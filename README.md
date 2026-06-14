# Tabular DL vs GBDT on fraud data

Supporting project for the central antifraud system
([`rec_sys_anti_frod`](../rec_sys_anti_frod)). It answers one question honestly:

> On the same fraud data, can a tabular Transformer (FT-Transformer) beat the
> gradient-boosting baseline (CatBoost / XGBoost) that antifraud teams actually
> ship — and if not, what do you have to get right to even get close?

The point isn't to crown a winner. It's to show three things a fraud ML engineer
is expected to know cold:

1. **In antifraud, GBDT is the baseline you have to beat, not the thing you reach
   for.** Trees on tabular features are the incumbent; a neural net has to justify
   its existence against them.
2. **How to encode features for a neural net** — categorical embeddings, and
   especially **numerical embeddings** (periodic and piecewise-linear), which are
   the difference between a Transformer that loses badly and one that's
   competitive.
3. **The target-encoding leakage trap** — the most common way people accidentally
   leak the label into their features, and how to do it correctly.

The feature encoder (`src/encoder.py`) is written to drop straight into the
central project: same reserved ids, same train-only fitting discipline, same
metric conventions.

---

## Data

[Sparkov "Credit Card Transactions Fraud Detection"](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
— 1.85M synthetic card transactions, label `is_fraud` (~0.5% positive). The CSVs
are reused in place from the central project (`../rec_sys_anti_frod/data`); set
`DL_VS_GBDT_DATA_DIR` to point elsewhere.

**Framing:** per-transaction (one row = one example). The *sequence* framing lives
in the central project; here every transaction is classified independently from
its own engineered features. Same features both ways, so the two projects describe
the same signal.

**Split — temporal, never random.** The train file's tail (`>= 2020-04-01`) is the
validation set; the separate test file is the test set. Shuffling across time
would let the model peek at the future — the cardinal fraud-modeling sin.

**Features** (mirrors `engineer()` in the central project):
- Categorical: `category`, `merchant`, `gender`, `state`, `job`.
- Numeric: `amt_log`, `dist` (haversine cardholder↔merchant), `age`,
  `city_pop_log`, and cyclical `hour_sin/cos`, `dow_sin/cos`.

---

## Results

Test-set metrics (full data). PR-AUC is the headline (base rate ≈ 0.4%, so ROC-AUC
is generous and reported only for reference); recall at low fixed FPR is what an
ops team actually lives with.

<!-- RESULTS_TABLE_START -->

| model | PR-AUC | ROC-AUC | R@FPR0.001 | R@FPR0.005 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|---|---|---|
| CatBoost | 0.8730 | 0.9974 | 0.8182 | 0.9221 | 0.9515 | 0.9897 |
| XGBoost | 0.7908 | 0.9922 | 0.7371 | 0.8331 | 0.8793 | 0.9613 |
| FT-Transformer (linear) | 0.8083 | 0.9942 | 0.7282 | 0.8545 | 0.8979 | 0.9683 |
| FT-Transformer (periodic) | 0.8713 | 0.9969 | 0.8196 | 0.9021 | 0.9375 | 0.9860 |
| FT-Transformer (ple) | 0.8692 | 0.9964 | 0.8149 | 0.8942 | 0.9310 | 0.9790 |

<!-- RESULTS_TABLE_END -->

**Reading the table:**
- **CatBoost sets the bar, and the Transformer reaches it only with proper numeric
  embeddings.** CatBoost (0.873 PR-AUC) and FT-Transformer with periodic embeddings
  (0.871) finish neck-and-neck on test; both clearly beat XGBoost. The honest
  conclusion isn't "DL wins" — it's "a well-encoded Transformer is *competitive*
  with the strongest tree baseline, and a badly-encoded one isn't." On a real fraud
  problem you'd still ship CatBoost (simpler, faster to train, no GPU) and keep the
  Transformer as the research/transfer track.
- **Numerical embeddings are the whole game for the Transformer.** The
  `linear → periodic → ple` ablation is the *same* model with only the numeric
  tokenizer swapped: PR-AUC goes 0.808 → 0.871 → 0.869. A raw linear token can't
  carve the sharp, non-monotone amount/distance thresholds a tree split gets for
  free; giving each scalar a *vector* (periodic frequencies or quantile bins) closes
  most of the gap to GBDT. This is the single most important encoding lesson here.
- **CatBoost ≫ XGBoost is a categorical-handling story.** Same trees, same features;
  the gap is `merchant` (≈690 values) and `job` (≈490). CatBoost's *ordered* target
  encoding extracts signal from those that XGBoost's native one-hot/split handling
  leaves on the table — and it does so without the leakage a hand-rolled target
  encoder would introduce (see below).

---

## Numerical embeddings (the `src/num_embeddings.py` ablation)

Each numeric feature → one `d_token` vector, three interchangeable modes
(Gorishniy et al., *On Embeddings for Numerical Features in Tabular DL*, 2022):

| mode | what it does | why it helps |
|---|---|---|
| `linear` | `x·W_f + b_f` per feature (vanilla FT-Transformer) | baseline; one direction in token space |
| `periodic` | `Linear(ReLU([sin, cos](2π·c_f·x)))`, frequencies `c_f` learned | captures repeated/threshold structure a linear map can't |
| `ple` | piecewise-linear over **train** quantile bins, then `Linear` | tree-like: each bin is its own coordinate, sharp boundaries |

PLE bin edges come from the **train** distribution only (stored in the encoder), so
the model never sees val/test quantiles — the same train-only discipline as the
categorical vocab.

---

## Target encoding & leakage (`src/target_encoding_leak.py`)

Target (mean) encoding replaces a category with the mean label for that category.
It's powerful and it's the #1 way people leak the label. A target encoding is
itself a P(fraud) estimate, so we rank transactions by the encoded value directly
and read ROC-AUC — no downstream classifier to muddy the signal. Two encodings:

- **`naive`** — category mean over the whole train set, applied to train itself.
  Each row's own label is baked into its feature → inflated train score; the
  **train≫test gap is the leak**.
- **`oof`** — out-of-fold: a row's encoding comes from the *other* folds only;
  test uses full-train stats. No row sees its own label → honest train score.

The sharp part is showing *when* it leaks, on two columns (full-data ROC-AUC):

| column | encoding | train | test | gap (leak) |
|---|---|---|---|---|
| `noise_id` (50k random ids, no signal) | naive | **0.950** | 0.506 | **0.444** |
| `noise_id` | oof | 0.498 | 0.506 | -0.008 |
| `merchant` (~690 values, real signal) | naive | 0.753 | 0.714 | 0.039 |
| `merchant` | oof | 0.717 | 0.714 | 0.003 |

- `noise_id` is a synthetic high-cardinality id with ~20 rows/category and **no**
  relation to the label. Naive TE memorizes each row's own label → train ROC 0.95,
  test ROC 0.51 (random). OOF removes the leak: both ≈ 0.5. This is the classic
  production blow-up from target-encoding an ID-like column.
- `merchant` is well-populated (~1600 rows/category), so one row barely moves its
  mean and naive leaks little — and it has genuine signal (test ROC 0.71). OOF still
  closes the residual gap.
- **Leakage scales with 1/(rows per category).** That's the rule to carry: target
  encoding is dangerous exactly on the high-cardinality columns you most want to
  encode. CatBoost's *ordered* target encoding removes this by construction — which
  is why the GBDT baseline gets native categoricals and we never hand-roll TE into
  the feature matrix (plot: `artifacts/te_leak.png`).

---

## Layout

```
src/
  config.py              schema, hyperparameters, FPR targets
  data.py                load Sparkov, engineer features, temporal split
  encoder.py             ⭐ TabularEncoder — reusable, train-only fit, PAD/UNK ids
  num_embeddings.py      ⭐ linear / periodic / ple numeric embeddings
  ft_transformer.py      FT-Transformer (feature tokenizer + [CLS] + encoder)
  losses.py              focal / weighted-BCE / BCE
  metrics.py             PR-AUC, ROC-AUC, recall@fixed-FPR (ported from central)
  train_gbdt.py          CatBoost + XGBoost baselines
  train_dl.py            FT-Transformer training + numeric-embedding ablation
  target_encoding_leak.py  ⭐ naive vs out-of-fold target encoding
  run_all.py             orchestrates both phases -> artifacts/comparison.md
  test_encoder.py        encoder reusability checks
artifacts/               metrics (json), comparison.md, te_leak.png, encoder.json
```

⭐ = the parts that carry the project's point / get reused downstream.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fast end-to-end smoke (minutes, subsampled):
python -m src.run_all --subsample 150000 --epochs 2

# full run (writes artifacts/comparison.md):
python -m src.run_all

# individual pieces:
python -m src.train_gbdt                         # CatBoost + XGBoost
python -m src.train_dl --all                     # FT-T linear/periodic/ple
python -m src.target_encoding_leak               # leakage demo + plot
python -m src.test_encoder                        # encoder contract checks
```

> macOS note: CatBoost/XGBoost and PyTorch ship separate OpenMP runtimes and
> segfault if a booster trains in a process that also imported torch. `run_all`
> runs each phase in its own subprocess to avoid this; run the scripts separately
> rather than importing both into one process.

---

## Honest takeaways

- GBDT is the baseline to beat in antifraud. CatBoost isn't beaten here, but a
  properly-encoded FT-Transformer *matches* it — so the framing "DL can't touch
  trees on tabular data" is too strong; "DL needs the right feature encoding to
  compete, and even then trees are the pragmatic default" is the accurate one.
- The Transformer only becomes competitive with proper **numerical embeddings**
  (0.808 → 0.871 from linear to periodic). The linear-token version is not close.
  That's the transferable lesson and the reason `num_embeddings.py` exists.
- Among trees, **CatBoost > XGBoost** on this data because of high-cardinality
  categorical handling, not hyperparameters — a concrete reason CatBoost is a strong
  default for fraud.
- The reusable win is `src/encoder.py` + `src/num_embeddings.py`: a train-only,
  leak-aware encoder with categorical embeddings and periodic/PLE numeric
  embeddings, ready to feed the central project's sequence model.
- ROC-AUC ≈ 0.99 for everything is a trap at 0.4% prevalence; PR-AUC and
  recall@low-FPR are the metrics that actually separate the models.
