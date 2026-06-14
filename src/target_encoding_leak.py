"""Target encoding: the leak that flatters your CV and dies in production.

Target (mean) encoding replaces a category with the mean of the label for that
category. It is powerful and it is the single most common way people leak the
label into their features. This module makes the leak *measurable*, and — just as
important — shows WHEN it bites and when it doesn't.

We score by the encoded value directly: a target encoding *is* a P(fraud) estimate,
so ranking transactions by it and taking ROC/PR-AUC measures exactly how good (or
leaky) the encoding is. No downstream classifier, no numerical games.

Two encodings, evaluated on train (which can cheat) and test (which can't):

  naive : category mean over the WHOLE train set, applied to train itself. Each
          row's own label is baked into its feature -> the train score is inflated;
          the train>>test gap is the leak.
  oof   : out-of-fold. A row's encoding is computed from the OTHER folds only, so
          no row sees its own label. Test uses full-train stats. The correct way.

Two columns, to show the leak's dependence on cardinality:

  merchant   : a REAL column, ~690 values over ~1.1M rows -> ~1600 rows/category.
               Each category mean is stable; one row barely moves it, so naive
               leaks little. Lesson: well-populated categories are relatively safe.
  noise_id   : a SYNTHETIC high-cardinality id (~50k random values, no relation to
               the label). ~20 rows/category, so a row's own label dominates its
               category mean. naive memorizes the label: train AUC -> ~1.0, test
               AUC -> ~0.5 (pure noise). This is the catastrophic case, and it's
               why target-encoding an ID-like column is a classic production
               blow-up.

  python -m src.target_encoding_leak
  python -m src.target_encoding_leak --subsample 200000   # faster
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import data as D

N_NOISE = 50_000        # cardinality of the synthetic id (few rows per category)
SMOOTHING = 10.0        # Bayesian smoothing toward the global mean


def _smoothed_mean(sum_y, count, global_mean, smoothing=SMOOTHING):
    return (sum_y + smoothing * global_mean) / (count + smoothing)


def _fit_category_means(cats, y, global_mean):
    dfg = pd.DataFrame({"c": np.asarray(cats), "y": np.asarray(y)})
    g = dfg.groupby("c")["y"].agg(["sum", "count"])
    return _smoothed_mean(g["sum"], g["count"], global_mean).to_dict()


def encode_naive(train_cats, train_y, eval_cats, global_mean):
    """Category mean from ALL of train, applied to train itself and to eval.
    The train encoding includes each row's own label -> leak."""
    means = _fit_category_means(train_cats, train_y, global_mean)
    enc_train = pd.Series(train_cats).map(means).fillna(global_mean).to_numpy()
    enc_eval = pd.Series(eval_cats).map(means).fillna(global_mean).to_numpy()
    return enc_train, enc_eval


def encode_oof(train_cats, train_y, eval_cats, global_mean, n_splits=5, seed=C.SEED):
    """Out-of-fold encoding for train (each fold encoded from the others); eval
    encoded from the full train. No row contributes to its own feature."""
    train_cats = pd.Series(np.asarray(train_cats)).reset_index(drop=True)
    y_arr = np.asarray(train_y)
    enc_train = np.empty(len(y_arr), dtype="float64")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, oof_idx in skf.split(train_cats, y_arr):
        means = _fit_category_means(train_cats.iloc[tr_idx], y_arr[tr_idx], global_mean)
        enc_train[oof_idx] = train_cats.iloc[oof_idx].map(means).fillna(global_mean).to_numpy()
    means_full = _fit_category_means(train_cats, y_arr, global_mean)
    enc_eval = pd.Series(eval_cats).map(means_full).fillna(global_mean).to_numpy()
    return enc_train, enc_eval


def _auc(y, score):
    # the encoding is itself a P(fraud) estimate -> rank directly
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score))}


def _eval_column(col_train, col_test, y_tr, y_te, global_mean):
    out = {}
    e_tr, e_te = encode_naive(col_train, y_tr, col_test, global_mean)
    out["naive"] = {"train": _auc(y_tr, e_tr), "test": _auc(y_te, e_te)}
    e_tr, e_te = encode_oof(col_train, y_tr, col_test, global_mean)
    out["oof"] = {"train": _auc(y_tr, e_tr), "test": _auc(y_te, e_te)}
    return out


def run(train_df, val_df, test_df, seed=C.SEED):
    y_tr = train_df[C.TARGET].to_numpy()
    y_te = test_df[C.TARGET].to_numpy()
    global_mean = float(y_tr.mean())

    rng = np.random.default_rng(seed)
    noise_tr = rng.integers(0, N_NOISE, len(train_df))
    noise_te = rng.integers(0, N_NOISE, len(test_df))

    results = {
        "merchant": _eval_column(train_df["merchant"], test_df["merchant"],
                                 y_tr, y_te, global_mean),
        "noise_id": _eval_column(noise_tr, noise_te, y_tr, y_te, global_mean),
    }
    return results, global_mean


def _print_table(results):
    print(f"\n{'column':<10} {'encoding':<7} {'train ROC':>10} {'test ROC':>10} {'gap (leak)':>11}")
    print("-" * 52)
    for col in ("noise_id", "merchant"):
        for enc in ("naive", "oof"):
            tr = results[col][enc]["train"]["roc_auc"]
            te = results[col][enc]["test"]["roc_auc"]
            print(f"{col:<10} {enc:<7} {tr:>10.4f} {te:>10.4f} {tr - te:>11.4f}")
        print()
    print("Reading:")
    print("  noise_id (rare categories): naive train ROC ~1.0 vs test ~0.5 — the")
    print("    encoding memorized each row's own label. OOF kills it: train ~= test ~0.5.")
    print("  merchant (well-populated):  naive's gap is small — many rows per category")
    print("    means one row barely moves the mean, so there's little to leak.")
    print("  => target encoding leaks in proportion to 1/(rows per category).")
    print("     CatBoost's *ordered* TE removes the leak by construction, which is why")
    print("     the GBDT baseline gets native categoricals and we never hand-roll TE.")


def _plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping plot: {e})")
        return
    cols = ["noise_id", "merchant"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col in zip(axes, cols):
        encs = ["naive", "oof"]
        train = [results[col][e]["train"]["roc_auc"] for e in encs]
        test = [results[col][e]["test"]["roc_auc"] for e in encs]
        x = np.arange(len(encs))
        w = 0.35
        ax.bar(x - w / 2, train, w, label="train ROC-AUC")
        ax.bar(x + w / 2, test, w, label="test ROC-AUC")
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["naive (leak)", "out-of-fold"])
        ax.set_title(f"target encoding of '{col}'")
        ax.set_ylim(0.4, 1.02)
        for i, (a, b) in enumerate(zip(train, test)):
            ax.text(i - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("ROC-AUC")
    axes[1].legend(loc="lower center")
    fig.suptitle("Target-encoding leakage: train↔test gap = label leaked into the feature")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None)
    args = ap.parse_args()

    print("loading data ...")
    train_df, val_df, test_df = D.load_splits(subsample=args.subsample)
    D.describe_splits(train_df, val_df, test_df)

    results, global_mean = run(train_df, val_df, test_df)
    _print_table(results)

    C.ART_DIR.mkdir(exist_ok=True)
    _plot(results, C.ART_DIR / "te_leak.png")
    with open(C.ART_DIR / "te_leak.json", "w") as f:
        json.dump({"global_mean": global_mean, "noise_cardinality": N_NOISE,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
