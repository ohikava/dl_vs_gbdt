"""GBDT baselines — the bar the neural net has to clear.

In antifraud, gradient boosting on tabular features is the incumbent. We give it
every fair advantage: native categorical handling (no manual, leak-prone target
encoding), class-imbalance weighting, and early stopping on val PR-AUC.

  python -m src.train_gbdt                      # CatBoost + XGBoost
  python -m src.train_gbdt --subsample 200000   # fast smoke run

Both models see the SAME engineered features as the FT-Transformer; the only
difference is the encoding (trees take raw/native categoricals, the net takes
embeddings). Metrics land in artifacts/gbdt_metrics.json.
"""
import argparse
import json

import numpy as np

from . import config as C
from . import data as D
from .metrics import compute_metrics, format_metrics


def _feature_frames(train_df, val_df, test_df):
    cols = C.CAT_COLS + C.NUM_COLS
    return (train_df[cols], val_df[cols], test_df[cols],
            train_df[C.TARGET].to_numpy(), val_df[C.TARGET].to_numpy(),
            test_df[C.TARGET].to_numpy())


def run_catboost(train_df, val_df, test_df):
    from catboost import CatBoostClassifier, Pool

    Xtr, Xva, Xte, ytr, yva, yte = _feature_frames(train_df, val_df, test_df)
    # CatBoost wants categoricals as strings; it applies *ordered* target encoding
    # internally, which is leak-free by construction (unlike a naive group-mean).
    for X in (Xtr, Xva, Xte):
        X[C.CAT_COLS] = X[C.CAT_COLS].astype(str)

    cat_idx = [Xtr.columns.get_loc(c) for c in C.CAT_COLS]
    train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
    val_pool = Pool(Xva, yva, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=8,
        loss_function="Logloss", eval_metric="PRAUC",
        auto_class_weights="Balanced",  # handles 0.5% prevalence
        random_seed=C.SEED, verbose=False,
        early_stopping_rounds=100,
    )
    model.fit(train_pool, eval_set=val_pool)
    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    return {
        "best_iteration": int(model.get_best_iteration()),
        "val": compute_metrics(yva, p_va), "test": compute_metrics(yte, p_te),
    }


def run_xgboost(train_df, val_df, test_df):
    import xgboost as xgb

    Xtr, Xva, Xte, ytr, yva, yte = _feature_frames(train_df, val_df, test_df)
    # XGBoost native categorical support: cast to pandas 'category' dtype.
    for X in (Xtr, Xva, Xte):
        for c in C.CAT_COLS:
            X[c] = X[c].astype("category")

    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    # High ceiling so val-PR-AUC early stopping (not the cap) decides when to stop —
    # at 2000 rounds XGBoost was still improving, which understated it.
    model = xgb.XGBClassifier(
        n_estimators=8000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=True,
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        early_stopping_rounds=100, random_state=C.SEED,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    return {
        "best_iteration": int(model.best_iteration),
        "val": compute_metrics(yva, p_va), "test": compute_metrics(yte, p_te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None)
    args = ap.parse_args()

    print("loading data ...")
    train_df, val_df, test_df = D.load_splits(subsample=args.subsample)
    D.describe_splits(train_df, val_df, test_df)

    results = {}
    print("\n=== CatBoost ===")
    results["catboost"] = run_catboost(train_df, val_df, test_df)
    print(format_metrics("CatBoost VAL", results["catboost"]["val"]))
    print(format_metrics("CatBoost TEST", results["catboost"]["test"]))

    print("\n=== XGBoost ===")
    results["xgboost"] = run_xgboost(train_df, val_df, test_df)
    print(format_metrics("XGBoost VAL", results["xgboost"]["val"]))
    print(format_metrics("XGBoost TEST", results["xgboost"]["test"]))

    C.ART_DIR.mkdir(exist_ok=True)
    out_path = C.ART_DIR / "gbdt_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
