"""Load the Sparkov fraud CSVs, engineer per-transaction features, and produce a
leak-free temporal split (train / val / test) as plain pandas DataFrames.

This is the per-row ("tabular") framing: each transaction is one independent
example. The sequence framing lives in the central project. Feature engineering
mirrors `rec_sys_anti_frod/src/preprocess.py::engineer` so both projects describe
the same signal.

Correctness decisions a reviewer will poke at:
  * Temporal split, never random: the train file's tail (>= VAL_CUTOFF) is the
    validation set; the separate test file is the test set. Shuffling across time
    would leak the future into the past — fatal in fraud.
  * No target-derived feature is created here. Categorical/numeric *encoders* are
    fit later, on train only (see encoder.py). Target encoding and its leakage
    traps are demonstrated separately (target_encoding_leak.py).
"""
import numpy as np
import pandas as pd

from . import config as C

_USECOLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "state", "city_pop", "job", "dob", "unix_time", "lat", "long",
    "merch_lat", "merch_long", "is_fraud",
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _check_data_exists():
    if not C.RAW_TRAIN.exists() or not C.RAW_TEST.exists():
        raise FileNotFoundError(
            f"Sparkov CSVs not found under {C.DATA_DIR}.\n"
            f"Expected: {C.RAW_TRAIN.name} and {C.RAW_TEST.name}.\n"
            "Download the Kaggle dataset 'kartik2112/fraud-detection' (Sparkov) "
            "and place fraudTrain.csv / fraudTest.csv there, or set "
            "DL_VS_GBDT_DATA_DIR to a directory that contains them."
        )


def engineer(df):
    """Derive the model-ready numeric features. Same definitions as the central
    project so the encoder transfers without surprises."""
    dt = df["trans_date_trans_time"]
    df["amt_log"] = np.log1p(df["amt"])
    df["dist"] = _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["age"] = (dt - df["dob"]).dt.days / 365.25
    df["city_pop_log"] = np.log1p(df["city_pop"])
    # cyclical time encodings (already bounded in [-1, 1])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def load_splits(subsample=None, seed=C.SEED):
    """Return (train_df, val_df, test_df), each with CAT_COLS + NUM_COLS + target.

    subsample: if set, keep at most this many rows per split (stratified by label,
    keeping ALL frauds) for a fast smoke run. The temporal order is preserved.
    """
    _check_data_exists()
    tr = pd.read_csv(C.RAW_TRAIN, usecols=_USECOLS,
                     parse_dates=["trans_date_trans_time", "dob"])
    te = pd.read_csv(C.RAW_TEST, usecols=_USECOLS,
                     parse_dates=["trans_date_trans_time", "dob"])

    cutoff = pd.Timestamp(C.VAL_CUTOFF)
    tr["split"] = np.where(tr["trans_date_trans_time"] >= cutoff, 1, 0).astype("int8")
    te["split"] = np.int8(2)

    df = pd.concat([tr, te], ignore_index=True)
    df = engineer(df)

    keep = C.CAT_COLS + C.NUM_COLS + [C.TARGET, "split"]
    df = df[keep]

    parts = []
    for s in (0, 1, 2):
        part = df[df["split"] == s].reset_index(drop=True)
        if subsample is not None and len(part) > subsample:
            part = _subsample_keep_pos(part, subsample, seed)
        parts.append(part.drop(columns="split"))
    return tuple(parts)


def _subsample_keep_pos(part, n, seed):
    """Keep every fraud row, fill the rest with random negatives, preserve order."""
    pos = part[part[C.TARGET] == 1]
    neg = part[part[C.TARGET] == 0]
    n_neg = max(0, n - len(pos))
    neg = neg.sample(n=min(n_neg, len(neg)), random_state=seed)
    out = pd.concat([pos, neg]).sort_index()
    return out.reset_index(drop=True)


def describe_splits(train_df, val_df, test_df):
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        y = d[C.TARGET].to_numpy()
        print(f"{name:5}: {len(d):>9,} rows | fraud {100 * y.mean():.3f}%")


if __name__ == "__main__":
    tr, va, te = load_splits()
    describe_splits(tr, va, te)
    print("cat cardinalities (train):",
          {c: tr[c].nunique() for c in C.CAT_COLS})
