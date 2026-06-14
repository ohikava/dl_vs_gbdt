"""TabularEncoder — the reusable feature encoder.

This is the piece that ports into the central project. It turns a raw
per-transaction DataFrame into two arrays a neural net can consume:

    X_cat : int64  [N, n_cat]   integer codes, ready for nn.Embedding
    X_num : float32[N, n_num]   standardized numerics (cyclical left as-is)

Design rules (the things a fraud reviewer checks):
  * Vocabularies and standardization stats are fit on TRAIN ONLY. `transform`
    never re-fits. Categories unseen at fit time map to UNK_ID, not a fresh id.
  * Reserved ids PAD_ID=0 / UNK_ID=1 / N_RESERVED=2 match the central project,
    so an Embedding(padding_idx=0) stays compatible.
  * NO target is touched here. Target encoding is leaky if done naively and is
    demonstrated/handled separately (see target_encoding_leak.py). Keeping it out
    of the encoder is the whole point.
  * Quantile bin edges for piecewise-linear numeric embeddings (PLE) are computed
    from train numerics and stored, so the DL model can build that representation
    without ever seeing val/test distributions.

The encoder is framework-agnostic (numpy in/out); the PyTorch embedding layers
live in num_embeddings.py and ft_transformer.py and consume these arrays.
"""
import json

import numpy as np
import pandas as pd

from . import config as C


class TabularEncoder:
    def __init__(self, cat_cols=None, num_cols=None,
                 standardize=None, ple_cols=None, ple_bins=C.PLE_BINS):
        self.cat_cols = list(cat_cols if cat_cols is not None else C.CAT_COLS)
        self.num_cols = list(num_cols if num_cols is not None else C.NUM_COLS)
        self.standardize = set(standardize if standardize is not None else C.NUM_STANDARDIZE)
        self.ple_cols = list(ple_cols if ple_cols is not None else C.NUM_PLE_COLS)
        self.ple_bins = ple_bins

        # fitted state
        self.cat_maps_ = {}      # col -> {category_value: int_code}
        self.vocab_sizes_ = {}   # col -> vocab size (incl. reserved)
        self.num_stats_ = {}     # col -> {"mean":..., "std":...}
        self.ple_edges_ = {}     # col -> np.ndarray of bin edges (in standardized space)
        self._fitted = False

    # ----- fit / transform -----
    def fit(self, df_train):
        """Fit vocabularies, standardization stats, and PLE bin edges on train."""
        for col in self.cat_cols:
            cats = pd.Index(pd.unique(df_train[col].dropna()))
            self.cat_maps_[col] = {c: i + C.N_RESERVED for i, c in enumerate(cats)}
            self.vocab_sizes_[col] = len(cats) + C.N_RESERVED

        for col in self.num_cols:
            if col in self.standardize:
                mu = float(df_train[col].mean())
                sd = float(df_train[col].std()) or 1.0
            else:
                mu, sd = 0.0, 1.0  # cyclical: already bounded, leave as-is
            self.num_stats_[col] = {"mean": mu, "std": sd}

        # PLE edges computed on the *standardized* train values so the encoder is
        # the single source of truth for the numeric space.
        for col in self.ple_cols:
            mu, sd = self.num_stats_[col]["mean"], self.num_stats_[col]["std"]
            x = (df_train[col].to_numpy(dtype="float64") - mu) / sd
            qs = np.linspace(0.0, 1.0, self.ple_bins + 1)
            edges = np.quantile(x, qs)
            edges = np.unique(edges)  # collapse ties from spiky distributions
            if len(edges) < 2:        # degenerate (constant) feature guard
                edges = np.array([x.min(), x.max() + 1e-6])
            self.ple_edges_[col] = edges.astype("float32")

        self._fitted = True
        return self

    def transform(self, df):
        """Map a DataFrame to (X_cat int64, X_num float32) using fitted state."""
        assert self._fitted, "call fit() on the train split before transform()"
        n = len(df)

        X_cat = np.empty((n, len(self.cat_cols)), dtype="int64")
        for j, col in enumerate(self.cat_cols):
            mapping = self.cat_maps_[col]
            codes = df[col].map(mapping)            # unseen -> NaN
            X_cat[:, j] = codes.fillna(C.UNK_ID).astype("int64").to_numpy()

        X_num = np.empty((n, len(self.num_cols)), dtype="float32")
        for j, col in enumerate(self.num_cols):
            mu, sd = self.num_stats_[col]["mean"], self.num_stats_[col]["std"]
            X_num[:, j] = (df[col].to_numpy(dtype="float32") - mu) / sd
        return X_cat, X_num

    def fit_transform(self, df_train):
        return self.fit(df_train).transform(df_train)

    # ----- accessors used by the model -----
    def vocab_sizes(self):
        """List of categorical vocab sizes, aligned with self.cat_cols order."""
        return [self.vocab_sizes_[c] for c in self.cat_cols]

    def n_cat(self):
        return len(self.cat_cols)

    def n_num(self):
        return len(self.num_cols)

    def ple_edge_list(self):
        """Bin edges per numeric feature, aligned with self.num_cols.

        Non-PLE features (e.g. cyclical) get None — the model uses a plain linear
        token for those even in PLE mode.
        """
        return [self.ple_edges_.get(c) for c in self.num_cols]

    # ----- persistence (for transfer into the central project) -----
    def state_dict(self):
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "standardize": sorted(self.standardize),
            "ple_cols": self.ple_cols,
            "ple_bins": self.ple_bins,
            "cat_maps": {c: {str(k): v for k, v in m.items()}
                         for c, m in self.cat_maps_.items()},
            "vocab_sizes": self.vocab_sizes_,
            "num_stats": self.num_stats_,
            "ple_edges": {c: e.tolist() for c, e in self.ple_edges_.items()},
        }

    def load_state_dict(self, st):
        self.cat_cols = st["cat_cols"]
        self.num_cols = st["num_cols"]
        self.standardize = set(st["standardize"])
        self.ple_cols = st["ple_cols"]
        self.ple_bins = st["ple_bins"]
        # NOTE: keys are loaded as strings; map() will still match original dtypes
        # for string columns. Numeric-keyed categoricals would need re-casting, but
        # all Sparkov categoricals here are strings.
        self.cat_maps_ = st["cat_maps"]
        self.vocab_sizes_ = st["vocab_sizes"]
        self.num_stats_ = st["num_stats"]
        self.ple_edges_ = {c: np.asarray(e, dtype="float32")
                           for c, e in st["ple_edges"].items()}
        self._fitted = True
        return self

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        enc = cls()
        with open(path) as f:
            enc.load_state_dict(json.load(f))
        return enc
