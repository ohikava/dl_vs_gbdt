"""Lightweight checks (run as a script, no pytest needed):

  python -m src.test_encoder

Verifies the reusability contract the central project depends on:
  * encoder fits on train only; an unseen category -> UNK_ID, never a new id
  * transform is deterministic and shape-correct
  * save/load round-trips
  * each NumericEmbeddings mode yields [B, n_num, d_token]
"""
import numpy as np
import pandas as pd
import torch

from . import config as C
from .encoder import TabularEncoder
from .num_embeddings import NumericEmbeddings


def _toy_df(merchants, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "category": rng.choice(["food", "gas", "net"], n),
        "merchant": rng.choice(merchants, n),
        "gender": rng.choice(["M", "F"], n),
        "state": rng.choice(["CA", "NY"], n),
        "job": rng.choice(["eng", "doc"], n),
        "amt_log": rng.normal(3, 1, n),
        "dist": rng.gamma(2, 1, n),
        "age": rng.normal(40, 10, n),
        "city_pop_log": rng.normal(10, 2, n),
        "hour_sin": rng.uniform(-1, 1, n),
        "hour_cos": rng.uniform(-1, 1, n),
        "dow_sin": rng.uniform(-1, 1, n),
        "dow_cos": rng.uniform(-1, 1, n),
        C.TARGET: rng.integers(0, 2, n),
    })


def test_unseen_category_maps_to_unk():
    train = _toy_df(["m1", "m2", "m3"])
    val = _toy_df(["m1", "m2", "m3", "m_NEW"])  # m_NEW unseen at fit time
    enc = TabularEncoder().fit(train)
    Xc, _ = enc.transform(val)
    j = enc.cat_cols.index("merchant")
    codes_new = Xc[val["merchant"].to_numpy() == "m_NEW", j]
    assert (codes_new == C.UNK_ID).all(), "unseen category must map to UNK_ID"
    # train vocab must NOT contain a code for the unseen category
    assert "m_NEW" not in enc.cat_maps_["merchant"]
    print("ok: unseen category -> UNK_ID")


def test_transform_deterministic_and_shaped():
    train = _toy_df(["m1", "m2"])
    enc = TabularEncoder().fit(train)
    a_c, a_n = enc.transform(train)
    b_c, b_n = enc.transform(train)
    assert np.array_equal(a_c, b_c) and np.allclose(a_n, b_n), "transform must be deterministic"
    assert a_c.shape == (len(train), len(C.CAT_COLS))
    assert a_n.shape == (len(train), len(C.NUM_COLS))
    assert a_c.dtype == np.int64 and a_n.dtype == np.float32
    # codes are within vocab range
    for j, col in enumerate(enc.cat_cols):
        assert a_c[:, j].max() < enc.vocab_sizes_[col]
    print("ok: transform deterministic, shaped, in-range")


def test_save_load_roundtrip(tmp="/tmp/_enc_test.json"):
    train = _toy_df(["m1", "m2", "m3"])
    enc = TabularEncoder().fit(train)
    enc.save(tmp)
    enc2 = TabularEncoder.load(tmp)
    a_c, a_n = enc.transform(train)
    b_c, b_n = enc2.transform(train)
    assert np.array_equal(a_c, b_c) and np.allclose(a_n, b_n), "save/load must round-trip"
    print("ok: save/load round-trip")


def test_num_embedding_modes():
    train = _toy_df(["m1", "m2"])
    enc = TabularEncoder().fit(train)
    _, Xn = enc.transform(train)
    x = torch.from_numpy(Xn)
    d = 16
    for mode in ("linear", "periodic", "ple"):
        m = NumericEmbeddings(enc.n_num(), d, mode=mode, ple_edges=enc.ple_edge_list())
        out = m(x)
        assert out.shape == (len(train), enc.n_num(), d), f"{mode}: bad shape {out.shape}"
        assert torch.isfinite(out).all(), f"{mode}: non-finite output"
    print("ok: linear/periodic/ple all yield [B, n_num, d_token]")


if __name__ == "__main__":
    test_unseen_category_maps_to_unk()
    test_transform_deterministic_and_shaped()
    test_save_load_roundtrip()
    test_num_embedding_modes()
    print("\nall encoder tests passed.")
