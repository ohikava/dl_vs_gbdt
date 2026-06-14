"""Train the FT-Transformer and ablate the numeric-embedding mode.

  python -m src.train_dl --num-emb linear      # vanilla FT-T
  python -m src.train_dl --num-emb periodic    # learnable periodic
  python -m src.train_dl --num-emb ple          # piecewise-linear (quantile bins)
  python -m src.train_dl --all                  # run all three, write dl_metrics.json
  python -m src.train_dl --all --subsample 150000 --epochs 2   # fast smoke run

Honest protocol: fit the encoder on TRAIN ONLY, train on train, select the epoch
by val PR-AUC, report that epoch's test metrics. Test is never used for selection.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from . import data as D
from .encoder import TabularEncoder
from .ft_transformer import FTTransformer
from .losses import make_loss
from .metrics import compute_metrics, format_metrics


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(X_cat, X_num, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X_cat), torch.from_numpy(X_num),
        torch.from_numpy(y.astype("float32")),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores = []
    for xc, xn, _ in loader:
        xc, xn = xc.to(device), xn.to(device)
        logits = model(xc, xn)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train_one(num_mode, enc, splits, device, epochs, batch_size, seed=C.SEED, verbose=True):
    (Xc_tr, Xn_tr, y_tr), (Xc_va, Xn_va, y_va), (Xc_te, Xn_te, y_te) = splits
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FTTransformer(
        cat_vocab_sizes=enc.vocab_sizes(), n_num=enc.n_num(),
        num_mode=num_mode, ple_edges=enc.ple_edge_list(),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    loss_fn = make_loss(C.LOSS, pos_weight=pos_weight, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    tr_loader = make_loader(Xc_tr, Xn_tr, y_tr, batch_size, shuffle=True)
    va_loader = make_loader(Xc_va, Xn_va, y_va, batch_size, shuffle=False)
    te_loader = make_loader(Xc_te, Xn_te, y_te, batch_size, shuffle=False)

    best_val_pr = -1.0
    best = None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for xc, xn, yb in tr_loader:
            xc, xn, yb = xc.to(device), xn.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xc, xn)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
        train_loss = running / len(y_tr)

        val_m = compute_metrics(y_va, predict(model, va_loader, device))
        history.append({"epoch": ep, "train_loss": train_loss, "val_pr_auc": val_m["pr_auc"]})
        if verbose:
            print(f"  [{num_mode}] epoch {ep}/{epochs} loss={train_loss:.4f} "
                  f"val_PR-AUC={val_m['pr_auc']:.4f} ({time.time()-t0:.1f}s)")
        if val_m["pr_auc"] > best_val_pr:
            best_val_pr = val_m["pr_auc"]
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best)
    val_m = compute_metrics(y_va, predict(model, va_loader, device))
    test_m = compute_metrics(y_te, predict(model, te_loader, device))
    if verbose:
        print(format_metrics(f"FT-T:{num_mode} VAL", val_m))
        print(format_metrics(f"FT-T:{num_mode} TEST", test_m))
    return {"num_mode": num_mode, "n_params": n_params,
            "val": val_m, "test": test_m, "history": history}


def encode_splits(enc, train_df, val_df, test_df):
    enc.fit(train_df)
    out = []
    for df in (train_df, val_df, test_df):
        Xc, Xn = enc.transform(df)
        out.append((Xc, Xn, df[C.TARGET].to_numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-emb", choices=["linear", "periodic", "ple"], default="ple")
    ap.add_argument("--all", action="store_true", help="run all three modes")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--subsample", type=int, default=None,
                    help="rows per split for a fast smoke run")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    print("loading data ...")
    train_df, val_df, test_df = D.load_splits(subsample=args.subsample)
    D.describe_splits(train_df, val_df, test_df)

    enc = TabularEncoder()
    splits = encode_splits(enc, train_df, val_df, test_df)
    C.ART_DIR.mkdir(exist_ok=True)
    enc.save(C.ART_DIR / "encoder.json")

    modes = ["linear", "periodic", "ple"] if args.all else [args.num_emb]
    results = {}
    for mode in modes:
        print(f"\n=== FT-Transformer | numeric embedding = {mode} ===")
        results[mode] = train_one(mode, enc, splits, device,
                                  epochs=args.epochs, batch_size=args.batch_size)

    out_path = C.ART_DIR / "dl_metrics.json"
    # merge with any prior single-mode runs so --num-emb runs accumulate
    prior = {}
    if out_path.exists():
        prior = json.load(open(out_path))
    prior.update(results)
    with open(out_path, "w") as f:
        json.dump(prior, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
