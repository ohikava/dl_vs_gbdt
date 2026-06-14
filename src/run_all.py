"""One entry point: run the GBDT baselines and the FT-Transformer ablation on the
same data, then write a single comparison table.

  python -m src.run_all                                   # full run
  python -m src.run_all --subsample 200000 --epochs 3     # fast end-to-end

Each phase runs in its OWN subprocess. This isn't just tidiness: on macOS,
CatBoost/XGBoost and PyTorch each ship their own OpenMP runtime, and training a
booster in a process that has also imported torch segfaults. Isolating the phases
sidesteps that entirely and keeps torch out of the tree-boosting process.

The subprocesses write artifacts/{gbdt_metrics.json, dl_metrics.json}; this script
reads them back and writes artifacts/comparison.md.
"""
import argparse
import json
import subprocess
import sys

from . import config as C

ROW_ORDER = [
    ("CatBoost", ("gbdt", "catboost")),
    ("XGBoost", ("gbdt", "xgboost")),
    ("FT-Transformer (linear)", ("dl", "linear")),
    ("FT-Transformer (periodic)", ("dl", "periodic")),
    ("FT-Transformer (ple)", ("dl", "ple")),
]


def _run_phase(module, extra):
    cmd = [sys.executable, "-u", "-m", module] + extra
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(C.ROOT))


def _table(gbdt, dl):
    cols = ["pr_auc", "roc_auc"] + [f"recall@fpr{f}" for f in C.FPR_TARGETS]
    headers = ["model", "PR-AUC", "ROC-AUC"] + [f"R@FPR{f}" for f in C.FPR_TARGETS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    src = {"gbdt": gbdt, "dl": dl}
    for label, (kind, key) in ROW_ORDER:
        entry = src.get(kind, {}).get(key)
        if not entry:
            continue
        m = entry["test"]
        cells = [label] + [f"{m[c]:.4f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--skip-gbdt", action="store_true")
    ap.add_argument("--skip-dl", action="store_true")
    args = ap.parse_args()

    C.ART_DIR.mkdir(exist_ok=True)
    common = []
    if args.subsample is not None:
        common += ["--subsample", str(args.subsample)]

    if not args.skip_gbdt:
        _run_phase("src.train_gbdt", common)
    if not args.skip_dl:
        _run_phase("src.train_dl", ["--all", "--epochs", str(args.epochs),
                                    "--batch-size", str(args.batch_size)] + common)

    gbdt = json.load(open(C.ART_DIR / "gbdt_metrics.json")) \
        if (C.ART_DIR / "gbdt_metrics.json").exists() else {}
    dl = json.load(open(C.ART_DIR / "dl_metrics.json")) \
        if (C.ART_DIR / "dl_metrics.json").exists() else {}

    table = _table(gbdt, dl)
    out = C.ART_DIR / "comparison.md"
    with open(out, "w") as f:
        f.write("# DL vs GBDT — test-set comparison (Sparkov fraud)\n\n")
        f.write(table + "\n")
    _inject_readme(table)
    print("\n" + table)
    print(f"\nsaved -> {out}")


def _inject_readme(table):
    """Drop the latest table into README between the RESULTS markers."""
    path = C.ROOT / "README.md"
    if not path.exists():
        return
    start, end = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
    txt = path.read_text()
    if start not in txt or end not in txt:
        return
    head = txt[:txt.index(start) + len(start)]
    tail = txt[txt.index(end):]
    path.write_text(f"{head}\n\n{table}\n\n{tail}")


if __name__ == "__main__":
    main()
