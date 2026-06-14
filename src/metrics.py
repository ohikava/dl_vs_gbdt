"""Fraud-appropriate metrics. Imbalance is ~0.5%, so PR-AUC and recall at a
fixed false-positive rate matter; ROC-AUC is reported only for reference.

Ported verbatim from the central project (rec_sys_anti_frod/src/metrics.py) so
that DL and GBDT numbers here are directly comparable to the sequence model there.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from . import config as C


def recall_at_fpr(y_true, y_score, target_fpr):
    """Highest recall (TPR) achievable while keeping FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(y_true.mean()),
    }
    for f in C.FPR_TARGETS:
        out[f"recall@fpr{f}"] = recall_at_fpr(y_true, y_score, f)
    return out


def format_metrics(name, m):
    parts = [
        f"PR-AUC {m['pr_auc']:.4f}",
        f"ROC-AUC {m['roc_auc']:.4f}",
    ]
    parts += [f"R@FPR{f}={m[f'recall@fpr{f}']:.3f}" for f in C.FPR_TARGETS]
    return f"[{name}] " + " | ".join(parts)
