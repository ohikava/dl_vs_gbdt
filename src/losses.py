"""Losses for a ~0.5% positive rate. Focal is the default; plain/weighted BCE
are kept for comparison. Mirrors the central project's loss choices.
"""
import torch
import torch.nn.functional as F

from . import config as C


def focal_loss(logits, targets, gamma=C.FOCAL_GAMMA, alpha=C.FOCAL_ALPHA):
    """Binary focal loss (Lin et al. 2017). Down-weights easy negatives, which
    dominate at 0.5% prevalence."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


def make_loss(name, pos_weight=None, device="cpu"):
    if name == "focal":
        return lambda logits, y: focal_loss(logits, y)
    if name == "weighted_bce":
        pw = torch.tensor([pos_weight], device=device) if pos_weight else None
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
    if name == "bce":
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y)
    raise ValueError(f"unknown loss: {name}")
