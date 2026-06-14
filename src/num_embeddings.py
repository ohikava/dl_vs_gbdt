"""Numerical feature embeddings for tabular DL — the three modes we ablate.

A plain MLP/transformer fed raw standardized numerics struggles to carve the
sharp, non-monotone boundaries that trees get for free. Gorishniy et al. (2022),
"On Embeddings for Numerical Features in Tabular Deep Learning", show that giving
each scalar a *vector* representation closes much of the gap. We implement three:

  linear   : x_f -> x_f * W_f + b_f         (vanilla FT-Transformer tokenizer)
  periodic : x_f -> Linear(ReLU([sin,cos](2*pi * c_f * x_f)))   (learnable freqs)
  ple      : x_f -> Linear(piecewise_linear_bins(x_f))          (quantile bins)

Every mode returns one token of width d_token per numeric feature, so they are
drop-in interchangeable inside the FT-Transformer. The mode is a constructor flag
— that flag IS the ablation in the plan.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from . import config as C


def _piecewise_linear_encode(x, edges):
    """Vectorized PLE (Gorishniy et al.). For value x and B bins with `edges`
    (length B+1), returns a length-B vector: bins fully below x -> 1, the bin
    containing x -> its fractional position, bins above -> 0.

    x: [B_batch] tensor (one feature). edges: [B+1] tensor. Returns [B_batch, B].
    """
    left = edges[:-1]                      # [B]
    right = edges[1:]                       # [B]
    width = (right - left).clamp_min(1e-6)  # [B]
    # fractional fill of each bin for each value
    frac = (x[:, None] - left[None, :]) / width[None, :]   # [B_batch, B]
    return frac.clamp(0.0, 1.0)


class NumericEmbeddings(nn.Module):
    """Embed n_num scalar features into [B, n_num, d_token].

    mode: "linear" | "periodic" | "ple"
    ple_edges: list (len n_num) of np.ndarray bin edges, or None per feature.
               Required for "ple"; features with None fall back to a linear token.
    """

    def __init__(self, n_num, d_token, mode="linear", ple_edges=None,
                 periodic_k=C.PERIODIC_K, periodic_sigma=C.PERIODIC_SIGMA):
        super().__init__()
        assert mode in ("linear", "periodic", "ple"), mode
        self.n_num = n_num
        self.d_token = d_token
        self.mode = mode

        if mode == "linear":
            # one affine map per feature: weight [n_num, d_token], bias [n_num, d_token]
            self.weight = nn.Parameter(torch.empty(n_num, d_token))
            self.bias = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.normal_(self.weight, std=d_token ** -0.5)
            nn.init.zeros_(self.bias)

        elif mode == "periodic":
            self.k = periodic_k
            # learnable frequencies c_f: [n_num, k], init ~ N(0, sigma)
            self.freqs = nn.Parameter(torch.randn(n_num, periodic_k) * periodic_sigma)
            # project [sin(2pi c x), cos(2pi c x)] (2k dims) -> d_token, per feature
            self.proj = nn.Parameter(torch.empty(n_num, 2 * periodic_k, d_token))
            self.proj_bias = nn.Parameter(torch.zeros(n_num, d_token))
            nn.init.normal_(self.proj, std=(2 * periodic_k) ** -0.5)

        elif mode == "ple":
            assert ple_edges is not None, "ple mode needs bin edges from the encoder"
            self.linear_fallback = NumericEmbeddings(n_num, d_token, mode="linear")
            # register per-feature edge buffers and a per-feature projection
            self._has_ple = []
            self.bins_per_feat = []
            projs = []
            for f, edges in enumerate(ple_edges):
                if edges is None:
                    self._has_ple.append(False)
                    self.bins_per_feat.append(0)
                    projs.append(None)
                    continue
                e = torch.as_tensor(np.asarray(edges), dtype=torch.float32)
                n_bins = len(e) - 1
                self.register_buffer(f"edges_{f}", e)
                self._has_ple.append(True)
                self.bins_per_feat.append(n_bins)
                p = nn.Linear(n_bins, d_token)
                projs.append(p)
            # ModuleList can't hold None; keep a parallel list of indices
            self.ple_projs = nn.ModuleList([p for p in projs if p is not None])
            self._proj_index = []
            k = 0
            for has in self._has_ple:
                self._proj_index.append(k if has else -1)
                if has:
                    k += 1

    def forward(self, x_num):
        """x_num: [B, n_num] float -> [B, n_num, d_token]."""
        if self.mode == "linear":
            # [B, n_num, 1] * [n_num, d_token] -> broadcast -> [B, n_num, d_token]
            return x_num.unsqueeze(-1) * self.weight + self.bias

        if self.mode == "periodic":
            # [B, n_num, k]
            ang = 2 * math.pi * x_num.unsqueeze(-1) * self.freqs
            feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,n_num,2k]
            feats = torch.relu(feats)
            # per-feature projection via einsum: [B,n_num,2k] x [n_num,2k,d] -> [B,n_num,d]
            out = torch.einsum("bnk,nkd->bnd", feats, self.proj) + self.proj_bias
            return out

        # ple
        B = x_num.shape[0]
        tokens = []
        for f in range(self.n_num):
            if self._has_ple[f]:
                edges = getattr(self, f"edges_{f}")
                enc = _piecewise_linear_encode(x_num[:, f], edges)   # [B, n_bins]
                proj = self.ple_projs[self._proj_index[f]]
                tokens.append(proj(enc))                              # [B, d_token]
            else:
                # cyclical feature: linear token from the fallback module
                lin = self.linear_fallback
                tokens.append(x_num[:, f:f + 1] * lin.weight[f] + lin.bias[f])
        return torch.stack(tokens, dim=1)                            # [B, n_num, d_token]
