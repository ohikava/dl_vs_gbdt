"""FT-Transformer (Gorishniy et al. 2021, "Revisiting Deep Learning Models for
Tabular Data").

Pipeline:
  1. Feature tokenizer: every feature (categorical AND numeric) becomes one
     d_token vector. Categoricals via nn.Embedding; numerics via NumericEmbeddings
     in one of three modes (linear / periodic / ple).
  2. Prepend a learnable [CLS] token.
  3. Run the token sequence through a pre-LN Transformer encoder.
  4. Classify from the final [CLS] representation.

The categorical tokenizer reuses the central project's convention: vocab includes
PAD/UNK reserved ids and Embedding uses padding_idx=PAD_ID. Numeric tokenization
is the swappable knob that turns "raw scalar into the transformer" (weak) into
"per-feature vector" (strong), which is the whole numerical-embeddings story.
"""
import torch
import torch.nn as nn

from . import config as C
from .num_embeddings import NumericEmbeddings


class FeatureTokenizer(nn.Module):
    def __init__(self, cat_vocab_sizes, n_num, d_token,
                 num_mode="linear", ple_edges=None):
        super().__init__()
        self.cat_emb = nn.ModuleList([
            nn.Embedding(v, d_token, padding_idx=C.PAD_ID) for v in cat_vocab_sizes
        ])
        self.num_emb = NumericEmbeddings(n_num, d_token, mode=num_mode,
                                         ple_edges=ple_edges)
        self.n_cat = len(cat_vocab_sizes)
        self.n_num = n_num

    def forward(self, x_cat, x_num):
        toks = []
        for j, emb in enumerate(self.cat_emb):
            toks.append(emb(x_cat[:, j]).unsqueeze(1))   # [B, 1, d]
        cat_tok = torch.cat(toks, dim=1) if toks else None  # [B, n_cat, d]
        num_tok = self.num_emb(x_num)                       # [B, n_num, d]
        if cat_tok is None:
            return num_tok
        return torch.cat([cat_tok, num_tok], dim=1)         # [B, n_cat+n_num, d]


class FTTransformer(nn.Module):
    def __init__(self, cat_vocab_sizes, n_num,
                 d_token=C.D_TOKEN, n_heads=C.N_HEADS, n_layers=C.N_LAYERS,
                 ffn_factor=C.FFN_FACTOR, dropout=C.DROPOUT,
                 attn_dropout=C.ATTN_DROPOUT, num_mode="linear", ple_edges=None):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_vocab_sizes, n_num, d_token,
                                          num_mode=num_mode, ple_edges=ple_edges)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * ffn_factor,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,  # pre-LN: stable for tabular depths
        )
        # set attention dropout (TransformerEncoderLayer ties it to `dropout`;
        # override the self-attn module's dropout explicitly)
        layer.self_attn.dropout = attn_dropout
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_cat, x_num):
        tok = self.tokenizer(x_cat, x_num)                  # [B, T, d]
        B = tok.shape[0]
        cls = self.cls.expand(B, -1, -1)                    # [B, 1, d]
        seq = torch.cat([cls, tok], dim=1)                  # [B, 1+T, d]
        enc = self.encoder(seq)
        cls_out = self.head_norm(enc[:, 0])                 # [B, d]
        return self.head(cls_out).squeeze(-1)               # [B]
