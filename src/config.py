"""Central config: paths, feature schema, hyperparameters.

One place to change things so data / encoder / models / training stay in sync.
Conventions are deliberately identical to the central project
(`rec_sys_anti_frod/src/config.py`) so the encoder ports over cleanly.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Reuse the Sparkov CSVs already downloaded for the central project instead of
# duplicating ~0.5 GB. Override DL_VS_GBDT_DATA_DIR to point elsewhere.
import os

_DEFAULT_DATA = ROOT.parent / "rec_sys_anti_frod" / "data"
DATA_DIR = Path(os.environ.get("DL_VS_GBDT_DATA_DIR", _DEFAULT_DATA))
ART_DIR = ROOT / "artifacts"

RAW_TRAIN = DATA_DIR / "fraudTrain.csv"
RAW_TEST = DATA_DIR / "fraudTest.csv"

# ----- temporal split -----
# train file spans 2019-01-01 .. 2020-06-21. Carve its tail as validation;
# the test file is always the test split. No random shuffling across time.
VAL_CUTOFF = "2020-04-01"

# ----- categorical encoding (shared reserved ids) -----
PAD_ID = 0
UNK_ID = 1
N_RESERVED = 2  # real categories are encoded starting at id = N_RESERVED

# ----- per-transaction feature schema -----
CAT_COLS = ["category", "merchant", "gender", "state", "job"]

# Numeric features. Cyclical *_sin/_cos are already bounded in [-1, 1] and are
# NOT standardized; everything else gets train-fit standardization.
NUM_COLS = [
    "amt_log", "dist", "age", "city_pop_log",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
NUM_STANDARDIZE = ["amt_log", "dist", "age", "city_pop_log"]
# Numeric features that get a quantile-binned (PLE) representation. The cyclical
# ones are excluded — piecewise-linear binning of a sin wave is meaningless.
NUM_PLE_COLS = ["amt_log", "dist", "age", "city_pop_log"]

TARGET = "is_fraud"

# ----- FT-Transformer hyperparameters -----
D_TOKEN = 64
N_HEADS = 8
N_LAYERS = 3
FFN_FACTOR = 2          # FFN hidden = D_TOKEN * FFN_FACTOR
DROPOUT = 0.1
ATTN_DROPOUT = 0.1
PLE_BINS = 16           # number of quantile bins for piecewise-linear encoding
PERIODIC_K = 24         # number of learnable frequencies per feature (periodic mode)
PERIODIC_SIGMA = 0.05   # init scale of periodic frequencies

# ----- training -----
BATCH_SIZE = 1024
LR = 2e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 8
LOSS = "focal"          # one of: "focal", "weighted_bce", "bce"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25      # weight on the positive (fraud) class
SEED = 42

# ----- evaluation -----
# recall measured at these fixed false-positive rates (low FPR matters in fraud)
FPR_TARGETS = [0.001, 0.005, 0.01, 0.05]
