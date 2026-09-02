import os
import platform
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT

SYSTEM = platform.system()

_data_root = os.getenv("DACON_DATA_ROOT")
if _data_root:
    DATA_ROOT = Path(_data_root).expanduser()
elif SYSTEM == "Windows":
    DATA_ROOT = PROJECT_ROOT / "data"
elif SYSTEM == "Linux":
    DATA_ROOT = Path("/data")
else:
    raise RuntimeError(f"Unsupported operating system: {SYSTEM}")

DATA = DATA_ROOT
MODEL = PROJECT_ROOT / "model"

STAGE1_MODEL = MODEL / "stage1"
STAGE2_MODEL = MODEL / "stage2"
STAGE2_VIDEOMAE_MODEL = MODEL / "stage2_videomae"
STAGE2_VIDEOMAE_CHECKPOINT = STAGE2_VIDEOMAE_MODEL / "best.pt"
STAGE3_MODEL = MODEL / "stage3"
# ============================================================
# Data directories
# ============================================================

RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"

TOOLS_DATA = PROJECT_ROOT / "src" / "tools" / "data"


# Stage 1
STAGE1_RAW = RAW_DATA / "stage1"
STAGE1_PROCESSED = PROCESSED_DATA / "stage1"

# Stage 2
STAGE2_RAW = RAW_DATA / "stage2"
STAGE2_PROCESSED = PROCESSED_DATA / "stage2"

# Stage 3
STAGE3_RAW = RAW_DATA / "stage3"
STAGE3_PROCESSED = PROCESSED_DATA / "stage3"

# Backward-compatible aliases for stage raw data roots.
STAGE1_DATA = STAGE1_RAW
STAGE2_DATA = STAGE2_RAW
STAGE3_DATA = STAGE3_RAW


# ============================================================
# Stage 1 datasets
# ============================================================

AIHUB_STAGE1_RAW = STAGE1_RAW / "aihub597"
AIHUB_STAGE1_PROCESSED = STAGE1_PROCESSED / "aihub597"
AIHUB_STAGE1_MANIFEST = AIHUB_STAGE1_PROCESSED / "manifest"

DLC_STAGE1_RAW = STAGE1_RAW / "dlc2021"
DLC_STAGE1_PROCESSED = STAGE1_PROCESSED / "dlc2021"
DLC_STAGE1_MANIFEST = DLC_STAGE1_PROCESSED / "manifest"

CCD_STAGE2_RAW = STAGE2_RAW / "CCD-1500"
CCD_STAGE2_PROCESSED = STAGE2_PROCESSED / "CCD-1500"
CCD_STAGE2_MANIFEST = CCD_STAGE2_PROCESSED / "manifest"
CCD_STAGE2_VIDEOMAE_MANIFEST = CCD_STAGE2_MANIFEST / "videomae"
CCD_STAGE2_VIDEOMAE_ALL_MANIFEST = CCD_STAGE2_VIDEOMAE_MANIFEST / "all.csv"
CCD_STAGE2_VIDEOMAE_TRAIN_MANIFEST = CCD_STAGE2_VIDEOMAE_MANIFEST / "train.csv"
CCD_STAGE2_VIDEOMAE_VAL_MANIFEST = CCD_STAGE2_VIDEOMAE_MANIFEST / "val.csv"

CCD_STAGE2_TOOL_DATA = TOOLS_DATA / "stage2" / "CCD-1500"
CCD_STAGE2_BOTSORT_TRACKS = CCD_STAGE2_TOOL_DATA / "tracks" / "botsort"
CCD_STAGE2_COLLISION_CANDIDATES = CCD_STAGE2_TOOL_DATA / "collision_candidates" / "collision_candidates.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
TRAIN_SOURCE_LIMIT = 4000

SIZE = 224
BATCH_SIZE = 2

S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

SEED = 42

