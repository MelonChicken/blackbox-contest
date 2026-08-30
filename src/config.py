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

STAGE1_DATA = DATA / "stage1"
STAGE2_DATA = DATA / "stage2"
STAGE3_DATA = DATA / "stage3"

STAGE1_MODEL = MODEL / "stage1"
STAGE2_MODEL = MODEL / "stage2"
STAGE3_MODEL = MODEL / "stage3"

AIHUB_STAGE1_RAW = DATA_ROOT / "raw" / "aihub597"
AIHUB_STAGE1_MANIFEST = DATA_ROOT / "processed" / "stage1" / "aihub597" / "manifest"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1
TRAIN_SOURCE_LIMIT = 4000

SIZE = 224
BATCH_SIZE = 2

S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

SEED = 42
