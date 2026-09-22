import os
import platform
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT


def _path_env(name: str, default: Path | str) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).lower() in {"1", "true", "yes", "on"}
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
MODEL = _path_env("DACON_MODEL_ROOT", PROJECT_ROOT / "model")

STAGE1_MODEL = MODEL / "stage1"
STAGE2_MODEL = MODEL / "stage2"
STAGE2_CHECKPOINT = STAGE2_MODEL / "best.pt"
STAGE2_VIDEOMAE_MODEL = STAGE2_MODEL
STAGE2_VIDEOMAE_CHECKPOINT = STAGE2_CHECKPOINT
STAGE3_MODEL = MODEL / "stage3"
STAGE3_VJEPA_MODEL = STAGE3_MODEL / "vjepa"
STAGE3_VJEPA_CHECKPOINT = _path_env("STAGE3_VJEPA_CHECKPOINT", STAGE3_MODEL / "vitl16.pth.tar")
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
STAGE3_RAW = _path_env("STAGE3_RAW", RAW_DATA)
STAGE3_PROCESSED = _path_env("STAGE3_PROCESSED", PROCESSED_DATA / "stage3")
COMMA2K19_STAGE3_RAW = _path_env("COMMA2K19_STAGE3_RAW", RAW_DATA / "comma2k19")
COMMA2K19_STAGE3_PROCESSED = STAGE3_PROCESSED
COMMA2K19_STAGE3_MANIFEST = _path_env("COMMA2K19_STAGE3_MANIFEST", COMMA2K19_STAGE3_PROCESSED / "manifest")
COMMA2K19_STAGE3_TRAIN_MANIFEST = COMMA2K19_STAGE3_MANIFEST / "train.csv"
COMMA2K19_STAGE3_VAL_MANIFEST = COMMA2K19_STAGE3_MANIFEST / "val.csv"
COMMA2K19_STAGE3_FRAME_CACHE = _path_env("COMMA2K19_STAGE3_FRAME_CACHE", STAGE3_PROCESSED / "comma2k19_frames")

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
STAGE2_MANIFEST = STAGE2_PROCESSED / "manifest"
STAGE2_ALL_MANIFEST = STAGE2_MANIFEST / "all.csv"
STAGE2_TRAIN_MANIFEST = STAGE2_MANIFEST / "train.csv"
STAGE2_VAL_MANIFEST = STAGE2_MANIFEST / "val.csv"
STAGE2_BACKBONE = "small"

CCD_STAGE2_TOOL_DATA = TOOLS_DATA / "stage2" / "CCD-1500"
CCD_STAGE2_BOTSORT_TRACKS = CCD_STAGE2_TOOL_DATA / "tracks" / "botsort"
CCD_STAGE2_COLLISION_CANDIDATES = CCD_STAGE2_TOOL_DATA / "collision_candidates" / "collision_candidates.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
TRAIN_SOURCE_LIMIT = 4000

SIZE = 224
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
STAGE3_EPOCHS = int(os.getenv("STAGE3_EPOCHS", "3"))
STAGE3_ARCH = "mvit_v2_s"
STAGE3_MVIT_PRETRAINED = True
STAGE3_MVIT_PRETRAINED_WEIGHTS = "KINETICS400_V1"
STAGE3_MVIT_PREPROCESS = "stage3_default"
STAGE3_MVIT_BACKBONE_LR = 1e-5
STAGE3_HEAD_LR = 1e-4
STAGE3_SAMPLE_PROFILE = os.getenv("STAGE3_SAMPLE_PROFILE", "mvit_full")
STAGE3_SAMPLE_PROFILES = {
    "r2_baseline_3k": {
        "train_stride": 50,
        "val_stride": 50,
    },
    "r2_medium": {
        "train_stride": 8,
        "val_stride": 4,
    },
    "full_ft_smoke": {
        "train_stride": 8,
        "val_stride": 4,
        "train_sample_limit": 3000,
        "val_sample_limit": 1000,
    },
    "mvit_full": {
        "train_stride": 8,
        "val_stride": 8,
        "train_sample_limit": None,
        "val_sample_limit": None,
    },
}
_STAGE3_SAMPLE_PROFILE = STAGE3_SAMPLE_PROFILES[STAGE3_SAMPLE_PROFILE]
STAGE3_NUM_FRAMES = int(os.getenv("STAGE3_NUM_FRAMES", "16"))
STAGE3_OUTPUT_HZ = float(os.getenv("STAGE3_OUTPUT_HZ", "10.0"))
STAGE3_SAMPLING_HZ = float(os.getenv("STAGE3_SAMPLING_HZ", str(STAGE3_OUTPUT_HZ)))
STAGE3_SAMPLING_POLICY = os.getenv("STAGE3_SAMPLING_POLICY", "centered_timestamp_nearest")
STAGE3_PAST_FRAMES = int(os.getenv("STAGE3_PAST_FRAMES", str(STAGE3_NUM_FRAMES // 2)))
STAGE3_FUTURE_FRAMES = int(os.getenv("STAGE3_FUTURE_FRAMES", str(STAGE3_NUM_FRAMES - STAGE3_PAST_FRAMES - 1)))
STAGE3_BOUNDARY_POLICY = os.getenv("STAGE3_BOUNDARY_POLICY", "edge_clamp")
STAGE3_MAX_ALIGNMENT_ERROR_SEC = float(os.getenv("STAGE3_MAX_ALIGNMENT_ERROR_SEC", "0.06"))
STAGE3_INVERT_STEERING = _bool_env("STAGE3_INVERT_STEERING", False)
if STAGE3_NUM_FRAMES != STAGE3_PAST_FRAMES + 1 + STAGE3_FUTURE_FRAMES:
    raise ValueError("STAGE3_NUM_FRAMES must equal STAGE3_PAST_FRAMES + 1 + STAGE3_FUTURE_FRAMES")
STAGE3_ACCEL_LABEL_MODE = "current"
STAGE3_ACCEL_THRESHOLD = 0.35
STAGE3_DECEL_THRESHOLD = 0.5
STAGE3_STOP_SPEED_THRESHOLD = 0.5
STAGE3_ACCEL_WINDOW_SECONDS = 0.8
STAGE3_STEER_THRESHOLD_DEG = 5.0
STAGE3_STEER_THRESHOLD = STAGE3_STEER_THRESHOLD_DEG
STAGE3_STEER_POSITIVE_IS = "LEFT"
STAGE3_LOSS_WEIGHTS = {"accel": 1.0, "steer": 1.0}
STAGE3_SELECTION_WEIGHTS = {"accel": 0.7, "steer": 0.3}
STAGE3_CLASS_WEIGHTS = {"accel": None,
    "steer": [1.5, 1.0, 1.5],}
# LEFT      150
# STRAIGHT  200
# RIGHT     150
STAGE3_TRAIN_TEMPORAL_STRIDE = _STAGE3_SAMPLE_PROFILE["train_stride"]
STAGE3_VAL_TEMPORAL_STRIDE = _STAGE3_SAMPLE_PROFILE["val_stride"]
STAGE3_NUM_WORKERS = 1
STAGE3_PREFETCH_FACTOR = 1
STAGE3_FRAME_CACHE_JPEG_QUALITY = 92
STAGE3_FRAME_CACHE_SIZE = 224
STAGE3_TRAIN_SAMPLE_LIMIT = _STAGE3_SAMPLE_PROFILE.get("train_sample_limit", 3000)
STAGE3_VAL_SAMPLE_LIMIT = _STAGE3_SAMPLE_PROFILE.get("val_sample_limit", 500)


S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

SEED = 42




















