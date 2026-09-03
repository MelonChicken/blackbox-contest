from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import COMMA2K19_STAGE3_RAW, S3_MEAN, S3_STD, STAGE3_NUM_FRAMES, STAGE3_RAW
from src.utils import clip

ACCEL_TO_ID = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
STEER_TO_ID = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}


def stage3_video_clip(path: str | Path, frame_index: int, frames: int = STAGE3_NUM_FRAMES) -> torch.Tensor:
    x, _ = clip(Path(path), frames, int(frame_index))
    return (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


class Stage3DaconDataset(Dataset):
    def __init__(self, labels: str | Path | pd.DataFrame, video_root: str | Path = STAGE3_RAW / "videos"):
        self.df = pd.read_csv(labels) if not isinstance(labels, pd.DataFrame) else labels.reset_index(drop=True)
        self.video_root = Path(video_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        video_path = self.video_root / f"{row.ID}.mp4"
        return {
            "video": stage3_video_clip(video_path, int(row.frame_index)),
            "accel_label": ACCEL_TO_ID[row.accel_label],
            "steer_label": STEER_TO_ID[row.steer_label],
            "timestamp": float(row.get("time_seconds", 0.0)),
            "video_path": str(video_path),
        }


class Comma2k19Stage3Dataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path | None = None, frames: int = STAGE3_NUM_FRAMES):
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else COMMA2K19_STAGE3_RAW
        self.frames = int(frames)
        self.df = pd.read_csv(self.manifest)

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        video_path = self._video_path(str(row.video_path))
        frame_index = int(row.frame_index)
        return {
            "video": stage3_video_clip(video_path, frame_index, self.frames),
            "accel_label": int(row.accel_label),
            "steer_label": int(row.steer_label),
            "timestamp": float(row.timestamp),
            "video_path": str(video_path),
        }
