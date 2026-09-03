from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import COMMA2K19_STAGE3_FRAME_CACHE, COMMA2K19_STAGE3_RAW, S3_MEAN, S3_STD, STAGE3_NUM_FRAMES, STAGE3_RAW
from src.utils import _crop_tensor, clip

ACCEL_TO_ID = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
STEER_TO_ID = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}


def stage3_video_clip(path: str | Path, frame_index: int, frames: int = STAGE3_NUM_FRAMES) -> torch.Tensor:
    x, _ = clip(Path(path), frames, int(frame_index))
    return (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


def _clip_indices(center: int, total: int, frames: int = STAGE3_NUM_FRAMES) -> np.ndarray:
    return np.clip(int(center) - frames // 2 + np.arange(frames), 0, total - 1).astype(int)


@lru_cache(maxsize=512)
def _cached_frame_lookup(cache_dir: str) -> tuple[int, dict[int, Path]]:
    root = Path(cache_dir)
    meta = pd.read_csv(root / "frames.csv")
    lookup = {int(row.original_frame_index): root / str(row.cached_path) for row in meta.itertuples(index=False)}
    return max(lookup) + 1, lookup


def stage3_cached_clip(cache_dir: str | Path, frame_index: int, frames: int = STAGE3_NUM_FRAMES) -> torch.Tensor:
    total, lookup = _cached_frame_lookup(str(cache_dir))
    indices = _clip_indices(frame_index, total, frames)
    tensors = []
    for idx in indices:
        path = lookup.get(int(idx))
        if path is None:
            raise FileNotFoundError(f"cached frame {idx} missing under {cache_dir}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read cached frame: {path}")
        tensors.append(_crop_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    x = torch.stack(tensors, dim=1)
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
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        frames: int = STAGE3_NUM_FRAMES,
        cache_root: str | Path | None = COMMA2K19_STAGE3_FRAME_CACHE,
    ):
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else COMMA2K19_STAGE3_RAW
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.frames = int(frames)
        self.df = pd.read_csv(self.manifest)

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _cache_dir(self, row) -> Path | None:
        if self.cache_root is None:
            return None
        if "route_id" in row and "segment_id" in row:
            cache_dir = self.cache_root / str(row.route_id) / str(row.segment_id)
        else:
            cache_dir = self.cache_root / Path(str(row.video_path)).with_suffix("")
        return cache_dir if (cache_dir / "frames.csv").is_file() else None

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        video_path = self._video_path(str(row.video_path))
        frame_index = int(row.frame_index)
        cache_dir = self._cache_dir(row)
        try:
            video = stage3_cached_clip(cache_dir, frame_index, self.frames) if cache_dir else stage3_video_clip(video_path, frame_index, self.frames)
        except FileNotFoundError:
            if cache_dir is None:
                raise
            video = stage3_video_clip(video_path, frame_index, self.frames)
        return {
            "video": video,
            "accel_label": int(row.accel_label),
            "steer_label": int(row.steer_label),
            "timestamp": float(row.timestamp),
            "video_path": str(video_path),
        }
