from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import S3_MEAN, S3_STD, STAGE3_NUM_FRAMES, STAGE3_NUSCENES_ROOT
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID
from src.datasets.kitti_stage3 import _label_id
from src.utils import _crop_tensor


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@lru_cache(maxsize=16)
def _scene_manifest(manifest: str, scene: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    return df[df.scene.astype(str) == str(scene)].sort_values("timestamp").reset_index(drop=True)


def _nearest_indices(timestamps: np.ndarray, center: float, frames: int) -> np.ndarray | None:
    targets = center - (np.arange(frames - 1, -1, -1, dtype=float) * 100_000.0)
    if targets[0] < timestamps[0] or targets[-1] > timestamps[-1]:
        return None
    idx = np.searchsorted(timestamps, targets)
    idx = np.clip(idx, 0, len(timestamps) - 1)
    prev = np.clip(idx - 1, 0, len(timestamps) - 1)
    return np.where(np.abs(timestamps[prev] - targets) <= np.abs(timestamps[idx] - targets), prev, idx).astype(int)


def nuscenes_stage3_clip(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read nuScenes frame: {path}")
        tensors.append(_crop_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    x = torch.stack(tensors, dim=1)
    return (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


class NuScenesStage3Dataset(Dataset):
    def __init__(self, manifest: str | Path | pd.DataFrame, root: str | Path | None = None, frames: int = STAGE3_NUM_FRAMES):
        self.manifest = Path(manifest) if not isinstance(manifest, pd.DataFrame) else None
        self.root = Path(root) if root is not None else STAGE3_NUSCENES_ROOT
        self.frames = int(frames)
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True)
        self.df = self._valid_clip_centers(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _scene_df(self, scene: str) -> pd.DataFrame:
        if self.manifest is None:
            return self.df[self.df.scene.astype(str) == str(scene)].sort_values("timestamp").reset_index(drop=True)
        return _scene_manifest(str(self.manifest), str(scene))

    def _valid_clip_centers(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for _, part in df.groupby("scene", sort=False):
            part = part.sort_values("timestamp")
            ts = part.timestamp.to_numpy(dtype=float)
            mask = [_nearest_indices(ts, float(t), self.frames) is not None for t in ts]
            keep.append(part.loc[mask])
        return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0].reset_index(drop=True)

    def clip_rows(self, row) -> pd.DataFrame:
        scene_df = self._scene_df(str(row.scene))
        ts = scene_df.timestamp.to_numpy(dtype=float)
        idx = _nearest_indices(ts, float(row.timestamp), self.frames)
        if idx is None:
            raise IndexError(f"cannot build in-scene nuScenes clip for {row.scene} timestamp={row.timestamp}")
        return scene_df.iloc[idx].reset_index(drop=True)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        clip_df = self.clip_rows(row)
        paths = [_path(self.root, str(p)) for p in clip_df.frame_path.tolist()]
        accel = _label_id(row.accel_label, ACCEL_TO_ID)
        steer = _label_id(row.steer_label, STEER_TO_ID)
        return {
            "video": nuscenes_stage3_clip(paths),
            "accel": accel,
            "steer": steer,
            "accel_label": accel,
            "steer_label": steer,
            "scene": str(row.scene),
            "frame_index": int(row.frame_idx),
            "timestamp": float(row.timestamp),
            "image_path": str(paths[-1]),
            "frame_paths": [str(p) for p in paths],
        }