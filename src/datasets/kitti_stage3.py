from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import KITTI_STAGE3_RAW, S3_MEAN, S3_STD, STAGE3_NUM_FRAMES, STAGE3_TARTANVO_HEIGHT, STAGE3_TARTANVO_WIDTH
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, _clip_indices
from src.utils import _crop_tensor


def _label_id(value, mapping: dict[str, int]) -> int:
    if isinstance(value, str) and not value.isdigit():
        return mapping[value]
    return int(value)


def _valid_series(values: pd.Series) -> pd.Series:
    if values.dtype == object:
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    return values.astype(bool)


@lru_cache(maxsize=128)
def _sequence_manifest(manifest: str, sequence_id: str) -> tuple[tuple[Path, ...], tuple[bool, ...]]:
    cols = ["sequence_id", "frame_index", "image_path"]
    df = pd.read_csv(manifest, keep_default_na=False)
    part = df[df.sequence_id.astype(str) == str(sequence_id)].sort_values("frame_index")
    valid = _valid_series(part.motion_valid).tolist() if "motion_valid" in part.columns else [True] * len(part)
    return tuple(Path(p) for p in part.image_path.tolist()), tuple(bool(v) for v in valid)


def kitti_stage3_clip(paths: tuple[Path, ...], frame_index: int, frames: int = STAGE3_NUM_FRAMES) -> torch.Tensor:
    indices = _clip_indices(frame_index, len(paths), frames)
    tensors = []
    for idx in indices:
        bgr = cv2.imread(str(paths[int(idx)]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read KITTI frame: {paths[int(idx)]}")
        tensors.append(_crop_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    x = torch.stack(tensors, dim=1)
    return (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


class KittiStage3Dataset(Dataset):
    def __init__(self, manifest: str | Path | pd.DataFrame, root: str | Path | None = None, frames: int = STAGE3_NUM_FRAMES, exclude_invalid_motion: bool = True):
        self.manifest = Path(manifest) if not isinstance(manifest, pd.DataFrame) else None
        self.df = pd.read_csv(manifest, keep_default_na=False) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True)
        self.root = Path(root) if root is not None else KITTI_STAGE3_RAW
        self.frames = int(frames)
        if exclude_invalid_motion and "motion_valid" in self.df.columns:
            self.df = self._valid_clip_centers(self.df).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _sequence_data(self, sequence_id: str) -> tuple[tuple[Path, ...], tuple[bool, ...]]:
        if self.manifest is None:
            part = self.df[self.df.sequence_id.astype(str) == str(sequence_id)].sort_values("frame_index")
            valid = _valid_series(part.motion_valid).tolist() if "motion_valid" in part.columns else [True] * len(part)
            return tuple(self._path(p) for p in part.image_path.tolist()), tuple(bool(v) for v in valid)
        paths, valid = _sequence_manifest(str(self.manifest), str(sequence_id))
        return tuple(self._path(str(p)) for p in paths), valid

    def _valid_clip_centers(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for _, part in df.groupby("sequence_id", sort=False):
            part = part.sort_values("frame_index")
            valid = _valid_series(part.motion_valid).to_numpy(dtype=bool)
            total = len(part)
            mask = []
            for frame_index in part.frame_index.astype(int):
                idx = _clip_indices(int(frame_index), total, self.frames)
                mask.append(bool(valid[idx].all()))
            keep.append(part.loc[mask])
        return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]

    def _intrinsics(self, row) -> torch.Tensor:
        width = float(getattr(row, "image_width", 0) or 0)
        height = float(getattr(row, "image_height", 0) or 0)
        fx, fy, cx, cy = (float(getattr(row, name, 0.0)) for name in ("fx", "fy", "cx", "cy"))
        if width > 0 and height > 0 and fx > 0 and fy > 0:
            fx *= STAGE3_TARTANVO_WIDTH / width
            fy *= STAGE3_TARTANVO_HEIGHT / height
            cx *= STAGE3_TARTANVO_WIDTH / width
            cy *= STAGE3_TARTANVO_HEIGHT / height
        return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        frame_index = int(row.frame_index)
        paths, _ = self._sequence_data(str(row.sequence_id))
        video = kitti_stage3_clip(paths, frame_index, self.frames)
        accel = _label_id(row.accel_label, ACCEL_TO_ID)
        steer = _label_id(row.steer_label, STEER_TO_ID)
        return {
            "video": video,
            "accel": accel,
            "steer": steer,
            "accel_label": accel,
            "steer_label": steer,
            "sequence_id": str(row.sequence_id),
            "frame_index": frame_index,
            "image_path": str(self._path(str(row.image_path))),
            "intrinsics": self._intrinsics(row),
        }
