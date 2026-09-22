from __future__ import annotations

from pathlib import Path

import cv2
import torch

from src.config import SIZE
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset, _cached_frame_lookup
from src.datasets.stage3_sampling import parse_clip_frame_indices
from src.utils import _crop_tensor

VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def stage3_vjepa_cached_clip(cache_dir: str | Path, frame_indices: list[int]) -> torch.Tensor:
    _, lookup = _cached_frame_lookup(str(cache_dir))
    frames = []
    for idx in frame_indices:
        path = lookup.get(int(idx))
        if path is None:
            raise FileNotFoundError(f"cached frame {idx} missing under {cache_dir}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read cached frame: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] == (SIZE, SIZE):
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0)
        else:
            frames.append(_crop_tensor(rgb))
    x = torch.stack(frames, dim=1)
    return (x - VJEPA_MEAN[:, None, None, None]) / VJEPA_STD[:, None, None, None]


class Comma2k19Stage3VJEPADataset(Comma2k19Stage3Dataset):
    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        clip_indices = parse_clip_frame_indices(str(row.clip_frame_indices), self.frames)
        cache_dir = self._cache_dir(row)
        if self.cache_root is None or cache_dir is None:
            raise self._missing_cache_error(self._expected_cache_dir(row))
        return {
            "video": stage3_vjepa_cached_clip(cache_dir, clip_indices),
            "accel_label": int(row.accel_label),
            "steer_label": int(row.steer_label),
        }
