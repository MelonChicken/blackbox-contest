from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import STAGE3_TARTANVO_FEATURE_CACHE


def _row_value(row, name: str):
    if hasattr(row, name):
        return getattr(row, name)
    return row[name]


def stage3_tartanvo_sample_key(row) -> str:
    try:
        return f"{_row_value(row, 'route_id')}__{_row_value(row, 'segment_id')}__{int(_row_value(row, 'frame_index'))}"
    except (KeyError, AttributeError):
        value = f"{_row_value(row, 'video_path')}|{int(_row_value(row, 'frame_index'))}"
        return hashlib.sha1(value.encode("utf-8")).hexdigest()


class Stage3TartanFeatureDataset(Dataset):
    def __init__(self, split: str, feature: str = "pose", root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE):
        self.split = split
        self.feature = feature
        self.root = Path(root)
        self.base = self.root / feature
        self.index_path = self.base / f"{split}_index.csv"
        if not self.index_path.is_file() and feature == "pose":
            self.base = self.root
            self.index_path = self.root / f"{split}_index.csv"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"TartanVO {feature} cache index not found: {self.index_path}")
        self.df = pd.read_csv(self.index_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        path = self.base / self.split / f"{row.sample_key}.pt"
        item = torch.load(path, map_location="cpu", weights_only=False)
        feature = item.get("feature", item.get("pose"))
        if feature is None:
            raise KeyError(f"cache sample has neither feature nor pose: {path}")
        return {
            "feature": feature.to(torch.float32),
            "accel_label": int(item["accel_label"]),
            "steer_label": int(item["steer_label"]),
        }


class Stage3TartanPoseDataset(Stage3TartanFeatureDataset):
    def __init__(self, split: str, root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE):
        super().__init__(split, feature="pose", root=root)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["pose"] = item["feature"]
        return item
