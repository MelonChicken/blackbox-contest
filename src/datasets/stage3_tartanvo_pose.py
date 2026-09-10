from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.config import SEED, STAGE3_TARTANVO_FEATURE_CACHE


WINDOWS_INVALID_FILENAME_CHARS = set('<>:"/\\|?*')


def sanitize_cache_filename(name: str) -> str:
    return "".join("_" if ch in WINDOWS_INVALID_FILENAME_CHARS else ch for ch in str(name))


def stage3_tartanvo_feature_path(split: str, sample_key: str) -> str:
    return str(Path(split) / f"{sanitize_cache_filename(sample_key)}.pt")


def stage3_tartanvo_legacy_join_path(split: str, sample_key: str) -> str:
    return str(Path(split) / f"{sanitize_cache_filename(str(sample_key).replace('|', '__'))}.pt")


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _row_value(row, name: str):
    if hasattr(row, name):
        return getattr(row, name)
    return row[name]


def stage3_tartanvo_sample_key(row) -> str:
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'sequence_id')}__{int(_row_value(row, 'frame_index'))}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'scene')}__{int(_row_value(row, 'frame_idx'))}__{int(_row_value(row, 'timestamp'))}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'ID')}__{int(_row_value(row, 'frame_index'))}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'route_id')}__{_row_value(row, 'segment_id')}__{int(_row_value(row, 'frame_index'))}")
    except (KeyError, AttributeError):
        value = f"{_row_value(row, 'video_path')}|{int(_row_value(row, 'frame_index'))}"
        return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _limit_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)


class Stage3TartanFeatureDataset(Dataset):
    def __init__(self, split: str, feature: str = "pose", root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE, dataset: str = "comma2k19", limit: int | None = None):
        self.split = split
        self.feature = feature
        self.source = dataset
        self.root = Path(root)
        self.base = self._base_dir(dataset, feature)
        self.index_path = self.base / f"{split}_index.csv"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"TartanVO {feature} cache index not found: {self.index_path}")
        self.df = _limit_df(pd.read_csv(self.index_path), limit)

    def _base_dir(self, dataset: str, feature: str) -> Path:
        source_base = self.root / feature / dataset
        if source_base.joinpath(f"{self.split}_index.csv").is_file():
            return source_base
        legacy = self.root / feature
        if dataset == "comma2k19" and legacy.joinpath(f"{self.split}_index.csv").is_file():
            return legacy
        if dataset == "comma2k19" and feature == "pose" and self.root.joinpath(f"{self.split}_index.csv").is_file():
            return self.root
        return source_base

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        raw_key = str(row.sample_key)
        candidates = []
        for column in ("feature_path", "path"):
            if column in self.df.columns and pd.notna(row[column]):
                rel = Path(str(row[column]))
                candidates.append(self.base / rel.with_name(sanitize_cache_filename(rel.name.replace('|', '__'))))
                candidates.append(self.base / rel.with_name(sanitize_cache_filename(rel.name)))
                candidates.append(self.base / rel)
        candidates.append(self.base / stage3_tartanvo_legacy_join_path(self.split, raw_key))
        candidates.append(self.base / stage3_tartanvo_feature_path(self.split, raw_key))
        candidates.append(self.base / self.split / f"{raw_key}.pt")
        path = next((candidate for candidate in candidates if _is_file(candidate)), candidates[0])
        item = torch.load(path, map_location="cpu", weights_only=False)
        feature = item.get("feature", item.get("pose"))
        if feature is None:
            raise KeyError(f"cache sample has neither feature nor pose: {path}")
        source = str(item.get("source", item.get("dataset", self.source)))
        sample_key = str(row.sample_key)
        accel = int(item.get("accel", item["accel_label"]))
        steer = int(item.get("steer", item["steer_label"]))
        sequence_id = item.get("sequence_id", getattr(row, "sequence_id", ""))
        frame_index = item.get("frame_index", getattr(row, "frame_index", -1))
        return {
            "feature": feature.to(torch.float32),
            "accel": accel,
            "steer": steer,
            "accel_label": accel,
            "steer_label": steer,
            "source": source,
            "sample_key": sample_key,
            "sequence_id": "" if sequence_id is None else str(sequence_id),
            "frame_index": -1 if frame_index is None else int(frame_index),
        }


class Stage3MixedTartanFeatureDataset(ConcatDataset):
    def __init__(self, split: str, feature: str = "latent", root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE, sources: tuple[str, ...] = ("comma2k19", "kitti"), limits: dict[str, int | None] | None = None):
        self.sources = tuple(sources)
        self.source_datasets = {source: Stage3TartanFeatureDataset(split, feature, root, dataset=source, limit=(limits or {}).get(source)) for source in self.sources}
        super().__init__(list(self.source_datasets.values()))

    @property
    def source_counts(self) -> dict[str, int]:
        return {source: len(ds) for source, ds in self.source_datasets.items()}


class Stage3TartanPoseDataset(Stage3TartanFeatureDataset):
    def __init__(self, split: str, root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE, dataset: str = "comma2k19", limit: int | None = None):
        super().__init__(split, feature="pose", root=root, dataset=dataset, limit=limit)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["pose"] = item["feature"]
        return item
