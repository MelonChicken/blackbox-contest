from __future__ import annotations

import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.config import (
    SEED,
    STAGE3_NUM_FRAMES,
    STAGE3_SEGMENT_FEATURE_LRU_SIZE,
    STAGE3_TARTANVO_FEATURE_CACHE,
)
from src.tools.stage3_comma_manifest import active_manifest_path, active_subset_name, segment_cache_index_name


WINDOWS_INVALID_FILENAME_CHARS = set('<>:"/\\|?*')
TARTANVO_ALIGNMENT_VERSION = "comma_frame_times_nearest_v1"
TARTANVO_SEGMENT_FEATURE_VERSION = "tartanvo_segment_latent_v1"
TARTANVO_SEGMENT_LAYOUT = "segment"


def sanitize_cache_filename(name: str) -> str:
    return "".join("_" if ch in WINDOWS_INVALID_FILENAME_CHARS else ch for ch in str(name))


def stage3_tartanvo_feature_path(split: str, sample_key: str) -> str:
    return str(Path(split) / f"{sanitize_cache_filename(sample_key)}.pt")


def stage3_tartanvo_segment_key(row) -> str:
    return sanitize_cache_filename(f"{_row_value(row, 'route_id')}__{_row_value(row, 'segment_id')}")


def stage3_tartanvo_window(frame_index: int, frames: int = STAGE3_NUM_FRAMES) -> tuple[int, int]:
    start = int(frame_index) - frames // 2
    return start, start + frames - 1


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


def _row_frame_index(row) -> int:
    try:
        return int(_row_value(row, "video_frame_index"))
    except (KeyError, AttributeError):
        return int(_row_value(row, "frame_index"))


def stage3_tartanvo_sample_key(row) -> str:
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'sequence_id')}__{_row_frame_index(row)}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'scene')}__{int(_row_value(row, 'frame_idx'))}__{int(_row_value(row, 'timestamp'))}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'ID')}__{_row_frame_index(row)}")
    except (KeyError, AttributeError):
        pass
    try:
        return sanitize_cache_filename(f"{_row_value(row, 'route_id')}__{_row_value(row, 'segment_id')}__{_row_frame_index(row)}")
    except (KeyError, AttributeError):
        value = f"{_row_value(row, 'video_path')}|{_row_frame_index(row)}"
        return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _limit_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else "segment_id" if "segment_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


class Stage3TartanFeatureDataset(Dataset):
    def __init__(self, split: str, feature: str = "pose", root: str | Path = STAGE3_TARTANVO_FEATURE_CACHE, dataset: str = "comma2k19", limit: int | None = None):
        self.split = split
        self.feature = feature
        self.source = dataset
        self.root = Path(root)
        self.base = self._base_dir(dataset, feature)
        self.segment_layout = dataset == "comma2k19" and feature == "latent" and self.base.parent.name == "segment_latent"
        self._segment_cache: OrderedDict[str, tuple[torch.Tensor, dict]] = OrderedDict()
        if self.segment_layout:
            self._init_segment_layout(limit)
            return
        self.index_path = self.base / f"{split}_index.csv"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"TartanVO {feature} cache index not found: {self.index_path}")
        self._check_alignment_metadata(dataset)
        self.df = _limit_df(pd.read_csv(self.index_path), limit)

    def _init_segment_layout(self, limit: int | None) -> None:
        self.subset_name = active_subset_name(self.split)
        self.index_path = self.base / segment_cache_index_name(self.split, self.subset_name)
        meta_path = self.base / "metadata.json"
        if not self.index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"comma2k19 segment TartanVO cache not found: {self.index_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_layout") != TARTANVO_SEGMENT_LAYOUT or meta.get("feature_version") != TARTANVO_SEGMENT_FEATURE_VERSION:
            raise RuntimeError(f"comma2k19 TartanVO segment cache version mismatch: {meta_path}")
        if meta.get("alignment_version") != TARTANVO_ALIGNMENT_VERSION:
            raise RuntimeError(f"comma2k19 TartanVO cache alignment mismatch: {meta_path}")
        index = pd.read_csv(self.index_path)
        self.segment_index = {str(row.segment_key): row._asdict() for row in index.itertuples(index=False)}
        manifest = active_manifest_path(self.split)
        if not manifest.is_file():
            raise FileNotFoundError(f"missing comma2k19 Stage3 manifest for {self.split}: {manifest}")
        df = pd.read_csv(manifest)
        versions = set(df.get("alignment_version", pd.Series(dtype=str)).dropna().astype(str))
        if versions != {TARTANVO_ALIGNMENT_VERSION}:
            raise RuntimeError(f"comma2k19 manifest must use {TARTANVO_ALIGNMENT_VERSION}; found {sorted(versions) or [None]}")
        stride = STAGE3_TRAIN_TEMPORAL_STRIDE if self.split == "train" else STAGE3_VAL_TEMPORAL_STRIDE
        df = _stride_manifest(df, stride)
        df["segment_key"] = [stage3_tartanvo_segment_key(row) for row in df.itertuples(index=False)]
        starts, ends, valids = [], [], []
        for row in df.itertuples(index=False):
            start, end = stage3_tartanvo_window(_row_frame_index(row))
            entry = self.segment_index.get(str(row.segment_key))
            valid = entry is not None and start >= 0 and end <= int(entry["num_pairs"])
            starts.append(start)
            ends.append(end)
            valids.append(valid)
        df["tartanvo_window_start"] = starts
        df["tartanvo_window_end"] = ends
        df["tartanvo_valid"] = valids
        dropped = int((~df["tartanvo_valid"]).sum())
        if dropped:
            warnings.warn(f"ignored {dropped} comma2k19 samples whose 16-frame TartanVO window crosses segment cache bounds")
        old_index = self.root / self.feature / self.source / f"{self.split}_index.csv"
        if old_index.is_file():
            warnings.warn(f"ignoring legacy sample-level comma2k19 TartanVO cache: {old_index}")
        self.df = _limit_df(df[df.tartanvo_valid].reset_index(drop=True), limit)

    def _check_alignment_metadata(self, dataset: str) -> None:
        if dataset != "comma2k19":
            return
        meta_path = self.base / f"{self.split}_metadata.json"
        if not meta_path.is_file():
            raise RuntimeError(f"comma2k19 TartanVO cache is invalid until regenerated with comma_frame_times_nearest_v1 alignment: {self.index_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("manifest_alignment_version") != TARTANVO_ALIGNMENT_VERSION:
            raise RuntimeError(f"comma2k19 TartanVO cache is invalid until regenerated with comma_frame_times_nearest_v1 alignment: {self.index_path}")

    def _base_dir(self, dataset: str, feature: str) -> Path:
        segment_base = self.root / "segment_latent" / dataset
        if dataset == "comma2k19" and feature == "latent" and (segment_base.joinpath(segment_cache_index_name(self.split, active_subset_name(self.split))).is_file() or segment_base.joinpath(f"{self.split}_index.csv").is_file()):
            return segment_base
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
        if self.segment_layout:
            return self._getitem_segment(index)
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
        frame_index = item.get("video_frame_index", item.get("frame_index", getattr(row, "video_frame_index", getattr(row, "frame_index", -1))))
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

    def _load_segment(self, segment_key: str) -> tuple[torch.Tensor, dict]:
        if segment_key in self._segment_cache:
            self._segment_cache.move_to_end(segment_key)
            return self._segment_cache[segment_key]
        entry = self.segment_index[segment_key]
        path = self.base / str(entry["feature_path"])
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("alignment_version") != TARTANVO_ALIGNMENT_VERSION or item.get("feature_version") != TARTANVO_SEGMENT_FEATURE_VERSION:
            raise RuntimeError(f"TartanVO segment cache version mismatch: {path}")
        features = item["features"].to(torch.float32)
        if features.ndim != 2 or int(features.shape[-1]) != 1536:
            raise RuntimeError(f"invalid TartanVO segment feature shape {tuple(features.shape)}: {path}")
        self._segment_cache[segment_key] = (features, item)
        while len(self._segment_cache) > STAGE3_SEGMENT_FEATURE_LRU_SIZE:
            self._segment_cache.popitem(last=False)
        return features, item

    def _getitem_segment(self, index: int) -> dict:
        row = self.df.iloc[index]
        segment_key = str(row.segment_key)
        features, _ = self._load_segment(segment_key)
        start, end = int(row.tartanvo_window_start), int(row.tartanvo_window_end)
        feature = features[start:end]
        if tuple(feature.shape) != (STAGE3_NUM_FRAMES - 1, 1536):
            raise RuntimeError(f"bad TartanVO slice shape {tuple(feature.shape)} for {segment_key}:{start}:{end}")
        accel = int(row.accel_label)
        steer = int(row.steer_label)
        return {
            "feature": feature,
            "accel": accel,
            "steer": steer,
            "accel_label": accel,
            "steer_label": steer,
            "source": self.source,
            "sample_key": stage3_tartanvo_sample_key(row),
            "sequence_id": segment_key,
            "frame_index": _row_frame_index(row),
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
