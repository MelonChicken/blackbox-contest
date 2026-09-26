from __future__ import annotations

from functools import lru_cache
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    COMMA2K19_STAGE3_FRAME_CACHE,
    COMMA2K19_STAGE3_RAW,
    S3_MEAN,
    S3_STD,
    SIZE,
    STAGE3_BOUNDARY_POLICY,
    STAGE3_FUTURE_FRAMES,
    STAGE3_NUM_FRAMES,
    STAGE3_OUTPUT_HZ,
    STAGE3_PAST_FRAMES,
    STAGE3_SAMPLING_HZ,
    STAGE3_SAMPLING_POLICY,
)
from src.datasets.stage3_labels import ACCEL_TO_ID, STEER_TO_ID
from src.datasets.stage3_sampling import (
    parse_clip_frame_indices,
    read_manifest_metadata,
    sampling_metadata,
    validate_sampling_metadata,
)
from src.utils import _crop_tensor, video_frames


def _normalize_clip(x: torch.Tensor) -> torch.Tensor:
    return (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]


def stage3_video_clip_indices(path: str | Path, frame_indices: list[int]) -> torch.Tensor:
    frames = video_frames(Path(path))
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")
    total = len(frames)
    bad = [idx for idx in frame_indices if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"clip frame index out of range for {path}: {bad[:8]} total={total}")
    x = torch.stack([_crop_tensor(frames[int(i)]) for i in frame_indices], 1)
    return _normalize_clip(x)


@lru_cache(maxsize=512)
def _cached_frame_lookup(cache_dir: str) -> tuple[int, dict[int, Path]]:
    root = Path(cache_dir)
    meta = pd.read_csv(root / "frames.csv")
    lookup = {int(row.original_frame_index): root / str(row.cached_path) for row in meta.itertuples(index=False)}
    return max(lookup) + 1, lookup


@lru_cache(maxsize=4096)
def _cached_frame_path(cache_dir: str, frame_index: int, max_fallback_delta: int = 2) -> Path:
    _, lookup = _cached_frame_lookup(cache_dir)
    frame_index = int(frame_index)
    path = lookup.get(frame_index)
    if path is not None:
        return path
    available = sorted(lookup)
    position = bisect_left(available, frame_index)
    candidates = available[max(0, position - 1):position + 1]
    if not candidates:
        raise FileNotFoundError(f"cached frame {frame_index} missing under {cache_dir}")
    nearest = min(candidates, key=lambda value: abs(value - frame_index))
    if abs(nearest - frame_index) > int(max_fallback_delta):
        raise FileNotFoundError(
            f"cached frame {frame_index} missing under {cache_dir}; nearest={nearest} exceeds fallback delta {max_fallback_delta}"
        )
    return lookup[nearest]


def stage3_cached_clip_indices(cache_dir: str | Path, frame_indices: list[int]) -> torch.Tensor:
    tensors = []
    for idx in frame_indices:
        path = _cached_frame_path(str(cache_dir), int(idx))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read cached frame: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] == (SIZE, SIZE):
            tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0)
        else:
            tensors.append(_crop_tensor(rgb))
    return _normalize_clip(torch.stack(tensors, dim=1))


def _expected_metadata(frames: int) -> dict:
    return sampling_metadata(
        sampling_hz=STAGE3_SAMPLING_HZ,
        sampling_policy=STAGE3_SAMPLING_POLICY,
        num_frames=frames,
        past_frames=STAGE3_PAST_FRAMES,
        future_frames=STAGE3_FUTURE_FRAMES,
        boundary_policy=STAGE3_BOUNDARY_POLICY,
        output_hz=STAGE3_OUTPUT_HZ,
    )


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
        validate_sampling_metadata(read_manifest_metadata(self.manifest), _expected_metadata(self.frames))
        if "clip_frame_indices" not in self.df.columns:
            raise RuntimeError(f"manifest lacks clip_frame_indices; regenerate it with timestamp-nearest Stage3 sampling: {self.manifest}")

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _expected_cache_dir(self, row) -> Path | None:
        if self.cache_root is None:
            return None
        if "route_id" in row and "segment_id" in row:
            return self.cache_root / str(row.route_id) / str(row.segment_id)
        return self.cache_root / Path(str(row.video_path)).with_suffix("")

    def _cache_dir(self, row) -> Path | None:
        cache_dir = self._expected_cache_dir(row)
        return cache_dir if cache_dir is not None and (cache_dir / "frames.csv").is_file() else None

    def _missing_cache_error(self, cache_dir: Path) -> FileNotFoundError:
        return FileNotFoundError(
            "missing comma2k19 Stage3 frame cache; refusing slow HEVC fallback: "
            f"{cache_dir / 'frames.csv'}\n"
            "Build it, e.g. python -m src.tools.cache_comma2k19_stage3_frames "
            f"--manifest {self.manifest} --temporal-stride 8 --limit-segments 0"
        )

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        video_path = self._video_path(str(row.video_path))
        frame_index = int(row.video_frame_index if "video_frame_index" in self.df.columns else row.frame_index)
        clip_indices = parse_clip_frame_indices(str(row.clip_frame_indices), self.frames)
        cache_dir = self._cache_dir(row)
        if self.cache_root is not None and cache_dir is None:
            raise self._missing_cache_error(self._expected_cache_dir(row))
        video = stage3_cached_clip_indices(cache_dir, clip_indices) if cache_dir else stage3_video_clip_indices(video_path, clip_indices)
        return {
            "video": video,
            "accel_label": int(row.accel_label),
            "steer_label": int(row.steer_label),
            "timestamp": float(row.target_timestamp if "target_timestamp" in self.df.columns else row.timestamp),
            "video_path": str(video_path),
            "frame_index": frame_index,
            "clip_frame_indices": torch.tensor(clip_indices, dtype=torch.long),
            "route_id": str(row.route_id) if "route_id" in self.df.columns else "",
            "segment_id": str(row.segment_id) if "segment_id" in self.df.columns else "",
        }
