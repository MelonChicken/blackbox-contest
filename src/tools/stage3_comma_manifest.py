from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_TARTANVO_FEATURE,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_TEMPORAL_STRIDE,
    SEED,
)
from src.tools.build_comma2k19_stage3_manifest import _video_timing


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = "segment_id" if "segment_id" in df.columns else "sequence_id" if "sequence_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "sequence_id" if "sequence_id" in df.columns else "segment_id" if "segment_id" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    per_group = max(1, limit // df[key].nunique() + 1)
    out = df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(per_group)
    return out.head(limit).sort_index().reset_index(drop=True)


def read_manifest(split: str) -> pd.DataFrame:
    path = COMMA2K19_STAGE3_TRAIN_MANIFEST if split == "train" else COMMA2K19_STAGE3_VAL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"missing comma2k19 {split} manifest: {path}")
    return pd.read_csv(path)


def route_series(df: pd.DataFrame) -> pd.Series:
    if "route_id" in df.columns:
        return df["route_id"].astype(str)
    return df["video_path"].astype(str).map(lambda x: str(Path(x).parent))


def frame_index_series(df: pd.DataFrame) -> pd.Series:
    if "video_frame_index" in df.columns:
        return df["video_frame_index"].astype(int)
    if "frame_index" in df.columns:
        return df["frame_index"].astype(int)
    raise KeyError("manifest has neither video_frame_index nor legacy frame_index")


def split_after_training_filters(df: pd.DataFrame, split: str) -> pd.DataFrame:
    stride = STAGE3_TRAIN_TEMPORAL_STRIDE if split == "train" else STAGE3_VAL_TEMPORAL_STRIDE
    limit = STAGE3_COMMA_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_COMMA_VAL_SAMPLE_LIMIT
    return _balanced_limit(_stride_manifest(df, stride), limit)


def print_route_overlap() -> None:
    train = read_manifest("train")
    val = read_manifest("val")
    train_routes = set(route_series(train))
    val_routes = set(route_series(val))
    overlap = train_routes & val_routes
    print(f"Train routes: {len(train_routes)}")
    print(f"Val routes: {len(val_routes)}")
    print(f"Route overlap: {len(overlap)}")
    if overlap:
        print("overlap examples:", sorted(overlap)[:10])


def feature_index(split: str) -> pd.DataFrame | None:
    path = STAGE3_TARTANVO_FEATURE_CACHE / STAGE3_TARTANVO_FEATURE / "comma2k19" / f"{split}_index.csv"
    return pd.read_csv(path) if path.is_file() else None


def abs_video_path(row) -> Path:
    p = Path(str(row.video_path))
    return p if p.is_absolute() else COMMA2K19_STAGE3_RAW / p


def cap_info(video_path: str | Path) -> tuple[float, int]:
    timing = _video_timing(Path(video_path))
    return timing.reported_fps, timing.decoded_frame_count


def label_counts(values: pd.Series, names: dict[int, str]) -> dict[str, int]:
    counts = values.astype(int).value_counts().sort_index()
    return {names[i]: int(counts.get(i, 0)) for i in sorted(names)}


def print_sample_count_audit(split: str) -> None:
    df = read_manifest(split)
    stride = STAGE3_TRAIN_TEMPORAL_STRIDE if split == "train" else STAGE3_VAL_TEMPORAL_STRIDE
    limit = STAGE3_COMMA_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_COMMA_VAL_SAMPLE_LIMIT
    after_stride = _stride_manifest(df, stride)
    final_df = _balanced_limit(after_stride, limit)
    cache = feature_index(split)
    print(f"[comma {split} sample count]")
    print(f"routes: {route_series(df).nunique()}")
    print(f"segments: {df[['route_id','segment_id']].drop_duplicates().shape[0] if {'route_id','segment_id'} <= set(df.columns) else df.video_path.nunique()}")
    print(f"raw possible 10Hz centers: {len(df)}")
    print(f"boundary-filtered centers: {len(df)}")
    print(f"after temporal stride ({stride}): {len(after_stride)}")
    print(f"after sample limit ({limit}): {len(final_df)}")
    print(f"cache index samples: {len(cache) if cache is not None else 'missing'}")
    print(f"final dataset len: {len(cache) if cache is not None else len(final_df)}")
    if len(df):
        key = 'segment_id' if 'segment_id' in df.columns else 'video_path'
        seg_key, seg = next(iter(df.groupby(key, sort=False)))
        video = abs_video_path(seg.iloc[0])
        fps, frames = cap_info(video) if video.is_file() else (20.0, int(frame_index_series(seg).max()) + 1)
        frame_index = frame_index_series(seg)
        valid = ((frame_index >= 0) & (frame_index < frames)).sum()
        used = split_after_training_filters(seg, split)
        print(f"segment {seg_key}")
        print(f"original frames: {frames}")
        print(f"10Hz candidate centers: {len(seg)}")
        print(f"valid 16-frame clips: {int(valid)}")
        print(f"final samples used: {len(used)}")


def disagreement(df: pd.DataFrame, accel_names: dict[int, str]) -> tuple[float, list[list[int]], dict[str, dict[str, int]]]:
    from src.datasets.stage3_labels import derive_acceleration_current, derive_acceleration_window_regression, derive_accel_label

    if "speed" not in df.columns:
        return float("nan"), [], {}
    speed = df.speed.to_numpy(float)
    current = derive_accel_label(speed, derive_acceleration_current(speed))
    alt = derive_accel_label(speed, derive_acceleration_window_regression(speed))
    matrix = np.zeros((4, 4), dtype=int)
    for a, b in zip(current, alt):
        matrix[int(a), int(b)] += 1
    agree = float((current == alt).mean()) if len(current) else float("nan")
    dist = {
        "current": {accel_names[i]: int((current == i).sum()) for i in accel_names},
        "window_regression": {accel_names[i]: int((alt == i).sum()) for i in accel_names},
    }
    return agree, matrix.tolist(), dist