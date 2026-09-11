from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_OUTPUT_HZ,
    STAGE3_TARTANVO_FEATURE,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_TEMPORAL_STRIDE,
)
from src.train.stage3 import _balanced_limit, _stride_manifest
from src.tools.build_comma2k19_stage3_manifest import ACCEL_NAMES, STEER_NAMES, _labels


def read_manifest(split: str) -> pd.DataFrame:
    path = COMMA2K19_STAGE3_TRAIN_MANIFEST if split == "train" else COMMA2K19_STAGE3_VAL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"missing comma2k19 {split} manifest: {path}")
    return pd.read_csv(path)


def route_series(df: pd.DataFrame) -> pd.Series:
    if "route_id" in df.columns:
        return df["route_id"].astype(str)
    return df["video_path"].astype(str).map(lambda x: str(Path(x).parent))


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


def cap_info(video_path: str | Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, frames


def abs_video_path(row) -> Path:
    p = Path(str(row.video_path))
    return p if p.is_absolute() else COMMA2K19_STAGE3_RAW / p


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
        fps, frames = cap_info(video) if video.is_file() else (20.0, int(seg.frame_index.max()) + 1)
        valid = ((seg.frame_index >= 0) & (seg.frame_index < frames)).sum()
        used = split_after_training_filters(seg, split)
        print(f"segment {seg_key}")
        print(f"original frames: {frames}")
        print(f"10Hz candidate centers: {len(seg)}")
        print(f"valid 16-frame clips: {int(valid)}")
        print(f"final samples used: {len(used)}")


def disagreement(df: pd.DataFrame) -> tuple[float, list[list[int]], dict[str, dict[str, int]]]:
    if not {"speed", "steering_angle"} <= set(df.columns):
        return float('nan'), [], {}
    _, current, _ = _labels(df.speed.to_numpy(float), df.steering_angle.to_numpy(float), False, "current")
    _, alt, _ = _labels(df.speed.to_numpy(float), df.steering_angle.to_numpy(float), False, "window_regression")
    matrix = np.zeros((4, 4), dtype=int)
    for a, b in zip(current, alt):
        matrix[int(a), int(b)] += 1
    agree = float((current == alt).mean()) if len(current) else float('nan')
    dist = {
        "current": {ACCEL_NAMES[i]: int((current == i).sum()) for i in ACCEL_NAMES},
        "window_regression": {ACCEL_NAMES[i]: int((alt == i).sum()) for i in ACCEL_NAMES},
    }
    return agree, matrix.tolist(), dist