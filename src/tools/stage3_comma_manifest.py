from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COMMA2K19_STAGE3_MANIFEST,
    COMMA2K19_STAGE3_RAW,
    SEED,
    STAGE3_TRAIN_SAMPLE_LIMIT,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_SAMPLE_LIMIT,
    STAGE3_VAL_TEMPORAL_STRIDE,
)


def _manifest_path(split: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"unknown split: {split}")
    return COMMA2K19_STAGE3_MANIFEST / f"{split}.csv"


def read_manifest(split: str) -> pd.DataFrame:
    path = _manifest_path(split)
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


def abs_video_path(row) -> Path:
    p = Path(str(row.video_path))
    return p if p.is_absolute() else COMMA2K19_STAGE3_RAW / p


def label_counts(values: pd.Series, names: dict[int, str]) -> dict[str, int]:
    counts = values.astype(int).value_counts().sort_index()
    return {names[i]: int(counts.get(i, 0)) for i in sorted(names)}


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else "video_path"
    return pd.concat([part.iloc[::stride] for _, part in df.groupby(key, sort=False)], ignore_index=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "segment_id" if "segment_id" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    per_group = max(1, limit // df[key].nunique() + 1)
    out = df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(per_group)
    return out.head(limit).sort_index().reset_index(drop=True)


def split_after_training_filters(df: pd.DataFrame, split: str) -> pd.DataFrame:
    stride = STAGE3_TRAIN_TEMPORAL_STRIDE if split == "train" else STAGE3_VAL_TEMPORAL_STRIDE
    limit = STAGE3_TRAIN_SAMPLE_LIMIT if split == "train" else STAGE3_VAL_SAMPLE_LIMIT
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


def print_sample_count_audit(split: str) -> None:
    df = read_manifest(split)
    final_df = split_after_training_filters(df, split)
    key_cols = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else ["video_path"]
    print(f"[comma {split} sample count]")
    print(f"routes: {route_series(df).nunique()}")
    print(f"segments: {df[key_cols].drop_duplicates().shape[0]}")
    print(f"manifest samples: {len(df)}")
    print(f"final dataset len: {len(final_df)}")


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
    return float((current == alt).mean()) if len(current) else float("nan"), matrix.tolist(), {
        "current": {accel_names[i]: int((current == i).sum()) for i in accel_names},
        "window_regression": {accel_names[i]: int((alt == i).sum()) for i in accel_names},
    }
