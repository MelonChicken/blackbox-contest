from __future__ import annotations

from pathlib import Path

import json
import zlib

import numpy as np
import pandas as pd

from src.config import (
    COMMA2K19_STAGE3_MANIFEST,
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_SUBSET_MANIFEST,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    STAGE3_COMMA_SUBSET_MODE,
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE,
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



def segment_key_columns(df: pd.DataFrame) -> list[str]:
    return ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else ["video_path"]


def subset_name(split: str, segments_per_route: int | None = None) -> str:
    n = segments_per_route if segments_per_route is not None else (STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE if split == "train" else STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE)
    return f"{split}_r{int(n)}"


def subset_manifest_path(split: str, name: str | None = None) -> Path:
    if name == "all" or (name is None and STAGE3_COMMA_SUBSET_MODE == "all"):
        return COMMA2K19_STAGE3_TRAIN_MANIFEST if split == "train" else COMMA2K19_STAGE3_VAL_MANIFEST
    if name in {None, ""}:
        name = subset_name(split)
    return COMMA2K19_STAGE3_SUBSET_MANIFEST / f"{name}.csv"


def active_subset_name(split: str) -> str | None:
    if STAGE3_COMMA_SUBSET_MODE == "all":
        return None
    if STAGE3_COMMA_SUBSET_MODE != "route_balanced":
        raise ValueError(f"unknown STAGE3_COMMA_SUBSET_MODE: {STAGE3_COMMA_SUBSET_MODE}")
    return subset_name(split)


def active_manifest_path(split: str) -> Path:
    name = active_subset_name(split)
    if name is None:
        return COMMA2K19_STAGE3_TRAIN_MANIFEST if split == "train" else COMMA2K19_STAGE3_VAL_MANIFEST
    return subset_manifest_path(split, name)


def segment_cache_index_name(split: str, subset: str | None = None) -> str:
    return f"{subset}_index.csv" if subset not in {None, "", "all"} else f"{split}_index.csv"


def route_balanced_subset(df: pd.DataFrame, segments_per_route: int, seed: int = SEED) -> pd.DataFrame:
    keys = segment_key_columns(df)
    segments = df[keys].drop_duplicates().copy()
    segments["route"] = route_series(segments if "route_id" in segments.columns else df.loc[segments.index])
    picked = []
    for route, part in segments.groupby("route", sort=True):
        route_seed = int(seed) + zlib.crc32(str(route).encode("utf-8"))
        sample = part.sample(n=min(int(segments_per_route), len(part)), random_state=route_seed).sort_values(keys).drop(columns="route")
        picked.append(sample)
    chosen = pd.concat(picked, ignore_index=True) if picked else segments[keys].head(0)
    return df.merge(chosen, on=keys, how="inner").reset_index(drop=True)


def _distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()} if col in df.columns else {}


def _segment_metadata() -> dict[str, int]:
    path = COMMA2K19_STAGE3_MANIFEST / "video_metadata.csv"
    if not path.is_file():
        return {}
    meta = pd.read_csv(path)
    if "segment_id" not in meta.columns or "decoded_frame_count" not in meta.columns:
        return {}
    out = {}
    for row in meta.itertuples(index=False):
        count = int(getattr(row, "decoded_frame_count", 0) or 0)
        if count > 0:
            key = str(getattr(row, "segment_id"))
            out[key] = count
            out[Path(key).name] = count
    return out


def _frame_times_count(row) -> int:
    video = abs_video_path(row)
    segment = video.parent
    for rel in ("global_pose/frame_times.npy", "global_pose/frame_times", "global_pos/frame_times.npy", "global_pos/frame_times", "frame_times.npy", "frame_times"):
        path = segment / rel
        if path.is_file():
            return int(len(np.asarray(np.load(path, allow_pickle=False)).squeeze()))
    return 0


def _segment_count_key(row) -> str:
    video = Path(str(row.video_path))
    return str(video.parent).replace("\\", "/")


def _estimated_pairs(df: pd.DataFrame) -> int:
    keys = segment_key_columns(df)
    meta = _segment_metadata()
    total = 0
    for row in df.drop_duplicates(keys).itertuples(index=False):
        count = 0
        if hasattr(row, "video_decoded_frames") and pd.notna(getattr(row, "video_decoded_frames")):
            count = int(getattr(row, "video_decoded_frames") or 0)
        if count <= 0:
            count = int(meta.get(_segment_count_key(row), meta.get(str(getattr(row, "segment_id", "")), 0)))
        if count <= 0:
            count = _frame_times_count(row)
        total += max(0, count - 1)
    return int(total)


def _summary(df: pd.DataFrame) -> dict:
    keys = segment_key_columns(df)
    pair_estimate = _estimated_pairs(df)
    return {
        "routes": int(route_series(df).nunique()),
        "segments": int(df[keys].drop_duplicates().shape[0]),
        "samples": int(len(df)),
        "accel": _distribution(df, "accel_label"),
        "steer": _distribution(df, "steer_label"),
        "estimated_pairs": int(pair_estimate),
        "estimated_cache_mb_fp32": float(pair_estimate * 1536 * 4 / 1024 / 1024),
    }


def _dist_delta(full: dict[str, int], subset: dict[str, int]) -> dict[str, float]:
    labels = sorted(set(full) | set(subset))
    ft, st = max(1, sum(full.values())), max(1, sum(subset.values()))
    return {label: (subset.get(label, 0) / st) - (full.get(label, 0) / ft) for label in labels}


def build_route_balanced_subsets(
    train_segments_per_route: int = STAGE3_COMMA_TRAIN_SEGMENTS_PER_ROUTE,
    val_segments_per_route: int = STAGE3_COMMA_VAL_SEGMENTS_PER_ROUTE,
    seed: int = SEED,
) -> dict:
    COMMA2K19_STAGE3_SUBSET_MANIFEST.mkdir(parents=True, exist_ok=True)
    config = {"train": int(train_segments_per_route), "val": int(val_segments_per_route)}
    metadata = {"mode": "route_balanced", "seed": int(seed), "splits": {}}
    for split, per_route in config.items():
        full = read_manifest(split)
        subset = route_balanced_subset(full, per_route, seed)
        name = subset_name(split, per_route)
        path = subset_manifest_path(split, name)
        subset.to_csv(path, index=False)
        full_summary, subset_summary = _summary(full), _summary(subset)
        metadata["splits"][split] = {
            "name": name,
            "path": str(path),
            "segments_per_route": int(per_route),
            "full": full_summary,
            "subset": subset_summary,
            "accel_delta": _dist_delta(full_summary["accel"], subset_summary["accel"]),
            "steer_delta": _dist_delta(full_summary["steer"], subset_summary["steer"]),
        }
    meta_path = COMMA2K19_STAGE3_SUBSET_MANIFEST / f"metadata_r{int(train_segments_per_route)}_r{int(val_segments_per_route)}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def print_subset_summary(metadata: dict, seconds_per_segment: float | None = None, mb_per_segment: float | None = None) -> None:
    total_segments = 0
    for split, info in metadata["splits"].items():
        sub = info["subset"]
        total_segments += sub["segments"]
        print(f"[{split} subset: {info['name']}]")
        print(f"routes: {sub['routes']}")
        print(f"segments: {sub['segments']}")
        print(f"samples: {sub['samples']}")
        print(f"accel: {sub['accel']}")
        print(f"steer: {sub['steer']}")
        print(f"estimated_pairs: {sub['estimated_pairs']}")
        print(f"estimated_cache_mb_fp32: {sub['estimated_cache_mb_fp32']:.2f}")
        print(f"accel_delta_vs_full: {info['accel_delta']}")
        print(f"steer_delta_vs_full: {info['steer_delta']}")
    total_mb = sum(info["subset"]["estimated_cache_mb_fp32"] for info in metadata["splits"].values())
    print(f"total_segments: {total_segments}")
    print(f"total_estimated_cache_mb_fp32: {total_mb:.2f}")
    if mb_per_segment is not None:
        print(f"estimated_cache_size_mb: {total_segments * float(mb_per_segment):.2f}")
    if seconds_per_segment is not None:
        print(f"estimated_cache_time_sec: {total_segments * float(seconds_per_segment):.1f}")

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