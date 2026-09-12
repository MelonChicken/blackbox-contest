from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    COMMA2K19_STAGE3_RAW,
    COMMA2K19_STAGE3_TRAIN_MANIFEST,
    COMMA2K19_STAGE3_VAL_MANIFEST,
    DEVICE,
    KITTI_STAGE3_TRAIN_MANIFEST,
    KITTI_STAGE3_VAL_MANIFEST,
    SEED,
    STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    STAGE3_COMMA_VAL_SAMPLE_LIMIT,
    STAGE3_KITTI_TRAIN_SAMPLE_LIMIT,
    STAGE3_KITTI_VAL_SAMPLE_LIMIT,
    STAGE3_NUSCENES_ROOT,
    STAGE3_NUSCENES_TRAIN_MANIFEST,
    STAGE3_NUSCENES_VAL_MANIFEST,
    STAGE3_NUM_FRAMES,
    STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE,
    STAGE3_TARTANVO_FEATURE_CACHE,
    STAGE3_TRAIN_TEMPORAL_STRIDE,
    STAGE3_VAL_TEMPORAL_STRIDE,
    TARTANVO_CHECKPOINT,
)
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID, Comma2k19Stage3Dataset
from src.datasets.kitti_stage3 import KittiStage3Dataset, _label_id
from src.datasets.nuscenes_stage3 import NuScenesStage3Dataset
from src.datasets.stage3_tartanvo_pose import (
    TARTANVO_SEGMENT_FEATURE_VERSION,
    stage3_tartanvo_feature_path,
    stage3_tartanvo_sample_key,
    stage3_tartanvo_segment_key,
)
from src.models import Stage3TartanVOGRU
from src.tools.build_comma2k19_stage3_manifest import ALIGNMENT_VERSION
from src.utils import _crop_tensor

CACHE_VERSION = 2
SEGMENT_CACHE_LAYOUT = "segment"


def _stride_manifest(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df.reset_index(drop=True)
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else "segment_id" if "segment_id" in df.columns else "sequence_id" if "sequence_id" in df.columns else "video_path"
    parts = [part.iloc[::stride] for _, part in df.groupby(key, sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df.reset_index(drop=True)


def _balanced_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    key = "sequence_id" if "sequence_id" in df.columns else "segment_id" if "segment_id" in df.columns else None
    if key is None:
        return df.sample(n=limit, random_state=SEED).sort_index().reset_index(drop=True)
    return df.groupby(key, group_keys=False).sample(frac=1.0, random_state=SEED).groupby(key, group_keys=False).head(max(1, limit // df[key].nunique() + 1)).head(limit).sort_index().reset_index(drop=True)


def _split_config(dataset: str, split: str):
    if dataset == "kitti":
        if split == "train":
            return KITTI_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_KITTI_TRAIN_SAMPLE_LIMIT
        if split == "val":
            return KITTI_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_KITTI_VAL_SAMPLE_LIMIT
    if dataset == "comma2k19":
        if split == "train":
            return COMMA2K19_STAGE3_TRAIN_MANIFEST, STAGE3_TRAIN_TEMPORAL_STRIDE, STAGE3_COMMA_TRAIN_SAMPLE_LIMIT
        if split == "val":
            return COMMA2K19_STAGE3_VAL_MANIFEST, STAGE3_VAL_TEMPORAL_STRIDE, STAGE3_COMMA_VAL_SAMPLE_LIMIT
    if dataset == "nuscenes":
        return (STAGE3_NUSCENES_TRAIN_MANIFEST if split == "train" else STAGE3_NUSCENES_VAL_MANIFEST), 1, None
    raise ValueError(f"unknown dataset/split: {dataset}/{split}")


def _dataset(dataset: str, manifest: Path, split: str):
    if dataset == "kitti":
        return KittiStage3Dataset(manifest)
    if dataset == "comma2k19":
        return Comma2k19Stage3Dataset(manifest)
    if dataset == "nuscenes":
        return NuScenesStage3Dataset(manifest, root=STAGE3_NUSCENES_ROOT)
    raise ValueError(f"unknown dataset: {dataset}")


def _cache_base(root: Path, feature: str, dataset: str) -> Path:
    return root / feature / dataset


def _segment_cache_base(root: Path, dataset: str) -> Path:
    return root / "segment_latent" / dataset


def _checkpoint_sha1(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_frame_index(row) -> int:
    return int(getattr(row, "video_frame_index", getattr(row, "frame_index", getattr(row, "frame_idx", -1))))


def _row_sequence_id(row):
    return getattr(row, "sequence_id", getattr(row, "scene", getattr(row, "ID", None)))


def _video_path(raw_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else raw_root / path


def _valid_segment_cache(path: Path, feature_dim: int) -> tuple[bool, dict | None]:
    if not path.is_file():
        return False, None
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        features = item["features"]
        ok = (
            item.get("alignment_version") == ALIGNMENT_VERSION
            and item.get("feature_version") == TARTANVO_SEGMENT_FEATURE_VERSION
            and int(item.get("feature_dim", features.shape[-1])) == feature_dim
            and features.ndim == 2
            and int(features.shape[-1]) == feature_dim
            and int(item.get("num_pairs", features.shape[0])) == int(features.shape[0])
        )
        return bool(ok), item if ok else None
    except Exception:
        return False, None


def _segment_groups(df: pd.DataFrame, max_segments: int | None = None) -> list[pd.DataFrame]:
    key = ["route_id", "segment_id"] if {"route_id", "segment_id"}.issubset(df.columns) else ["video_path"]
    groups = [part.reset_index(drop=True) for _, part in df.groupby(key, sort=False)]
    return groups[:max_segments] if max_segments else groups


def _extract_segment_features(model: Stage3TartanVOGRU, video_path: Path, pair_batch_size: int) -> tuple[torch.Tensor, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    features, frames = [], []
    decoded = 0
    start = time.perf_counter()

    def run_chunk(chunk: list[torch.Tensor]) -> None:
        if len(chunk) < 2:
            return
        video = torch.stack(chunk, dim=1).unsqueeze(0).to(DEVICE, non_blocking=True)
        video = (video - model.s3_mean) / model.s3_std
        features.append(model.feature_sequence(video).squeeze(0).cpu().to(torch.float32))

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(_crop_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            decoded += 1
            if len(frames) == int(pair_batch_size) + 1:
                run_chunk(frames)
                frames = [frames[-1]]
        run_chunk(frames)
    finally:
        cap.release()
    if decoded < 2:
        raise RuntimeError(f"segment decoded fewer than 2 frames: {video_path}")
    return torch.cat(features, dim=0), decoded, time.perf_counter() - start


def cache_segment_split(
    split: str,
    root: Path = STAGE3_TARTANVO_FEATURE_CACHE,
    overwrite: bool = False,
    pair_batch_size: int = STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE,
    max_segments: int | None = None,
) -> None:
    manifest, _, _ = _split_config("comma2k19", split)
    df = pd.read_csv(manifest)
    versions = set(df.get("alignment_version", pd.Series(dtype=str)).dropna().astype(str))
    if versions != {ALIGNMENT_VERSION}:
        raise RuntimeError(f"comma2k19 manifest must use {ALIGNMENT_VERSION}; found {sorted(versions) or [None]}")
    base = _segment_cache_base(root, "comma2k19")
    out_dir = base / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Stage3TartanVOGRU(load_pretrained=True, feature="latent").to(DEVICE).eval()
    groups = _segment_groups(df, max_segments)
    rows = []
    cached = skipped = failed = 0
    started = time.perf_counter()
    progress = tqdm(groups, desc=f"comma {split} cache", unit="segment")
    with torch.inference_mode():
        for n, group in enumerate(progress, start=1):
            row = group.iloc[0]
            key = stage3_tartanvo_segment_key(row)
            rel_path = Path(split) / f"{key}.pt"
            path = base / rel_path
            ok, existing = _valid_segment_cache(path, model.feature_dim)
            elapsed = 0.0
            if ok and not overwrite:
                features = existing["features"]
                num_frames = int(existing.get("num_frames", int(features.shape[0]) + 1))
                skipped += 1
            else:
                try:
                    features, num_frames, elapsed = _extract_segment_features(
                        model,
                        _video_path(COMMA2K19_STAGE3_RAW, str(row.video_path)),
                        int(pair_batch_size),
                    )
                    if int(features.shape[0]) != num_frames - 1:
                        raise RuntimeError(f"num_pairs {features.shape[0]} != num_frames-1 {num_frames - 1}")
                    torch.save({
                        "features": features,
                        "frame_indices": torch.arange(int(features.shape[0]), dtype=torch.long),
                        "segment_id": str(row.segment_id),
                        "route_id": str(row.route_id),
                        "alignment_version": ALIGNMENT_VERSION,
                        "feature_version": TARTANVO_SEGMENT_FEATURE_VERSION,
                        "feature_dim": model.feature_dim,
                        "num_frames": int(num_frames),
                        "num_pairs": int(features.shape[0]),
                        "feature_type": "latent",
                        "encoder": "TartanVO",
                        "pretrained_loaded": bool(model.tartanvo.pretrained_loaded),
                    }, path)
                    cached += 1
                except Exception as exc:
                    failed += 1
                    warnings.warn(f"failed segment {key}: {exc}")
                    continue
            num_pairs = int(features.shape[0])
            rows.append({
                "route_id": str(row.route_id),
                "segment_id": str(row.segment_id),
                "segment_key": key,
                "feature_path": str(rel_path),
                "num_frames": int(num_frames),
                "num_pairs": num_pairs,
                "feature_dim": model.feature_dim,
                "alignment_version": ALIGNMENT_VERSION,
                "feature_version": TARTANVO_SEGMENT_FEATURE_VERSION,
            })
            total_elapsed = time.perf_counter() - started
            eta = (total_elapsed / n) * (len(groups) - n) if n else 0.0
            progress.set_postfix_str(
                f"{n}/{len(groups)} {key} frames={num_frames} pairs={num_pairs} elapsed={elapsed:.1f}s ETA={eta/60:.1f}m cached={cached} skipped={skipped} failed={failed}"
            )
    pd.DataFrame(rows).to_csv(base / f"{split}_index.csv", index=False)
    metadata = {
        "alignment_version": ALIGNMENT_VERSION,
        "cache_layout": SEGMENT_CACHE_LAYOUT,
        "feature_type": "latent",
        "feature_dim": model.feature_dim,
        "encoder": "TartanVO",
        "pretrained_loaded": bool(model.tartanvo.pretrained_loaded),
        "window_pairs": STAGE3_NUM_FRAMES - 1,
        "feature_version": TARTANVO_SEGMENT_FEATURE_VERSION,
        "pair_batch_size": int(pair_batch_size),
        "dataset": "comma2k19",
        "splits": {split: {"segments": len(rows), "cached": cached, "skipped": skipped, "failed": failed}},
    }
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def cache_split(split: str, feature: str, root: Path = STAGE3_TARTANVO_FEATURE_CACHE, overwrite: bool = False, dataset: str = "comma2k19", batch_size: int = 4, max_samples: int | None = None) -> None:
    if feature not in {"pose", "latent"}:
        raise ValueError(f"unknown feature: {feature}")
    manifest, stride, limit = _split_config(dataset, split)
    ds = _dataset(dataset, manifest, split)
    if dataset == "comma2k19":
        versions = set(ds.df.get("alignment_version", pd.Series(dtype=str)).dropna().astype(str))
        if versions != {ALIGNMENT_VERSION}:
            raise RuntimeError(f"comma2k19 manifest must use {ALIGNMENT_VERSION}; found {sorted(versions) or [None]}")
    ds.df = _balanced_limit(_stride_manifest(ds.df, stride), limit)
    if max_samples:
        ds.df = ds.df.head(max_samples).reset_index(drop=True)

    base = _cache_base(root, feature, dataset)
    out_dir = base / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Stage3TartanVOGRU(load_pretrained=True, feature=feature).to(DEVICE).eval()
    rows = []
    pending = []

    def flush() -> None:
        if not pending:
            return
        items = [x[0] for x in pending]
        video = torch.stack([item["video"] for item in items]).to(DEVICE, non_blocking=True)
        intrinsics = torch.stack([item["intrinsics"] for item in items]).to(DEVICE, non_blocking=True) if all("intrinsics" in item for item in items) else None
        cached_batch = model.feature_sequence(video, intrinsics=intrinsics).cpu().to(torch.float32)
        for cached, (item, row, key, path, accel_label, steer_label) in zip(cached_batch, pending):
            payload = {
                "feature": cached,
                "feature_mode": feature,
                "feature_dim": int(cached.shape[-1]),
                "accel_label": accel_label,
                "steer_label": steer_label,
                "sample_key": key,
                "source": dataset,
                "video_frame_index": _row_frame_index(row),
            }
            for name in ("video_path", "image_path", "sequence_id", "scene", "timestamp", "frame_paths"):
                if name in item:
                    payload[name] = item[name]
                elif hasattr(row, name):
                    payload[name] = getattr(row, name)
            torch.save(payload, path)
        pending.clear()

    with torch.inference_mode():
        for i in tqdm(range(len(ds)), desc=f"cache {dataset} tartanvo {feature} {split}"):
            row = ds.df.iloc[i]
            key = stage3_tartanvo_sample_key(row)
            feature_path = stage3_tartanvo_feature_path(split, key)
            path = base / feature_path
            accel_label = _label_id(row.accel_label, ACCEL_TO_ID)
            steer_label = _label_id(row.steer_label, STEER_TO_ID)
            if overwrite or not path.exists():
                pending.append((ds[i], row, key, path, accel_label, steer_label))
                if len(pending) >= batch_size:
                    flush()
            out_row = {"sample_key": key, "feature_path": feature_path, "accel_label": accel_label, "steer_label": steer_label, "sequence_id": _row_sequence_id(row), "video_frame_index": _row_frame_index(row)}
            for name in ("alignment_version", "alignment_source", "target_timestamp", "video_pts_sec", "alignment_error_sec"):
                if hasattr(row, name):
                    out_row[name] = getattr(row, name)
            rows.append(out_row)
        flush()
    pd.DataFrame(rows).to_csv(base / f"{split}_index.csv", index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "dataset": dataset,
        "feature_mode": feature,
        "feature_dim": model.feature_dim,
        "num_frames": STAGE3_NUM_FRAMES,
        "temporal_stride": stride,
        "sample_limit": limit,
        "max_samples": max_samples,
        "seed": SEED,
        "tartanvo_checkpoint": str(TARTANVO_CHECKPOINT),
        "tartanvo_checkpoint_name": TARTANVO_CHECKPOINT.name,
        "tartanvo_checkpoint_sha1": _checkpoint_sha1(TARTANVO_CHECKPOINT),
        "preprocessing": "Stage3 dataset video -> unnormalize S3_MEAN/S3_STD -> bilinear resize 448x640 -> TartanVO pairs",
        "temporal_length": STAGE3_NUM_FRAMES - 1,
        "split": split,
        "samples": len(rows),
        "manifest_alignment_version": str(ds.df.alignment_version.iloc[0]) if "alignment_version" in ds.df.columns and len(ds.df) else None,
    }
    (base / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen TartanVO features for Stage3.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--feature", choices=["pose", "latent"], default="latent")
    parser.add_argument("--dataset", choices=["comma2k19", "kitti", "nuscenes"], default="comma2k19")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-layout", choices=["sample", "segment"], default="segment")
    parser.add_argument("--pair-batch-size", type=int, default=STAGE3_TARTANVO_CACHE_PAIR_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-segments", type=int)
    args = parser.parse_args()
    if args.cache_layout == "segment":
        if args.dataset != "comma2k19" or args.feature != "latent":
            raise ValueError("--cache-layout segment is only implemented for comma2k19 latent")
        cache_segment_split(args.split, overwrite=args.overwrite, pair_batch_size=args.pair_batch_size, max_segments=args.max_segments)
        return
    cache_split(args.split, args.feature, overwrite=args.overwrite, dataset=args.dataset, batch_size=args.batch_size, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
