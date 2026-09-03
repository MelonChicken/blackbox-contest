from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    COMMA2K19_STAGE3_MANIFEST,
    COMMA2K19_STAGE3_RAW,
    STAGE3_ACCEL_THRESHOLD,
    STAGE3_DECEL_THRESHOLD,
    STAGE3_OUTPUT_HZ,
    STAGE3_STEER_THRESHOLD,
    STAGE3_STOP_SPEED_THRESHOLD,
)

VIDEO_EXT = {".hevc", ".mp4", ".mkv", ".avi", ".mov"}
ACCEL_NAMES = {0: "ACCELERATING", 1: "DECELERATING", 2: "CONSTANT", 3: "STOPPED"}
STEER_NAMES = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False)).squeeze()


def _series_one(segment: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    base = segment / "processed_log" / "CAN" / name
    for t_name in ("t", "t.npy"):
        for v_name in ("value", "value.npy"):
            t_path, v_path = base / t_name, base / v_name
            if t_path.is_file() and v_path.is_file():
                return _load_array(t_path).astype(float), _load_array(v_path).astype(float)
    flat_t = next(segment.rglob(f"processed_log__CAN__{name}__t*"), None)
    flat_v = next(segment.rglob(f"processed_log__CAN__{name}__value*"), None)
    if flat_t and flat_v:
        return _load_array(flat_t).astype(float), _load_array(flat_v).astype(float)
    return None


def _series(segment: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = (name, "car_speed") if name == "speed" else (name,)
    for candidate in names:
        found = _series_one(segment, candidate)
        if found is not None:
            return found
    raise FileNotFoundError(f"missing processed_log/CAN/{name}/{{t,value}} under {segment}")


def _direct_video(segment: Path) -> Path:
    videos = sorted(p for p in segment.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise FileNotFoundError("missing video file")
    return videos[0]


def _frame_times(segment: Path, video: Path) -> np.ndarray:
    for rel in ("global_pose/frame_times", "global_pose/frame_times.npy", "global_pos/frame_times", "global_pos/frame_times.npy"):
        path = segment / rel
        if path.is_file():
            return _load_array(path).astype(float)
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if count <= 0:
        raise ValueError(f"cannot read frame count: {video}")
    return np.arange(count, dtype=float) / fps


def _align_time_base(target_t: np.ndarray, state_t: np.ndarray) -> np.ndarray:
    if state_t.min() <= target_t.max() and target_t.min() <= state_t.max():
        return state_t
    return state_t - state_t[0] + target_t[0]


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width, dtype=float) / width, mode="same")


def _labels(speed: np.ndarray, steering: np.ndarray, invert_steering: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accel = np.gradient(_smooth(speed), 1.0 / STAGE3_OUTPUT_HZ)
    accel_label = np.full(len(speed), 2, dtype=np.int64)
    accel_label[accel > STAGE3_ACCEL_THRESHOLD] = 0
    accel_label[accel < -STAGE3_DECEL_THRESHOLD] = 1
    accel_label[speed < STAGE3_STOP_SPEED_THRESHOLD] = 3

    steering = -steering if invert_steering else steering
    steer_label = np.full(len(steering), 1, dtype=np.int64)
    steer_label[steering > STAGE3_STEER_THRESHOLD] = 0
    steer_label[steering < -STAGE3_STEER_THRESHOLD] = 2
    return accel, accel_label, steer_label


def _rows(video_path: str, frame_t, speed_t, speed_v, steer_t, steer_v, invert_steering: bool) -> list[dict]:
    frame_t = np.asarray(frame_t, dtype=float).squeeze()
    if len(frame_t) < 2:
        raise ValueError("need at least two frame timestamps")
    start, end = float(frame_t[0]), float(frame_t[-1])
    target_t = np.arange(start, end + 1e-9, 1.0 / STAGE3_OUTPUT_HZ)
    speed_t = _align_time_base(target_t, np.asarray(speed_t, dtype=float).squeeze())
    steer_t = _align_time_base(target_t, np.asarray(steer_t, dtype=float).squeeze())
    speed = np.interp(target_t, speed_t, np.asarray(speed_v, dtype=float).squeeze())
    steering = np.interp(target_t, steer_t, np.asarray(steer_v, dtype=float).squeeze())
    acceleration, accel_label, steer_label = _labels(speed, steering, invert_steering)
    frame_index = np.clip(np.rint(np.interp(target_t, frame_t, np.arange(len(frame_t)))).astype(int), 0, len(frame_t) - 1)
    return [
        {
            "video_path": video_path,
            "frame_index": int(frame_index[i]),
            "timestamp": float(target_t[i] - start),
            "speed": float(speed[i]),
            "acceleration": float(acceleration[i]),
            "steering_angle": float(steering[i]),
            "accel_label": int(accel_label[i]),
            "steer_label": int(steer_label[i]),
        }
        for i in range(len(target_t))
    ]


def _segment_dirs(raw_root: Path) -> list[Path]:
    chunks = sorted(p for p in raw_root.glob("Chunk_*") if p.is_dir())
    segments = []
    for chunk in chunks:
        for route in sorted(p for p in chunk.iterdir() if p.is_dir()):
            for segment in sorted(p for p in route.iterdir() if p.is_dir()):
                if (segment / "video.hevc").is_file():
                    segments.append(segment)
    return segments


def _local_rows(segment: Path, raw_root: Path, invert_steering: bool) -> list[dict]:
    video = _direct_video(segment)
    frame_t = _frame_times(segment, video)
    speed_t, speed_v = _series(segment, "speed")
    steer_t, steer_v = _series(segment, "steering_angle")
    rel_video = video.relative_to(raw_root) if video.is_relative_to(raw_root) else video
    rows = _rows(str(rel_video), frame_t, speed_t, speed_v, steer_t, steer_v, invert_steering)
    route_id = segment.parent.name
    segment_id = segment.name
    for row in rows:
        row["route_id"] = route_id
        row["segment_id"] = segment_id
    return rows


def _hf_video_path(row: dict, video_dir: Path) -> str:
    video = row["video"]
    if isinstance(video, dict) and video.get("path"):
        return str(Path(video["path"]))
    segment_id = row.get("segment_id", str(len(list(video_dir.glob("*.hevc")))))
    out = video_dir / f"{str(segment_id).replace('/', '__').replace('|', '_')}.hevc"
    data = video.get("bytes") if isinstance(video, dict) else video
    if data is None:
        raise ValueError("HF row video has neither path nor bytes")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out)


def _hf_rows(split: str, limit: int | None, video_dir: Path, invert_steering: bool) -> list[dict]:
    from datasets import Video, load_dataset

    ds = load_dataset("commaai/comma2k19", split=split)
    if "video" in ds.features:
        ds = ds.cast_column("video", Video(decode=False))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for row in ds:
        log = row["log"]
        video_path = _hf_video_path(row, video_dir)
        rows.extend(
            _rows(
                video_path,
                log["global_pose__frame_times"],
                log["processed_log__CAN__speed__t"],
                log["processed_log__CAN__speed__value"],
                log["processed_log__CAN__steering_angle__t"],
                log["processed_log__CAN__steering_angle__value"],
                invert_steering,
            )
        )
    return rows


def _write_splits(df: pd.DataFrame, out_dir: Path, val_ratio: float) -> tuple[Path, Path]:
    if df.empty:
        raise RuntimeError("no valid comma2k19 samples were produced")

    split_key = df["route_id"].astype(str) if "route_id" in df.columns else df.video_path.astype(str)
    routes = sorted(split_key.unique())
    val_count = max(1, int(round(len(routes) * val_ratio))) if val_ratio > 0 and len(routes) > 1 else 0
    val_routes = set(routes[-val_count:]) if val_count else set()
    df["split"] = np.where(split_key.isin(val_routes), "val", "train")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.csv", out_dir / "val.csv"
    df[df.split == "train"].to_csv(train_path, index=False)
    df[df.split == "val"].to_csv(val_path, index=False)
    for split, part in df.groupby("split"):
        print(f"{split}: {len(part)} samples")
        print(f"{split} routes: {part.route_id.nunique() if 'route_id' in part.columns else 'N/A'}")
        print("accel:", {ACCEL_NAMES[k]: int(v) for k, v in part.accel_label.value_counts().sort_index().items()})
        print("steer:", {STEER_NAMES[k]: int(v) for k, v in part.steer_label.value_counts().sort_index().items()})
    return train_path, val_path


def build_manifest(raw_root: Path, out_dir: Path, val_ratio: float, limit: int | None, invert_steering: bool, hf_split: str | None) -> tuple[Path, Path]:
    if hf_split:
        rows = _hf_rows(hf_split, limit, out_dir.parent / "videos", invert_steering)
    else:
        segments = _segment_dirs(raw_root)
        if limit:
            segments = segments[:limit]
        if not segments:
            raise FileNotFoundError(f"no comma2k19 Chunk_* segment directories found under {raw_root}")
        rows = []
        for segment in segments:
            try:
                rows.extend(_local_rows(segment, raw_root, invert_steering))
            except Exception as exc:
                warnings.warn(f"skip incomplete segment {segment}: {exc}")
    return _write_splits(pd.DataFrame(rows), out_dir, val_ratio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=COMMA2K19_STAGE3_RAW)
    parser.add_argument("--out-dir", type=Path, default=COMMA2K19_STAGE3_MANIFEST)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-split", default=None, help="Use HuggingFace load_dataset('commaai/comma2k19', split=...) instead of local Chunk_* files.")
    parser.add_argument("--invert-steering", action="store_true")
    args = parser.parse_args()
    build_manifest(args.raw_root, args.out_dir, args.val_ratio, args.limit, args.invert_steering, args.hf_split)


if __name__ == "__main__":
    main()
