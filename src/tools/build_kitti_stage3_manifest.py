from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    KITTI_ACCEL_THRESHOLD,
    KITTI_INVERT_YAW,
    KITTI_MAX_ABS_ACCELERATION,
    KITTI_MAX_SPEED,
    KITTI_SPEED_SMOOTH_METHOD,
    KITTI_SPEED_SMOOTH_WINDOW,
    KITTI_STAGE3_MANIFEST,
    KITTI_STAGE3_RAW,
    KITTI_STAGE3_SPLIT_CSV,
    KITTI_STAGE3_SPLIT_JSON,
    KITTI_STOP_SPEED_THRESHOLD,
    KITTI_YAW_RATE_THRESHOLD,
    SEED,
)

DT = 0.1
ACCEL_NAMES = {0: "ACCELERATING", 1: "DECELERATING", 2: "CONSTANT", 3: "STOPPED"}
STEER_NAMES = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}


def _read_cam(path: Path) -> tuple[float, float, float, float]:
    values = np.loadtxt(path, dtype=float).reshape(3, 3)
    return float(values[0, 0]), float(values[1, 1]), float(values[0, 2]), float(values[1, 2])


def _read_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=float)
    poses = np.atleast_2d(poses)
    if poses.shape[1] != 12:
        raise ValueError(f"poses.txt must have 12 floats per line: {path}")
    return poses.reshape(-1, 3, 4)


def _smooth(values: np.ndarray, window: int, method: str) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    if method == "median":
        return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy(dtype=float)
    if method == "moving_average":
        return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    raise ValueError(f"unknown KITTI speed smoothing method: {method}")


def _image_resolution(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _labels(speed: np.ndarray, acceleration: np.ndarray, yaw_rate: np.ndarray, invert_yaw: bool, yaw_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    accel = np.full(len(speed), 2, dtype=np.int64)
    accel[acceleration > KITTI_ACCEL_THRESHOLD] = 0
    accel[acceleration < -KITTI_ACCEL_THRESHOLD] = 1
    accel[speed < KITTI_STOP_SPEED_THRESHOLD] = 3

    signed_yaw = -yaw_rate if invert_yaw else yaw_rate
    steer = np.full(len(speed), 1, dtype=np.int64)
    steer[signed_yaw > yaw_threshold] = 0
    steer[signed_yaw < -yaw_threshold] = 2
    return accel, steer


def _invalid_reason(speed: float, acceleration: float, max_speed: float | None, max_abs_acceleration: float | None) -> str:
    reasons = []
    if max_speed is not None and speed > max_speed:
        reasons.append("speed")
    if max_abs_acceleration is not None and abs(acceleration) > max_abs_acceleration:
        reasons.append("acceleration")
    return "+".join(reasons)


def _sequence_rows(seq: Path, raw_root: Path, smooth_window: int, smooth_method: str, invert_yaw: bool, yaw_threshold: float, max_speed: float | None, max_abs_acceleration: float | None) -> tuple[list[dict], dict]:
    images = sorted(seq.glob("*.jpg"))
    npy_count = len(list(seq.glob("*.npy")))
    cam_path = seq / "cam.txt"
    poses_path = seq / "poses.txt"
    cam_exists = cam_path.is_file()
    pose_count = 0
    resolution = None
    usable = 0
    rows: list[dict] = []

    if not images or not poses_path.is_file():
        info = {"sequence_id": seq.name, "image_count": len(images), "pose_count": pose_count, "cam_exists": cam_exists, "resolution": resolution, "usable_aligned_frame_count": usable, "npy_count": npy_count}
        return rows, info

    poses = _read_poses(poses_path)
    pose_count = len(poses)
    usable = min(len(images), pose_count)
    if len(images) != pose_count:
        warnings.warn(f"{seq.name}: image_count={len(images)} pose_count={pose_count}; using {usable}")
    if usable < 2:
        info = {"sequence_id": seq.name, "image_count": len(images), "pose_count": pose_count, "cam_exists": cam_exists, "resolution": resolution, "usable_aligned_frame_count": usable, "npy_count": npy_count}
        return rows, info

    images, poses = images[:usable], poses[:usable]
    width, height = _image_resolution(images[0])
    resolution = f"{width}x{height}"
    fx = fy = cx = cy = np.nan
    if cam_exists:
        fx, fy, cx, cy = _read_cam(cam_path)

    r = poses[:, :3, :3]
    t = poses[:, :3, 3]
    distance = np.r_[0.0, np.linalg.norm(t[1:] - t[:-1], axis=1)]
    raw_speed = distance / DT
    speed = _smooth(raw_speed, smooth_window, smooth_method)
    acceleration = np.r_[0.0, np.diff(speed) / DT]
    yaw_delta = np.zeros(usable, dtype=float)
    for i in range(1, usable):
        rel = r[i - 1].T @ r[i]
        yaw_delta[i] = np.arctan2(rel[0, 2], rel[2, 2])
    yaw_rate = yaw_delta / DT
    accel_label, steer_label = _labels(speed, acceleration, yaw_rate, invert_yaw, yaw_threshold)

    invalid_count = 0
    for i, image in enumerate(images):
        reason = _invalid_reason(float(speed[i]), float(acceleration[i]), max_speed, max_abs_acceleration)
        invalid_count += bool(reason)
        rows.append({
            "dataset": "kitti",
            "sequence_id": seq.name,
            "frame_index": i,
            "image_path": str(image.relative_to(raw_root)),
            "raw_speed": float(raw_speed[i]),
            "speed": float(speed[i]),
            "acceleration": float(acceleration[i]),
            "yaw_rate": float(yaw_rate[i]),
            "motion_valid": not bool(reason),
            "motion_invalid_reason": reason,
            "accel_label": int(accel_label[i]),
            "steer_label": int(steer_label[i]),
            "split": "",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_width": width,
            "image_height": height,
        })
    info = {
        "sequence_id": seq.name,
        "image_count": len(images),
        "pose_count": pose_count,
        "cam_exists": cam_exists,
        "resolution": resolution,
        "usable_aligned_frame_count": usable,
        "npy_count": npy_count,
        "max_speed": float(np.max(speed)),
        "max_abs_acceleration": float(np.max(np.abs(acceleration))),
        "max_abs_yaw_rate": float(np.max(np.abs(yaw_rate))),
        "invalid_motion_frames": invalid_count,
    }
    return rows, info


def _split_sequences(sequences: list[str], out_csv: Path, out_json: Path, val_ratio: float) -> dict[str, str]:
    if out_csv.is_file() and out_json.is_file():
        df = pd.read_csv(out_csv)
        return dict(zip(df.sequence_id.astype(str), df.split.astype(str)))
    rng = np.random.default_rng(SEED)
    seq = np.array(sorted(sequences), dtype=object)
    rng.shuffle(seq)
    val_count = max(1, int(round(len(seq) * val_ratio))) if len(seq) > 1 and val_ratio > 0 else 0
    val = set(seq[:val_count])
    split = {str(s): ("val" if s in val else "train") for s in seq}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"sequence_id": k, "split": v} for k, v in sorted(split.items())]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def _print_stats(name: str, values: pd.Series, percentiles: list[float]) -> None:
    arr = values.to_numpy(dtype=float)
    print(f"{name}: mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}")
    print(f"{name} percentiles:", {p: float(np.percentile(arr, p)) for p in percentiles})


def _print_distribution(name: str, values: pd.Series, names: dict[int, str]) -> None:
    total = max(1, len(values))
    counts = values.value_counts().sort_index()
    print(name)
    for key, label in names.items():
        count = int(counts.get(key, 0))
        print(f"  {label}: {count} ({count / total:.4f})")


def _save_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    for col, title in (("raw_speed", "Raw speed"), ("speed", "Smoothed speed"), ("acceleration", "Acceleration")):
        plt.figure()
        plt.hist(df[col].to_numpy(dtype=float), bins=100)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(title)
        plt.grid(True)
        path = diag / f"kitti_{col}_hist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"histogram saved: {path}")


def build_manifest(raw_root: Path, out_dir: Path, smooth_window: int, smooth_method: str, invert_yaw: bool, yaw_threshold: float, val_ratio: float, max_speed: float | None, max_abs_acceleration: float | None) -> tuple[Path, Path]:
    seqs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    rows: list[dict] = []
    infos = []
    for seq in seqs:
        seq_rows, info = _sequence_rows(seq, raw_root, smooth_window, smooth_method, invert_yaw, yaw_threshold, max_speed, max_abs_acceleration)
        rows.extend(seq_rows)
        infos.append(info)
        print(f"{info['sequence_id']}: images={info['image_count']} poses={info['pose_count']} cam={info['cam_exists']} resolution={info['resolution']} usable={info['usable_aligned_frame_count']} npy={info['npy_count']}")
    if not rows:
        raise RuntimeError(f"no usable KITTI frames under {raw_root}")

    df = pd.DataFrame(rows)
    split = _split_sequences(sorted(df.sequence_id.unique()), KITTI_STAGE3_SPLIT_CSV, KITTI_STAGE3_SPLIT_JSON, val_ratio)
    df["split"] = df.sequence_id.map(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path, train_path, val_path = out_dir / "all.csv", out_dir / "train.csv", out_dir / "val.csv"
    df.to_csv(all_path, index=False)
    df[df.split == "train"].to_csv(train_path, index=False)
    df[df.split == "val"].to_csv(val_path, index=False)

    print(f"motion pipeline: pose translation -> raw speed -> {smooth_method}(window={smooth_window}) speed -> acceleration")
    print(f"sequence_count={df.sequence_id.nunique()} usable_frame_count={len(df)}")
    print(f"train_sequences={df[df.split == 'train'].sequence_id.nunique()} val_sequences={df[df.split == 'val'].sequence_id.nunique()}")
    _print_stats("raw_speed", df.raw_speed, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    _print_stats("speed", df.speed, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    _print_stats("acceleration", df.acceleration, [1, 5, 25, 50, 75, 95, 99])
    print("abs(acceleration) percentiles:", {p: float(np.percentile(np.abs(df.acceleration), p)) for p in [90, 95, 97.5, 99, 99.5, 99.9]})
    print("abs(yaw_rate) percentiles:", {p: float(np.percentile(np.abs(df.yaw_rate), p)) for p in [50, 75, 90, 95, 97.5, 99]})
    print(f"chosen_yaw_threshold={yaw_threshold} invert_yaw={invert_yaw}")
    print(f"motion safeguard max_speed={max_speed} max_abs_acceleration={max_abs_acceleration} invalid_frames={int((~df.motion_valid).sum())}")
    print("sequence motion maxima:")
    for info in sorted(infos, key=lambda x: x.get("max_abs_acceleration", -1), reverse=True):
        if "max_speed" in info:
            print(f"  {info['sequence_id']}: max_speed={info['max_speed']:.6f} max_abs_acceleration={info['max_abs_acceleration']:.6f} max_abs_yaw_rate={info['max_abs_yaw_rate']:.6f} invalid={info['invalid_motion_frames']}")
    _print_distribution("accel distribution", df.accel_label, ACCEL_NAMES)
    _print_distribution("steer distribution", df.steer_label, STEER_NAMES)
    _save_histograms(df, out_dir)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KITTI Stage3 manifest from image/cam/pose sequences.")
    parser.add_argument("--raw-root", type=Path, default=KITTI_STAGE3_RAW)
    parser.add_argument("--out-dir", type=Path, default=KITTI_STAGE3_MANIFEST)
    parser.add_argument("--smooth-window", type=int, default=KITTI_SPEED_SMOOTH_WINDOW)
    parser.add_argument("--smooth-method", choices=["moving_average", "median"], default=KITTI_SPEED_SMOOTH_METHOD)
    parser.add_argument("--invert-yaw", action="store_true", default=KITTI_INVERT_YAW)
    parser.add_argument("--yaw-threshold", type=float, default=KITTI_YAW_RATE_THRESHOLD)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=KITTI_MAX_SPEED)
    parser.add_argument("--max-abs-acceleration", type=float, default=KITTI_MAX_ABS_ACCELERATION)
    args = parser.parse_args()
    build_manifest(args.raw_root, args.out_dir, args.smooth_window, args.smooth_method, args.invert_yaw, args.yaw_threshold, args.val_ratio, args.max_speed, args.max_abs_acceleration)


if __name__ == "__main__":
    main()
