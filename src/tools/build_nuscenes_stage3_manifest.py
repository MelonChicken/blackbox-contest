from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    SEED,
    STAGE3_ACCEL_THRESHOLD,
    STAGE3_DECEL_THRESHOLD,
    STAGE3_NUSCENES_PROCESSED,
    STAGE3_NUSCENES_ROOT,
    STAGE3_NUSCENES_VERSION,
    STAGE3_NUSCENES_YAW_RATE_THRESHOLD,
    STAGE3_NUSCENES_YAW_SIGN,
    STAGE3_STOP_SPEED_THRESHOLD,
)
from src.tools.build_kitti_stage3_manifest import ACCEL_NAMES, STEER_NAMES, _print_distribution, _print_stats


def _labels(speed: np.ndarray, accel: np.ndarray, yaw_rate: np.ndarray, yaw_sign: int, yaw_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    accel_label = np.full(len(speed), 2, dtype=np.int64)
    accel_label[accel > STAGE3_ACCEL_THRESHOLD] = 0
    accel_label[accel < -STAGE3_DECEL_THRESHOLD] = 1
    accel_label[np.abs(speed) < STAGE3_STOP_SPEED_THRESHOLD] = 3

    signed_yaw = yaw_rate * int(yaw_sign)
    steer_label = np.full(len(speed), 1, dtype=np.int64)
    steer_label[signed_yaw > yaw_threshold] = 0
    steer_label[signed_yaw < -yaw_threshold] = 2
    return accel_label, steer_label


def _cam_front_frames(nusc, scene: dict) -> list[dict]:
    sample = nusc.get("sample", scene["first_sample_token"])
    token = sample["data"]["CAM_FRONT"]
    frames = []
    while token:
        sd = nusc.get("sample_data", token)
        frames.append({"timestamp": int(sd["timestamp"]), "filename": sd["filename"], "is_key_frame": bool(sd["is_key_frame"])})
        token = sd["next"]
    return frames


def _arrays(messages: list[dict], key: str, index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray([m["utime"] for m in messages], dtype=np.float64)
    if index is None:
        values = np.asarray([m[key] for m in messages], dtype=np.float64)
    else:
        values = np.asarray([m[key][index] for m in messages], dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], values[order]


def _scene_rows(nusc, nusc_can, scene: dict, root: Path, yaw_sign: int, yaw_threshold: float, min_stationary: int) -> list[dict]:
    scene_name = scene["name"]
    frames = _cam_front_frames(nusc, scene)
    try:
        pose = nusc_can.get_messages(scene_name, "pose")
        steer = nusc_can.get_messages(scene_name, "steeranglefeedback")
    except Exception as exc:
        print(f"[SKIP] {scene_name}: CAN loading failed: {exc}")
        return []
    if not pose or not steer:
        print(f"[SKIP] {scene_name}: pose={len(pose)} steer={len(steer)}")
        return []

    pose_ts, speed = _arrays(pose, "vel", 0)
    _, accel = _arrays(pose, "accel", 0)
    _, yaw_rate = _arrays(pose, "rotation_rate", 2)
    steer_ts, steering = _arrays(steer, "value")
    valid_start = max(pose_ts[0], steer_ts[0])
    valid_end = min(pose_ts[-1], steer_ts[-1])

    rows = []
    for frame_idx, frame in enumerate(frames):
        t = float(frame["timestamp"])
        path = root / frame["filename"]
        if t < valid_start or t > valid_end or not path.is_file():
            continue
        values = {
            "speed": float(np.interp(t, pose_ts, speed)),
            "accel": float(np.interp(t, pose_ts, accel)),
            "yaw_rate": float(np.interp(t, pose_ts, yaw_rate)),
            "steering_raw": float(np.interp(t, steer_ts, steering)),
        }
        if not np.isfinite(list(values.values())).all():
            continue
        rows.append({
            "scene": scene_name,
            "description": scene.get("description", ""),
            "frame_idx": frame_idx,
            "timestamp": int(t),
            "frame_path": frame["filename"],
            "is_key_frame": bool(frame["is_key_frame"]),
            **values,
        })
    if not rows:
        print(f"[SKIP] {scene_name}: no frames inside CAN/image range")
        return []

    df = pd.DataFrame(rows)
    stationary = df[(df.speed.abs() < 0.5) & (df.yaw_rate.abs() < 0.01)]
    if len(stationary) >= min_stationary:
        offset = float(stationary.steering_raw.median())
    else:
        offset = 0.0
        print(f"[WARN] {scene_name}: stationary samples={len(stationary)} < {min_stationary}; steering offset fallback=0")
    df["steering_offset"] = offset
    df["steering_centered"] = df.steering_raw - offset
    accel_label, steer_label = _labels(df.speed.to_numpy(float), df.accel.to_numpy(float), df.yaw_rate.to_numpy(float), yaw_sign, yaw_threshold)
    df["accel_label"] = accel_label
    df["steer_label"] = steer_label
    gaps = np.diff(df.timestamp.to_numpy(float)) / 1_000_000.0
    bad_gaps = int(((gaps < 0.04) | (gaps > 0.12)).sum())
    if bad_gaps:
        print(f"[WARN] {scene_name}: abnormal timestamp gaps={bad_gaps}")
    print(f"{scene_name}: CAM={len(frames)} valid={len(df)} pose={len(pose)} steer={len(steer)} offset={offset:.6f}")
    return df.to_dict("records")


def _split_scenes(df: pd.DataFrame, val_ratio: float) -> pd.Series:
    scenes = np.array(sorted(df.scene.unique()), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_ratio))) if len(scenes) > 1 and val_ratio > 0 else 0
    val = set(scenes[:n_val])
    return df.scene.map(lambda x: "val" if x in val else "train")


def build_manifest(root: Path, version: str, out_dir: Path, yaw_sign: int, yaw_threshold: float, val_ratio: float, min_stationary: int) -> Path:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(root), verbose=True)
    nusc_can = NuScenesCanBus(dataroot=str(root))
    rows = []
    for scene in nusc.scene:
        rows.extend(_scene_rows(nusc, nusc_can, scene, root, yaw_sign, yaw_threshold, min_stationary))
    if not rows:
        raise RuntimeError(f"no usable nuScenes frames under {root}")

    df = pd.DataFrame(rows)
    df["split"] = _split_scenes(df, val_ratio)
    cols = ["scene", "description", "frame_idx", "timestamp", "frame_path", "is_key_frame", "speed", "accel", "yaw_rate", "steering_raw", "steering_centered", "steering_offset", "accel_label", "steer_label", "split"]
    df = df[cols]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "nuscenes_mini_manifest.csv" if "mini" in version else "nuscenes_trainval_manifest.csv"
    path = out_dir / name
    df.to_csv(path, index=False)
    df[df.split == "train"].to_csv(out_dir / "train.csv", index=False)
    df[df.split == "val"].to_csv(out_dir / "val.csv", index=False)
    (out_dir / "split.json").write_text(json.dumps(dict(sorted(zip(df.scene, df.split))), indent=2), encoding="utf-8")

    print(f"number of scenes: {df.scene.nunique()}")
    print(f"number of valid frames: {len(df)}")
    _print_stats("speed", df.speed, [0, 25, 50, 75, 100])
    _print_stats("acceleration", df.accel, [0, 25, 50, 75, 100])
    _print_stats("yaw_rate", df.yaw_rate, [0, 25, 50, 75, 100])
    _print_stats("steering_raw", df.steering_raw, [0, 25, 50, 75, 100])
    _print_stats("steering_centered", df.steering_centered, [0, 25, 50, 75, 100])
    _print_distribution("accel class distribution", df.accel_label, ACCEL_NAMES)
    _print_distribution("steering class distribution", df.steer_label, STEER_NAMES)
    print(f"Manifest saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nuScenes Stage3 frame-level manifest.")
    parser.add_argument("--root", type=Path, default=STAGE3_NUSCENES_ROOT)
    parser.add_argument("--version", default=STAGE3_NUSCENES_VERSION)
    parser.add_argument("--out-dir", type=Path, default=STAGE3_NUSCENES_PROCESSED)
    parser.add_argument("--yaw-sign", type=int, choices=[-1, 1], default=STAGE3_NUSCENES_YAW_SIGN)
    parser.add_argument("--yaw-threshold", type=float, default=STAGE3_NUSCENES_YAW_RATE_THRESHOLD)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-stationary", type=int, default=5)
    args = parser.parse_args()
    build_manifest(args.root, args.version, args.out_dir, args.yaw_sign, args.yaw_threshold, args.val_ratio, args.min_stationary)


if __name__ == "__main__":
    main()