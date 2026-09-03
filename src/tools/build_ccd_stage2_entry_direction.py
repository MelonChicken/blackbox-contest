from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import CCD_STAGE2_BOTSORT_TRACKS, CCD_STAGE2_MANIFEST, STAGE2_MANIFEST

MANIFEST_PATH = CCD_STAGE2_MANIFEST / "ego_candidates.csv"
TRACK_DIR = CCD_STAGE2_BOTSORT_TRACKS
OUTPUT_PATH = STAGE2_MANIFEST / "ccd_stage2_entry_direction_pseudo_labels.csv"
DEBUG_DIR = STAGE2_MANIFEST / "entry_direction_debug"

FPS = 10.0
YOLO_CONF_THRESHOLD = 0.25
MIN_TRACK_LENGTH = 5
LOOKBACK_FRAMES = 20
MAX_END_DISTANCE = 10
MIN_TARGET_SCORE = 0.25
MIN_TARGET_SCORE_MARGIN = 0.04
ENTRY_PERSISTENCE_FRAMES = 3
MIN_ENTRY_TO_COLLISION_FRAMES = 2
MAX_ENTRY_TO_COLLISION_FRAMES = 45
MIN_LATERAL_DISPLACEMENT = 0.04
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
EGO_LANE_ROI_CONFIG = {
    "top_y": 0.48,
    "bottom_y": 0.98,
    "top_left_x": 0.43,
    "top_right_x": 0.57,
    "bottom_left_x": 0.18,
    "bottom_right_x": 0.82,
}
TARGET_SCORE_WEIGHTS = {
    "collision_proximity": 0.22,
    "bbox_growth": 0.16,
    "center_proximity": 0.20,
    "lateral_motion": 0.14,
    "track_stability": 0.18,
    "detection_confidence": 0.10,
}
STAGE2_ENTRY_CANDIDATE_SCORE_WEIGHTS = {
    "ego_path_convergence": 0.22,
    "bbox_growth": 0.14,
    "collision_proximity": 0.18,
    "path_intersection": 0.18,
    "lateral_convergence": 0.12,
    "collision_region": 0.10,
}
DIRECTION_LABELS = {0: "LEFT", 1: "RIGHT"}


def clamp01(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def normalize_video_id(value: object) -> str:
    return str(value).zfill(6)


def ego_lane_proxy_polygon(width: int, height: int) -> np.ndarray:
    c = EGO_LANE_ROI_CONFIG
    points = [
        (c["top_left_x"] * width, c["top_y"] * height),
        (c["top_right_x"] * width, c["top_y"] * height),
        (c["bottom_right_x"] * width, c["bottom_y"] * height),
        (c["bottom_left_x"] * width, c["bottom_y"] * height),
    ]
    return np.asarray(points, dtype=np.int32)


def point_inside_roi(x_norm: float, y_norm: float) -> bool:
    polygon = ego_lane_proxy_polygon(1000, 1000)
    return cv2.pointPolygonTest(polygon, (float(x_norm) * 1000, float(y_norm) * 1000), False) >= 0


def load_manifest(limit: int | None = None, video_ids: list[str] | None = None) -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing CCD ego manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH, dtype={"video_id": str, "source_id": str})
    df["video_id"] = df["video_id"].map(normalize_video_id)
    ego_involved = df["ego_involved"].astype(str).str.lower().isin({"true", "1", "yes"})
    if not ego_involved.all():
        raise RuntimeError(f"CCD ego manifest contains non-ego rows: {(~ego_involved).sum()}")
    df = df[ego_involved].reset_index(drop=True)
    if video_ids:
        wanted = {normalize_video_id(v) for v in video_ids}
        df = df[df["video_id"].isin(wanted)].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def load_tracks(video_id: str) -> pd.DataFrame:
    path = TRACK_DIR / f"{normalize_video_id(video_id)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing track CSV: {path}. Run python -m src.tools.build_ccd_vehicles_tracks first.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    df["video_id"] = df["video_id"].map(normalize_video_id)
    return df[df["confidence"].astype(float) >= YOLO_CONF_THRESHOLD].copy()


def robust_delta(values: np.ndarray, n: int = 3) -> float:
    if len(values) < 2:
        return 0.0
    n = min(n, len(values))
    return float(np.median(values[-n:]) - np.median(values[:n]))


def _empty_target_scores() -> dict[str, float]:
    return {
        "collision_proximity_score": 0.0,
        "bbox_growth_score": 0.0,
        "center_proximity_score": 0.0,
        "lateral_motion_score": 0.0,
        "track_stability_score": 0.0,
        "detection_confidence_score": 0.0,
        "target_score": 0.0,
    }


def target_features(track: pd.DataFrame, collision_frame: int) -> dict[str, Any]:
    track = track.sort_values("frame")
    pre = track[(track["frame"] <= collision_frame) & (track["frame"] >= max(0, collision_frame - LOOKBACK_FRAMES))]
    frames_all = track["frame"].astype(int).to_numpy()
    base = {
        "track_id": int(track["track_id"].iloc[0]),
        "track_start_frame": int(frames_all.min()) if len(frames_all) else -1,
        "track_end_frame": int(frames_all.max()) if len(frames_all) else -1,
        "track_length": int(track["frame"].nunique()),
        "mean_detection_confidence": float(track["confidence"].mean()) if len(track) else 0.0,
        "start_x": np.nan,
        "end_x": np.nan,
        "end_y": np.nan,
        "signed_lateral_motion": 0.0,
        "target_reject_reason": "",
        **_empty_target_scores(),
    }
    if pre.empty:
        base["target_reject_reason"] = "track_exists_but_geometry_fail"
        return base
    frames = pre["frame"].astype(int).to_numpy()
    if len(pre) < MIN_TRACK_LENGTH:
        base["target_reject_reason"] = "track_too_short"
    last_frame = int(frames.max())
    end_distance = collision_frame - last_frame
    if end_distance > MAX_END_DISTANCE:
        base["target_reject_reason"] = "track_exists_but_collision_proximity_fail"

    xs = pre["bottom_x_norm"].astype(float).to_numpy()
    ys = pre["bottom_y_norm"].astype(float).to_numpy()
    areas = pre["area_norm"].astype(float).to_numpy()
    conf = pre["confidence"].astype(float).to_numpy()
    start_x = float(np.median(xs[: min(3, len(xs))]))
    end_x = float(np.median(xs[-min(3, len(xs)) :]))
    end_y = float(np.median(ys[-min(3, len(ys)) :]))
    lateral_delta = robust_delta(xs)
    area_delta = robust_delta(areas)
    collision_proximity = 1.0 - min(max(end_distance, 0) / MAX_END_DISTANCE, 1.0)
    bbox_growth_score = clamp01(area_delta / 0.05)
    center_proximity_score = 1.0 - min(abs(end_x - 0.5) / 0.5, 1.0)
    lateral_motion_score = clamp01((abs(start_x - 0.5) - abs(end_x - 0.5)) / 0.25)
    track_stability_score = clamp01(len(np.unique(frames)) / min(LOOKBACK_FRAMES + 1, collision_frame + 1))
    detection_confidence_score = clamp01(float(np.mean(conf)))
    scores = {
        "collision_proximity_score": collision_proximity,
        "bbox_growth_score": bbox_growth_score,
        "center_proximity_score": center_proximity_score,
        "lateral_motion_score": lateral_motion_score,
        "track_stability_score": track_stability_score,
        "detection_confidence_score": detection_confidence_score,
    }
    target_score = sum(scores[f"{k}_score"] * w for k, w in TARGET_SCORE_WEIGHTS.items())
    if not base["target_reject_reason"] and target_score < MIN_TARGET_SCORE:
        base["target_reject_reason"] = "track_exists_but_below_target_score"
    base.update({
        "start_x": start_x,
        "end_x": end_x,
        "end_y": end_y,
        "signed_lateral_motion": lateral_delta,
        "target_score": float(target_score),
        **scores,
    })
    return base


def rank_target_candidates(tracks: pd.DataFrame, collision_frame: int) -> list[dict[str, Any]]:
    candidates = [target_features(track, collision_frame) for _, track in tracks.groupby("track_id")]
    candidates.sort(key=lambda row: row["target_score"], reverse=True)
    for rank, row in enumerate(candidates, start=1):
        row["candidate_rank"] = rank
    return candidates


def select_target_track(tracks: pd.DataFrame, collision_frame: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    if tracks.empty:
        return None, [], "no_vehicle_detection"
    if "track_id" not in tracks.columns or tracks["track_id"].isna().all():
        return None, [], "vehicle_detected_but_no_track"
    candidates = rank_target_candidates(tracks, collision_frame)
    viable = [row for row in candidates if not row["target_reject_reason"]]
    if not viable:
        return candidates[0] if candidates else None, candidates, candidates[0]["target_reject_reason"] if candidates else "no_vehicle_detection"
    best = viable[0]
    second = viable[1]["target_score"] if len(viable) > 1 else 0.0
    best["target_score_margin"] = float(best["target_score"] - second)
    if len(viable) > 1 and best["target_score_margin"] < MIN_TARGET_SCORE_MARGIN:
        return best, candidates, "ambiguous_entry_candidate"
    return best, candidates, ""

def detect_entry(track: pd.DataFrame, collision_frame: int) -> tuple[bool, int | None, dict[str, Any], str]:
    pre = track[track["frame"] <= collision_frame].sort_values("frame").copy()
    if len(pre) < MIN_TRACK_LENGTH:
        return False, None, {"lane_crossing_clarity": 0.0, "crossing_persistence": 0.0}, "track_too_short"
    inside = [point_inside_roi(x, y) for x, y in zip(pre["bottom_x_norm"], pre["bottom_y_norm"])]
    frames = pre["frame"].astype(int).tolist()
    if inside[0]:
        return False, None, {"lane_crossing_clarity": 0.0, "crossing_persistence": 0.0}, "already_inside_roi"
    for i in range(1, len(frames)):
        if inside[i] and not inside[i - 1]:
            window = inside[i : i + ENTRY_PERSISTENCE_FRAMES]
            persistence = float(np.mean(window)) if window else 0.0
            if len(window) >= ENTRY_PERSISTENCE_FRAMES and persistence >= 2 / 3:
                entry_frame = int(frames[i])
                gap = collision_frame - entry_frame
                clarity = clamp01((sum(inside[i:]) / max(1, len(inside[i:]))) * (gap / 10.0))
                if gap < MIN_ENTRY_TO_COLLISION_FRAMES:
                    return False, entry_frame, {"lane_crossing_clarity": clarity, "crossing_persistence": persistence}, "entry_too_close_to_collision"
                if gap > MAX_ENTRY_TO_COLLISION_FRAMES:
                    return False, entry_frame, {"lane_crossing_clarity": clarity, "crossing_persistence": persistence}, "entry_too_far_from_collision"
                return True, entry_frame, {"lane_crossing_clarity": clarity, "crossing_persistence": persistence}, ""
    return False, None, {"lane_crossing_clarity": 0.0, "crossing_persistence": 0.0}, "no_lane_crossing"


def infer_direction(track: pd.DataFrame, collision_frame: int, entry_frame: int | None) -> tuple[bool, int | None, dict[str, float], str]:
    pre = track[track["frame"] <= collision_frame].sort_values("frame")
    if entry_frame is not None:
        pre = pre[pre["frame"] <= entry_frame]
    empty = {
        "x_start": 0.0,
        "x_mid": 0.0,
        "x_end": 0.0,
        "delta_x": 0.0,
        "early_to_late_delta_x": 0.0,
        "trajectory_slope": 0.0,
        "lateral_displacement": 0.0,
        "trajectory_consistency": 0.0,
    }
    if len(pre) < MIN_TRACK_LENGTH:
        return False, None, empty, "direction_track_too_short"

    frames = pre["frame"].astype(float).to_numpy()
    xs = pre["bottom_x_norm"].astype(float).to_numpy()
    x_start = float(np.median(xs[: min(3, len(xs))]))
    x_mid = float(np.median(xs[max(0, len(xs) // 2 - 1) : min(len(xs), len(xs) // 2 + 2)]))
    x_end = float(np.median(xs[-min(3, len(xs)) :]))
    delta_x = x_end - x_start
    early_to_late_delta_x = robust_delta(xs)
    slope = float(delta_x / max(1.0, frames[-1] - frames[0]))
    motion = early_to_late_delta_x if abs(early_to_late_delta_x) >= abs(delta_x) * 0.5 else delta_x
    lateral_displacement = abs(motion)
    center_origin = abs(x_start - 0.5) < 0.05
    scores = {
        "x_start": x_start,
        "x_mid": x_mid,
        "x_end": x_end,
        "delta_x": float(delta_x),
        "early_to_late_delta_x": float(early_to_late_delta_x),
        "trajectory_slope": slope,
        "lateral_displacement": lateral_displacement,
        "trajectory_consistency": 0.0,
    }
    if lateral_displacement < MIN_LATERAL_DISPLACEMENT:
        reason = "direction_center_origin" if center_origin and abs(slope) < 0.003 else "direction_low_lateral_displacement"
        return False, None, scores, reason

    expected_sign = 1 if motion > 0 else -1
    deltas = np.diff(xs)
    meaningful = deltas[np.abs(deltas) >= 0.003]
    consistency = float(np.mean(np.sign(meaningful) == expected_sign)) if len(meaningful) else 0.0
    scores["trajectory_consistency"] = consistency
    direction = 0 if expected_sign > 0 else 1
    if consistency < 0.45:
        return False, direction, scores, "direction_inconsistent_trajectory"
    return True, direction, scores, ""

def confidence_level(value: float) -> str:
    if value >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if value >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def score_stage2_entry_candidate(tracks: pd.DataFrame, collision_frame: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    if tracks.empty:
        return None, [], "no_target_track"
    rows = []
    for _, track in tracks.groupby("track_id"):
        f = target_features(track, collision_frame)
        if f is None:
            continue
        track_pre = track[(track["frame"] <= collision_frame) & (track["frame"] >= max(0, collision_frame - LOOKBACK_FRAMES))].sort_values("frame")
        inside = [point_inside_roi(x, y) for x, y in zip(track_pre["bottom_x_norm"], track_pre["bottom_y_norm"])]
        path_intersection_score = 1.0 if any(inside) else 0.0
        ego_path_convergence_score = clamp01((abs(f["start_x"] - 0.5) - abs(f["end_x"] - 0.5)) / 0.25)
        lateral_convergence_score = f["lateral_motion_score"]
        collision_region_score = clamp01(1.0 - (abs(f["end_x"] - 0.5) / 0.42)) * clamp01(f["end_y"] / 0.65)
        components = {
            "ego_path_convergence_score": ego_path_convergence_score,
            "bbox_growth_score": f["bbox_growth_score"],
            "collision_proximity_score": f["collision_proximity_score"],
            "path_intersection_score": path_intersection_score,
            "lateral_convergence_score": lateral_convergence_score,
            "collision_region_score": collision_region_score,
        }
        score = sum(components[f"{name}_score"] * weight for name, weight in STAGE2_ENTRY_CANDIDATE_SCORE_WEIGHTS.items())
        rows.append({**f, **components, "stage2_entry_candidate_score": float(score)})
    if not rows:
        return None, [], "no_target_track"
    rows.sort(key=lambda row: row["stage2_entry_candidate_score"], reverse=True)
    best = rows[0]
    second = rows[1]["stage2_entry_candidate_score"] if len(rows) > 1 else 0.0
    best["stage2_entry_candidate_score_margin"] = float(best["stage2_entry_candidate_score"] - second)
    return best, rows, ""


def stage2_entry_status(entry_valid: bool, failure_reason: str) -> str:
    if entry_valid:
        return "valid"
    if failure_reason in {
        "no_target_track",
        "no_vehicle_detection",
        "vehicle_detected_but_no_track",
        "track_too_short",
        "track_exists_but_below_target_score",
        "track_exists_but_collision_proximity_fail",
        "track_exists_but_geometry_fail",
    }:
        return "no_target"
    if failure_reason == "already_inside_roi":
        return "already_inside"
    if failure_reason == "no_lane_crossing":
        return "no_crossing"
    if failure_reason == "entry_too_far_from_collision":
        return "not_observable"
    return "ambiguous"



def candidate_debug_values(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = candidates[0] if candidates else {}
    second = candidates[1] if len(candidates) > 1 else {}
    return {
        "num_candidate_tracks": len(candidates),
        "best_candidate_score": best.get("target_score", 0.0),
        "second_candidate_score": second.get("target_score", 0.0),
        "target_score_margin": best.get("target_score_margin", best.get("target_score", 0.0) - second.get("target_score", 0.0)),
        "best_track_length": best.get("track_length", 0),
        "best_track_start_frame": best.get("track_start_frame", np.nan),
        "best_track_end_frame": best.get("track_end_frame", np.nan),
        "best_collision_proximity": best.get("collision_proximity_score", 0.0),
        "best_bbox_growth": best.get("bbox_growth_score", 0.0),
        "best_center_proximity": best.get("center_proximity_score", 0.0),
        "best_lateral_motion": best.get("lateral_motion_score", 0.0),
        "best_track_stability": best.get("track_stability_score", 0.0),
        "best_detection_confidence": best.get("detection_confidence_score", 0.0),
    }

def process_video(row: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    video_id = normalize_video_id(row["video_id"])
    collision_frame = int(row["accident_start_frame"])
    tracks = load_tracks(video_id)
    entry_candidate, entry_candidates, candidate_failure = score_stage2_entry_candidate(tracks, collision_frame)
    base = {
        "video_id": video_id,
        "video_path": row["video_path"],
        "ego_involved": True,
        "ego_source": row.get("ego_source", "ccd_official_egoinvolve"),
        "collision_frame": collision_frame,
        "stage2_entry_status": "no_target" if candidate_failure else "ambiguous",
        "stage2_entry_candidate_score": 0.0 if entry_candidate is None else entry_candidate["stage2_entry_candidate_score"],
        "stage2_entry_candidate_confidence": "low" if entry_candidate is None else confidence_level(entry_candidate["stage2_entry_candidate_score"]),
        "target_confidence": 0.0,
        "target_confidence_level": "low",
        "num_vehicle_detections": int(len(tracks)),
        "num_tracks": int(tracks["track_id"].nunique()) if "track_id" in tracks else 0,
        "num_candidate_tracks": 0,
        "best_candidate_score": 0.0,
        "second_candidate_score": 0.0,
        "best_track_length": 0,
        "best_track_start_frame": np.nan,
        "best_track_end_frame": np.nan,
        "best_collision_proximity": 0.0,
        "best_bbox_growth": 0.0,
        "best_center_proximity": 0.0,
        "best_lateral_motion": 0.0,
        "best_track_stability": 0.0,
        "best_detection_confidence": 0.0,
        "ego_path_convergence_score": 0.0 if entry_candidate is None else entry_candidate["ego_path_convergence_score"],
        "path_intersection_score": 0.0 if entry_candidate is None else entry_candidate["path_intersection_score"],
        "lateral_convergence_score": 0.0 if entry_candidate is None else entry_candidate["lateral_convergence_score"],
        "collision_region_score": 0.0 if entry_candidate is None else entry_candidate["collision_region_score"],
        "target_track_id": np.nan,
        "target_score": 0.0,
        "target_score_margin": 0.0,
        "entry_valid": False,
        "entry_frame": np.nan,
        "entry_to_collision_frames": np.nan,
        "entry_to_collision_seconds": np.nan,
        "direction_valid": False,
        "direction": np.nan,
        "direction_name": "",
        "entry_confidence": 0.0,
        "direction_confidence": 0.0,
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "track_start_frame": np.nan,
        "track_end_frame": np.nan,
        "track_length": 0,
        "mean_detection_confidence": 0.0,
        "bbox_growth_score": 0.0,
        "center_proximity_score": 0.0,
        "lateral_motion_score": 0.0,
        "track_stability_score": 0.0,
        "detection_confidence_score": 0.0,
        "collision_proximity_score": 0.0,
        "lane_crossing_clarity": 0.0,
        "crossing_persistence": 0.0,
        "direction_x_start": 0.0,
        "direction_x_mid": 0.0,
        "direction_x_end": 0.0,
        "direction_delta_x": 0.0,
        "direction_early_to_late_delta_x": 0.0,
        "direction_trajectory_slope": 0.0,
        "direction_lateral_displacement": 0.0,
        "direction_trajectory_consistency": 0.0,
        "failure_reason": candidate_failure,
    }

    target, candidates, target_failure = select_target_track(tracks, collision_frame)
    base.update(candidate_debug_values(candidates or entry_candidates))
    if target is None:
        base["stage2_entry_status"] = "no_target"
        base["failure_reason"] = target_failure or "no_target_track"
        return base, entry_candidates
    if target_failure:
        base["stage2_entry_status"] = "ambiguous" if target_failure == "ambiguous_entry_candidate" else "no_target"
        base["failure_reason"] = target_failure
        return base, candidates

    target_conf = float(np.mean([target["target_score"], target["track_stability_score"], target["detection_confidence_score"], clamp01(target.get("target_score_margin", 0.0) / 0.20)]))
    base.update({
        "target_track_id": target["track_id"],
        "target_score": target["target_score"],
        "target_score_margin": target.get("target_score_margin", 0.0),
        "target_confidence": target_conf,
        "target_confidence_level": confidence_level(target_conf),
        "track_start_frame": target["track_start_frame"],
        "track_end_frame": target["track_end_frame"],
        "track_length": target["track_length"],
        "mean_detection_confidence": target["mean_detection_confidence"],
        "bbox_growth_score": target["bbox_growth_score"],
        "center_proximity_score": target["center_proximity_score"],
        "lateral_motion_score": target["lateral_motion_score"],
        "track_stability_score": target["track_stability_score"],
        "detection_confidence_score": target["detection_confidence_score"],
        "collision_proximity_score": target["collision_proximity_score"],
    })
    track = tracks[tracks["track_id"] == target["track_id"]]
    entry_valid, entry_frame, entry_scores, entry_failure = detect_entry(track, collision_frame)
    direction_valid, direction, direction_scores, direction_failure = infer_direction(track, collision_frame, entry_frame if entry_valid else None)
    base.update({
        "entry_valid": bool(entry_valid),
        "entry_frame": entry_frame if entry_frame is not None and entry_valid else np.nan,
        "lane_crossing_clarity": entry_scores["lane_crossing_clarity"],
        "crossing_persistence": entry_scores["crossing_persistence"],
        "direction_valid": bool(direction_valid),
        "direction": direction if direction is not None and direction_valid else np.nan,
        "direction_name": DIRECTION_LABELS.get(direction, "") if direction_valid else "",
        "direction_x_start": direction_scores["x_start"],
        "direction_x_mid": direction_scores["x_mid"],
        "direction_x_end": direction_scores["x_end"],
        "direction_delta_x": direction_scores["delta_x"],
        "direction_early_to_late_delta_x": direction_scores["early_to_late_delta_x"],
        "direction_trajectory_slope": direction_scores["trajectory_slope"],
        "direction_lateral_displacement": direction_scores["lateral_displacement"],
        "direction_trajectory_consistency": direction_scores["trajectory_consistency"],
    })
    if entry_valid and entry_frame is not None:
        gap = collision_frame - int(entry_frame)
        base["entry_to_collision_frames"] = gap
        base["entry_to_collision_seconds"] = float(gap / FPS)
    entry_conf = np.mean([
        target["track_stability_score"],
        target["detection_confidence_score"],
        entry_scores["lane_crossing_clarity"],
        entry_scores["crossing_persistence"],
        clamp01(target.get("target_score_margin", 0.0) / 0.20),
    ]) if entry_valid else 0.0
    direction_conf = np.mean([
        clamp01(direction_scores["lateral_displacement"] / 0.12),
        direction_scores["trajectory_consistency"],
        target["track_stability_score"],
        target["collision_proximity_score"],
        target["detection_confidence_score"],
    ]) if direction_valid else 0.0
    overall = min(float(entry_conf), float(direction_conf)) if entry_valid and direction_valid else max(float(entry_conf), float(direction_conf)) * 0.5
    reasons = [r for r in (entry_failure, direction_failure) if r]
    base.update({
        "stage2_entry_status": stage2_entry_status(bool(entry_valid), entry_failure),
        "entry_confidence": float(clamp01(entry_conf)),
        "direction_confidence": float(clamp01(direction_conf)),
        "overall_confidence": float(clamp01(overall)),
        "confidence_level": confidence_level(float(clamp01(overall))),
        "failure_reason": ";".join(reasons),
    })
    return base, candidates
def print_distribution(name: str, values: pd.Series) -> None:
    values = values.dropna().astype(float)
    print(f"{name}:")
    for stat in ("mean", "median", "min", "max"):
        print(f"  {stat}: {float(getattr(values, stat)()) if not values.empty else float('nan'):.3f}")


def print_level_counts(name: str, series: pd.Series) -> None:
    print(f"{name} confidence:")
    for level in ("high", "medium", "low"):
        print(f"  {level}: {int((series == level).sum())}")


def print_stats(df: pd.DataFrame) -> None:
    total = len(df)
    target_ok = int(df["target_track_id"].notna().sum())
    target_ambiguous = int((df["failure_reason"] == "ambiguous_entry_candidate").sum())
    no_target = int(((df["stage2_entry_status"] == "no_target") | df["failure_reason"].isin([
        "no_vehicle_detection",
        "vehicle_detected_but_no_track",
        "track_too_short",
        "track_exists_but_below_target_score",
        "track_exists_but_collision_proximity_fail",
        "track_exists_but_geometry_fail",
        "no_target_track",
    ])).sum())
    entry_ok = int(df["entry_valid"].astype(bool).sum())
    direction_ok = int(df["direction_valid"].astype(bool).sum())
    direction_only = int((~df["entry_valid"].astype(bool) & df["direction_valid"].astype(bool)).sum())
    print("=== CCD Official Ego Stage2 Candidates ===")
    print(f"Official ego videos: {total}")
    print(f"Target selected: {target_ok}")
    print(f"Target ambiguous: {target_ambiguous}")
    print(f"No target: {no_target}")
    print("No target breakdown:")
    for reason in (
        "no_vehicle_detection",
        "vehicle_detected_but_no_track",
        "track_too_short",
        "track_exists_but_below_target_score",
        "track_exists_but_collision_proximity_fail",
        "track_exists_but_geometry_fail",
        "no_target_track",
    ):
        print(f"  {reason}: {int((df['failure_reason'] == reason).sum())}")
    print(f"Entry valid: {entry_ok}")
    print(f"Entry already occurred: {int((df['stage2_entry_status'] == 'already_inside').sum())}")
    print(f"Entry no crossing: {int((df['stage2_entry_status'] == 'no_crossing').sum())}")
    print(f"Direction valid: {direction_ok}")
    print(f"Direction-only valid: {direction_only}")
    print(f"Direction ambiguous: {int(total - direction_ok)}")
    print_level_counts("Target", df["target_confidence_level"])
    print_level_counts("Entry", df["entry_confidence"].map(confidence_level))
    print_level_counts("Direction", df["direction_confidence"].map(confidence_level))
    print_distribution("best candidate score", df["best_candidate_score"])
    print_distribution("selected target score distribution", df.loc[df["target_track_id"].notna(), "target_score"])
    print_distribution("rejected best candidate score distribution", df.loc[df["target_track_id"].isna(), "best_candidate_score"])
    print_distribution("entry_confidence distribution", df["entry_confidence"])
    print_distribution("direction_confidence distribution", df["direction_confidence"])
    print_distribution("target_confidence distribution", df["target_confidence"])
    print("direction class distribution:")
    print(df.loc[df["direction_valid"].astype(bool), "direction_name"].value_counts().to_string() or "none")
    print("stage2_entry_status:")
    print(df["stage2_entry_status"].value_counts().to_string() or "none")
    print("failure_reason:")
    reasons = Counter(reason for cell in df["failure_reason"].fillna("") for reason in str(cell).split(";") if reason)
    print(pd.Series(reasons).sort_values(ascending=False).to_string() if reasons else "none")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build CCD Stage2 entry/direction pseudo-labels from official CCD ego samples and existing YOLO tracks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR)
    parser.add_argument("--preview-groups", action="store_true")
    parser.add_argument("--preview-per-group", type=int, default=5)
    args = parser.parse_args()

    manifest = load_manifest(limit=args.limit, video_ids=args.video_id)
    rows = []
    debug_rows = []
    for idx, row in manifest.iterrows():
        out, candidates = process_video(row)
        rows.append(out)
        for rank, candidate in enumerate(candidates[:5], start=1):
            debug_rows.append({"video_id": out["video_id"], "candidate_rank": rank, **candidate})
        print(f"[{idx + 1}/{len(manifest)}] {out['video_id']} status={out['stage2_entry_status']} candidate_score={out['stage2_entry_candidate_score']:.3f} target={out['target_track_id']} entry={out['entry_frame']} direction={out['direction_name']} confidence={out['overall_confidence']:.3f} reason={out['failure_reason']}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(debug_rows).to_csv(args.debug_dir / "target_candidates_debug.csv", index=False)
    print_stats(df)
    if args.preview_groups:
        from src.tools.preview_ccd_stage2_entry_direction import write_group_contact_sheets

        counts = write_group_contact_sheets(df, args.output, args.debug_dir / "preview_groups", per_group=args.preview_per_group)
        print(f"Representative preview: {counts}")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.debug_dir / 'target_candidates_debug.csv'}")

if __name__ == "__main__":
    main()
