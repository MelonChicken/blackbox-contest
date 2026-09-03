from __future__ import annotations

import ast
import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CCD_STAGE2_COLLISION_CANDIDATES,
    CCD_STAGE2_MANIFEST,
    CCD_STAGE2_RAW,
    STAGE2_ALL_MANIFEST,
    STAGE2_MANIFEST,
    STAGE2_TRAIN_MANIFEST,
    STAGE2_VAL_MANIFEST,
)

CCD_ROOT = CCD_STAGE2_RAW
ANNOTATION_PATH = CCD_ROOT / "Crash-1500.txt"
VIDEO_DIR = CCD_ROOT / "videos"
CCD_ALL_MANIFEST_PATH = CCD_STAGE2_MANIFEST / "all.csv"
EGO_MANIFEST_PATH = CCD_STAGE2_MANIFEST / "ego_candidates.csv"
COLLISION_CANDIDATES_PATH = CCD_STAGE2_COLLISION_CANDIDATES
PSEUDO_LABEL_PATH = STAGE2_MANIFEST / "ccd_stage2_entry_direction_pseudo_labels.csv"
EXPECTED_NUM_FRAMES = 50
EXPECTED_FPS = 10.0
MISSING_LABEL = -1
SEED = 42
VAL_SIZE = 0.15


def parse_annotation_line(line: str) -> dict:
    line = line.strip()
    if not line:
        raise ValueError("Empty annotation line")

    label_start = line.find("[")
    label_end = line.find("]")
    if label_start == -1 or label_end == -1:
        raise ValueError(f"Could not locate binlabels: {line[:100]}")

    vidname = line[:label_start].rstrip(",")
    binlabels = ast.literal_eval(line[label_start : label_end + 1])
    fields = next(csv.reader([line[label_end + 1 :].lstrip(",")]))
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields after binlabels, got {len(fields)}: {fields}")

    startframe, youtube_id, timing, weather, ego_involve = fields
    if len(binlabels) != EXPECTED_NUM_FRAMES:
        raise ValueError(f"{vidname}: expected {EXPECTED_NUM_FRAMES} labels, got {len(binlabels)}")
    if any(label not in (0, 1) for label in binlabels):
        raise ValueError(f"{vidname}: binlabels contains values other than 0/1")

    positive = [idx for idx, label in enumerate(binlabels) if label == 1]
    accident_start_frame = positive[0] if positive else -1
    accident_end_frame = positive[-1] if positive else -1
    return {
        "video_id": vidname,
        "source_id": youtube_id,
        "source_start_frame": int(startframe),
        "timing": timing,
        "weather": weather,
        "ego_involved": ego_involve.strip().lower() == "yes",
        "accident_start_frame": accident_start_frame,
        "accident_end_frame": accident_end_frame,
        "num_accident_frames": len(positive),
        "total_frames": EXPECTED_NUM_FRAMES,
        "fps": EXPECTED_FPS,
    }


def validate_temporal_labels(row: dict) -> None:
    if row["accident_start_frame"] < 0:
        raise ValueError(f"{row['video_id']}: crash video has no positive frame")
    if row["accident_end_frame"] < row["accident_start_frame"]:
        raise ValueError(f"{row['video_id']}: accident_end_frame < accident_start_frame")
    expected = row["accident_end_frame"] - row["accident_start_frame"] + 1
    if expected != row["num_accident_frames"]:
        raise ValueError(f"{row['video_id']}: non-contiguous accident labels detected")


def build_ccd_manifest() -> pd.DataFrame:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_PATH}")
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    rows = []
    with ANNOTATION_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = parse_annotation_line(line)
                validate_temporal_labels(row)
                video_path = VIDEO_DIR / f"{row['video_id']}.mp4"
                row["video_path"] = str(video_path)
                row["video_exists"] = video_path.exists()
                rows.append(row)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse line {line_number}") from exc

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No CCD annotations were parsed")
    if df["video_id"].duplicated().any():
        duplicated = df.loc[df["video_id"].duplicated(keep=False), "video_id"].tolist()
        raise RuntimeError(f"Duplicate video IDs detected: {duplicated[:10]}")
    return df


def build_manifest() -> pd.DataFrame:
    return build_ccd_manifest()

def write_ccd_manifests() -> pd.DataFrame:
    CCD_STAGE2_MANIFEST.mkdir(parents=True, exist_ok=True)
    df = build_ccd_manifest()
    missing = df.loc[~df["video_exists"], ["video_id", "video_path"]]
    if not missing.empty:
        print(missing.head(10).to_string(index=False))
        raise RuntimeError(f"{len(missing)} video files are missing")

    df.to_csv(CCD_ALL_MANIFEST_PATH, index=False)
    ego_df = df[df["ego_involved"]].copy().reset_index(drop=True)
    ego_df["ego_source"] = "ccd_official_egoinvolve"
    if not ego_df["ego_involved"].all():
        raise RuntimeError("ego_candidates.csv would contain non-ego rows")
    ego_df.to_csv(EGO_MANIFEST_PATH, index=False)
    print_summary(df, ego_df)
    return ego_df


def _normalize_video_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(6)


def load_collision_candidates() -> pd.DataFrame:
    if not COLLISION_CANDIDATES_PATH.exists():
        return pd.DataFrame(columns=["video_id"])
    df = pd.read_csv(COLLISION_CANDIDATES_PATH, dtype={"video_id": str})
    df["video_id"] = _normalize_video_id(df["video_id"])
    return df


def build_top1_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "candidate_rank" not in candidates.columns:
        return pd.DataFrame(columns=["video_id"])
    top1 = candidates[candidates["candidate_rank"] == 1].copy()
    keep = [c for c in ("video_id", "track_id", "candidate_score", "approach_side", "first_frame", "last_frame") if c in top1]
    return top1[keep].copy()


def direction_from_candidate(value: object) -> int:
    if value == "left":
        return 0
    if value == "right":
        return 1
    return MISSING_LABEL


def build_stage2_manifest() -> pd.DataFrame:
    if not EGO_MANIFEST_PATH.exists():
        write_ccd_manifests()
    ego = pd.read_csv(EGO_MANIFEST_PATH, dtype={"video_id": str, "source_id": str})
    ego["video_id"] = _normalize_video_id(ego["video_id"])
    top1 = build_top1_candidates(load_collision_candidates())
    df = ego.merge(top1, on="video_id", how="left", validate="one_to_one")
    pseudo = pd.DataFrame(columns=["video_id"])
    if PSEUDO_LABEL_PATH.exists():
        pseudo = pd.read_csv(PSEUDO_LABEL_PATH, dtype={"video_id": str})
        pseudo["video_id"] = _normalize_video_id(pseudo["video_id"])
        df = df.merge(pseudo, on="video_id", how="left", validate="one_to_one", suffixes=("", "_pseudo"))

    direction = df.get("direction", pd.Series(index=df.index, dtype=float))
    entry_valid = df.get("entry_valid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    direction_valid = df.get("direction_valid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    fallback_direction = df.get("approach_side", pd.Series(index=df.index, dtype=object)).map(direction_from_candidate)
    direction = direction.where(direction_valid, fallback_direction if not PSEUDO_LABEL_PATH.exists() else MISSING_LABEL)
    has_direction = direction.fillna(MISSING_LABEL).astype(int).ge(0)
    out = pd.DataFrame(
        {
            "video_id": df["video_id"],
            "video_path": df["video_path"],
            "source_id": df.get("source_id", ""),
            "ego_involved": True,
            "ego_source": df.get("ego_source", pd.Series("ccd_official_egoinvolve", index=df.index)),
            "collision_frame": df["accident_start_frame"].astype(int),
            "entry_frame": df.get("entry_frame", pd.Series(MISSING_LABEL, index=df.index)).where(entry_valid, MISSING_LABEL).fillna(MISSING_LABEL).astype(int),
            "direction": direction.fillna(MISSING_LABEL).astype(int),
            "avoidance": MISSING_LABEL,
            "collision_source": "ccd_accident_start",
            "entry_source": entry_valid.map(lambda ok: "ccd_yolo_track_roi_pseudo" if ok else "missing"),
            "direction_source": has_direction.map(lambda ok: "ccd_yolo_track_direction_pseudo" if ok else "missing"),
            "avoidance_source": "missing",
            "collision_confidence": 1.0,
            "entry_confidence": df.get("entry_confidence", pd.Series(0.0, index=df.index)).fillna(0.0),
            "direction_confidence": df.get("direction_confidence", pd.Series(0.0, index=df.index)).fillna(0.0),
            "overall_confidence": df.get("overall_confidence", pd.Series(0.0, index=df.index)).fillna(0.0),
            "confidence_level": df.get("confidence_level", pd.Series("low", index=df.index)).fillna("low"),
        }
    )
    return out


def write_stage2_manifest() -> pd.DataFrame:
    STAGE2_MANIFEST.mkdir(parents=True, exist_ok=True)
    df = build_stage2_manifest()
    df.to_csv(STAGE2_ALL_MANIFEST, index=False)
    print(f"Saved: {STAGE2_ALL_MANIFEST}")
    print(f"Rows: {len(df)}")
    return df


def split_stage2_manifest() -> None:
    if not STAGE2_ALL_MANIFEST.exists():
        write_stage2_manifest()
    df = pd.read_csv(STAGE2_ALL_MANIFEST, dtype={"video_id": str, "source_id": str})
    groups = df["source_id"] if "source_id" in df.columns else df.get("video_id", df.index)
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED).split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_df.to_csv(STAGE2_TRAIN_MANIFEST, index=False)
    val_df.to_csv(STAGE2_VAL_MANIFEST, index=False)
    print(f"Saved: {STAGE2_TRAIN_MANIFEST}")
    print(f"Saved: {STAGE2_VAL_MANIFEST}")


def print_summary(df: pd.DataFrame, ego_df: pd.DataFrame | None = None) -> None:
    print("=== CCD Stage2 Manifest ===")
    print(f"Total annotations: {len(df)}")
    print(f"Videos found: {int(df['video_exists'].sum())}/{len(df)}")
    print(f"Ego candidates: {len(ego_df) if ego_df is not None else int(df['ego_involved'].sum())}/{len(df)}")
    official_ego = len(ego_df) if ego_df is not None else int(df["ego_involved"].sum())
    print("=== CCD Official Ego Filter ===")
    print(f"Total crash videos: {len(df)}")
    print(f"Official ego-involved: {official_ego}")
    print(f"Excluded non-ego: {len(df) - official_ego}")
    print(f"Unique source videos: {df['source_id'].nunique()}")
    print("Timing:")
    for key, value in Counter(df["timing"]).items():
        print(f"  {key}: {value}")


def build_stage2_flow(include_tracking: bool = True) -> None:
    steps: list[tuple[str, Callable[[], object]]] = [("ccd manifest", write_ccd_manifests)]
    if include_tracking:
        from src.tools import build_ccd_collision_candidates, build_ccd_vehicles_tracks

        steps += [("vehicle tracks", build_ccd_vehicles_tracks.main), ("collision candidates", build_ccd_collision_candidates.main)]
    steps += [("stage2 manifest", write_stage2_manifest), ("stage2 split", split_stage2_manifest)]
    for index, (name, run_step) in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {name}")
        run_step()
        print()


def main() -> None:
    build_stage2_flow()


if __name__ == "__main__":
    main()
