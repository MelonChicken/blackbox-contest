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
from src.tools.stage2_human_labels import load_stage2_human_labels, print_human_label_summary

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


def _apply_human_labels(out: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["dataset"] = "ccd"
    out["n_frames"] = EXPECTED_NUM_FRAMES
    out["fps"] = EXPECTED_FPS
    out["human_reviewed"] = False
    out["ambiguous"] = False
    out["tier"] = ""
    out["lane_score"] = float("nan")
    out["labeler"] = ""
    out["notes"] = ""
    out["label_issue"] = ""
    out["collision_valid"] = out["collision_frame"].ge(0)
    out["entry_valid"] = out["entry_frame"].ge(0)
    out["direction_valid"] = out["direction"].ge(0)
    out["avoidance_valid"] = out["avoidance"].ge(0)
    if human.empty:
        return out

    ccd = human[human["dataset"] == "ccd"].copy()
    if not ccd.empty:
        keep = [
            "video_id",
            "collision_frame",
            "entry_frame",
            "direction",
            "avoidance",
            "collision_valid",
            "entry_valid",
            "direction_valid",
            "avoidance_valid",
            "human_reviewed",
            "ambiguous",
            "tier",
            "lane_score",
            "labeler",
            "notes",
            "label_issue",
        ]
        out = out.merge(ccd[keep], on="video_id", how="left", validate="one_to_one", suffixes=("", "_human"))
        reviewed = out["human_reviewed_human"].eq(True)
        for task in ("collision", "entry", "direction", "avoidance"):
            valid = out[f"{task}_valid_human"].eq(True)
            target = f"{task}_frame" if task in {"collision", "entry"} else task
            out.loc[valid, target] = pd.to_numeric(out.loc[valid, f"{target}_human"], errors="raise")
            out[target] = pd.to_numeric(out[target], errors="raise").astype(int)
            out.loc[valid, f"{task}_source"] = "human_manual"
            out.loc[valid, f"{task}_confidence"] = 1.0
            out.loc[reviewed, f"{task}_valid"] = valid.loc[reviewed].astype(bool)
        for column in ("ambiguous", "tier", "lane_score", "labeler", "notes", "label_issue"):
            human_column = f"{column}_human"
            values = out.loc[reviewed, human_column]
            if column == "ambiguous":
                values = values.astype(bool)
            elif column == "lane_score":
                values = pd.to_numeric(values, errors="coerce")
            else:
                values = values.fillna("").astype(str)
            out.loc[reviewed, column] = values
        out["human_reviewed"] = reviewed
        out.loc[reviewed, "overall_confidence"] = 1.0
        out.loc[reviewed, "confidence_level"] = "human"
        out = out.drop(columns=[column for column in out.columns if column.endswith("_human")])

    aihub = human[human["dataset"] == "aihub"].copy()
    if not aihub.empty:
        added = pd.DataFrame(
            {
                "video_id": aihub["video_id"],
                "video_path": aihub["video_path"],
                "source_id": aihub["source_id"],
                "ego_involved": True,
                "ego_source": "human_stage2_label",
                "collision_frame": aihub["collision_frame"],
                "entry_frame": aihub["entry_frame"],
                "direction": aihub["direction"],
                "avoidance": aihub["avoidance"],
                "collision_source": "human_manual",
                "entry_source": aihub["entry_valid"].map(lambda valid: "human_manual" if valid else "missing"),
                "direction_source": aihub["direction_valid"].map(lambda valid: "human_manual" if valid else "missing"),
                "avoidance_source": aihub["avoidance_valid"].map(lambda valid: "human_manual" if valid else "missing"),
                "collision_confidence": 1.0,
                "entry_confidence": aihub["entry_valid"].astype(float),
                "direction_confidence": aihub["direction_valid"].astype(float),
                "avoidance_confidence": aihub["avoidance_valid"].astype(float),
                "overall_confidence": 1.0,
                "confidence_level": "human",
                "dataset": "aihub",
                "n_frames": aihub["n_frames"],
                "fps": aihub["fps"],
                "human_reviewed": True,
                "ambiguous": aihub["ambiguous"],
                "tier": aihub["tier"],
                "lane_score": aihub["lane_score"],
                "labeler": aihub["labeler"],
                "notes": aihub["notes"],
                "label_issue": aihub["label_issue"],
                "collision_valid": aihub["collision_valid"],
                "entry_valid": aihub["entry_valid"],
                "direction_valid": aihub["direction_valid"],
                "avoidance_valid": aihub["avoidance_valid"],
            }
        )
        out = pd.concat([out, added], ignore_index=True, sort=False)
    return out


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
            "avoidance_confidence": 0.0,
            "overall_confidence": df.get("overall_confidence", pd.Series(0.0, index=df.index)).fillna(0.0),
            "confidence_level": df.get("confidence_level", pd.Series("low", index=df.index)).fillna("low"),
        }
    )
    human = load_stage2_human_labels()
    print_human_label_summary(human)
    return _apply_human_labels(out, human)


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
    train_parts = []
    val_parts = []
    datasets = df["dataset"] if "dataset" in df.columns else pd.Series("ccd", index=df.index)
    for _, part in df.groupby(datasets, sort=True):
        part = part.reset_index(drop=True)
        if len(part) < 2:
            train_parts.append(part)
            continue
        groups = part["source_id"] if "source_id" in part.columns else part.get("video_id", part.index)
        train_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED).split(part, groups=groups)
        )
        train_parts.append(part.iloc[train_idx])
        val_parts.append(part.iloc[val_idx])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
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
