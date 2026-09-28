from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    AIHUB_STAGE1_RAW,
    CCD_STAGE2_RAW,
    DATA_ROOT,
    STAGE2_HUMAN_LABEL_ROOT,
)


MISSING_LABEL = -1
REQUIRED_COLUMNS = {
    "id",
    "video",
    "NEW_collision_frame",
    "NEW_entry_frame",
    "NEW_entry_side",
    "NEW_evasion_space",
    "n_frames",
}
LABEL_SUFFIXES = {".csv", ".txt"}


def _int_label(value: Any, *, allowed: set[int] | None = None) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return MISSING_LABEL
    result = int(float(value))
    if allowed is not None and result not in allowed:
        return MISSING_LABEL
    return result


def _direction_label(value: Any) -> int:
    normalized = str(value).strip().upper()
    if normalized == "LEFT":
        return 0
    if normalized == "RIGHT":
        return 1
    return MISSING_LABEL


def _dataset_name(video_id: str, n_frames: int) -> str:
    return "aihub" if video_id.startswith("bb_") or n_frames > 50 else "ccd"


def _resolve_video_path(raw_path: Any, dataset: str, video_id: str) -> Path:
    value = Path(str(raw_path).replace("\\", "/"))
    candidates = [value] if value.is_absolute() else [DATA_ROOT / value]
    if dataset == "ccd":
        candidates.extend(
            [
                CCD_STAGE2_RAW / f"{video_id}.mp4",
                CCD_STAGE2_RAW / "videos" / f"{video_id}.mp4",
            ]
        )
    else:
        candidates.extend(
            [
                AIHUB_STAGE1_RAW / value.name,
                AIHUB_STAGE1_RAW / "1.Training" / "원천데이터_231108_add" / value.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def discover_human_label_files(root: str | Path = STAGE2_HUMAN_LABEL_ROOT) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LABEL_SUFFIXES):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(columns):
            files.append(path)
    return files


def _normalize_file(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, dtype={"id": str})
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Human label file is missing columns {missing}: {path}")

    rows = []
    for raw in source.to_dict("records"):
        video_id = str(raw["id"]).strip()
        n_frames = _int_label(raw["n_frames"])
        if not video_id or n_frames <= 0:
            raise ValueError(f"Invalid id/n_frames in {path}: id={video_id!r}, n_frames={n_frames}")
        dataset = _dataset_name(video_id, n_frames)
        collision = _int_label(raw["NEW_collision_frame"])
        entry = _int_label(raw["NEW_entry_frame"])
        direction = _direction_label(raw["NEW_entry_side"])
        avoidance = _int_label(raw["NEW_evasion_space"], allowed={0, 1})
        collision_valid = 0 <= collision < n_frames
        entry_in_range = 0 <= entry < n_frames
        entry_after_collision = entry_in_range and collision_valid and entry > collision
        entry_valid = entry_in_range and not entry_after_collision
        label_issue = "entry_after_collision" if entry_after_collision else ""
        if not collision_valid:
            raise ValueError(
                f"Human collision label is outside the video range: {video_id} "
                f"collision={collision}, n_frames={n_frames}"
            )

        rows.append(
            {
                "video_id": video_id.zfill(6) if dataset == "ccd" else video_id,
                "dataset": dataset,
                "video_path": str(_resolve_video_path(raw["video"], dataset, video_id)),
                "source_id": video_id,
                "n_frames": n_frames,
                "fps": 10.0 if dataset == "ccd" else float("nan"),
                "collision_frame": collision,
                "entry_frame": entry if entry_valid else MISSING_LABEL,
                "direction": direction,
                "avoidance": avoidance,
                "collision_valid": collision_valid,
                "entry_valid": entry_valid,
                "direction_valid": direction in {0, 1},
                "avoidance_valid": avoidance in {0, 1},
                "human_reviewed": True,
                "ambiguous": str(raw.get("ambiguous", "")).strip().upper() == "Y",
                "tier": str(raw.get("tier", "")).strip(),
                "lane_score": pd.to_numeric(raw.get("lane_score"), errors="coerce"),
                "labeler": str(raw.get("labeler", "")).strip(),
                "notes": str(raw.get("notes", "")).strip(),
                "label_issue": label_issue,
                "label_file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_stage2_human_labels(root: str | Path = STAGE2_HUMAN_LABEL_ROOT) -> pd.DataFrame:
    files = discover_human_label_files(root)
    if not files:
        return pd.DataFrame(columns=["video_id", "dataset"])
    result = pd.concat([_normalize_file(path) for path in files], ignore_index=True)
    duplicates = result.duplicated(["dataset", "video_id"], keep=False)
    if duplicates.any():
        values = result.loc[duplicates, ["dataset", "video_id", "label_file"]]
        raise ValueError(f"Duplicate Stage 2 human labels:\n{values.to_string(index=False)}")
    return result


def print_human_label_summary(labels: pd.DataFrame) -> None:
    print("=== Stage 2 Human Labels ===")
    if labels.empty:
        print(f"No label files found under: {STAGE2_HUMAN_LABEL_ROOT}")
        return
    for dataset, rows in labels.groupby("dataset", sort=True):
        print(
            f"{dataset}: rows={len(rows)} "
            f"collision={int(rows['collision_valid'].sum())} "
            f"entry={int(rows['entry_valid'].sum())} "
            f"direction={int(rows['direction_valid'].sum())} "
            f"avoidance={int(rows['avoidance_valid'].sum())}"
        )
    issues = labels[labels["label_issue"].ne("")]
    if not issues.empty:
        print("Excluded task labels requiring review:")
        print(issues[["dataset", "video_id", "label_issue"]].to_string(index=False))
