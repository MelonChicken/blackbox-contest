from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CCD_STAGE2_COLLISION_CANDIDATES,
    CCD_STAGE2_MANIFEST,
    STAGE2_ALL_MANIFEST,
    STAGE2_MANIFEST,
)


EGO_MANIFEST_PATH = CCD_STAGE2_MANIFEST / "ego_candidates.csv"
COLLISION_CANDIDATES_PATH = CCD_STAGE2_COLLISION_CANDIDATES
OUTPUT_PATH = STAGE2_ALL_MANIFEST
MISSING_LABEL = -1


def _normalize_video_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(6)


def load_ego_manifest(path: str | Path = EGO_MANIFEST_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CCD ego manifest not found: {path}")
    df = pd.read_csv(path, dtype={"video_id": str, "source_id": str})
    df["video_id"] = _normalize_video_id(df["video_id"])
    return df


def load_collision_candidates(path: str | Path = COLLISION_CANDIDATES_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["video_id"])
    df = pd.read_csv(path, dtype={"video_id": str})
    df["video_id"] = _normalize_video_id(df["video_id"])
    return df


def build_top1_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "candidate_rank" not in candidates.columns:
        return pd.DataFrame(columns=["video_id"])
    top1 = candidates[candidates["candidate_rank"] == 1].copy()
    keep = [
        column
        for column in [
            "video_id",
            "track_id",
            "candidate_score",
            "approach_side",
            "first_frame",
            "last_frame",
        ]
        if column in top1.columns
    ]
    return top1[keep].copy()


def direction_from_candidate(value: object) -> int:
    if value == "left":
        return 0
    if value == "right":
        return 1
    return MISSING_LABEL


def build_manifest(
    ego_manifest_path: str | Path = EGO_MANIFEST_PATH,
    collision_candidates_path: str | Path = COLLISION_CANDIDATES_PATH,
) -> pd.DataFrame:
    ego = load_ego_manifest(ego_manifest_path)
    candidates = load_collision_candidates(collision_candidates_path)
    top1 = build_top1_candidates(candidates)

    df = ego.merge(top1, on="video_id", how="left", validate="one_to_one")
    has_candidate = df.get("track_id", pd.Series(index=df.index, dtype=object)).notna()
    direction = df.get("approach_side", pd.Series(index=df.index, dtype=object)).map(direction_from_candidate)

    manifest = pd.DataFrame(
        {
            "video_path": df["video_path"],
            "collision_frame": df["accident_start_frame"].astype(int),
            "entry_frame": MISSING_LABEL,
            "direction": direction.fillna(MISSING_LABEL).astype(int),
            "avoidance": MISSING_LABEL,
            "collision_source": "ccd_accident_start",
            "entry_source": "missing",
            "direction_source": has_candidate.map(lambda valid: "ccd_approach_side" if valid else "missing"),
            "avoidance_source": "missing",
            "collision_confidence": 1.0,
            "entry_confidence": 0.0,
        }
    )
    manifest.insert(0, "video_id", df["video_id"])
    return manifest


def main() -> None:
    STAGE2_MANIFEST.mkdir(parents=True, exist_ok=True)
    df = build_manifest()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Collision labels: {int((df['collision_frame'] >= 0).sum())}")
    print(f"Entry labels: {int((df['entry_frame'] >= 0).sum())}")
    print(f"Direction labels: {int((df['direction'] >= 0).sum())}")
    print(f"Avoidance labels: {int((df['avoidance'] >= 0).sum())}")


if __name__ == "__main__":
    main()
