from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CCD_STAGE2_MANIFEST


# ============================================================
# Configuration
# ============================================================

TOOLS_DIR = Path(__file__).resolve().parent

EGO_MANIFEST_PATH = (
    CCD_STAGE2_MANIFEST
    / "ego_candidates.csv"
)

COLLISION_CANDIDATES_PATH = (
    TOOLS_DIR
    / "data"
    / "stage2"
    / "CCD-1500"
    / "collision_candidates"
    / "collision_candidates.csv"
)

OUTPUT_DIR = (
    CCD_STAGE2_MANIFEST
    / "videomae"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "all.csv"
)


# ============================================================
# Loading
# ============================================================

def load_ego_manifest() -> pd.DataFrame:
    if not EGO_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {EGO_MANIFEST_PATH}"
        )

    df = pd.read_csv(
        EGO_MANIFEST_PATH,
        dtype={
            "video_id": str,
            "source_id": str,
        },
    )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.zfill(6)
    )

    return df


def load_collision_candidates() -> pd.DataFrame:
    if not COLLISION_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            "Collision candidate file not found: "
            f"{COLLISION_CANDIDATES_PATH}"
        )

    df = pd.read_csv(
        COLLISION_CANDIDATES_PATH,
        dtype={
            "video_id": str,
        },
    )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.zfill(6)
    )

    return df


# ============================================================
# Weak label helpers
# ============================================================

def build_top1_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    영상별 collision candidate rank 1만 사용.

    여기서 victim_track_id는 GT가 아니라
    weak/pseudo candidate임.
    """

    top1 = candidates[
        candidates["candidate_rank"] == 1
    ].copy()

    keep_columns = [
        "video_id",
        "track_id",
        "candidate_score",
        "approach_side",
        "start_x",
        "end_x",
        "first_frame",
        "last_frame",
        "window_coverage",
    ]

    top1 = top1[
        keep_columns
    ].copy()

    top1 = top1.rename(
        columns={
            "track_id": (
                "victim_track_id"
            ),
            "candidate_score": (
                "victim_candidate_score"
            ),
            "approach_side": (
                "victim_approach_side"
            ),
            "first_frame": (
                "victim_first_frame"
            ),
            "last_frame": (
                "victim_last_frame"
            ),
        }
    )

    return top1


def infer_weak_side(
    row: pd.Series,
) -> tuple[int, bool]:
    """
    매우 단순한 weak entry-side label.

    left  -> 0
    right -> 1

    center는 신뢰할 수 없으므로 invalid.
    """

    side = row.get(
        "victim_approach_side"
    )

    if side == "left":
        return 0, True

    if side == "right":
        return 1, True

    return -1, False


# ============================================================
# Build
# ============================================================

def build_manifest() -> pd.DataFrame:
    ego = load_ego_manifest()

    candidates = (
        load_collision_candidates()
    )

    top1 = build_top1_candidates(
        candidates
    )

    df = ego.merge(
        top1,
        on="video_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Collision
    # --------------------------------------------------------

    # CCD에서 직접 제공되는 temporal annotation.
    df["collision_frame"] = (
        df["accident_start_frame"]
        .astype(int)
    )

    df["collision_valid"] = True

    # 현재는 accident onset을 collision GT로 사용하지만,
    # semantic 차이를 기록해둔다.
    df["collision_label_source"] = (
        "ccd_accident_start"
    )

    # --------------------------------------------------------
    # Victim candidate
    # --------------------------------------------------------

    df["victim_valid"] = (
        df["victim_track_id"]
        .notna()
    )

    # --------------------------------------------------------
    # Entry frame
    # --------------------------------------------------------

    # 아직 신뢰할 수 있는 pseudo label을
    # 만들지 않았으므로 masking.
    df["entry_frame"] = -1
    df["entry_valid"] = False

    # --------------------------------------------------------
    # Entry side
    # --------------------------------------------------------

    side_results = df.apply(
        infer_weak_side,
        axis=1,
    )

    df["entry_side"] = [
        value
        for value, valid
        in side_results
    ]

    df["side_valid"] = [
        valid
        for value, valid
        in side_results
    ]

    df["side_label_source"] = np.where(
        df["side_valid"],
        "victim_track_start_side",
        "none",
    )

    # --------------------------------------------------------
    # Evasion space
    # --------------------------------------------------------

    # 아직 label 없음.
    df["evasion_space"] = -1
    df["space_valid"] = False

    # --------------------------------------------------------
    # Pseudo-label confidence
    # --------------------------------------------------------

    df["collision_weight"] = 1.0

    df["entry_weight"] = 0.0

    df["side_weight"] = (
        df["victim_candidate_score"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    df.loc[
        ~df["side_valid"],
        "side_weight",
    ] = 0.0

    df["space_weight"] = 0.0

    # --------------------------------------------------------
    # Keep useful columns
    # --------------------------------------------------------

    columns = [
        "video_id",
        "video_path",
        "source_id",

        # metadata
        "timing",
        "weather",

        # collision
        "collision_frame",
        "collision_valid",
        "collision_weight",
        "collision_label_source",

        # victim pseudo-track
        "victim_track_id",
        "victim_candidate_score",
        "victim_valid",

        "victim_first_frame",
        "victim_last_frame",

        "start_x",
        "end_x",
        "victim_approach_side",

        # entry
        "entry_frame",
        "entry_valid",
        "entry_weight",

        # side
        "entry_side",
        "side_valid",
        "side_weight",
        "side_label_source",

        # space
        "evasion_space",
        "space_valid",
        "space_weight",
    ]

    df = df[
        columns
    ].copy()

    return df


# ============================================================
# Summary
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    print(
        "=== CCD VideoMAE Manifest ==="
    )

    print(
        f"Total videos: {len(df)}"
    )

    print()

    print(
        "Valid labels:"
    )

    print(
        f"  collision: "
        f"{int(df['collision_valid'].sum())}"
    )

    print(
        f"  entry: "
        f"{int(df['entry_valid'].sum())}"
    )

    print(
        f"  side: "
        f"{int(df['side_valid'].sum())}"
    )

    print(
        f"  space: "
        f"{int(df['space_valid'].sum())}"
    )

    print()

    print(
        "Victim candidates:"
    )

    print(
        int(
            df["victim_valid"]
            .sum()
        )
    )

    print()

    if df["side_valid"].any():
        print(
            "Weak side labels:"
        )

        valid_side = df[
            df["side_valid"]
        ]

        print(
            valid_side[
                "entry_side"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()

    print(
        "Collision frame:"
    )

    print(
        df[
            "collision_frame"
        ]
        .describe()
        .to_string()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = build_manifest()

    print_summary(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()