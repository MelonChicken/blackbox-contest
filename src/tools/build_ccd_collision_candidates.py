from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CCD_STAGE2_BOTSORT_TRACKS,
    CCD_STAGE2_COLLISION_CANDIDATES,
    CCD_STAGE2_MANIFEST,
)


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = (
    CCD_STAGE2_MANIFEST
    / "ego_candidates.csv"
)

TRACK_DIR = CCD_STAGE2_BOTSORT_TRACKS

OUTPUT_PATH = CCD_STAGE2_COLLISION_CANDIDATES

OUTPUT_DIR = OUTPUT_PATH.parent
TOP_K = 3

# 사고 전 얼마나 볼 것인가
LOOKBACK_FRAMES = 15

# 지나치게 짧은 track 제외
MIN_TRACK_LENGTH = 3

# 사고 시점에서 몇 frame 떨어져 있어도 후보로 인정할지
MAX_END_DISTANCE = 6

# 초기 ego corridor approximation
EGO_CENTER_X = 0.5
EGO_CENTER_Y = 1.0


# ============================================================
# Loading
# ============================================================

def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(
        MANIFEST_PATH,
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


def load_tracks(
    video_id: str,
) -> pd.DataFrame:
    path = TRACK_DIR / f"{video_id}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Track file not found: {path}"
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return df


# ============================================================
# Helpers
# ============================================================

def robust_first_last(
    values: np.ndarray,
    n: int = 3,
) -> tuple[float, float]:
    """
    단일 frame noise를 줄이기 위해
    첫 n개 / 마지막 n개의 median을 사용.
    """

    if len(values) == 0:
        return np.nan, np.nan

    n = min(n, len(values))

    start = float(
        np.median(values[:n])
    )

    end = float(
        np.median(values[-n:])
    )

    return start, end


def ego_distance(
    x: float,
    y: float,
) -> float:
    """
    normalized 좌표 기준으로
    화면 중앙 하단 ego collision zone과 거리.
    """

    dx = x - EGO_CENTER_X

    # y 방향 거리를 조금 덜 강하게 반영
    dy = (y - EGO_CENTER_Y) * 0.7

    return float(
        np.sqrt(
            dx ** 2
            + dy ** 2
        )
    )


def normalize_positive(
    value: float,
    scale: float,
) -> float:
    """
    0 이상인 값을 대략 [0, 1]로 변환.
    """

    if value <= 0:
        return 0.0

    return float(
        min(
            value / scale,
            1.0,
        )
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_track_features(
    track_df: pd.DataFrame,
    accident_frame: int,
) -> dict | None:

    track_df = (
        track_df
        .sort_values("frame")
        .copy()
    )

    # 사고 이후의 track은 제외하고
    # 사고 이전 + 사고 frame만 사용
    track_df = track_df[
        track_df["frame"]
        <= accident_frame
    ]

    if track_df.empty:
        return None

    window_start = max(
        0,
        accident_frame
        - LOOKBACK_FRAMES,
    )

    track_df = track_df[
        track_df["frame"]
        >= window_start
    ]

    if len(track_df) < MIN_TRACK_LENGTH:
        return None

    frames = (
        track_df["frame"]
        .astype(int)
        .to_numpy()
    )

    first_frame = int(
        frames.min()
    )

    last_frame = int(
        frames.max()
    )

    track_length = len(
        np.unique(frames)
    )

    end_distance_frames = (
        accident_frame
        - last_frame
    )

    if (
        end_distance_frames
        > MAX_END_DISTANCE
    ):
        return None

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    xs = (
        track_df["bottom_x_norm"]
        .astype(float)
        .to_numpy()
    )

    ys = (
        track_df["bottom_y_norm"]
        .astype(float)
        .to_numpy()
    )

    start_x, end_x = robust_first_last(
        xs
    )

    start_y, end_y = robust_first_last(
        ys
    )

    signed_lateral_motion = (
        end_x - start_x
    )

    lateral_motion = abs(
        signed_lateral_motion
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    areas = (
        track_df["area_norm"]
        .astype(float)
        .to_numpy()
    )

    start_area, end_area = (
        robust_first_last(areas)
    )

    max_area = float(
        np.max(areas)
    )

    area_growth = (
        end_area
        - start_area
    )

    if start_area > 1e-8:
        area_growth_ratio = (
            end_area
            / start_area
        )
    else:
        area_growth_ratio = 1.0

    # --------------------------------------------------------
    # Ego approach
    # --------------------------------------------------------

    start_ego_distance = (
        ego_distance(
            start_x,
            start_y,
        )
    )

    end_ego_distance = (
        ego_distance(
            end_x,
            end_y,
        )
    )

    ego_approach = (
        start_ego_distance
        - end_ego_distance
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    expected_window = (
        accident_frame
        - window_start
        + 1
    )

    window_coverage = (
        track_length
        / expected_window
    )

    reaches_accident = (
        last_frame
        == accident_frame
    )

    # --------------------------------------------------------
    # Side information
    # --------------------------------------------------------

    if start_x < 0.4:
        approach_side = "left"

    elif start_x > 0.6:
        approach_side = "right"

    else:
        approach_side = "center"

    toward_center = (
        abs(start_x - 0.5)
        - abs(end_x - 0.5)
    )

    # --------------------------------------------------------
    # Raw feature return
    # --------------------------------------------------------

    return {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "track_length": track_length,

        "frames_before_accident_end": (
            end_distance_frames
        ),

        "reaches_accident": (
            reaches_accident
        ),

        "window_coverage": (
            window_coverage
        ),

        "start_x": start_x,
        "end_x": end_x,
        "start_y": start_y,
        "end_y": end_y,

        "signed_lateral_motion": (
            signed_lateral_motion
        ),

        "lateral_motion": (
            lateral_motion
        ),

        "toward_center": (
            toward_center
        ),

        "approach_side": (
            approach_side
        ),

        "start_area": (
            start_area
        ),

        "end_area": (
            end_area
        ),

        "max_area": (
            max_area
        ),

        "area_growth": (
            area_growth
        ),

        "area_growth_ratio": (
            area_growth_ratio
        ),

        "start_ego_distance": (
            start_ego_distance
        ),

        "end_ego_distance": (
            end_ego_distance
        ),

        "ego_approach": (
            ego_approach
        ),
    }


# ============================================================
# Scoring
# ============================================================

def calculate_candidate_score(
    f: dict,
) -> dict:

    # --------------------------------------------------------
    # 1. Accident proximity
    # --------------------------------------------------------

    accident_proximity = (
        1.0
        - min(
            f[
                "frames_before_accident_end"
            ]
            / MAX_END_DISTANCE,
            1.0,
        )
    )

    # --------------------------------------------------------
    # 2. Track continuity
    # --------------------------------------------------------

    continuity = min(
        f["window_coverage"],
        1.0,
    )

    # --------------------------------------------------------
    # 3. Ego-path approach
    # --------------------------------------------------------

    approach_score = (
        normalize_positive(
            f["ego_approach"],
            scale=0.30,
        )
    )

    # --------------------------------------------------------
    # 4. Center approach
    # --------------------------------------------------------

    center_score = (
        normalize_positive(
            f["toward_center"],
            scale=0.25,
        )
    )

    # --------------------------------------------------------
    # 5. Lateral movement
    # --------------------------------------------------------

    lateral_score = (
        normalize_positive(
            f["lateral_motion"],
            scale=0.25,
        )
    )

    # --------------------------------------------------------
    # 6. Object becoming larger
    # --------------------------------------------------------

    area_score = (
        normalize_positive(
            f["area_growth"],
            scale=0.08,
        )
    )

    # --------------------------------------------------------
    # 7. End position near ego corridor
    # --------------------------------------------------------

    final_position_score = (
        1.0
        - min(
            f["end_ego_distance"]
            / 0.5,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Weighted score
    # --------------------------------------------------------

    score = (
        accident_proximity * 0.20
        + continuity * 0.15
        + approach_score * 0.20
        + center_score * 0.15
        + lateral_score * 0.10
        + area_score * 0.10
        + final_position_score * 0.10
    )

    return {
        "accident_proximity_score": (
            accident_proximity
        ),

        "continuity_score": (
            continuity
        ),

        "ego_approach_score": (
            approach_score
        ),

        "center_approach_score": (
            center_score
        ),

        "lateral_score": (
            lateral_score
        ),

        "area_score": (
            area_score
        ),

        "final_position_score": (
            final_position_score
        ),

        "candidate_score": (
            float(score)
        ),
    }


# ============================================================
# Video processing
# ============================================================

def process_video(
    video_id: str,
    accident_frame: int,
    tracks: pd.DataFrame,
) -> pd.DataFrame:

    if tracks.empty:
        return pd.DataFrame()

    rows = []

    for (
        track_id,
        track_df,
    ) in tracks.groupby(
        "track_id"
    ):

        features = (
            extract_track_features(
                track_df,
                accident_frame,
            )
        )

        if features is None:
            continue

        scores = (
            calculate_candidate_score(
                features
            )
        )

        # 가장 많이 등장한 class 사용
        class_name = (
            track_df[
                "class_name"
            ]
            .mode()
            .iloc[0]
        )

        confidence = float(
            track_df[
                "confidence"
            ].mean()
        )

        rows.append(
            {
                "video_id": (
                    video_id
                ),

                "accident_frame": (
                    accident_frame
                ),

                "track_id": int(
                    track_id
                ),

                "class_name": (
                    class_name
                ),

                "mean_detection_confidence": (
                    confidence
                ),

                **features,
                **scores,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            "candidate_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["candidate_rank"] = (
        np.arange(
            1,
            len(result) + 1,
        )
    )

    return result.head(
        TOP_K
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        load_manifest()
    )

    print(
        "=== CCD Collision Candidate Builder ==="
    )

    print(
        f"Videos: {len(manifest)}"
    )

    print(
        f"Tracker: BoT-SORT"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print()

    all_results = []

    no_candidates = []

    for idx, row in (
        manifest.iterrows()
    ):

        video_id = (
            row["video_id"]
        )

        accident_frame = int(
            row[
                "accident_start_frame"
            ]
        )

        try:
            tracks = load_tracks(
                video_id
            )

            result = process_video(
                video_id=video_id,
                accident_frame=(
                    accident_frame
                ),
                tracks=tracks,
            )

            if result.empty:
                no_candidates.append(
                    video_id
                )

                print(
                    f"[{idx + 1}/{len(manifest)}] "
                    f"{video_id} | "
                    f"no candidate"
                )

                continue

            all_results.append(
                result
            )

            best = result.iloc[0]

            print(
                f"[{idx + 1}/{len(manifest)}] "
                f"{video_id} | "
                f"track={int(best['track_id'])} | "
                f"score={best['candidate_score']:.3f} | "
                f"side={best['approach_side']} | "
                f"x={best['start_x']:.2f}"
                f"->{best['end_x']:.2f}"
            )

        except Exception as exc:

            print(
                f"[{idx + 1}/{len(manifest)}] "
                f"{video_id} FAILED: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No collision candidates generated."
        )

    output = pd.concat(
        all_results,
        ignore_index=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=== Completed ==="
    )

    print(
        f"Videos with candidates: "
        f"{output['video_id'].nunique()}"
    )

    print(
        f"No candidate videos: "
        f"{len(no_candidates)}"
    )

    print()

    print(
        "Top-1 candidate score:"
    )

    top1 = output[
        output[
            "candidate_rank"
        ]
        == 1
    ]

    print(
        top1[
            "candidate_score"
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        "Top-1 approach side:"
    )

    print(
        top1[
            "approach_side"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
