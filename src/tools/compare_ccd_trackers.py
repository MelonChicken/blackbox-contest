from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CCD_STAGE2_MANIFEST

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MANIFEST_PATH = CCD_STAGE2_MANIFEST / 'ego_candidates.csv'

TOOLS_DIR = Path(__file__).resolve().parent

TRACK_ROOT = (
    TOOLS_DIR
    / "data"
    / "stage2"
    / "CCD-1500"
    / "tracks"
)

BYTETRACK_DIR = TRACK_ROOT / "bytetrack"
BOTSORT_DIR = TRACK_ROOT / "botsort"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage2"
    / "CCD-1500"
    / "tracker_comparison"
)

DETAIL_OUTPUT = OUTPUT_DIR / "track_comparison.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "summary.csv"

# 사고 직전 몇 frame을 중요하게 볼지
WINDOW_SIZE = 10

# 너무 짧게 잡힌 track은 candidate에서 제외
MIN_TRACK_FRAMES = 2


# ============================================================
# Loading
# ============================================================

def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

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


def load_track_file(
    track_dir: Path,
    video_id: str,
) -> pd.DataFrame:
    path = track_dir / f"{video_id}.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return df


# ============================================================
# Metrics
# ============================================================

def count_gaps(frames: list[int]) -> tuple[int, int, int]:
    """
    Returns:
        missing_frames
        num_gaps
        max_gap
    """

    if len(frames) <= 1:
        return 0, 0, 0

    frames = sorted(set(frames))

    missing_frames = 0
    num_gaps = 0
    max_gap = 0

    for prev_frame, next_frame in zip(
        frames[:-1],
        frames[1:],
    ):
        gap = next_frame - prev_frame - 1

        if gap > 0:
            missing_frames += gap
            num_gaps += 1
            max_gap = max(
                max_gap,
                gap,
            )

    return (
        missing_frames,
        num_gaps,
        max_gap,
    )


def analyze_track(
    track_df: pd.DataFrame,
    accident_frame: int,
    window_start: int,
    window_end: int,
) -> dict:
    frames = sorted(
        track_df["frame"]
        .astype(int)
        .unique()
        .tolist()
    )

    first_frame = min(frames)
    last_frame = max(frames)

    full_track_length = len(frames)

    window_frames = [
        frame
        for frame in frames
        if window_start <= frame <= window_end
    ]

    window_length = (
        window_end
        - window_start
        + 1
    )

    window_coverage = (
        len(window_frames)
        / window_length
    )

    (
        missing_frames,
        num_gaps,
        max_gap,
    ) = count_gaps(window_frames)

    reaches_accident = (
        accident_frame in frames
    )

    frames_from_accident = (
        last_frame - accident_frame
    )

    # 사고 시점에 가까울수록 0에 가까움.
    end_distance_to_accident = abs(
        frames_from_accident
    )

    # trajectory 특성
    ordered = track_df.sort_values("frame")

    start_x = float(
        ordered.iloc[0]["bottom_x_norm"]
    )

    end_x = float(
        ordered.iloc[-1]["bottom_x_norm"]
    )

    lateral_motion = abs(
        end_x - start_x
    )

    start_area = float(
        ordered.iloc[0]["area_norm"]
    )

    end_area = float(
        ordered.iloc[-1]["area_norm"]
    )

    max_area = float(
        ordered["area_norm"].max()
    )

    area_growth = (
        end_area - start_area
    )

    return {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "track_length": full_track_length,

        "window_track_frames": len(
            window_frames
        ),
        "window_coverage": window_coverage,

        "missing_frames": missing_frames,
        "num_gaps": num_gaps,
        "max_gap": max_gap,

        "reaches_accident": reaches_accident,
        "frames_from_accident": (
            frames_from_accident
        ),
        "end_distance_to_accident": (
            end_distance_to_accident
        ),

        "start_x": start_x,
        "end_x": end_x,
        "lateral_motion": lateral_motion,

        "start_area": start_area,
        "end_area": end_area,
        "max_area": max_area,
        "area_growth": area_growth,
    }


def analyze_tracker(
    tracker_name: str,
    track_df: pd.DataFrame,
    video_id: str,
    accident_frame: int,
) -> pd.DataFrame:
    if track_df.empty:
        return pd.DataFrame()

    window_start = max(
        0,
        accident_frame - WINDOW_SIZE,
    )

    window_end = accident_frame

    rows: list[dict] = []

    for track_id, group in track_df.groupby(
        "track_id"
    ):
        metrics = analyze_track(
            track_df=group,
            accident_frame=accident_frame,
            window_start=window_start,
            window_end=window_end,
        )

        if (
            metrics["track_length"]
            < MIN_TRACK_FRAMES
        ):
            continue

        rows.append(
            {
                "video_id": video_id,
                "tracker": tracker_name,
                "track_id": int(track_id),
                "accident_frame": (
                    accident_frame
                ),
                "window_start": window_start,
                "window_end": window_end,
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # 사고 직전 안정성을 우선해서 candidate score 구성.
    #
    # 이 score는 실제 충돌 상대 차량을 확정하는 score가 아니라
    # tracker continuity 비교용 score임.
    result["continuity_score"] = (
        result["window_coverage"] * 0.65
        + result["reaches_accident"].astype(float)
        * 0.20
        + (
            1.0
            - (
                result["end_distance_to_accident"]
                / max(
                    1,
                    WINDOW_SIZE,
                )
            ).clip(0, 1)
        )
        * 0.15
    )

    return result


# ============================================================
# Best candidate
# ============================================================

def get_best_track(
    df: pd.DataFrame,
) -> pd.Series | None:
    if df.empty:
        return None

    # continuity가 가장 높은 track 선택
    best = (
        df.sort_values(
            [
                "continuity_score",
                "window_coverage",
                "track_length",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return best


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    all_details: list[pd.DataFrame] = []
    summaries: list[dict] = []

    print("=== CCD Tracker Comparison ===")
    print(f"Videos: {len(manifest)}")
    print(
        f"Collision window: "
        f"-{WINDOW_SIZE} ~ 0 frames"
    )
    print()

    for idx, row in manifest.iterrows():
        video_id = row["video_id"]

        accident_frame = int(
            row["accident_start_frame"]
        )

        byte_df = load_track_file(
            BYTETRACK_DIR,
            video_id,
        )

        bot_df = load_track_file(
            BOTSORT_DIR,
            video_id,
        )

        byte_analysis = analyze_tracker(
            tracker_name="bytetrack",
            track_df=byte_df,
            video_id=video_id,
            accident_frame=accident_frame,
        )

        bot_analysis = analyze_tracker(
            tracker_name="botsort",
            track_df=bot_df,
            video_id=video_id,
            accident_frame=accident_frame,
        )

        if not byte_analysis.empty:
            all_details.append(
                byte_analysis
            )

        if not bot_analysis.empty:
            all_details.append(
                bot_analysis
            )

        byte_best = get_best_track(
            byte_analysis
        )

        bot_best = get_best_track(
            bot_analysis
        )

        summary = {
            "video_id": video_id,
            "accident_frame": accident_frame,

            "bytetrack_tracks": (
                byte_df["track_id"].nunique()
                if not byte_df.empty
                else 0
            ),

            "botsort_tracks": (
                bot_df["track_id"].nunique()
                if not bot_df.empty
                else 0
            ),
        }

        # --------------------------------------------
        # ByteTrack
        # --------------------------------------------

        if byte_best is not None:
            summary.update(
                {
                    "bytetrack_best_id": int(
                        byte_best[
                            "track_id"
                        ]
                    ),

                    "bytetrack_coverage": float(
                        byte_best[
                            "window_coverage"
                        ]
                    ),

                    "bytetrack_track_length": int(
                        byte_best[
                            "track_length"
                        ]
                    ),

                    "bytetrack_missing": int(
                        byte_best[
                            "missing_frames"
                        ]
                    ),

                    "bytetrack_gaps": int(
                        byte_best[
                            "num_gaps"
                        ]
                    ),

                    "bytetrack_max_gap": int(
                        byte_best[
                            "max_gap"
                        ]
                    ),

                    "bytetrack_reaches_accident": bool(
                        byte_best[
                            "reaches_accident"
                        ]
                    ),

                    "bytetrack_score": float(
                        byte_best[
                            "continuity_score"
                        ]
                    ),
                }
            )

        else:
            summary.update(
                {
                    "bytetrack_best_id": None,
                    "bytetrack_coverage": 0.0,
                    "bytetrack_track_length": 0,
                    "bytetrack_missing": 0,
                    "bytetrack_gaps": 0,
                    "bytetrack_max_gap": 0,
                    "bytetrack_reaches_accident": False,
                    "bytetrack_score": 0.0,
                }
            )

        # --------------------------------------------
        # BoT-SORT
        # --------------------------------------------

        if bot_best is not None:
            summary.update(
                {
                    "botsort_best_id": int(
                        bot_best[
                            "track_id"
                        ]
                    ),

                    "botsort_coverage": float(
                        bot_best[
                            "window_coverage"
                        ]
                    ),

                    "botsort_track_length": int(
                        bot_best[
                            "track_length"
                        ]
                    ),

                    "botsort_missing": int(
                        bot_best[
                            "missing_frames"
                        ]
                    ),

                    "botsort_gaps": int(
                        bot_best[
                            "num_gaps"
                        ]
                    ),

                    "botsort_max_gap": int(
                        bot_best[
                            "max_gap"
                        ]
                    ),

                    "botsort_reaches_accident": bool(
                        bot_best[
                            "reaches_accident"
                        ]
                    ),

                    "botsort_score": float(
                        bot_best[
                            "continuity_score"
                        ]
                    ),
                }
            )

        else:
            summary.update(
                {
                    "botsort_best_id": None,
                    "botsort_coverage": 0.0,
                    "botsort_track_length": 0,
                    "botsort_missing": 0,
                    "botsort_gaps": 0,
                    "botsort_max_gap": 0,
                    "botsort_reaches_accident": False,
                    "botsort_score": 0.0,
                }
            )

        summaries.append(summary)

        print(
            f"[{idx + 1}/{len(manifest)}] "
            f"{video_id} | "
            f"Byte={summary['bytetrack_coverage']:.2f} | "
            f"BoT={summary['botsort_coverage']:.2f}"
        )

    # ========================================================
    # Save
    # ========================================================

    if all_details:
        detail_df = pd.concat(
            all_details,
            ignore_index=True,
        )

        detail_df.to_csv(
            DETAIL_OUTPUT,
            index=False,
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df["coverage_diff"] = (
        summary_df["botsort_coverage"]
        - summary_df["bytetrack_coverage"]
    )

    summary_df["score_diff"] = (
        summary_df["botsort_score"]
        - summary_df["bytetrack_score"]
    )

    summary_df["winner"] = "tie"

    summary_df.loc[
        summary_df["score_diff"] > 0.05,
        "winner",
    ] = "botsort"

    summary_df.loc[
        summary_df["score_diff"] < -0.05,
        "winner",
    ] = "bytetrack"

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # Overall summary
    # ========================================================

    print()
    print("=== Overall Comparison ===")

    print(
        "Mean collision-window coverage:"
    )

    print(
        f"  ByteTrack: "
        f"{summary_df['bytetrack_coverage'].mean():.4f}"
    )

    print(
        f"  BoT-SORT : "
        f"{summary_df['botsort_coverage'].mean():.4f}"
    )

    print()
    print(
        "Mean continuity score:"
    )

    print(
        f"  ByteTrack: "
        f"{summary_df['bytetrack_score'].mean():.4f}"
    )

    print(
        f"  BoT-SORT : "
        f"{summary_df['botsort_score'].mean():.4f}"
    )

    print()
    print("Winner counts:")

    print(
        summary_df[
            "winner"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Reached accident frame:"
    )

    print(
        f"  ByteTrack: "
        f"{summary_df['bytetrack_reaches_accident'].mean():.4f}"
    )

    print(
        f"  BoT-SORT : "
        f"{summary_df['botsort_reaches_accident'].mean():.4f}"
    )

    print()
    print(
        f"Saved details: {DETAIL_OUTPUT}"
    )

    print(
        f"Saved summary: {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()