from __future__ import annotations

import ast
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from src.config import CCD_STAGE2_RAW, CCD_STAGE2_PROCESSED, CCD_STAGE2_MANIFEST

# ============================================================
# Configuration
# ============================================================

CCD_ROOT = CCD_STAGE2_RAW

ANNOTATION_PATH = CCD_ROOT / "Crash-1500.txt"
VIDEO_DIR = CCD_ROOT / "videos"

OUTPUT_DIR = CCD_STAGE2_MANIFEST

ALL_MANIFEST_PATH = OUTPUT_DIR / "all.csv"
EGO_MANIFEST_PATH = OUTPUT_DIR / "ego_candidates.csv"

EXPECTED_NUM_FRAMES = 50
EXPECTED_FPS = 10.0


# ============================================================
# Parsing
# ============================================================

def parse_annotation_line(line: str) -> dict:
    """
    Parse one line from Crash-1500.txt.

    Format:
        vidname,
        [50 binary frame labels],
        startframe,
        youtubeID,
        timing,
        weather,
        egoinvolve
    """

    line = line.strip()

    if not line:
        raise ValueError("Empty annotation line")

    # binlabels 내부에도 comma가 있으므로 일반 split(",") 사용 불가.
    label_start = line.find("[")
    label_end = line.find("]")

    if label_start == -1 or label_end == -1:
        raise ValueError(f"Could not locate binlabels: {line[:100]}")

    vidname = line[:label_start].rstrip(",")

    binlabels_text = line[label_start:label_end + 1]
    remainder = line[label_end + 1:].lstrip(",")

    binlabels = ast.literal_eval(binlabels_text)

    fields = next(csv.reader([remainder]))

    if len(fields) != 5:
        raise ValueError(
            f"Expected 5 fields after binlabels, got {len(fields)}: {fields}"
        )

    startframe, youtube_id, timing, weather, ego_involve = fields

    if len(binlabels) != EXPECTED_NUM_FRAMES:
        raise ValueError(
            f"{vidname}: expected {EXPECTED_NUM_FRAMES} labels, "
            f"got {len(binlabels)}"
        )

    if any(label not in (0, 1) for label in binlabels):
        raise ValueError(
            f"{vidname}: binlabels contains values other than 0/1"
        )

    positive_indices = [
        idx
        for idx, label in enumerate(binlabels)
        if label == 1
    ]

    if positive_indices:
        accident_start_frame = positive_indices[0]
        accident_end_frame = positive_indices[-1]
        num_accident_frames = len(positive_indices)
    else:
        accident_start_frame = -1
        accident_end_frame = -1
        num_accident_frames = 0

    return {
        "video_id": vidname,
        "source_id": youtube_id,
        "source_start_frame": int(startframe),
        "timing": timing,
        "weather": weather,
        "ego_involved": ego_involve.strip().lower() == "yes",
        "accident_start_frame": accident_start_frame,
        "accident_end_frame": accident_end_frame,
        "num_accident_frames": num_accident_frames,
        "total_frames": EXPECTED_NUM_FRAMES,
        "fps": EXPECTED_FPS,
    }


# ============================================================
# Validation
# ============================================================

def validate_temporal_labels(
    video_id: str,
    accident_start_frame: int,
    accident_end_frame: int,
    num_accident_frames: int,
) -> None:
    """
    Basic consistency checks for parsed temporal labels.
    """

    if accident_start_frame < 0:
        raise ValueError(
            f"{video_id}: crash video has no positive frame"
        )

    if accident_end_frame < accident_start_frame:
        raise ValueError(
            f"{video_id}: accident_end_frame < accident_start_frame"
        )

    expected_positive_count = (
        accident_end_frame - accident_start_frame + 1
    )

    if expected_positive_count != num_accident_frames:
        raise ValueError(
            f"{video_id}: non-contiguous accident labels detected. "
            f"start={accident_start_frame}, "
            f"end={accident_end_frame}, "
            f"positive_count={num_accident_frames}"
        )


# ============================================================
# Manifest construction
# ============================================================

def build_manifest() -> pd.DataFrame:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_PATH}"
        )

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}"
        )

    rows = []

    with ANNOTATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = parse_annotation_line(line)

                validate_temporal_labels(
                    video_id=row["video_id"],
                    accident_start_frame=row["accident_start_frame"],
                    accident_end_frame=row["accident_end_frame"],
                    num_accident_frames=row["num_accident_frames"],
                )

                video_path = (
                    VIDEO_DIR / f"{row['video_id']}.mp4"
                )

                row["video_path"] = str(video_path)
                row["video_exists"] = video_path.exists()

                rows.append(row)

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse line {line_number}"
                ) from exc

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No CCD annotations were parsed")

    if df["video_id"].duplicated().any():
        duplicated = df.loc[
            df["video_id"].duplicated(keep=False),
            "video_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate video IDs detected: {duplicated[:10]}"
        )

    return df


# ============================================================
# Reporting
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    print("=== CCD Stage 2 Manifest ===")

    print(f"Total annotations: {len(df)}")

    print(
        f"Videos found: "
        f"{int(df['video_exists'].sum())}/{len(df)}"
    )

    print(
        f"Ego involved: "
        f"{int(df['ego_involved'].sum())}"
    )

    print(
        f"Ego not involved: "
        f"{int((~df['ego_involved']).sum())}"
    )

    print()

    print("Timing:")
    for key, value in Counter(df["timing"]).items():
        print(f"  {key}: {value}")

    print()

    print("Weather:")
    for key, value in Counter(df["weather"]).items():
        print(f"  {key}: {value}")

    print()

    print("Accident start frame:")
    print(
        df["accident_start_frame"]
        .describe()
        .to_string()
    )

    print()

    source_counts = df["source_id"].value_counts()

    print(
        f"Unique source videos: {df['source_id'].nunique()}"
    )

    print(
        f"Sources producing >1 clip: "
        f"{int((source_counts > 1).sum())}"
    )

    print(
        f"Maximum clips from one source: "
        f"{int(source_counts.max())}"
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

    missing = df.loc[
        ~df["video_exists"],
        ["video_id", "video_path"],
    ]

    if not missing.empty:
        print()
        print("Missing video examples:")
        print(
            missing
            .head(10)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"{len(missing)} video files are missing"
        )

    # Full official CCD annotation manifest.
    df.to_csv(
        ALL_MANIFEST_PATH,
        index=False,
    )

    # DACON Stage 2에 우선 사용할 ego-involved 후보.
    ego_df = (
        df[df["ego_involved"]]
        .copy()
        .reset_index(drop=True)
    )

    ego_df.to_csv(
        EGO_MANIFEST_PATH,
        index=False,
    )

    print()
    print(f"Saved: {ALL_MANIFEST_PATH}")
    print(f"Saved: {EGO_MANIFEST_PATH}")

    print(
        f"Ego candidates: "
        f"{len(ego_df)}/{len(df)}"
    )


if __name__ == "__main__":
    main()