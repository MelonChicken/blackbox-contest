from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from src.config import CCD_STAGE2_MANIFEST

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MANIFEST_PATH = CCD_STAGE2_MANIFEST / 'ego_candidates.csv'


TRACK_DIR  = Path(
    "data/stage2/CCD-1500/tracks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage2"
    / "CCD-1500"
    / "preview"
    / "tracks"
)

# 직접 확인하고 싶은 video id를 넣으면 됨
VIDEO_IDS = [
    "000001",
    "000002",
    "000003",
    "000004",
]

# bbox confidence 표시 여부
SHOW_CONFIDENCE = True

# track trajectory를 몇 frame까지 이어 그릴지
TRAIL_LENGTH = 15


# ============================================================
# Utilities
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


def load_tracks(video_id: str) -> pd.DataFrame:
    track_path = TRACK_DIR / f"{video_id}.csv"

    if not track_path.exists():
        raise FileNotFoundError(
            f"Track file not found: {track_path}"
        )

    df = pd.read_csv(track_path)

    if df.empty:
        return df

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.zfill(6)
    )

    return df


# ============================================================
# Drawing
# ============================================================

def draw_track(
    frame,
    row,
) -> None:
    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    track_id = int(row["track_id"])
    class_name = str(row["class_name"])
    confidence = float(row["confidence"])

    # track마다 약간 다른 색을 deterministic하게 생성
    color = (
        int((track_id * 53) % 255),
        int((track_id * 97) % 255),
        int((track_id * 193) % 255),
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = f"{class_name} #{track_id}"

    if SHOW_CONFIDENCE:
        label += f" {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    # bottom-center point
    bottom_x = int(row["bottom_x"])
    bottom_y = int(row["bottom_y"])

    cv2.circle(
        frame,
        (bottom_x, bottom_y),
        5,
        color,
        -1,
    )


def draw_trajectory(
    frame,
    track_df: pd.DataFrame,
    frame_idx: int,
) -> None:
    if track_df.empty:
        return

    current_tracks = track_df[
        track_df["frame"] == frame_idx
    ]

    if current_tracks.empty:
        return

    for _, current in current_tracks.iterrows():
        track_id = int(current["track_id"])

        history = track_df[
            (track_df["track_id"] == track_id)
            & (track_df["frame"] <= frame_idx)
            & (
                track_df["frame"]
                >= frame_idx - TRAIL_LENGTH
            )
        ].sort_values("frame")

        if len(history) < 2:
            continue

        color = (
            int((track_id * 53) % 255),
            int((track_id * 97) % 255),
            int((track_id * 193) % 255),
        )

        points = []

        for _, row in history.iterrows():
            points.append(
                (
                    int(row["bottom_x"]),
                    int(row["bottom_y"]),
                )
            )

        for p1, p2 in zip(
            points[:-1],
            points[1:],
        ):
            cv2.line(
                frame,
                p1,
                p2,
                color,
                2,
            )


# ============================================================
# Preview
# ============================================================

def preview_video(
    video_id: str,
    video_path: Path,
    accident_start_frame: int,
    tracks: pd.DataFrame,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_id}_tracks.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        # --------------------------------------------
        # Draw trajectory first
        # --------------------------------------------

        draw_trajectory(
            frame,
            tracks,
            frame_idx,
        )

        # --------------------------------------------
        # Draw current detections
        # --------------------------------------------

        if not tracks.empty:
            current = tracks[
                tracks["frame"] == frame_idx
            ]

            for _, row in current.iterrows():
                draw_track(
                    frame,
                    row,
                )

        # --------------------------------------------
        # General frame information
        # --------------------------------------------

        cv2.putText(
            frame,
            f"Video: {video_id}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Frame: {frame_idx} "
                f"/ Accident: {accident_start_frame}"
            ),
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frames_to_accident = (
            frame_idx - accident_start_frame
        )

        cv2.putText(
            frame,
            f"Relative: {frames_to_accident:+d}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------
        # Accident frame marker
        # --------------------------------------------

        if frame_idx == accident_start_frame:
            cv2.rectangle(
                frame,
                (3, 3),
                (width - 4, height - 4),
                (0, 0, 255),
                8,
            )

            text = "ACCIDENT START"

            text_size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                3,
            )

            text_width = text_size[0]

            cv2.putText(
                frame,
                text,
                (
                    max(
                        10,
                        (width - text_width) // 2,
                    ),
                    130,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    print("=== CCD Track Preview ===")

    for video_id in VIDEO_IDS:
        video_id = str(video_id).zfill(6)

        rows = manifest[
            manifest["video_id"] == video_id
        ]

        if rows.empty:
            print(
                f"[{video_id}] "
                f"Not found in manifest"
            )
            continue

        row = rows.iloc[0]

        video_path = Path(
            row["video_path"]
        )

        accident_start_frame = int(
            row["accident_start_frame"]
        )

        try:
            tracks = load_tracks(video_id)

            print(
                f"[{video_id}] "
                f"rows={len(tracks)}, "
                f"tracks="
                f"{tracks['track_id'].nunique() if not tracks.empty else 0}"
            )

            preview_video(
                video_id=video_id,
                video_path=video_path,
                accident_start_frame=(
                    accident_start_frame
                ),
                tracks=tracks,
            )

        except Exception as exc:
            print(
                f"[{video_id}] FAILED: {exc}"
            )


if __name__ == "__main__":
    main()