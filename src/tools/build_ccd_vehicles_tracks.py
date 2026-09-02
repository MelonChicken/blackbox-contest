from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.config import CCD_STAGE2_BOTSORT_TRACKS, CCD_STAGE2_MANIFEST

# ============================================================
# Configuration
# ============================================================



MANIFEST_PATH = CCD_STAGE2_MANIFEST / 'ego_candidates.csv'


OUTPUT_DIR = CCD_STAGE2_BOTSORT_TRACKS


# 우선 작은 모델로 pipeline 검증
YOLO_MODEL = "yolo11m.pt"

# COCO vehicle classes
VEHICLE_NAMES = {
    "car",
    "bus",
    "truck",
}

CONF_THRESHOLD = 0.25

# 처음에는 전체 801개가 아니라
# 몇 개만 테스트할 수 있도록 지원
VIDEO_LIMIT: int | None = None


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


def get_vehicle_class_ids(model: YOLO) -> set[int]:
    """
    Resolve COCO class IDs dynamically instead of hardcoding
    2=car, 5=bus, 7=truck.
    """

    names = model.names

    return {
        class_id
        for class_id, class_name in names.items()
        if class_name in VEHICLE_NAMES
    }


# ============================================================
# Tracking
# ============================================================

def track_video(
    model: YOLO,
    video_id: str,
    video_path: Path,
    accident_start_frame: int,
    vehicle_class_ids: set[int],
) -> pd.DataFrame:
    """
    Run YOLO + ByteTrack on one CCD video.

    Returns one row per tracked vehicle per frame.
    """

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
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    rows: list[dict] = []

    frame_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            # tracker="bytetrack.yaml"
            tracker="botsort.yaml",
            classes=sorted(vehicle_class_ids),
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        if not results:
            frame_idx += 1
            continue

        result = results[0]
        boxes = result.boxes

        # 차량이 하나도 검출되지 않은 frame
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        # ByteTrack에서 ID가 아직 할당되지 않은 경우
        if boxes.id is None:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )
        confidences = boxes.conf.cpu().numpy()

        track_ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        height, width = frame.shape[:2]

        for bbox, cls_id, conf, track_id in zip(
            xyxy,
            class_ids,
            confidences,
            track_ids,
        ):
            x1, y1, x2, y2 = bbox.tolist()

            class_name = model.names[cls_id]

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # road-plane에서 차량 위치를 대략 대표할 point
            bottom_x = center_x
            bottom_y = y2

            rows.append(
                {
                    "video_id": video_id,
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),

                    "cx": float(center_x),
                    "cy": float(center_y),

                    "bottom_x": float(bottom_x),
                    "bottom_y": float(bottom_y),

                    "width": float(bbox_width),
                    "height": float(bbox_height),
                    "area": float(bbox_area),

                    # 영상 해상도가 달라도 비교할 수 있도록
                    # normalized coordinate도 같이 저장
                    "cx_norm": float(center_x / width),
                    "cy_norm": float(center_y / height),

                    "bottom_x_norm": float(
                        bottom_x / width
                    ),
                    "bottom_y_norm": float(
                        bottom_y / height
                    ),

                    "area_norm": float(
                        bbox_area / (width * height)
                    ),

                    "accident_start_frame": (
                        accident_start_frame
                    ),

                    "frames_to_accident": (
                        frame_idx
                        - accident_start_frame
                    ),

                    "fps": float(fps),
                    "video_frame_count": frame_count,
                }
            )

        frame_idx += 1

    cap.release()

    return pd.DataFrame(rows)


# ============================================================
# Reporting
# ============================================================

def print_track_summary(
    video_id: str,
    tracks: pd.DataFrame,
) -> None:
    if tracks.empty:
        print(
            f"[{video_id}] "
            f"No vehicle tracks detected"
        )
        return

    unique_tracks = (
        tracks["track_id"]
        .nunique()
    )

    classes = (
        tracks["class_name"]
        .value_counts()
        .to_dict()
    )

    print(
        f"[{video_id}] "
        f"detections={len(tracks)}, "
        f"tracks={unique_tracks}, "
        f"classes={classes}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_manifest()

    if VIDEO_LIMIT is not None:
        df = df.head(VIDEO_LIMIT)

    print("=== CCD Vehicle Tracking ===")
    print(f"Videos: {len(df)}")
    print(f"Model: {YOLO_MODEL}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print()

    model = YOLO(YOLO_MODEL)

    vehicle_class_ids = get_vehicle_class_ids(
        model
    )

    print("Vehicle classes:")

    for class_id in sorted(
        vehicle_class_ids
    ):
        print(
            f"  {class_id}: "
            f"{model.names[class_id]}"
        )

    print()

    failed: list[str] = []

    for idx, row in df.iterrows():
        video_id = row["video_id"]

        video_path = Path(
            row["video_path"]
        )

        accident_start_frame = int(
            row["accident_start_frame"]
        )

        output_path = (
            OUTPUT_DIR
            / f"{video_id}.csv"
        )

        # 이미 처리된 영상은 skip
        if output_path.exists():
            print(
                f"[{idx + 1}/{len(df)}] "
                f"{video_id}: skipped"
            )
            continue

        print(
            f"[{idx + 1}/{len(df)}] "
            f"{video_id}"
        )

        try:
            tracks = track_video(
                model=model,
                video_id=video_id,
                video_path=video_path,
                accident_start_frame=(
                    accident_start_frame
                ),
                vehicle_class_ids=(
                    vehicle_class_ids
                ),
            )

            print_track_summary(
                video_id,
                tracks,
            )

            tracks.to_csv(
                output_path,
                index=False,
            )

        except Exception as exc:
            failed.append(video_id)

            print(
                f"  FAILED: {exc}"
            )

    print()
    print("=== Completed ===")
    print(
        f"Failed videos: {len(failed)}"
    )

    if failed:
        print(
            "Failed IDs:",
            ", ".join(failed[:20]),
        )


if __name__ == "__main__":
    main()
