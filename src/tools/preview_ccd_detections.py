from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.config import CCD_STAGE2_MANIFEST


CCD_MANIFEST = CCD_STAGE2_MANIFEST / 'ego_candidates.csv'

OUTPUT_DIR = Path(
    "data/stage2/CCD-1500/preview/yolo"
)

MODEL_NAME = "yolo11n.pt"

NUM_VIDEOS = 10

VEHICLE_NAMES = {
    "car",
    "bus",
    "truck",
}


def process_video(
    model: YOLO,
    video_path: Path,
    output_path: Path,
) -> None:
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

        result = model(
            frame,
            verbose=False,
        )[0]

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            class_name = result.names[cls_id]

            if class_name not in VEHICLE_NAMES:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{class_name} "
                f"{conf:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        CCD_MANIFEST,
        dtype={
            "video_id": str,
            "source_id": str,
        },
    )

    # CSV에서 000001이 1로 변환되는 것을 방지
    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.zfill(6)
    )

    model = YOLO(MODEL_NAME)

    sample_df = df.head(NUM_VIDEOS)

    for _, row in sample_df.iterrows():
        video_id = row["video_id"]

        video_path = Path(
            row["video_path"]
        )

        output_path = (
            OUTPUT_DIR
            / f"{video_id}_yolo.mp4"
        )

        print(
            f"[YOLO] {video_id}"
        )

        process_video(
            model=model,
            video_path=video_path,
            output_path=output_path,
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()