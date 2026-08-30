from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    PROJECT_ROOT,
    STAGE1_MODEL,
    DLC_STAGE1_RAW,
    STAGE2_MODEL,
    STAGE2_RAW,
    STAGE3_MODEL,
    STAGE3_RAW,
)
from src.inference import predict_stage1, predict_stage2, predict_stage3

SMOKE_DIR = PROJECT_ROOT / "sample_evaluation_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

EXPECTED_COLUMNS = {
    "stage1": ["ID", "answer"],
    "stage2": ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    "stage3": ["ID", "sample_index", "accel_label", "steer_label"],
}


def _copy_public_samples() -> Path:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    (SMOKE_DIR / "stage1" / "videos").mkdir(parents=True)
    (SMOKE_DIR / "stage2" / "images").mkdir(parents=True)
    (SMOKE_DIR / "stage3" / "videos").mkdir(parents=True)

    for label, folder in [("O", "original"), ("R", "rerecorded")]:
        source_dir = DLC_DLC_STAGE1_RAW / folder
        for index, path in enumerate(sorted(source_dir.glob("*")), 1):
            target = SMOKE_DIR / "stage1" / "videos" / f"SAMPLE_S1_{label}_{index:03d}{path.suffix.lower()}"
            shutil.copy2(path, target)

    for index, video_path in enumerate(sorted((STAGE2_RAW / "videos").glob("*")), 1):
        frame_dir = SMOKE_DIR / "stage2" / "images" / f"SAMPLE_S2_{index:03d}"
        frame_dir.mkdir()
        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"frame_{frame_index:06d}.jpg"), image)
            frame_index += 1
        capture.release()

    for path in sorted((STAGE3_RAW / "videos").glob("*")):
        shutil.copy2(path, SMOKE_DIR / "stage3" / "videos" / path.name)
    return SMOKE_DIR


def _check_columns(name, frame):
    expected = EXPECTED_COLUMNS[name]
    actual = list(frame.columns)
    if actual != expected:
        raise AssertionError(f"{name} columns mismatch: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke inference for all three stages.")
    parser.add_argument("--data-root", type=Path, default=SMOKE_DIR)
    parser.add_argument("--create-sample", action="store_true", help="Create sample_evaluation_data from data/ first.")
    parser.add_argument("--save-csv", action="store_true", help="Save predictions under output/.")
    args = parser.parse_args()

    data_root = _copy_public_samples() if args.create_sample else args.data_root
    predictions = {
        "stage1": predict_stage1(data_root / "stage1", STAGE1_MODEL),
        "stage2": predict_stage2(data_root / "stage2", STAGE2_MODEL),
        "stage3": predict_stage3(data_root / "stage3", STAGE3_MODEL),
    }
    for stage, frame in predictions.items():
        _check_columns(stage, frame)
        print(f"{stage}: {len(frame):,} rows")
        if args.save_csv:
            OUTPUT_DIR.mkdir(exist_ok=True)
            frame.to_csv(OUTPUT_DIR / f"{stage}_submission.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
