from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, STAGE3_ACCEL_LABEL_MODE, STAGE3_ACCEL_THRESHOLD, STAGE3_DECEL_THRESHOLD, STAGE3_STEER_POSITIVE_IS, STAGE3_STEER_THRESHOLD_DEG, STAGE3_STOP_SPEED_THRESHOLD
from src.datasets.stage3_labels import ACCEL_NAMES, STEER_NAMES
from src.tools.stage3_comma_manifest import disagreement, label_counts, print_route_overlap, print_sample_count_audit, read_manifest


def _audit_split(split: str) -> None:
    df = read_manifest(split)
    print(f"[comma {split}]")
    print(f"samples: {len(df)}")
    print("accel distribution:", label_counts(df.accel_label, ACCEL_NAMES))
    print("steer distribution:", label_counts(df.steer_label, STEER_NAMES))
    agree, matrix, dist = disagreement(df, ACCEL_NAMES)
    print(f"window-regression agreement: {agree:.4f}" if agree == agree else "window-regression agreement: unavailable")
    print("accel disagreement confusion current(row) vs window_regression(col):", matrix)
    print("accel mode distributions:", dist)
    cols = [c for c in ["sample_index", "target_timestamp", "speed", "acceleration", "accel_label", "steering_angle", "steer_label", "route_id", "segment_id", "video_frame_index", "video_frame_timestamp", "alignment_error_sec"] if c in df.columns]
    out = PROJECT_ROOT / f"stage3_comma_{split}_label_audit_samples.csv"
    df[cols].head(20).to_csv(out, index=False)
    print(f"sample csv: {out}")


def main() -> None:
    print("accel source signal: processed_log/CAN/speed")
    print("speed: interpolated to 10Hz target timestamps")
    print("acceleration current: gradient(moving_average(speed, width=5), dt=0.1)")
    print(f"accel label mode: {STAGE3_ACCEL_LABEL_MODE}")
    print(f"accel threshold: +{STAGE3_ACCEL_THRESHOLD}, decel threshold: -{STAGE3_DECEL_THRESHOLD}")
    print(f"STOPPED speed < {STAGE3_STOP_SPEED_THRESHOLD}")
    print("steer source signal: processed_log/CAN/steering_angle")
    print(f"steer threshold deg: {STAGE3_STEER_THRESHOLD_DEG}")
    print(f"steer sign convention: positive -> {STAGE3_STEER_POSITIVE_IS}; LEFT=0 STRAIGHT=1 RIGHT=2")
    for split in ("train", "val"):
        try:
            _audit_split(split)
            print_sample_count_audit(split)
        except FileNotFoundError as exc:
            print(exc)
    try:
        print_route_overlap()
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()