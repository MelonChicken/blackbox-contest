from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.datasets.stage3_sampling import build_centered_clip_indices

def _submission_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    path = root / "submission" / "inference.py"
    spec = importlib.util.spec_from_file_location("submission_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_centered_clip_indices_n20_len16():
    expected = {
        0: [0] * 9 + list(range(1, 8)),
        1: [0] * 8 + list(range(1, 9)),
        8: list(range(16)),
        12: list(range(4, 20)),
        19: list(range(11, 20)) + [19] * 7,
    }
    for center, indices in expected.items():
        assert build_centered_clip_indices(center, 20, 16) == indices


def test_submission_sampler_matches_src():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "submission"))
    path = root / "submission" / "inference.py"
    spec = importlib.util.spec_from_file_location("submission_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for center in range(20):
        assert module.build_centered_clip_indices(center, 20, 16) == build_centered_clip_indices(center, 20, 16)


def test_stage3_submission_shape_contract_without_model():
    ids = ["a", "b"]
    rows = [
        {"ID": video_id, "sample_index": i, "accel_label": "CONSTANT", "steer_label": "STRAIGHT"}
        for video_id, n in zip(ids, [3, 2])
        for i in range(n)
    ]
    df = pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
    assert list(df.columns) == ["ID", "sample_index", "accel_label", "steer_label"]
    assert len(df) == 5
    assert df.groupby("ID").sample_index.apply(list).to_dict() == {"a": [0, 1, 2], "b": [0, 1]}
    assert set(df.accel_label) <= {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}
    assert set(df.steer_label) <= {"LEFT", "STRAIGHT", "RIGHT"}



def test_stage3_decoded_video_rows_are_per_frame_and_per_id():
    module = _submission_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = {"a": 5, "b": 3}
        for name, n in specs.items():
            path = root / f"{name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            assert writer.isOpened()
            for i in range(n):
                frame = np.full((24, 32, 3), i * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

        rows = []
        for name, n in specs.items():
            frames = module._stage3_frames(root / f"{name}.mp4")
            assert len(frames) == n
            rows.extend(
                {"ID": name, "sample_index": i, "accel_label": "CONSTANT", "steer_label": "STRAIGHT"}
                for i in range(len(frames))
            )

    df = pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
    assert len(df) == sum(specs.values())
    assert df.groupby("ID").sample_index.apply(list).to_dict() == {"a": [0, 1, 2, 3, 4], "b": [0, 1, 2]}
    assert list(df.columns) == ["ID", "sample_index", "accel_label", "steer_label"]
    assert set(df.accel_label) <= {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}
    assert set(df.steer_label) <= {"LEFT", "STRAIGHT", "RIGHT"}
