from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.datasets.stage3_sampling import (
    build_centered_clip_indices,
    build_timestamp_nearest_clip,
    json_dumps_compact,
    nearest_frame_indices_for_timestamps,
    parse_clip_frame_indices,
    sampling_metadata,
)

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

def test_timestamp_nearest_clip_20hz_to_10hz():
    frame_times = np.arange(0.0, 2.0001, 0.05)
    clip = build_timestamp_nearest_clip(frame_times, 1.0, 10.0, 8, 7, max_alignment_error_sec=0.026)
    assert clip["clip_frame_indices"] == [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    assert np.allclose(clip["clip_target_timestamps"], np.arange(0.2, 1.7001, 0.1))
    assert max(clip["clip_alignment_errors_sec"]) < 1e-9


def test_timestamp_nearest_clip_edge_clamp_start_and_end():
    frame_times = np.arange(0.0, 2.0001, 0.05)
    start = build_timestamp_nearest_clip(frame_times, 0.1, 10.0, 8, 7)
    end = build_timestamp_nearest_clip(frame_times, 1.9, 10.0, 8, 7)
    assert start["clip_frame_indices"][:8] == [0] * 8
    assert len(start["clip_frame_indices"]) == 16
    assert end["clip_frame_indices"][-6:] == [40] * 6
    assert len(end["clip_frame_indices"]) == 16


def test_irregular_frame_times_nearest_and_alignment_error():
    indices, errors = nearest_frame_indices_for_timestamps([0.0, 0.04, 0.11, 0.21], [0.1])
    assert indices == [2]
    assert abs(errors[0] - 0.01) < 1e-9


def test_alignment_error_limit_raises():
    try:
        nearest_frame_indices_for_timestamps([0.0, 1.0], [0.4], max_alignment_error_sec=0.1)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("expected alignment error failure")


def test_clip_indices_json_round_trip():
    values = [4, 6, 8, 10]
    encoded = json_dumps_compact(values)
    assert parse_clip_frame_indices(encoded, 4) == values
    try:
        parse_clip_frame_indices("[1, 2.5]", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected non-int JSON value failure")


def test_row_thinning_does_not_change_internal_clip_indices():
    rows = [
        {"sample_index": i, "clip_frame_indices": json_dumps_compact([i, i + 2])}
        for i in range(4)
    ]
    thinned = pd.DataFrame(rows).iloc[::2].reset_index(drop=True)
    assert [parse_clip_frame_indices(v, 2) for v in thinned.clip_frame_indices] == [[0, 2], [2, 4]]


def test_cache_needed_frames_union_uses_all_clip_indices():
    from src.tools.cache_comma2k19_stage3_frames import _needed_frames

    df = pd.DataFrame({
        "clip_frame_indices": [json_dumps_compact([4, 6, 8]), json_dumps_compact([6, 10, 12])]
    })
    old = __import__("src.config", fromlist=["STAGE3_NUM_FRAMES"]).STAGE3_NUM_FRAMES
    assert old == 16
    try:
        _needed_frames(df)
    except ValueError:
        pass
    df = pd.DataFrame({
        "clip_frame_indices": [json_dumps_compact(list(range(16))), json_dumps_compact(list(range(8, 24)))]
    })
    assert _needed_frames(df).tolist() == list(range(24))


def test_dataset_uses_manifest_clip_indices_not_center_window():
    import json
    import src.datasets.comma2k19_stage3 as ds_mod

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "train.csv"
        meta = sampling_metadata(
            sampling_hz=10.0,
            sampling_policy="centered_timestamp_nearest",
            num_frames=16,
            past_frames=8,
            future_frames=7,
            boundary_policy="edge_clamp",
            output_hz=10.0,
        )
        (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        wanted = list(range(0, 32, 2))
        pd.DataFrame([{
            "video_path": "fake.hevc",
            "target_timestamp": 1.0,
            "video_frame_index": 20,
            "clip_frame_indices": json_dumps_compact(wanted),
            "accel_label": 2,
            "steer_label": 1,
        }]).to_csv(manifest, index=False)

        seen = []
        old_fn = ds_mod.stage3_video_clip_indices
        ds_mod.stage3_video_clip_indices = lambda path, indices: seen.append(list(indices)) or torch.zeros(3, 16, 224, 224)
        try:
            item = ds_mod.Comma2k19Stage3Dataset(manifest, root=root, cache_root=None)[0]
        finally:
            ds_mod.stage3_video_clip_indices = old_fn
        assert seen == [wanted]
        assert item["video"].shape == (3, 16, 224, 224)
