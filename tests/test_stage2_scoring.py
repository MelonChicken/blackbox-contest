from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.tools.build_ccd_stage2_manifest import _apply_human_labels
from src.tools.stage2_human_labels import load_stage2_human_labels
from src.train.stage2 import _frame_metrics, selection_metric


def test_accuracy_at_point_three_seconds_uses_video_fps() -> None:
    metrics = _frame_metrics(
        pred_frames=[13, 14, 19],
        target_frames=[10, 10, 10],
        fps_values=[10.0, 10.0, 30.0],
        frame_counts=[50, 50, 150],
    )
    assert metrics["accuracy_at_0_3s"] == 2 / 3


def test_official_stage2_weighted_score() -> None:
    metrics = {
        "val_collision_accuracy_at_0_3s": 0.8,
        "val_entry_accuracy_at_0_3s": 0.6,
        "val_direction_macro_f1": 0.5,
        "val_avoidance_macro_f1": 0.4,
    }
    expected = 0.35 * 0.8 + 0.35 * 0.6 + 0.15 * 0.5 + 0.15 * 0.4
    assert abs(selection_metric(metrics) - expected) < 1e-12


def test_human_label_normalization_and_temporal_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="stage2_human_labels_") as tmp:
        root = Path(tmp)
        pd.DataFrame(
            [
                {
                    "tier": "1-HIGH",
                    "id": "000001",
                    "video": "raw/ccd/000001.mp4",
                    "NEW_collision_frame": 30,
                    "NEW_entry_frame": 20,
                    "NEW_entry_side": "LEFT",
                    "NEW_evasion_space": 1,
                    "n_frames": 50,
                },
                {
                    "tier": "3-LOW",
                    "id": "000002",
                    "video": "raw/ccd/000002.mp4",
                    "NEW_collision_frame": 20,
                    "NEW_entry_frame": 25,
                    "NEW_entry_side": "UNKNOWN",
                    "NEW_evasion_space": 0,
                    "n_frames": 50,
                },
            ]
        ).to_csv(root / "labels.csv", index=False)

        labels = load_stage2_human_labels(root)
        good = labels.loc[labels.video_id == "000001"].iloc[0]
        review = labels.loc[labels.video_id == "000002"].iloc[0]
        assert int(good.direction) == 0
        assert bool(good.entry_valid)
        assert int(review.entry_frame) == -1
        assert not bool(review.entry_valid)
        assert review.label_issue == "entry_after_collision"


def test_human_labels_override_pseudo_labels_and_append_aihub() -> None:
    base = pd.DataFrame(
        [
            {
                "video_id": "000001",
                "video_path": "ccd.mp4",
                "source_id": "source",
                "collision_frame": 29,
                "entry_frame": -1,
                "direction": 1,
                "avoidance": -1,
                "collision_source": "ccd_official",
                "entry_source": "missing",
                "direction_source": "pseudo",
                "avoidance_source": "missing",
                "collision_confidence": 1.0,
                "entry_confidence": 0.0,
                "direction_confidence": 0.4,
                "avoidance_confidence": 0.0,
                "overall_confidence": 0.4,
                "confidence_level": "medium",
            }
        ]
    )
    human = pd.DataFrame(
        [
            {
                "video_id": "000001",
                "dataset": "ccd",
                "video_path": "ccd.mp4",
                "source_id": "000001",
                "n_frames": 50,
                "fps": 10.0,
                "collision_frame": 30,
                "entry_frame": 20,
                "direction": 0,
                "avoidance": 1,
                "collision_valid": True,
                "entry_valid": True,
                "direction_valid": True,
                "avoidance_valid": True,
                "human_reviewed": True,
                "ambiguous": False,
                "tier": "1-HIGH",
                "lane_score": 0.8,
                "labeler": "tester",
                "notes": "",
                "label_issue": "",
            }
        ]
    )
    result = _apply_human_labels(base, human)
    row = result.iloc[0]
    assert int(row.collision_frame) == 30
    assert int(row.entry_frame) == 20
    assert int(row.direction) == 0
    assert int(row.avoidance) == 1
    assert row.direction_source == "human_manual"
