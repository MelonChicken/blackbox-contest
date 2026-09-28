from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import torch

from src.tools.build_ccd_stage2_manifest import _apply_human_labels, _balanced_group_split, validate_stage2_manifest
from src.tools.stage2_human_labels import load_stage2_human_labels
from src.train.stage2 import _frame_metrics, _masked_cross_entropy, balanced_sample_weights, classification_loss_weights, selection_metric


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


def test_balanced_sampling_boosts_auxiliary_and_minority_rows() -> None:
    rows = pd.DataFrame(
        [
            {"entry_frame": -1, "direction": -1, "avoidance": -1},
            {"entry_frame": 10, "direction": 0, "avoidance": 1},
            {"entry_frame": -1, "direction": 1, "avoidance": 1},
            {"entry_frame": -1, "direction": 1, "avoidance": 1},
        ]
    )
    weights = balanced_sample_weights(rows)
    assert weights[1] > weights[2] > weights[0]
    class_weights = classification_loss_weights(rows)
    assert class_weights["direction"][0] > class_weights["direction"][1]


def test_pseudo_sample_weight_reduces_loss_contribution() -> None:
    logits = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([0])
    human = _masked_cross_entropy(logits, target, sample_weight=torch.tensor([1.0]))
    pseudo = _masked_cross_entropy(logits, target, sample_weight=torch.tensor([0.5]))
    assert torch.isclose(pseudo, human * 0.5)


def test_group_split_has_no_source_leakage_and_keeps_sparse_labels() -> None:
    rows = []
    for group in range(20):
        rows.append(
            {
                "video_id": f"v{group:02d}",
                "video_path": f"v{group:02d}.mp4",
                "source_id": f"s{group:02d}",
                "collision_frame": 20,
                "entry_frame": 10 if group % 3 == 0 else -1,
                "direction": group % 2 if group % 3 == 0 else -1,
                "avoidance": group % 2 if group % 4 == 0 else -1,
                "dataset": "ccd",
                "n_frames": 50,
            }
        )
    frame = pd.DataFrame(rows)
    validate_stage2_manifest(frame)
    train, val = _balanced_group_split(frame)
    assert not (set(train.source_id) & set(val.source_id))
    assert (val.entry_frame >= 0).any()
    assert (val.direction >= 0).any()
    assert (val.avoidance >= 0).any()
