from __future__ import annotations

import torch

from src.config import DEVICE
from src.train.stage3 import _loss, _official_task_metrics


def test_official_metrics_exclude_stopped_frames_from_steer():
    metrics = _official_task_metrics(
        accel_pred=[2, 3, 0, 1],
        accel_target=[2, 3, 0, 1],
        steer_pred=[0, 2, 1, 2],
        steer_target=[0, 0, 1, 2],
    )

    assert metrics["accel"]["accuracy"] == 1.0
    assert metrics["steer"]["accuracy"] == 1.0
    assert metrics["steer"]["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_steer_loss_is_zero_for_all_stopped_batch():
    accel = torch.zeros(2, 4, device=DEVICE, requires_grad=True)
    steer = torch.randn(2, 3, device=DEVICE, requires_grad=True)
    batch = {
        "accel_label": torch.tensor([3, 3]),
        "steer_label": torch.tensor([0, 2]),
    }

    total, _, steer_loss = _loss(accel, steer, batch)
    total.backward()

    assert steer_loss.item() == 0.0
    assert torch.count_nonzero(steer.grad).item() == 0
