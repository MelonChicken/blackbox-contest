from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import torch
from transformers import VideoMAEConfig

from src.datasets.stage2_dataset import Stage2Dataset
from src.models.stage2_videomae import Stage2VideoMAE
from src.train.stage2 import build_optimizer, compute_stage2_loss


def _tiny_config() -> dict:
    return {
        "backbone_variant": "small",
        "num_frames": 4,
        "image_size": 32,
        "direction_classes": 2,
        "avoidance_classes": 2,
        "dropout": 0.0,
        "hf_config": VideoMAEConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            num_frames=4,
            tubelet_size=2,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            qkv_bias=True,
        ).to_dict(),
    }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "video": torch.randn(4, 4, 3, 32, 32),
        "sampled_indices": torch.arange(4).repeat(4, 1),
        "collision_frame": torch.tensor([1, 1, 2, 2]),
        "entry_frame": torch.tensor([0, -1, 1, -1]),
        "collision_index": torch.tensor([1, 1, 2, 2]),
        "entry_index": torch.tensor([0, -1, 1, -1]),
        "direction": torch.tensor([-1, 1, 0, -1]),
        "avoidance": torch.tensor([-1, -1, -1, -1]),
    }


def _has_grad(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters())


def _run(tasks: tuple[str, ...]) -> dict[str, float]:
    model = Stage2VideoMAE.from_config(_tiny_config(), use_pretrained=False)
    outputs = model(_batch()["video"])
    loss, metrics = compute_stage2_loss(outputs, _batch(), active_tasks=tasks)
    assert torch.isfinite(loss)
    loss.backward()
    assert _has_grad(model.backbone)
    assert _has_grad(model.collision_head)
    assert _has_grad(model.entry_head) == ("entry" in tasks and metrics.get("entry_supervised_count", 0) > 0)
    assert _has_grad(model.direction_head) == ("direction" in tasks and metrics.get("direction_supervised_count", 0) > 0)
    assert not _has_grad(model.avoidance_head)
    build_optimizer(model)
    return metrics


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (48, 48))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for _ in range(5):
        writer.write(torch.zeros(48, 48, 3, dtype=torch.uint8).numpy())
    writer.release()


def _dataset_missing_label_check() -> None:
    with tempfile.TemporaryDirectory(prefix="stage2_multitask_smoke_") as tmp:
        root = Path(tmp)
        video = root / "sample.avi"
        _write_video(video)
        manifest = root / "manifest.csv"
        pd.DataFrame([{"video_id": "sample", "video_path": str(video), "collision_frame": 2, "entry_frame": -1, "direction": -1, "avoidance": -1}]).to_csv(manifest, index=False)
        item = Stage2Dataset(manifest, num_frames=4, image_size=32)[0]
        assert int(item["entry_frame"]) == -1
        assert int(item["entry_index"]) == -1
        assert int(item["direction"]) == -1


def main() -> None:
    _dataset_missing_label_check()
    cd = _run(("collision", "direction"))
    ce = _run(("collision", "entry"))
    ced = _run(("collision", "entry", "direction"))

    all_missing = _batch()
    all_missing["entry_index"] = torch.full((4,), -1)
    all_missing["entry_frame"] = torch.full((4,), -1)
    model = Stage2VideoMAE.from_config(_tiny_config(), use_pretrained=False)
    loss, metrics = compute_stage2_loss(model(all_missing["video"]), all_missing, active_tasks=("collision", "entry"))
    assert metrics["entry_supervised_count"] == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert not _has_grad(model.entry_head)

    print("Stage2 multitask smoke passed")
    print("collision+direction", cd)
    print("collision+entry", ce)
    print("collision+entry+direction", ced)
    print("mixed batch: entry-only, direction-only, both, neither passed")


if __name__ == "__main__":
    main()
