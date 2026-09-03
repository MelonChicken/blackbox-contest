from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEConfig

from src.datasets.stage2_dataset import Stage2Dataset
from src.inference.stage2 import run_stage2_clip
from src.models.stage2_videomae import Stage2VideoMAE
from src.train.stage2 import compute_stage2_loss, evaluate, save_stage2_checkpoint


def _write_video(path: Path, frames: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for i in range(frames):
            frame = torch.zeros(64, 96, 3, dtype=torch.uint8).numpy()
            frame[:, :, 0] = (i * 11) % 255
            frame[16:48, 20 + i % 20 : 44 + i % 20, 1] = 220
            writer.write(frame)
    finally:
        writer.release()


def _tiny_config() -> dict:
    hf_config = VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        qkv_bias=True,
    ).to_dict()
    return {
        "backbone_variant": "small",
        "num_frames": 16,
        "image_size": 224,
        "direction_classes": 2,
        "avoidance_classes": 2,
        "dropout": 0.1,
        "hf_config": hf_config,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage2_smoke_") as tmp:
        root = Path(tmp)
        video_a = root / "videos" / "sample_a.avi"
        video_b = root / "videos" / "sample_b.avi"
        _write_video(video_a)
        _write_video(video_b)
        manifest = root / "manifest.csv"
        pd.DataFrame(
            [
                {
                    "video_id": "sample_a",
                    "video_path": str(video_a),
                    "collision_frame": 8,
                    "entry_frame": 4,
                    "direction": 1,
                    "avoidance": 0,
                    "collision_source": "smoke",
                    "entry_source": "smoke",
                    "direction_source": "smoke",
                    "avoidance_source": "smoke",
                },
                {
                    "video_id": "sample_b",
                    "video_path": str(video_b),
                    "collision_frame": 9,
                    "entry_frame": -1,
                    "direction": -1,
                    "avoidance": -1,
                    "collision_source": "smoke",
                    "entry_source": "missing",
                    "direction_source": "missing",
                    "avoidance_source": "missing",
                },
            ]
        ).to_csv(manifest, index=False)

        dataset = Stage2Dataset(manifest, num_frames=16, image_size=224)
        assert len(dataset) == 2
        item = dataset[0]
        assert tuple(item["video"].shape) == (16, 3, 224, 224)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert tuple(batch["video"].shape) == (2, 16, 3, 224, 224)

        model = Stage2VideoMAE.from_config(_tiny_config(), use_pretrained=False)
        outputs = model(batch["video"])
        assert tuple(outputs["collision_logits"].shape) == (2, 16)
        assert tuple(outputs["entry_logits"].shape) == (2, 16)
        assert outputs["direction_logits"].shape[0] == 2
        assert outputs["avoidance_logits"].shape[0] == 2

        loss, metrics = compute_stage2_loss(outputs, batch, active_tasks=("collision", "entry"))
        loss.backward()
        assert model.collision_head.weight.grad is not None
        assert model.entry_head.weight.grad is not None
        inactive_loss, inactive_metrics = compute_stage2_loss(outputs, batch, active_tasks=("collision",))
        assert "entry_loss" not in inactive_metrics
        assert float(inactive_loss.detach()) != float(loss.detach())

        val_metrics = evaluate(model, loader, torch.device("cpu"), active_tasks=("collision", "entry"))
        assert "val_collision_mean_abs_original_frame_error" in val_metrics
        assert "val_entry_mean_abs_original_frame_error" in val_metrics
        assert "val_selection_metric" in val_metrics

        ckpt = root / "best_collision_entry.pt"
        save_stage2_checkpoint(ckpt, model, epoch=1, metrics=val_metrics, history=[], active_tasks=("collision", "entry"))
        code = (
            "import torch, sys; "
            "from src.train.stage2 import load_stage2_checkpoint; "
            "m=load_stage2_checkpoint(sys.argv[1], 'cpu'); "
            "print(len(m.state_dict()))"
        )
        subprocess.run([sys.executable, "-c", code, str(ckpt)], check=True)

        reloaded = Stage2VideoMAE.from_config(torch.load(ckpt, map_location="cpu", weights_only=False)["model_config"])
        reloaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        reloaded.eval()
        result = run_stage2_clip(item["video"], item["sampled_indices"].numpy(), reloaded, torch.device("cpu"))
        assert {"collision_frame", "entry_frame", "entry_side", "evasion_space"}.issubset(result)
        print("Stage2 smoke passed")
        print(f"loss={float(loss.detach()):.6f}")
        print(metrics)
        print(val_metrics)
        print(result)


if __name__ == "__main__":
    main()
