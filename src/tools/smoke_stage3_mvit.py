from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Stage3MViT
from src.train.stage3 import _datasets, _stage3_collate


def _as_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.shape[1] != 3 and video.shape[2] == 3:
        video = video.permute(0, 2, 1, 3, 4).contiguous()
    assert tuple(video.shape[1:]) == (3, 16, 224, 224), tuple(video.shape)
    return video


def _fixture_sample() -> dict:
    return {
        "video": torch.randn(3, 16, 224, 224),
        "accel_label": 1,
        "steer_label": 1,
    }


def _dataset_sample() -> tuple[dict, str]:
    try:
        train, _, _ = _datasets()
        return train[0], "real"
    except FileNotFoundError:
        return _fixture_sample(), "fixture"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage3MViT(pretrained=False).to(device)

    dummy = torch.randn(2, 3, 16, 224, 224, device=device)
    accel, steer = model(dummy)
    assert tuple(accel.shape) == (2, 4), tuple(accel.shape)
    assert tuple(steer.shape) == (2, 3), tuple(steer.shape)
    print("model forward: ok", tuple(accel.shape), tuple(steer.shape))

    sample, source = _dataset_sample()
    clip = _as_bcthw(sample["video"])
    print(f"dataset sample: ok ({source})", tuple(clip.shape))

    batch = _stage3_collate([sample])
    video = _as_bcthw(batch["video"]).to(device)
    target_accel = batch["accel_label"].to(device)
    target_steer = batch["steer_label"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    accel, steer = model(video)
    loss = nn.functional.cross_entropy(accel, target_accel) + nn.functional.cross_entropy(steer, target_steer)
    opt.zero_grad(); loss.backward(); opt.step()
    print("training batch: ok", float(loss.detach().cpu()))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stage3_mvit_smoke.pt"
        torch.save({"model": model.state_dict(), "arch": "mvit_v2_s", "model_config": model.model_config()}, path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        reloaded = Stage3MViT(pretrained=False).to(device)
        reloaded.load_state_dict(checkpoint["model"], strict=True)
        reloaded.eval()
        with torch.inference_mode():
            accel, steer = reloaded(dummy[:1])
        assert tuple(accel.shape) == (1, 4), tuple(accel.shape)
        assert tuple(steer.shape) == (1, 3), tuple(steer.shape)
    print("checkpoint reload: ok")
    print("stage3 mvit smoke passed")


if __name__ == "__main__":
    main()
