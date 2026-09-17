from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.stage3_vjepa2 import Stage3VJEPA2Frozen


class TinyVJEPA(nn.Module):
    embed_dim = 32

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def _model(probe: str, input_size: int) -> Stage3VJEPA2Frozen:
    return Stage3VJEPA2Frozen(
        backbone=TinyVJEPA(),
        probe_type=probe,
        input_size=input_size,
        num_frames=16,
        probe_dim=32,
        probe_heads=4,
        dropout=0.0,
    )


def _batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, 3, 16, 224, 224)
    return x, torch.tensor([0, 3])[:batch_size], torch.tensor([1, 2])[:batch_size]


def _check_forward(model: Stage3VJEPA2Frozen, x: torch.Tensor, label: str) -> None:
    accel, steer = model(x)
    assert tuple(accel.shape) == (x.size(0), 4), tuple(accel.shape)
    assert tuple(steer.shape) == (x.size(0), 3), tuple(steer.shape)
    assert torch.isfinite(accel).all() and torch.isfinite(steer).all()
    print(f"PASS {label}: accel={tuple(accel.shape)} steer={tuple(steer.shape)} tokens={tuple(model.tokens(x).shape)}")


def _check_backward(model: Stage3VJEPA2Frozen, x: torch.Tensor, accel_y: torch.Tensor, steer_y: torch.Tensor) -> None:
    opt_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert opt_params and not any(n.startswith("backbone.") for n, _ in opt_params)
    opt = torch.optim.AdamW((p for _, p in opt_params), lr=1e-4, weight_decay=0.05)
    accel, steer = model(x)
    loss = F.cross_entropy(accel, accel_y) + F.cross_entropy(steer, steer_y)
    assert torch.isfinite(loss)
    opt.zero_grad(); loss.backward(); opt.step()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(p.grad is not None for n, p in model.named_parameters() if not n.startswith("backbone."))
    print(f"PASS backward: loss={float(loss.detach()):.6f} trainable={sum(p.numel() for _, p in opt_params)}")


def _check_reload(model: Stage3VJEPA2Frozen, x: torch.Tensor) -> None:
    model.eval()
    with torch.inference_mode():
        before = model(x)[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vjepa_smoke.pt"
        torch.save({"arch": "vjepa2_1_vitb_frozen", "model": model.state_dict(), "model_config": model.model_config()}, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        reloaded = _model(ckpt["model_config"]["probe_type"], ckpt["model_config"]["input_size"])
        reloaded.load_state_dict(ckpt["model"], strict=True)
        reloaded.eval()
        with torch.inference_mode():
            after = reloaded(x)[0]
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    print("PASS checkpoint reload: logits match")


def _bench(model: Stage3VJEPA2Frozen, x: torch.Tensor, device: torch.device) -> None:
    model = model.to(device).eval()
    x = x.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(3):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak = 0.0
    latency_ms = (time.perf_counter() - start) * 1000 / (3 * x.size(0))
    print(f"PASS bench {device.type}: peak_mb={peak:.1f} latency_ms_per_clip={latency_ms:.2f}")


def _official(args) -> None:
    if not args.repo or not args.checkpoint:
        print("SKIP official checkpoint load: pass --repo and --checkpoint")
        return
    model = Stage3VJEPA2Frozen(repo_dir=args.repo, checkpoint=args.checkpoint, input_size=args.input_size, probe_type="mean")
    x, _, _ = _batch(1)
    _check_forward(model, x, "official checkpoint forward")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Stage3 V-JEPA2 frozen probes.")
    parser.add_argument("--repo", default="", help="local facebookresearch/vjepa2 checkout")
    parser.add_argument("--checkpoint", default="", help="local vjepa2_1_vitb_dist_vitG_384.pt")
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    _official(args)
    x, accel_y, steer_y = _batch()
    mean224 = _model("mean", 224)
    _check_forward(mean224, x, "224 input / mean probe")
    mean384 = _model("mean", 384)
    _check_forward(mean384, x, "224 cache resized to 384 / mean probe")
    cross = _model("cross_attention", 224)
    _check_forward(cross, x, "224 input / cross_attention probe")
    _check_backward(cross, x, accel_y, steer_y)
    _check_reload(cross, x)
    _bench(_model("mean", 224), x[:1], torch.device("cpu"))
    if torch.cuda.is_available():
        _bench(_model("mean", 224), x[:1], torch.device("cuda"))
        _bench(_model("mean", 384), x[:1], torch.device("cuda"))
    else:
        print("SKIP CUDA AMP/VRAM benchmark: CUDA unavailable")
    print("stage3 vjepa2 smoke passed")


if __name__ == "__main__":
    main()