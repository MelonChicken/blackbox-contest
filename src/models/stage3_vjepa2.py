from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.config import S3_MEAN, S3_STD

VJEPA2_IMAGENET_MEAN = (0.485, 0.456, 0.406)
VJEPA2_IMAGENET_STD = (0.229, 0.224, 0.225)


def _clean_vjepa_state_dict(state: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}


def _load_official_vjepa2_encoder(repo_dir: str | Path, checkpoint: str | Path | None, backbone_name: str) -> nn.Module:
    repo_dir = Path(repo_dir).expanduser()
    checkpoint = Path(checkpoint).expanduser() if checkpoint else None
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"V-JEPA2 repo not found for offline torch.hub load: {repo_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V-JEPA2 checkpoint not found: {checkpoint}")
    loaded = torch.hub.load(str(repo_dir), backbone_name, source="local", pretrained=False)
    encoder = loaded[0] if isinstance(loaded, tuple) else loaded
    if checkpoint is None:
        return encoder
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "ema_encoder" in payload:
        state = payload["ema_encoder"]
    elif "target_encoder" in payload:
        state = payload["target_encoder"]
    elif "encoder" in payload:
        state = payload["encoder"]
    else:
        state = payload
    missing, unexpected = encoder.load_state_dict(_clean_vjepa_state_dict(state), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected V-JEPA2 checkpoint keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"Missing V-JEPA2 checkpoint keys: {missing[:8]}")
    return encoder


class Stage3VJEPAMeanProbe(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, tokens: torch.Tensor):
        z = self.dropout(self.norm(tokens.mean(dim=1)))
        return self.accel(z), self.steer(z)


class Stage3VJEPACrossAttentionProbe(nn.Module):
    def __init__(self, dim: int, probe_dim: int, heads: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, probe_dim) if dim != probe_dim else nn.Identity()
        self.accel_query = nn.Parameter(torch.zeros(1, 1, probe_dim))
        self.steer_query = nn.Parameter(torch.zeros(1, 1, probe_dim))
        self.attn = nn.MultiheadAttention(probe_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(probe_dim)
        self.mlp = nn.Sequential(nn.Linear(probe_dim, probe_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(probe_dim * 2, probe_dim))
        self.dropout = nn.Dropout(dropout)
        self.accel = nn.Linear(probe_dim, 4)
        self.steer = nn.Linear(probe_dim, 3)
        nn.init.normal_(self.accel_query, std=0.02)
        nn.init.normal_(self.steer_query, std=0.02)

    def _pool(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = query.expand(tokens.size(0), -1, -1)
        attended, _ = self.attn(q, tokens, tokens, need_weights=False)
        z = self.norm(q + attended).squeeze(1)
        return self.norm(z + self.mlp(z))

    def forward(self, tokens: torch.Tensor):
        tokens = self.dropout(self.proj(tokens))
        return self.accel(self._pool(self.accel_query, tokens)), self.steer(self._pool(self.steer_query, tokens))


class Stage3VJEPA2Frozen(nn.Module):
    def __init__(
        self,
        backbone: nn.Module | None = None,
        repo_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        backbone_name: str = "vjepa2_1_vit_base_384",
        probe_type: str = "mean",
        input_size: int = 224,
        num_frames: int = 16,
        freeze_backbone: bool = True,
        probe_dim: int = 384,
        probe_heads: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.probe_type = probe_type
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        self.freeze_backbone = bool(freeze_backbone)
        self.probe_dim = int(probe_dim)
        self.probe_heads = int(probe_heads)
        self.dropout_value = float(dropout)
        self.checkpoint_path = str(checkpoint) if checkpoint is not None else ""
        self.repo_dir = str(repo_dir) if repo_dir is not None else ""
        self.backbone = backbone or _load_official_vjepa2_encoder(repo_dir, checkpoint, backbone_name)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", 768)))
        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone
        if self.freeze_backbone:
            self.backbone.eval()
        self.probe = Stage3VJEPAMeanProbe(self.embed_dim, dropout) if probe_type == "mean" else Stage3VJEPACrossAttentionProbe(self.embed_dim, probe_dim, probe_heads, dropout)
        self.register_buffer("s3_mean", S3_MEAN[:, None, :, :].clone())
        self.register_buffer("s3_std", S3_STD[:, None, :, :].clone())
        self.register_buffer("vjepa_mean", torch.tensor(VJEPA2_IMAGENET_MEAN)[:, None, None, None])
        self.register_buffer("vjepa_std", torch.tensor(VJEPA2_IMAGENET_STD)[:, None, None, None])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Stage3VJEPA2Frozen expects [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[2] != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {x.shape[2]}")
        x = (x * self.s3_std + self.s3_mean).clamp(0.0, 1.0)
        if x.shape[-2:] != (self.input_size, self.input_size):
            b, c, t, h, w = x.shape
            # 224->384 upsampling adds no new video information; it only matches the official ViT-B/16 input grid.
            x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            x = x.reshape(b, t, c, self.input_size, self.input_size).permute(0, 2, 1, 3, 4).contiguous()
        return (x - self.vjepa_mean) / self.vjepa_std

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        with torch.set_grad_enabled(not self.freeze_backbone):
            tokens = self.backbone(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens.ndim != 3:
            raise RuntimeError(f"V-JEPA2 backbone must return [B,N,D] tokens, got {tuple(tokens.shape)}")
        return tokens

    def forward(self, x: torch.Tensor):
        return self.probe(self.tokens(x))

    def model_config(self) -> dict:
        return {
            "probe_type": self.probe_type,
            "input_size": self.input_size,
            "num_frames": self.num_frames,
            "backbone_name": self.backbone_name,
            "backbone_frozen": self.freeze_backbone,
            "backbone_checkpoint": self.checkpoint_path,
            "backbone_repo": self.repo_dir,
            "feature_dim": self.embed_dim,
            "probe_dim": self.probe_dim,
            "probe_heads": self.probe_heads,
            "dropout": self.dropout_value,
            "accel_classes": ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"],
            "steer_classes": ["LEFT", "STRAIGHT", "RIGHT"],
            "normalization": {"mean": VJEPA2_IMAGENET_MEAN, "std": VJEPA2_IMAGENET_STD},
        }