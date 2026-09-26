from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .vjepa.models.vision_transformer import vit_large


def _encoder_state(payload: dict) -> dict[str, torch.Tensor]:
    state = payload.get("target_encoder", payload.get("encoder", payload))
    if not isinstance(state, dict):
        raise TypeError("V-JEPA encoder checkpoint does not contain a state dict")
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }


def _load_encoder(checkpoint_path: str | Path) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing bundled V-JEPA encoder: {checkpoint_path}")

    encoder = vit_large(
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_SiLU=False,
        tight_SiLU=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    incompatible = encoder.load_state_dict(_encoder_state(payload), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "bundled V-JEPA encoder is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return encoder


class TaskQueryAttention(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.accel_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.steer_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.accel = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4)
        )
        self.steer = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3)
        )

    def _pool(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        query = query.expand(tokens.size(0), -1, -1)
        attention = torch.softmax(
            torch.matmul(query, tokens.transpose(1, 2)) / (tokens.size(-1) ** 0.5), dim=-1
        )
        return torch.matmul(attention, tokens).squeeze(1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.norm(tokens)
        return self.accel(self._pool(self.accel_query, tokens)), self.steer(
            self._pool(self.steer_query, tokens)
        )


class Stage3VJEPA(nn.Module):
    def __init__(self, encoder_checkpoint: str | Path, head_state: dict[str, torch.Tensor]):
        super().__init__()
        self.encoder = _load_encoder(encoder_checkpoint)
        self.head = TaskQueryAttention(dim=self.encoder.embed_dim)
        self.head.load_state_dict(head_state, strict=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.encoder(x))
