from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import STAGE3_NUM_FRAMES, STAGE3_VJEPA_CHECKPOINT
from src.third_party.vjepa.models.vision_transformer import vit_large


def _load_vjepa_encoder(checkpoint: str | Path = STAGE3_VJEPA_CHECKPOINT) -> nn.Module:
    encoder = vit_large(
        img_size=224,
        patch_size=16,
        num_frames=STAGE3_NUM_FRAMES,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_SiLU=False,
        tight_SiLU=False,
    )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing V-JEPA checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("target_encoder", payload.get("encoder", payload))
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    encoder.load_state_dict(state, strict=False)
    return encoder


class TaskQueryAttention(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.accel_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.steer_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.accel = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))
        self.steer = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def _pool(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = query.expand(tokens.size(0), -1, -1)
        attn = torch.softmax(torch.matmul(q, tokens.transpose(1, 2)) / (tokens.size(-1) ** 0.5), dim=-1)
        return torch.matmul(attn, tokens).squeeze(1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.norm(tokens)
        return self.accel(self._pool(self.accel_query, tokens)), self.steer(self._pool(self.steer_query, tokens))


class Stage3VJEPA(nn.Module):
    def __init__(self, checkpoint: str | Path = STAGE3_VJEPA_CHECKPOINT):
        super().__init__()
        self.encoder = _load_vjepa_encoder(checkpoint)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = TaskQueryAttention(dim=self.encoder.embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens = self.encoder(x)
        return self.head(tokens)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

