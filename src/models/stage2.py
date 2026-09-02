from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    VideoMAEModel,
)


def infer_videomae_config_from_state_dict(
    state_dict: dict,
    num_frames: int = 16,
) -> dict:
    projection = state_dict[
        "encoder.embeddings.patch_embeddings.projection.weight"
    ]

    hidden_size = int(
        projection.shape[0]
    )

    num_channels = int(
        projection.shape[1]
    )

    tubelet_size = int(
        projection.shape[2]
    )

    patch_size = int(
        projection.shape[3]
    )

    layer_indices = [
        int(key.split(".")[3])
        for key
        in state_dict
        if key.startswith("encoder.encoder.layer.")
    ]

    if not layer_indices:
        raise RuntimeError(
            "Cannot infer VideoMAE config: encoder layers are missing from checkpoint."
        )

    num_hidden_layers = (
        max(layer_indices) + 1
    )

    intermediate_size = int(
        state_dict[
            "encoder.encoder.layer.0.intermediate.dense.weight"
        ].shape[0]
    )

    num_attention_heads = {
        384: 6,
        768: 12,
        1024: 16,
    }.get(hidden_size)

    if num_attention_heads is None:
        for candidate in range(16, 0, -1):
            if hidden_size % candidate == 0:
                num_attention_heads = candidate
                break

    return VideoMAEConfig(
        image_size=224,
        patch_size=patch_size,
        num_channels=num_channels,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        qkv_bias=True,
    ).to_dict()


class Stage2Temporal(nn.Module):
    """
    ResNet18 frame feature sequence를 입력으로 받는 기존 Stage 2 GRU 모델.
    VideoMAE 전환 이후 학습/inference 기본 경로에서는 사용하지 않는다.
    """

    def __init__(self):
        super().__init__()

        self.r = nn.GRU(
            512,
            192,
            2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )

        self.tc = nn.Linear(
            384,
            1,
        )

        self.te = nn.Linear(
            384,
            1,
        )

        self.scene = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, 4),
        )

    def logits(self, x):
        h, _ = self.r(x)

        return (
            self.tc(h).squeeze(-1),
            self.te(h).squeeze(-1),
            h,
        )

    def forward(self, x):
        collision, entry, h = self.logits(x)
        ci, ei = collision.argmax(1), entry.argmax(1)
        b = torch.arange(
            len(h),
            device=h.device,
        )

        return (
            ci,
            ei,
            self.scene(
                torch.cat(
                    [h[b, ci], h[b, ei]],
                    dim=1,
                )
            ),
        )


class Stage2VideoMAE(nn.Module):
    """
    Stage 2 VideoMAE 모델.

    현재 학습된 head는 collision head이며, side head는 구조 호환을 위해 유지한다.
    """

    def __init__(
        self,
        pretrained_name: str = (
            "MCG-NJU/"
            "videomae-small-finetuned-ssv2"
        ),
        config_dict: dict | None = None,
        num_input_frames: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        if config_dict is None:
            pretrained = (
                VideoMAEForVideoClassification
                .from_pretrained(
                    pretrained_name
                )
            )

            self.encoder = pretrained.videomae

        else:
            config = VideoMAEConfig.from_dict(
                config_dict
            )

            self.encoder = VideoMAEModel(
                config
            )

        config = self.encoder.config

        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.image_size = config.image_size
        self.num_input_frames = num_input_frames

        spatial_size = (
            self.image_size
            // self.patch_size
        )

        self.num_spatial_tokens = (
            spatial_size
            * spatial_size
        )

        self.num_temporal_tokens = (
            num_input_frames
            // self.tubelet_size
        )

        self.temporal_norm = nn.LayerNorm(
            self.hidden_size
        )

        self.collision_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_size,
                1,
            ),
        )

        self.side_head = nn.Sequential(
            nn.LayerNorm(
                self.hidden_size
            ),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_size,
                2,
            ),
        )

    def _temporal_features(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden: [B, sequence_length, D]
        return: [B, temporal_tokens, D]
        """

        batch_size = hidden.shape[0]

        expected_tokens = (
            self.num_temporal_tokens
            * self.num_spatial_tokens
        )

        if hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected VideoMAE token count: "
                f"got={hidden.shape[1]}, "
                f"expected={expected_tokens}"
            )

        hidden = hidden.reshape(
            batch_size,
            self.num_temporal_tokens,
            self.num_spatial_tokens,
            self.hidden_size,
        )

        return hidden.mean(
            dim=2
        )

    def forward(
        self,
        video: torch.Tensor,
    ) -> dict:
        """
        video: [B, T, C, H, W]
        """

        outputs = self.encoder(
            pixel_values=video,
        )

        temporal = self._temporal_features(
            outputs.last_hidden_state
        )

        temporal = self.temporal_norm(
            temporal
        )

        collision_logits = self.collision_head(
            temporal
        ).squeeze(-1)

        collision_logits = F.interpolate(
            collision_logits.unsqueeze(1),
            size=self.num_input_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        global_feature = temporal.mean(
            dim=1
        )

        side_logits = self.side_head(
            global_feature
        )

        return {
            "collision_logits": collision_logits,
            "side_logits": side_logits,
            "temporal_features": temporal,
        }

    def freeze_backbone(
        self,
        unfreeze_last_n: int = 2,
    ) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        layers = self.encoder.encoder.layer

        if unfreeze_last_n > 0:
            for layer in layers[-unfreeze_last_n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        layernorm = getattr(
            self.encoder,
            "layernorm",
            None,
        )

        if layernorm is not None:
            for parameter in layernorm.parameters():
                parameter.requires_grad = True
