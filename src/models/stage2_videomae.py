from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEModel


STAGE2_BACKBONES = {
    "small": "MCG-NJU/videomae-small-finetuned-ssv2",
    "base": "MCG-NJU/videomae-base-finetuned-ssv2",
}

DEFAULT_STAGE2_CONFIG: dict[str, Any] = {
    "backbone_variant": "small",
    "num_frames": 16,
    "image_size": 224,
    "direction_classes": 2,
    "avoidance_classes": 2,
    "dropout": 0.2,
}


def default_hf_config(num_frames: int = 16, image_size: int = 224) -> dict[str, Any]:
    return {
        "image_size": image_size,
        "patch_size": 16,
        "num_channels": 3,
        "num_frames": num_frames,
        "tubelet_size": 2,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "qkv_bias": True,
    }


class Stage2VideoMAE(nn.Module):
    def __init__(
        self,
        backbone_variant: str = "small",
        num_frames: int = 16,
        image_size: int = 224,
        direction_classes: int = 2,
        avoidance_classes: int = 2,
        dropout: float = 0.2,
        hf_config: dict[str, Any] | None = None,
        use_pretrained: bool = False,
    ):
        super().__init__()
        if backbone_variant not in STAGE2_BACKBONES:
            raise ValueError(f"Unknown Stage2 backbone variant: {backbone_variant}")

        self.backbone_variant = backbone_variant
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.direction_classes = int(direction_classes)
        self.avoidance_classes = int(avoidance_classes)
        self.dropout = float(dropout)

        if use_pretrained:
            pretrained = VideoMAEForVideoClassification.from_pretrained(STAGE2_BACKBONES[backbone_variant])
            self.backbone = pretrained.videomae
        else:
            config = VideoMAEConfig.from_dict(hf_config or default_hf_config(self.num_frames, self.image_size))
            self.backbone = VideoMAEModel(config)

        hidden_size = int(self.backbone.config.hidden_size)
        self.head_dropout = nn.Dropout(self.dropout)
        self.collision_head = nn.Linear(hidden_size, 1)
        self.entry_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, self.direction_classes)
        self.avoidance_head = nn.Linear(hidden_size, self.avoidance_classes)

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone_variant": self.backbone_variant,
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "direction_classes": self.direction_classes,
            "avoidance_classes": self.avoidance_classes,
            "dropout": self.dropout,
            "hf_config": self.backbone.config.to_dict(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], use_pretrained: bool = False) -> "Stage2VideoMAE":
        merged = deepcopy(DEFAULT_STAGE2_CONFIG)
        merged.update(config)
        return cls(
            backbone_variant=merged["backbone_variant"],
            num_frames=merged["num_frames"],
            image_size=merged["image_size"],
            direction_classes=merged["direction_classes"],
            avoidance_classes=merged["avoidance_classes"],
            dropout=merged["dropout"],
            hf_config=merged.get("hf_config"),
            use_pretrained=use_pretrained,
        )

    def freeze_backbone(self, unfreeze_last_n: int = 2) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None and unfreeze_last_n > 0:
            for block in layers[-unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        layernorm = getattr(self.backbone, "layernorm", None)
        if layernorm is not None:
            for parameter in layernorm.parameters():
                parameter.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        batch_size, token_count, hidden_size = tokens.shape
        temporal_count = max(1, self.num_frames // int(self.backbone.config.tubelet_size))
        spatial_count = max(1, token_count // temporal_count)
        temporal_tokens = tokens[:, : temporal_count * spatial_count].reshape(
            batch_size, temporal_count, spatial_count, hidden_size
        )
        temporal_features = temporal_tokens.mean(dim=2)
        temporal_features = self.head_dropout(temporal_features)

        collision_logits = self.collision_head(temporal_features).squeeze(-1)
        entry_logits = self.entry_head(temporal_features).squeeze(-1)
        if collision_logits.shape[1] != self.num_frames:
            collision_logits = F.interpolate(
                collision_logits.unsqueeze(1), size=self.num_frames, mode="linear", align_corners=False
            ).squeeze(1)
            entry_logits = F.interpolate(
                entry_logits.unsqueeze(1), size=self.num_frames, mode="linear", align_corners=False
            ).squeeze(1)

        global_feature = self.head_dropout(temporal_features.mean(dim=1))
        return {
            "collision_logits": collision_logits,
            "entry_logits": entry_logits,
            "direction_logits": self.direction_head(global_feature),
            "avoidance_logits": self.avoidance_head(global_feature),
        }


def build_stage2_model(config: dict[str, Any] | None = None, use_pretrained: bool = False) -> Stage2VideoMAE:
    return Stage2VideoMAE.from_config(config or DEFAULT_STAGE2_CONFIG, use_pretrained=use_pretrained)
