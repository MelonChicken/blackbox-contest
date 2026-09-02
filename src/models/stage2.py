# 1. Configuration

# 1.1. 관련 패키지 임포트
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    VideoMAEForVideoClassification,
)




class Stage2Temporal(nn.Module):
    """
    입력은 이미 다른 모델에서 추출한 frame-wise feature sequence가 된다.
    입력 feature dimension은 512이다.
    Stage 2: 사고 주요시점·상황 분석 모델
    """

    def __init__(self):
        # nn.Module의 초기화를 수행
        super().__init__()

        # 각 frame feature가 512차원 input_size = 512
        # GRU의 hidden representation 크기는 192 hidden_size = 192
        # GRU layer를 두 층 num_layers = 2
        # 입력 shape이 [B, T, feature] 가 되도록 한다. batch_first=True
        # 양방향 GRU로 설정. 이를 통해 representation의 수가 192x2=384가 된다. bidirectional=True
        # GRU layer 사이에 dropout을 적용 dropout=0.15
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)

        # collision head 확인. 각 timestep의 384차원 feature를 scalar 하나로 줄인다.
        # 즉 각 frame마다 "이 frame이 collision frame일 가능성이 얼마나 높은가?"에 대응되는 logit 하나를 생성
        self.tc = nn.Linear(384, 1)
        # entry head 확인. 각 frame마다 하나의 score를 만들지만 이번에는 entry frame을 찾기 위한 것
        self.te = nn.Linear(384, 1)
        # scene classifier. 각 클래스별로 입력으로 받는 feature에 대하여 scene logit을 계산한다.
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def logits(self, x):
        # 입력에 대해 GRU를 통과시킨다.
        # 입력: x = [B,T,512]
        # 출력: h = [B,T,384]
        h, _ = self.r(x)
        # collision score : tc를 통해 확인 [B, T]
        # entry score : te를 통해 확인 [B, T]
        return self.tc(h).squeeze(-1), self.te(h).squeeze(-1), h

    def forward(self, x):
        # logit을 계산한다.
        collision, entry, h = self.logits(x)
        # collision frame과 entry frame 선택
        ci, ei = collision.argmax(1), entry.argmax(1)
        # 각 batch sample에서 선택한 frame의 hidden state를 가져온다.
        b = torch.arange(len(h), device=h.device)
        # 두 feature를 concatenate 한 뒤 self.scene()에 넣는다.
        # predicted collision index
        # predicted entry index
        # scene classification logits
        # 위의 세가지를 최종적으로 제출한다.
        return ci, ei, self.scene(torch.cat([h[b, ci], h[b, ei]], 1))

class Stage2VideoMAE(nn.Module):
    """
    Stage 2 VideoMAE.

    Initial version:
        - collision temporal prediction
        - entry-side auxiliary prediction

    Later:
        - entry temporal head
        - evasion-space head
    """

    def __init__(
        self,
        pretrained_name: str = (
            "MCG-NJU/"
            "videomae-small-finetuned-ssv2"
        ),
        num_input_frames: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        pretrained = (
            VideoMAEForVideoClassification
            .from_pretrained(
                pretrained_name
            )
        )

        # classification head 제거하고
        # pretrained VideoMAE encoder만 사용
        self.encoder = (
            pretrained.videomae
        )

        config = self.encoder.config

        self.hidden_size = (
            config.hidden_size
        )

        self.patch_size = (
            config.patch_size
        )

        self.tubelet_size = (
            config.tubelet_size
        )

        self.image_size = (
            config.image_size
        )

        self.num_input_frames = (
            num_input_frames
        )

        # Spatial patch 개수
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

        # ----------------------------------------------------
        # Temporal collision head
        # ----------------------------------------------------

        self.temporal_norm = (
            nn.LayerNorm(
                self.hidden_size
            )
        )

        self.collision_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_size,
                1,
            ),
        )

        # ----------------------------------------------------
        # Global side head
        # ----------------------------------------------------

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

    # ========================================================
    # Feature extraction
    # ========================================================

    def _temporal_features(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden:
            [B, sequence_length, D]

        VideoMAE sequence:
            temporal tubelets
            × spatial patches

        return:
            [B, temporal_tokens, D]
        """

        batch_size = (
            hidden.shape[0]
        )

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

        # Spatial mean pooling
        temporal = hidden.mean(
            dim=2
        )

        return temporal

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        video: torch.Tensor,
    ) -> dict:
        """
        video:
            [B, T, C, H, W]
        """

        outputs = self.encoder(
            pixel_values=video,
        )

        hidden = (
            outputs.last_hidden_state
        )

        temporal = (
            self._temporal_features(
                hidden
            )
        )

        temporal = (
            self.temporal_norm(
                temporal
            )
        )

        # [B, 8, 1]
        collision_logits = (
            self.collision_head(
                temporal
            )
        )

        # [B, 8]
        collision_logits = (
            collision_logits
            .squeeze(-1)
        )

        # tubelet level 8
        # ->
        # sampled-frame level 16
        collision_logits = (
            F.interpolate(
                collision_logits
                .unsqueeze(1),

                size=(
                    self.num_input_frames
                ),

                mode="linear",

                align_corners=False,
            )
            .squeeze(1)
        )

        # Global representation
        global_feature = (
            temporal.mean(
                dim=1
            )
        )

        side_logits = (
            self.side_head(
                global_feature
            )
        )

        return {
            "collision_logits": (
                collision_logits
            ),

            "side_logits": (
                side_logits
            ),

            "temporal_features": (
                temporal
            ),
        }

    def freeze_backbone(
            self,
            unfreeze_last_n: int = 2,
    ) -> None:

        # 전체 encoder freeze
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # 마지막 N개 transformer block만 학습
        layers = self.encoder.encoder.layer

        if unfreeze_last_n > 0:
            for layer in layers[-unfreeze_last_n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        # final layernorm이 실제로 존재할 때만 unfreeze
        layernorm = getattr(
            self.encoder,
            "layernorm",
            None,
        )

        if layernorm is not None:
            for parameter in layernorm.parameters():
                parameter.requires_grad = True