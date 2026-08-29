from __future__ import annotations

import math
import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class RecaptureTransform:
    """
    Synthetic screen-recapture augmentation.

    Input:
        clip: float Tensor [T, C, H, W], range [0, 1]

    Output:
        float Tensor [T, C, H, W], range [0, 1]
    """

    def __init__(self):
        pass

    def __call__(
        self,
        clip: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        rng = random.Random(seed)

        clip = clip.clone()

        # 최소 2종류 이상의 artifact가 발생하도록 구성
        transforms = [
            self._perspective,
            self._moire,
            self._gamma_flicker,
            self._blur,
            self._resample,
        ]

        n = rng.randint(2, 4)

        selected = rng.sample(transforms, n)

        for transform in selected:
            clip = transform(clip, rng)

        return clip.clamp(0.0, 1.0)

    @staticmethod
    def _perspective(
        clip: torch.Tensor,
        rng: random.Random,
    ):
        _, _, h, w = clip.shape

        amount = rng.uniform(0.01, 0.05)

        dx = int(w * amount)
        dy = int(h * amount)

        startpoints = [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]

        endpoints = [
            [
                rng.randint(0, dx),
                rng.randint(0, dy),
            ],
            [
                w - 1 - rng.randint(0, dx),
                rng.randint(0, dy),
            ],
            [
                w - 1 - rng.randint(0, dx),
                h - 1 - rng.randint(0, dy),
            ],
            [
                rng.randint(0, dx),
                h - 1 - rng.randint(0, dy),
            ],
        ]

        return TF.perspective(
            clip,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    @staticmethod
    def _moire(
        clip: torch.Tensor,
        rng: random.Random,
    ):
        t, _, h, w = clip.shape

        device = clip.device
        dtype = clip.dtype

        yy = torch.linspace(
            0,
            1,
            h,
            device=device,
            dtype=dtype,
        )

        xx = torch.linspace(
            0,
            1,
            w,
            device=device,
            dtype=dtype,
        )

        y, x = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        frequency = rng.uniform(15.0, 50.0)
        angle = rng.uniform(0.0, math.pi)
        amplitude = rng.uniform(0.015, 0.06)

        direction = (
            x * math.cos(angle)
            + y * math.sin(angle)
        )

        frames = []

        initial_phase = rng.uniform(0.0, math.pi * 2)
        temporal_shift = rng.uniform(-0.15, 0.15)

        for i in range(t):
            phase = initial_phase + i * temporal_shift

            pattern = torch.sin(
                2
                * math.pi
                * frequency
                * direction
                + phase
            )

            pattern = pattern.unsqueeze(0)

            frame = clip[i] + amplitude * pattern

            frames.append(frame)

        return torch.stack(frames)

    @staticmethod
    def _gamma_flicker(
        clip: torch.Tensor,
        rng: random.Random,
    ):
        t = clip.shape[0]

        gamma = rng.uniform(0.80, 1.25)

        clip = clip.clamp(1e-6, 1.0).pow(gamma)

        amplitude = rng.uniform(0.01, 0.08)
        frequency = rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, math.pi * 2)

        frames = []

        for i in range(t):
            brightness = 1.0 + amplitude * math.sin(
                phase
                + 2
                * math.pi
                * frequency
                * i
                / max(t - 1, 1)
            )

            frames.append(
                clip[i] * brightness
            )

        return torch.stack(frames)

    @staticmethod
    def _blur(
        clip: torch.Tensor,
        rng: random.Random,
    ):
        sigma = rng.uniform(0.3, 1.2)

        return TF.gaussian_blur(
            clip,
            kernel_size=[3, 3],
            sigma=[sigma, sigma],
        )

    @staticmethod
    def _resample(
        clip: torch.Tensor,
        rng: random.Random,
    ):
        _, _, h, w = clip.shape

        scale = rng.uniform(0.60, 0.90)

        small_h = max(32, int(h * scale))
        small_w = max(32, int(w * scale))

        clip = TF.resize(
            clip,
            [small_h, small_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        clip = TF.resize(
            clip,
            [h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        return clip