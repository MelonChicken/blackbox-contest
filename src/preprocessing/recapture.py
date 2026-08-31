from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode


class RecaptureTransform:
    """
    Synthetic screen-recapture augmentation.

    재촬영 여부와 비교적 직접적으로 관련된
    display-camera acquisition feature만 합성한다.

    포함하는 특징:
        - display subpixel / sampling aliasing
        - perspective distortion
        - temporal display flicker
        - rolling brightness band
        - camera micro-jitter

    포함하지 않는 특징:
        - 일반적인 blur
        - 일반적인 resize / resampling degradation
        - static gamma / brightness variation

    위의 일반적인 영상 열화는 ORIGINAL에서도 발생할 수 있으므로
    Dataset의 common augmentation에서 두 class에 모두 적용한다.

    Parameters
    ----------
    profile:
        "train":
            training synthetic distribution

        "holdout":
            validation synthetic distribution.
            training과 완전히 같은 parameter range만 반복해서
            validation score가 과대평가되는 것을 조금 줄이기 위한 용도다.

            실제 physical recapture validation을 대체하지는 않는다.

    Input
    -----
    clip:
        float Tensor [T, C, H, W]
        range [0, 1]

    timestamps:
        각 sampled frame의 실제 시간(sec).
        shape [T]

        None이면 normalized time [0, 1]을 fallback으로 사용한다.

    Output
    ------
    float Tensor [T, C, H, W]
    range [0, 1]
    """

    def __init__(
        self,
        profile: str = "train",
    ):
        if profile not in {
            "train",
            "holdout",
        }:
            raise ValueError(
                f"Unsupported profile: {profile}"
            )

        self.profile = profile

        # 일부 RERECORDED는 육안상 거의 깨끗하게 만들어
        # "artifact가 강하면 rerecorded"라는 shortcut을 줄인다.
        self.near_clean_prob = 0.15

        if profile == "train":

            self.artifact_prob = {
                "display_aliasing": 0.35,
                "perspective": 0.30,
                "display_flicker": 0.50,
                "rolling_band": 0.30,
                "camera_jitter": 0.40,
            }

            self.severity_weights = [
                0.35,  # weak
                0.50,  # medium
                0.15,  # strong
            ]

        else:

            # validation에서는 확률 및 parameter range를
            # training과 조금 다르게 사용한다.
            self.artifact_prob = {
                "display_aliasing": 0.25,
                "perspective": 0.20,
                "display_flicker": 0.50,
                "rolling_band": 0.35,
                "camera_jitter": 0.30,
            }

            self.severity_weights = [
                0.60,
                0.35,
                0.05,
            ]

    def __call__(
        self,
        clip: torch.Tensor,
        seed: int | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        clip에 synthetic recapture feature를 적용한다.
        """

        if clip.ndim != 4:
            raise ValueError(
                "clip must have shape [T, C, H, W]"
            )

        if clip.shape[1] != 3:
            raise ValueError(
                "clip must have 3 RGB channels"
            )

        rng = random.Random(seed)

        clip = clip.clone()

        times = self._prepare_timestamps(
            clip,
            timestamps,
        )

        # ---------------------------------------
        # Near-clean positive
        # ---------------------------------------
        #
        # 일부 RERECORDED에는 아주 약한 recapture feature 하나만
        # 넣는다.
        #
        # 실제 재촬영 영상도 항상 moire, blur, perspective가
        # 눈에 띄는 것은 아니기 때문이다.
        # ---------------------------------------

        if rng.random() < self.near_clean_prob:

            transform_name = rng.choice(
                [
                    "display_aliasing",
                    "display_flicker",
                    "camera_jitter",
                ]
            )

            return self._apply_named_transform(
                clip=clip,
                transform_name=transform_name,
                rng=rng,
                severity="weak",
                timestamps=times,
            ).clamp(
                0.0,
                1.0,
            )

        # ---------------------------------------
        # Normal synthetic recapture
        # ---------------------------------------

        severity = rng.choices(
            population=[
                "weak",
                "medium",
                "strong",
            ],
            weights=self.severity_weights,
            k=1,
        )[0]

        applied = []

        # Display-side effect를 먼저 적용한다.

        if (
            rng.random()
            < self.artifact_prob["display_aliasing"]
        ):
            clip = self._display_aliasing(
                clip,
                rng,
                severity,
                times,
            )

            applied.append(
                "display_aliasing"
            )

        # Camera geometry

        if (
            rng.random()
            < self.artifact_prob["perspective"]
        ):
            clip = self._perspective(
                clip,
                rng,
                severity,
            )

            applied.append(
                "perspective"
            )

        if (
            rng.random()
            < self.artifact_prob["camera_jitter"]
        ):
            clip = self._camera_jitter(
                clip,
                rng,
                severity,
            )

            applied.append(
                "camera_jitter"
            )

        # Temporal display-camera interaction

        if (
            rng.random()
            < self.artifact_prob["display_flicker"]
        ):
            clip = self._display_flicker(
                clip,
                rng,
                severity,
                times,
            )

            applied.append(
                "display_flicker"
            )

        if (
            rng.random()
            < self.artifact_prob["rolling_band"]
        ):
            clip = self._rolling_band(
                clip,
                rng,
                severity,
                times,
            )

            applied.append(
                "rolling_band"
            )

        # ---------------------------------------
        # 모든 artifact가 우연히 선택되지 않은 경우
        # ---------------------------------------
        #
        # ORIGINAL과 완전히 동일한 positive를 만들기보다는
        # 약한 recapture-specific feature 하나만 넣는다.
        # ---------------------------------------

        if not applied:

            transform_name = rng.choice(
                [
                    "display_aliasing",
                    "display_flicker",
                    "camera_jitter",
                ]
            )

            clip = self._apply_named_transform(
                clip=clip,
                transform_name=transform_name,
                rng=rng,
                severity="weak",
                timestamps=times,
            )

        return clip.clamp(
            0.0,
            1.0,
        )

    def _apply_named_transform(
        self,
        clip: torch.Tensor,
        transform_name: str,
        rng: random.Random,
        severity: str,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:

        if transform_name == "display_aliasing":

            return self._display_aliasing(
                clip,
                rng,
                severity,
                timestamps,
            )

        if transform_name == "perspective":

            return self._perspective(
                clip,
                rng,
                severity,
            )

        if transform_name == "display_flicker":

            return self._display_flicker(
                clip,
                rng,
                severity,
                timestamps,
            )

        if transform_name == "rolling_band":

            return self._rolling_band(
                clip,
                rng,
                severity,
                timestamps,
            )

        if transform_name == "camera_jitter":

            return self._camera_jitter(
                clip,
                rng,
                severity,
            )

        raise ValueError(
            f"Unknown transform: {transform_name}"
        )

    @staticmethod
    def _prepare_timestamps(
        clip: torch.Tensor,
        timestamps: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        temporal transform에서 사용할 실제 시간축을 준비한다.
        """

        t = clip.shape[0]

        if timestamps is None:

            return torch.linspace(
                0.0,
                1.0,
                steps=t,
                device=clip.device,
                dtype=clip.dtype,
            )

        timestamps = torch.as_tensor(
            timestamps,
            device=clip.device,
            dtype=clip.dtype,
        ).reshape(-1)

        if timestamps.numel() != t:
            raise ValueError(
                "timestamps length must match "
                "the number of frames"
            )

        # 절대 시작 시간이 의미를 가지지는 않으므로
        # 첫 sampled frame을 t=0으로 맞춘다.
        timestamps = (
            timestamps
            - timestamps[0]
        )

        return timestamps

    @staticmethod
    def _severity_range(
        severity: str,
        weak: tuple[float, float],
        medium: tuple[float, float],
        strong: tuple[float, float],
    ) -> tuple[float, float]:

        ranges = {
            "weak": weak,
            "medium": medium,
            "strong": strong,
        }

        if severity not in ranges:
            raise ValueError(
                f"Unknown severity: {severity}"
            )

        return ranges[severity]

    @staticmethod
    def _make_smooth_noise(
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        rng: random.Random,
        grid_size: int = 6,
    ) -> torch.Tensor:
        """
        작은 random field를 bicubic upsampling하여
        부드러운 spatial perturbation을 만든다.

        display lattice가 화면 전체에서 완벽히 동일한
        수학적 격자가 되는 것을 줄이기 위한 용도다.
        """

        seed = rng.randrange(
            0,
            2**31 - 1,
        )

        generator = torch.Generator(
            device=device,
        )

        generator.manual_seed(
            seed
        )

        noise = torch.rand(
            (
                1,
                1,
                grid_size,
                grid_size,
            ),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        noise = (
            noise * 2.0
            - 1.0
        )

        noise = F.interpolate(
            noise,
            size=(
                h,
                w,
            ),
            mode="bicubic",
            align_corners=False,
        )

        return noise[
            0,
            0,
        ]

    def _display_aliasing(
        self,
        clip: torch.Tensor,
        rng: random.Random,
        severity: str,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        display pixel/subpixel structure를 약하게 근사한다.

        중요한 점:
            최종 moire pattern을 sine wave로 직접 그리지 않는다.

        대신

            display lattice
                +
            channel-dependent subpixel offset
                +
            약한 spatial irregularity
                +
            small temporal phase change

        를 합성한다.

        Dataset에서는 이 transform을 224보다 큰 intermediate
        resolution에서 수행하고 마지막에 224로 downsample한다.

        따라서 실제 moire-like aliasing은 주로
        lattice와 최종 sampling grid 사이의 interaction으로
        발생하도록 유도한다.
        """

        t, _, h, w = clip.shape

        device = clip.device
        dtype = clip.dtype

        strength_range = self._severity_range(
            severity,
            weak=(0.003, 0.010),
            medium=(0.007, 0.018),
            strong=(0.012, 0.030),
        )

        strength = rng.uniform(
            *strength_range
        )

        # ---------------------------------------
        # Pixel pitch
        # ---------------------------------------
        #
        # 정수 pitch 하나에 고정하면 synthetic signature가
        # 너무 강해질 수 있으므로 실수값으로 둔다.
        # ---------------------------------------

        if self.profile == "train":

            pitch_x = rng.uniform(
                2.2,
                5.0,
            )

            pitch_y = rng.uniform(
                2.4,
                5.5,
            )

        else:

            pitch_x = rng.uniform(
                1.9,
                5.8,
            )

            pitch_y = rng.uniform(
                2.0,
                6.2,
            )

        # display orientation

        angle = rng.uniform(
            -math.radians(8.0),
            math.radians(8.0),
        )

        cos_a = math.cos(
            angle
        )

        sin_a = math.sin(
            angle
        )

        yy = torch.arange(
            h,
            device=device,
            dtype=dtype,
        )

        xx = torch.arange(
            w,
            device=device,
            dtype=dtype,
        )

        y, x = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        cx = (
            w - 1
        ) * 0.5

        cy = (
            h - 1
        ) * 0.5

        x0 = (
            x - cx
        )

        y0 = (
            y - cy
        )

        u = (
            cos_a * x0
            + sin_a * y0
        )

        v = (
            -sin_a * x0
            + cos_a * y0
        )

        # ---------------------------------------
        # Smooth spatial irregularity
        # ---------------------------------------

        warp_u = self._make_smooth_noise(
            h,
            w,
            device,
            dtype,
            rng,
            grid_size=rng.randint(
                4,
                8,
            ),
        )

        warp_v = self._make_smooth_noise(
            h,
            w,
            device,
            dtype,
            rng,
            grid_size=rng.randint(
                4,
                8,
            ),
        )

        warp_amount = rng.uniform(
            0.05,
            0.30,
        )

        u = (
            u
            + warp_amount
            * warp_u
        )

        v = (
            v
            + warp_amount
            * warp_v
        )

        # ---------------------------------------
        # RGB subpixel layout
        # ---------------------------------------

        layouts = [
            [0.0, 1.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.5, 0.25],
        ]

        channel_offsets = rng.choice(
            layouts
        )

        channel_scales = torch.tensor(
            [
                rng.uniform(
                    0.85,
                    1.15,
                ),
                rng.uniform(
                    0.85,
                    1.15,
                ),
                rng.uniform(
                    0.85,
                    1.15,
                ),
            ],
            device=device,
            dtype=dtype,
        ).reshape(
            3,
            1,
            1,
        )

        # display-camera relative movement를
        # 아주 미세한 lattice phase drift로 근사한다.

        phase_velocity = rng.uniform(
            -0.25,
            0.25,
        )

        frames = []

        for i in range(t):

            temporal_shift = (
                phase_velocity
                * timestamps[i]
            )

            channel_patterns = []

            for channel in range(3):

                offset = (
                    channel_offsets[channel]
                    + temporal_shift
                )

                # --------------------------------
                # Horizontal subpixel structure
                # --------------------------------

                cell_x = torch.remainder(
                    (
                        u / pitch_x
                    )
                    - offset,
                    1.0,
                )

                # circular distance to center
                dist_x = torch.minimum(
                    cell_x,
                    1.0 - cell_x,
                )

                # Gaussian-like subpixel aperture
                aperture_x = torch.exp(
                    -(
                        dist_x
                        / 0.22
                    )
                    ** 2
                )

                # --------------------------------
                # Weak row / pixel-grid structure
                # --------------------------------

                cell_y = torch.remainder(
                    v / pitch_y,
                    1.0,
                )

                dist_y = torch.minimum(
                    cell_y,
                    1.0 - cell_y,
                )

                aperture_y = torch.exp(
                    -(
                        dist_y
                        / 0.30
                    )
                    ** 2
                )

                # vertical RGB subpixel structure가 중심이고
                # row structure는 약하게 섞는다.

                pattern = (
                    0.80
                    * aperture_x
                    + 0.20
                    * aperture_y
                )

                # zero mean으로 바꿔 전체 brightness가
                # 일정 방향으로 변하지 않게 한다.

                pattern = (
                    pattern
                    - pattern.mean()
                )

                channel_patterns.append(
                    pattern
                )

            modulation = torch.stack(
                channel_patterns,
                dim=0,
            )

            modulation = (
                modulation
                * channel_scales
            )

            frame = clip[i]

            # 밝은 영역과 어두운 영역에서 항상 동일하게
            # lattice가 나타나는 것을 피한다.

            luminance = frame.mean(
                dim=0,
                keepdim=True,
            )

            local_strength = (
                0.35
                + 0.65
                * luminance
            )

            frame = (
                frame
                * (
                    1.0
                    + strength
                    * modulation
                    * local_strength
                )
            )

            frames.append(
                frame
            )

        return torch.stack(
            frames,
            dim=0,
        )

    def _perspective(
        self,
        clip: torch.Tensor,
        rng: random.Random,
        severity: str,
    ) -> torch.Tensor:
        """
        화면과 카메라 optical axis가 완전히 정렬되지 않았을 때의
        약한 perspective distortion을 합성한다.

        reflect padding 후 center crop하여
        기존 fill=0 방식의 검은 삼각형이
        RERECORDED shortcut이 되지 않도록 한다.
        """

        _, _, h, w = clip.shape

        amount_range = self._severity_range(
            severity,
            weak=(
                0.002,
                0.008,
            ),
            medium=(
                0.005,
                0.018,
            ),
            strong=(
                0.010,
                0.030,
            ),
        )

        amount = rng.uniform(
            *amount_range
        )

        pad = max(
            8,
            int(
                max(h, w)
                * 0.10
            ),
        )

        padded = F.pad(
            clip,
            (
                pad,
                pad,
                pad,
                pad,
            ),
            mode="reflect",
        )

        _, _, hp, wp = (
            padded.shape
        )

        dx = max(
            1,
            int(
                wp
                * amount
            ),
        )

        dy = max(
            1,
            int(
                hp
                * amount
            ),
        )

        startpoints = [
            [
                0,
                0,
            ],
            [
                wp - 1,
                0,
            ],
            [
                wp - 1,
                hp - 1,
            ],
            [
                0,
                hp - 1,
            ],
        ]

        endpoints = [
            [
                rng.randint(
                    0,
                    dx,
                ),
                rng.randint(
                    0,
                    dy,
                ),
            ],
            [
                wp
                - 1
                - rng.randint(
                    0,
                    dx,
                ),
                rng.randint(
                    0,
                    dy,
                ),
            ],
            [
                wp
                - 1
                - rng.randint(
                    0,
                    dx,
                ),
                hp
                - 1
                - rng.randint(
                    0,
                    dy,
                ),
            ],
            [
                rng.randint(
                    0,
                    dx,
                ),
                hp
                - 1
                - rng.randint(
                    0,
                    dy,
                ),
            ],
        ]

        warped = TF.perspective(
            padded,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            fill=0,
        )

        warped = TF.center_crop(
            warped,
            [
                h,
                w,
            ],
        )

        return warped

    def _camera_jitter(
        self,
        clip: torch.Tensor,
        rng: random.Random,
        severity: str,
    ) -> torch.Tensor:
        """
        handheld camera의 작은 frame-to-frame viewpoint 변화를
        근사한다.

        큰 흔들림이 아니라 subpixel~수 pixel 수준의
        micro-jitter만 생성한다.
        """

        _, _, h, w = (
            clip.shape
        )

        translation_range = (
            self._severity_range(
                severity,
                weak=(
                    0.001,
                    0.003,
                ),
                medium=(
                    0.002,
                    0.005,
                ),
                strong=(
                    0.004,
                    0.008,
                ),
            )
        )

        rotation_range = (
            self._severity_range(
                severity,
                weak=(
                    0.02,
                    0.10,
                ),
                medium=(
                    0.05,
                    0.25,
                ),
                strong=(
                    0.15,
                    0.50,
                ),
            )
        )

        translation_fraction = (
            rng.uniform(
                *translation_range
            )
        )

        max_dx = max(
            1,
            int(
                w
                * translation_fraction
            ),
        )

        max_dy = max(
            1,
            int(
                h
                * translation_fraction
            ),
        )

        max_angle = rng.uniform(
            *rotation_range
        )

        pad = max(
            6,
            int(
                max(h, w)
                * 0.05
            ),
        )

        # 완전히 독립적인 jitter보다
        # 약간의 random walk를 사용한다.

        dx = 0
        dy = 0
        angle = 0.0

        frames = []

        for frame in clip:

            dx += rng.choice(
                [
                    -1,
                    0,
                    1,
                ]
            )

            dy += rng.choice(
                [
                    -1,
                    0,
                    1,
                ]
            )

            dx = max(
                -max_dx,
                min(
                    max_dx,
                    dx,
                ),
            )

            dy = max(
                -max_dy,
                min(
                    max_dy,
                    dy,
                ),
            )

            angle += rng.uniform(
                -max_angle * 0.25,
                max_angle * 0.25,
            )

            angle = max(
                -max_angle,
                min(
                    max_angle,
                    angle,
                ),
            )

            padded = F.pad(
                frame,
                (
                    pad,
                    pad,
                    pad,
                    pad,
                ),
                mode="reflect",
            )

            transformed = TF.affine(
                padded,
                angle=angle,
                translate=[
                    dx,
                    dy,
                ],
                scale=1.0,
                shear=[
                    0.0,
                    0.0,
                ],
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                fill=0,
            )

            transformed = (
                TF.center_crop(
                    transformed,
                    [
                        h,
                        w,
                    ],
                )
            )

            frames.append(
                transformed
            )

        return torch.stack(
            frames,
            dim=0,
        )

    def _display_flicker(
        self,
        clip: torch.Tensor,
        rng: random.Random,
        severity: str,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        display refresh와 camera sampling의 시간축 불일치로
        관측될 수 있는 약한 global brightness variation을
        근사한다.

        sampled clip index가 아니라 실제 timestamp를 사용한다.
        """

        amplitude_range = (
            self._severity_range(
                severity,
                weak=(
                    0.003,
                    0.012,
                ),
                medium=(
                    0.008,
                    0.028,
                ),
                strong=(
                    0.018,
                    0.045,
                ),
            )
        )

        amplitude = rng.uniform(
            *amplitude_range
        )

        if self.profile == "train":

            beat_frequency = rng.uniform(
                0.4,
                5.0,
            )

        else:

            beat_frequency = rng.uniform(
                0.2,
                7.0,
            )

        phase = rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        # 단일 완벽한 sine signature를 약하게 하기 위해
        # 작은 harmonic을 하나 섞는다.

        harmonic_strength = rng.uniform(
            0.05,
            0.25,
        )

        harmonic_phase = rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        primary = torch.sin(
            (
                2.0
                * math.pi
                * beat_frequency
                * timestamps
            )
            + phase
        )

        secondary = torch.sin(
            (
                2.0
                * math.pi
                * (
                    beat_frequency
                    * 2.0
                )
                * timestamps
            )
            + harmonic_phase
        )

        brightness = (
            1.0
            + amplitude
            * (
                primary
                + harmonic_strength
                * secondary
            )
        )

        brightness = brightness.reshape(
            -1,
            1,
            1,
            1,
        )

        return (
            clip
            * brightness
        )

    def _rolling_band(
        self,
        clip: torch.Tensor,
        rng: random.Random,
        severity: str,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        display refresh와 rolling-shutter 계열의 interaction에서
        관측될 수 있는 움직이는 brightness band를 단순 근사한다.

        강한 줄무늬가 아니라 낮은 amplitude의 modulation만 사용한다.
        """

        t, _, h, w = (
            clip.shape
        )

        device = clip.device
        dtype = clip.dtype

        amplitude_range = (
            self._severity_range(
                severity,
                weak=(
                    0.003,
                    0.010,
                ),
                medium=(
                    0.007,
                    0.022,
                ),
                strong=(
                    0.015,
                    0.035,
                ),
            )
        )

        amplitude = rng.uniform(
            *amplitude_range
        )

        yy = torch.linspace(
            0.0,
            1.0,
            h,
            device=device,
            dtype=dtype,
        )

        xx = torch.linspace(
            0.0,
            1.0,
            w,
            device=device,
            dtype=dtype,
        )

        y, x = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        # 완벽한 수평 stripe에 고정하지 않는다.

        slope = rng.uniform(
            -0.20,
            0.20,
        )

        direction = (
            y
            + slope * x
        )

        spatial_frequency = rng.uniform(
            0.6,
            2.2,
        )

        temporal_rate = rng.uniform(
            0.4,
            5.0,
        )

        phase = rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        # band intensity가 공간 전체에서 동일하지 않도록
        # smooth envelope를 추가한다.

        envelope = (
            0.70
            + 0.30
            * self._make_smooth_noise(
                h,
                w,
                device,
                dtype,
                rng,
                grid_size=rng.randint(
                    3,
                    6,
                ),
            )
        )

        frames = []

        for i in range(t):

            temporal_phase = (
                2.0
                * math.pi
                * temporal_rate
                * timestamps[i]
            )

            band = torch.sin(
                (
                    2.0
                    * math.pi
                    * spatial_frequency
                    * direction
                )
                + temporal_phase
                + phase
            )

            band = (
                band
                * envelope
            )

            modulation = (
                1.0
                + amplitude
                * band
            )

            frame = (
                clip[i]
                * modulation.unsqueeze(0)
            )

            frames.append(
                frame
            )

        return torch.stack(
            frames,
            dim=0,
        )