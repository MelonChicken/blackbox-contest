from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode

from src.config import (
    AIHUB_STAGE1_RAW,
    PROJECT_ROOT,
)
from src.preprocessing.recapture import RecaptureTransform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FRAMES = 16

MODEL_SIZE = 224
RECAPTURE_SIZE = 320

SEED = 42

OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "tools"
    / "data"
    / "stage1"
    / "aihub597"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Video loading
# --------------------------------------------------

def frame_to_tensor(
    frame,
) -> torch.Tensor:
    """
    RGB uint8 numpy image [H, W, C]
    ->
    float Tensor [C, H, W], range [0, 1]
    """

    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        / 255.0
    )


def load_clip(
    video_path: Path,
    frames: int = FRAMES,
    recapture_size: int = RECAPTURE_SIZE,
):
    """
    AIHubStage1Dataset과 동일한 방식으로
    영상 전체에서 균등하게 frame을 선택한다.

    ORIGINAL과 RERECORDED가 서로 다른 resize path를
    가지지 않도록 하나의 shared intermediate clip만 만든다.

    Returns
    -------
    base_clip:
        Tensor [T, C, recapture_size, recapture_size]

    timestamps:
        Tensor [T]

    indices:
        실제 sampled frame index 목록

    fps:
        source video FPS
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    try:
        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            raise RuntimeError(
                f"Invalid frame count: "
                f"{video_path}"
            )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if fps <= 1e-6:
            raise RuntimeError(
                f"Invalid FPS: "
                f"{fps}"
            )

        indices = (
            torch.linspace(
                0,
                total_frames - 1,
                steps=frames,
            )
            .round()
            .long()
            .tolist()
        )

        result = []

        for frame_index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Failed to decode frame "
                    f"{frame_index}"
                )

            # BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # -----------------------------------
            # Shared intermediate representation
            #
            # ORIGINAL / RERECORDED 모두
            # 동일한 320x320 source에서 출발
            # -----------------------------------

            frame = cv2.resize(
                frame,
                (
                    recapture_size,
                    recapture_size,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            result.append(
                frame_to_tensor(
                    frame
                )
            )

        base_clip = torch.stack(
            result,
            dim=0,
        )

        timestamps = torch.tensor(
            [
                frame_index / fps
                for frame_index in indices
            ],
            dtype=torch.float32,
        )

        return (
            base_clip,
            timestamps,
            indices,
            fps,
        )

    finally:
        cap.release()


# --------------------------------------------------
# Resize to model input
# --------------------------------------------------

def resize_to_model_size(
    clip: torch.Tensor,
) -> torch.Tensor:
    """
    Dataset의 _resize_to_model_size()와 동일한 처리.

    [T, C, 320, 320]
    ->
    [T, C, 224, 224]
    """

    return TF.resize(
        clip,
        [
            MODEL_SIZE,
            MODEL_SIZE,
        ],
        interpolation=(
            InterpolationMode.BILINEAR
        ),
        antialias=True,
    )


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def save_preview(
    original: torch.Tensor,
    recaptured: torch.Tensor,
):
    frame_indices = [
        0,
        5,
        10,
        15,
    ]

    fig, axes = plt.subplots(
        3,
        len(frame_indices),
        figsize=(16, 10),
    )

    for col, idx in enumerate(
        frame_indices
    ):

        original_frame = (
            original[idx]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )

        recaptured_frame = (
            recaptured[idx]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )

        diff = (
            original[idx]
            - recaptured[idx]
        ).abs()

        # 실제 데이터에는 영향 없음.
        # 사람이 difference를 확인하기 위한 시각화용 증폭.
        diff_display = (
            diff
            * 8.0
        ).clamp(
            0,
            1,
        )

        diff_display = (
            diff_display
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        axes[0, col].imshow(
            original_frame
        )

        axes[0, col].set_title(
            f"Original #{idx}"
        )

        axes[0, col].axis(
            "off"
        )

        axes[1, col].imshow(
            recaptured_frame
        )

        axes[1, col].set_title(
            f"Recaptured #{idx}"
        )

        axes[1, col].axis(
            "off"
        )

        axes[2, col].imshow(
            diff_display
        )

        axes[2, col].set_title(
            f"Difference x8 #{idx}"
        )

        axes[2, col].axis(
            "off"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "recapture_preview.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"saved: {output_path}"
    )


# --------------------------------------------------
# Video export
# --------------------------------------------------

def tensor_frame_to_bgr(
    frame: torch.Tensor,
):
    """
    Tensor [C, H, W], RGB [0, 1]
    ->
    OpenCV BGR uint8
    """

    frame = (
        frame
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    frame = (
        frame
        * 255.0
    ).round().astype(
        "uint8"
    )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR,
    )


def save_comparison_video(
    original: torch.Tensor,
    recaptured: torch.Tensor,
):
    """
    왼쪽 ORIGINAL / 오른쪽 RERECORDED 비교 영상을 저장한다.

    display FPS는 시각화용이며,
    RecaptureTransform 자체는 실제 timestamp를 사용한다.
    """

    output_path = (
        OUTPUT_DIR
        / "recapture_comparison.mp4"
    )

    display_fps = 4.0

    width = (
        MODEL_SIZE
        * 2
    )

    height = MODEL_SIZE

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        display_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video: "
            f"{output_path}"
        )

    try:
        for i in range(
            original.shape[0]
        ):

            left = tensor_frame_to_bgr(
                original[i]
            )

            right = tensor_frame_to_bgr(
                recaptured[i]
            )

            comparison = cv2.hconcat(
                [
                    left,
                    right,
                ]
            )

            cv2.line(
                comparison,
                (
                    MODEL_SIZE,
                    0,
                ),
                (
                    MODEL_SIZE,
                    MODEL_SIZE - 1,
                ),
                (
                    255,
                    255,
                    255,
                ),
                1,
            )

            cv2.putText(
                comparison,
                "ORIGINAL",
                (
                    8,
                    22,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                comparison,
                "RERECORDED",
                (
                    MODEL_SIZE + 8,
                    22,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            writer.write(
                comparison
            )

    finally:
        writer.release()

    print(
        f"saved: {output_path}"
    )


# --------------------------------------------------
# Difference statistics
# --------------------------------------------------

def print_difference_stats(
    original: torch.Tensor,
    recaptured: torch.Tensor,
):
    diff = (
        original
        - recaptured
    ).abs()

    print()
    print(
        "=== Recapture Difference ==="
    )

    print(
        f"Mean absolute difference: "
        f"{diff.mean().item():.6f}"
    )

    print(
        f"Max absolute difference: "
        f"{diff.max().item():.6f}"
    )

    frame_mae = diff.mean(
        dim=(
            1,
            2,
            3,
        )
    )

    print(
        "Frame MAE:"
    )

    for i, value in enumerate(
        frame_mae.tolist()
    ):
        print(
            f"  #{i:02d}: "
            f"{value:.6f}"
        )


def print_difference_decomposition(
    base_clip: torch.Tensor,
    original: torch.Tensor,
    recaptured: torch.Tensor,
):
    """
    resize path 자체와 RecaptureTransform의 효과를 분리한다.

    새 pipeline에서는 ORIGINAL과 pipeline_only가
    동일해야 하므로 Resize-path-only MAE는 0이어야 한다.
    """

    pipeline_only = (
        resize_to_model_size(
            base_clip.clone()
        )
    )

    pipeline_diff = (
        original
        - pipeline_only
    ).abs()

    recapture_diff = (
        pipeline_only
        - recaptured
    ).abs()

    total_diff = (
        original
        - recaptured
    ).abs()

    print()
    print(
        "=== Difference Decomposition ==="
    )

    print(
        f"Resize-path-only MAE: "
        f"{pipeline_diff.mean().item():.6f}"
    )

    print(
        f"Recapture-effect MAE: "
        f"{recapture_diff.mean().item():.6f}"
    )

    print(
        f"Total MAE: "
        f"{total_diff.mean().item():.6f}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

video_path = (
    AIHUB_STAGE1_RAW
    / "1.Training"
    / "원천데이터_231108_add"
    / "bb_1_180313_vehicle_254_37494.mp4"
)

(
    base_clip,
    timestamps,
    sampled_indices,
    source_fps,
) = load_clip(
    video_path
)


# --------------------------------------------------
# ORIGINAL
#
# 320 shared base
# ->
# 동일한 224 resize
# --------------------------------------------------

original = (
    resize_to_model_size(
        base_clip.clone()
    )
)


# --------------------------------------------------
# RERECORDED
#
# 320 shared base
# ->
# RecaptureTransform
# ->
# 동일한 224 resize
# --------------------------------------------------

transform = RecaptureTransform(
    profile="train",
)

recaptured_base = transform(
    base_clip.clone(),
    seed=SEED,
    timestamps=timestamps,
)

recaptured = (
    resize_to_model_size(
        recaptured_base
    )
)


# --------------------------------------------------
# Information
# --------------------------------------------------

print(
    f"Source FPS: "
    f"{source_fps:.3f}"
)

print(
    "Sampled frame indices:",
    sampled_indices,
)

print(
    "Sampled timestamps:",
    [
        round(
            value,
            3,
        )
        for value
        in timestamps.tolist()
    ],
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print_difference_stats(
    original,
    recaptured,
)

print_difference_decomposition(
    base_clip,
    original,
    recaptured,
)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

save_preview(
    original,
    recaptured,
)

save_comparison_video(
    original,
    recaptured,
)