from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from src.config import AIHUB_STAGE1_RAW, PROJECT_ROOT
from src.preprocessing.recapture import RecaptureTransform


def load_clip(
    video_path: Path,
    frames: int = 16,
    size: int = 224,
):
    cap = cv2.VideoCapture(str(video_path))

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indices = torch.linspace(
        0,
        total - 1,
        steps=frames,
    ).round().long()

    result = []

    for idx in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx),
        )

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Failed to decode frame: {idx}"
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (size, size),
        )

        frame = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        result.append(frame)

    cap.release()

    return torch.stack(result)


video_path = AIHUB_STAGE1_RAW / "1.Training" / "원천데이터_231108_add" / "bb_1_180313_vehicle_254_37494.mp4"

original = load_clip(
    video_path
)

transform = RecaptureTransform()

recaptured = transform(
    original,
    seed=42,
)
frame_indices = [
    0,
    5,
    10,
    15,
]

fig, axes = plt.subplots(
    2,
    len(frame_indices),
    figsize=(16, 7),
)

for col, idx in enumerate(frame_indices):

    axes[0, col].imshow(
        original[idx]
        .permute(1, 2, 0)
        .numpy()
    )

    axes[0, col].set_title(
        f"Original #{idx}"
    )

    axes[0, col].axis("off")

    axes[1, col].imshow(
        recaptured[idx]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )

    axes[1, col].set_title(
        f"Recaptured #{idx}"
    )

    axes[1, col].axis("off")


plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "recapture_preview.png",
    dpi=150,
)

plt.close()

print(
    "saved: recapture_preview.png"
)
