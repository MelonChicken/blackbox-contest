from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import COMMA2K19_STAGE3_TRAIN_MANIFEST, DEVICE
from src.datasets.comma2k19_stage3 import Comma2k19Stage3Dataset
from src.models import Stage3MViT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=COMMA2K19_STAGE3_TRAIN_MANIFEST)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    dataset = Comma2k19Stage3Dataset(args.manifest, root=args.root)
    sample = dataset[0]
    assert sample["video"].ndim == 4
    assert 0 <= sample["accel_label"] <= 3
    assert 0 <= sample["steer_label"] <= 2

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    model = Stage3MViT().to(DEVICE).eval()
    with torch.inference_mode():
        accel, steer = model(batch["video"].to(DEVICE))
        loss = nn.functional.cross_entropy(accel, batch["accel_label"].to(DEVICE))
        loss = loss + nn.functional.cross_entropy(steer, batch["steer_label"].to(DEVICE))
    print("ok")
    print("video_shape", tuple(batch["video"].shape))
    print("accel_logits", tuple(accel.shape), "steer_logits", tuple(steer.shape), "loss", float(loss.cpu()))


if __name__ == "__main__":
    main()
