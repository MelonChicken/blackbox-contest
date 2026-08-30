import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import DEVICE, EPOCHS, SEED, STAGE2_MODEL, STAGE2_RAW
from src.models import Stage2Temporal
from src.utils import set_seed, video_frames

set_seed(SEED)


def _resnet_backbone():
    try:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        print("Warning: failed to load ImageNet weights; using weights=None.")
        model = resnet18(weights=None)
    return model


def fit_stage2():
    out = STAGE2_MODEL
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STAGE2_RAW / "labels.csv")
    backbone = _resnet_backbone()

    torch.save(backbone.state_dict(), out / "resnet18-f37072fd.pth")

    backbone.fc = nn.Identity()
    backbone.to(DEVICE).eval()

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    sequences = []

    with torch.inference_mode():
        for row in df.itertuples():
            frames = video_frames(STAGE2_RAW / row.path)
            batches = []

            for start in range(0, len(frames), 64):
                x = torch.stack(
                    [transform(Image.fromarray(frame)) for frame in frames[start:start + 64]]
                ).to(DEVICE)
                batches.append(backbone(x).float().cpu())

            sequences.append((torch.cat(batches), min(int(row.t_collision), len(frames) - 1)))

    temporal = Stage2Temporal().to(DEVICE)
    opt = torch.optim.AdamW(temporal.parameters(), 2e-4)

    for _ in range(max(1, EPOCHS)):
        temporal.train()
        for seq, target in sequences:
            collision, _, _ = temporal.logits(seq[None].to(DEVICE))
            loss = nn.functional.cross_entropy(collision, torch.tensor([target], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({"model": temporal.state_dict()}, out / "best.pt")
