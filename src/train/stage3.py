import pandas as pd
import torch
from torch import nn

from src.config import DEVICE, EPOCHS, S3_MEAN, S3_STD, SEED, STAGE3_MODEL, STAGE3_RAW
from src.models import Stage3MViT
from src.utils import clip, set_seed

set_seed(SEED)


def fit_stage3():
    out = STAGE3_MODEL
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STAGE3_RAW / "labels.csv")
    amap = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
    smap = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}

    model = Stage3MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)

    for _ in range(EPOCHS):
        model.train()
        for row in df.itertuples():
            x, _ = clip(STAGE3_RAW / "videos" / f"{row.ID}.mp4", 16, int(row.frame_index))
            x = (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]
            accel, steer = model(x[None].to(DEVICE))
            loss = nn.functional.cross_entropy(
                accel,
                torch.tensor([amap[row.accel_label]], device=DEVICE),
            )
            loss += nn.functional.cross_entropy(
                steer,
                torch.tensor([smap[row.steer_label]], device=DEVICE),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({"model": model.state_dict()}, out / "best.pt")
