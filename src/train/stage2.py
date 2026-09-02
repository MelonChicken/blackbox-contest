from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    SEED,
    STAGE2_BACKBONE,
    STAGE2_CHECKPOINT,
    STAGE2_TRAIN_MANIFEST,
    STAGE2_VAL_MANIFEST,
)
from src.datasets.stage2_dataset import MISSING_LABEL, Stage2Dataset
from src.models.stage2_videomae import DEFAULT_STAGE2_CONFIG, Stage2VideoMAE, build_stage2_model
from src.utils import set_seed


NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.05
UNFREEZE_LAST_N = 2


def build_loaders(
    train_manifest: str | Path = STAGE2_TRAIN_MANIFEST,
    val_manifest: str | Path | None = STAGE2_VAL_MANIFEST,
) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = Stage2Dataset(train_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_manifest is not None and Path(val_manifest).exists():
        val_dataset = Stage2Dataset(val_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader


def build_optimizer(model: Stage2VideoMAE) -> torch.optim.Optimizer:
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": LR_BACKBONE})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": LR_HEAD})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.ne(MISSING_LABEL)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def compute_stage2_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    collision_loss = _masked_cross_entropy(outputs["collision_logits"], batch["collision_index"].to(outputs["collision_logits"].device))
    entry_loss = _masked_cross_entropy(outputs["entry_logits"], batch["entry_index"].to(outputs["entry_logits"].device))
    direction_loss = _masked_cross_entropy(outputs["direction_logits"], batch["direction"].to(outputs["direction_logits"].device))
    avoidance_loss = _masked_cross_entropy(outputs["avoidance_logits"], batch["avoidance"].to(outputs["avoidance_logits"].device))
    total = collision_loss + entry_loss + direction_loss + avoidance_loss
    metrics = {
        "collision_loss": float(collision_loss.detach().cpu()),
        "entry_loss": float(entry_loss.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
        "avoidance_loss": float(avoidance_loss.detach().cpu()),
    }
    return total, metrics


def train_one_epoch(model: Stage2VideoMAE, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="stage2 train"):
        video = batch["video"].to(device, non_blocking=True)
        outputs = model(video)
        loss, _ = compute_stage2_loss(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.detach().cpu())
    return running / max(1, len(loader))


@torch.inference_mode()
def evaluate(model: Stage2VideoMAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    running = 0.0
    for batch in tqdm(loader, desc="stage2 val"):
        video = batch["video"].to(device, non_blocking=True)
        outputs = model(video)
        loss, _ = compute_stage2_loss(outputs, batch)
        running += float(loss.detach().cpu())
    return running / max(1, len(loader))


def save_stage2_checkpoint(path: str | Path, model: Stage2VideoMAE, epoch: int, val_loss: float | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "epoch": int(epoch),
            "val_loss": val_loss,
        },
        path,
    )


def load_stage2_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Stage2VideoMAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError("Stage2 checkpoint must contain model_config. Retrain with src.train.stage2.fit_stage2().")
    model = Stage2VideoMAE.from_config(checkpoint["model_config"], use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def fit_stage2() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    train_loader, val_loader = build_loaders()
    config = dict(DEFAULT_STAGE2_CONFIG)
    config.update({"backbone_variant": STAGE2_BACKBONE, "num_frames": NUM_FRAMES, "image_size": IMAGE_SIZE})
    model = build_stage2_model(config, use_pretrained=True).to(device)
    model.freeze_backbone(unfreeze_last_n=UNFREEZE_LAST_N)
    optimizer = build_optimizer(model)

    best_loss = float("inf")
    for epoch in range(1, max(1, EPOCHS) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device) if val_loader is not None else train_loss
        print(f"stage2 epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_stage2_checkpoint(STAGE2_CHECKPOINT, model, epoch=epoch, val_loss=val_loss)
