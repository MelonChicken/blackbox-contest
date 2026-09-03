from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
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
from src.datasets.stage2_dataset import MISSING_LABEL, Stage2Dataset, original_frame_to_sample_index, sample_frame_indices
from src.models.stage2_videomae import DEFAULT_STAGE2_CONFIG, Stage2VideoMAE, build_stage2_model, build_stage2_from_checkpoint
from src.utils import set_seed

NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.05
UNFREEZE_LAST_N = 2
BEST_METRIC = "val_mean_abs_original_frame_error"


def _frame_count(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    finally:
        cap.release()


def _valid_collision(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["collision_frame"].fillna(MISSING_LABEL).astype(int).ge(0)].copy()


def _row_sampled_indices(row) -> np.ndarray:
    frame_count = _frame_count(row.video_path)
    if frame_count <= 0:
        raise RuntimeError(f"Cannot read frame count: {row.video_path}")
    return sample_frame_indices(frame_count, NUM_FRAMES)


def _metrics(pred_frames: list[int], target_frames: list[int]) -> dict[str, float]:
    errors = np.abs(np.asarray(pred_frames, dtype=np.float32) - np.asarray(target_frames, dtype=np.float32))
    return {
        "mean_abs_frame_error": float(errors.mean()),
        "median_abs_frame_error": float(np.median(errors)),
        "acc_within_1_frame": float((errors <= 1).mean()),
        "acc_within_2_frame": float((errors <= 2).mean()),
    }


def compute_trivial_baselines(train_manifest: str | Path = STAGE2_TRAIN_MANIFEST, val_manifest: str | Path = STAGE2_VAL_MANIFEST) -> dict[str, dict[str, float]]:
    train = _valid_collision(pd.read_csv(train_manifest))
    val = _valid_collision(pd.read_csv(val_manifest))
    train_frames = train["collision_frame"].astype(int)

    median_frame = int(round(float(train_frames.median())))
    rel_values = []
    sampled_positions = []
    for row in train.itertuples(index=False):
        n = _frame_count(row.video_path)
        if n <= 1:
            continue
        sampled = sample_frame_indices(n, NUM_FRAMES)
        rel_values.append(int(row.collision_frame) / (n - 1))
        sampled_positions.append(original_frame_to_sample_index(int(row.collision_frame), sampled))

    rel_mean = float(np.mean(rel_values))
    counts = pd.Series(sampled_positions).value_counts().sort_index()
    common_pos = int(counts.idxmax())
    print("=== train sampled collision position distribution ===")
    for pos in range(NUM_FRAMES):
        print(f"position {pos}: {int(counts.get(pos, 0))}")

    target = []
    median_pred = []
    rel_pred = []
    sampled_pred = []
    for row in val.itertuples(index=False):
        n = _frame_count(row.video_path)
        sampled = sample_frame_indices(n, NUM_FRAMES)
        target.append(int(row.collision_frame))
        median_pred.append(median_frame)
        rel_pred.append(int(round(rel_mean * (n - 1))))
        sampled_pred.append(int(sampled[common_pos]))

    return {
        "median": _metrics(median_pred, target),
        "relative_position": _metrics(rel_pred, target),
        "most_common_sampled_position": _metrics(sampled_pred, target),
        "common_sampled_position": {"position": float(common_pos)},
    }


def build_loaders(train_manifest: str | Path = STAGE2_TRAIN_MANIFEST, val_manifest: str | Path | None = STAGE2_VAL_MANIFEST) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = Stage2Dataset(train_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = None
    if val_manifest is not None and Path(val_manifest).exists():
        val_dataset = Stage2Dataset(val_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
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


def compute_stage2_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["collision_logits"]
    target = batch["collision_index"].to(logits.device)
    valid = target.ne(MISSING_LABEL)
    if not bool(valid.any()):
        loss = logits.sum() * 0.0
    else:
        loss = F.cross_entropy(logits[valid], target[valid])
    return loss, {"collision_loss": float(loss.detach().cpu())}


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
def evaluate(model: Stage2VideoMAE, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    pred_frames = []
    target_frames = []
    pred_positions = []
    for batch in tqdm(loader, desc="stage2 val"):
        video = batch["video"].to(device, non_blocking=True)
        outputs = model(video)
        loss, _ = compute_stage2_loss(outputs, batch)
        losses.append(float(loss.detach().cpu()))
        pred_idx = outputs["collision_logits"].argmax(dim=1).cpu()
        for i, pos in enumerate(pred_idx.tolist()):
            sampled = batch["sampled_indices"][i].numpy()
            pred_positions.append(int(pos))
            pred_frames.append(int(sampled[pos]))
            target_frames.append(int(batch["collision_frame"][i]))
    metrics = _metrics(pred_frames, target_frames)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    counts = pd.Series(pred_positions).value_counts().sort_index()
    for pos in range(NUM_FRAMES):
        metrics[f"pred_pos_{pos}"] = float(counts.get(pos, 0))
    return metrics


def save_stage2_checkpoint(path: str | Path, model: Stage2VideoMAE, epoch: int, metrics: dict[str, float], history: list[dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.get_config(), "num_frames": model.num_frames, "epoch": int(epoch), "val_frame_mae": metrics["mean_abs_frame_error"]}, path)


def load_stage2_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Stage2VideoMAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_stage2_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def fit_stage2() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    baselines = compute_trivial_baselines()
    print("=== trivial baselines ===")
    for name, values in baselines.items():
        print(name, values)

    train_loader, val_loader = build_loaders()
    config = dict(DEFAULT_STAGE2_CONFIG)
    config.update({"backbone_variant": STAGE2_BACKBONE, "num_frames": NUM_FRAMES, "image_size": IMAGE_SIZE})
    model = build_stage2_model(config, use_pretrained=True).to(device)
    model.freeze_backbone(unfreeze_last_n=UNFREEZE_LAST_N)
    optimizer = build_optimizer(model)

    best_mae = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, EPOCHS) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {"mean_abs_frame_error": train_loss, "loss": train_loss}
        row = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(f"stage2 epoch={epoch} train_loss={train_loss:.5f} val_mae={val_metrics['mean_abs_frame_error']:.5f} val_median={val_metrics['median_abs_frame_error']:.5f} acc1={val_metrics['acc_within_1_frame']:.5f} acc2={val_metrics['acc_within_2_frame']:.5f}")
        if val_metrics["mean_abs_frame_error"] < best_mae:
            best_mae = val_metrics["mean_abs_frame_error"]
            save_stage2_checkpoint(STAGE2_CHECKPOINT, model, epoch=epoch, metrics=val_metrics, history=history)
            print(f"saved best checkpoint: {STAGE2_CHECKPOINT} {BEST_METRIC}={best_mae:.5f}")

