from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CCD_STAGE2_VIDEOMAE_TRAIN_MANIFEST,
    CCD_STAGE2_VIDEOMAE_VAL_MANIFEST,
    DEVICE,
    SEED,
    STAGE2_VIDEOMAE_CHECKPOINT,
    STAGE2_VIDEOMAE_MODEL,
)

from src.datasets.ccddataset import (
    CCDStage2VideoMAEDataset,
)

from src.models.stage2 import (
    Stage2VideoMAE,
)

from src.utils import set_seed


# ============================================================
# Configuration
# ============================================================

set_seed(SEED)

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 1
NUM_WORKERS = 2

EPOCHS = 8

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4

WEIGHT_DECAY = 0.05

UNFREEZE_LAST_N = 2

SIDE_LOSS_WEIGHT = 0.0

TRAIN_MANIFEST = CCD_STAGE2_VIDEOMAE_TRAIN_MANIFEST

VAL_MANIFEST = CCD_STAGE2_VIDEOMAE_VAL_MANIFEST
CHECKPOINT_DIR = STAGE2_VIDEOMAE_MODEL

CHECKPOINT_PATH = STAGE2_VIDEOMAE_CHECKPOINT

# ============================================================
# Dataloader
# ============================================================

def build_loaders():
    train_dataset = (
        CCDStage2VideoMAEDataset(
            manifest_path=(
                TRAIN_MANIFEST
            ),
            num_frames=NUM_FRAMES,
            image_size=IMAGE_SIZE,
        )
    )

    val_dataset = (
        CCDStage2VideoMAEDataset(
            manifest_path=(
                VAL_MANIFEST
            ),
            num_frames=NUM_FRAMES,
            image_size=IMAGE_SIZE,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model: Stage2VideoMAE,
):
    backbone_parameters = []

    head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if name.startswith(
            "encoder."
        ):
            backbone_parameters.append(
                parameter
            )
        else:
            head_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    backbone_parameters
                ),
                "lr": LR_BACKBONE,
            },
            {
                "params": (
                    head_parameters
                ),
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# ============================================================
# Loss
# ============================================================

def compute_loss(
    outputs: dict,
    batch: dict,
    collision_criterion,
    side_criterion,
):
    collision_logits = (
        outputs[
            "collision_logits"
        ]
    )

    collision_target = (
        batch[
            "collision_index"
        ]
    )

    collision_loss = (
        collision_criterion(
            collision_logits,
            collision_target,
        )
    )

    total_loss = collision_loss

    side_loss_value = torch.tensor(
        0.0,
        device=collision_logits.device,
    )

    # --------------------------------------------------------
    # Side auxiliary loss
    # --------------------------------------------------------

    if SIDE_LOSS_WEIGHT > 0:
        side_valid = (
            batch["side_valid"]
        )

        if side_valid.any():
            side_logits = (
                outputs[
                    "side_logits"
                ][side_valid]
            )

            side_target = (
                batch[
                    "entry_side"
                ][side_valid]
            )

            side_weight = (
                batch[
                    "side_weight"
                ][side_valid]
            )

            raw_side_loss = (
                side_criterion(
                    side_logits,
                    side_target,
                )
            )

            side_loss_value = (
                raw_side_loss
                * side_weight
            ).mean()

            total_loss = (
                total_loss
                + SIDE_LOSS_WEIGHT
                * side_loss_value
            )

    return (
        total_loss,
        collision_loss,
        side_loss_value,
    )


# ============================================================
# Train epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    collision_criterion,
    side_criterion,
):
    model.train()

    running_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Train",
    ):
        video = (
            batch["video"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        for key in [
            "collision_index",
            "entry_side",
            "side_valid",
            "side_weight",
        ]:
            batch[key] = (
                batch[key]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            video
        )

        (
            loss,
            _,
            _,
        ) = compute_loss(
            outputs,
            batch,
            collision_criterion,
            side_criterion,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

    return (
        running_loss
        / len(loader)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
):
    model.eval()

    sampled_errors = []
    original_errors = []

    predictions = []
    targets = []

    for batch in tqdm(
        loader,
        desc="Validation",
    ):
        video = (
            batch["video"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        outputs = model(
            video
        )

        collision_logits = (
            outputs[
                "collision_logits"
            ]
        )

        pred_index = (
            collision_logits
            .argmax(dim=1)
            .cpu()
        )

        target_index = (
            batch[
                "collision_index"
            ]
        )

        sampled_indices = (
            batch[
                "sampled_indices"
            ]
        )

        collision_frame = (
            batch[
                "collision_frame"
            ]
        )

        # --------------------------------------------
        # Error in sampled 16-frame coordinates
        # --------------------------------------------

        sampled_error = (
            pred_index
            - target_index
        ).abs()

        sampled_errors.extend(
            sampled_error.tolist()
        )

        # --------------------------------------------
        # Convert prediction back to original CCD frame
        # --------------------------------------------

        batch_indices = torch.arange(
            pred_index.shape[0]
        )

        pred_original = (
            sampled_indices[
                batch_indices,
                pred_index,
            ]
        )

        original_error = (
            pred_original
            - collision_frame
        ).abs()

        original_errors.extend(
            original_error.tolist()
        )

        predictions.extend(
            pred_original.tolist()
        )

        targets.extend(
            collision_frame.tolist()
        )

    sampled_mae = float(
        np.mean(
            sampled_errors
        )
    )

    original_frame_mae = float(
        mean_absolute_error(
            targets,
            predictions,
        )
    )

    return {
        "sampled_mae": (
            sampled_mae
        ),

        "original_frame_mae": (
            original_frame_mae
        ),
    }


# ============================================================
# Fit
# ============================================================

def fit_stage2():
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_loader,
        val_loader,
    ) = build_loaders()

    print(
        "=== Stage 2 VideoMAE ==="
    )

    print(
        f"Train videos: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val videos: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Frames: {NUM_FRAMES}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    model = Stage2VideoMAE(
        num_input_frames=(
            NUM_FRAMES
        )
    )

    model.freeze_backbone(
        unfreeze_last_n=(
            UNFREEZE_LAST_N
        )
    )

    model = model.to(
        DEVICE
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,} / "
        f"{total:,}"
    )

    optimizer = (
        build_optimizer(
            model
        )
    )

    collision_criterion = (
        nn.CrossEntropyLoss()
    )

    # reduction none:
    # pseudo-label confidence weight 적용 가능
    side_criterion = (
        nn.CrossEntropyLoss(
            reduction="none"
        )
    )

    best_mae = float(
        "inf"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        train_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                collision_criterion=(
                    collision_criterion
                ),
                side_criterion=(
                    side_criterion
                ),
            )
        )

        metrics = validate(
            model,
            val_loader,
        )

        print(
            f"[Stage 2] "
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"sample_mae="
            f"{metrics['sampled_mae']:.3f} | "
            f"frame_mae="
            f"{metrics['original_frame_mae']:.3f}"
        )

        current_mae = (
            metrics[
                "original_frame_mae"
            ]
        )

        if current_mae < best_mae:
            best_mae = (
                current_mae
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "num_frames": (
                        NUM_FRAMES
                    ),

                    "videomae_config": (
                        model.encoder.config.to_dict()
                    ),

                    "epoch": epoch,

                    "val_frame_mae": (
                        current_mae
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "[Stage 2] "
                "Best model updated: "
                f"{best_mae:.3f}"
            )


if __name__ == "__main__":
    fit_stage2()
