from __future__ import annotations

import argparse
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
    STAGE2_MODEL,
    STAGE2_TRAIN_MANIFEST,
    STAGE2_VAL_MANIFEST,
)
from src.datasets.stage2_dataset import MISSING_LABEL, Stage2Dataset, original_frame_to_sample_index, sample_frame_indices
from src.models.stage2_videomae import DEFAULT_STAGE2_CONFIG, Stage2VideoMAE, build_stage2_from_checkpoint, build_stage2_model, stage2_config_from_checkpoint
from src.utils import set_seed

NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 2
LR_BACKBONE = 1e-5
LR_COLLISION_HEAD = 1e-4
LR_ENTRY_HEAD = 1e-4
LR_DIRECTION_HEAD = 1e-4
LR_AVOIDANCE_HEAD = 1e-4
WEIGHT_DECAY = 0.05
UNFREEZE_LAST_N = 2
TASK_ORDER = ("collision", "entry", "direction", "avoidance")
FRAME_TASKS = ("collision", "entry")
CLASSIFICATION_TASKS = ("direction", "avoidance")
ACTIVE_STAGE2_TASKS = ("collision", "direction")
LOSS_WEIGHTS = {"collision": 1.0, "entry": 1.0, "direction": 1.0}
STAGE2_INIT_CHECKPOINT: str | Path | None = STAGE2_MODEL / "archive" / "collision_only_best.pt"
STAGE2_MIN_PSEUDO_LABEL_CONFIDENCE: float | None = None
SELECTION_METRIC = "val_selection_metric"


def _active_tasks(tasks: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    selected = tuple(tasks or ACTIVE_STAGE2_TASKS)
    unknown = sorted(set(selected) - set(TASK_ORDER))
    if unknown:
        raise ValueError(f"Unknown Stage2 task(s): {unknown}")
    return selected


def _experiment_name(tasks: tuple[str, ...] | list[str] | None = None) -> str:
    return "_".join(_active_tasks(tasks))


def _checkpoint_path(kind: str, tasks: tuple[str, ...] | list[str] | None = None) -> Path:
    return STAGE2_MODEL / f"{kind}_{_experiment_name(tasks)}.pt"


STAGE2_EXPERIMENT_NAME = _experiment_name()
STAGE2_BEST_CHECKPOINT = _checkpoint_path("best")
STAGE2_LAST_CHECKPOINT = _checkpoint_path("last")


def _target_name(task: str) -> str:
    return f"{task}_index" if task in FRAME_TASKS else task


def _valid_target(target: torch.Tensor) -> torch.Tensor:
    return target.ne(MISSING_LABEL)


def split_supervision_counts(manifest_path: str | Path) -> dict[str, Any]:
    df = pd.read_csv(manifest_path)
    entry = df.get("entry_frame", pd.Series(MISSING_LABEL, index=df.index)).fillna(MISSING_LABEL).astype(int).ge(0)
    direction = df.get("direction", pd.Series(MISSING_LABEL, index=df.index)).fillna(MISSING_LABEL).astype(int).ge(0)
    direction_values = df.loc[direction, "direction"].astype(int) if "direction" in df else pd.Series(dtype=int)
    return {
        "rows": int(len(df)),
        "collision": int(len(df)),
        "entry": int(entry.sum()),
        "direction": int(direction.sum()),
        "left": int((direction_values == 0).sum()),
        "right": int((direction_values == 1).sum()),
    }


def print_stage2_task_summary(
    train_manifest: str | Path = STAGE2_TRAIN_MANIFEST,
    val_manifest: str | Path = STAGE2_VAL_MANIFEST,
    active_tasks: tuple[str, ...] | list[str] | None = None,
) -> None:
    tasks = _active_tasks(active_tasks)
    print("=== Stage2 Tasks ===")
    for task in TASK_ORDER:
        print(f"{task}: {'ON' if task in tasks else 'OFF'}")
    print("=== Stage2 Split Supervision ===")
    for name, path in (("Train", train_manifest), ("Val", val_manifest)):
        counts = split_supervision_counts(path)
        print(f"{name}:")
        print(f"  rows: {counts['rows']}")
        print(f"  entry valid: {counts['entry']}")
        print(f"  direction valid: {counts['direction']}")
        print(f"  LEFT: {counts['left']}")
        print(f"  RIGHT: {counts['right']}")


def _frame_count(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    finally:
        cap.release()


def _valid_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    column = f"{task}_frame"
    return df[df[column].fillna(MISSING_LABEL).astype(int).ge(0)].copy()


def _frame_metrics(pred_frames: list[int], target_frames: list[int]) -> dict[str, float]:
    if not target_frames:
        return {
            "mean_abs_original_frame_error": float("nan"),
            "median_abs_original_frame_error": float("nan"),
            "acc_within_1_frame": float("nan"),
            "acc_within_2_frames": float("nan"),
        }
    errors = np.abs(np.asarray(pred_frames, dtype=np.float32) - np.asarray(target_frames, dtype=np.float32))
    return {
        "mean_abs_original_frame_error": float(errors.mean()),
        "median_abs_original_frame_error": float(np.median(errors)),
        "acc_within_1_frame": float((errors <= 1).mean()),
        "acc_within_2_frames": float((errors <= 2).mean()),
    }


def _classification_metrics(pred: list[int], target: list[int], num_classes: int = 2) -> dict[str, Any]:
    if not target:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "confusion_matrix": [], "left_recall": float("nan"), "right_recall": float("nan")}
    pred_arr = np.asarray(pred, dtype=np.int64)
    target_arr = np.asarray(target, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, yhat in zip(target_arr, pred_arr):
        if 0 <= y < num_classes and 0 <= yhat < num_classes:
            matrix[y, yhat] += 1
    f1_scores = []
    for cls in range(num_classes):
        tp = matrix[cls, cls]
        fp = matrix[:, cls].sum() - tp
        fn = matrix[cls, :].sum() - tp
        denom = (2 * tp) + fp + fn
        f1_scores.append(float((2 * tp) / denom) if denom else 0.0)
    metrics = {
        "accuracy": float((pred_arr == target_arr).mean()),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": matrix.tolist(),
    }
    if num_classes == 2:
        metrics["left_recall"] = float(matrix[0, 0] / matrix[0, :].sum()) if matrix[0, :].sum() else float("nan")
        metrics["right_recall"] = float(matrix[1, 1] / matrix[1, :].sum()) if matrix[1, :].sum() else float("nan")
    return metrics


def _trivial_baseline_for_task(task: str, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, dict[str, float]]:
    column = f"{task}_frame"
    train = _valid_frame(train, task)
    val = _valid_frame(val, task)
    if train.empty or val.empty:
        return {}

    median_frame = int(round(float(train[column].astype(int).median())))
    rel_values = []
    sampled_positions = []
    for row in train.itertuples(index=False):
        n = _frame_count(row.video_path)
        if n <= 1:
            continue
        sampled = sample_frame_indices(n, NUM_FRAMES)
        frame = int(getattr(row, column))
        rel_values.append(frame / (n - 1))
        sampled_positions.append(original_frame_to_sample_index(frame, sampled))
    if not rel_values:
        return {}

    rel_mean = float(np.mean(rel_values))
    counts = pd.Series(sampled_positions).value_counts().sort_index()
    common_pos = int(counts.idxmax())
    print(f"=== train sampled {task} position distribution ===")
    for pos in range(NUM_FRAMES):
        print(f"{task}_position {pos}: {int(counts.get(pos, 0))}")

    target = []
    median_pred = []
    rel_pred = []
    sampled_pred = []
    for row in val.itertuples(index=False):
        n = _frame_count(row.video_path)
        sampled = sample_frame_indices(n, NUM_FRAMES)
        target.append(int(getattr(row, column)))
        median_pred.append(median_frame)
        rel_pred.append(int(round(rel_mean * (n - 1))))
        sampled_pred.append(int(sampled[common_pos]))

    return {
        "median": _frame_metrics(median_pred, target),
        "relative_position": _frame_metrics(rel_pred, target),
        "most_common_sampled_position": _frame_metrics(sampled_pred, target),
        "common_sampled_position": {"position": float(common_pos)},
    }


def compute_trivial_baselines(
    train_manifest: str | Path = STAGE2_TRAIN_MANIFEST,
    val_manifest: str | Path = STAGE2_VAL_MANIFEST,
    active_tasks: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    train = pd.read_csv(train_manifest)
    val = pd.read_csv(val_manifest)
    baselines = {}
    for task in _active_tasks(active_tasks):
        if task in FRAME_TASKS:
            baselines[task] = _trivial_baseline_for_task(task, train, val)
    return baselines


def build_loaders(train_manifest: str | Path = STAGE2_TRAIN_MANIFEST, val_manifest: str | Path | None = STAGE2_VAL_MANIFEST) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = Stage2Dataset(train_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, min_pseudo_label_confidence=STAGE2_MIN_PSEUDO_LABEL_CONFIDENCE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = None
    if val_manifest is not None and Path(val_manifest).exists():
        val_dataset = Stage2Dataset(val_manifest, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, min_pseudo_label_confidence=STAGE2_MIN_PSEUDO_LABEL_CONFIDENCE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def build_optimizer(model: Stage2VideoMAE) -> torch.optim.Optimizer:
    groups_by_name: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "collision_head": [],
        "entry_head": [],
        "direction_head": [],
        "avoidance_head": [],
        "other_heads": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            groups_by_name["backbone"].append(parameter)
        elif name.startswith("collision_head."):
            groups_by_name["collision_head"].append(parameter)
        elif name.startswith("entry_head."):
            groups_by_name["entry_head"].append(parameter)
        elif name.startswith("direction_head."):
            groups_by_name["direction_head"].append(parameter)
        elif name.startswith("avoidance_head."):
            groups_by_name["avoidance_head"].append(parameter)
        else:
            groups_by_name["other_heads"].append(parameter)

    lr_by_name = {
        "backbone": LR_BACKBONE,
        "collision_head": LR_COLLISION_HEAD,
        "entry_head": LR_ENTRY_HEAD,
        "direction_head": LR_DIRECTION_HEAD,
        "avoidance_head": LR_AVOIDANCE_HEAD,
        "other_heads": LR_COLLISION_HEAD,
    }
    groups = [
        {"params": params, "lr": lr_by_name[name], "name": name}
        for name, params in groups_by_name.items()
        if params
    ]
    print("[Stage 2] optimizer parameter groups:")
    for group in groups:
        print(f"  {group['name']}: lr={group['lr']} tensors={len(group['params'])}")
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.ne(MISSING_LABEL)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def compute_stage2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    active_tasks: tuple[str, ...] | list[str] | None = None,
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = loss_weights or LOSS_WEIGHTS
    losses = {}
    counts = {}
    for task in _active_tasks(active_tasks):
        logits = outputs[f"{task}_logits"]
        target = batch[_target_name(task)].to(logits.device)
        valid = _valid_target(target)
        counts[task] = int(valid.sum().detach().cpu())
        losses[task] = _masked_cross_entropy(logits, target)

    total = None
    for task, loss in losses.items():
        weighted = float(weights.get(task, 1.0)) * loss
        total = weighted if total is None else total + weighted
    if total is None:
        total = outputs["collision_logits"].sum() * 0.0

    metrics = {f"{task}_loss": float(loss.detach().cpu()) for task, loss in losses.items()}
    metrics.update({f"{task}_supervised_count": float(count) for task, count in counts.items()})
    metrics["total_loss"] = float(total.detach().cpu())
    return total, metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def train_one_epoch(
    model: Stage2VideoMAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    active_tasks: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    model.train()
    rows = []
    for batch in tqdm(loader, desc="stage2 train"):
        video = batch["video"].to(device, non_blocking=True)
        outputs = model(video)
        loss, metrics = compute_stage2_loss(outputs, batch, active_tasks=active_tasks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append(metrics)
    summary = _mean_metrics(rows)
    for task in _active_tasks(active_tasks):
        key = f"{task}_supervised_count"
        summary[key] = float(sum(row.get(key, 0.0) for row in rows))
    return summary


@torch.inference_mode()
def evaluate(
    model: Stage2VideoMAE,
    loader: DataLoader,
    device: torch.device,
    active_tasks: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    tasks = _active_tasks(active_tasks)
    model.eval()
    loss_rows = []
    frame_rows = {task: {"pred": [], "target": [], "positions": []} for task in tasks if task in FRAME_TASKS}
    class_rows = {task: {"pred": [], "target": []} for task in tasks if task in CLASSIFICATION_TASKS}
    for batch in tqdm(loader, desc="stage2 val"):
        video = batch["video"].to(device, non_blocking=True)
        outputs = model(video)
        _, loss_metrics = compute_stage2_loss(outputs, batch, active_tasks=tasks)
        loss_rows.append(loss_metrics)

        for task, rows in frame_rows.items():
            pred_idx = outputs[f"{task}_logits"].argmax(dim=1).cpu()
            target = batch[f"{task}_frame"].cpu()
            for i, pos in enumerate(pred_idx.tolist()):
                if int(target[i]) == MISSING_LABEL:
                    continue
                sampled = batch["sampled_indices"][i].numpy()
                rows["positions"].append(int(pos))
                rows["pred"].append(int(sampled[pos]))
                rows["target"].append(int(target[i]))

        for task, rows in class_rows.items():
            pred = outputs[f"{task}_logits"].argmax(dim=1).cpu()
            target = batch[task].cpu()
            for yhat, y in zip(pred.tolist(), target.tolist()):
                if int(y) == MISSING_LABEL:
                    continue
                rows["pred"].append(int(yhat))
                rows["target"].append(int(y))

    metrics = {f"val_{key}": value for key, value in _mean_metrics(loss_rows).items()}
    for task in tasks:
        key = f"{task}_supervised_count"
        metrics[f"val_{key}"] = float(sum(row.get(key, 0.0) for row in loss_rows))
    for task, rows in frame_rows.items():
        for name, value in _frame_metrics(rows["pred"], rows["target"]).items():
            metrics[f"val_{task}_{name}"] = value
        counts = pd.Series(rows["positions"]).value_counts().sort_index()
        for pos in range(NUM_FRAMES):
            metrics[f"val_{task}_pred_pos_{pos}"] = float(counts.get(pos, 0))

    for task, rows in class_rows.items():
        for name, value in _classification_metrics(rows["pred"], rows["target"]).items():
            metrics[f"val_{task}_{name}"] = value

    metrics[SELECTION_METRIC] = selection_metric(metrics, tasks)
    return metrics


def selection_metric(metrics: dict[str, Any], active_tasks: tuple[str, ...] | list[str] | None = None) -> float:
    tasks = _active_tasks(active_tasks)
    collision_mae = metrics.get("val_collision_mean_abs_original_frame_error")
    if collision_mae is not None and (tasks == ("collision",) or tasks == ("collision", "direction")):
        return float(collision_mae)

    values = []
    for task in tasks:
        if task in FRAME_TASKS:
            value = float(metrics[f"val_{task}_mean_abs_original_frame_error"])
            if not np.isnan(value):
                values.append(value)
        elif task in CLASSIFICATION_TASKS:
            accuracy = float(metrics[f"val_{task}_accuracy"])
            if not np.isnan(accuracy):
                values.append(1.0 - accuracy)
    return float(np.sum(values)) if values else float("inf")


def _checkpoint_payload(
    model: Stage2VideoMAE,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    active_tasks: tuple[str, ...],
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "num_frames": model.num_frames,
        "epoch": int(epoch),
        "val_frame_mae": metrics.get("val_collision_mean_abs_original_frame_error"),
        "active_tasks": list(active_tasks),
        "active_stage2_tasks": list(active_tasks),
        "loss_weights": dict(loss_weights),
        "val_collision_metrics": {key: value for key, value in metrics.items() if key.startswith("val_collision_")},
        "val_entry_metrics": {key: value for key, value in metrics.items() if key.startswith("val_entry_")},
        "val_direction_metrics": {key: value for key, value in metrics.items() if key.startswith("val_direction_")},
        "selection_metric_name": SELECTION_METRIC,
        "selection_metric": metrics.get(SELECTION_METRIC),
        "metrics": metrics,
        "history": history,
    }


def save_stage2_checkpoint(
    path: str | Path,
    model: Stage2VideoMAE,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    active_tasks: tuple[str, ...] | list[str] | None = None,
    loss_weights: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    if "archive" in path.parts:
        raise RuntimeError(f"Refusing to write Stage2 checkpoint under archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = _active_tasks(active_tasks)
    torch.save(_checkpoint_payload(model, epoch, metrics, history, tasks, loss_weights or LOSS_WEIGHTS), path)


def load_stage2_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Stage2VideoMAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_stage2_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def _mapped_init_state_dict(model: Stage2VideoMAE, checkpoint: dict[str, Any]) -> tuple[dict[str, torch.Tensor], list[str]]:
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    mapped = {}
    skipped = []
    allowed_prefixes = ("backbone.", "encoder.", "collision_head.")
    for key, value in source.items():
        if not key.startswith(allowed_prefixes):
            skipped.append(key)
            continue
        candidates = [key]
        if key.startswith("encoder."):
            candidates.append("backbone." + key.removeprefix("encoder."))
        if key.startswith("collision_head.1."):
            candidates.append("collision_head." + key.removeprefix("collision_head.1."))
        for candidate in candidates:
            if candidate in target and tuple(target[candidate].shape) == tuple(value.shape):
                mapped[candidate] = value
                break
        else:
            skipped.append(key)
    return mapped, skipped


def init_stage2_from_checkpoint(model: Stage2VideoMAE, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mapped, skipped = _mapped_init_state_dict(model, checkpoint)
    result = model.load_state_dict(mapped, strict=False)
    expected_missing = {"entry_head.weight", "entry_head.bias", "direction_head.weight", "direction_head.bias", "avoidance_head.weight", "avoidance_head.bias"}
    unexpected_missing = sorted(set(result.missing_keys) - expected_missing)
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing keys when initializing Stage2 from {checkpoint_path}: {unexpected_missing}")
    loaded_backbone = sum(key.startswith("backbone.") for key in mapped)
    loaded_collision = sum(key.startswith("collision_head.") for key in mapped)
    print(f"[Stage 2] init_checkpoint={checkpoint_path}")
    print(f"[Stage 2] Loaded backbone tensors: {loaded_backbone}")
    print(f"[Stage 2] Loaded collision head tensors: {loaded_collision}")
    print("[Stage 2] Random/new: entry head, direction head")
    print(f"[Stage 2] Skipped legacy tensors: {len(skipped)}")
    print(f"[Stage 2] missing_after_init={list(result.missing_keys)}")
    print(f"[Stage 2] skipped_init_keys={skipped[:20]}{' ...' if len(skipped) > 20 else ''}")


def build_training_model(init_checkpoint: str | Path | None = STAGE2_INIT_CHECKPOINT) -> Stage2VideoMAE:
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        config = dict(DEFAULT_STAGE2_CONFIG)
        config.update(stage2_config_from_checkpoint(checkpoint))
        model = Stage2VideoMAE.from_config(config, use_pretrained=False)
        init_stage2_from_checkpoint(model, init_checkpoint)
        return model
    config = dict(DEFAULT_STAGE2_CONFIG)
    config.update({"backbone_variant": STAGE2_BACKBONE, "num_frames": NUM_FRAMES, "image_size": IMAGE_SIZE})
    return build_stage2_model(config, use_pretrained=True)


def _print_metrics(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, Any], active_tasks: tuple[str, ...]) -> None:
    print(f"[Stage 2] Epoch {epoch}/{max(1, EPOCHS)} active_tasks={'+'.join(active_tasks)}")
    print("Train supervised:")
    for task in active_tasks:
        print(f"  {task}: {int(train_metrics.get(f'{task}_supervised_count', 0))}")
    print("Val supervised:")
    for task in active_tasks:
        print(f"  {task}: {int(val_metrics.get(f'val_{task}_supervised_count', 0))}")
    for key in ("total_loss", "collision_loss", "entry_loss", "direction_loss", "avoidance_loss"):
        if key in train_metrics:
            print(f"train_{key}={train_metrics[key]:.5f}")
    for task in active_tasks:
        if task in FRAME_TASKS:
            print(f"val_{task}_mean_abs_original_frame_error={val_metrics[f'val_{task}_mean_abs_original_frame_error']:.5f}")
            print(f"val_{task}_median_abs_original_frame_error={val_metrics[f'val_{task}_median_abs_original_frame_error']:.5f}")
            print(f"val_{task}_acc_within_1_frame={val_metrics[f'val_{task}_acc_within_1_frame']:.5f}")
            print(f"val_{task}_acc_within_2_frames={val_metrics[f'val_{task}_acc_within_2_frames']:.5f}")
        else:
            print(f"val_{task}_accuracy={val_metrics[f'val_{task}_accuracy']:.5f}")
            print(f"val_{task}_macro_f1={val_metrics[f'val_{task}_macro_f1']:.5f}")
            print(f"val_{task}_confusion_matrix={val_metrics[f'val_{task}_confusion_matrix']}")
            if task == "direction":
                print(f"val_{task}_left_recall={val_metrics[f'val_{task}_left_recall']:.5f}")
                print(f"val_{task}_right_recall={val_metrics[f'val_{task}_right_recall']:.5f}")
    print(f"val_selection_metric={val_metrics[SELECTION_METRIC]:.5f}")


def fit_stage2(active_tasks: tuple[str, ...] | list[str] | None = None) -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    active_tasks = _active_tasks(active_tasks)
    best_checkpoint = _checkpoint_path("best", active_tasks)
    last_checkpoint = _checkpoint_path("last", active_tasks)
    print(f"[Stage 2] experiment={_experiment_name(active_tasks)}")
    print(f"[Stage 2] best_checkpoint={best_checkpoint}")
    print(f"[Stage 2] last_checkpoint={last_checkpoint}")
    print_stage2_task_summary(active_tasks=active_tasks)

    baselines = compute_trivial_baselines(active_tasks=active_tasks)
    print("=== trivial baselines ===")
    for task, task_baselines in baselines.items():
        print(f"{task.title()} trivial baseline")
        for name, values in task_baselines.items():
            print(name, values)

    train_loader, val_loader = build_loaders()
    model = build_training_model().to(device)
    model.freeze_backbone(unfreeze_last_n=UNFREEZE_LAST_N)
    optimizer = build_optimizer(model)

    best_value = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(1, EPOCHS) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, active_tasks=active_tasks)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, active_tasks=active_tasks)
        else:
            val_metrics = {"val_total_loss": train_metrics["total_loss"], SELECTION_METRIC: train_metrics["total_loss"]}
        row = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}, **val_metrics}
        history.append(row)
        _print_metrics(epoch, train_metrics, val_metrics, active_tasks)
        save_stage2_checkpoint(last_checkpoint, model, epoch=epoch, metrics=val_metrics, history=history, active_tasks=active_tasks)
        if float(val_metrics[SELECTION_METRIC]) < best_value:
            best_value = float(val_metrics[SELECTION_METRIC])
            save_stage2_checkpoint(best_checkpoint, model, epoch=epoch, metrics=val_metrics, history=history, active_tasks=active_tasks)
            print(f"saved best checkpoint: {best_checkpoint} {SELECTION_METRIC}={best_value:.5f}")


def _parse_tasks(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage2 VideoMAE multi-task ablations.")
    parser.add_argument("--tasks", default=",".join(ACTIVE_STAGE2_TASKS), help="Comma-separated tasks, e.g. collision,direction")
    parser.add_argument("--print-split", action="store_true", help="Only print Stage2 task/split supervision counts.")
    args = parser.parse_args()
    tasks = _parse_tasks(args.tasks)
    if args.print_split:
        print_stage2_task_summary(active_tasks=tasks)
        return
    fit_stage2(active_tasks=tasks)


if __name__ == "__main__":
    main()
