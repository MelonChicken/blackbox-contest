from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader

from src import config as C
from src.datasets.stage3_tartanvo_pose import Stage3TartanFeatureDataset
from src.train.stage3 import (
    _build_stage3_model,
    _class_weights,
    _loss,
    _model_outputs,
    _stage3_collate,
)


def _train_dataset() -> Stage3TartanFeatureDataset:
    return Stage3TartanFeatureDataset(
        "train",
        C.STAGE3_TARTANVO_FEATURE,
        root=C.STAGE3_TARTANVO_FEATURE_CACHE,
        dataset="comma2k19",
        limit=C.STAGE3_COMMA_TRAIN_SAMPLE_LIMIT,
    )


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Stage3 model has no trainable parameters.")
    return torch.optim.AdamW(trainable, C.STAGE3_HEAD_LR)


def _run_candidate(dataset, batch_size: int, batches: int, warmup: int, safety: float) -> dict:
    device = torch.device(C.DEVICE)
    if device.type != "cuda":
        raise RuntimeError("cached batch benchmark requires CUDA")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=C.STAGE3_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=C.STAGE3_NUM_WORKERS > 0,
        collate_fn=_stage3_collate,
    )
    model = _build_stage3_model(pretrained=True).to(device).train()
    opt = _optimizer(model)
    accel_weight = _class_weights("accel")
    steer_weight = _class_weights("steer")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    measured_batches = measured_samples = 0
    start = None

    try:
        for step, batch in enumerate(loader):
            if step >= warmup + batches:
                break
            if step == warmup:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
            accel, steer = _model_outputs(model, batch)
            loss, _, _ = _loss(accel, steer, batch, accel_weight, steer_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step >= warmup:
                measured_batches += 1
                measured_samples += int(batch["accel_label"].numel())
    except torch.cuda.OutOfMemoryError:
        del model, opt, loader
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "oom": True}

    torch.cuda.synchronize(device)
    elapsed = max(1e-9, time.perf_counter() - start) if start is not None else 0.0
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    _, total = torch.cuda.mem_get_info(device)
    batches_sec = measured_batches / elapsed if elapsed else 0.0
    samples_sec = measured_samples / elapsed if elapsed else 0.0
    iters = math.ceil(len(dataset) / batch_size)

    del model, opt, loader
    torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "oom": False,
        "batches": measured_batches,
        "samples": measured_samples,
        "seconds": elapsed,
        "batches_sec": batches_sec,
        "samples_sec": samples_sec,
        "peak_alloc_gb": peak_alloc / 1024**3,
        "peak_reserved_gb": peak_reserved / 1024**3,
        "total_gb": total / 1024**3,
        "safe": peak_reserved <= total * safety,
        "iters_per_epoch": iters,
        "epoch_seconds": iters / batches_sec if batches_sec else float("inf"),
    }


def _fmt_time(seconds: float) -> str:
    if seconds == float("inf"):
        return "inf"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:04.1f}"


def _print_result(row: dict) -> None:
    if row.get("oom"):
        print(f"bs={row['batch_size']:<3} OOM")
        return
    print(
        f"bs={row['batch_size']:<3} "
        f"peak_alloc={row['peak_alloc_gb']:.2f}GB "
        f"peak_reserved={row['peak_reserved_gb']:.2f}/{row['total_gb']:.2f}GB "
        f"batches/sec={row['batches_sec']:.2f} "
        f"samples/sec={row['samples_sec']:.2f} "
        f"iters/epoch={row['iters_per_epoch']} "
        f"epoch?{_fmt_time(row['epoch_seconds'])} "
        f"safe={row['safe']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cached Stage3 TartanVO batch sizes with real forward/backward.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--safety", type=float, default=0.85, help="max safe fraction of total VRAM reserved")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dataset = _train_dataset()
    print("[Stage3 cached batch benchmark]")
    print(f"sample_profile={C.STAGE3_SAMPLE_PROFILE}")
    print(f"train_samples={len(dataset)}")
    print(f"feature_shape=[15, 1536]")
    print(f"mode={C.STAGE3_TARTANVO_MODE} unfreeze={C.STAGE3_TARTANVO_UNFREEZE}")
    print(f"num_workers={C.STAGE3_NUM_WORKERS} measured_batches={args.batches} warmup={args.warmup}")

    results = []
    for batch_size in args.batch_sizes:
        row = _run_candidate(dataset, batch_size, args.batches, args.warmup, args.safety)
        results.append(row)
        _print_result(row)

    viable = [row for row in results if not row.get("oom") and row.get("safe")]
    if not viable:
        print("recommendation: no safe batch size found; keep BATCH_SIZE=2 or lower --safety after inspecting VRAM")
        return
    best = max(viable, key=lambda row: row["samples_sec"])
    print(f"recommendation: BATCH_SIZE={best['batch_size']} (fastest safe, peak_reserved={best['peak_reserved_gb']:.2f}GB)")


if __name__ == "__main__":
    main()