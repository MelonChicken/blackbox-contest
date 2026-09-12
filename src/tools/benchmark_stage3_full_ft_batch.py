from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C
from src.models import Stage3TartanVOGRU


class SyntheticRawStage3Dataset(Dataset):
    def __init__(self, length: int):
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {
            "video": torch.rand(3, C.STAGE3_NUM_FRAMES, C.SIZE, C.SIZE),
            "accel_label": index % 4,
            "steer_label": index % 3,
        }
from src.train.stage3 import _class_weights, _datasets, _loss, _model_outputs, _stage3_collate


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    tartan = [p for p in model.tartanvo.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("tartanvo.")]
    return torch.optim.AdamW([
        {"params": tartan, "lr": C.STAGE3_TARTANVO_LR},
        {"params": head, "lr": C.STAGE3_HEAD_LR},
    ])


def _run_candidate(dataset, val_samples: int, batch_size: int, batches: int, warmup: int, safety: float) -> dict:
    device = torch.device(C.DEVICE)
    if device.type != "cuda":
        raise RuntimeError("full FT batch benchmark requires CUDA")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=_stage3_collate)
    model = Stage3TartanVOGRU(load_pretrained=True).to(device).train()
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
    samples_sec = measured_samples / elapsed if elapsed else 0.0
    sec_iter = elapsed / measured_batches if measured_batches else float("inf")
    del model, opt, loader
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "oom": False,
        "samples_sec": samples_sec,
        "sec_iter": sec_iter,
        "peak_alloc_gb": peak_alloc / 1024**3,
        "peak_reserved_gb": peak_reserved / 1024**3,
        "total_gb": total / 1024**3,
        "safe": peak_reserved <= total * safety,
        "train_epoch_seconds": math.ceil(len(dataset) / batch_size) * sec_iter,
        "val_seconds": math.ceil(val_samples / batch_size) * sec_iter,
    }


def _fmt_time(seconds: float) -> str:
    if seconds == float("inf"):
        return "inf"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:04.1f}"


def _print(row: dict) -> None:
    if row.get("oom"):
        print(f"bs={row['batch_size']:<2} OOM")
        return
    print(
        f"bs={row['batch_size']:<2} "
        f"peak_alloc={row['peak_alloc_gb']:.2f}GB "
        f"peak_reserved={row['peak_reserved_gb']:.2f}/{row['total_gb']:.2f}GB "
        f"samples/sec={row['samples_sec']:.2f} "
        f"sec/iter={row['sec_iter']:.3f} "
        f"train_epoch={_fmt_time(row['train_epoch_seconds'])} "
        f"val={_fmt_time(row['val_seconds'])} "
        f"safe={row['safe']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage3 full TartanVO fine-tuning batch sizes.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--safety", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if C.STAGE3_TARTANVO_MODE != "finetune" or C.STAGE3_TARTANVO_UNFREEZE != "full":
        raise RuntimeError('set STAGE3_TARTANVO_MODE="finetune" and STAGE3_TARTANVO_UNFREEZE="full"')

    try:
        train, val, _ = _datasets()
        val_samples = sum(len(v) for v in val.values())
        data_source = "raw dataset"
    except FileNotFoundError as exc:
        train = SyntheticRawStage3Dataset(C.STAGE3_TRAIN_SAMPLE_LIMIT)
        val_samples = C.STAGE3_VAL_SAMPLE_LIMIT
        data_source = f"synthetic raw video ({exc})"
    print("[Stage3 full FT batch benchmark]")
    print(f"profile={C.STAGE3_SAMPLE_PROFILE} train_samples={len(train)} val_samples={val_samples}")
    print(f"data={data_source}")
    print(f"mode={C.STAGE3_TARTANVO_MODE} unfreeze={C.STAGE3_TARTANVO_UNFREEZE}")
    results = []
    for batch_size in args.batch_sizes:
        row = _run_candidate(train, val_samples, batch_size, args.batches, args.warmup, args.safety)
        results.append(row)
        _print(row)
    viable = [r for r in results if not r.get("oom") and r["safe"]]
    if viable:
        best = max(viable, key=lambda r: r["samples_sec"])
        print(f"recommendation: BATCH_SIZE={best['batch_size']} (fastest safe)")
    else:
        print("recommendation: no safe batch size found")


if __name__ == "__main__":
    main()