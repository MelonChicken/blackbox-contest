from __future__ import annotations

import json
import random

import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DEVICE, STAGE3_TARTANVO_FEATURE, STAGE3_TARTANVO_FEATURE_CACHE, TARTANVO_CHECKPOINT
from src.datasets.comma2k19_stage3 import ACCEL_TO_ID, STEER_TO_ID
from src.models import Stage3TartanVOGRU
from src.train.stage3 import _count_names, _datasets, _loss, _stage3_collate


def _metadata(source: str, split: str = "train") -> dict:
    path = STAGE3_TARTANVO_FEATURE_CACHE / STAGE3_TARTANVO_FEATURE / source / f"{split}_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(dataset):
    if hasattr(dataset, "df"):
        return dataset.df.accel_label.astype(int).tolist(), dataset.df.steer_label.astype(int).tolist()
    accel, steer = [], []
    for child in getattr(dataset, "datasets", []):
        a, s = _labels(child)
        accel.extend(a); steer.extend(s)
    return accel, steer


def _print_dist(name: str, dataset) -> None:
    accel_names = {v: k for k, v in ACCEL_TO_ID.items()}
    steer_names = {v: k for k, v in STEER_TO_ID.items()}
    accel, steer = _labels(dataset)
    print(f"[{name}]")
    print("  accel:", _count_names(accel, accel_names))
    print("  steer :", _count_names(steer, steer_names))


def main() -> None:
    train, val, summary = _datasets()
    print("train samples by source")
    for source, count in summary["train_sources"].items():
        print(f"{source:<11}: {count}")
    print("val samples by source")
    for source, count in summary["val_sources"].items():
        print(f"{source:<11}: {count}")
    assert "nuscenes" in summary["train_sources"], summary["train_sources"]
    assert "nuscenes" not in summary["val_sources"], summary["val_sources"]
    assert "dacon" not in summary["train_sources"], summary["train_sources"]
    assert "dacon" not in summary["val_sources"], summary["val_sources"]

    metas = {source: _metadata(source) for source in summary["train_sources"]}
    ref = next(iter(metas.values()))
    keys = ("feature_dim", "feature_mode", "num_frames", "tartanvo_checkpoint_sha1")
    for source, meta in metas.items():
        print(f"{source} metadata:", {k: meta.get(k) for k in keys})
        assert all(meta.get(k) == ref.get(k) for k in keys), source

    shapes, dtypes = {}, {}
    for source, ds in train.source_datasets.items():
        item = ds[0]
        shapes[source] = tuple(item["feature"].shape)
        dtypes[source] = str(item["feature"].dtype)
        _print_dist(f"{source} train", ds)
    print("feature tensor shape:", shapes)
    print("feature dtype:", dtypes)
    assert len(set(shapes.values())) == 1, shapes
    assert len(set(dtypes.values())) == 1, dtypes
    _print_dist("all train", train)

    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_stage3_collate)
    batch = next(iter(loader))
    print("random batch shape:", tuple(batch["feature"].shape))
    print("random batch source:", [train[random.randrange(len(train))]["source"] for _ in range(min(4, len(train)))])
    model = Stage3TartanVOGRU(load_pretrained=False).to(DEVICE)
    accel, steer = model.forward_feature(batch["feature"].to(DEVICE))
    loss, _, _ = _loss(accel, steer, batch)
    print("forward logits:", tuple(accel.shape), tuple(steer.shape))
    print("loss:", float(loss.detach().cpu()))
    print("checkpoint:", TARTANVO_CHECKPOINT)
    print("mixed feature smoke passed")


if __name__ == "__main__":
    main()
