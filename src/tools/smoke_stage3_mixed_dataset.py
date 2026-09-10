from __future__ import annotations

from torch.utils.data import ConcatDataset

from src.config import STAGE3_NUM_FRAMES
from src.train.stage3 import _datasets, _stage3_collate


def _leaf_datasets(dataset):
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _leaf_datasets(child)
    else:
        yield dataset


def main() -> None:
    train, val, summary = _datasets()
    assert "nuScenes" in summary["train_sources"], summary["train_sources"]
    assert "nuScenes" not in summary["val_sources"], summary["val_sources"]
    samples = []
    for ds in _leaf_datasets(train):
        sample = ds[0]
        samples.append(sample)
        clip = sample["video"]
        assert tuple(clip.shape) == (3, STAGE3_NUM_FRAMES, 224, 224), tuple(clip.shape)
        assert "accel_label" in sample
        assert "steer_label" in sample
    batch = _stage3_collate(samples)
    assert tuple(batch["video"].shape[1:]) == (3, STAGE3_NUM_FRAMES, 224, 224), tuple(batch["video"].shape)
    print("mixed Stage3 smoke passed")
    print("train sources:", summary["train_sources"])
    print("val sources:", summary["val_sources"])


if __name__ == "__main__":
    main()
