from __future__ import annotations

_EXPORTS = {
    "AIHubStage1Dataset": "src.datasets.aihubDataset",
    "Stage2Dataset": "src.datasets.stage2_dataset",
    "Comma2k19Stage3Dataset": "src.datasets.comma2k19_stage3",
    "Stage3DaconDataset": "src.datasets.comma2k19_stage3",
    "NuScenesStage3Dataset": "src.datasets.nuscenes_stage3",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value