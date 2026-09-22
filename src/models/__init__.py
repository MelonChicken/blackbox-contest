__all__ = [
    "Stage1MViT",
    "Stage2VideoMAE",
    "Stage3MViT",
    "Stage3VJEPA",
    "build_stage2_model",
]


def __getattr__(name: str):
    if name == "Stage1MViT":
        from .stage1 import Stage1MViT
        return Stage1MViT
    if name in {"Stage2VideoMAE", "build_stage2_model"}:
        from .stage2_videomae import Stage2VideoMAE, build_stage2_model
        return {"Stage2VideoMAE": Stage2VideoMAE, "build_stage2_model": build_stage2_model}[name]
    if name == "Stage3MViT":
        from .stage3 import Stage3MViT
        return Stage3MViT
    if name == "Stage3VJEPA":
        from .stage3_vjepa import Stage3VJEPA
        return Stage3VJEPA
    raise AttributeError(name)
