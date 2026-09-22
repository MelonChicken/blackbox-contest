__all__ = ["fit_stage1", "fit_stage2", "fit_stage3", "fit_stage3_vjepa"]


def __getattr__(name: str):
    if name == "fit_stage1":
        from .stage1 import fit_stage1
        return fit_stage1
    if name == "fit_stage2":
        from .stage2 import fit_stage2
        return fit_stage2
    if name == "fit_stage3":
        from .stage3 import fit_stage3
        return fit_stage3
    if name == "fit_stage3_vjepa":
        from .stage3_vjepa import fit_stage3_vjepa
        return fit_stage3_vjepa
    raise AttributeError(name)
