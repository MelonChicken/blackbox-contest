from __future__ import annotations

import argparse
import py_compile
import re
import sys
import tokenize
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, STAGE3_VJEPA_CHECKPOINT, STAGE3_VJEPA_MODEL

SUBMISSION_DIR = PROJECT_ROOT / "submission"
MODEL_DIR = SUBMISSION_DIR / "model"
REQUIREMENTS = SUBMISSION_DIR / "requirements.txt"
INFERENCE_OUT = SUBMISSION_DIR / "inference.py"
SUBMIT_ZIP = PROJECT_ROOT / "submit.zip"
ROOT_ZIP_ENTRIES = {"model/", "inference.py", "requirements.txt"}
STAGE3_ARCH = "vjepa_vitl_224"
STAGE3_HEAD_SOURCE = STAGE3_VJEPA_MODEL / "best.pt"
STAGE3_ENCODER_SOURCE = STAGE3_VJEPA_CHECKPOINT
STAGE3_DIR = MODEL_DIR / "stage3"
STAGE3_HEAD_OUT = STAGE3_DIR / "best.pt"
STAGE3_ENCODER_OUT = STAGE3_DIR / "encoder.pt"


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def validate_inference(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    with tokenize.open(path) as f:
        text = f.read()
    for name in ("predict_stage1", "predict_stage2", "predict_stage3"):
        if not re.search(rf"^def\s+{name}\s*\(", text, flags=re.MULTILINE):
            raise RuntimeError(f"missing function: {name}")

    if re.search(r"^\s*(from\s+src\.|import\s+src\.)", text, flags=re.MULTILINE):
        raise RuntimeError("submission/inference.py must not import repo-local src.*")

    if re.search(r"\bResNet18_Weights\b|resnet18-f37072fd\.pth", text):
        raise RuntimeError("submission/inference.py still references downloadable ResNet weights")

    compile(text, str(path), "exec")




def validate_stage2_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", {})
    if "model_config" in checkpoint:
        required_prefixes = ("collision_head.", "entry_head.", "direction_head.", "avoidance_head.")
    elif "videomae_config" in checkpoint:
        required_prefixes = ("encoder.", "collision_head.", "side_head.")
    else:
        raise RuntimeError("Stage2 checkpoint contains neither model_config nor videomae_config.")
    missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state_dict)]
    if missing:
        raise RuntimeError(f"Stage2 checkpoint is missing head weights: {missing}")


def _extract_encoder_state(payload: dict) -> dict:
    state = payload.get("target_encoder", payload.get("encoder", payload))
    if not isinstance(state, dict) or not state:
        raise RuntimeError("V-JEPA checkpoint contains no encoder state dict")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise RuntimeError("V-JEPA encoder state contains non-tensor values")
    return state


def prepare_stage3_vjepa() -> None:
    if not STAGE3_HEAD_SOURCE.is_file():
        raise FileNotFoundError(f"missing Stage3 V-JEPA head: {STAGE3_HEAD_SOURCE}")
    if not STAGE3_ENCODER_SOURCE.is_file():
        raise FileNotFoundError(f"missing Stage3 V-JEPA encoder: {STAGE3_ENCODER_SOURCE}")

    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(STAGE3_HEAD_SOURCE, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != STAGE3_ARCH or "head" not in checkpoint:
        raise RuntimeError("Stage3 head checkpoint is not a vjepa_vitl_224 checkpoint")
    checkpoint = dict(checkpoint)
    checkpoint["encoder_filename"] = STAGE3_ENCODER_OUT.name
    checkpoint.pop("encoder_checkpoint", None)
    torch.save(checkpoint, STAGE3_HEAD_OUT)

    encoder_is_current = (
        STAGE3_ENCODER_OUT.is_file()
        and STAGE3_ENCODER_OUT.stat().st_mtime_ns >= STAGE3_ENCODER_SOURCE.stat().st_mtime_ns
    )
    if not encoder_is_current:
        payload = torch.load(
            STAGE3_ENCODER_SOURCE, map_location="cpu", weights_only=True, mmap=True
        )
        torch.save({"target_encoder": _extract_encoder_state(payload)}, STAGE3_ENCODER_OUT)


def validate_stage3_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != STAGE3_ARCH:
        raise RuntimeError(f"Stage3 checkpoint arch must be {STAGE3_ARCH}")
    head = checkpoint.get("head")
    expected = {
        "accel_query": (1, 1, 1024),
        "steer_query": (1, 1, 1024),
        "norm.weight": (1024,),
        "norm.bias": (1024,),
        "accel.0.weight": (512, 1024),
        "accel.0.bias": (512,),
        "accel.3.weight": (4, 512),
        "accel.3.bias": (4,),
        "steer.0.weight": (512, 1024),
        "steer.0.bias": (512,),
        "steer.3.weight": (3, 512),
        "steer.3.bias": (3,),
    }
    if not isinstance(head, dict) or set(head) != set(expected):
        raise RuntimeError("Stage3 checkpoint contains an invalid V-JEPA head state")
    bad_shapes = {
        key: tuple(head[key].shape)
        for key, shape in expected.items()
        if tuple(head[key].shape) != shape
    }
    if bad_shapes:
        raise RuntimeError(f"Stage3 V-JEPA head shape mismatch: {bad_shapes}")
    if not all(bool(torch.isfinite(value).all()) for value in head.values()):
        raise RuntimeError("Stage3 V-JEPA head contains non-finite weights")
    encoder_filename = checkpoint.get("encoder_filename")
    if encoder_filename != STAGE3_ENCODER_OUT.name or not STAGE3_ENCODER_OUT.is_file():
        raise RuntimeError("Stage3 checkpoint does not reference the bundled encoder.pt")


def validate_inputs() -> None:
    required = [
        INFERENCE_OUT,
        REQUIREMENTS,
        MODEL_DIR / "stage1" / "best.pt",
        MODEL_DIR / "stage2" / "best.pt",
        MODEL_DIR / "stage3" / "best.pt",
    ]
    missing = [_relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing submission file(s): {missing}")
    validate_stage2_checkpoint(MODEL_DIR / "stage2" / "best.pt")
    validate_stage3_checkpoint(MODEL_DIR / "stage3" / "best.pt")


def _iter_submission_files():
    for path in sorted(SUBMISSION_DIR.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".ipynb", ".zip"}:
            continue
        yield path, path.relative_to(SUBMISSION_DIR).as_posix()


def _iter_extra_model_files():
    return
    yield


def build_zip() -> Path:
    if SUBMIT_ZIP.exists():
        SUBMIT_ZIP.unlink()

    with zipfile.ZipFile(SUBMIT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in _iter_submission_files():
            zf.write(path, arcname)
        for path, arcname in _iter_extra_model_files():
            zf.write(path, arcname)

    validate_zip(SUBMIT_ZIP)
    return SUBMIT_ZIP


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

    roots = set()
    for name in names:
        first = name.split("/", 1)[0]
        roots.add(f"{first}/" if "/" in name else first)
        if "__pycache__" in name or name.endswith((".pyc", ".ipynb")) or name.startswith("data/"):
            raise RuntimeError(f"forbidden zip member: {name}")

    if roots != ROOT_ZIP_ENTRIES:
        raise RuntimeError(f"unexpected zip root entries: {sorted(roots)}")

    required = {
        "inference.py",
        "requirements.txt",
        "model/stage1/best.pt",
        "model/stage2/best.pt",
        "model/stage3/best.pt",
        "model/stage3/encoder.pt",
        "model/__init__.py",
        "model/stage1/__init__.py",
        "model/stage1/mvit.py",
        "model/stage2/__init__.py",
        "model/stage2/videomae.py",
        "model/stage3/__init__.py",
        "model/stage3/heads.py",
        "model/stage3/vjepa/models/vision_transformer.py",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"missing zip member(s): {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission/ and build submit.zip.")
    parser.parse_args()

    prepare_stage3_vjepa()
    validate_inference(INFERENCE_OUT)
    validate_inputs()
    zip_path = build_zip()
    print(f"generated: {zip_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

