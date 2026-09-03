from __future__ import annotations

import argparse
import py_compile
import re
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT

SUBMISSION_DIR = PROJECT_ROOT / "submission"
MODEL_DIR = SUBMISSION_DIR / "model"
REQUIREMENTS = SUBMISSION_DIR / "requirements.txt"
INFERENCE_OUT = SUBMISSION_DIR / "inference.py"
SUBMIT_ZIP = PROJECT_ROOT / "submit.zip"
ROOT_ZIP_ENTRIES = {"model/", "inference.py", "requirements.txt"}


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def validate_inference(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    for name in ("predict_stage1", "predict_stage2", "predict_stage3"):
        if not re.search(rf"^def\s+{name}\s*\(", text, flags=re.MULTILINE):
            raise RuntimeError(f"missing function: {name}")

    if re.search(r"^\s*(from\s+src\.|import\s+src\.)", text, flags=re.MULTILINE):
        raise RuntimeError(
            "submission/inference.py must be self-contained; found a src.* import"
        )

    if re.search(r"\bResNet18_Weights\b|\bresnet18\b|resnet18-f37072fd\.pth", text):
        raise RuntimeError("submission/inference.py still references the old Stage 2 ResNet path")

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


def _iter_submission_files():
    for path in sorted(SUBMISSION_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SUBMISSION_DIR).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".ipynb"}:
            continue
        yield path


def build_zip() -> Path:
    if SUBMIT_ZIP.exists():
        SUBMIT_ZIP.unlink()

    with zipfile.ZipFile(SUBMIT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _iter_submission_files():
            zf.write(path, path.relative_to(SUBMISSION_DIR).as_posix())

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
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"missing zip member(s): {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission/ and build submit.zip.")
    parser.parse_args()

    validate_inference(INFERENCE_OUT)
    validate_inputs()
    zip_path = build_zip()
    print(f"generated: {zip_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
