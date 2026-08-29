from __future__ import annotations

import argparse
import py_compile
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
REQUIREMENTS = ROOT / "requirements.txt"
INFERENCE_OUT = ROOT / "inference.py"
SUBMIT_ZIP = ROOT / "submit.zip"
REQUIRED_CHECKPOINTS = [
    MODEL_DIR / "stage1" / "best.pt",
    MODEL_DIR / "stage2" / "best.pt",
    MODEL_DIR / "stage2" / "resnet18-f37072fd.pth",
    MODEL_DIR / "stage3" / "best.pt",
]
ROOT_ZIP_ENTRIES = {"model/", "inference.py", "requirements.txt"}


def validate_inference(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    for name in ("predict_stage1", "predict_stage2", "predict_stage3"):
        if not re.search(rf"^def\s+{name}\s*\(", text, flags=re.MULTILINE):
            raise RuntimeError(f"missing function: {name}")
    py_compile.compile(str(path), doraise=True)


def validate_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_CHECKPOINTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model checkpoint(s): {missing}")
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError("requirements.txt is missing")


def _iter_model_files():
    for path in sorted(MODEL_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".ipynb"} or "__pycache__" in path.parts:
            continue
        yield path


def build_zip() -> Path:
    if SUBMIT_ZIP.exists():
        SUBMIT_ZIP.unlink()
    with zipfile.ZipFile(SUBMIT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(INFERENCE_OUT, "inference.py")
        zf.write(REQUIREMENTS, "requirements.txt")
        for path in _iter_model_files():
            zf.write(path, path.relative_to(ROOT).as_posix())
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
        "model/stage1/best.pt",
        "model/stage2/best.pt",
        "model/stage2/resnet18-f37072fd.pth",
        "model/stage3/best.pt",
        "inference.py",
        "requirements.txt",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"missing zip member(s): {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DACON inference.py and build submit.zip.")
    parser.parse_args()

    validate_inference(INFERENCE_OUT)
    print(f"validated: {INFERENCE_OUT.relative_to(ROOT)}")
    validate_inputs()
    zip_path = build_zip()
    print(f"generated: {zip_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
