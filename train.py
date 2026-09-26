from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical training entrypoint for one stage.")
    parser.add_argument("stage", nargs="?", choices=("stage1", "stage2", "stage3", "stage3_vjepa"), default="stage3_vjepa")
    args = parser.parse_args()
    if args.stage == "stage1":
        from src.train.stage1 import fit_stage1
        fit_stage1()
    elif args.stage == "stage2":
        from src.train.stage2 import fit_stage2
        fit_stage2()
    elif args.stage == "stage3_vjepa":
        from src.train.stage3_vjepa import fit_stage3_vjepa
        fit_stage3_vjepa()
    else:
        from src.train.stage3 import fit_stage3
        fit_stage3()


if __name__ == "__main__":
    main()
