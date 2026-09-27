# Stage 2 Training Pipeline

Stage 2 predicts four outputs from each accident video:

- collision frame
- entry frame
- entry side (`LEFT` or `RIGHT`)
- evasion-space availability (`0` or `1`)

## Human labels

The server label root defaults to:

```text
/data/processed/stage2/CCD_lane_labeling/
```

Override it with `STAGE2_HUMAN_LABEL_ROOT` when necessary. Label files are discovered by schema rather than filename. A CSV or text file is treated as a Stage 2 human-label file when it contains `id`, `video`, `NEW_collision_frame`, `NEW_entry_frame`, `NEW_entry_side`, `NEW_evasion_space`, and `n_frames`.

Human `NEW_*` labels take precedence over official or pseudo labels. `UNKNOWN` and `-1` remain missing labels and are masked independently for each task. An entry frame later than the collision frame is retained in the audit columns but excluded from entry supervision with `label_issue=entry_after_collision`.

Build the canonical manifests with:

```bash
python -m src.stage2.manifest
```

CCD and AIHub are split independently, using `source_id` groups, so adding AIHub rows does not change the deterministic CCD group split.

## Official validation score

Collision and entry predictions are converted from frame error to seconds using each video's FPS. Their metric is `Accuracy@0.3 seconds`. Direction and evasion use macro-F1.

```text
Stage2 score =
    0.35 * collision Accuracy@0.3s
  + 0.35 * entry Accuracy@0.3s
  + 0.15 * direction macro-F1
  + 0.15 * evasion macro-F1
```

The highest `val_stage2_score` checkpoint is saved as `model/stage2/best.pt`. All four tasks are active by default. Ablations supplied through `--tasks` keep task-specific checkpoint names.

## Training

```bash
python -m src.stage2.train
```

Missing labels do not contribute to loss. Default loss weights are `1.0` for collision and entry and `0.5` for direction and avoidance. Checkpoint selection always follows the official score rather than frame MAE.

## Inference contract

Stage 2 inference uses all four model heads. It returns sampled source-frame numbers for collision and entry, `LEFT` or `RIGHT` for entry side, and `0` or `1` for evasion space. Argmax frame outputs are always drawn from the decoded video range.

