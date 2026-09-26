# DACON Blackbox Video Pipeline

This repository keeps one active submission pipeline per stage.

## Active Models

- Stage 1: MViT-v2-S recapture classifier
- Stage 2: VideoMAE collision and direction model
- Stage 3: frozen V-JEPA ViT-L/16 encoder with task-query acceleration and steering heads

## Data Flow

```text
manifest -> frame cache or dataset images -> train -> checkpoint -> submission inference
```

Stage 1 and Stage 2 currently read dataset inputs directly, so their cache command is a no-op wrapper. Stage 3 uses a comma2k19 JPEG frame cache.

## Canonical Commands

```bash
python -m src.stage1.manifest
python -m src.stage1.cache
python -m src.stage1.train

python -m src.stage2.manifest
python -m src.stage2.cache
python -m src.stage2.train

python -m src.stage3.manifest
python -m src.stage3.cache
python train.py
```

Root training wrapper:

```bash
python train.py stage1
python train.py stage2
python train.py                  # defaults to stage3_vjepa
python train.py stage3           # legacy MViT training
```

## Checkpoints

Default checkpoint roots are under `model/` unless `DACON_MODEL_ROOT` is set.

- Stage 1: `model/stage1/best.pt`
- Stage 2: `model/stage2/best.pt`
- Stage 3 head: `model/stage3/vjepa/best.pt`
- Stage 3 encoder source: `model/stage3/vitl16.pth.tar`

The Stage 3 training checkpoint stores only the task-query head. The submission builder extracts `target_encoder` from the local V-JEPA checkpoint and writes a compact `submission/model/stage3/encoder.pt`; no model download is performed during inference.

## Submission

```bash
python -m src.tools.build_submission
```

Submission inference lives in `submission/inference.py`. Stage 3 loads both `submission/model/stage3/best.pt` and the bundled `submission/model/stage3/encoder.pt` entirely offline.

## Stage 3 Sampling

Stage 3 training and cache generation use timestamp-nearest sampling aligned to the submission cadence:

- `sampling_hz = 10.0`
- cache/manifest frames: `16`
- V-JEPA model input: `8` frames sampled across the same 1.5 sec window
- train row stride: `8`, with offsets `0..7` rotated across 8 epochs
- batch size: `4`; BF16 enabled when the GPU supports it
- offsets: `[-8, -7, ..., 6, 7]`
- coverage: `1.5 sec`
- center: manifest `target_timestamp`
- frame choice: nearest source frame timestamp
- boundary policy: `clamp`
- image size: `224`

The checkpoint stores the selected frame positions, so submission inference supports both existing 16-frame and new 8-frame heads without online downloads.
