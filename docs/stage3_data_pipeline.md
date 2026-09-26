# Stage 3 Data Pipeline

Stage 3 is comma2k19 + V-JEPA ViT-L/16.

```text
data/raw/comma2k19
-> data/processed/stage3/manifest/{train,val}.csv
-> data/processed/stage3/comma2k19_frames/
-> src.datasets.comma2k19_stage3_vjepa.Comma2k19Stage3VJEPADataset
-> src.models.stage3_vjepa.Stage3VJEPA
-> model/stage3/vjepa/best.pt
```

## Manifest

Build and check:

```bash
python -m src.tools.audit_stage3_comma_sync
python -m src.tools.build_comma2k19_stage3_manifest
python -m src.tools.check_stage3_comma_alignment
python -m src.tools.audit_stage3_comma_labels
```

Required outputs:

```text
data/processed/stage3/manifest/train.csv
data/processed/stage3/manifest/val.csv
data/processed/stage3/manifest/metadata.json
```

Required columns:

| column | meaning |
|---|---|
| `route_id` | comma2k19 route id |
| `segment_id` | route segment id |
| `video_path` | source video path relative to raw root |
| `sample_index` | 10 Hz sample index |
| `target_timestamp` | CAN/video-clock timestamp |
| `video_frame_index` | nearest decoded frame index |
| `video_frame_timestamp` | selected frame timestamp |
| `alignment_error_sec` | frame/timestamp alignment error |
| `clip_frame_indices` | JSON list of MViT clip frame indices |
| `speed` | interpolated speed |
| `steering_angle` | interpolated steering angle |
| `acceleration` | derived acceleration |
| `accel_label` | 4-class acceleration target |
| `steer_label` | 3-class steering target |

## Frame Cache

Build frame cache before training. The dataset refuses slow HEVC fallback when cache is configured.

```bash
python -m src.tools.cache_comma2k19_stage3_frames
```

Required layout:

```text
data/processed/stage3/comma2k19_frames/<route_id>/<segment_id>/frames.csv
```

## Training

```bash
python train.py
```

Selection metric is weighted macro-F1:

```text
0.7 * accel_macro_f1 + 0.3 * steer_macro_f1
```

Outputs:

```text
model/stage3/vjepa/best.pt
model/stage3/vjepa/*_history.csv
```
