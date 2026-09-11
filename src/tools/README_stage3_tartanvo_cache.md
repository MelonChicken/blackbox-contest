# Stage3 TartanVO Feature Cache

This cache is training-only. Submission keeps the online `tartanvo_gru` path and loads the full `best.pt` state_dict.

Run on the data server after comma2k19 manifests and videos are available:

```powershell
uv run python -m src.tools.cache_stage3_tartanvo_feature --split train --feature latent
uv run python -m src.tools.cache_stage3_tartanvo_feature --split val --feature latent
uv run python train.py
```

Pose cache remains available:

```powershell
uv run python -m src.tools.cache_stage3_tartanvo_feature --split train --feature pose
uv run python -m src.tools.cache_stage3_tartanvo_feature --split val --feature pose
```

Required config values in `src/config.py`:

```python
STAGE3_ARCH = "tartanvo_gru"
STAGE3_TARTANVO_USE_FEATURE_CACHE = True
STAGE3_TARTANVO_FEATURE = "latent"  # or "pose"
STAGE3_TARTANVO_FEATURE_NORM = "layernorm"  # or "none"
```

Cache layout:

```text
data/processed/stage3/tartanvo_features/
    latent/train/<sample_key>.pt
    latent/val/<sample_key>.pt
    latent/train_index.csv
    latent/val_index.csv
    latent/train_metadata.json
    latent/val_metadata.json
    pose/...
```

Each `.pt` stores `feature [15,D]`, labels, `sample_key`, `video_path`, and `frame_index`.
