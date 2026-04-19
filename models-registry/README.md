# Model registry (B6.2)

This directory is the default `MODEL_REGISTRY_URI` backend. It holds
versioned model artifact bundles plus an `active.json` pointer that the
FastAPI `/models/active` endpoint reads.

## Layout

```
models-registry/
  active.json            # currently-served version
  v0/                    # bootstrap: no binary artifacts
  v1/
    manifest.json
    master_xgb.pkl
    master_lgb.pkl
    meta_stacker.pkl
    calibrator.pkl
  v2/
    ...
```

## Promoting a new version

```python
from pathlib import Path
from models.registry import get_registry, publish_version

publish_version(
    get_registry(),
    version="v1",
    artifacts={
        "master_xgb.pkl":    Path("artifacts/master_xgb.pkl"),
        "master_lgb.pkl":    Path("artifacts/master_lgb.pkl"),
        "meta_stacker.pkl":  Path("artifacts/meta_stacker.pkl"),
        "calibrator.pkl":    Path("artifacts/calibrator.pkl"),
    },
    trained_at="2026-04-19T03:00:00Z",
    notes="v1 — recalibrated on 2024 data, adds macro features.",
)
```

That writes the files into `v1/` and flips `active.json` atomically.

## Switching to S3 / R2

Set the env var and (optionally) a custom endpoint for R2:

```bash
export MODEL_REGISTRY_URI=s3://protrader-models/registry
export S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com   # R2 only
```

The registry lazy-imports boto3 only when an `s3://` URI is in play, so
dev installs stay boto3-free.
