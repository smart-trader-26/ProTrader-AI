"""
Model registry router (B6.3).

  GET /api/v1/models/active   → current model version + train date

Today this is hard-coded to the in-process hybrid model — when B6.2 lands
(R2/S3 versioned `models/v{n}/master_xgb.pkl`), this endpoint reads the
loaded version from disk metadata.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from services.prediction_service import MODEL_VERSION

router = APIRouter(prefix="/models", tags=["models"])


class ActiveModel(BaseModel):
    name: str
    version: str
    trained_at: str | None = None
    source: str
    notes: str | None = None


@router.get("/active", response_model=ActiveModel, summary="Currently-served model")
def active_model() -> ActiveModel:
    # Best-effort train-date proxy: mtime of hybrid_model.py. Real value will
    # come from the model registry blob metadata in B6.2.
    trained_at: str | None = None
    try:
        path = "models/hybrid_model.py"
        if os.path.exists(path):
            trained_at = datetime.fromtimestamp(os.path.getmtime(path), UTC).isoformat()
    except OSError:
        pass

    return ActiveModel(
        name="hybrid",
        version=MODEL_VERSION,
        trained_at=trained_at,
        source="in-process",
        notes="Switches to R2/S3 registry in B6.2.",
    )
