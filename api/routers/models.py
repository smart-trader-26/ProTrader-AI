"""
Model registry router (B6.3).

  GET /api/v1/models/active    → active model spec (from registry)
  GET /api/v1/models/versions  → versions available in the registry

Reads from `models.registry.get_registry()`, whose backend is selected by
the `MODEL_REGISTRY_URI` env var (file:// for local dev, s3:// for prod).
A bootstrap `models-registry/active.json` ships with the repo so fresh
installs return a sane spec without any extra setup.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from models.registry import ActiveModelSpec, get_registry

log = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active", response_model=ActiveModelSpec, summary="Currently-served model")
def active_model() -> ActiveModelSpec:
    try:
        return get_registry().read_active()
    except Exception as e:  # noqa: BLE001
        log.warning("registry read failed: %s", e)
        raise HTTPException(status_code=503, detail=f"registry unavailable: {e}") from e


@router.get("/versions", summary="List known model versions in the registry")
def list_versions() -> dict[str, list[str] | str]:
    try:
        reg = get_registry()
        return {"backend": reg.backend, "versions": reg.list_versions()}
    except Exception as e:  # noqa: BLE001
        log.warning("registry list failed: %s", e)
        raise HTTPException(status_code=503, detail=f"registry unavailable: {e}") from e
