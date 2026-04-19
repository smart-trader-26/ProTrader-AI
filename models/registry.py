"""
Model registry (B6.2).

A tiny indirection so the hybrid model's binary artifacts can live outside
the code repo — on R2 / S3 in production, on local disk in dev — without
the rest of the backend caring where.

Layout (both backends):

    <root>/
      active.json                    # {"version": "vN", "sha256": "...", ...}
      v0/
        manifest.json                # per-version metadata (kept next to artifacts)
        master_xgb.pkl
        master_lgb.pkl
        ...
      v1/
        manifest.json
        ...

`ActiveModelSpec` is the Pydantic shape the router surfaces. Storage
backend is picked from `MODEL_REGISTRY_URI`:

  • `file:///abs/path` or `./rel/path` → :class:`LocalFileRegistry`
  • `s3://bucket/prefix`               → :class:`S3Registry` (boto3, lazy-import)
  • unset                              → local bootstrap at `<repo>/models-registry`

Artifact downloads are cached under `~/.cache/protrader/models/<version>/`
so a cold boot hits the blob store once per (version, artifact).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

CACHE_ROOT = Path(os.environ.get("PROTRADER_MODEL_CACHE") or (Path.home() / ".cache" / "protrader" / "models"))


class ActiveModelSpec(BaseModel):
    """Shape of `active.json` — also what `/models/active` surfaces."""

    name: str = "hybrid"
    version: str = Field(description="Opaque version tag, e.g. 'v3' or 'v2026-04-19'.")
    trained_at: str | None = None
    source: str = "registry"
    backend: str = "local"  # "local" | "s3"
    artifacts: list[str] = Field(default_factory=list, description="Filenames under <root>/<version>/")
    sha256: str | None = Field(default=None, description="Optional hash of the canonical artifact bundle.")
    notes: str | None = None


class Registry(Protocol):
    """Storage-agnostic interface."""

    backend: str

    def read_active(self) -> ActiveModelSpec: ...
    def write_active(self, spec: ActiveModelSpec) -> None: ...
    def fetch_artifact(self, version: str, filename: str) -> Path: ...
    def list_versions(self) -> list[str]: ...


# ─── local file backend ────────────────────────────────────────────────────


@dataclass
class LocalFileRegistry:
    root: Path
    backend: str = field(default="local", init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── active.json ──
    def _active_path(self) -> Path:
        return self.root / "active.json"

    def read_active(self) -> ActiveModelSpec:
        p = self._active_path()
        if not p.exists():
            return _bootstrap_spec(backend=self.backend)
        with self._lock, p.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        data.setdefault("backend", self.backend)
        return ActiveModelSpec(**data)

    def write_active(self, spec: ActiveModelSpec) -> None:
        p = self._active_path()
        payload = spec.model_dump(mode="json")
        payload["backend"] = self.backend
        with self._lock, p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # ── artifacts ──
    def fetch_artifact(self, version: str, filename: str) -> Path:
        src = self.root / version / filename
        if not src.exists():
            raise FileNotFoundError(f"artifact missing: {src}")
        return src

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())


# ─── s3 / r2 backend ───────────────────────────────────────────────────────


@dataclass
class S3Registry:
    bucket: str
    prefix: str
    backend: str = field(default="s3", init=False)

    def _key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p)

    def _client(self):  # noqa: ANN401 — boto3 S3.Client
        import boto3  # lazy — only needed when MODEL_REGISTRY_URI is s3://

        endpoint_url = os.environ.get("S3_ENDPOINT_URL") or None  # R2 or MinIO
        return boto3.client("s3", endpoint_url=endpoint_url)

    def read_active(self) -> ActiveModelSpec:
        try:
            obj = self._client().get_object(Bucket=self.bucket, Key=self._key("active.json"))
            body = obj["Body"].read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("S3 active.json read failed, returning bootstrap: %s", e)
            return _bootstrap_spec(backend=self.backend)
        data: dict[str, Any] = json.loads(body)
        data.setdefault("backend", self.backend)
        return ActiveModelSpec(**data)

    def write_active(self, spec: ActiveModelSpec) -> None:
        payload = spec.model_dump(mode="json")
        payload["backend"] = self.backend
        self._client().put_object(
            Bucket=self.bucket,
            Key=self._key("active.json"),
            Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

    def fetch_artifact(self, version: str, filename: str) -> Path:
        cache_dir = CACHE_ROOT / version
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / filename
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        key = self._key(version, filename)
        tmp = dst.with_suffix(dst.suffix + ".part")
        self._client().download_file(self.bucket, key, str(tmp))
        tmp.replace(dst)
        return dst

    def list_versions(self) -> list[str]:
        resp = self._client().list_objects_v2(Bucket=self.bucket, Prefix=self._key() + "/", Delimiter="/")
        out: list[str] = []
        for cp in resp.get("CommonPrefixes") or []:
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            if name and name != "active.json":
                out.append(name)
        return sorted(out)


# ─── factory ───────────────────────────────────────────────────────────────


def _bootstrap_spec(backend: str) -> ActiveModelSpec:
    """Returned when `active.json` is missing — lets a fresh install boot."""
    from services.prediction_service import MODEL_VERSION  # local import to avoid cycle

    return ActiveModelSpec(
        version=MODEL_VERSION,
        source="in-process",
        backend=backend,
        notes="bootstrap — no active.json in registry; using the in-process hybrid model.",
    )


def _default_local_root() -> Path:
    return Path(__file__).resolve().parent.parent / "models-registry"


_SINGLETON: Registry | None = None
_SINGLETON_LOCK = threading.Lock()


def get_registry() -> Registry:
    """Resolve the registry once per process based on `MODEL_REGISTRY_URI`."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = _build_from_env()
        return _SINGLETON


def reset_registry_for_tests() -> None:
    """Dump the process-wide singleton — lets tests inject a fresh factory."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None


def _build_from_env() -> Registry:
    from config.settings import MODEL_REGISTRY_URI

    uri = MODEL_REGISTRY_URI.strip()
    if not uri:
        return LocalFileRegistry(root=_default_local_root())

    parsed = urlparse(uri) if "://" in uri else None
    if parsed and parsed.scheme == "file":
        # urllib doesn't round-trip Windows drive letters cleanly: `file://C:/x`
        # parses as netloc='C:' / path='/x'. Stitch them back together and strip
        # the leading slash only when it's followed by a `<letter>:` drive.
        raw = (parsed.netloc + parsed.path) if parsed.netloc else parsed.path
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return LocalFileRegistry(root=Path(raw or "/"))
    if parsed and parsed.scheme == "s3":
        bucket = parsed.netloc
        prefix = (parsed.path or "").lstrip("/")
        if not bucket:
            raise ValueError(f"S3 URI missing bucket: {uri!r}")
        return S3Registry(bucket=bucket, prefix=prefix)
    # Bare path: treat as local.
    return LocalFileRegistry(root=Path(uri))


# ─── helpers ───────────────────────────────────────────────────────────────


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_version(
    registry: Registry,
    version: str,
    artifacts: dict[str, Path],
    trained_at: str | None = None,
    notes: str | None = None,
) -> ActiveModelSpec:
    """
    Upload a bundle of artifacts under `<root>/<version>/` and flip
    `active.json` to point at it. Convenience helper for CLI / CI jobs —
    not called from the request path.
    """
    if not isinstance(registry, LocalFileRegistry):
        raise NotImplementedError(
            "publish_version currently only supports LocalFileRegistry — use aws cli for s3"
        )
    version_dir = registry.root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    for name, src in artifacts.items():
        shutil.copy2(src, version_dir / name)
    spec = ActiveModelSpec(
        version=version,
        trained_at=trained_at,
        artifacts=sorted(artifacts.keys()),
        backend=registry.backend,
        notes=notes,
    )
    registry.write_active(spec)
    return spec
