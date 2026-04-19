"""B6.2 — model registry: local round-trip + URI-based factory selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.registry import (
    ActiveModelSpec,
    LocalFileRegistry,
    S3Registry,
    _build_from_env,
    publish_version,
    reset_registry_for_tests,
    sha256_of,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_registry_for_tests()
    yield
    reset_registry_for_tests()


def test_local_registry_reads_bootstrap_when_no_active_json(tmp_path: Path):
    reg = LocalFileRegistry(root=tmp_path)
    spec = reg.read_active()
    assert spec.source == "in-process"
    assert spec.version


def test_local_registry_round_trip(tmp_path: Path):
    reg = LocalFileRegistry(root=tmp_path)
    spec = ActiveModelSpec(version="v7", artifacts=["master_xgb.pkl"])
    reg.write_active(spec)
    back = reg.read_active()
    assert back.version == "v7"
    assert back.artifacts == ["master_xgb.pkl"]
    assert back.backend == "local"


def test_publish_version_copies_artifacts_and_updates_active(tmp_path: Path):
    # Create a fake artifact to publish.
    artifact = tmp_path / "src" / "master_xgb.pkl"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x00\x01fake-pickle")

    root = tmp_path / "registry"
    reg = LocalFileRegistry(root=root)
    spec = publish_version(
        reg,
        version="v1",
        artifacts={"master_xgb.pkl": artifact},
        notes="unit test",
    )
    assert spec.version == "v1"
    assert (root / "v1" / "master_xgb.pkl").exists()
    # active.json now points at v1
    assert reg.read_active().version == "v1"
    # list_versions sees it
    assert "v1" in reg.list_versions()


def test_fetch_artifact_raises_for_missing_file(tmp_path: Path):
    reg = LocalFileRegistry(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.fetch_artifact("v9", "nonexistent.pkl")


def test_sha256_of_matches_hashlib(tmp_path: Path):
    import hashlib

    p = tmp_path / "blob"
    p.write_bytes(b"hello world" * 1000)
    assert sha256_of(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_factory_picks_local_from_file_uri(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("config.settings.MODEL_REGISTRY_URI", f"file://{tmp_path}")
    reg = _build_from_env()
    assert isinstance(reg, LocalFileRegistry)
    assert reg.root == tmp_path.resolve()


def test_factory_picks_local_from_bare_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("config.settings.MODEL_REGISTRY_URI", str(tmp_path))
    reg = _build_from_env()
    assert isinstance(reg, LocalFileRegistry)


def test_factory_picks_s3_from_s3_uri(monkeypatch):
    monkeypatch.setattr("config.settings.MODEL_REGISTRY_URI", "s3://my-bucket/registry")
    reg = _build_from_env()
    assert isinstance(reg, S3Registry)
    assert reg.bucket == "my-bucket"
    assert reg.prefix == "registry"


def test_factory_defaults_to_local_bootstrap(monkeypatch):
    monkeypatch.setattr("config.settings.MODEL_REGISTRY_URI", "")
    reg = _build_from_env()
    assert isinstance(reg, LocalFileRegistry)
    # Should land on the repo's bootstrap directory.
    assert reg.root.name == "models-registry"
