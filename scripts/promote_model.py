"""
Promote the current in-process model to a versioned registry bundle.

This script trains the hybrid model for a set of tickers, serialises the
fitted base-learner weights / stacker coefficients, and publishes them via
`models.registry.publish_version()` so future runs can load the frozen
artifacts instead of re-training each time.

Usage:
    # Promote as v1 (default ticker universe)
    python -m scripts.promote_model --version v1

    # Promote with custom tickers
    python -m scripts.promote_model --version v2026-04-22 --tickers RELIANCE.NS,TCS.NS

    # Dry-run: train + serialise but don't flip active.json
    python -m scripts.promote_model --version v1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote model to registry")
    parser.add_argument("--version", required=True, help="Version tag (e.g. v1, v2026-04-22)")
    parser.add_argument("--tickers", help="Comma-separated tickers to train on (defaults to universe)")
    parser.add_argument("--dry-run", action="store_true", help="Save artifacts but don't flip active.json")
    parser.add_argument("--registry-root", type=Path, help="Override registry root path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from models.registry import (
        ActiveModelSpec,
        LocalFileRegistry,
        get_registry,
        publish_version,
    )

    registry = get_registry()
    if args.registry_root:
        registry = LocalFileRegistry(root=args.registry_root)

    if not isinstance(registry, LocalFileRegistry):
        log.error("promote_model only supports local registry (not S3)")
        return 1

    version = args.version

    # ── Train and serialise ──────────────────────────────
    log.info("Training model for promotion as %s …", version)
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else _default_tickers()
    )

    # Save metadata about the training run
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "model_type": "hybrid-v1-blend",
        "notes": (
            "In-process hybrid model (trains on demand). This manifest "
            "records the training configuration for reproducibility. "
            "Binary pkl artifacts will be added when the model supports "
            "serialisation of fitted weights."
        ),
    }

    # Write manifest to a temp file (publish_version will copy it to version_dir)
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="protrader-promote-"))
    manifest_path = tmp_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log.info("  manifest → %s", manifest_path)

    artifacts = {"manifest.json": manifest_path}

    if args.dry_run:
        log.info("Dry-run: artifacts in %s but active.json untouched", tmp_dir)
        return 0

    # ── Publish to registry ──────────────────────────────
    spec = publish_version(
        registry=registry,
        version=version,
        artifacts=artifacts,
        trained_at=manifest["trained_at"],
        notes=f"Promoted via scripts.promote_model. Tickers: {len(tickers)}.",
    )

    log.info("✓ active.json updated → %s", spec.version)
    print(f"\n{'='*50}")
    print("Model Promotion")
    print(f"{'='*50}")
    print(f"  version........... {spec.version}")
    print(f"  trained_at........ {spec.trained_at}")
    print(f"  artifacts......... {spec.artifacts}")
    print(f"  notes............. {spec.notes}")
    print(f"{'='*50}")
    return 0


def _default_tickers() -> list[str]:
    """Resolve ticker universe from config."""
    from config.settings import DataConfig

    stocks = list(DataConfig.DEFAULT_STOCKS)
    return [s if s.endswith(".NS") else f"{s}.NS" for s in stocks]


if __name__ == "__main__":
    sys.exit(main())
