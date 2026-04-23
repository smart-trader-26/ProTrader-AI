"""
Auto-populate Upstox instrument-key map from the official BOD file.

Downloads the NSE equity instrument list from Upstox's CDN, filters for
segment=NSE_EQ + instrument_type=EQ, and writes a JSON map from
``{TRADING_SYMBOL}.NS`` → ``instrument_key`` into
``config/upstox_instruments.json``.

Run once after KYC, or daily via cron to pick up IPOs / symbol changes:

    python -m scripts.populate_instruments
    python -m scripts.populate_instruments --out /custom/path/instruments.json

The output file is the same one ``data.upstox_client`` reads at runtime.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_NSE_BOD_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "config" / "upstox_instruments.json"
_REQUEST_TIMEOUT = 30.0


def download_nse_instruments(url: str = _NSE_BOD_URL) -> list[dict]:
    """Download and decompress the NSE BOD instrument JSON."""
    log.info("downloading %s …", url)
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"expected list, got {type(data).__name__}")
    return data


def build_instrument_map(instruments: list[dict]) -> dict[str, str]:
    """
    Filter for NSE_EQ equities and build {TRADING_SYMBOL.NS: instrument_key}.

    Skips entries without a trading_symbol or instrument_key.
    """
    result: dict[str, str] = {}
    for inst in instruments:
        seg = inst.get("segment", "")
        itype = inst.get("instrument_type", "")
        if seg != "NSE_EQ" or itype != "EQ":
            continue
        symbol = inst.get("trading_symbol", "").strip()
        key = inst.get("instrument_key", "").strip()
        if not symbol or not key:
            continue
        # yfinance-style ticker: RELIANCE.NS
        yf_ticker = f"{symbol}.NS"
        result[yf_ticker] = key
    return result


def write_map(mapping: dict[str, str], path: Path) -> None:
    """Write the instrument map as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def populate(out_path: Path | None = None, url: str = _NSE_BOD_URL) -> dict[str, str]:
    """Full pipeline: download → filter → write. Returns the map."""
    instruments = download_nse_instruments(url)
    log.info("  total instruments in BOD file: %d", len(instruments))
    mapping = build_instrument_map(instruments)
    log.info("  NSE_EQ equities mapped: %d", len(mapping))
    dest = out_path or _DEFAULT_OUT
    write_map(mapping, dest)
    log.info("  written to %s", dest)
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate Upstox instrument-key map from the official BOD file."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        mapping = populate(out_path=args.out)
        print(f"✓ {len(mapping)} instruments written to {args.out}")
        return 0
    except Exception as e:
        log.exception("populate failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
