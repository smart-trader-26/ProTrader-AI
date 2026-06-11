"""
Tests for the P4-2 swing screener service.

No network: the universe download and the directional signal are both stubbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import services.swing_scan_service as scan_svc


class FakeStats:
    tau_star = 0.63
    oos_precision = 0.606
    oos_base_rate = 0.58
    oos_coverage = 0.056


class FakeSignal:
    """Deterministic stand-in for DirectionalSignal: fires on tickers whose
    last close is above 100, with prob proportional to the close."""

    horizon = 30
    stats = FakeStats()

    def __init__(self, trained: bool = True):
        self._trained = trained

    def is_trained(self) -> bool:
        return self._trained

    def predict(self, close: pd.Series, ticker: str = "") -> dict:
        last = float(close.iloc[-1])
        p = min(0.9, max(0.1, last / 200.0))
        return {
            "signal": "UP" if p >= self.stats.tau_star else "NEUTRAL",
            "prob_up": p,
            "conviction": 0.5 if p >= self.stats.tau_star else 0.0,
        }


def _fake_closes(tickers_levels: dict[str, float], bars: int = 300) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=bars)
    return pd.DataFrame(
        {t: np.full(bars, level) for t, level in tickers_levels.items()}, index=idx
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    scan_svc._cache = None
    scan_svc._cache_at = None
    yield
    scan_svc._cache = None
    scan_svc._cache_at = None


def test_scan_fires_sorted_and_counts(monkeypatch):
    closes = _fake_closes({"AAA.NS": 150.0, "BBB.NS": 80.0, "CCC.NS": 180.0})
    monkeypatch.setattr(scan_svc, "_download_closes", lambda u: closes)
    monkeypatch.setattr(
        "models.directional_signal.get_signal", lambda: FakeSignal(trained=True)
    )

    res = scan_svc.scan(universe=["AAA.NS", "BBB.NS", "CCC.NS"])
    assert res.trained is True
    assert res.n_scanned == 3
    # 150/200=0.75 and 180/200=0.9 clear τ*=0.63; 80/200=0.40 does not.
    assert res.n_fired == 2
    # Fired rows first, sorted by prob descending.
    assert [r.ticker for r in res.rows[:2]] == ["CCC.NS", "AAA.NS"]
    assert res.rows[0].signal == "UP"
    assert res.rows[-1].signal == "NEUTRAL"
    assert res.expected_hit_rate == pytest.approx(0.606)
    assert res.tau_star == pytest.approx(0.63)


def test_scan_untrained_returns_no_rows(monkeypatch):
    monkeypatch.setattr(
        "models.directional_signal.get_signal", lambda: FakeSignal(trained=False)
    )
    # Download must never be attempted when the model is untrained.
    monkeypatch.setattr(
        scan_svc, "_download_closes",
        lambda u: (_ for _ in ()).throw(AssertionError("should not download")),
    )
    res = scan_svc.scan(universe=["AAA.NS"])
    assert res.trained is False
    assert res.rows == []


def test_scan_skips_short_history(monkeypatch):
    closes = _fake_closes({"AAA.NS": 150.0}, bars=300)
    closes["SHORT.NS"] = np.nan
    closes.iloc[-50:, closes.columns.get_loc("SHORT.NS")] = 120.0  # only 50 bars
    monkeypatch.setattr(scan_svc, "_download_closes", lambda u: closes)
    monkeypatch.setattr(
        "models.directional_signal.get_signal", lambda: FakeSignal(trained=True)
    )
    res = scan_svc.scan(universe=["AAA.NS", "SHORT.NS", "MISSING.NS"])
    assert res.n_scanned == 1
    assert res.n_errors == 2


def test_default_scan_is_cached(monkeypatch):
    calls = {"n": 0}

    def counting_download(universe):
        calls["n"] += 1
        return _fake_closes({"AAA.NS": 150.0})

    monkeypatch.setattr(scan_svc, "_download_closes", counting_download)
    monkeypatch.setattr(scan_svc, "_default_universe", lambda: ["AAA.NS"])
    monkeypatch.setattr(
        "models.directional_signal.get_signal", lambda: FakeSignal(trained=True)
    )

    first = scan_svc.scan()
    second = scan_svc.scan()
    assert calls["n"] == 1
    assert first.cached is False
    assert second.cached is True

    # refresh=True busts the cache.
    third = scan_svc.scan(refresh=True)
    assert calls["n"] == 2
    assert third.cached is False


def test_download_failure_degrades_gracefully(monkeypatch):
    def boom(universe):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(scan_svc, "_download_closes", boom)
    monkeypatch.setattr(
        "models.directional_signal.get_signal", lambda: FakeSignal(trained=True)
    )
    res = scan_svc.scan(universe=["AAA.NS", "BBB.NS"])
    assert res.trained is True
    assert res.rows == []
    assert res.n_errors == 2
