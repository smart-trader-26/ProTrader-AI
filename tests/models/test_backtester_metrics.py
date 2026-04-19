"""
Tests for A8.2 (Sortino/Calmar/Expectancy) + A8.3 (Diebold-Mariano) extensions
to `models/backtester.py`.
"""

from __future__ import annotations

import numpy as np

from models.backtester import _compute_metrics_from_returns, diebold_mariano_test


def test_sortino_higher_when_no_downside():
    """All-positive returns → sortino is high (downside std ~0) but we floor it to 0."""
    returns = np.array([0.01] * 20)
    m = _compute_metrics_from_returns(returns)
    # No downside → sortino = 0 (our floor); Sharpe is well-defined
    assert m["sortino_ratio"] == 0.0
    assert m["sharpe_ratio"] > 0


def test_sortino_defined_when_mixed_returns():
    returns = np.array([0.02, -0.005, 0.01, -0.003, 0.015, -0.01])
    m = _compute_metrics_from_returns(returns)
    assert m["sortino_ratio"] != 0.0
    # Sortino should exceed Sharpe when upside >> downside std
    assert m["sortino_ratio"] > m["sharpe_ratio"]


def test_calmar_positive_on_profitable_series_with_drawdown():
    # 1 year of returns with a clear drawdown mid-year so Calmar is finite
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 252)
    m = _compute_metrics_from_returns(returns)
    assert m["calmar_ratio"] != 0  # should be finite non-zero with mixed returns
    assert m["max_drawdown"] < 0


def test_calmar_infinity_on_strictly_monotonic_gains():
    """No drawdown at all → Calmar is mathematically ∞; we surface that."""
    returns = np.linspace(0.001, 0.002, 252)
    m = _compute_metrics_from_returns(returns)
    assert m["max_drawdown"] == 0
    assert m["calmar_ratio"] == float("inf")


def test_expectancy_matches_definition():
    # Two wins of +0.02, one loss of -0.01 → E = (2/3)*0.02 + (1/3)*-0.01 = 0.01
    returns = np.array([0.02, 0.02, -0.01])
    m = _compute_metrics_from_returns(returns)
    expected = (2 / 3) * 0.02 + (1 / 3) * -0.01
    assert abs(m["expectancy"] - expected) < 1e-9


def test_dm_test_null_on_identical_errors():
    """Identical error series → d=0 → DM stat = 0 → p = 1 (cannot reject)."""
    err = np.random.normal(0, 1, 100)
    result = diebold_mariano_test(err, err.copy())
    assert abs(result["dm_stat"]) < 1e-6
    assert result["p_value"] > 0.99
    assert result["reject_null_at_5pct"] is False


def test_dm_test_rejects_when_model_clearly_better():
    """Model errors much smaller than benchmark → negative DM stat, p small."""
    rng = np.random.default_rng(42)
    n = 500
    model_err = rng.normal(0, 0.5, n)
    bench_err = rng.normal(0, 2.0, n)
    result = diebold_mariano_test(model_err, bench_err, loss="squared")
    assert result["dm_stat"] < 0  # model has lower squared error
    assert result["reject_null_at_5pct"] is True


def test_dm_test_short_sample_returns_safe_defaults():
    result = diebold_mariano_test(np.array([0.1, 0.2]), np.array([0.1, 0.3]))
    assert result["p_value"] == 1.0
    assert result["reject_null_at_5pct"] is False
