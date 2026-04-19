"""
Tests for A8.4 3-way split + reveal_holdout gate.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from services.backtest_split import (
    DEFAULT_SPLIT,
    describe_split,
    reveal_holdout,
    split_train_val_holdout,
)


def _daily_frame(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="B")
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_split_respects_default_boundaries():
    df = _daily_frame("2020-01-01", "2025-06-30")
    train, val, holdout = split_train_val_holdout(df, include_holdout=True)

    assert train.index.max().date() <= date(2022, 12, 31)
    assert val.index.min().date() > date(2022, 12, 31)
    assert val.index.max().date() <= date(2023, 12, 31)
    assert holdout is not None and holdout.index.min().date() > date(2023, 12, 31)


def test_default_hides_holdout():
    df = _daily_frame("2020-01-01", "2025-06-30")
    _, _, holdout = split_train_val_holdout(df)
    assert holdout is None


def test_reveal_holdout_refuses_without_exact_phrase():
    df = _daily_frame("2020-01-01", "2025-06-30")
    with pytest.raises(PermissionError):
        reveal_holdout(df, confirm="please")
    with pytest.raises(PermissionError):
        reveal_holdout(df, confirm="")


def test_reveal_holdout_works_with_confirmation():
    df = _daily_frame("2020-01-01", "2025-06-30")
    holdout = reveal_holdout(df, confirm="I have not touched the holdout")
    assert not holdout.empty
    assert holdout.index.min().date() > DEFAULT_SPLIT.val_end


def test_reveal_holdout_empty_frame_raises():
    # Frame that ends before val_end → holdout should be empty → ValueError
    df = _daily_frame("2020-01-01", "2022-06-30")
    with pytest.raises(ValueError):
        reveal_holdout(df, confirm="I have not touched the holdout")


def test_describe_split_reports_row_counts():
    df = _daily_frame("2022-01-01", "2024-06-30")
    d = describe_split(df)
    assert d["train"]["n_rows"] > 0
    assert d["val"]["n_rows"] > 0
    assert d["holdout"]["n_rows"] > 0
    # No overlap — rows partition into exactly the three splits
    assert d["train"]["n_rows"] + d["val"]["n_rows"] + d["holdout"]["n_rows"] == len(df)
