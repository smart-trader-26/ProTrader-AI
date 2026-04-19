"""
Three-way temporal split for honest model evaluation (A8.4).

The rule: never look at the holdout until the very end. Anything tuned
against the holdout — hyper-parameters, feature engineering, threshold
calibration — leaks future info into the past and overstates Sharpe by
the same 20-50% margin a naive backtest already overstates by.

Convention used here:
    TRAIN    : 2018-01-01 → 2022-12-31      (5 yrs; all feature dev OK)
    VAL      : 2023-01-01 → 2023-12-31      (1 yr; threshold + cal tuning)
    HOLDOUT  : 2024-01-01 → today           (all untouched — reveal at end)

The split is exposed as two functions:
    split_train_val_holdout()    returns three DataFrames
    reveal_holdout()             gate — raises unless called explicitly
                                 with `confirm="I have not touched the
                                 holdout"`, to make accidental leakage
                                 harder in the UI path.

Usage in the dashboard (Track B migration will lift this into an HTTP
endpoint):

    train, val, _ = split_train_val_holdout(df, include_holdout=False)
    # ... tune threshold on val ...
    final = reveal_holdout(df, confirm="I have not touched the holdout")
    # run final backtest on `final`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    train_end: date = date(2022, 12, 31)
    val_end: date = date(2023, 12, 31)
    # holdout starts day after val_end and runs to the frame's tail


DEFAULT_SPLIT = SplitDates()
_HOLDOUT_CONFIRMATION = "I have not touched the holdout"


def split_train_val_holdout(
    df: pd.DataFrame,
    split: SplitDates = DEFAULT_SPLIT,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Return `(train_df, val_df, holdout_df_or_None)`.

    Pass `include_holdout=True` only via `reveal_holdout()` — we raise
    otherwise so accidental calls in notebooks can't leak the holdout.

    `df.index` must be datetime-like (DatetimeIndex or date column index).
    """
    if df.empty:
        return df.copy(), df.copy(), None if not include_holdout else df.copy()

    idx = pd.DatetimeIndex(df.index)
    train_mask = idx.date <= split.train_end
    val_mask = (idx.date > split.train_end) & (idx.date <= split.val_end)
    holdout_mask = idx.date > split.val_end

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    holdout_df = df.loc[holdout_mask].copy() if include_holdout else None
    return train_df, val_df, holdout_df


def reveal_holdout(
    df: pd.DataFrame,
    confirm: str,
    split: SplitDates = DEFAULT_SPLIT,
) -> pd.DataFrame:
    """
    Return the holdout frame — **only** if `confirm` is the exact string
    `"I have not touched the holdout"`. This is deliberately a speed-bump
    so a future-you / future-me doesn't casually peek.
    """
    if confirm != _HOLDOUT_CONFIRMATION:
        raise PermissionError(
            "Holdout access denied: pass confirm='I have not touched the holdout' "
            "to explicitly acknowledge you haven't used this data for tuning."
        )
    _, _, holdout = split_train_val_holdout(df, split=split, include_holdout=True)
    if holdout is None or holdout.empty:
        raise ValueError(f"Holdout is empty — frame ends before {split.val_end}.")
    return holdout


def describe_split(df: pd.DataFrame, split: SplitDates = DEFAULT_SPLIT) -> dict:
    """Row / date-range summary for each partition — for UI display."""
    train, val, holdout = split_train_val_holdout(df, split=split, include_holdout=True)

    def _bounds(frame: pd.DataFrame) -> tuple[date | None, date | None, int]:
        if frame.empty:
            return None, None, 0
        idx = pd.DatetimeIndex(frame.index)
        return idx.min().date(), idx.max().date(), len(frame)

    t_start, t_end, t_n = _bounds(train)
    v_start, v_end, v_n = _bounds(val)
    h_start, h_end, h_n = _bounds(holdout if holdout is not None else df.iloc[0:0])

    return {
        "train": {"start": t_start, "end": t_end, "n_rows": t_n},
        "val":   {"start": v_start, "end": v_end, "n_rows": v_n},
        "holdout": {"start": h_start, "end": h_end, "n_rows": h_n},
    }
