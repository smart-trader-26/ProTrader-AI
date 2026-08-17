"""
Study 1b -- sensitivity of the scale-normalized weight to the trailing-error
window length W. The deployed code hardcodes MAX_ERROR_WINDOW=10
(config/settings.py:160); this checks whether that specific choice is what
produces the genuine weight differentiation reported in Study 1, or whether
it is an artifact of that one setting.

Reuses the real per-day cross-sectional mean squared errors already computed
and saved by Study 1 (results/study1_daily.csv, columns mse_tech/mse_vol) --
no new downloads, no new model fits, just re-deriving the rolling-window
uncertainty and softmax weights from the same real error series at
different window lengths.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"

WINDOWS = [5, 10, 15, 20, 30]


def weights_for_window(mse_tech, mse_vol, window):
    recent_t, recent_v = [], []
    w_series = []
    for et, ev in zip(mse_tech, mse_vol):
        sigma2_t = np.mean(recent_t) if recent_t else 1.0
        sigma2_v = np.mean(recent_v) if recent_v else 1.0
        tau = max((sigma2_t + sigma2_v) / 2.0, 1e-12)
        wt_exp, wv_exp = np.exp(-sigma2_t / tau), np.exp(-sigma2_v / tau)
        w_series.append(wt_exp / (wt_exp + wv_exp))
        recent_t.append(et)
        recent_v.append(ev)
        if len(recent_t) > window:
            recent_t.pop(0)
        if len(recent_v) > window:
            recent_v.pop(0)
    return np.array(w_series)


def main():
    daily = pd.read_csv(RES / "study1_daily.csv")
    results = {}
    for w in WINDOWS:
        series = weights_for_window(daily["mse_tech"].values, daily["mse_vol"].values, w)
        results[w] = {
            "mean_w_tech": float(series.mean()),
            "std_w_tech": float(series.std()),
            "min_w_tech": float(series.min()),
            "max_w_tech": float(series.max()),
        }
        print(f"W={w:>2}: mean={series.mean():.4f} std={series.std():.4f} "
              f"range=[{series.min():.3f},{series.max():.3f}]")

    with open(RES / "study1b_window_ablation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
