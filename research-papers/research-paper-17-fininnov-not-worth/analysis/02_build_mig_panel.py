"""
Step 2 - turn scored headlines into a symbol-session panel carrying the gated
aggregate A and every competing aggregator we intend to race it against.

Timing convention (copied verbatim from the paper-14 pipeline, which was audited
after a timezone-join look-ahead leak was found in an earlier study):
  every headline carries a UTC publication stamp; it is converted to
  America/New_York and assigned to the session whose 16:00 ET close first follows
  it.  The aggregate for session d therefore contains only information public
  before that close, and it is used to predict returns realised from session d+1
  onwards.  Nothing on the right-hand side of a forecast is ever stamped later
  than the left-hand side.

Aggregators built for every (symbol, session) with at least one headline:

  A_mig      relevance-weighted product aggregate, Eq. (2): sum(r^2 s) / sum(r)
             over events clearing the materiality floor mu_0, r = nu * mu
  pol_mean   plain mean polarity - the standard additive recipe
  pol_relf   mean polarity restricted to material events (a hard relevance
             filter; this is the baseline that vendor pipelines actually use)
  pol_cnt    count-weighted mean polarity
  add_comb   mean of the *additive* combiner (s + nu + mu)/3, the natural
             non-multiplicative alternative with the same three inputs
  A_nu       product aggregate using novelty only  (ablation)
  A_mu       product aggregate using materiality only (ablation)
  nu_bar, mu_bar, n_news, frac_material  descriptive companions

Forward targets are cumulative market-adjusted log returns over H sessions
starting at d+1, for H in {1, 5, 21}, plus their signs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT.parent / "research-paper-14-aece-not-sent" / "data-cache"

MU0 = 0.15          # materiality floor of Eq. (2), as in the deployed system
HORIZONS = (1, 5, 21)

# The FNSPID coverage universe mixes operating companies with exchange-traded
# funds.  Materiality - "how sensitive is this firm's fundamental value to the
# event" - has no clean meaning for a country or commodity fund, so the funds are
# excluded from the study universe.  They are listed explicitly rather than
# pattern-matched so the exclusion is auditable.
NON_EQUITY = {
    "EWJ",   # iShares MSCI Japan
    "SLV",   # iShares Silver Trust
    "GXC",   # SPDR S&P China
    "PGJ",   # Invesco Golden Dragon China
    "QQQ",   # Invesco QQQ Trust
    "FXP",   # ProShares UltraShort FTSE China 50
    "EWI",   # iShares MSCI Italy
    "YINN",  # Direxion Daily FTSE China Bull 3X
}


def assign_session(ts_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Map each UTC stamp to the first trading session closing at/after it."""
    et = ts_utc.dt.tz_convert("America/New_York")
    day = et.dt.normalize().dt.tz_localize(None)
    before_close = et.dt.hour < 16
    eff = day.where(before_close, day + pd.Timedelta(days=1))
    idx = sessions.searchsorted(eff.values, side="left")
    ok = idx < len(sessions)
    out = pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")
    out.loc[ok] = sessions[idx[ok]]
    return out


def load_scores(cache_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(cache_db))
    df = pd.read_sql("SELECT k, symbol, title, nu, mu, s FROM scores", con)
    con.close()
    return df


def aggregate(df: pd.DataFrame, mu0: float = MU0) -> pd.DataFrame:
    """Per (symbol, session) aggregation of the three axes."""
    d = df.copy()
    d["r"] = d["nu"] * d["mu"]
    d["a"] = d["s"] * d["r"]
    d["material"] = (d["mu"] >= mu0).astype(np.int8)

    keys = ["symbol", "session"]
    g = d.groupby(keys)

    # additive and filtered baselines over *all* events of the session
    d["_add"] = (d["s"] + d["nu"] + d["mu"]) / 3.0
    base = pd.DataFrame(
        {
            "pol_mean": g["s"].mean(),
            "nu_bar": g["nu"].mean(),
            "mu_bar": g["mu"].mean(),
            "n_news": g["s"].size(),
            "frac_material": g["material"].mean(),
            "add_comb": g["_add"].mean(),
        }
    )
    # count-weighted polarity: repeated coverage of one story gets more weight,
    # which is what a naive "sum the sentiment" pipeline effectively does
    base["pol_cnt"] = g["s"].sum() / np.sqrt(base["n_news"])

    def weighted_agg(frame: pd.DataFrame, weight: pd.Series, name: str) -> pd.Series:
        """sum(w^2 s) / sum(w) per session, aligned to the full session index."""
        tmp = frame.assign(_w=weight.to_numpy(), _w2s=(weight.to_numpy() ** 2) * frame["s"])
        gg = tmp.groupby(keys).agg(num=("_w2s", "sum"), den=("_w", "sum"))
        val = np.where(gg["den"] > 0, gg["num"] / gg["den"], 0.0)
        return pd.Series(val, index=gg.index, name=name).reindex(base.index).fillna(0.0)

    # Eq. (2): relevance-weighted, restricted to events clearing the floor mu_0
    m = d[d["material"] == 1]
    base["A_mig"] = weighted_agg(m, m["r"], "A_mig")
    # ablations: one axis at a time, same weighting scheme, all events
    base["A_nu"] = weighted_agg(d, d["nu"], "A_nu")
    base["A_mu"] = weighted_agg(d, d["mu"], "A_mu")
    # hard relevance filter: mean polarity of material events only
    base["pol_relf"] = (
        m.groupby(keys)["s"].mean().reindex(base.index).fillna(0.0)
    )
    base["n_material"] = (
        m.groupby(keys)["s"].size().reindex(base.index).fillna(0).astype(int)
    )
    return base.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "cache" / "mig_panel.parquet"))
    ap.add_argument("--mu0", type=float, default=MU0,
                    help="materiality floor of Eq. (2); varied in the robustness study")
    args = ap.parse_args()

    scores = load_scores(ROOT / "cache" / "scores.db")
    print(f"scored (symbol,title) pairs: {len(scores):,}  symbols: {scores.symbol.nunique()}")

    # keep only symbols whose scoring pass finished, so no session is aggregated
    # from a partially scored set of headlines
    import hashlib as _h
    import re as _re

    _ws = _re.compile(r"\s+")
    corpus_all = pd.read_parquet(DATA / "corpus.parquet")
    corpus_all["k"] = [
        _h.sha1(f"{sy}|{_ws.sub(' ', str(ti).strip().lower())}".encode()).hexdigest()
        for sy, ti in zip(corpus_all["Stock_symbol"], corpus_all["Article_title"])
    ]
    want = corpus_all.drop_duplicates("k").groupby("Stock_symbol").size()
    have = scores.groupby("symbol").size()
    complete = [s for s in have.index if s in want.index and have[s] >= 0.98 * want[s]]
    complete = sorted(set(complete) - NON_EQUITY)
    scores = scores[scores["symbol"].isin(complete)]
    print(f"symbols fully scored and in scope: {len(complete)}")

    # the cache key was built above; join the three axes onto every headline row
    corpus = corpus_all[corpus_all["Stock_symbol"].isin(set(scores["symbol"]))].copy()
    corpus = corpus.merge(scores[["k", "nu", "mu", "s"]], on="k", how="inner")
    print(f"headlines with axes: {len(corpus):,}")

    panel = pd.read_parquet(DATA / "panel.parquet")
    panel = panel[panel["symbol"].isin(set(corpus["Stock_symbol"]))].copy()
    sessions = pd.DatetimeIndex(sorted(panel["session"].unique()))

    corpus["ts"] = pd.to_datetime(corpus["Date"], format="mixed", utc=True, errors="coerce")
    corpus = corpus.dropna(subset=["ts"])
    corpus["session"] = assign_session(corpus["ts"], sessions)
    corpus = corpus.dropna(subset=["session"]).rename(columns={"Stock_symbol": "symbol"})
    print(f"headlines assigned to a session: {len(corpus):,}")

    events = corpus[["symbol", "session", "nu", "mu", "s"]].reset_index(drop=True)
    events.to_parquet(ROOT / "cache" / "events.parquet", index=False)

    agg = aggregate(events, mu0=args.mu0)
    print(f"symbol-session cells with scored news: {len(agg):,}")

    panel = panel.sort_values(["symbol", "session"]).reset_index(drop=True)
    # forward cumulative market-adjusted return, strictly from d+1
    for h in HORIZONS:
        fwd = (
            panel.groupby("symbol")["ret_adj"]
            .apply(lambda x: x.shift(-1).rolling(h).sum().shift(-(h - 1)))
            .reset_index(level=0, drop=True)
        )
        panel[f"fwd{h}"] = fwd.values
        panel[f"up{h}"] = (panel[f"fwd{h}"] > 0).astype(float)
        panel.loc[panel[f"fwd{h}"].isna(), f"up{h}"] = np.nan
        # raw (unadjusted) forward return, for the drift-aware base rate
        fwdr = (
            panel.groupby("symbol")["ret"]
            .apply(lambda x: x.shift(-1).rolling(h).sum().shift(-(h - 1)))
            .reset_index(level=0, drop=True)
        )
        panel[f"fwd{h}_raw"] = fwdr.values
        panel[f"up{h}_raw"] = (panel[f"fwd{h}_raw"] > 0).astype(float)
        panel.loc[panel[f"fwd{h}_raw"].isna(), f"up{h}_raw"] = np.nan

    merged = panel.merge(agg, on=["symbol", "session"], how="left", suffixes=("", "_mig"))
    for c in ("A_mig", "A_nu", "A_mu", "pol_mean", "pol_relf", "pol_cnt", "add_comb",
              "nu_bar", "mu_bar", "frac_material"):
        merged[c] = merged[c].fillna(0.0)
    merged["n_news_scored"] = merged["n_news_mig"].fillna(0) if "n_news_mig" in merged else 0
    merged["has_scored_news"] = (merged["n_news_scored"] > 0).astype(np.int8)

    out = Path(args.out)
    merged.to_parquet(out, index=False)

    meta = {
        "symbols": int(merged["symbol"].nunique()),
        "sessions": int(merged["session"].nunique()),
        "rows": int(len(merged)),
        "rows_with_scored_news": int(merged["has_scored_news"].sum()),
        "headlines_scored": int(len(corpus)),
        "date_min": str(merged["session"].min().date()),
        "date_max": str(merged["session"].max().date()),
        "mu0": args.mu0,
        "median_headlines_per_covered_session": float(
            merged.loc[merged["has_scored_news"] == 1, "n_news_scored"].median()
        ),
    }
    (ROOT / "results" / "panel_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {out}")

    cov = merged[merged["has_scored_news"] == 1]
    print("\naggregate summary on news sessions:")
    print(cov[["A_mig", "pol_mean", "pol_relf", "add_comb", "nu_bar", "mu_bar"]]
          .describe().to_string())


if __name__ == "__main__":
    main()
