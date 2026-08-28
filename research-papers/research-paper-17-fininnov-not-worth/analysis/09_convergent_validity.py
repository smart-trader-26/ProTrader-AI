"""
Step 9 - do the two gates measure different things, or is the instrument
collapsing them?

Novelty and materiality, as scored by the language model, correlate at about 0.87.
That single fact drives every null in this study, and it has two very different
explanations:

  (world)      genuinely new firm news usually *is* the consequential news, so the
               two properties really do travel together in this corpus;
  (instrument) the model cannot hold two related concepts apart under one prompt
               and is effectively reporting one "importance" score twice.

The two are distinguishable without any further model calls, because each axis has
an external criterion it should track *if* it is measuring what it claims:

  * Novelty should track **mechanical staleness**.  Following the stale-news
    literature, we measure how much a headline repeats what was already written
    about the same ticker: one minus the maximum TF-IDF cosine similarity to that
    ticker's headlines over the preceding 30 days.  This is computed only from
    text published strictly *before* the headline, so it is causal and needs no
    returns.
  * Materiality should track **outcome magnitude**, and specifically it should
    track it in a way novelty does not.  Materiality is defined as the sensitivity
    of value to the event regardless of direction, so it should predict the
    *absolute* market-adjusted return of the session, independently of sign.

The decisive quantities are therefore partial, not raw.  If the axes are distinct,
novelty should predict mechanical staleness after controlling for materiality, and
materiality should predict absolute return after controlling for novelty.  If the
instrument is collapsing them, neither partial relationship survives and the two
columns are interchangeable.

Note on what may be used where: the absolute-return criterion is used *only* to
validate the instrument, never as an input to any signal.  Using an outcome to
check whether a measurement means what it says is legitimate; using it to build
the measurement would not be.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
DATA = ROOT.parent / "research-paper-14-aece-not-sent" / "data-cache"

WINDOW_DAYS = 30
_WS = re.compile(r"\s+")


def key_of(symbol: str, title: str) -> str:
    return hashlib.sha1(
        f"{symbol}|{_WS.sub(' ', str(title).strip().lower())}".encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
def load_scored_corpus() -> pd.DataFrame:
    con = sqlite3.connect(str(ROOT / "cache" / "scores.db"))
    scores = pd.read_sql("SELECT k, nu, mu, s FROM scores", con)
    con.close()

    panel_syms = set(pd.read_parquet(ROOT / "cache" / "mig_panel.parquet",
                                     columns=["symbol"])["symbol"].unique())
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    corpus = corpus[corpus["Stock_symbol"].isin(panel_syms)].copy()
    corpus["k"] = [key_of(sy, ti) for sy, ti in
                   zip(corpus["Stock_symbol"], corpus["Article_title"])]
    corpus = corpus.merge(scores, on="k", how="inner")
    corpus["ts"] = pd.to_datetime(corpus["Date"], format="mixed", utc=True, errors="coerce")
    corpus = corpus.dropna(subset=["ts"]).sort_values(["Stock_symbol", "ts"])
    return corpus.rename(columns={"Stock_symbol": "symbol", "Article_title": "title"})


def mechanical_staleness(corpus: pd.DataFrame) -> pd.DataFrame:
    """One minus the max TF-IDF cosine similarity to the same ticker's prior 30 days.

    The vectoriser is fitted once on the whole corpus so that inverse document
    frequencies are comparable across tickers, and the comparison set for each
    headline is strictly earlier in time, so nothing here can see the future.
    """
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", sublinear_tf=True,
        min_df=2, max_df=0.4, ngram_range=(1, 2), dtype=np.float32,
    )
    X = vec.fit_transform(corpus["title"].astype(str))
    X = X.tocsr()
    print(f"tf-idf matrix: {X.shape[0]:,} x {X.shape[1]:,}")

    out_sim = np.full(len(corpus), np.nan, dtype=np.float64)
    out_cnt = np.zeros(len(corpus), dtype=np.int32)

    pos = 0
    for sym, grp in corpus.groupby("symbol", sort=False):
        idx = np.arange(pos, pos + len(grp))
        pos += len(grp)
        ts = grp["ts"].to_numpy()
        Xs = X[idx]
        lo = 0
        for i in range(len(grp)):
            cutoff = ts[i] - np.timedelta64(WINDOW_DAYS, "D")
            while lo < i and ts[lo] < cutoff:
                lo += 1
            if i == lo:                      # nothing published in the window
                continue
            sims = (Xs[lo:i] @ Xs[i].T).toarray().ravel()
            out_cnt[idx[i]] = i - lo
            if sims.size:
                out_sim[idx[i]] = float(sims.max())
        print(f"  {sym}: {len(grp):,} headlines", flush=True)

    corpus = corpus.copy()
    corpus["max_sim_30d"] = out_sim
    corpus["n_prior_30d"] = out_cnt
    corpus["nu_mech"] = 1.0 - corpus["max_sim_30d"]
    return corpus


# ---------------------------------------------------------------------------
def partial_corr(df: pd.DataFrame, x: str, y: str, z: str) -> float:
    """Correlation of x and y after linearly removing z from both."""
    d = df[[x, y, z]].dropna()
    if len(d) < 100:
        return np.nan
    zz = np.column_stack([np.ones(len(d)), d[z].to_numpy()])
    rx = d[x].to_numpy() - zz @ np.linalg.lstsq(zz, d[x].to_numpy(), rcond=None)[0]
    ry = d[y].to_numpy() - zz @ np.linalg.lstsq(zz, d[y].to_numpy(), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_t(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with standard errors clustered on one dimension (date)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    codes = pd.factorize(groups)[0]
    order = np.argsort(codes, kind="stable")
    Xs, us, gs = X[order], resid[order], codes[order]
    meat = np.zeros((X.shape[1], X.shape[1]))
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(gs)]):
        sg = Xs[lo:hi].T @ us[lo:hi]
        meat += np.outer(sg, sg)
    cov = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def standardise(v: np.ndarray) -> np.ndarray:
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    corpus = load_scored_corpus()
    print(f"scored headlines with timestamps: {len(corpus):,}")

    cache = ROOT / "cache" / "staleness.parquet"
    if cache.exists():
        corpus = pd.read_parquet(cache)
        print("loaded cached staleness")
    else:
        corpus = mechanical_staleness(corpus)
        corpus.to_parquet(cache, index=False)

    have = corpus.dropna(subset=["nu_mech"])
    print(f"headlines with a prior-30d comparison set: {len(have):,} "
          f"({100 * len(have) / len(corpus):.1f}%)")

    out: dict = {
        "n_headlines": int(len(corpus)),
        "n_with_prior": int(len(have)),
        "window_days": WINDOW_DAYS,
        "median_prior_docs": float(have["n_prior_30d"].median()),
    }

    # ---- 1. raw and partial correlations against mechanical staleness -----
    out["corr_nu_mu"] = float(corpus["nu"].corr(corpus["mu"]))
    out["raw"] = {
        "nu_llm_vs_nu_mech": float(have["nu"].corr(have["nu_mech"], method="spearman")),
        "mu_llm_vs_nu_mech": float(have["mu"].corr(have["nu_mech"], method="spearman")),
        "s_llm_vs_nu_mech": float(have["s"].abs().corr(have["nu_mech"], method="spearman")),
    }
    out["partial"] = {
        "nu_vs_mech_given_mu": partial_corr(have, "nu", "nu_mech", "mu"),
        "mu_vs_mech_given_nu": partial_corr(have, "mu", "nu_mech", "nu"),
    }
    print("\n=== criterion 1: mechanical staleness ===")
    print(json.dumps({"raw": out["raw"], "partial": out["partial"]}, indent=1))

    # ---- 2. the shape of the relationship, before any linear summary ------
    # A raw rank correlation is the wrong summary if the relationship is not
    # monotone, so the decile profile is computed first and reported as such.
    prof = have.copy()
    prof["decile"] = pd.qcut(prof["nu_mech"], 10, labels=False, duplicates="drop")
    g = prof.groupby("decile").agg(
        n=("nu", "size"), nu=("nu", "mean"), mu=("mu", "mean"),
        max_sim=("max_sim_30d", "mean"), prior=("n_prior_30d", "mean"),
    )
    out["decile_profile"] = g.reset_index().to_dict(orient="records")
    print("\nmean LLM novelty by decile of mechanical novelty:")
    print(g.to_string(float_format=lambda v: f"{v:,.3f}"))
    peak = int(g["nu"].idxmax())
    out["profile_shape"] = {
        "peak_decile": peak,
        "nu_at_peak": float(g.loc[peak, "nu"]),
        "nu_at_stalest": float(g.loc[g.index.min(), "nu"]),
        "nu_at_freshest": float(g.loc[g.index.max(), "nu"]),
        "monotone_increasing": bool(g["nu"].is_monotonic_increasing),
    }

    # ---- 3. horse race for the staleness criterion ------------------------
    # The size of the comparison set is a mechanical confound: a headline with few
    # prior documents cannot match anything and so scores as fresh by construction.
    d = have.dropna(subset=["nu", "mu", "nu_mech"]).copy()
    d["date"] = d["ts"].dt.date
    d["log_prior"] = np.log1p(d["n_prior_30d"].to_numpy())
    X = np.column_stack([np.ones(len(d)), standardise(d["nu"].to_numpy()),
                         standardise(d["mu"].to_numpy()),
                         standardise(d["log_prior"].to_numpy())])
    beta, se = cluster_t(standardise(d["nu_mech"].to_numpy()), X, d["date"].to_numpy())
    out["staleness_regression"] = {
        "n": int(len(d)),
        "nu_beta": float(beta[1]), "nu_t": float(beta[1] / se[1]),
        "mu_beta": float(beta[2]), "mu_t": float(beta[2] / se[2]),
        "logprior_beta": float(beta[3]), "logprior_t": float(beta[3] / se[3]),
    }
    print("\nstaleness ~ nu + mu + log(prior docs) :",
          json.dumps(out["staleness_regression"], indent=1))

    # restricting to headlines with a substantial comparison set removes the
    # sparse-window artefact that drives the raw rank correlation negative
    dense = have[have["n_prior_30d"] >= 20]
    out["raw"]["nu_llm_vs_nu_mech_dense"] = float(
        dense["nu"].corr(dense["nu_mech"], method="spearman"))
    out["raw"]["mu_llm_vs_nu_mech_dense"] = float(
        dense["mu"].corr(dense["nu_mech"], method="spearman"))
    out["raw"]["n_dense"] = int(len(dense))
    print(f"dense subsample (>=20 prior docs, n={len(dense):,}): "
          f"nu {out['raw']['nu_llm_vs_nu_mech_dense']:.3f}, "
          f"mu {out['raw']['mu_llm_vs_nu_mech_dense']:.3f}")

    # ---- 3. materiality against outcome magnitude -------------------------
    panel = pd.read_parquet(ROOT / "cache" / "mig_panel.parquet",
                            columns=["symbol", "session", "ret_adj", "lrv_innov"])
    ev = pd.read_parquet(ROOT / "cache" / "events.parquet")
    ev = ev.merge(panel, on=["symbol", "session"], how="inner").dropna(subset=["ret_adj"])
    ev["abs_ret"] = ev["ret_adj"].abs()
    print(f"\n=== criterion 2: outcome magnitude ({len(ev):,} events) ===")

    out["raw"]["nu_llm_vs_absret"] = float(ev["nu"].corr(ev["abs_ret"], method="spearman"))
    out["raw"]["mu_llm_vs_absret"] = float(ev["mu"].corr(ev["abs_ret"], method="spearman"))
    out["partial"]["mu_vs_absret_given_nu"] = partial_corr(ev, "mu", "abs_ret", "nu")
    out["partial"]["nu_vs_absret_given_mu"] = partial_corr(ev, "nu", "abs_ret", "mu")

    X = np.column_stack([np.ones(len(ev)), standardise(ev["nu"].to_numpy()),
                         standardise(ev["mu"].to_numpy())])
    beta, se = cluster_t(ev["abs_ret"].to_numpy() * 1e4, X, ev["session"].to_numpy())
    # Also fit with a standardised dependent variable, so that the coefficients on
    # the two criteria are expressed in the same units (SDs of criterion per SD of
    # axis) and can honestly share one axis in the figure.
    beta_z, se_z = cluster_t(standardise(ev["abs_ret"].to_numpy()), X,
                             ev["session"].to_numpy())
    out["absret_regression"] = {
        "n": int(len(ev)),
        "nu_beta_bps": float(beta[1]), "nu_t": float(beta[1] / se[1]),
        "mu_beta_bps": float(beta[2]), "mu_t": float(beta[2] / se[2]),
        "nu_beta_z": float(beta_z[1]), "nu_t_z": float(beta_z[1] / se_z[1]),
        "mu_beta_z": float(beta_z[2]), "mu_t_z": float(beta_z[2] / se_z[2]),
    }
    print("|ret| ~ nu + mu :", json.dumps(out["absret_regression"], indent=1))

    # ---- 4. the sanity check: near-duplicate headlines --------------------
    dup = have[have["max_sim_30d"] > 0.8]
    fresh = have[have["max_sim_30d"] < 0.2]
    out["duplicates"] = {
        "n_near_duplicate": int(len(dup)),
        "mean_nu_near_duplicate": float(dup["nu"].mean()) if len(dup) else None,
        "mean_nu_fresh": float(fresh["nu"].mean()) if len(fresh) else None,
        "mean_mu_near_duplicate": float(dup["mu"].mean()) if len(dup) else None,
        "mean_mu_fresh": float(fresh["mu"].mean()) if len(fresh) else None,
    }
    print("\n=== near-duplicate check ===")
    print(json.dumps(out["duplicates"], indent=1))

    (RESULTS / "convergent_validity.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULTS / 'convergent_validity.json'}")


if __name__ == "__main__":
    main()
