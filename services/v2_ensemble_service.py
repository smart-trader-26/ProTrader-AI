"""
V2 sentiment-ensemble service.

Loads the 4-learner stack + fine-tuned FinBERT from HuggingFace
(`EnteiTiger3/protrader-sentiment-v2`) on first call and caches them in the
default HF cache. Subsequent calls are in-process only.

Model contract (must mirror sentiment_analysis_v2.py → `ensemble_predict`):

  For each headline:
      sv  ∈ {-1, 0, +1}   via v2 FinBERT (pipeline)
      sc  ∈ [0, 1]        confidence of that label
      cat ∈ VALID_CATEGORIES  via keyword classifier (reused from news_sentiment)

  Aggregate across the day:
      top_cat       = mode(cat)
      w_sentiment   = Σ(sv · sc) / Σ(sc)           in [-1, +1]

  Feature row (12 cols, exact training order):
      Sentiment_<cat>: w_sentiment if cat == top_cat else 0   (6 cols)
      Count_<cat>:     1           if cat == top_cat else 0   (6 cols)

  Then each base learner emits prob_up; the LR stacker consumes those 4
  probs (in training order: LogReg, RandomForest, XGBoost, LightGBM) and
  returns the consensus.

If the stacker pkl is missing we fall back to the same weighted-average
rule the Colab UI uses (weight = |p - 0.5| · 2 + 0.1).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from config.settings import HF_REPO_ID, HF_TOKEN
from data.news_sentiment import categorize_headline
from schemas.v2_ensemble import V2EnsemblePrediction, V2ModelBreakdown

VALID_CATEGORIES = [
    "Earnings & Output", "Analyst Ratings", "Market Action",
    "Deals & Acquisitions", "Macro & Policy", "Other",
]
SENTIMENT_FEATURES = [f"Sentiment_{c}" for c in VALID_CATEGORIES]
COUNT_FEATURES = [f"Count_{c}" for c in VALID_CATEGORIES]
ALL_FEATURES = SENTIMENT_FEATURES + COUNT_FEATURES

# Base-learner order matters: the stacker was trained on this exact column order.
_BASE_ORDER = ("logreg", "random_forest", "xgboost", "lightgbm")

_PKL_FILES = {
    "logreg":        "master_lr.pkl",
    "random_forest": "master_rf.pkl",
    "xgboost":       "master_xgb.pkl",
    "lightgbm":      "master_lgb.pkl",
}
_STACKER_FILE = "master_stacker.pkl"
_SENT_DIR = "sentiment_model"

_MODEL_VERSION = "v2-finbert-4lrn-stack"


@dataclass
class _V2Bundle:
    learners: dict  # {'logreg': est, ...}
    stacker: object | None
    sentiment_pipe: object


_bundle: _V2Bundle | None = None
_load_lock = threading.Lock()


def _download(filename: str) -> str:
    """Download one file from the repo; returns local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename,
        token=HF_TOKEN or None, repo_type="model",
    )


def _snapshot_sentiment_dir() -> str:
    """Download the `sentiment_model/` subdir; returns its local path."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(
        repo_id=HF_REPO_ID, token=HF_TOKEN or None, repo_type="model",
        allow_patterns=f"{_SENT_DIR}/*",
    )
    return os.path.join(root, _SENT_DIR)


def _load() -> _V2Bundle:
    """Load + cache the full v2 bundle. Thread-safe, idempotent."""
    global _bundle
    if _bundle is not None:
        return _bundle
    with _load_lock:
        if _bundle is not None:
            return _bundle

        from transformers import pipeline

        learners = {name: joblib.load(_download(f)) for name, f in _PKL_FILES.items()}

        try:
            stacker = joblib.load(_download(_STACKER_FILE))
        except Exception:
            stacker = None

        sent_dir = _snapshot_sentiment_dir()
        # framework="pt" is a hard invariant — see CLAUDE.md §2.
        sentiment_pipe = pipeline(
            "sentiment-analysis", model=sent_dir, tokenizer=sent_dir, framework="pt",
        )

        _bundle = _V2Bundle(learners=learners, stacker=stacker, sentiment_pipe=sentiment_pipe)
        return _bundle


def _sv_from_label(label: str) -> int:
    norm = (label or "").lower()
    if norm in ("positive", "pos", "label_2"):
        return 1
    if norm in ("negative", "neg", "label_0"):
        return -1
    return 0


def _score_headline(text: str, pipe) -> tuple[int, float, str]:
    """Return (sv, sc, category) for a single headline."""
    if not text:
        return 0, 0.0, "Other"
    try:
        out = pipe(text[:512])[0]
        sv = _sv_from_label(out.get("label", "neutral"))
        sc = float(out.get("score", 0.5))
    except Exception:
        sv, sc = 0, 0.5
    return sv, sc, categorize_headline(text)


def _build_feature_row(top_cat: str, w_sentiment: float) -> pd.DataFrame:
    row = {f"Sentiment_{c}": (w_sentiment if c == top_cat else 0.0) for c in VALID_CATEGORIES}
    row.update({f"Count_{c}": (1 if c == top_cat else 0) for c in VALID_CATEGORIES})
    return pd.DataFrame([row])[ALL_FEATURES].astype("float32")


def _ensemble(feature_row: pd.DataFrame, bundle: _V2Bundle) -> tuple[dict[str, float], float, bool]:
    probs = {
        name: float(bundle.learners[name].predict_proba(feature_row)[0][1])
        for name in _BASE_ORDER
    }
    if bundle.stacker is not None:
        stack_in = np.array([[probs[n] for n in _BASE_ORDER]], dtype="float32")
        consensus = float(bundle.stacker.predict_proba(stack_in)[0][1])
        return probs, consensus, True
    # Weighted-average fallback — matches the Colab UI.
    vals = [probs[n] for n in _BASE_ORDER]
    weights = [abs(p - 0.5) * 2 + 0.1 for p in vals]
    consensus = sum(p * w for p, w in zip(vals, weights)) / sum(weights)
    return probs, float(consensus), False


def predict_v2(ticker: str, headlines: list[dict]) -> V2EnsemblePrediction:
    """
    Run the v2 ensemble on a set of headlines.

    Args:
        ticker: e.g. "RELIANCE.NS"
        headlines: list of dicts, each with a "title" key (other keys ignored).
            Accepts title-only or {"title": ..., "description": ...}.

    Returns:
        V2EnsemblePrediction with prob_up, breakdown, aggregates, and
        whether the stacker was used (vs weighted-average fallback).

    Raises:
        ValueError: if `headlines` is empty (caller should short-circuit
            before loading the model to save a download round-trip).
    """
    if not headlines:
        raise ValueError("predict_v2 requires at least one headline")

    bundle = _load()

    svs: list[float] = []
    scs: list[float] = []
    cats: list[str] = []
    for h in headlines:
        title = (h.get("title") or "").strip()
        desc = (h.get("description") or "").strip()
        text = f"{title}. {desc[:400]}" if desc else title
        if not text:
            continue
        sv, sc, cat = _score_headline(text, bundle.sentiment_pipe)
        svs.append(sv)
        scs.append(sc)
        cats.append(cat)

    if not svs:
        raise ValueError("predict_v2: all headlines were empty after cleaning")

    svs_arr = np.asarray(svs, dtype=float)
    scs_arr = np.asarray(scs, dtype=float)
    total_w = float(scs_arr.sum())
    w_sent = float((svs_arr * scs_arr).sum() / total_w) if total_w > 0 else 0.0
    w_sent = max(-1.0, min(1.0, w_sent))

    cat_series = pd.Series(cats)
    top_cat = cat_series.mode().iloc[0]
    cat_counts = cat_series.value_counts().reindex(VALID_CATEGORIES, fill_value=0)

    feature_row = _build_feature_row(top_cat, w_sent)
    probs, prob_up, stacker_ok = _ensemble(feature_row, bundle)

    return V2EnsemblePrediction(
        ticker=ticker,
        made_at=datetime.now(UTC),
        n_headlines=len(svs),
        top_category=top_cat,
        weighted_sentiment=w_sent,
        category_counts={k: int(v) for k, v in cat_counts.items()},
        prob_up=prob_up,
        model_breakdown=V2ModelBreakdown(
            logreg=probs["logreg"],
            random_forest=probs["random_forest"],
            xgboost=probs["xgboost"],
            lightgbm=probs["lightgbm"],
        ),
        stacker_available=stacker_ok,
        model_version=_MODEL_VERSION,
    )


def is_configured() -> bool:
    """True if HF_TOKEN is present — gate UI rendering cleanly when missing."""
    return bool(HF_TOKEN) and bool(HF_REPO_ID)
