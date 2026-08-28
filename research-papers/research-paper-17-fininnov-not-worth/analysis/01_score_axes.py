"""
Step 1 - score the three MIG axes (novelty, materiality, polarity) for every
headline of the study universe with an instruction-tuned language model.

This is the measurement instrument of the paper: the same anchored prompt that the
deployed NSE system uses, applied to the FNSPID US headline corpus so that the gated
aggregate A can enter the forecasting model that produces the headline result.

Design points that matter for the paper:

  * Content-addressed cache.  Every scored item is memoised under
    sha1(symbol | normalised headline), so a run is deterministic and resumable and
    a re-run costs nothing.  This is the reproducibility claim of Section 2.2.
  * Symbol-complete ordering.  Symbols are scored one at a time to completion.  If
    the run is interrupted, the symbols that are finished are *fully* scored, so the
    aggregate A is never computed from a partially scored session.
  * Adaptive concurrency.  The API's per-minute limits are discovered at run time:
    workers are reduced on 429/503 and slowly restored, so the job self-paces
    instead of hammering the endpoint.
  * A test-retest slice.  A fixed subsample is scored a second time with the cache
    bypassed, which gives the per-axis reliability (ICC) reported in the paper.

Usage
    python 01_score_axes.py --symbols 60            # score the 60 best-covered names
    python 01_score_axes.py --retest 2000           # test-retest pass, cache bypassed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_DIR = ROOT / "cache"
DATA = ROOT.parent / "research-paper-14-aece-not-sent" / "data-cache"

MODEL = "gemini-3.6-flash"  # overridden by --model; recorded per row in the cache
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

# --------------------------------------------------------------------------------
# The anchored prompt.  Identical in structure to the deployed NSE prompt: a role, a
# definition per axis, four numerical anchors per axis, and a strict JSON contract.
# --------------------------------------------------------------------------------
PROMPT_HEADER = """You are a quantitative equity analyst. For each item below (a news \
headline about a US-listed company, given with its ticker) return three independent \
scores. Judge each axis on its own; do not let one axis contaminate another.

novelty (nu, 0.0 to 1.0): how much genuinely NEW information the item carries relative \
to what the market already knew about this company.
  0.0 = an exact restatement of news already priced weeks ago, or a generic market list
  0.2 = an in-line, widely expected result or a routine scheduled disclosure
  0.6 = an unexpected but modest development
  1.0 = a sudden shock such as a surprise regulatory ban, fraud disclosure or bid

materiality (mu, 0.0 to 1.0): how sensitive this company's fundamental value is to the \
event, IGNORING its direction and IGNORING how fresh it is.
  0.0 = boilerplate, listicles, price-move recaps, "stocks to watch" round-ups
  0.4 = a contract, product or ruling worth a few percent of revenue
  0.8 = a transformative deal, guidance reset or loss of a core franchise
  1.0 = an existential event (solvency, delisting, breakup)

polarity (s, -1.0 to 1.0): the SIGN and size of the implied change in value.
  -1.0 = fraud allegation, catastrophic loss
  -0.4 = a disappointing but survivable miss
   0.0 = a neutral reshuffle or a purely descriptive item
  +0.4 = a solid but unspectacular positive
  +1.0 = a large earnings beat or a takeover at a big premium

Return ONLY a JSON array with exactly one object per input item, in the SAME order and \
with the SAME length as the input, each of the form
{"i": <index>, "nu": <float>, "mu": <float>, "s": <float>}
No prose, no code fence, no commentary.

ITEMS:
"""

_WS = re.compile(r"\s+")


def norm_title(t: str) -> str:
    """Normalisation used by the cache key: case-folded, whitespace-collapsed."""
    return _WS.sub(" ", str(t).strip().lower())


def key_of(symbol: str, title: str) -> str:
    return hashlib.sha1(f"{symbol}|{norm_title(title)}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------
def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS scores (
               k TEXT PRIMARY KEY, symbol TEXT, title TEXT,
               nu REAL, mu REAL, s REAL, model TEXT, scored_at REAL)"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS ix_scores_symbol ON scores(symbol)")
    con.execute(
        """CREATE TABLE IF NOT EXISTS retest (
               k TEXT, symbol TEXT, title TEXT, nu REAL, mu REAL, s REAL,
               pass INTEGER, model TEXT, scored_at REAL)"""
    )
    con.commit()
    return con


# --------------------------------------------------------------------------------
# API call with backoff
# --------------------------------------------------------------------------------
class RateState:
    """Shared pacing state: a fixed worker pool plus an adaptive inter-request gap.

    Concurrency is held constant (a semaphore of `workers` permits) and the *spacing*
    between requests is what adapts.  Every throttle widens the gap multiplicatively;
    a run of clean responses narrows it again.  That keeps the job just under the
    endpoint's per-minute ceiling without the permit bookkeeping drifting.
    """

    def __init__(self, workers: int, lo: int = 1, hi: int = 12):
        self.lock = threading.Lock()
        self.permits = threading.Semaphore(workers)
        self.workers = workers
        self.lo, self.hi = lo, hi
        self.gap = float(globals().get("INITIAL_GAP", 0.0))  # spacing between requests
        self.next_slot = 0.0      # monotonic time of the next allowed request
        self.ok_since_throttle = 0
        self.throttles = 0

    def wait_turn(self):
        """Serialise request starts so they are at least `gap` seconds apart."""
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + self.gap
        sleep_for = start - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    def on_throttle(self):
        with self.lock:
            self.ok_since_throttle = 0
            self.throttles += 1
            self.gap = min(max(self.gap * 1.6, 1.0), 30.0)

    def on_success(self):
        with self.lock:
            self.ok_since_throttle += 1
            # Recover briskly.  A widened gap slows the job so much that a slow
            # recovery rule (say, 15 clean responses per step) can take the best
            # part of an hour to walk back down from a single throttle burst.
            if self.ok_since_throttle >= 5 and self.gap > 0:
                self.gap = max(self.gap * 0.6, 0.0 if self.gap < 0.3 else 0.3)
                self.ok_since_throttle = 0


class DailyQuotaExhausted(RuntimeError):
    """Raised when the API's per-day request cap is hit; the run must stop, not retry."""


def call_model(api_key: str, items: list[dict], rate: RateState, max_tries: int = 7):
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": PROMPT_HEADER + json.dumps(items, ensure_ascii=False)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        }
    ).encode("utf-8")

    delay = 4.0
    for attempt in range(max_tries):
        rate.wait_turn()
        req = urllib.request.Request(
            ENDPOINT.format(m=MODEL),
            data=payload,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.load(resp)
            parts = body["candidates"][0]["content"]["parts"]
            text = parts[-1]["text"]
            out = json.loads(text)
            rate.on_success()
            return out
        except urllib.error.HTTPError as exc:
            code = exc.code
            try:
                raw = exc.read().decode()
            except Exception:
                raw = ""
            if code == 429 and "PerDay" in raw:
                # A daily cap is not something backoff can wait out inside a run.
                raise DailyQuotaExhausted(raw[:400])
            if code in (429, 500, 502, 503, 504):
                rate.on_throttle()
                time.sleep(delay + random.uniform(0, 3))
                delay = min(delay * 1.8, 90)
                continue
            raise
        except Exception:
            time.sleep(delay + random.uniform(0, 3))
            delay = min(delay * 1.8, 90)
            continue
    return None


def coerce(rec: dict) -> tuple[float, float, float] | None:
    try:
        nu = float(rec["nu"])
        mu = float(rec["mu"])
        s = float(rec["s"])
    except Exception:
        return None
    if not all(map(lambda v: v == v, (nu, mu, s))):  # NaN guard
        return None
    return min(max(nu, 0.0), 1.0), min(max(mu, 0.0), 1.0), min(max(s, -1.0), 1.0)


# --------------------------------------------------------------------------------
# Main scoring loop
# --------------------------------------------------------------------------------
def load_universe(n_symbols: int) -> pd.DataFrame:
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    panel = pd.read_parquet(DATA / "panel.parquet", columns=["symbol"])
    keep = set(panel["symbol"].unique())
    corpus = corpus[corpus["Stock_symbol"].isin(keep)].copy()

    counts = corpus["Stock_symbol"].value_counts()
    chosen = list(counts.head(n_symbols).index)
    corpus = corpus[corpus["Stock_symbol"].isin(chosen)].copy()

    corpus["k"] = [
        key_of(sy, ti) for sy, ti in zip(corpus["Stock_symbol"], corpus["Article_title"])
    ]
    pairs = corpus.drop_duplicates("k")[["k", "Stock_symbol", "Article_title"]]
    pairs = pairs.rename(columns={"Stock_symbol": "symbol", "Article_title": "title"})
    # symbol-complete ordering: best-covered symbols first
    order = {s: i for i, s in enumerate(chosen)}
    pairs["ord"] = pairs["symbol"].map(order)
    return pairs.sort_values(["ord", "title"]).reset_index(drop=True)


def run_scoring(api_key: str, n_symbols: int, batch: int, workers: int, status_path: Path):
    con = open_cache(CACHE_DIR / "scores.db")
    pairs = load_universe(n_symbols)
    done = {r[0] for r in con.execute("SELECT k FROM scores")}
    todo = pairs[~pairs["k"].isin(done)].reset_index(drop=True)

    print(f"universe: {pairs['symbol'].nunique()} symbols, {len(pairs)} (symbol,title) pairs")
    print(f"cached:   {len(done)}   to score: {len(todo)}", flush=True)
    if todo.empty:
        return

    batches = [todo.iloc[i : i + batch] for i in range(0, len(todo), batch)]
    rate = RateState(workers)
    write_lock = threading.Lock()
    counter = {"n": 0, "fail": 0, "t0": time.time()}
    work: "queue.Queue" = queue.Queue()
    for b in batches:
        work.put(b)

    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                chunk = work.get_nowait()
            except queue.Empty:
                return
            rate.permits.acquire()
            try:
                items = [
                    {"i": i, "sym": sy, "h": str(ti)[:300]}
                    for i, (sy, ti) in enumerate(zip(chunk["symbol"], chunk["title"]))
                ]
                try:
                    out = call_model(api_key, items, rate)
                except DailyQuotaExhausted as exc:
                    if not stop.is_set():
                        print(f"DAILY QUOTA EXHAUSTED for {MODEL}: {exc}", flush=True)
                    stop.set()
                    return
                rows = []
                if out and isinstance(out, list):
                    by_i = {}
                    for rec in out:
                        if isinstance(rec, dict) and "i" in rec:
                            try:
                                by_i[int(rec["i"])] = rec
                            except Exception:
                                pass
                    for i, (k, sy, ti) in enumerate(
                        zip(chunk["k"], chunk["symbol"], chunk["title"])
                    ):
                        rec = by_i.get(i)
                        if rec is None:
                            continue
                        v = coerce(rec)
                        if v is None:
                            continue
                        rows.append((k, sy, str(ti), v[0], v[1], v[2], MODEL, time.time()))
                with write_lock:
                    if rows:
                        con.executemany(
                            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)", rows
                        )
                        con.commit()
                    counter["n"] += len(rows)
                    counter["fail"] += len(chunk) - len(rows)
                    el = time.time() - counter["t0"]
                    rpm = counter["n"] / el * 60 if el > 0 else 0
                    status = {
                        "scored_this_run": counter["n"],
                        "failed": counter["fail"],
                        "cached_before": len(done),
                        "total_target": len(pairs),
                        "elapsed_min": round(el / 60, 1),
                        "headlines_per_min": round(rpm),
                        "workers": rate.workers,
                        "gap_s": round(rate.gap, 2),
                        "throttles": rate.throttles,
                        "eta_min": round((len(todo) - counter["n"]) / rpm, 1) if rpm > 0 else None,
                    }
                    status_path.write_text(json.dumps(status, indent=1))
                    if counter["n"] % (batch * 10) < len(rows):
                        print(json.dumps(status), flush=True)
            finally:
                rate.permits.release()
                work.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(rate.hi)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("scoring pass complete:", json.dumps(json.loads(status_path.read_text())), flush=True)


def run_retest(api_key: str, n: int, batch: int, workers: int):
    """Second independent pass over a fixed subsample, cache bypassed -> reliability."""
    con = open_cache(CACHE_DIR / "scores.db")
    have = pd.read_sql("SELECT k, symbol, title FROM scores", con)
    if have.empty:
        print("nothing scored yet")
        return
    sample = have.sample(min(n, len(have)), random_state=17).reset_index(drop=True)
    rate = RateState(workers)
    rows = []
    for i in range(0, len(sample), batch):
        chunk = sample.iloc[i : i + batch]
        items = [
            {"i": j, "sym": sy, "h": str(ti)[:300]}
            for j, (sy, ti) in enumerate(zip(chunk["symbol"], chunk["title"]))
        ]
        try:
            out = call_model(api_key, items, rate)
        except DailyQuotaExhausted as exc:
            # Stop cleanly and keep what was already scored: the rows collected so
            # far are perfectly good, and discarding them on the way out would
            # throw away the whole pass for the sake of its unfinished tail.
            print(f"daily quota reached after {len(rows)} rows: {str(exc)[:120]}", flush=True)
            break
        if not out:
            continue
        by_i = {int(r["i"]): r for r in out if isinstance(r, dict) and "i" in r}
        for j, (k, sy, ti) in enumerate(zip(chunk["k"], chunk["symbol"], chunk["title"])):
            rec = by_i.get(j)
            if rec is None:
                continue
            v = coerce(rec)
            if v is None:
                continue
            rows.append((k, sy, str(ti), v[0], v[1], v[2], 2, MODEL, time.time()))
        print(f"retest {min(i + batch, len(sample))}/{len(sample)}", flush=True)
    con.executemany("INSERT INTO retest VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    print("retest rows:", len(rows))


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=60)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retest", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gap", type=float, default=0.0, help="initial seconds between requests")
    args = ap.parse_args()

    MODEL = args.model
    globals()["INITIAL_GAP"] = args.gap

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT.parent.parent / ".env", override=True)
    except Exception:
        pass
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        sys.exit("GEMINI_API_KEY not set")

    if args.retest:
        run_retest(api_key, args.retest, args.batch, args.workers)
    else:
        run_scoring(
            api_key, args.symbols, args.batch, args.workers, CACHE_DIR / "score_status.json"
        )


if __name__ == "__main__":
    main()
