"""Step 2: four mutually independent sentiment measurements of every headline.

m1 : FinBERT (ProsusAI/finbert)      transformer, financial phrase corpus
m2 : Loughran-McDonald lexicon       finance-specific word lists
m3 : VADER                           rule-based valence with intensifiers
m4 : Harvard General Inquirer IV-4   general-purpose psychosocial lexicon

The scorers come from four different families and share neither training
data nor vocabulary construction, which is what licenses the errors-in-
variables treatment in step 4: each is a noisy indicator of one latent
sentiment.  Three indicators identify the reliability ratio; the fourth
makes the one-factor model over-identified and therefore testable.

Unique headline strings are scored once and mapped back, because the corpus
repeats wire copy across symbols.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def score_lexicon(texts: list[str], which: str) -> np.ndarray:
    """Polarity from a bag-of-words lexicon ('LM' or 'HIV4')."""
    import pysentiment2 as ps

    lex = ps.LM() if which == "LM" else ps.HIV4()
    out = np.empty(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        sc = lex.get_score(lex.tokenize(t))
        p, n = sc["Positive"], sc["Negative"]
        out[i] = 0.0 if (p + n) == 0 else (p - n) / (p + n)
    return out


def score_vader(texts: list[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    an = SentimentIntensityAnalyzer()
    out = np.empty(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = an.polarity_scores(t)["compound"]
    return out


def score_finbert(texts: list[str], batch: int, max_len: int,
                  quantize: bool, device: str = "cpu") -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_grad_enabled(False)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    if use_cuda:
        # fp16 on GPU; dynamic int8 quantisation is a CPU-only path
        mdl = mdl.half().to("cuda")
    elif quantize:
        mdl = torch.quantization.quantize_dynamic(
            mdl, {torch.nn.Linear}, dtype=torch.qint8)
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    ip, ineg = lab["positive"], lab["negative"]

    out = np.empty(len(texts), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        if use_cuda:
            enc = {k: v.to("cuda") for k, v in enc.items()}
        p = torch.softmax(mdl(**enc).logits.float(), dim=-1)
        out[s:s + batch] = (p[:, ip] - p[:, ineg]).cpu().numpy()
        if s % (batch * 200) == 0 and s:
            el = time.time() - t0
            print(f"  {s:,}/{len(texts):,}  {s/el:.0f} hdl/s  "
                  f"eta {(len(texts)-s)/(s/el)/60:.1f} min", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="pilot mode: score only this many unique headlines")
    ap.add_argument("--shard", default="",
                    help="'i/n': score only the i-th of n shards of the "
                         "remaining headlines, writing its own cache file")
    ap.add_argument("--checkpoint-every", type=int, default=20000,
                    help="flush scores to disk every this many headlines")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--skip-finbert", action="store_true",
                    help="score lexicons only; m1 is filled later "
                         "from the GPU pass (m1_gpu.parquet)")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    corpus = pd.read_parquet(os.path.join(args.indir, "corpus.parquet"))
    corpus["Article_title"] = corpus["Article_title"].fillna("").str.strip()
    corpus = corpus[corpus["Article_title"].str.len() > 0]
    print(f"corpus rows: {len(corpus):,}")

    uniq = pd.Index(corpus["Article_title"].unique())
    print(f"unique headlines: {len(uniq):,} "
          f"({100*len(uniq)/len(corpus):.1f}% of rows)")
    if args.sample and args.sample < len(uniq):
        rng = np.random.default_rng(20260725)
        uniq = pd.Index(rng.choice(uniq.to_numpy(), args.sample, replace=False))
        corpus = corpus[corpus["Article_title"].isin(set(uniq))]
        print(f"PILOT: {len(uniq):,} headlines, {len(corpus):,} rows")

    import glob as _glob

    cache = os.path.join(args.indir, "headline_scores.parquet")
    COLS = ["Article_title", "m1_finbert", "m2_lm", "m3_vader", "m4_hiv4"]

    def read_all_caches() -> pd.DataFrame:
        frames = []
        for p in [cache] + sorted(
                _glob.glob(os.path.join(args.indir,
                                        "headline_scores_shard*.parquet"))):
            if os.path.exists(p):
                frames.append(pd.read_parquet(p))
        if not frames:
            return pd.DataFrame(columns=COLS)
        return (pd.concat(frames, ignore_index=True)
                .drop_duplicates("Article_title"))

    have = read_all_caches()
    todo = uniq.difference(pd.Index(have["Article_title"]))
    print(f"cached: {len(have):,}   to score: {len(todo):,}", flush=True)

    if args.shard and len(todo):
        i, n = (int(x) for x in args.shard.split("/"))
        todo = pd.Index(sorted(todo))          # deterministic split
        todo = todo[i::n]
        out_path = os.path.join(args.indir,
                                f"headline_scores_shard{i}.parquet")
        print(f"shard {i}/{n}: {len(todo):,} headlines -> {out_path}",
              flush=True)
        # append semantics: never drop rows an earlier run already scored
        done_parts = ([pd.read_parquet(out_path)]
                      if os.path.exists(out_path) else [])
        texts_all = todo.tolist()
        step = max(args.checkpoint_every, args.batch)
        for s in range(0, len(texts_all), step):
            texts = texts_all[s:s + step]
            m2 = score_lexicon(texts, "LM")
            m3 = score_vader(texts)
            m4 = score_lexicon(texts, "HIV4")
            if args.skip_finbert:
                m1 = np.full(len(texts), np.nan, dtype=np.float32)
            else:
                m1 = score_finbert(texts, args.batch, args.max_len,
                                   not args.no_quantize, args.device)
            done_parts.append(pd.DataFrame(
                {"Article_title": texts, "m1_finbert": m1, "m2_lm": m2,
                 "m3_vader": m3, "m4_hiv4": m4}))
            pd.concat(done_parts, ignore_index=True).to_parquet(
                out_path, index=False)
            print(f"  checkpoint {min(s + step, len(texts_all)):,}"
                  f"/{len(texts_all):,}", flush=True)
        print("shard done", flush=True)
        return

    if len(todo):
        # single-process path (also merges after shards ran)
        texts = sorted(todo)
        print("scoring remaining in one process ...")
        m2 = score_lexicon(texts, "LM")
        m3 = score_vader(texts)
        m4 = score_lexicon(texts, "HIV4")
        if args.skip_finbert:
            m1 = np.full(len(texts), np.nan, dtype=np.float32)
        else:
            m1 = score_finbert(texts, args.batch, args.max_len,
                               not args.no_quantize, args.device)
        have = pd.concat([have, pd.DataFrame(
            {"Article_title": texts, "m1_finbert": m1, "m2_lm": m2,
             "m3_vader": m3, "m4_hiv4": m4})], ignore_index=True)
    have = have.drop_duplicates("Article_title")
    gpu_path = os.path.join(args.indir, "m1_gpu.parquet")
    if os.path.exists(gpu_path):
        g = pd.read_parquet(gpu_path).drop_duplicates("Article_title")
        have = have.merge(g.rename(columns={"m1_finbert": "m1_gpu"}),
                          on="Article_title", how="left")
        n_over = int(have["m1_gpu"].notna().sum())
        have["m1_finbert"] = have["m1_gpu"].fillna(have["m1_finbert"])
        have = have.drop(columns=["m1_gpu"])
        print(f"m1 overridden with GPU scores for {n_over:,} headlines")
    n_nan = int(have["m1_finbert"].isna().sum())
    if n_nan:
        raise SystemExit(f"{n_nan} headlines still lack m1 - run 02b first")
    have.to_parquet(cache, index=False)
    sc = have

    df = corpus.merge(sc, on="Article_title", how="inner")
    df.to_parquet(os.path.join(args.indir, "corpus_scored.parquet"),
                  index=False)
    cols = ["m1_finbert", "m2_lm", "m3_vader", "m4_hiv4"]
    print(f"wrote corpus_scored.parquet  rows={len(df):,}")
    print(df[cols].describe())
    print("headline-level correlation matrix:")
    print(df[cols].corr().round(3))


if __name__ == "__main__":
    main()
