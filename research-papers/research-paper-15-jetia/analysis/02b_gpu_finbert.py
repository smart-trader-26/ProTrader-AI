"""FinBERT scores for every unique headline, on the GPU, in one pass.

Writes m1_gpu.parquet.  The merge step in 02_score_sentiment.py overrides
the m1 column with these values wherever they exist, so the transformer
measurement is homogeneous (same fp16 model) across the whole corpus
rather than a mix of CPU-quantised and GPU runs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=40)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA not available"
    torch.set_grad_enabled(False)
    print("device:", torch.cuda.get_device_name(0), flush=True)

    corpus = pd.read_parquet(os.path.join(args.indir, "corpus.parquet"),
                             columns=["Article_title"])
    corpus["Article_title"] = corpus["Article_title"].fillna("").str.strip()
    texts = sorted(corpus.loc[corpus["Article_title"].str.len() > 0,
                              "Article_title"].unique())
    out_path = os.path.join(args.indir, "m1_gpu.parquet")
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_parquet(out_path,
                                   columns=["Article_title"])
                   ["Article_title"])
        texts = [t for t in texts if t not in done]
    print(f"to score on GPU: {len(texts):,} (already done {len(done):,})",
          flush=True)
    if not texts:
        return

    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = (AutoModelForSequenceClassification.from_pretrained(name)
           .eval().half().to("cuda"))
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    ip, ineg = lab["positive"], lab["negative"]

    scores = np.empty(len(texts), dtype=np.float32)
    t0 = time.time()
    B = args.batch
    flush_every = 50000
    written = 0
    parts = ([pd.read_parquet(out_path)] if done else [])
    for s in range(0, len(texts), B):
        chunk = texts[s:s + B]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        p = torch.softmax(mdl(**enc).logits.float(), dim=-1)
        scores[s:s + B] = (p[:, ip] - p[:, ineg]).cpu().numpy()
        n = min(s + B, len(texts))
        if n - written >= flush_every or n == len(texts):
            frame = pd.DataFrame({"Article_title": texts[:n],
                                  "m1_finbert": scores[:n]})
            pd.concat(parts + [frame], ignore_index=True).to_parquet(
                out_path, index=False)
            written = n
            el = time.time() - t0
            print(f"  {n:,}/{len(texts):,}  {n/el:.0f} hdl/s  "
                  f"eta {(len(texts)-n)/max(n/el,1)/60:.1f} min", flush=True)
    print("GPU scoring done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
