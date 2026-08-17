"""Live progress of the paper-14 pipeline. Run any time:

    .venv\\Scripts\\python.exe research-papers\\research-paper-14-aece\\progress.py
"""
import datetime
import glob
import os
import subprocess
import sys

VENV_PY = r"c:\Users\divya\Desktop\finance\.venv\Scripts\python.exe"
try:
    import pandas as pd
    import pyarrow  # noqa: F401  (parquet engine)
except ImportError:
    # launched with a Python that lacks pyarrow: re-run in the project venv
    if os.path.exists(VENV_PY) and os.path.abspath(sys.executable) != \
            os.path.abspath(VENV_PY):
        raise SystemExit(subprocess.call([VENV_PY] + sys.argv))
    raise

W = ("C:/Users/divya/AppData/Local/Temp/claude/"
     "c--Users-divya-Desktop-finance/"
     "46e0220a-4c07-4d85-a952-1dd92b27aa0d/scratchpad/work")
HERE = os.path.dirname(os.path.abspath(__file__))

corpus = pd.read_parquet(os.path.join(W, "corpus.parquet"),
                         columns=["Article_title"])
total = corpus["Article_title"].str.strip().nunique()

cached = 0
p = os.path.join(W, "headline_scores.parquet")
if os.path.exists(p):
    cached = len(pd.read_parquet(p, columns=["Article_title"]))

shard_done, latest = 0, None
for f in glob.glob(os.path.join(W, "headline_scores_shard*.parquet")):
    shard_done += len(pd.read_parquet(f, columns=["Article_title"]))
    m = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    latest = m if latest is None or m > latest else latest

done = min(cached + shard_done, total)
todo = total - done

gpu_done = 0
g = os.path.join(W, "m1_gpu.parquet")
if os.path.exists(g):
    gpu_done = len(pd.read_parquet(g, columns=["Article_title"]))

print(f"unique headlines      : {total:>9,}")
print(f"lexicon scores (CPU)  : {done:>9,}  ({100 * done / total:.1f}%)")
print(f"FinBERT scores (GPU)  : {gpu_done:>9,}  "
      f"({100 * gpu_done / total:.1f}%)"
      + ("" if gpu_done else "  [waiting on CUDA install]"))
print(f"last checkpoint       : {latest:%H:%M:%S}" if latest else "")
remain = max(todo / 600, (total - gpu_done) / 800) / 60
print(f"scoring ETA           : ~{remain:.0f} min scoring, then "
      f"~60-70 min analysis & manuscript")

res = os.path.join(HERE, "results")
for tag, label in [("kernel.json", "kernel (return)"),
                   ("kernel_lrv_innov.json", "kernel (volatility)"),
                   ("filter_comparison.csv", "filters (return)"),
                   ("filter_comparison_lrv_innov.csv", "filters (vol)"),
                   ("closed_loop.json", "loop (return)"),
                   ("closed_loop_lrv_innov.json", "loop (vol)")]:
    f = os.path.join(res, tag)
    if os.path.exists(f):
        m = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        stale = " (pilot)" if m.date() < datetime.date.today() else ""
        print(f"  {label:<22} written {m:%H:%M}{stale}")

doc = os.path.join(HERE, "aece_matched_filter.docx")
if os.path.exists(doc):
    m = datetime.datetime.fromtimestamp(os.path.getmtime(doc))
    print(f"FINAL MANUSCRIPT: {doc} ({m:%H:%M})")
