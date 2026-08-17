# Cached derived data (all from real public sources; nothing synthetic)

These files let the analysis (steps 04-07 and the manuscript build) be
reproduced exactly without re-downloading FNSPID or re-scoring headlines.

- `corpus.parquet` — FNSPID headlines for the 300-symbol study universe
  (source: https://huggingface.co/datasets/Zihan1004/FNSPID, Stock_news/All_external.csv)
- `headline_scores.parquet` — all four sentiment scores per unique headline;
  m1 is the homogeneous fp16 GPU FinBERT score (m2 LM, m3 VADER, m4 HIV4)
- `prices.parquet` — yfinance daily Close/High/Low, adjusted, as fetched on
  2026-07-26 (kept because Yahoo's history is not perfectly stable over time)
- `panel.parquet` — the aligned symbol-session panel used by steps 04-06
- `dataset.json`, `universe.csv` — corpus metadata and symbol list

To reproduce from here, point `--indir` of analysis steps 04-06 at this
folder (or copy these files into a work dir and run `run_final.sh` stages).
