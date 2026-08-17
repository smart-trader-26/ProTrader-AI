"""Step 1: stream the FNSPID news corpus, pick the study universe, extract its rows.

Input : FNSPID Stock_news/All_external.csv (real Benzinga headlines, ~5.7 GB)
        https://huggingface.co/datasets/Zihan1004/FNSPID
Output: corpus.parquet  (Date, Article_title, Stock_symbol, Publisher)
        symbol_counts.csv

No synthetic rows are created anywhere in this pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

COLS = ["Date", "Article_title", "Stock_symbol", "Publisher"]


def pass1_counts(src: str) -> pd.DataFrame:
    """Count headlines per symbol and per year without loading the file."""
    sym = Counter()
    sym_year = Counter()
    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=["Date", "Stock_symbol"],
            column_types={"Date": pa.string(), "Stock_symbol": pa.string()},
        ),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
    )
    n = 0
    for batch in reader:
        s = batch.column("Stock_symbol").to_pylist()
        d = batch.column("Date").to_pylist()
        for a, b in zip(s, d):
            if not a or not b:
                continue
            sym[a] += 1
            sym_year[(a, b[:4])] += 1
        n += batch.num_rows
        if n % 2_000_000 < batch.num_rows:
            print(f"  scanned {n:,} rows", flush=True)
    print(f"  total rows: {n:,}")
    df = pd.DataFrame(sorted(sym.items(), key=lambda kv: -kv[1]),
                      columns=["symbol", "n_articles"])
    yr = pd.DataFrame([(k[0], k[1], v) for k, v in sym_year.items()],
                      columns=["symbol", "year", "n"])
    return df, yr


def pass2_extract(src: str, keep: set[str], out: str) -> int:
    """Second pass: write only the rows of the selected universe."""
    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=COLS,
            column_types={c: pa.string() for c in COLS},
        ),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
    )
    writer = None
    kept = 0
    try:
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            mask = pa.compute.is_in(tbl.column("Stock_symbol"),
                                    value_set=pa.array(sorted(keep)))
            tbl = tbl.filter(mask)
            if tbl.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            kept += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top", type=int, default=100,
                    help="number of most-covered symbols to keep")
    ap.add_argument("--min-year", type=int, default=2010)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    counts_path = os.path.join(args.outdir, "symbol_counts.csv")
    year_path = os.path.join(args.outdir, "symbol_year_counts.csv")

    if os.path.exists(counts_path):
        counts = pd.read_csv(counts_path)
        print("reusing symbol counts")
    else:
        print("pass 1: counting headlines per symbol ...")
        counts, yr = pass1_counts(args.src)
        counts.to_csv(counts_path, index=False)
        yr.to_csv(year_path, index=False)

    # Universe: the most densely covered symbols. Dense coverage is required
    # because the response kernel is estimated per symbol at daily resolution.
    uni = counts.head(args.top)["symbol"].tolist()
    uni = [s for s in uni if s.isalpha() and 1 <= len(s) <= 5]
    print(f"universe: {len(uni)} symbols, "
          f"{counts.head(args.top)['n_articles'].sum():,} headlines")
    print(", ".join(uni[:40]))

    out = os.path.join(args.outdir, "corpus.parquet")
    print("pass 2: extracting universe rows ...")
    kept = pass2_extract(args.src, set(uni), out)
    print(f"wrote {kept:,} rows -> {out}")
    pd.Series(uni, name="symbol").to_csv(
        os.path.join(args.outdir, "universe.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
