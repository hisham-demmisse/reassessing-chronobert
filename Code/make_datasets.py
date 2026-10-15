"""Build the FNSPID news CSVs that the finance notebooks read.

Writes, next to this script:
    dataset/fnspid_2019_2022.csv  (ChronoBERT_experiments, ChronoBERT_rolling_covid)
    dataset/fnspid_2010_2023.csv  (ChronoBERT_fama_macbeth_sp500_10-22)

The cleaning follows the news-loading cell of paper_replication.ipynb: keep
Date, Stock_symbol and Article_title, and drop rows with an unparseable date,
an empty headline or an empty ticker. Date is kept as the original UTC string
because the notebooks parse it themselves with pd.to_datetime(...).dt.date.
The CSVs originally used were not kept, so these are a reconstruction.

Usage:
    python make_datasets.py                  # downloads All_external.csv from Hugging Face
    python make_datasets.py path/to/All_external.csv
"""

import os
import sys

import pandas as pd

WINDOWS = {"fnspid_2019_2022.csv": (2019, 2022), "fnspid_2010_2023.csv": (2010, 2023)}
COLUMNS = ["Date", "Stock_symbol", "Article_title"]


def clean(chunk):
    chunk = chunk.copy()
    chunk["Stock_symbol"] = chunk["Stock_symbol"].fillna("").astype(str).str.strip()
    chunk["Article_title"] = chunk["Article_title"].fillna("").astype(str)
    parsed = pd.to_datetime(chunk["Date"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce", utc=True)
    chunk["year"] = parsed.dt.year
    keep = (
        parsed.notna()
        & (chunk["Article_title"].str.strip() != "")
        & (chunk["Stock_symbol"] != "")
        & (chunk["Stock_symbol"].str.lower() != "nan")
    )
    return chunk[keep]


def build(news_csv, out_dir, chunksize=200_000):
    os.makedirs(out_dir, exist_ok=True)
    parts = {name: [] for name in WINDOWS}
    for chunk in pd.read_csv(news_csv, usecols=COLUMNS, chunksize=chunksize, dtype={"Stock_symbol": str}):
        chunk = clean(chunk)
        for name, (first, last) in WINDOWS.items():
            parts[name].append(chunk[chunk["year"].between(first, last)][COLUMNS])
    for name, frames in parts.items():
        df = pd.concat(frames, ignore_index=True).sort_values(["Date", "Stock_symbol"])
        df.to_csv(os.path.join(out_dir, name), index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        news_csv = sys.argv[1]
    else:
        from huggingface_hub import hf_hub_download
        news_csv = hf_hub_download(repo_id="Zihan1004/FNSPID", repo_type="dataset", filename="Stock_news/All_external.csv")
    build(news_csv, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
