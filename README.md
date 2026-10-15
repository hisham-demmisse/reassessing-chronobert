# Reassessing ChronoBERT on Public Financial News

UCL Statistical Natural Language Processing coursework, 2025/26, group project. Code only; the written report is not included.

## What we did

We replicated the finance pipeline of ChronoBERT (He et al., 2025) on FNSPID news headlines and tested whether its look-ahead-bias conclusions survive a change of dataset and of regression framework (expanding window, rolling window, Fama-MacBeth). We also tested whether ChronoBERT checkpoints show the knowledge cutoffs they claim.

## Contents

- `Code/paper_replication.ipynb`: downloads FNSPID, builds stock-day headline embeddings, fits monthly ridge regressions and forms long-short portfolios
- `Code/make_datasets.py`: builds the `dataset/` CSVs the three notebooks below read
- `Code/ChronoBERT_experiments.ipynb`: bootstrap intervals, sub-periods, parameter robustness, embedding geometry, sector and permutation tests (2019 to 2022)
- `Code/ChronoBERT_fama_macbeth_sp500_10-22.ipynb`: Fama-MacBeth ridge regression, decile sorts and Ledoit-Wolf Sharpe tests on S&P 100 stocks
- `Code/ChronoBERT_rolling_covid.ipynb`: rolling regression over the COVID period across ChronoBERT checkpoints
- `Code/knowledge_cutoff.ipynb`: paired temporal discrimination test on the checkpoints
- `Code/chronological_tests.ipynb`: single-mask and pairwise A/B question-bank tests of chronological consistency

## Running it

Python 3.9 with `torch transformers datasets huggingface_hub scikit-learn scipy pandas numpy yfinance matplotlib seaborn tqdm`. Run the notebooks from inside `Code/`.

- The finance notebooks need a GPU and several GB of FNSPID from Hugging Face (`Zihan1004/FNSPID`). `paper_replication.ipynb` downloads what it needs.
- `ChronoBERT_experiments`, `ChronoBERT_fama_macbeth_sp500_10-22` and `ChronoBERT_rolling_covid` read `Code/dataset/fnspid_2019_2022.csv` or `Code/dataset/fnspid_2010_2023.csv`. Build them with `python Code/make_datasets.py`, or supply your own CSVs with the columns `Date` (UTC timestamp string), `Stock_symbol` and `Article_title`. The original CSVs were not kept, so `make_datasets.py` is a reconstruction from the cleaning in `paper_replication.ipynb`.
- `chronological_tests.ipynb` needs two question-bank CSVs that are not in the repository; its first cell lists the columns.
- `knowledge_cutoff.ipynb` needs only the model downloads.
- The finance notebooks fetch prices from yfinance at run time.

## Authors

Group project; collaborators not named here.

Scope: one-term coursework with small samples and no tuning budget; results are indicative.
