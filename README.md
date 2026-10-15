# Reassessing ChronoBERT on Public Financial News

UCL Statistical Natural Language Processing coursework, 2025/26, group project. Code only; the written report is not included.

## What we did

We replicated the ChronoBERT finance pipeline (He et al., 2025) on FNSPID news headlines, then tested whether its look-ahead-bias conclusions hold on a different dataset and under different regression frameworks (expanding window, rolling window, Fama-MacBeth). We also tested whether the ChronoBERT checkpoints show the knowledge cutoffs they claim. The notebooks replicate the pipeline and run the robustness and knowledge-cutoff tests described below; see Known limitations before reading any number as a result.

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

## Known limitations

- News timing differs between notebooks. `paper_replication` moves headlines published after 16:00 ET to the next day. `ChronoBERT_experiments`, `ChronoBERT_fama_macbeth_sp500_10-22` and `ChronoBERT_rolling_covid` use the UTC calendar date, so some news that arrives after the close is traded at that day's close, and the absolute Sharpe ratios in those three notebooks are too high.
- The notebooks also differ in pooling (CLS or mean), unit (headline or stock-day), stock universe, sample period and FinBERT checkpoint. A difference between two notebooks is not the effect of the regression framework alone.
- The COVID universe in `ChronoBERT_experiments` and `ChronoBERT_rolling_covid` was picked with hindsight. `ChronoBERT_fama_macbeth_sp500_10-22` applies the current S&P 100 list back to 2010.
- Portfolios are equal-weighted and rebalanced daily with no transaction costs, and turnover is not reported. On some days the long and short legs hold only a few stocks.
- The bootstrap intervals resample days independently for each model, and the permutation test assumes the two strategies' daily returns are exchangeable, so neither tests whether two Sharpe ratios are equal. The Ledoit-Wolf test does, but our implementation uses an approximate gradient. No test is corrected for multiple comparisons.
- The Fama-MacBeth sample starts in 2010, when none of the ChronoBERT checkpoints compared is free of look-ahead. The 2019 to 2022 sub-period is the cleanest comparison.
- `paper_replication` runs in debug mode by default (40,000 stock-days per year, quintile portfolios), and the subsampling result rests on five repeats.
- The knowledge-cutoff test uses one prompt template over 26 years, and in two of the pairs the "wrong" year is also correct. The question banks for `chronological_tests` are not included, and this repository does not say how they were built.

## Authors

Group project; collaborators not named here.

Scope: one-term coursework with small samples and no tuning budget; results are indicative.
