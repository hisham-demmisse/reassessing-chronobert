# Reassessing ChronoBERT on Public Financial News

UCL Statistical Natural Language Processing coursework, 2025/26, group project. Code only; the written report is not included.

## What we did

We replicated the finance pipeline of ChronoBERT (He et al., 2025) on FNSPID news headlines and tested whether its look-ahead-bias conclusions survive a change of dataset and of regression framework (expanding window, rolling window, Fama-MacBeth). We also tested whether ChronoBERT checkpoints show the knowledge cutoffs they claim. The notebooks replicate the pipeline and run the robustness and knowledge-cutoff tests described below; see Known limitations before reading any number as a result.

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

- News timing differs across notebooks. `paper_replication` assigns headlines after 16:00 ET to the next day; `ChronoBERT_experiments`, `ChronoBERT_fama_macbeth_sp500_10-22` and `ChronoBERT_rolling_covid` use the UTC calendar date, so some after-close news is traded at the same day's close. Absolute Sharpe ratios in those three notebooks are optimistic.
- The notebooks differ in more than the regression framework: pooling (CLS or mean), unit (headline or stock-day), stock universe, sample period and FinBERT checkpoint all change between them. Differences between notebooks should not be read as the effect of the framework alone.
- The COVID universe (`ChronoBERT_experiments`, `ChronoBERT_rolling_covid`) was chosen with hindsight, and the S&P 100 list in `ChronoBERT_fama_macbeth_sp500_10-22` is the current one applied back to 2010.
- Portfolios are equal-weighted and rebalanced daily with no transaction costs; turnover is not reported. Daily long and short legs can contain only a few stocks.
- Bootstrap intervals resample days independently and per model, and the permutation test assumes the two strategies' daily returns are exchangeable. Neither is a test of equal Sharpe ratios. The Ledoit-Wolf test is the appropriate one; its implementation uses an approximate gradient, and no test is corrected for multiple comparisons.
- The Fama-MacBeth sample starts in 2010, when none of the ChronoBERT checkpoints compared is free of look-ahead; the cleanest comparison is the 2019 to 2022 sub-period.
- `paper_replication` defaults to debug mode (40,000 stock-days per year, quintile portfolios); the subsampling result is based on five repeats.
- The knowledge-cutoff test uses one prompt template and 26 years, and in two pairs the "wrong" year is also a correct answer. The question banks for `chronological_tests` are not included, and how they were built is not documented here.
- Results are from a one-term coursework project and are indicative only.

## Authors

Group project; collaborators not named here.

Scope: one-term coursework with small samples and no tuning budget; results are indicative.
