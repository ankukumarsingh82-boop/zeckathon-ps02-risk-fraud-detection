# PS 02: Risk & Fraud Early-Warning System ("would have caught IL&FS?")

This is a local application plus a reproducible pipeline. Each quarter it scores a frozen book of real listed Indian non-financial companies (`data/reference/universe_frozen.csv`, frozen 2026-10-29: 893 universe names plus 44 sector peers; 930 eligible borrowers at 2026-09-30, a stand-in for the brief's 800-borrower book). It flags a hard-capped top-k = min(40, floor(5% x N)) review list, explains each flag with its top 3 drivers, and backtests the system against 15 documented Indian corporate defaults.

## Run the app (one command)
```bash
make setup      # once: creates .venv and installs requirements (Python 3.11+; WeasyPrint needs system pango for the PDF only)
make app        # = .venv/bin/streamlit run app.py   -> open http://localhost:8501
```
The app works fully offline from the committed dataset (`outputs/`, `data/`). It does not need an API key.
Tabs:
- **Watch-list**: any quarter, top 40, flagged names within the cap, with drivers.
- **Company drill-down**: score history vs the alert cutoff (the score of the k-th name), SHAP and plain-language drivers, Z''/F/M scores, soft signals, data provenance.
- **Metrics**: Polish PR-AUC, Indian recall at 4Q and 8Q, leakage checks.
- **Indian default cases**: event dates and sources.
- **Policy & costs**.
- **Provenance & data gaps**.

Optional: run `export FIRECRAWL_API_KEY=...` before `make app` to turn on the read-only "check for newer filings" button. It never changes the scores.

## Reproduce everything
```bash
make all        # = python run_all.py : Polish model -> Indian scoring -> 60-day-lag and large/mid-cap sensitivities -> report + dashboard (about 8 min)
make test       # pytest: alert cap, leakage, soft-signal timing, cost arithmetic, report/json consistency
make scrape     # optional: re-fetch public screener.in pages (polite, cached), then rerun
```
Seeds are fixed (42). Outputs:
- `outputs/`: metrics JSON, scores, watch-list, provenance.
- `reports/report.md` and `reports/report.pdf`.
- `dashboard/index.html`: static dashboard.

## Layout
- `src/polish_model.py`: UCI Polish distress model. Horizon split: train 1-3year, validation 4year, untouched holdout 5year. Monotone LightGBM vs logistic, isotonic calibration, cost threshold.
- `src/scrape_screener.py`: cached scraper for public annual statements, schedules and prices. Logs URL and access time.
- `src/india_features.py`: rebuilds the Polish ratios from Indian statements, plus Altman Z'', Piotroski F, Beneish M, as-of dates and provenance.
- `src/score_india.py`: quarterly scoring, SHAP drivers, recall at 4Q/8Q at a 5% alert rate, expected cost, watch-list, leakage checks, data gaps.
- `src/make_report.py`: figures, policy, report (MD/PDF), static dashboard.
- `data/reference/cases.csv`, `peers.csv`, `soft_signals.csv`: hand-researched event dates and signals with source URLs.

- `tests/test_pipeline.py`: pytest suite. Run it after `make all`.

## Headline results (from `outputs/*.json`; see `reports/report.pdf`)
- Polish untouched holdout: PR-AUC 0.369 (no-skill 0.069, 6.94% positives), ROC-AUC 0.823. With native-NaN handling the PR-AUC is 0.561, but that benefit does not transfer to Indian data.
- Indian backtest (15 non-financial defaulters): composite recall is 1/15 at 4Q and 0/15 at 8Q, with alerts capped at min(40, 5%). IL&FS group companies are not caught. Every case's rank is in report section 5.2.
- Survivorship bias: the universe comes from current listings. See report section 7.

Nothing is deployed or pushed anywhere. The repo is a local git repository with no remote.
