# NFL passing-yard notebook

Open nori-nfl-passing-yards.ipynb in Jupyter or upload it to Colab. The notebook includes the helper implementations: downloading the notebook alone is sufficient. Python 3.11 is recommended.

The default path installs the public packages, downloads 2024/2025 nflverse play-by-play and official player statistics, creates Q1/halftime rows, downloads public Nori-6M weights, and generates fresh 2025 Week 1 predictions. No saved predictions, embedded dataset, website data asset, private checkpoint, or research checkout is required. Set MAX_WEEK=18 for a full-season run. CPU is supported; GPU is optional. The first run downloads data and model weights; later runs reuse the cache.

Forecast outputs include MAE, RMSE, pinball loss, central 80% coverage, quantiles, a plot and a run manifest. The public baseline is intentionally distinct from the blog's selected configuration and does not claim its 24.1% return. It selects the first observed QB passer, not an independently verified anticipated starter; current-week outcomes are excluded from context. Historical corrected play data do not establish real-time publication availability.

RUN_MARKET_BACKTEST defaults to True. It downloads historical Kalshi markets and quotes and exports candidate/rejection records, one-contract selections and quote-based cost sensitivities. It never places orders or establishes fills/depth. API/network failures or unavailable matching market data preserve the forecast outputs, write kalshi_status.json with available=false and null ROI, and allow the notebook to finish. Successful requests remain cached. Programming errors and corrupt caches still raise.

The adjacent Python modules are reviewable mirrors of the embedded implementations. Keep the notebook copies synchronized when changing them. The prompt file describes how to adapt this baseline.

Offline tests:

    python -m pytest examples/notebooks/test_nfl_passing_yards.py -q

Full-season forecast performance and betting profitability must be measured after running the chosen configuration; neither is hard-coded or promised.

Exact blog parity is tracked in NFL_REPRODUCTION_PARITY.md. Historical-only inputs permit reconstruction in principle, but the current feature, context, and timing settings differ and exact parity has not been demonstrated.
