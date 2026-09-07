# NFL passing-yard notebook

Open nori-nfl-passing-yards.ipynb in Jupyter or upload it to Colab. The notebook includes the helper implementations: downloading the notebook alone is sufficient. Python 3.11 is recommended.

The default path installs the public packages, downloads 2024/2025 nflverse play-by-play and official player statistics, creates Q1/halftime rows, downloads public Nori-6M weights, and generates fresh 2025 Week 1 predictions. No saved predictions, embedded dataset, website data asset, private checkpoint, or research checkout is required. Set MAX_WEEK=18 for a full-season run. CPU is supported; GPU is optional. The first run downloads data and model weights; later runs reuse the cache.

Forecast outputs include MAE, RMSE, pinball loss, central 80% coverage, quantiles, a plot and a run manifest. The public baseline is intentionally distinct from the blog's selected configuration and does not claim its 24.1% return. It selects the first observed QB passer, not an independently verified anticipated starter; current-week outcomes are excluded from context. Historical corrected play data do not establish real-time publication availability.

RUN_MARKET_BACKTEST defaults to False. Enabling it downloads historical Kalshi markets and quotes and exports candidate/rejection records, one-contract selections and quote-based cost sensitivities. It never places orders or establishes fills/depth. Network failures stop the run and retain successful cached requests for retry.

The adjacent Python modules are reviewable mirrors of the embedded implementations. Keep the notebook copies synchronized when changing them. The prompt file describes how to adapt this baseline.

Offline tests:

    python -m pytest examples/notebooks/test_nfl_passing_yards.py -q

Full-season forecast performance and betting profitability must be measured after running the chosen configuration; neither is hard-coded or promised.
