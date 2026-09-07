"""Generate readable notebook cells, with immutable public companion downloads.

Run after committing/pushing helpers: python build_nfl_blog_notebook.py --source-ref SHA.
Execute the resulting notebook separately to populate outputs; this generator never
embeds saved predictions or claims execution happened.
"""

import argparse
import hashlib
from pathlib import Path
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parent


def build(source_ref):
    if len(source_ref) != 40 or any(c not in "0123456789abcdef" for c in source_ref):
        raise ValueError("--source-ref must be the public 40-character helper commit SHA")
    sources = [
        ROOT / name for name in ("nfl_blog_inference.py", "nfl_blog_strategy.py", "nfl_passing_yards_markets.py")
    ]
    sources += sorted((ROOT / "nfl_blog_features").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cells = []

    def md(s):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))

    md("""
    # NFL passing-yard predictions and Kalshi backtest with Nori

    Run every cell from a fresh Python 3.11 Jupyter or Colab environment. This
    notebook downloads public nflverse football data, reconstructs the blog's
    feature transformations, runs fresh Nori predictions for all 18 regular-season
    weeks of 2025, and optionally downloads Kalshi quotes to replay its strategy.
    **It never downloads precomputed research predictions or requires an internal repository.**
    The first run computes predictions; subsequent runs can reuse verified outputs
    generated locally from matching inputs, model, configuration, and runtime.

    The original feature rules, pruning, model weights, and betting decisions are
    ported here—not the earlier simplified baseline. Results below are measured
    from the downloaded inputs. Public nflverse files can be revised: a fresh
    2020 source differs from the original cached snapshot, so matching the original
    24.1% number is a result to verify, not a value to hard-code.

    CPU works but the full run can take hours. A GPU is optional. To smoke-test,
    change WEEKS to (1,); that is a partial run, not the full-season reproduction.
    Kalshi is enabled by default; an API outage does not discard your forecasts.
    """)
    md("""## 1. Install the measured package versions
    Run this before importing the packages. If Jupyter already imported different
    versions, restart its kernel after installation and run again.
    """)
    code("""
    import sys
    import subprocess
    import importlib.metadata as metadata

    required = {
        "synthefy-nori": "0.19.0", "nflreadpy": "0.1.5", "numpy": "2.4.6",
        "pandas": "3.0.5", "polars": "1.44.0", "torch": "2.13.0",
        "matplotlib": "3.11.1", "requests": "2.34.2", "huggingface-hub": "1.28.0",
        "scipy": "1.17.1", "scikit-learn": "1.9.0", "pyarrow": "25.0.1",
    }
    missing = []
    for package, wanted in required.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != wanted:
            missing.append(f"{package}=={wanted}")
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    print({name: metadata.version(name) for name in required})
    """)
    md("""## 2. Download the public implementation
    The feature transformations are too long for one readable notebook cell.
    Their adjacent source files are fetched from a fixed public Git commit and
    checked against SHA-256 hashes. Downloading this notebook alone is sufficient.
    The downloads are ordinary source code, not data or precomputed forecasts.
    """)
    code(f"""
from pathlib import Path
import hashlib
import json
import os
import requests

SOURCE_REF = {source_ref!r}
SOURCE_HASHES = {hashes!r}
SOURCE_DIR = Path("nfl_blog_source") / SOURCE_REF
base_url = f"https://raw.githubusercontent.com/Synthefy/synthefy-nori/{{SOURCE_REF}}/examples/notebooks"
for filename, expected_hash in SOURCE_HASHES.items():
    destination = SOURCE_DIR / filename
    if not destination.exists():
        response = requests.get(f"{{base_url}}/{{filename}}", timeout=60)
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Helper hash mismatch: {{destination}}; remove this file and retry")
sys.path.insert(0, str(SOURCE_DIR.resolve()))
print(f"Verified {{len(SOURCE_HASHES)}} public source files at {{SOURCE_REF}}")
""")
    code("""
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from IPython.display import display
    from nfl_blog_features import build_blog_features, blog_feature_columns
    from nfl_blog_inference import run_blog_predictions, prediction_metrics, modeling_rows
    from nfl_blog_strategy import run_kalshi_backtest, probability_over_line, summarize

    CACHE_DIR = Path(os.environ.get("NFL_CACHE_DIR", "nfl_blog_cache"))
    OUTPUT_DIR = Path(os.environ.get("NFL_OUTPUT_DIR", "nfl_blog_outputs"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEEKS = tuple(range(1, 19))  # All regular-season weeks; use (1,) only for a smoke test.
    DEVICE = os.environ.get("NFL_DEVICE", "cpu")  # For example: "cuda:0".
    RUN_MARKET_BACKTEST = True
    EXECUTION_COST = 0.05  # Additional modeled cost per one-contract purchase.
    REFRESH_SOURCE_DATA = False
    RESUME_PREDICTIONS = True  # Reuse only matching, fingerprinted local computations.
    print({"weeks": WEEKS, "device": DEVICE, "kalshi": RUN_MARKET_BACKTEST})
    """)
    md("""## 3. Pull football data and build the two checkpoint tables
    nflreadpy downloads schedules, player statistics, depth charts, participation,
    play-by-play, and team metadata. Full-game rolling form uses 2016–2017 as
    warmup. Model context starts in 2018 and expands only with earlier weeks.

    The 140 numeric candidates include QB form, offense, defense, game/season
    context, game state at the checkpoint, and the QB's matching Q1/halftime
    history. The original selected model uses identity to compute individual
    history; it does **not** feed categorical QB IDs or weather to Nori.

    Q1 uses the earliest timestamped Q2 record as its boundary, excluding that
    record. Halftime uses the last Q2 record. The decision is two minutes later.
    Finalized play data may include corrections and do not prove feed arrival time.
    """)
    code("""
    features = build_blog_features(CACHE_DIR, OUTPUT_DIR, refresh=REFRESH_SOURCE_DATA)
    overview = pd.DataFrame([
        {"checkpoint": h, "rows": table.height,
         "eligible_2025_rows": modeling_rows(table).filter(pl.col("season") == 2025).height,
         "candidate_features": len(blog_feature_columns())}
        for h, table in features.items()
    ])
    display(overview)
    """)
    code("""
    example_columns = ["game_id", "week", "actual_qb_name", "live_decision_utc",
                       "live_qb_passing_yards", "live_qb_attempts", "official_passing_yards"]
    history_columns = [c for c in blog_feature_columns() if c.startswith("checkpoint_history_")][:3]
    display(features["q1"].filter(pl.col("season") == 2025)
            .select(example_columns + history_columns).head(8).to_pandas())
    print("All candidate feature names:")
    print("\\n".join(blog_feature_columns()))
    """)
    md("""## 4. Run Nori—fresh predictions, week by week
    For each week and checkpoint, rank candidate features using correlations
    computed on past context only. Drop a feature when its absolute correlation
    with an already-kept feature exceeds 0.75. Use the entire retained context:
    no subsampling or different lightweight model is substituted.

    The pinned public Nori checkpoint predicts **remaining** yards. Adding the
    yards already thrown converts every quantile into a final-yard prediction.
    fit() stores the context; it does not train new model weights. Each week's
    predictions and chosen columns are saved as they finish.
    On a rerun, RESUME_PREDICTIONS checks local provenance fingerprints before
    reusing a completed week. Changed inputs, selected features, model, runtime,
    or device trigger fresh inference. Set it to False to recompute every week.
    """)
    code("""
    predictions = run_blog_predictions(
        features["q1"], features["halftime"], OUTPUT_DIR,
        weeks=WEEKS, device=DEVICE, season=2025, resume=RESUME_PREDICTIONS,
    )
    """)
    md("""## 5. Measure forecast error and coverage
    MAE and RMSE below use the predictive median. Pinball loss measures quantile
    accuracy; P10–P90 coverage is the fraction of final totals inside the predicted
    middle 80%. These are forecast metrics, not betting returns.
    """)
    code("""
    metrics = {h: prediction_metrics(frame) for h, frame in predictions.items()}
    display(pd.DataFrame(metrics).T.drop(columns=["pinball_loss"]))
    display(pd.DataFrame({h: values["pinball_loss"] for h, values in metrics.items()}).rename_axis("quantile"))
    (OUTPUT_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    """)
    code("""
    display(predictions["q1"].select(
        "week", "actual_qb_name", "live_qb_passing_yards", "nori_p10",
        "nori_median", "nori_p90", "official_passing_yards", "context_rows", "selected_feature_count"
    ).head(12).to_pandas())
    """)
    md("""## 6. See the distribution in 5-yard bins
    Prefer the blog's Burrow example when that week is included; otherwise use the
    first predicted row. This plot is computed from this run's quantiles. Values
    are not copied from the blog. Bar heights approximate probability mass by
    interpolating the predictive CDF at half-yard bin boundaries.
    """)
    code("""
    example = predictions["q1"].filter(
        (pl.col("game_id") == "2025_13_CIN_BAL") & (pl.col("actual_qb_name") == "Joe Burrow")
    )
    row = (example if example.height else predictions["q1"].head(1)).to_dicts()[0]
    line = 200
    taus = row["nori_quantile_taus"]
    values = row["nori_quantile_values"]
    max_yards = max(500, int(np.ceil(max(values) / 5) * 5))
    boundaries = np.arange(-0.5, max_yards + 5, 5)
    cdf = 1 - np.array([probability_over_line(taus, values, x) for x in boundaries])
    mass = np.maximum(0, np.diff(cdf))
    centers = boundaries[:-1] + 2.5
    p_over = probability_over_line(taus, values, line - 0.5)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(centers, 100 * mass, width=4.5,
           color=np.where(centers >= line, "#33856b", "#c98a66"))
    ax.axvline(line - 0.5, color="#b42318", label=f"{line}+ yards; Nori probability {p_over:.1%}")
    ax.set(xlabel="Final passing yards", ylabel="Probability per 5-yard bin (%)",
           title=f"{row['actual_qb_name']} — Week {row['week']} — after Q1")
    ax.legend()
    plt.show()
    fig.savefig(OUTPUT_DIR / "passing_yards_distribution.png", bbox_inches="tight")
    """)
    md("""## 7. Replay the original Kalshi rule
    After Q1, select the quoted line whose midpoint is closest to 50¢, then its
    higher-edge side. Buy only if probability × $1 − price − fee is at least 10¢.
    If Q1 does not buy, select the maximum-edge halftime candidate. Its edge must
    reach 10¢ and its probability for the **same line and side** must be at least
    the Q1 probability. At most one contract is bought per quarterback-game.

    Use the latest minute-close bid/ask between the anchor and decision; exclude
    quotes over five minutes old or with spreads over 10¢. YES buys at the YES ask;
    NO buys at one minus the YES bid. Modeled taker fees and the extra execution
    allowance reduce reported profit. Settlement comes from the public market.
    These are **quote-based simulations, not verified fills**.

    Missing API responses stop only the market stage. They never become invented
    quotes, inferred fills, or a profitable result on an incomplete download.
    """)
    code("""
    if RUN_MARKET_BACKTEST:
        decisions, market_report = run_kalshi_backtest(
            predictions["q1"], predictions["halftime"], CACHE_DIR,
            execution_cost=EXECUTION_COST,
        )
    else:
        decisions = pl.DataFrame()
        market_report = {"status": "disabled", "roi": None}
    print(json.dumps(market_report, indent=2))
    (OUTPUT_DIR / "kalshi_status.json").write_text(json.dumps(market_report, indent=2))
    """)
    code("""
    if market_report["status"] == "quote_based_simulation":
        decisions.write_parquet(OUTPUT_DIR / "kalshi_decisions.parquet")
        trades = decisions.filter(pl.col("bet_taken"))
        display(trades.select(
            "week", "actual_qb_name", "selected_horizon", "line", "side",
            "yes_bid", "yes_ask", "model_probability", "entry_price", "fee",
            "expected_net_edge", "stressed_capital", "stressed_profit"
        ).to_pandas())
        trades.write_csv(OUTPUT_DIR / "kalshi_selected_trades.csv")
        sensitivities = pd.DataFrame([summarize(decisions, cost) for cost in (0.0, 0.05, 0.10)])
        display(sensitivities[["execution_cost", "bets", "capital", "pnl", "roi"]])
    else:
        print("Forecasts are saved. No betting return is reported without available market data.")
    """)
    md("""## 8. Keep the provenance with the results
    Feature input hashes, pinned model identity, per-week feature selections,
    forecast metrics, and the market status are saved beside predictions. The
    original strategy was refined on 2025: this is an exploratory reproduction,
    not an untouched test or a guarantee of future returns. Public source revisions
    and runtime differences can change numbers even with the same algorithm.
    """)
    code("""
    run_manifest = {
        "source_ref": SOURCE_REF, "source_hashes": SOURCE_HASHES,
        "package_versions": {p: metadata.version(p) for p in required},
        "weeks": WEEKS, "device": DEVICE, "execution_cost": EXECUTION_COST,
        "market_status": market_report["status"],
        "resume_verified_local_predictions": RESUME_PREDICTIONS,
        "precomputed_research_predictions": False,
        "input_hash_file": "feature_input_hashes.json",
    }
    (OUTPUT_DIR / "notebook_run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    display(pd.DataFrame([{"file": p.name, "bytes": p.stat().st_size}
                          for p in sorted(OUTPUT_DIR.iterdir()) if p.is_file()]))
    print(f"Outputs: {OUTPUT_DIR.resolve()}")
    """)
    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
    )
    for i, cell in enumerate(nb.cells):
        cell.id = f"nfl-blog-{i:02d}"
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "nori-nfl-passing-yards.ipynb")
    print(
        f"Generated {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells; no outputs fabricated"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", required=True)
    build(parser.parse_args().source_ref)
