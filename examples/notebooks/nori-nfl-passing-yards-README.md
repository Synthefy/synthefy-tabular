# Reproduce the NFL passing-yard pipeline

Open nori-nfl-passing-yards.ipynb in Python 3.11 Jupyter or upload it to Colab.
Run cells in order. Downloading the notebook alone is sufficient: it retrieves
its companion Python sources from a pinned public Git commit and checks hashes.
No internal repository, downloaded research predictions, private model or website
data asset is required.

The notebook downloads public nflverse sources with nflreadpy, reconstructs the
original blog feature transformations, generates fresh predictions with the
pinned public Nori checkpoint, and downloads archived Kalshi markets and quotes.
All 18 regular-season weeks of 2025 are scored by default at Q1 and halftime.
Full CPU execution can take hours; choose DEVICE="cuda:0" for an available GPU,
or WEEKS=(1,) for a clearly labelled partial smoke test. Changing evaluation weeks
does not truncate earlier-week context. Data and model weights are cached.
RESUME_PREDICTIONS=True reuses completed local prediction files only after their
input/model/configuration/runtime/device fingerprints match. Set False to force
recomputation; an empty output directory always starts with fresh inference.

## Original model and strategy, not the earlier small baseline

The feature port uses 140 numeric candidates, past-context-only correlation
pruning at 0.75, expanding context starting in 2018, and matching checkpoint
history. QB identity determines individual rolling histories but is not supplied
as a categorical feature to the original selected model. Weather is excluded.
Nori predicts remaining yards; observed yards are added to each quantile.

The strategy picks a market-balanced line at Q1 and a maximum-edge line at
halftime only if Q1 did not enter. Halftime must meet the 10-cent edge threshold
and same-line/side probability confirmation against Q1. There is at most one
one-contract position per QB-game. It reports fees and configurable extra
execution costs, with no live orders or assumed guaranteed fills.

The original historical-input reference check reproduced all 544 decisions and
the 42 selected trades (24 wins, $19.34 deployed and $4.66 quote-based profit at
5 cents extra cost). That is a port verification using original cached inputs,
not a promised result of a new download. A fresh public 2020 input differs from
the old snapshot. Revised sources can change rolling context and predictions.
The notebook measures the current run and never hard-codes 24.1% as its output.

## Missing market data

RUN_MARKET_BACKTEST defaults to True. Network/API failure preserves predictions
and records status="unavailable", with null ROI. A partial failed download does
not produce a partial return. Successful raw requests remain cached for retry.
Programming errors and corrupt source caches still fail visibly.

## Outputs

The notebook saves rebuilt feature tables, raw-input SHA-256 hashes, per-week
predictions, feature selections, model/runtime manifests, MAE, RMSE, pinball loss,
P10–P90 coverage, a 5-yard distribution plot and—when available—Kalshi decisions,
selected trades and cost sensitivities. Outputs shown in the notebook must come
from executing those cells, never copying older predictions into them.

The 2025 strategy was selected through experimentation, not independent test
validation. Finalized play data do not prove original feed arrival times.
Minute-close quotes do not establish order-book quantity or fills.

## Maintaining the notebook

Companion files are ordinary adjacent Python sources. After changing and pushing
them, regenerate the notebook with their immutable public commit SHA, then execute
it to save rendered outputs:

    python examples/notebooks/build_nfl_blog_notebook.py --source-ref COMMIT_SHA

The generator hashes the current companion files; make sure they match that commit.
It does not run inference. Unit tests are offline:

    PYTHONPATH=examples/notebooks pytest examples/notebooks/nfl_blog_features/tests examples/notebooks/nfl_blog_strategy_test.py

See nori-nfl-passing-yards-prompt.txt for the matching coding-assistant prompt.
