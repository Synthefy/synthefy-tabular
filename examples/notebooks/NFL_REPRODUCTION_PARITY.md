# NFL blog reproduction parity

The notebook currently runs a public-source baseline. **The exact blog feature
pipeline has not yet been ported.** A successful notebook run must not be
described as reproducing the blog's 24.1% return.

## Verified model availability

The original Nori-6M weights are downloadable without credentials from
`Synthefy/Nori`, revision `157d6be39b5ba8809e4296d50abf3f41f3b72947`,
filename `nori.pt`. The anonymous download endpoint returned HTTP 200 after a
redirect. Its linked SHA-256 matches the original experiment's recorded hash:

```text
a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323
```

[Pinned weights](https://huggingface.co/Synthefy/Nori/resolve/157d6be39b5ba8809e4296d50abf3f41f3b72947/nori.pt)

This verifies the checkpoint, not every historical data download. nflverse
provides the source families used by the experiment, but their currently
available contents have not all been compared with the original snapshots.
Historical feeds can receive corrections. Do not promise identical predictions
or returns until input hashes and derived rows have been compared.

## Required parity checks

| Component | Original blog experiment | Baseline difference / required check |
| --- | --- | --- |
| Features | QB, offense, defense, context, season context, live usage, and nine checkpoint-history features | Port the original feature builders and preserve feature order; the baseline's reduced aggregates are not equivalent. |
| Checkpoint history | Prior 3/8-game checkpoint yards, attempts, YPA and remaining yards; history count capped at eight | Match eligibility, missing values and weekly embargo exactly. |
| Categories | Winning arm recorded no categorical columns | The baseline adds ordinal QB/team/opponent columns. Keep categorical alternatives separate from exact reproduction. |
| Context | 2016 warmup; eligible context starts 2018; weekly expanding history with no row cap | Baseline defaults use shorter history and capped context. Original Q1 Week 1 2025 had 3,710 context rows. |
| Pruning | Weekly training-only Pearson pruning at 0.75 against remaining yards | Restore the original selection algorithm and selected-column manifests. |
| Q1 timing | Earliest timestamped Q2 record anchors the decision; include Q1 records strictly before it; decide 120 seconds later | Last-Q1-play timing is not equivalent. |
| Identity and eligibility | Original starter selection, team aliases and evaluation eligibility | Match rows and identities before comparing forecasts. |
| Inference | Pinned checkpoint, float32 numeric matrix, exact memory policy, no dropped context | Pin package/runtime behavior as well as weights; compare native quantile banks. |
| Metrics | Median-based MAE/RMSE and reported P10/P50/P90 pinball summaries | Mean-based errors or all-quantile pinball are different metrics. |
| Selection | Q1 market-balanced selection; halftime max-edge fallback with same-side/same-line probability confirmation | Reproduce quote timestamps, filters, fees and selection order, not just the headline edge threshold. |

## Original source provenance

The original research sources live in the monorepo under
`experiments/2026-08-25-nfl-qb-passing-yards/`; these references document the
implementation to port and are **not public notebook runtime dependencies**:

- `prepare.py`, `data.py`: source downloads and pregame dataset construction.
- `live_features.py`, `live_flow_features.py`, `live_q1_features.py`: checkpoint
  boundaries, identity correction and live features.
- `season_context_features.py`: prior-week season context.
- `checkpoint_history_ablation.py`, `live_evaluate.py`: feature history,
  pruning, context and inference settings.
- `checkpoint_history_report.py`: fixed-rule strategy replay.
- `GAME_FLOW_PROTOCOL.md`, `Q1_TIMING_PROTOCOL_V2.md`,
  `CHECKPOINT_HISTORY_PROTOCOL.md`: original protocol details.

The recorded Q1 feature-source SHA-256 before adding checkpoint history is:

```text
a78310f1e30f6793533e02aadaf65188ce33880b8a19e38995a70d1fc55352c8
```

Package versions, source snapshots and ordered feature manifests must accompany
the eventual exact port. A matching weight file alone is insufficient.

## Acceptance conditions for an exact reproduction

These are reference assertions, **not results obtained by this baseline**:

1. Match source snapshots, derived rows, eligibility, labels and anchor times;
   where serialization changes, compare logical values in addition to hashes.
2. Cover 544 QB-game rows across 272 games per checkpoint in 2025.
3. Match weekly context rows, selected columns and original prediction banks
   within an explicitly reported numerical tolerance.
4. Match the original 42 selections in 39 games, including 24 winning contracts,
   their sides, thresholds, quoted prices and timestamps.
5. At one contract per selection, reproduce $17.24 deployed including modeled
   fees and $6.76 profit before the additional execution allowance.
6. With a further $0.05 per contract, reproduce $19.34 deployed and $4.66 profit:
   `4.66 / 19.34 = 24.0951%`, rounded to 24.1%.

Unavailable Kalshi history must produce an explicit skipped-market status, not
fabricated prices or a claimed reproduction. These targets describe a
quote-based simulation, not verified fills or realized earnings.
