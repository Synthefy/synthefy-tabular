# Original NFL blog feature pipeline

This package ports the original feature transformations. It does not consume
saved predictions, feature tables, or private research modules.

build_blog_features(cache_dir, output_dir, refresh=False) returns a dictionary
with q1 and halftime Polars frames and saves both rebuilt frames plus SHA-256
hashes of cached raw Parquet inputs. blog_feature_columns() returns the ordered
140 numeric candidate columns before training-only correlation pruning.

Public sources are loaded with nflreadpy: schedules and player statistics
(2016–2025), depth charts and participation (2018–2025), play-by-play
(2016–2025 for full-game rolling features; 2018–2025 for checkpoints), and teams.
A missing required input fails visibly rather than substituting another model.
Weather, injuries, betting lines, and categorical player ID are not part of
the selected numeric model. QB identity is used for individual rolling history.

The checkpoint rules preserve the experiment: Q1 uses the earliest timestamped
Q2 record as its boundary (excluding that record); halftime uses the final Q2
record. Decisions occur two minutes later. Historical checkpoint means use
eligible earlier-week games only. Finalized nflverse data may contain later
corrections and does not establish original feed arrival latency.

Pregame form is recomputed for the actual observed starter. Full-game form
uses 2016–2017 as warmup; checkpoint history starts in 2018. Unused columns
from other research arms are intentionally omitted, but all 140 selected
candidate columns and all decision timestamps/labels are preserved.

## Port verification

Rebuilding from original cached public raw inputs yielded 4,254 rows for
each checkpoint. QB/game keys, targets, identities, eligibility and decision
timestamps matched the original feature artifacts exactly. 137 of 140
candidate columns matched bitwise; the other three defense EPA averages had
maximum absolute difference 2.09e-16 from floating-point aggregation order.
This is transformation parity, not a promise that revised upstream downloads
will be byte-identical to the original cached source snapshot.

The package includes 29 offline tests copied from the original invariant suite
and adapted only for imports. Run from the repository root:

    PYTHONPATH=examples/notebooks pytest examples/notebooks/nfl_blog_features/tests
