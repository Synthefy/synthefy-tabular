# NFL blog reproduction checks

The notebook now ports the original feature builders, weekly inference, and
Kalshi selection rule. It downloads public inputs and computes predictions;
it does not download precomputed research forecasts. Verified local results can
be resumed only when input, model, runtime, device, and file hashes match.

## Original-input port verification

These checks used original cached research inputs as reference fixtures outside
the public repository. They verify implementation equivalence, not the return
from a new download.

- Both checkpoint tables contain 4,254 rows. Keys, labels, eligibility, player
  identities and decision timestamps match the original.
- Of 140 numeric candidate columns, 137 match bitwise. Three defense EPA
  aggregations differ by at most 2.09e-16 from summation order; their actual
  Float32 model inputs match.
- All 36 weekly original-input context counts and selected-column lists match.
- Given original predictions and quotes, the strategy reproduces all 544
  decisions: 42 bets in 39 games, with 24 wins.
- At one contract per bet and an extra 5-cent execution allowance, those
  reference selections cost $19.34 and produce $4.66 quote-based profit,
  or 24.0951% return on deployed contract cost.

## Public source revisions

A fresh empty-cache build succeeds from nflverse through nflreadpy. Its keys,
labels, eligibility and decision timestamps match the original. Some **2020**
source values have changed: 3,018 play EPA values, 21 play classifications,
eight dropback indicators, and 427 player-game passing-EPA values, plus smaller
CPOE/success corrections. Attempts, passing yards, air yards and sacks in the
player-stat table remain unchanged. These corrections propagate into historical
rolling context; they must not be patched to force an old result.

The original raw snapshots still exist in private research storage, but are not
public notebook dependencies. The notebook records hashes of the sources it
actually downloads. Historical data can be corrected after publication.

Repeated builds from the same downloaded sources select the same columns and
produce identical Float32 model matrices across all 36 origins, despite tiny
Float64 aggregation differences.

## Model and inference

The original public weights are pinned to Synthefy/Nori revision
157d6be39b5ba8809e4296d50abf3f41f3b72947, file nori.pt, SHA-256:

    a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323

[Pinned weights](https://huggingface.co/Synthefy/Nori/resolve/157d6be39b5ba8809e4296d50abf3f41f3b72947/nori.pt)

Inference uses the original numeric candidates, past-context-only Pearson
pruning at 0.75, expanding 2018+ context with no row cap, and exact memory
policy. QB identities group their historical form but are not categorical
model inputs in this selected arm. Weather is excluded. The target is remaining
passing yards; observed yards are added back to the quantile bank.

A one-query CPU check against the original GPU forecast differed by 0.0489 yards
in the mean, 0.0395 in the median, and at most 0.1463 across the quantile bank.
CPU disables mixed precision. This is close numerical agreement, not a claim
of bit-identical predictions across hardware.

## Fresh-run status

The complete public-data 2025 run is in progress. Its measured forecast metrics,
trade selections and return will be saved by executing the notebook cells.
Do not treat the original-input reference return above as this run's output.

## Market handling

The Q1 selection uses the market-balanced line; halftime is a maximum-edge
fallback with same-line/side probability confirmation. Only one position can
be opened per QB-game. Quotes must lie between the checkpoint anchor and the
decision, meet age/spread filters, and are priced at the buying side of the
quote. No future transaction price determines entry.

Kalshi is enabled by default. A failed market download reports unavailable and
null ROI while retaining forecasts. It does not fabricate quotes or report
profit on a failed partial download. Minute-close quotes do not establish
available size or guaranteed fills. The strategy was refined on 2025 data;
its results are exploratory, not independent validation.
