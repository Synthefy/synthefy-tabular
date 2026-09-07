"""Independent pregame context and market feature groups.

The two builders are intentionally separate so an evaluation can add sports
context without adding market information, or add the market group as an
explicitly labeled ablation.

The nflverse schedule dictionary defines spread_line as the closing spread
from the home-team perspective: positive means the home team was favored and
negative means the away team was favored. nflreadr's official clean_homeaway
example negates spread_line for the away row. Accordingly,
market_team_spread is positive when the row's team was favored and negative
when it was the underdog. This is a favored-by convention, not the common
sportsbook display convention where a favorite is shown with a minus sign.

Sources:
https://github.com/nflverse/nflfastR/blob/master/NEWS.md
https://nflreadr.nflverse.com/reference/clean_homeaway.html

Important timestamp caveat: nflverse exposes a closing spread and closing
total, not a timestamped quote history. These fields are suitable for a
market-assisted prediction ablation, but not as proof that the same line was
tradable at the experiment's T-60 decision time.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "context_is_home",
    "context_is_neutral_site",
    "context_team_rest_days",
    "context_rest_advantage_days",
    "context_is_short_week",
    "context_is_post_bye",
    "context_roof_outdoors",
    "context_roof_open",
    "context_roof_closed",
    "context_roof_dome",
    "context_roof_other",
    "context_roof_missing",
    "context_temperature_f",
    "context_wind_mph",
)

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_team_spread",
    "market_game_total",
)

_BASE_KEYS = ("game_id", "team")
_ROOF_VALUES = ("outdoors", "open", "closed", "dome")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _validate_base_rows(base_rows: pl.DataFrame) -> None:
    _require_columns(base_rows, _BASE_KEYS, frame_name="base_rows")
    if base_rows.select(pl.struct(_BASE_KEYS).is_duplicated().any()).item():
        raise ValueError("base_rows must have exactly one row per game_id/team")


def _schedule_rows(
    base_rows: pl.DataFrame,
    schedules: pl.DataFrame,
    schedule_columns: Sequence[str],
) -> pl.DataFrame:
    _validate_base_rows(base_rows)
    required_schedule_columns = ("game_id", "home_team", "away_team", *schedule_columns)
    _require_columns(schedules, required_schedule_columns, frame_name="schedules")

    schedule_lookup = schedules.select(
        pl.col("game_id"),
        pl.col("home_team").alias("_schedule_home_team"),
        pl.col("away_team").alias("_schedule_away_team"),
        *[pl.col(column).alias(f"_schedule_{column}") for column in schedule_columns],
    )
    if schedule_lookup.select(pl.col("game_id").is_duplicated().any()).item():
        raise ValueError("schedules must have exactly one row per game_id")

    rows = (
        base_rows.with_row_index("_feature_row_order")
        .join(
            schedule_lookup,
            on="game_id",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .with_columns(
            (pl.col("team") == pl.col("_schedule_home_team")).alias("_feature_is_home"),
            (pl.col("team") == pl.col("_schedule_away_team")).alias("_feature_is_away"),
        )
    )
    missing_games = rows.filter(pl.col("_schedule_home_team").is_null()).get_column("game_id").unique().to_list()
    if missing_games:
        raise ValueError(f"base_rows contain game_ids absent from schedules: {missing_games[:5]}")

    invalid_teams = rows.filter(
        ~(pl.col("_feature_is_home").fill_null(False) ^ pl.col("_feature_is_away").fill_null(False))
    ).select("game_id", "team")
    if invalid_teams.height:
        examples = invalid_teams.head(5).rows()
        raise ValueError(f"base_rows teams must match exactly one schedule side: {examples}")
    return rows


def _nullable_binary(condition: pl.Expr, *, present: pl.Expr, alias: str) -> pl.Expr:
    return pl.when(present).then(condition.cast(pl.Float64)).otherwise(None).alias(alias)


def _finish(rows: pl.DataFrame, base_rows: pl.DataFrame, feature_columns: Sequence[str]) -> pl.DataFrame:
    output = rows.sort("_feature_row_order").select(*base_rows.columns, *feature_columns)
    if output.height != base_rows.height:
        raise RuntimeError("feature builder changed the number of input rows")
    if output.select(pl.struct(_BASE_KEYS).is_duplicated().any()).item():
        raise RuntimeError("feature builder produced duplicate game_id/team rows")
    return output


def build_context_features(base_rows: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Add numeric sports-context features without market or result fields.

    context_is_post_bye uses at least 11 days between games. That threshold
    separates a normal/mini-bye Thursday-to-Sunday gap (10 days) from the
    shortest ordinary NFL bye transition, Sunday-to-Thursday (11 days).
    context_is_home follows nflverse's designated home team even at a neutral
    site; context_is_neutral_site preserves that distinction.
    Missing rest, location, roof, temperature, and wind values remain missing;
    no weather value is imputed.
    """

    schedule_columns = (
        "location",
        "home_rest",
        "away_rest",
        "roof",
        "temp",
        "wind",
    )
    rows = _schedule_rows(base_rows, schedules, schedule_columns).with_columns(
        pl.when(pl.col("_feature_is_home"))
        .then(pl.col("_schedule_home_rest"))
        .otherwise(pl.col("_schedule_away_rest"))
        .cast(pl.Float64)
        .alias("_feature_team_rest"),
        pl.when(pl.col("_feature_is_home"))
        .then(pl.col("_schedule_away_rest"))
        .otherwise(pl.col("_schedule_home_rest"))
        .cast(pl.Float64)
        .alias("_feature_opponent_rest"),
        pl.col("_schedule_location").cast(pl.String).str.to_lowercase().str.strip_chars().alias("_feature_location"),
        pl.col("_schedule_roof").cast(pl.String).str.to_lowercase().str.strip_chars().alias("_feature_roof"),
    )

    roof_present = pl.col("_feature_roof").is_not_null()
    rows = rows.with_columns(
        pl.col("_feature_is_home").cast(pl.Float64).alias("context_is_home"),
        _nullable_binary(
            pl.col("_feature_location") == "neutral",
            present=pl.col("_feature_location").is_not_null(),
            alias="context_is_neutral_site",
        ),
        pl.col("_feature_team_rest").alias("context_team_rest_days"),
        (pl.col("_feature_team_rest") - pl.col("_feature_opponent_rest")).alias("context_rest_advantage_days"),
        _nullable_binary(
            pl.col("_feature_team_rest") < 7.0,
            present=pl.col("_feature_team_rest").is_not_null(),
            alias="context_is_short_week",
        ),
        _nullable_binary(
            pl.col("_feature_team_rest") >= 11.0,
            present=pl.col("_feature_team_rest").is_not_null(),
            alias="context_is_post_bye",
        ),
        *[
            _nullable_binary(
                pl.col("_feature_roof") == roof,
                present=roof_present,
                alias=f"context_roof_{roof}",
            )
            for roof in _ROOF_VALUES
        ],
        _nullable_binary(
            ~pl.col("_feature_roof").is_in(_ROOF_VALUES),
            present=roof_present,
            alias="context_roof_other",
        ),
        pl.col("_feature_roof").is_null().cast(pl.Float64).alias("context_roof_missing"),
        pl.col("_schedule_temp").cast(pl.Float64).alias("context_temperature_f"),
        pl.col("_schedule_wind").cast(pl.Float64).alias("context_wind_mph"),
    )
    return _finish(rows, base_rows, CONTEXT_FEATURE_COLUMNS)


def build_market_features(base_rows: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Add closing spread/total features as an independent market ablation.

    Positive market_team_spread means the row's team was favored. The source
    spread_line is already positive when the home team was favored, so only
    away-team rows are negated.
    """

    rows = _schedule_rows(base_rows, schedules, ("spread_line", "total_line")).with_columns(
        pl.when(pl.col("_feature_is_home"))
        .then(pl.col("_schedule_spread_line"))
        .otherwise(-pl.col("_schedule_spread_line"))
        .cast(pl.Float64)
        .alias("market_team_spread"),
        pl.col("_schedule_total_line").cast(pl.Float64).alias("market_game_total"),
    )
    return _finish(rows, base_rows, MARKET_FEATURE_COLUMNS)
