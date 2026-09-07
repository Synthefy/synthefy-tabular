from __future__ import annotations

import polars as pl

_TREND_BASES = (
    "qb_attempts_avg",
    "qb_passing_yards_avg",
    "qb_ypa",
    "qb_epa_per_dropback",
    "qb_cpoe",
    "qb_air_yards_per_attempt",
    "qb_sack_rate",
    "offense_plays_per_game",
    "offense_epa_per_play",
    "offense_neutral_pass_rate",
    "defense_dropbacks_faced_per_game",
    "defense_passing_yards_allowed_per_game",
    "defense_epa_per_dropback_allowed",
    "defense_sack_rate",
    "defense_qb_hit_or_sack_rate",
    "defense_explosive_completed_pass_rate",
)

_DISPERSION_BASES = (
    "qb_attempts_avg",
    "qb_passing_yards_avg",
    "qb_epa_per_dropback",
    "offense_epa_per_play",
    "defense_passing_yards_allowed_per_game",
    "defense_epa_per_dropback_allowed",
)

TREND_FEATURE_COLUMNS = tuple(
    [f"trend_{base}_last3_minus_last8" for base in _TREND_BASES]
    + [f"dispersion_{base}_last3_vs_last8" for base in _DISPERSION_BASES]
)

MATCHUP_FEATURE_COLUMNS = (
    "matchup_expected_attempts_last8",
    "matchup_expected_passing_yards_last8",
    "matchup_expected_dropbacks_last8",
    "matchup_epa_blend_last8",
    "matchup_sack_pressure_last8",
    "matchup_air_explosive_last8",
    "matchup_wind_air_penalty_last8",
    "matchup_rest_pace_last8",
)


def _require_columns(rows: pl.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(rows.columns)
    if missing:
        raise ValueError(f"Derived features require columns: {sorted(missing)}")


def _paired_mean(left: str, right: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(left).is_not_null() & pl.col(right).is_not_null())
        .then((pl.col(left).cast(pl.Float64) + pl.col(right).cast(pl.Float64)) / 2.0)
        .otherwise(None)
        .alias(alias)
    )


def build_derived_features(rows: pl.DataFrame) -> pl.DataFrame:
    """Append algebraic pregame features without introducing new observations."""
    required = {
        *(f"{base}_last3" for base in _TREND_BASES),
        *(f"{base}_last8" for base in _TREND_BASES),
        "context_roof_outdoors",
        "context_wind_mph",
        "context_rest_advantage_days",
    }
    _require_columns(rows, required)
    overlap = set(TREND_FEATURE_COLUMNS + MATCHUP_FEATURE_COLUMNS).intersection(rows.columns)
    if overlap:
        raise ValueError(f"Rows already contain derived features: {sorted(overlap)}")

    trend_expressions = [
        (pl.col(f"{base}_last3") - pl.col(f"{base}_last8")).alias(f"trend_{base}_last3_minus_last8")
        for base in _TREND_BASES
    ]
    dispersion_expressions = [
        (pl.col(f"{base}_last3") - pl.col(f"{base}_last8")).abs().alias(f"dispersion_{base}_last3_vs_last8")
        for base in _DISPERSION_BASES
    ]
    matchup_expressions = [
        _paired_mean(
            "qb_attempts_avg_last8",
            "defense_dropbacks_faced_per_game_last8",
            "matchup_expected_attempts_last8",
        ),
        _paired_mean(
            "qb_passing_yards_avg_last8",
            "defense_passing_yards_allowed_per_game_last8",
            "matchup_expected_passing_yards_last8",
        ),
        _paired_mean(
            "offense_plays_per_game_last8",
            "defense_dropbacks_faced_per_game_last8",
            "matchup_expected_dropbacks_last8",
        ),
        _paired_mean(
            "qb_epa_per_dropback_last8",
            "defense_epa_per_dropback_allowed_last8",
            "matchup_epa_blend_last8",
        ),
        (pl.col("qb_sack_rate_last8") + pl.col("defense_sack_rate_last8")).alias("matchup_sack_pressure_last8"),
        (pl.col("qb_air_yards_per_attempt_last8") * pl.col("defense_explosive_completed_pass_rate_last8")).alias(
            "matchup_air_explosive_last8"
        ),
        (pl.col("context_roof_outdoors") * pl.col("context_wind_mph") * pl.col("qb_air_yards_per_attempt_last8")).alias(
            "matchup_wind_air_penalty_last8"
        ),
        (pl.col("context_rest_advantage_days") * pl.col("offense_plays_per_game_last8")).alias(
            "matchup_rest_pace_last8"
        ),
    ]
    return rows.with_columns(*trend_expressions, *dispersion_expressions, *matchup_expressions)
