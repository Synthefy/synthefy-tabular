from __future__ import annotations

import polars as pl

from .data import DataConfig

_ROLLING_COMPONENTS = (
    "offense_plays",
    "offense_epa_value",
    "offense_epa_plays",
    "neutral_dropbacks",
    "neutral_plays",
)


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator))
        .otherwise(None)
        .alias(alias)
    )


def _franchise_team(column: str, alias: str) -> pl.Expr:
    return pl.when(pl.col(column).is_in(["OAK", "LV"])).then(pl.lit("LV")).otherwise(pl.col(column)).alias(alias)


def _game_kickoffs(schedules: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    return (
        schedules.filter(
            (pl.col("game_type") == config.season_type)
            & pl.col("season").is_between(config.warmup_start_season, config.test_season)
        )
        .select(
            "game_id",
            pl.concat_str(["gameday", "gametime"], separator=" ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="America/New_York", strict=True)
            .dt.convert_time_zone("UTC")
            .alias("history_game_kickoff_utc"),
        )
        .unique("game_id")
    )


def aggregate_offense_team_games(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    config: DataConfig,
) -> pl.DataFrame:
    is_dropback = pl.col("qb_dropback").fill_null(0.0) == 1.0
    is_rush = pl.col("rush_attempt").fill_null(0.0) == 1.0
    is_neutral = (pl.col("qtr") <= config.neutral_max_quarter) & pl.col("wp").is_between(
        config.neutral_wp_lower, config.neutral_wp_upper, closed="both"
    )
    return (
        pbp.filter(
            (pl.col("season_type") == config.season_type)
            & pl.col("season").is_between(config.warmup_start_season, config.test_season)
            & pl.col("game_id").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("play_id").is_not_null()
            & pl.col("play_type").is_in(["pass", "run"])
            & (pl.col("qb_kneel").fill_null(0.0) != 1.0)
            & (pl.col("qb_spike").fill_null(0.0) != 1.0)
            & (is_dropback | is_rush)
        )
        .with_columns(
            _franchise_team("posteam", "history_team"),
            pl.lit(1.0).alias("offense_plays"),
            pl.when(pl.col("epa").is_not_null()).then(pl.col("epa")).otherwise(0.0).alias("offense_epa_value"),
            pl.when(pl.col("epa").is_not_null()).then(1.0).otherwise(0.0).alias("offense_epa_plays"),
            pl.when(is_neutral & is_dropback).then(1.0).otherwise(0.0).alias("neutral_dropbacks"),
            pl.when(is_neutral).then(1.0).otherwise(0.0).alias("neutral_plays"),
        )
        .group_by(["game_id", "season", "week", "history_team"])
        .agg(*[pl.col(component).sum() for component in _ROLLING_COMPONENTS])
        .join(_game_kickoffs(schedules, config), on="game_id", how="inner", validate="m:1")
        .with_columns(
            pl.lit(1.0).alias("history_game_count"),
            (pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("history_week_order"),
        )
        .sort(["history_team", "history_week_order", "history_game_kickoff_utc"])
    )


def _offense_history(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    config: DataConfig,
) -> pl.DataFrame:
    history = aggregate_offense_team_games(pbp, schedules, config)
    for window in config.offense_rolling_windows:
        history = history.with_columns(
            *[
                pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("history_team")
                .alias(f"{component}_last{window}_sum")
                for component in _ROLLING_COMPONENTS
            ],
            pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("history_team")
            .alias(f"history_games_last{window}"),
        )

    return history.with_columns(
        *[
            pl.col(component).cum_sum().over(["history_team", "season"]).alias(f"{component}_season_sum")
            for component in _ROLLING_COMPONENTS
        ],
        pl.col("history_game_count").cum_sum().over(["history_team", "season"]).alias("history_games_season"),
    )


def offense_feature_columns(windows: tuple[int, ...]) -> list[str]:
    features = [
        "offense_plays_lag1",
        "offense_epa_per_play_lag1",
        "offense_neutral_pass_rate_lag1",
    ]
    for window in windows:
        features.extend(
            [
                f"offense_history_games_last{window}",
                f"offense_plays_per_game_last{window}",
                f"offense_epa_per_play_last{window}",
                f"offense_neutral_pass_rate_last{window}",
            ]
        )
    features.extend(
        [
            "offense_history_games_season",
            "offense_plays_per_game_season",
            "offense_epa_per_play_season",
            "offense_neutral_pass_rate_season",
        ]
    )
    return features


def _feature_expressions(windows: tuple[int, ...]) -> list[pl.Expr]:
    expressions = [
        pl.col("offense_plays").alias("offense_plays_lag1"),
        _safe_ratio("offense_epa_value", "offense_epa_plays", "offense_epa_per_play_lag1"),
        _safe_ratio("neutral_dropbacks", "neutral_plays", "offense_neutral_pass_rate_lag1"),
    ]
    for window in windows:
        suffix = f"last{window}"
        count = f"history_games_{suffix}"
        expressions.extend(
            [
                pl.col(count).alias(f"offense_history_games_{suffix}"),
                _safe_ratio(f"offense_plays_{suffix}_sum", count, f"offense_plays_per_game_{suffix}"),
                _safe_ratio(
                    f"offense_epa_value_{suffix}_sum",
                    f"offense_epa_plays_{suffix}_sum",
                    f"offense_epa_per_play_{suffix}",
                ),
                _safe_ratio(
                    f"neutral_dropbacks_{suffix}_sum",
                    f"neutral_plays_{suffix}_sum",
                    f"offense_neutral_pass_rate_{suffix}",
                ),
            ]
        )
    expressions.extend(
        [
            pl.col("history_games_season").alias("offense_history_games_season"),
            _safe_ratio(
                "offense_plays_season_sum",
                "history_games_season",
                "offense_plays_per_game_season",
            ),
            _safe_ratio(
                "offense_epa_value_season_sum",
                "offense_epa_plays_season_sum",
                "offense_epa_per_play_season",
            ),
            _safe_ratio(
                "neutral_dropbacks_season_sum",
                "neutral_plays_season_sum",
                "offense_neutral_pass_rate_season",
            ),
        ]
    )
    return expressions


def build_offense_rolling_features(
    base_rows: pl.DataFrame,
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    config: DataConfig,
) -> pl.DataFrame:
    history = _offense_history(pbp, schedules, config).with_columns(
        *_feature_expressions(config.offense_rolling_windows)
    )
    feature_columns = offense_feature_columns(config.offense_rolling_windows)
    history_features = history.select(
        "history_team",
        "history_week_order",
        pl.col("season").alias("offense_history_season"),
        pl.col("game_id").alias("offense_previous_game_id"),
        *feature_columns,
    )

    rows = (
        base_rows.with_columns(
            _franchise_team("team", "feature_team"),
            (pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("prediction_week_order"),
        )
        .sort("prediction_week_order")
        .join_asof(
            history_features.sort("history_week_order"),
            left_on="prediction_week_order",
            right_on="history_week_order",
            by_left="feature_team",
            by_right="history_team",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
    )

    season_value_columns = [
        column for column in feature_columns if column.endswith("_season") and column != "offense_history_games_season"
    ]
    rows = rows.with_columns(
        *[
            pl.when(pl.col("offense_history_season") == pl.col("season"))
            .then(pl.col(column))
            .otherwise(None)
            .alias(column)
            for column in season_value_columns
        ],
        pl.when(pl.col("offense_history_season") == pl.col("season"))
        .then(pl.col("offense_history_games_season"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("offense_history_games_season"),
    )
    count_columns = [column for column in feature_columns if column.startswith("offense_history_games_last")]
    return (
        rows.with_columns(*[pl.col(column).fill_null(0.0).alias(column) for column in count_columns])
        .drop(
            "feature_team",
            "prediction_week_order",
            "history_team",
            "history_week_order",
            strict=False,
        )
        .sort(["kickoff_utc", "game_id", "team"])
    )


def summarize_offense_features(rows: pl.DataFrame, windows: tuple[int, ...]) -> dict[str, int]:
    return {
        "offense_feature_columns": len(offense_feature_columns(windows)),
        "rows_with_prior_offense_game": rows.filter(pl.col("offense_previous_game_id").is_not_null()).height,
        "rows_without_prior_offense_game": rows.filter(pl.col("offense_previous_game_id").is_null()).height,
    }
