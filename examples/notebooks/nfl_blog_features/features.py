from __future__ import annotations

import polars as pl

from .data import DataConfig

_ROLLING_COMPONENTS = (
    "attempts",
    "passing_yards",
    "passing_air_yards",
    "sacks_suffered",
    "dropbacks",
    "passing_epa_value",
    "passing_epa_dropbacks",
    "passing_cpoe_weighted",
    "passing_cpoe_attempts",
)


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator))
        .otherwise(None)
        .alias(alias)
    )


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


def _qb_history(player_stats: pl.DataFrame, schedules: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    history = (
        player_stats.filter(
            (pl.col("season_type") == config.season_type)
            & (pl.col("position") == "QB")
            & pl.col("player_id").is_not_null()
        )
        .select(
            "player_id",
            "game_id",
            "season",
            "week",
            pl.col("attempts").cast(pl.Float64),
            pl.col("passing_yards").cast(pl.Float64),
            pl.col("passing_air_yards").cast(pl.Float64),
            pl.col("sacks_suffered").cast(pl.Float64),
            pl.col("passing_epa").cast(pl.Float64),
            pl.col("passing_cpoe").cast(pl.Float64),
        )
        .join(_game_kickoffs(schedules, config), on="game_id", how="inner", validate="m:1")
        .with_columns(
            (pl.col("attempts") + pl.col("sacks_suffered")).alias("dropbacks"),
            pl.lit(1.0).alias("history_game_count"),
            (pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("history_week_order"),
        )
        .filter(pl.col("dropbacks") > 0)
        .with_columns(
            pl.when(pl.col("passing_epa").is_not_null())
            .then(pl.col("passing_epa"))
            .otherwise(0.0)
            .alias("passing_epa_value"),
            pl.when(pl.col("passing_epa").is_not_null())
            .then(pl.col("dropbacks"))
            .otherwise(0.0)
            .alias("passing_epa_dropbacks"),
            pl.when(pl.col("passing_cpoe").is_not_null())
            .then(pl.col("passing_cpoe") * pl.col("attempts"))
            .otherwise(0.0)
            .alias("passing_cpoe_weighted"),
            pl.when(pl.col("passing_cpoe").is_not_null())
            .then(pl.col("attempts"))
            .otherwise(0.0)
            .alias("passing_cpoe_attempts"),
        )
        .unique(["player_id", "game_id"], keep="first", maintain_order=True)
        .sort(["player_id", "history_week_order", "history_game_kickoff_utc"])
    )

    for window in config.qb_rolling_windows:
        history = history.with_columns(
            *[
                pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("player_id")
                .alias(f"{component}_last{window}_sum")
                for component in _ROLLING_COMPONENTS
            ],
            pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("player_id")
            .alias(f"history_games_last{window}"),
        )

    history = history.with_columns(
        *[
            pl.col(component).cum_sum().over(["player_id", "season"]).alias(f"{component}_season_sum")
            for component in _ROLLING_COMPONENTS
        ],
        pl.col("history_game_count").cum_sum().over(["player_id", "season"]).alias("history_games_season"),
    )
    return history


def qb_feature_columns(windows: tuple[int, ...]) -> list[str]:
    features = [
        "qb_attempts_lag1",
        "qb_passing_yards_lag1",
        "qb_ypa_lag1",
        "qb_epa_per_dropback_lag1",
        "qb_cpoe_lag1",
        "qb_air_yards_per_attempt_lag1",
        "qb_sack_rate_lag1",
    ]
    for window in windows:
        features.extend(
            [
                f"qb_history_games_last{window}",
                f"qb_attempts_avg_last{window}",
                f"qb_passing_yards_avg_last{window}",
                f"qb_ypa_last{window}",
                f"qb_epa_per_dropback_last{window}",
                f"qb_cpoe_last{window}",
                f"qb_air_yards_per_attempt_last{window}",
                f"qb_sack_rate_last{window}",
            ]
        )
    features.extend(
        [
            "qb_history_games_season",
            "qb_attempts_avg_season",
            "qb_passing_yards_avg_season",
            "qb_ypa_season",
            "qb_epa_per_dropback_season",
            "qb_cpoe_season",
            "qb_air_yards_per_attempt_season",
            "qb_sack_rate_season",
            "qb_days_since_previous_game",
        ]
    )
    return features


def _feature_expressions(windows: tuple[int, ...]) -> list[pl.Expr]:
    expressions = [
        pl.col("attempts").alias("qb_attempts_lag1"),
        pl.col("passing_yards").alias("qb_passing_yards_lag1"),
        _safe_ratio("passing_yards", "attempts", "qb_ypa_lag1"),
        _safe_ratio("passing_epa_value", "passing_epa_dropbacks", "qb_epa_per_dropback_lag1"),
        _safe_ratio("passing_cpoe_weighted", "passing_cpoe_attempts", "qb_cpoe_lag1"),
        _safe_ratio("passing_air_yards", "attempts", "qb_air_yards_per_attempt_lag1"),
        _safe_ratio("sacks_suffered", "dropbacks", "qb_sack_rate_lag1"),
    ]
    for window in windows:
        suffix = f"last{window}"
        count = f"history_games_{suffix}"
        expressions.extend(
            [
                pl.col(count).alias(f"qb_history_games_{suffix}"),
                _safe_ratio(f"attempts_{suffix}_sum", count, f"qb_attempts_avg_{suffix}"),
                _safe_ratio(f"passing_yards_{suffix}_sum", count, f"qb_passing_yards_avg_{suffix}"),
                _safe_ratio(f"passing_yards_{suffix}_sum", f"attempts_{suffix}_sum", f"qb_ypa_{suffix}"),
                _safe_ratio(
                    f"passing_epa_value_{suffix}_sum",
                    f"passing_epa_dropbacks_{suffix}_sum",
                    f"qb_epa_per_dropback_{suffix}",
                ),
                _safe_ratio(
                    f"passing_cpoe_weighted_{suffix}_sum",
                    f"passing_cpoe_attempts_{suffix}_sum",
                    f"qb_cpoe_{suffix}",
                ),
                _safe_ratio(
                    f"passing_air_yards_{suffix}_sum",
                    f"attempts_{suffix}_sum",
                    f"qb_air_yards_per_attempt_{suffix}",
                ),
                _safe_ratio(
                    f"sacks_suffered_{suffix}_sum",
                    f"dropbacks_{suffix}_sum",
                    f"qb_sack_rate_{suffix}",
                ),
            ]
        )
    expressions.extend(
        [
            pl.col("history_games_season").alias("qb_history_games_season"),
            _safe_ratio("attempts_season_sum", "history_games_season", "qb_attempts_avg_season"),
            _safe_ratio(
                "passing_yards_season_sum",
                "history_games_season",
                "qb_passing_yards_avg_season",
            ),
            _safe_ratio("passing_yards_season_sum", "attempts_season_sum", "qb_ypa_season"),
            _safe_ratio(
                "passing_epa_value_season_sum",
                "passing_epa_dropbacks_season_sum",
                "qb_epa_per_dropback_season",
            ),
            _safe_ratio(
                "passing_cpoe_weighted_season_sum",
                "passing_cpoe_attempts_season_sum",
                "qb_cpoe_season",
            ),
            _safe_ratio(
                "passing_air_yards_season_sum",
                "attempts_season_sum",
                "qb_air_yards_per_attempt_season",
            ),
            _safe_ratio("sacks_suffered_season_sum", "dropbacks_season_sum", "qb_sack_rate_season"),
        ]
    )
    return expressions


def build_qb_rolling_features(
    base_rows: pl.DataFrame,
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    config: DataConfig,
    *,
    qb_id_column: str = "anticipated_qb_id",
) -> pl.DataFrame:
    history = _qb_history(player_stats, schedules, config).with_columns(
        *_feature_expressions(config.qb_rolling_windows)
    )
    feature_columns = qb_feature_columns(config.qb_rolling_windows)
    history_features = history.select(
        pl.col("player_id").alias("history_qb_id"),
        "history_week_order",
        pl.col("season").alias("qb_history_season"),
        pl.col("game_id").alias("qb_previous_game_id"),
        "history_game_kickoff_utc",
        *[column for column in feature_columns if column != "qb_days_since_previous_game"],
    )

    rows = (
        base_rows.with_columns((pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("prediction_week_order"))
        .sort("prediction_week_order")
        .join_asof(
            history_features.sort("history_week_order"),
            left_on="prediction_week_order",
            right_on="history_week_order",
            by_left=qb_id_column,
            by_right="history_qb_id",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
    )

    season_value_columns = [
        column for column in feature_columns if column.endswith("_season") and column != "qb_history_games_season"
    ]
    rows = rows.with_columns(
        *[
            pl.when(pl.col("qb_history_season") == pl.col("season")).then(pl.col(column)).otherwise(None).alias(column)
            for column in season_value_columns
        ],
        pl.when(pl.col("qb_history_season") == pl.col("season"))
        .then(pl.col("qb_history_games_season"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("qb_history_games_season"),
        (pl.col("kickoff_utc") - pl.col("history_game_kickoff_utc"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("qb_days_since_previous_game"),
    )
    count_columns = [column for column in feature_columns if column.startswith("qb_history_games_last")]
    return (
        rows.with_columns(*[pl.col(column).fill_null(0.0).alias(column) for column in count_columns])
        .drop(
            "prediction_week_order",
            "history_week_order",
            "history_qb_id",
            strict=False,
        )
        .sort(["kickoff_utc", "game_id", "team"])
    )


def summarize_qb_features(rows: pl.DataFrame, windows: tuple[int, ...]) -> dict[str, int]:
    return {
        "qb_feature_columns": len(qb_feature_columns(windows)),
        "rows_with_prior_qb_game": rows.filter(pl.col("qb_previous_game_id").is_not_null()).height,
        "rows_without_prior_qb_game": rows.filter(pl.col("qb_previous_game_id").is_null()).height,
    }
