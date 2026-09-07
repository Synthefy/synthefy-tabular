from __future__ import annotations

import polars as pl

_ROLLING_COMPONENTS = (
    "defense_dropbacks",
    "defense_passing_yards_allowed",
    "defense_epa_value",
    "defense_epa_dropbacks",
    "defense_sacks",
    "defense_qb_hits_or_sacks",
    "defense_explosive_completed_passes",
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


def _validated_windows(windows: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(int(window) for window in windows)
    if not normalized or any(window <= 0 for window in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"Rolling windows must be unique positive integers, got {windows}")
    return normalized


def defense_feature_columns(windows: tuple[int, ...]) -> list[str]:
    """Return the numeric, model-facing defense feature contract."""
    windows = _validated_windows(windows)
    features: list[str] = []
    for window in windows:
        suffix = f"last{window}"
        features.extend(
            [
                f"defense_history_games_{suffix}",
                f"defense_dropbacks_faced_per_game_{suffix}",
                f"defense_passing_yards_allowed_per_game_{suffix}",
                f"defense_epa_per_dropback_allowed_{suffix}",
                f"defense_sack_rate_{suffix}",
                f"defense_qb_hit_or_sack_rate_{suffix}",
                f"defense_explosive_completed_pass_rate_{suffix}",
            ]
        )
    features.extend(
        [
            "defense_history_games_season",
            "defense_dropbacks_faced_per_game_season",
            "defense_passing_yards_allowed_per_game_season",
            "defense_epa_per_dropback_allowed_season",
            "defense_sack_rate_season",
            "defense_qb_hit_or_sack_rate_season",
            "defense_explosive_completed_pass_rate_season",
        ]
    )
    return features


def _required_columns(frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _passing_yards_expression(pbp_rows: pl.DataFrame) -> pl.Expr:
    if "passing_yards" in pbp_rows.columns:
        return pl.col("passing_yards").cast(pl.Float64).fill_null(0.0)
    if "yards_gained" in pbp_rows.columns:
        return pl.col("yards_gained").cast(pl.Float64).fill_null(0.0)
    raise ValueError("Play-by-play rows need passing_yards or yards_gained")


def _aggregate_defense_team_games(
    pbp_rows: pl.DataFrame,
    schedules: pl.DataFrame,
    season_type: str,
) -> pl.DataFrame:
    _required_columns(
        schedules,
        {"game_id", "season", "week", "game_type"},
        "Schedules",
    )
    _required_columns(
        pbp_rows,
        {"game_id", "defteam", "qb_dropback", "sack", "qb_hit", "complete_pass", "epa"},
        "Play-by-play rows",
    )
    schedule_games = (
        schedules.filter(pl.col("game_type") == season_type).select("game_id", "season", "week").unique("game_id")
    )
    if "season_type" in pbp_rows.columns:
        pbp_rows = pbp_rows.filter(pl.col("season_type") == season_type)

    is_dropback = pl.col("qb_dropback").fill_null(0.0) == 1.0
    is_sack = pl.col("sack").fill_null(0.0) == 1.0
    is_qb_hit = pl.col("qb_hit").fill_null(0.0) == 1.0
    is_complete = pl.col("complete_pass").fill_null(0.0) == 1.0
    passing_yards = _passing_yards_expression(pbp_rows)
    valid_play = is_dropback & pl.col("game_id").is_not_null() & pl.col("defteam").is_not_null()
    if "play_type" in pbp_rows.columns:
        valid_play &= pl.col("play_type").fill_null("") != "no_play"
    if "qb_kneel" in pbp_rows.columns:
        valid_play &= pl.col("qb_kneel").fill_null(0.0) != 1.0
    if "qb_spike" in pbp_rows.columns:
        valid_play &= pl.col("qb_spike").fill_null(0.0) != 1.0

    return (
        pbp_rows.filter(valid_play)
        .select(
            "game_id",
            _franchise_team("defteam", "history_defense_team"),
            pl.lit(1.0).alias("defense_dropbacks"),
            pl.when(~is_sack).then(passing_yards).otherwise(0.0).alias("defense_passing_yards_allowed"),
            pl.when(pl.col("epa").is_not_null()).then(pl.col("epa")).otherwise(0.0).alias("defense_epa_value"),
            pl.when(pl.col("epa").is_not_null()).then(1.0).otherwise(0.0).alias("defense_epa_dropbacks"),
            pl.when(is_sack).then(1.0).otherwise(0.0).alias("defense_sacks"),
            pl.when(is_qb_hit | is_sack).then(1.0).otherwise(0.0).alias("defense_qb_hits_or_sacks"),
            pl.when(is_complete & ~is_sack & (passing_yards >= 20.0))
            .then(1.0)
            .otherwise(0.0)
            .alias("defense_explosive_completed_passes"),
        )
        .group_by(["game_id", "history_defense_team"])
        .agg(*[pl.col(component).sum() for component in _ROLLING_COMPONENTS])
        .join(schedule_games, on="game_id", how="inner", validate="m:1")
        .with_columns(
            pl.lit(1.0).alias("history_game_count"),
            (pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("history_week_order"),
        )
        .sort(["history_defense_team", "history_week_order", "game_id"])
    )


def _defense_history(
    pbp_rows: pl.DataFrame,
    schedules: pl.DataFrame,
    season_type: str,
    windows: tuple[int, ...],
) -> pl.DataFrame:
    history = _aggregate_defense_team_games(pbp_rows, schedules, season_type)
    for window in windows:
        history = history.with_columns(
            *[
                pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("history_defense_team")
                .alias(f"{component}_last{window}_sum")
                for component in _ROLLING_COMPONENTS
            ],
            pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("history_defense_team")
            .alias(f"history_games_last{window}"),
        )
    return history.with_columns(
        *[
            pl.col(component).cum_sum().over(["history_defense_team", "season"]).alias(f"{component}_season_sum")
            for component in _ROLLING_COMPONENTS
        ],
        pl.col("history_game_count").cum_sum().over(["history_defense_team", "season"]).alias("history_games_season"),
    )


def _period_feature_expressions(period: str) -> list[pl.Expr]:
    count = f"history_games_{period}"
    return [
        pl.col(count).alias(f"defense_history_games_{period}"),
        _safe_ratio(
            f"defense_dropbacks_{period}_sum",
            count,
            f"defense_dropbacks_faced_per_game_{period}",
        ),
        _safe_ratio(
            f"defense_passing_yards_allowed_{period}_sum",
            count,
            f"defense_passing_yards_allowed_per_game_{period}",
        ),
        _safe_ratio(
            f"defense_epa_value_{period}_sum",
            f"defense_epa_dropbacks_{period}_sum",
            f"defense_epa_per_dropback_allowed_{period}",
        ),
        _safe_ratio(
            f"defense_sacks_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_sack_rate_{period}",
        ),
        _safe_ratio(
            f"defense_qb_hits_or_sacks_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_qb_hit_or_sack_rate_{period}",
        ),
        _safe_ratio(
            f"defense_explosive_completed_passes_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_explosive_completed_pass_rate_{period}",
        ),
    ]


def _feature_expressions(windows: tuple[int, ...]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for window in windows:
        expressions.extend(_period_feature_expressions(f"last{window}"))
    expressions.extend(_period_feature_expressions("season"))
    return expressions


def build_defense_features(
    base_rows: pl.DataFrame,
    pbp_rows: pl.DataFrame,
    schedules: pl.DataFrame,
    season_type: str,
    windows: tuple[int, ...],
) -> pl.DataFrame:
    """Attach opponent-defense features using games strictly before each row's week."""
    windows = _validated_windows(windows)
    _required_columns(base_rows, {"game_id", "season", "week", "opponent_team"}, "Base rows")
    feature_columns = defense_feature_columns(windows)
    overlapping = set(feature_columns).intersection(base_rows.columns)
    if overlapping:
        raise ValueError(f"Base rows already contain defense features: {sorted(overlapping)}")

    history = _defense_history(pbp_rows, schedules, season_type, windows).with_columns(*_feature_expressions(windows))
    history_features = history.select(
        "history_defense_team",
        "history_week_order",
        pl.col("season").alias("defense_history_season"),
        *feature_columns,
    )
    rows = (
        base_rows.with_row_index("_defense_base_order")
        .with_columns(
            _franchise_team("opponent_team", "defense_feature_team"),
            (pl.col("season").cast(pl.Int64) * 100 + pl.col("week")).alias("prediction_week_order"),
        )
        .sort("prediction_week_order")
        .join_asof(
            history_features.sort("history_week_order"),
            left_on="prediction_week_order",
            right_on="history_week_order",
            by_left="defense_feature_team",
            by_right="history_defense_team",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
    )

    season_value_columns = [
        column for column in feature_columns if column.endswith("_season") and column != "defense_history_games_season"
    ]
    count_columns = [column for column in feature_columns if column.startswith("defense_history_games_last")]
    return (
        rows.with_columns(
            *[
                pl.when(pl.col("defense_history_season") == pl.col("season"))
                .then(pl.col(column))
                .otherwise(None)
                .alias(column)
                for column in season_value_columns
            ],
            pl.when(pl.col("defense_history_season") == pl.col("season"))
            .then(pl.col("defense_history_games_season"))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("defense_history_games_season"),
            *[pl.col(column).fill_null(0.0).alias(column) for column in count_columns],
        )
        .sort("_defense_base_order")
        .drop(
            "_defense_base_order",
            "defense_feature_team",
            "prediction_week_order",
            "history_defense_team",
            "history_week_order",
            "defense_history_season",
            strict=False,
        )
    )
