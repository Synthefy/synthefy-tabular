"""Historical and checkpoint feature construction for the NFL Nori notebook."""

from __future__ import annotations

availability_features__STATUS_BUCKETS = ("out", "doubtful", "questionable")


def availability_features_availability_feature_columns() -> list[str]:
    """Return the numeric columns produced by :func:`build_availability_features`."""
    columns = ["availability_report_coverage", "availability_relevant_player_count"]
    for group in ("ol", "pass_catcher"):
        columns.extend((f"availability_{group}_{status}_count" for status in availability_features__STATUS_BUCKETS))
    return columns


import numpy as checkpoint_history_np

import polars as checkpoint_history_pl

checkpoint_history_WINDOWS = (3, 8)

checkpoint_history_SOURCES = (
    "live_qb_passing_yards",
    "live_qb_attempts",
    "live_qb_ypa",
    "remaining_passing_yards",
)


def checkpoint_history_add_checkpoint_history(
    rows: checkpoint_history_pl.DataFrame,
) -> tuple[checkpoint_history_pl.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Use only earlier season/weeks; never current-game or same-week labels.

    Call separately for Q1 and halftime. Cold starters have missing averages
    and a zero history count. Preserve input order and all original columns.
    """
    keys = ["season", "week", "game_id", "actual_qb_id"]
    if rows.select(keys).is_duplicated().any():
        raise ValueError("duplicate QB-game checkpoint rows")
    records = rows.to_dicts()
    histories: dict[str, list[dict]] = {}
    features: dict[int, dict] = {}
    groups: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(records):
        if row["actual_qb_id"] is None:
            raise ValueError("checkpoint history requires QB identity")
        groups.setdefault((row["season"], row["week"]), []).append(i)
    averages = tuple(
        (
            f"checkpoint_history_{source}_mean_last{window}"
            for window in checkpoint_history_WINDOWS
            for source in checkpoint_history_SOURCES
        )
    ) + ("checkpoint_history_games_last8",)
    deviations = tuple(
        (
            f"checkpoint_deviation_{source}_last{window}"
            for window in checkpoint_history_WINDOWS
            for source in checkpoint_history_SOURCES[:2]
        )
    )
    for period, indices in sorted(groups.items()):
        for i in indices:
            row = records[i]
            prior = histories.get(row["actual_qb_id"], [])
            prior = [r for r in prior if r["kickoff_utc"] < row["kickoff_utc"]]
            f = {"checkpoint_history_games_last8": float(min(len(prior), 8))}
            for window in checkpoint_history_WINDOWS:
                recent = prior[-window:]
                for source in checkpoint_history_SOURCES:
                    values = [
                        r[source] for r in recent if r[source] is not None and checkpoint_history_np.isfinite(r[source])
                    ]
                    mean = float(checkpoint_history_np.mean(values)) if values else None
                    f[f"checkpoint_history_{source}_mean_last{window}"] = mean
                    if source in checkpoint_history_SOURCES[:2]:
                        current = row[source]
                        f[f"checkpoint_deviation_{source}_last{window}"] = (
                            float(current - mean) if current is not None and mean is not None else None
                        )
            features[i] = f
        for i in sorted(indices, key=lambda j: records[j]["kickoff_utc"]):
            row = records[i]
            if row.get("live_evaluation_eligible", True):
                histories.setdefault(row["actual_qb_id"], []).append(row)
    return (
        rows.hstack(
            checkpoint_history_pl.DataFrame(
                [features[i] for i in range(len(records))],
                schema={c: checkpoint_history_pl.Float64 for c in (*averages, *deviations)},
            )
        ),
        averages,
        deviations,
    )


from collections.abc import Sequence as context_features_Sequence

import polars as context_features_pl

context_features_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
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

context_features_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_team_spread",
    "market_game_total",
)

context_features__BASE_KEYS = ("game_id", "team")

context_features__ROOF_VALUES = ("outdoors", "open", "closed", "dome")


def context_features__require_columns(
    frame: context_features_pl.DataFrame,
    columns: context_features_Sequence[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def context_features__validate_base_rows(
    base_rows: context_features_pl.DataFrame,
) -> None:
    context_features__require_columns(base_rows, context_features__BASE_KEYS, frame_name="base_rows")
    if base_rows.select(context_features_pl.struct(context_features__BASE_KEYS).is_duplicated().any()).item():
        raise ValueError("base_rows must have exactly one row per game_id/team")


def context_features__schedule_rows(
    base_rows: context_features_pl.DataFrame,
    schedules: context_features_pl.DataFrame,
    schedule_columns: context_features_Sequence[str],
) -> context_features_pl.DataFrame:
    context_features__validate_base_rows(base_rows)
    required_schedule_columns = ("game_id", "home_team", "away_team", *schedule_columns)
    context_features__require_columns(schedules, required_schedule_columns, frame_name="schedules")
    schedule_lookup = schedules.select(
        context_features_pl.col("game_id"),
        context_features_pl.col("home_team").alias("_schedule_home_team"),
        context_features_pl.col("away_team").alias("_schedule_away_team"),
        *[context_features_pl.col(column).alias(f"_schedule_{column}") for column in schedule_columns],
    )
    if schedule_lookup.select(context_features_pl.col("game_id").is_duplicated().any()).item():
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
            (context_features_pl.col("team") == context_features_pl.col("_schedule_home_team")).alias(
                "_feature_is_home"
            ),
            (context_features_pl.col("team") == context_features_pl.col("_schedule_away_team")).alias(
                "_feature_is_away"
            ),
        )
    )
    missing_games = (
        rows.filter(context_features_pl.col("_schedule_home_team").is_null()).get_column("game_id").unique().to_list()
    )
    if missing_games:
        raise ValueError(f"base_rows contain game_ids absent from schedules: {missing_games[:5]}")
    invalid_teams = rows.filter(
        ~(
            context_features_pl.col("_feature_is_home").fill_null(False)
            ^ context_features_pl.col("_feature_is_away").fill_null(False)
        )
    ).select("game_id", "team")
    if invalid_teams.height:
        examples = invalid_teams.head(5).rows()
        raise ValueError(f"base_rows teams must match exactly one schedule side: {examples}")
    return rows


def context_features__nullable_binary(
    condition: context_features_pl.Expr,
    *,
    present: context_features_pl.Expr,
    alias: str,
) -> context_features_pl.Expr:
    return (
        context_features_pl.when(present).then(condition.cast(context_features_pl.Float64)).otherwise(None).alias(alias)
    )


def context_features__finish(
    rows: context_features_pl.DataFrame,
    base_rows: context_features_pl.DataFrame,
    feature_columns: context_features_Sequence[str],
) -> context_features_pl.DataFrame:
    output = rows.sort("_feature_row_order").select(*base_rows.columns, *feature_columns)
    if output.height != base_rows.height:
        raise RuntimeError("feature builder changed the number of input rows")
    if output.select(context_features_pl.struct(context_features__BASE_KEYS).is_duplicated().any()).item():
        raise RuntimeError("feature builder produced duplicate game_id/team rows")
    return output


def context_features_build_context_features(
    base_rows: context_features_pl.DataFrame, schedules: context_features_pl.DataFrame
) -> context_features_pl.DataFrame:
    """Add numeric sports-context features without market or result fields.

    context_is_post_bye uses at least 11 days between games. That threshold
    separates a normal/mini-bye Thursday-to-Sunday gap (10 days) from the
    shortest ordinary NFL bye transition, Sunday-to-Thursday (11 days).
    context_is_home follows nflverse's designated home team even at a neutral
    site; context_is_neutral_site preserves that distinction.
    Missing rest, location, roof, temperature, and wind values remain missing;
    no weather value is imputed.
    """
    schedule_columns = ("location", "home_rest", "away_rest", "roof", "temp", "wind")
    rows = context_features__schedule_rows(base_rows, schedules, schedule_columns).with_columns(
        context_features_pl.when(context_features_pl.col("_feature_is_home"))
        .then(context_features_pl.col("_schedule_home_rest"))
        .otherwise(context_features_pl.col("_schedule_away_rest"))
        .cast(context_features_pl.Float64)
        .alias("_feature_team_rest"),
        context_features_pl.when(context_features_pl.col("_feature_is_home"))
        .then(context_features_pl.col("_schedule_away_rest"))
        .otherwise(context_features_pl.col("_schedule_home_rest"))
        .cast(context_features_pl.Float64)
        .alias("_feature_opponent_rest"),
        context_features_pl.col("_schedule_location")
        .cast(context_features_pl.String)
        .str.to_lowercase()
        .str.strip_chars()
        .alias("_feature_location"),
        context_features_pl.col("_schedule_roof")
        .cast(context_features_pl.String)
        .str.to_lowercase()
        .str.strip_chars()
        .alias("_feature_roof"),
    )
    roof_present = context_features_pl.col("_feature_roof").is_not_null()
    rows = rows.with_columns(
        context_features_pl.col("_feature_is_home").cast(context_features_pl.Float64).alias("context_is_home"),
        context_features__nullable_binary(
            context_features_pl.col("_feature_location") == "neutral",
            present=context_features_pl.col("_feature_location").is_not_null(),
            alias="context_is_neutral_site",
        ),
        context_features_pl.col("_feature_team_rest").alias("context_team_rest_days"),
        (context_features_pl.col("_feature_team_rest") - context_features_pl.col("_feature_opponent_rest")).alias(
            "context_rest_advantage_days"
        ),
        context_features__nullable_binary(
            context_features_pl.col("_feature_team_rest") < 7.0,
            present=context_features_pl.col("_feature_team_rest").is_not_null(),
            alias="context_is_short_week",
        ),
        context_features__nullable_binary(
            context_features_pl.col("_feature_team_rest") >= 11.0,
            present=context_features_pl.col("_feature_team_rest").is_not_null(),
            alias="context_is_post_bye",
        ),
        *[
            context_features__nullable_binary(
                context_features_pl.col("_feature_roof") == roof,
                present=roof_present,
                alias=f"context_roof_{roof}",
            )
            for roof in context_features__ROOF_VALUES
        ],
        context_features__nullable_binary(
            ~context_features_pl.col("_feature_roof").is_in(context_features__ROOF_VALUES),
            present=roof_present,
            alias="context_roof_other",
        ),
        context_features_pl.col("_feature_roof")
        .is_null()
        .cast(context_features_pl.Float64)
        .alias("context_roof_missing"),
        context_features_pl.col("_schedule_temp").cast(context_features_pl.Float64).alias("context_temperature_f"),
        context_features_pl.col("_schedule_wind").cast(context_features_pl.Float64).alias("context_wind_mph"),
    )
    return context_features__finish(rows, base_rows, context_features_CONTEXT_FEATURE_COLUMNS)


from collections.abc import Callable as data_Callable

from dataclasses import dataclass as data_dataclass

from pathlib import Path as data_Path

import nflreadpy as data_nfl

import polars as data_pl


@data_dataclass(frozen=True)
class data_DataConfig:
    warmup_start_season: int
    first_eligible_season: int
    validation_season: int
    test_season: int
    season_type: str
    prediction_cutoff_minutes: int
    cache_dir: data_Path
    base_dataset_path: data_Path
    feature_dataset_path: data_Path
    qb_rolling_windows: tuple[int, ...]
    offense_rolling_windows: tuple[int, ...]
    neutral_wp_lower: float
    neutral_wp_upper: float
    neutral_max_quarter: int
    include_starter_mismatch_trades: bool
    defense_rolling_windows: tuple[int, ...] = (3, 8)

    @property
    def stat_seasons(self) -> list[int]:
        return list(range(self.warmup_start_season, self.test_season + 1))

    @property
    def depth_chart_seasons(self) -> list[int]:
        return list(range(self.first_eligible_season, self.test_season + 1))


def data__load_cached(
    cache_path: data_Path,
    loader: data_Callable[[list[int]], data_pl.DataFrame],
    seasons: list[int],
    refresh: bool,
) -> data_pl.DataFrame:
    if cache_path.exists() and (not refresh):
        return data_pl.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = loader(seasons)
    frame.write_parquet(cache_path)
    return frame


def data__load_pbp_starter_fields(seasons: list[int]) -> data_pl.DataFrame:
    fields = ("game_id", "season_type", "play_id", "posteam", "passer_player_id")
    return data_pl.concat(
        [data_nfl.load_pbp(season).select(*fields) for season in seasons],
        how="diagonal_relaxed",
    )


def data__load_pbp_offense_fields(seasons: list[int]) -> data_pl.DataFrame:
    fields = (
        "game_id",
        "season",
        "season_type",
        "week",
        "play_id",
        "posteam",
        "qtr",
        "wp",
        "play_type",
        "qb_dropback",
        "rush_attempt",
        "qb_kneel",
        "qb_spike",
        "epa",
    )
    return data_pl.concat(
        [data_nfl.load_pbp(season).select(*fields) for season in seasons],
        how="diagonal_relaxed",
    )


def data__load_pbp_defense_fields(seasons: list[int]) -> data_pl.DataFrame:
    fields = (
        "game_id",
        "season_type",
        "play_id",
        "defteam",
        "play_type",
        "qb_dropback",
        "qb_kneel",
        "qb_spike",
        "sack",
        "qb_hit",
        "complete_pass",
        "passing_yards",
        "epa",
    )
    return data_pl.concat(
        [data_nfl.load_pbp(season).select(*fields) for season in seasons],
        how="diagonal_relaxed",
    )


def data_load_nflverse_tables(config: data_DataConfig, refresh: bool = False) -> dict[str, data_pl.DataFrame]:
    season_tag = f"{config.warmup_start_season}_{config.test_season}"
    depth_tag = f"{config.first_eligible_season}_{config.test_season}"
    pbp_starter_seasons = config.depth_chart_seasons
    return {
        "schedules": data__load_cached(
            config.cache_dir / f"schedules_{season_tag}.parquet",
            data_nfl.load_schedules,
            config.stat_seasons,
            refresh,
        ),
        "player_stats": data__load_cached(
            config.cache_dir / f"player_stats_{season_tag}.parquet",
            data_nfl.load_player_stats,
            config.stat_seasons,
            refresh,
        ),
        "participation": data__load_cached(
            config.cache_dir / f"participation_{depth_tag}.parquet",
            data_nfl.load_participation,
            config.depth_chart_seasons,
            refresh,
        ),
        "pbp_starter_plays": data__load_cached(
            config.cache_dir / f"pbp_starter_plays_{pbp_starter_seasons[0]}_{pbp_starter_seasons[-1]}.parquet",
            data__load_pbp_starter_fields,
            pbp_starter_seasons,
            refresh,
        ),
        "pbp_offense_plays": data__load_cached(
            config.cache_dir / f"pbp_offense_plays_{season_tag}.parquet",
            data__load_pbp_offense_fields,
            config.stat_seasons,
            refresh,
        ),
        "pbp_defense_plays": data__load_cached(
            config.cache_dir / f"pbp_defense_plays_{season_tag}.parquet",
            data__load_pbp_defense_fields,
            config.stat_seasons,
            refresh,
        ),
        "depth_charts": data__load_cached(
            config.cache_dir / f"depth_charts_{depth_tag}.parquet",
            data_nfl.load_depth_charts,
            config.depth_chart_seasons,
            refresh,
        ),
    }


def data_normalize_schedules(
    schedules: data_pl.DataFrame,
    season_type: str,
    first_eligible_season: int,
    test_season: int,
    cutoff_minutes: int,
) -> data_pl.DataFrame:
    games = (
        schedules.filter(
            (data_pl.col("game_type") == season_type)
            & data_pl.col("season").is_between(first_eligible_season, test_season)
        )
        .with_columns(
            data_pl.concat_str(["gameday", "gametime"], separator=" ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="America/New_York", strict=True)
            .dt.convert_time_zone("UTC")
            .alias("kickoff_utc")
        )
        .with_columns(
            (data_pl.col("kickoff_utc") - data_pl.duration(minutes=cutoff_minutes)).alias("prediction_cutoff_utc")
        )
    )
    shared = [
        "game_id",
        "season",
        "week",
        "gameday",
        "gametime",
        "kickoff_utc",
        "prediction_cutoff_utc",
        "roof",
        "surface",
        "temp",
        "wind",
    ]
    home = games.select(
        *shared,
        data_pl.col("home_team").alias("team"),
        data_pl.col("away_team").alias("opponent_team"),
        data_pl.lit(1, dtype=data_pl.Int8).alias("is_home"),
        data_pl.col("home_rest").alias("rest_days"),
    )
    away = games.select(
        *shared,
        data_pl.col("away_team").alias("team"),
        data_pl.col("home_team").alias("opponent_team"),
        data_pl.lit(0, dtype=data_pl.Int8).alias("is_home"),
        data_pl.col("away_rest").alias("rest_days"),
    )
    return data_pl.concat([home, away]).sort(["kickoff_utc", "game_id", "team"])


def data__unique_starters(
    frame: data_pl.DataFrame, keys: list[str], id_column: str, name_column: str
) -> data_pl.DataFrame:
    return (
        frame.filter(data_pl.col(id_column).is_not_null())
        .group_by(keys)
        .agg(
            data_pl.col(id_column).n_unique().alias("starter_candidate_count"),
            data_pl.col(id_column).first().alias("anticipated_qb_id"),
            data_pl.col(name_column).first().alias("anticipated_qb_name"),
        )
        .with_columns(
            data_pl.when(data_pl.col("starter_candidate_count") == 1)
            .then(data_pl.col("anticipated_qb_id"))
            .otherwise(None)
            .alias("anticipated_qb_id"),
            data_pl.when(data_pl.col("starter_candidate_count") == 1)
            .then(data_pl.col("anticipated_qb_name"))
            .otherwise(None)
            .alias("anticipated_qb_name"),
        )
    )


def data_select_anticipated_starters(
    schedule_rows: data_pl.DataFrame, depth_charts: data_pl.DataFrame
) -> data_pl.DataFrame:
    depth_charts = depth_charts.with_columns(
        data_pl.col("season").cast(data_pl.Int64),
        data_pl.col("week").cast(data_pl.Int64),
        data_pl.col("club_code").cast(data_pl.String),
        data_pl.col("depth_team").cast(data_pl.String),
        data_pl.col("position").cast(data_pl.String),
        data_pl.col("dt").cast(data_pl.String),
        data_pl.col("team").cast(data_pl.String),
        data_pl.col("pos_abb").cast(data_pl.String),
        data_pl.col("pos_rank").cast(data_pl.Int64),
    )
    legacy = depth_charts.filter(
        data_pl.col("dt").is_null()
        & (data_pl.col("position") == "QB")
        & (data_pl.col("depth_team").cast(data_pl.String) == "1")
    )
    legacy = data__unique_starters(
        legacy,
        keys=["season", "week", "club_code"],
        id_column="gsis_id",
        name_column="full_name",
    ).rename({"club_code": "team"})
    legacy = legacy.with_columns(
        data_pl.lit("weekly_depth_chart").alias("starter_source"),
        data_pl.lit(None, dtype=data_pl.Datetime(time_zone="UTC")).alias("starter_snapshot_utc"),
        data_pl.lit(False).alias("starter_cutoff_verified"),
    )
    current = depth_charts.filter(
        data_pl.col("dt").is_not_null() & (data_pl.col("pos_abb") == "QB") & (data_pl.col("pos_rank") == 1)
    ).with_columns(data_pl.col("dt").str.to_datetime(time_zone="UTC", strict=True).alias("starter_snapshot_utc"))
    current = data__unique_starters(
        current,
        keys=["starter_snapshot_utc", "team"],
        id_column="gsis_id",
        name_column="player_name",
    ).with_columns(
        data_pl.lit("timestamped_depth_chart").alias("starter_source"),
        data_pl.lit(True).alias("starter_cutoff_verified"),
    )
    legacy_rows = schedule_rows.filter(data_pl.col("season") < 2025).join(
        legacy, on=["season", "week", "team"], how="left", validate="m:1"
    )
    current_rows = (
        schedule_rows.filter(data_pl.col("season") >= 2025)
        .sort("prediction_cutoff_utc")
        .join_asof(
            current.sort("starter_snapshot_utc"),
            left_on="prediction_cutoff_utc",
            right_on="starter_snapshot_utc",
            by="team",
            strategy="backward",
            check_sortedness=False,
        )
    )
    return data_pl.concat([legacy_rows, current_rows], how="diagonal_relaxed").sort(["kickoff_utc", "game_id", "team"])


def data_actual_starters_from_participation(
    participation: data_pl.DataFrame,
) -> data_pl.DataFrame:
    players = data_pl.col("offense_players").str.split(";")
    positions = data_pl.col("offense_positions").str.split(";")
    qb_appearances = (
        participation.filter(
            data_pl.col("nflverse_game_id").is_not_null()
            & data_pl.col("possession_team").is_not_null()
            & data_pl.col("play_id").is_not_null()
            & data_pl.col("offense_players").is_not_null()
            & data_pl.col("offense_positions").is_not_null()
        )
        .select(
            data_pl.col("nflverse_game_id").alias("game_id"),
            data_pl.col("possession_team").alias("team"),
            "play_id",
            players.alias("offense_player_ids"),
            positions.alias("offense_player_positions"),
        )
        .filter(data_pl.col("offense_player_ids").list.len() == data_pl.col("offense_player_positions").list.len())
        .explode(["offense_player_ids", "offense_player_positions"], empty_as_null=True)
        .filter(data_pl.col("offense_player_positions") == "QB")
        .select(
            "game_id",
            "team",
            "play_id",
            data_pl.col("offense_player_ids").alias("actual_qb_id"),
        )
        .unique()
    )
    first_qb_play = qb_appearances.group_by(["game_id", "team"]).agg(
        data_pl.col("play_id").min().alias("actual_starter_play_id")
    )
    return (
        qb_appearances.join(
            first_qb_play,
            left_on=["game_id", "team", "play_id"],
            right_on=["game_id", "team", "actual_starter_play_id"],
            how="inner",
        )
        .group_by(["game_id", "team"])
        .agg(
            data_pl.col("play_id").first().alias("actual_starter_play_id"),
            data_pl.col("actual_qb_id").n_unique().alias("actual_starter_candidate_count"),
            data_pl.col("actual_qb_id").first(),
        )
        .with_columns(
            data_pl.when(data_pl.col("actual_starter_candidate_count") == 1)
            .then(data_pl.col("actual_qb_id"))
            .otherwise(None)
            .alias("actual_qb_id"),
            data_pl.lit("first_offensive_qb_participation").alias("actual_starter_source"),
        )
    )


def data_actual_starters_from_first_passer(
    pbp_starter_plays: data_pl.DataFrame,
) -> data_pl.DataFrame:
    return (
        pbp_starter_plays.filter(
            (data_pl.col("season_type") == "REG")
            & data_pl.col("game_id").is_not_null()
            & data_pl.col("posteam").is_not_null()
            & data_pl.col("play_id").is_not_null()
            & data_pl.col("passer_player_id").is_not_null()
        )
        .sort(["game_id", "posteam", "play_id"])
        .group_by(["game_id", "posteam"], maintain_order=True)
        .agg(
            data_pl.col("play_id").first().alias("actual_starter_play_id"),
            data_pl.col("passer_player_id").first().alias("actual_qb_id"),
        )
        .rename({"posteam": "team"})
        .with_columns(
            data_pl.when(
                (data_pl.col("game_id").str.slice(0, 4).cast(data_pl.Int32) < 2020) & (data_pl.col("team") == "LV")
            )
            .then(data_pl.lit("OAK"))
            .otherwise(data_pl.col("team"))
            .alias("team"),
            data_pl.lit(1, dtype=data_pl.UInt32).alias("actual_starter_candidate_count"),
            data_pl.lit("first_passer_fallback").alias("actual_starter_source"),
        )
    )


def data_attach_postgame_outcomes(
    rows: data_pl.DataFrame,
    participation: data_pl.DataFrame,
    pbp_starter_plays: data_pl.DataFrame,
    player_stats: data_pl.DataFrame,
) -> data_pl.DataFrame:
    participation_starters = data_actual_starters_from_participation(participation).filter(
        data_pl.col("actual_qb_id").is_not_null()
    )
    fallback_starters = data_actual_starters_from_first_passer(pbp_starter_plays)
    actual_starters = data_pl.concat([participation_starters, fallback_starters], how="diagonal_relaxed").unique(
        ["game_id", "team"], keep="first", maintain_order=True
    )
    recorded_passers = (
        pbp_starter_plays.filter(data_pl.col("passer_player_id").is_not_null())
        .select("game_id", data_pl.col("passer_player_id").alias("actual_qb_id"))
        .unique()
        .with_columns(data_pl.lit(True).alias("actual_recorded_pass"))
    )
    qb_outcomes = (
        player_stats.filter(data_pl.col("season_type") == "REG")
        .select(
            "game_id",
            data_pl.col("player_id").alias("actual_qb_id"),
            data_pl.col("player_display_name").alias("actual_qb_name"),
            data_pl.col("passing_yards").cast(data_pl.Float64).alias("official_passing_yards"),
        )
        .unique(["game_id", "actual_qb_id"])
    )
    return (
        rows.join(actual_starters, on=["game_id", "team"], how="left", validate="1:1")
        .join(qb_outcomes, on=["game_id", "actual_qb_id"], how="left", validate="m:1")
        .join(recorded_passers, on=["game_id", "actual_qb_id"], how="left", validate="m:1")
        .with_columns(data_pl.col("actual_recorded_pass").fill_null(False))
        .with_columns(
            (
                data_pl.col("actual_qb_id").is_not_null()
                & data_pl.col("official_passing_yards").is_null()
                & ~data_pl.col("actual_recorded_pass")
            ).alias("target_zero_no_pass_attempt")
        )
        .with_columns(
            data_pl.when(data_pl.col("target_zero_no_pass_attempt"))
            .then(data_pl.lit(0.0))
            .otherwise(data_pl.col("official_passing_yards"))
            .alias("official_passing_yards")
        )
        .with_columns(
            (
                data_pl.col("anticipated_qb_id").is_not_null()
                & data_pl.col("actual_qb_id").is_not_null()
                & (data_pl.col("anticipated_qb_id") == data_pl.col("actual_qb_id"))
            ).alias("starter_matches_actual")
        )
        .with_columns(
            (data_pl.col("starter_matches_actual") & data_pl.col("official_passing_yards").is_not_null()).alias(
                "evaluation_eligible"
            )
        )
    )


def data_build_base_qb_game_table(
    schedules: data_pl.DataFrame,
    depth_charts: data_pl.DataFrame,
    participation: data_pl.DataFrame,
    pbp_starter_plays: data_pl.DataFrame,
    player_stats: data_pl.DataFrame,
    config: data_DataConfig,
) -> data_pl.DataFrame:
    schedule_rows = data_normalize_schedules(
        schedules=schedules,
        season_type=config.season_type,
        first_eligible_season=config.first_eligible_season,
        test_season=config.test_season,
        cutoff_minutes=config.prediction_cutoff_minutes,
    )
    rows = data_select_anticipated_starters(schedule_rows, depth_charts)
    rows = data_attach_postgame_outcomes(rows, participation, pbp_starter_plays, player_stats)
    return rows.with_columns(
        data_pl.when(data_pl.col("anticipated_qb_id").is_null())
        .then(data_pl.lit("missing_or_ambiguous_qb1"))
        .otherwise(data_pl.lit("eligible_starter_row"))
        .alias("row_status")
    )


import polars as defense_features_pl

defense_features__ROLLING_COMPONENTS = (
    "defense_dropbacks",
    "defense_passing_yards_allowed",
    "defense_epa_value",
    "defense_epa_dropbacks",
    "defense_sacks",
    "defense_qb_hits_or_sacks",
    "defense_explosive_completed_passes",
)


def defense_features__safe_ratio(numerator: str, denominator: str, alias: str) -> defense_features_pl.Expr:
    return (
        defense_features_pl.when(defense_features_pl.col(denominator) > 0)
        .then(
            defense_features_pl.col(numerator).cast(defense_features_pl.Float64) / defense_features_pl.col(denominator)
        )
        .otherwise(None)
        .alias(alias)
    )


def defense_features__franchise_team(column: str, alias: str) -> defense_features_pl.Expr:
    return (
        defense_features_pl.when(defense_features_pl.col(column).is_in(["OAK", "LV"]))
        .then(defense_features_pl.lit("LV"))
        .otherwise(defense_features_pl.col(column))
        .alias(alias)
    )


def defense_features__validated_windows(windows: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple((int(window) for window in windows))
    if not normalized or any((window <= 0 for window in normalized)) or len(normalized) != len(set(normalized)):
        raise ValueError(f"Rolling windows must be unique positive integers, got {windows}")
    return normalized


def defense_features_defense_feature_columns(windows: tuple[int, ...]) -> list[str]:
    """Return the numeric, model-facing defense feature contract."""
    windows = defense_features__validated_windows(windows)
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


def defense_features__required_columns(frame: defense_features_pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def defense_features__passing_yards_expression(
    pbp_rows: defense_features_pl.DataFrame,
) -> defense_features_pl.Expr:
    if "passing_yards" in pbp_rows.columns:
        return defense_features_pl.col("passing_yards").cast(defense_features_pl.Float64).fill_null(0.0)
    if "yards_gained" in pbp_rows.columns:
        return defense_features_pl.col("yards_gained").cast(defense_features_pl.Float64).fill_null(0.0)
    raise ValueError("Play-by-play rows need passing_yards or yards_gained")


def defense_features__aggregate_defense_team_games(
    pbp_rows: defense_features_pl.DataFrame,
    schedules: defense_features_pl.DataFrame,
    season_type: str,
) -> defense_features_pl.DataFrame:
    defense_features__required_columns(schedules, {"game_id", "season", "week", "game_type"}, "Schedules")
    defense_features__required_columns(
        pbp_rows,
        {"game_id", "defteam", "qb_dropback", "sack", "qb_hit", "complete_pass", "epa"},
        "Play-by-play rows",
    )
    schedule_games = (
        schedules.filter(defense_features_pl.col("game_type") == season_type)
        .select("game_id", "season", "week")
        .unique("game_id")
    )
    if "season_type" in pbp_rows.columns:
        pbp_rows = pbp_rows.filter(defense_features_pl.col("season_type") == season_type)
    is_dropback = defense_features_pl.col("qb_dropback").fill_null(0.0) == 1.0
    is_sack = defense_features_pl.col("sack").fill_null(0.0) == 1.0
    is_qb_hit = defense_features_pl.col("qb_hit").fill_null(0.0) == 1.0
    is_complete = defense_features_pl.col("complete_pass").fill_null(0.0) == 1.0
    passing_yards = defense_features__passing_yards_expression(pbp_rows)
    valid_play = (
        is_dropback
        & defense_features_pl.col("game_id").is_not_null()
        & defense_features_pl.col("defteam").is_not_null()
    )
    if "play_type" in pbp_rows.columns:
        valid_play &= defense_features_pl.col("play_type").fill_null("") != "no_play"
    if "qb_kneel" in pbp_rows.columns:
        valid_play &= defense_features_pl.col("qb_kneel").fill_null(0.0) != 1.0
    if "qb_spike" in pbp_rows.columns:
        valid_play &= defense_features_pl.col("qb_spike").fill_null(0.0) != 1.0
    return (
        pbp_rows.filter(valid_play)
        .select(
            "game_id",
            defense_features__franchise_team("defteam", "history_defense_team"),
            defense_features_pl.lit(1.0).alias("defense_dropbacks"),
            defense_features_pl.when(~is_sack)
            .then(passing_yards)
            .otherwise(0.0)
            .alias("defense_passing_yards_allowed"),
            defense_features_pl.when(defense_features_pl.col("epa").is_not_null())
            .then(defense_features_pl.col("epa"))
            .otherwise(0.0)
            .alias("defense_epa_value"),
            defense_features_pl.when(defense_features_pl.col("epa").is_not_null())
            .then(1.0)
            .otherwise(0.0)
            .alias("defense_epa_dropbacks"),
            defense_features_pl.when(is_sack).then(1.0).otherwise(0.0).alias("defense_sacks"),
            defense_features_pl.when(is_qb_hit | is_sack).then(1.0).otherwise(0.0).alias("defense_qb_hits_or_sacks"),
            defense_features_pl.when(is_complete & ~is_sack & (passing_yards >= 20.0))
            .then(1.0)
            .otherwise(0.0)
            .alias("defense_explosive_completed_passes"),
        )
        .group_by(["game_id", "history_defense_team"])
        .agg(*[defense_features_pl.col(component).sum() for component in defense_features__ROLLING_COMPONENTS])
        .join(schedule_games, on="game_id", how="inner", validate="m:1")
        .with_columns(
            defense_features_pl.lit(1.0).alias("history_game_count"),
            (
                defense_features_pl.col("season").cast(defense_features_pl.Int64) * 100
                + defense_features_pl.col("week")
            ).alias("history_week_order"),
        )
        .sort(["history_defense_team", "history_week_order", "game_id"])
    )


def defense_features__defense_history(
    pbp_rows: defense_features_pl.DataFrame,
    schedules: defense_features_pl.DataFrame,
    season_type: str,
    windows: tuple[int, ...],
) -> defense_features_pl.DataFrame:
    history = defense_features__aggregate_defense_team_games(pbp_rows, schedules, season_type)
    for window in windows:
        history = history.with_columns(
            *[
                defense_features_pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("history_defense_team")
                .alias(f"{component}_last{window}_sum")
                for component in defense_features__ROLLING_COMPONENTS
            ],
            defense_features_pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("history_defense_team")
            .alias(f"history_games_last{window}"),
        )
    return history.with_columns(
        *[
            defense_features_pl.col(component)
            .cum_sum()
            .over(["history_defense_team", "season"])
            .alias(f"{component}_season_sum")
            for component in defense_features__ROLLING_COMPONENTS
        ],
        defense_features_pl.col("history_game_count")
        .cum_sum()
        .over(["history_defense_team", "season"])
        .alias("history_games_season"),
    )


def defense_features__period_feature_expressions(
    period: str,
) -> list[defense_features_pl.Expr]:
    count = f"history_games_{period}"
    return [
        defense_features_pl.col(count).alias(f"defense_history_games_{period}"),
        defense_features__safe_ratio(
            f"defense_dropbacks_{period}_sum",
            count,
            f"defense_dropbacks_faced_per_game_{period}",
        ),
        defense_features__safe_ratio(
            f"defense_passing_yards_allowed_{period}_sum",
            count,
            f"defense_passing_yards_allowed_per_game_{period}",
        ),
        defense_features__safe_ratio(
            f"defense_epa_value_{period}_sum",
            f"defense_epa_dropbacks_{period}_sum",
            f"defense_epa_per_dropback_allowed_{period}",
        ),
        defense_features__safe_ratio(
            f"defense_sacks_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_sack_rate_{period}",
        ),
        defense_features__safe_ratio(
            f"defense_qb_hits_or_sacks_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_qb_hit_or_sack_rate_{period}",
        ),
        defense_features__safe_ratio(
            f"defense_explosive_completed_passes_{period}_sum",
            f"defense_dropbacks_{period}_sum",
            f"defense_explosive_completed_pass_rate_{period}",
        ),
    ]


def defense_features__feature_expressions(
    windows: tuple[int, ...],
) -> list[defense_features_pl.Expr]:
    expressions: list[defense_features_pl.Expr] = []
    for window in windows:
        expressions.extend(defense_features__period_feature_expressions(f"last{window}"))
    expressions.extend(defense_features__period_feature_expressions("season"))
    return expressions


def defense_features_build_defense_features(
    base_rows: defense_features_pl.DataFrame,
    pbp_rows: defense_features_pl.DataFrame,
    schedules: defense_features_pl.DataFrame,
    season_type: str,
    windows: tuple[int, ...],
) -> defense_features_pl.DataFrame:
    """Attach opponent-defense features using games strictly before each row's week."""
    windows = defense_features__validated_windows(windows)
    defense_features__required_columns(base_rows, {"game_id", "season", "week", "opponent_team"}, "Base rows")
    feature_columns = defense_features_defense_feature_columns(windows)
    overlapping = set(feature_columns).intersection(base_rows.columns)
    if overlapping:
        raise ValueError(f"Base rows already contain defense features: {sorted(overlapping)}")
    history = defense_features__defense_history(pbp_rows, schedules, season_type, windows).with_columns(
        *defense_features__feature_expressions(windows)
    )
    history_features = history.select(
        "history_defense_team",
        "history_week_order",
        defense_features_pl.col("season").alias("defense_history_season"),
        *feature_columns,
    )
    rows = (
        base_rows.with_row_index("_defense_base_order")
        .with_columns(
            defense_features__franchise_team("opponent_team", "defense_feature_team"),
            (
                defense_features_pl.col("season").cast(defense_features_pl.Int64) * 100
                + defense_features_pl.col("week")
            ).alias("prediction_week_order"),
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
                defense_features_pl.when(
                    defense_features_pl.col("defense_history_season") == defense_features_pl.col("season")
                )
                .then(defense_features_pl.col(column))
                .otherwise(None)
                .alias(column)
                for column in season_value_columns
            ],
            defense_features_pl.when(
                defense_features_pl.col("defense_history_season") == defense_features_pl.col("season")
            )
            .then(defense_features_pl.col("defense_history_games_season"))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("defense_history_games_season"),
            *[defense_features_pl.col(column).fill_null(0.0).alias(column) for column in count_columns],
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


derived_features__TREND_BASES = (
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

derived_features__DISPERSION_BASES = (
    "qb_attempts_avg",
    "qb_passing_yards_avg",
    "qb_epa_per_dropback",
    "offense_epa_per_play",
    "defense_passing_yards_allowed_per_game",
    "defense_epa_per_dropback_allowed",
)

derived_features_TREND_FEATURE_COLUMNS = tuple(
    [f"trend_{base}_last3_minus_last8" for base in derived_features__TREND_BASES]
    + [f"dispersion_{base}_last3_vs_last8" for base in derived_features__DISPERSION_BASES]
)

derived_features_MATCHUP_FEATURE_COLUMNS = (
    "matchup_expected_attempts_last8",
    "matchup_expected_passing_yards_last8",
    "matchup_expected_dropbacks_last8",
    "matchup_epa_blend_last8",
    "matchup_sack_pressure_last8",
    "matchup_air_explosive_last8",
    "matchup_wind_air_penalty_last8",
    "matchup_rest_pace_last8",
)

import polars as features_pl

features__ROLLING_COMPONENTS = (
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


def features__safe_ratio(numerator: str, denominator: str, alias: str) -> features_pl.Expr:
    return (
        features_pl.when(features_pl.col(denominator) > 0)
        .then(features_pl.col(numerator).cast(features_pl.Float64) / features_pl.col(denominator))
        .otherwise(None)
        .alias(alias)
    )


def features__game_kickoffs(schedules: features_pl.DataFrame, config: data_DataConfig) -> features_pl.DataFrame:
    return (
        schedules.filter(
            (features_pl.col("game_type") == config.season_type)
            & features_pl.col("season").is_between(config.warmup_start_season, config.test_season)
        )
        .select(
            "game_id",
            features_pl.concat_str(["gameday", "gametime"], separator=" ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="America/New_York", strict=True)
            .dt.convert_time_zone("UTC")
            .alias("history_game_kickoff_utc"),
        )
        .unique("game_id")
    )


def features__qb_history(
    player_stats: features_pl.DataFrame,
    schedules: features_pl.DataFrame,
    config: data_DataConfig,
) -> features_pl.DataFrame:
    history = (
        player_stats.filter(
            (features_pl.col("season_type") == config.season_type)
            & (features_pl.col("position") == "QB")
            & features_pl.col("player_id").is_not_null()
        )
        .select(
            "player_id",
            "game_id",
            "season",
            "week",
            features_pl.col("attempts").cast(features_pl.Float64),
            features_pl.col("passing_yards").cast(features_pl.Float64),
            features_pl.col("passing_air_yards").cast(features_pl.Float64),
            features_pl.col("sacks_suffered").cast(features_pl.Float64),
            features_pl.col("passing_epa").cast(features_pl.Float64),
            features_pl.col("passing_cpoe").cast(features_pl.Float64),
        )
        .join(
            features__game_kickoffs(schedules, config),
            on="game_id",
            how="inner",
            validate="m:1",
        )
        .with_columns(
            (features_pl.col("attempts") + features_pl.col("sacks_suffered")).alias("dropbacks"),
            features_pl.lit(1.0).alias("history_game_count"),
            (features_pl.col("season").cast(features_pl.Int64) * 100 + features_pl.col("week")).alias(
                "history_week_order"
            ),
        )
        .filter(features_pl.col("dropbacks") > 0)
        .with_columns(
            features_pl.when(features_pl.col("passing_epa").is_not_null())
            .then(features_pl.col("passing_epa"))
            .otherwise(0.0)
            .alias("passing_epa_value"),
            features_pl.when(features_pl.col("passing_epa").is_not_null())
            .then(features_pl.col("dropbacks"))
            .otherwise(0.0)
            .alias("passing_epa_dropbacks"),
            features_pl.when(features_pl.col("passing_cpoe").is_not_null())
            .then(features_pl.col("passing_cpoe") * features_pl.col("attempts"))
            .otherwise(0.0)
            .alias("passing_cpoe_weighted"),
            features_pl.when(features_pl.col("passing_cpoe").is_not_null())
            .then(features_pl.col("attempts"))
            .otherwise(0.0)
            .alias("passing_cpoe_attempts"),
        )
        .unique(["player_id", "game_id"], keep="first", maintain_order=True)
        .sort(["player_id", "history_week_order", "history_game_kickoff_utc"])
    )
    for window in config.qb_rolling_windows:
        history = history.with_columns(
            *[
                features_pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("player_id")
                .alias(f"{component}_last{window}_sum")
                for component in features__ROLLING_COMPONENTS
            ],
            features_pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("player_id")
            .alias(f"history_games_last{window}"),
        )
    history = history.with_columns(
        *[
            features_pl.col(component).cum_sum().over(["player_id", "season"]).alias(f"{component}_season_sum")
            for component in features__ROLLING_COMPONENTS
        ],
        features_pl.col("history_game_count").cum_sum().over(["player_id", "season"]).alias("history_games_season"),
    )
    return history


def features_qb_feature_columns(windows: tuple[int, ...]) -> list[str]:
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


def features__feature_expressions(windows: tuple[int, ...]) -> list[features_pl.Expr]:
    expressions = [
        features_pl.col("attempts").alias("qb_attempts_lag1"),
        features_pl.col("passing_yards").alias("qb_passing_yards_lag1"),
        features__safe_ratio("passing_yards", "attempts", "qb_ypa_lag1"),
        features__safe_ratio("passing_epa_value", "passing_epa_dropbacks", "qb_epa_per_dropback_lag1"),
        features__safe_ratio("passing_cpoe_weighted", "passing_cpoe_attempts", "qb_cpoe_lag1"),
        features__safe_ratio("passing_air_yards", "attempts", "qb_air_yards_per_attempt_lag1"),
        features__safe_ratio("sacks_suffered", "dropbacks", "qb_sack_rate_lag1"),
    ]
    for window in windows:
        suffix = f"last{window}"
        count = f"history_games_{suffix}"
        expressions.extend(
            [
                features_pl.col(count).alias(f"qb_history_games_{suffix}"),
                features__safe_ratio(f"attempts_{suffix}_sum", count, f"qb_attempts_avg_{suffix}"),
                features__safe_ratio(
                    f"passing_yards_{suffix}_sum",
                    count,
                    f"qb_passing_yards_avg_{suffix}",
                ),
                features__safe_ratio(
                    f"passing_yards_{suffix}_sum",
                    f"attempts_{suffix}_sum",
                    f"qb_ypa_{suffix}",
                ),
                features__safe_ratio(
                    f"passing_epa_value_{suffix}_sum",
                    f"passing_epa_dropbacks_{suffix}_sum",
                    f"qb_epa_per_dropback_{suffix}",
                ),
                features__safe_ratio(
                    f"passing_cpoe_weighted_{suffix}_sum",
                    f"passing_cpoe_attempts_{suffix}_sum",
                    f"qb_cpoe_{suffix}",
                ),
                features__safe_ratio(
                    f"passing_air_yards_{suffix}_sum",
                    f"attempts_{suffix}_sum",
                    f"qb_air_yards_per_attempt_{suffix}",
                ),
                features__safe_ratio(
                    f"sacks_suffered_{suffix}_sum",
                    f"dropbacks_{suffix}_sum",
                    f"qb_sack_rate_{suffix}",
                ),
            ]
        )
    expressions.extend(
        [
            features_pl.col("history_games_season").alias("qb_history_games_season"),
            features__safe_ratio("attempts_season_sum", "history_games_season", "qb_attempts_avg_season"),
            features__safe_ratio(
                "passing_yards_season_sum",
                "history_games_season",
                "qb_passing_yards_avg_season",
            ),
            features__safe_ratio("passing_yards_season_sum", "attempts_season_sum", "qb_ypa_season"),
            features__safe_ratio(
                "passing_epa_value_season_sum",
                "passing_epa_dropbacks_season_sum",
                "qb_epa_per_dropback_season",
            ),
            features__safe_ratio(
                "passing_cpoe_weighted_season_sum",
                "passing_cpoe_attempts_season_sum",
                "qb_cpoe_season",
            ),
            features__safe_ratio(
                "passing_air_yards_season_sum",
                "attempts_season_sum",
                "qb_air_yards_per_attempt_season",
            ),
            features__safe_ratio(
                "sacks_suffered_season_sum",
                "dropbacks_season_sum",
                "qb_sack_rate_season",
            ),
        ]
    )
    return expressions


def features_build_qb_rolling_features(
    base_rows: features_pl.DataFrame,
    player_stats: features_pl.DataFrame,
    schedules: features_pl.DataFrame,
    config: data_DataConfig,
    *,
    qb_id_column: str = "anticipated_qb_id",
) -> features_pl.DataFrame:
    history = features__qb_history(player_stats, schedules, config).with_columns(
        *features__feature_expressions(config.qb_rolling_windows)
    )
    feature_columns = features_qb_feature_columns(config.qb_rolling_windows)
    history_features = history.select(
        features_pl.col("player_id").alias("history_qb_id"),
        "history_week_order",
        features_pl.col("season").alias("qb_history_season"),
        features_pl.col("game_id").alias("qb_previous_game_id"),
        "history_game_kickoff_utc",
        *[column for column in feature_columns if column != "qb_days_since_previous_game"],
    )
    rows = (
        base_rows.with_columns(
            (features_pl.col("season").cast(features_pl.Int64) * 100 + features_pl.col("week")).alias(
                "prediction_week_order"
            )
        )
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
            features_pl.when(features_pl.col("qb_history_season") == features_pl.col("season"))
            .then(features_pl.col(column))
            .otherwise(None)
            .alias(column)
            for column in season_value_columns
        ],
        features_pl.when(features_pl.col("qb_history_season") == features_pl.col("season"))
        .then(features_pl.col("qb_history_games_season"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("qb_history_games_season"),
        (features_pl.col("kickoff_utc") - features_pl.col("history_game_kickoff_utc"))
        .dt.total_days()
        .cast(features_pl.Float64)
        .alias("qb_days_since_previous_game"),
    )
    count_columns = [column for column in feature_columns if column.startswith("qb_history_games_last")]
    return (
        rows.with_columns(*[features_pl.col(column).fill_null(0.0).alias(column) for column in count_columns])
        .drop("prediction_week_order", "history_week_order", "history_qb_id", strict=False)
        .sort(["kickoff_utc", "game_id", "team"])
    )


import polars as offense_features_pl

offense_features__ROLLING_COMPONENTS = (
    "offense_plays",
    "offense_epa_value",
    "offense_epa_plays",
    "neutral_dropbacks",
    "neutral_plays",
)


def offense_features__safe_ratio(numerator: str, denominator: str, alias: str) -> offense_features_pl.Expr:
    return (
        offense_features_pl.when(offense_features_pl.col(denominator) > 0)
        .then(
            offense_features_pl.col(numerator).cast(offense_features_pl.Float64) / offense_features_pl.col(denominator)
        )
        .otherwise(None)
        .alias(alias)
    )


def offense_features__franchise_team(column: str, alias: str) -> offense_features_pl.Expr:
    return (
        offense_features_pl.when(offense_features_pl.col(column).is_in(["OAK", "LV"]))
        .then(offense_features_pl.lit("LV"))
        .otherwise(offense_features_pl.col(column))
        .alias(alias)
    )


def offense_features__game_kickoffs(
    schedules: offense_features_pl.DataFrame, config: data_DataConfig
) -> offense_features_pl.DataFrame:
    return (
        schedules.filter(
            (offense_features_pl.col("game_type") == config.season_type)
            & offense_features_pl.col("season").is_between(config.warmup_start_season, config.test_season)
        )
        .select(
            "game_id",
            offense_features_pl.concat_str(["gameday", "gametime"], separator=" ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="America/New_York", strict=True)
            .dt.convert_time_zone("UTC")
            .alias("history_game_kickoff_utc"),
        )
        .unique("game_id")
    )


def offense_features_aggregate_offense_team_games(
    pbp: offense_features_pl.DataFrame,
    schedules: offense_features_pl.DataFrame,
    config: data_DataConfig,
) -> offense_features_pl.DataFrame:
    is_dropback = offense_features_pl.col("qb_dropback").fill_null(0.0) == 1.0
    is_rush = offense_features_pl.col("rush_attempt").fill_null(0.0) == 1.0
    is_neutral = (offense_features_pl.col("qtr") <= config.neutral_max_quarter) & offense_features_pl.col(
        "wp"
    ).is_between(config.neutral_wp_lower, config.neutral_wp_upper, closed="both")
    return (
        pbp.filter(
            (offense_features_pl.col("season_type") == config.season_type)
            & offense_features_pl.col("season").is_between(config.warmup_start_season, config.test_season)
            & offense_features_pl.col("game_id").is_not_null()
            & offense_features_pl.col("posteam").is_not_null()
            & offense_features_pl.col("play_id").is_not_null()
            & offense_features_pl.col("play_type").is_in(["pass", "run"])
            & (offense_features_pl.col("qb_kneel").fill_null(0.0) != 1.0)
            & (offense_features_pl.col("qb_spike").fill_null(0.0) != 1.0)
            & (is_dropback | is_rush)
        )
        .with_columns(
            offense_features__franchise_team("posteam", "history_team"),
            offense_features_pl.lit(1.0).alias("offense_plays"),
            offense_features_pl.when(offense_features_pl.col("epa").is_not_null())
            .then(offense_features_pl.col("epa"))
            .otherwise(0.0)
            .alias("offense_epa_value"),
            offense_features_pl.when(offense_features_pl.col("epa").is_not_null())
            .then(1.0)
            .otherwise(0.0)
            .alias("offense_epa_plays"),
            offense_features_pl.when(is_neutral & is_dropback).then(1.0).otherwise(0.0).alias("neutral_dropbacks"),
            offense_features_pl.when(is_neutral).then(1.0).otherwise(0.0).alias("neutral_plays"),
        )
        .group_by(["game_id", "season", "week", "history_team"])
        .agg(*[offense_features_pl.col(component).sum() for component in offense_features__ROLLING_COMPONENTS])
        .join(
            offense_features__game_kickoffs(schedules, config),
            on="game_id",
            how="inner",
            validate="m:1",
        )
        .with_columns(
            offense_features_pl.lit(1.0).alias("history_game_count"),
            (
                offense_features_pl.col("season").cast(offense_features_pl.Int64) * 100
                + offense_features_pl.col("week")
            ).alias("history_week_order"),
        )
        .sort(["history_team", "history_week_order", "history_game_kickoff_utc"])
    )


def offense_features__offense_history(
    pbp: offense_features_pl.DataFrame,
    schedules: offense_features_pl.DataFrame,
    config: data_DataConfig,
) -> offense_features_pl.DataFrame:
    history = offense_features_aggregate_offense_team_games(pbp, schedules, config)
    for window in config.offense_rolling_windows:
        history = history.with_columns(
            *[
                offense_features_pl.col(component)
                .rolling_sum(window_size=window, min_samples=1)
                .over("history_team")
                .alias(f"{component}_last{window}_sum")
                for component in offense_features__ROLLING_COMPONENTS
            ],
            offense_features_pl.col("history_game_count")
            .rolling_sum(window_size=window, min_samples=1)
            .over("history_team")
            .alias(f"history_games_last{window}"),
        )
    return history.with_columns(
        *[
            offense_features_pl.col(component)
            .cum_sum()
            .over(["history_team", "season"])
            .alias(f"{component}_season_sum")
            for component in offense_features__ROLLING_COMPONENTS
        ],
        offense_features_pl.col("history_game_count")
        .cum_sum()
        .over(["history_team", "season"])
        .alias("history_games_season"),
    )


def offense_features_offense_feature_columns(windows: tuple[int, ...]) -> list[str]:
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


def offense_features__feature_expressions(
    windows: tuple[int, ...],
) -> list[offense_features_pl.Expr]:
    expressions = [
        offense_features_pl.col("offense_plays").alias("offense_plays_lag1"),
        offense_features__safe_ratio("offense_epa_value", "offense_epa_plays", "offense_epa_per_play_lag1"),
        offense_features__safe_ratio("neutral_dropbacks", "neutral_plays", "offense_neutral_pass_rate_lag1"),
    ]
    for window in windows:
        suffix = f"last{window}"
        count = f"history_games_{suffix}"
        expressions.extend(
            [
                offense_features_pl.col(count).alias(f"offense_history_games_{suffix}"),
                offense_features__safe_ratio(
                    f"offense_plays_{suffix}_sum",
                    count,
                    f"offense_plays_per_game_{suffix}",
                ),
                offense_features__safe_ratio(
                    f"offense_epa_value_{suffix}_sum",
                    f"offense_epa_plays_{suffix}_sum",
                    f"offense_epa_per_play_{suffix}",
                ),
                offense_features__safe_ratio(
                    f"neutral_dropbacks_{suffix}_sum",
                    f"neutral_plays_{suffix}_sum",
                    f"offense_neutral_pass_rate_{suffix}",
                ),
            ]
        )
    expressions.extend(
        [
            offense_features_pl.col("history_games_season").alias("offense_history_games_season"),
            offense_features__safe_ratio(
                "offense_plays_season_sum",
                "history_games_season",
                "offense_plays_per_game_season",
            ),
            offense_features__safe_ratio(
                "offense_epa_value_season_sum",
                "offense_epa_plays_season_sum",
                "offense_epa_per_play_season",
            ),
            offense_features__safe_ratio(
                "neutral_dropbacks_season_sum",
                "neutral_plays_season_sum",
                "offense_neutral_pass_rate_season",
            ),
        ]
    )
    return expressions


def offense_features_build_offense_rolling_features(
    base_rows: offense_features_pl.DataFrame,
    pbp: offense_features_pl.DataFrame,
    schedules: offense_features_pl.DataFrame,
    config: data_DataConfig,
) -> offense_features_pl.DataFrame:
    history = offense_features__offense_history(pbp, schedules, config).with_columns(
        *offense_features__feature_expressions(config.offense_rolling_windows)
    )
    feature_columns = offense_features_offense_feature_columns(config.offense_rolling_windows)
    history_features = history.select(
        "history_team",
        "history_week_order",
        offense_features_pl.col("season").alias("offense_history_season"),
        offense_features_pl.col("game_id").alias("offense_previous_game_id"),
        *feature_columns,
    )
    rows = (
        base_rows.with_columns(
            offense_features__franchise_team("team", "feature_team"),
            (
                offense_features_pl.col("season").cast(offense_features_pl.Int64) * 100
                + offense_features_pl.col("week")
            ).alias("prediction_week_order"),
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
            offense_features_pl.when(
                offense_features_pl.col("offense_history_season") == offense_features_pl.col("season")
            )
            .then(offense_features_pl.col(column))
            .otherwise(None)
            .alias(column)
            for column in season_value_columns
        ],
        offense_features_pl.when(offense_features_pl.col("offense_history_season") == offense_features_pl.col("season"))
        .then(offense_features_pl.col("offense_history_games_season"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("offense_history_games_season"),
    )
    count_columns = [column for column in feature_columns if column.startswith("offense_history_games_last")]
    return (
        rows.with_columns(*[offense_features_pl.col(column).fill_null(0.0).alias(column) for column in count_columns])
        .drop(
            "feature_team",
            "prediction_week_order",
            "history_team",
            "history_week_order",
            strict=False,
        )
        .sort(["kickoff_utc", "game_id", "team"])
    )


from collections.abc import Sequence as season_context_features_Sequence

from collections.abc import Mapping as season_context_features_Mapping

from typing import Any as season_context_features_Any

import polars as season_context_features_pl

season_context_features_SEASON_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "season_context_week_fraction",
    "season_context_team_games_played",
    "season_context_team_games_remaining",
    "season_context_team_win_pct",
    "season_context_opponent_win_pct",
    "season_context_win_pct_delta",
    "season_context_team_point_diff_per_game",
    "season_context_opponent_point_diff_per_game",
    "season_context_point_diff_per_game_delta",
    "season_context_team_conference_rank_fraction",
    "season_context_opponent_conference_rank_fraction",
    "season_context_conference_rank_fraction_delta",
    "season_context_team_division_rank_fraction",
    "season_context_opponent_division_rank_fraction",
    "season_context_division_rank_fraction_delta",
    "season_context_team_gap_to_conference_7th",
    "season_context_opponent_gap_to_conference_7th",
    "season_context_team_gap_to_division_leader",
    "season_context_opponent_gap_to_division_leader",
    "season_context_team_max_wins_over_conference_7th",
    "season_context_opponent_max_wins_over_conference_7th",
)

season_context_features__BASE_KEYS = ("game_id", "team")

season_context_features__SNAPSHOT_FIELDS = (
    "games_played",
    "games_remaining",
    "win_pct",
    "point_diff_per_game",
    "conference_rank_fraction",
    "division_rank_fraction",
    "gap_to_conference_7th",
    "gap_to_division_leader",
    "max_wins_over_conference_7th",
)


def season_context_features__require_columns(
    frame: season_context_features_pl.DataFrame,
    columns: season_context_features_Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def season_context_features__season_rules(season: int) -> tuple[int, int]:
    """Return regular-season weeks and scheduled games per team."""
    return (17, 16) if season <= 2020 else (18, 17)


def season_context_features__average_rank_fraction(
    values: season_context_features_Mapping[str, tuple[float, float]],
    members: season_context_features_Sequence[str],
) -> dict[str, float | None]:
    """Rank descending with average ranks for exact metric ties."""
    available = [team for team in members if team in values]
    if not available:
        return {team: None for team in members}
    ordered = sorted(available, key=lambda team: (-values[team][0], -values[team][1], team))
    denominator = max(len(ordered) - 1, 1)
    result: dict[str, float | None] = {team: None for team in members}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        fraction = (average_rank - 1.0) / denominator
        for team in ordered[index:end]:
            result[team] = float(fraction)
        index = end
    return result


def season_context_features__standings_snapshots(
    schedules: season_context_features_pl.DataFrame,
    teams: season_context_features_pl.DataFrame,
    seasons: season_context_features_Sequence[int],
) -> season_context_features_pl.DataFrame:
    season_context_features__require_columns(
        schedules,
        (
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ),
        label="schedules",
    )
    season_context_features__require_columns(teams, ("team_abbr", "team_conf", "team_division"), label="teams")
    metadata_rows = teams.select("team_abbr", "team_conf", "team_division").unique().to_dicts()
    if len(metadata_rows) != teams.get_column("team_abbr").n_unique():
        raise ValueError("teams metadata maps one abbreviation to multiple conference/division rows")
    metadata = {str(row["team_abbr"]): (str(row["team_conf"]), str(row["team_division"])) for row in metadata_rows}
    regular = schedules.filter(
        (season_context_features_pl.col("game_type") == "REG")
        & season_context_features_pl.col("season").is_in([int(season) for season in seasons])
    ).sort(["season", "week", "game_id"] if "game_id" in schedules.columns else ["season", "week"])
    output: list[dict[str, season_context_features_Any]] = []
    for season in sorted({int(value) for value in seasons}):
        season_games = regular.filter(season_context_features_pl.col("season") == season)
        scheduled_teams = sorted(
            set(season_games.get_column("home_team").drop_nulls().to_list())
            | set(season_games.get_column("away_team").drop_nulls().to_list())
        )
        if not scheduled_teams:
            raise ValueError(f"schedules contain no regular-season games for {season}")
        missing_metadata = sorted(set(scheduled_teams).difference(metadata))
        if missing_metadata:
            raise ValueError(f"teams metadata is missing season {season} codes: {missing_metadata}")
        regular_weeks, games_per_team = season_context_features__season_rules(season)
        state = {
            team: {
                "games": 0.0,
                "wins": 0.0,
                "ties": 0.0,
                "points_for": 0.0,
                "points_against": 0.0,
            }
            for team in scheduled_teams
        }
        games_by_week: dict[int, list[dict[str, season_context_features_Any]]] = {}
        for row in season_games.select("week", "home_team", "away_team", "home_score", "away_score").to_dicts():
            games_by_week.setdefault(int(row["week"]), []).append(row)
        for week in range(1, regular_weeks + 1):
            win_pct = {
                team: (values["wins"] + 0.5 * values["ties"]) / values["games"]
                for team, values in state.items()
                if values["games"] > 0
            }
            point_diff = {
                team: (values["points_for"] - values["points_against"]) / values["games"]
                for team, values in state.items()
                if values["games"] > 0
            }
            rank_values = {team: (win_pct[team], point_diff[team]) for team in win_pct}
            conference_ranks: dict[str, float | None] = {}
            division_ranks: dict[str, float | None] = {}
            conference_cutoffs: dict[str, tuple[float, float]] = {}
            division_leaders: dict[str, float] = {}
            for conference in sorted({metadata[team][0] for team in scheduled_teams}):
                members = [team for team in scheduled_teams if metadata[team][0] == conference]
                conference_ranks.update(season_context_features__average_rank_fraction(rank_values, members))
                ordered = sorted(
                    [team for team in members if team in rank_values],
                    key=lambda team: (
                        -rank_values[team][0],
                        -rank_values[team][1],
                        team,
                    ),
                )
                if ordered:
                    cutoff = ordered[min(6, len(ordered) - 1)]
                    conference_cutoffs[conference] = (
                        win_pct[cutoff],
                        state[cutoff]["wins"],
                    )
            for division in sorted({metadata[team][1] for team in scheduled_teams}):
                members = [team for team in scheduled_teams if metadata[team][1] == division]
                division_ranks.update(season_context_features__average_rank_fraction(rank_values, members))
                observed = [win_pct[team] for team in members if team in win_pct]
                if observed:
                    division_leaders[division] = max(observed)
            for team in scheduled_teams:
                conference, division = metadata[team]
                values = state[team]
                games_played = values["games"]
                games_remaining = float(max(0, games_per_team - int(games_played)))
                team_win_pct = win_pct.get(team)
                cutoff = conference_cutoffs.get(conference)
                output.append(
                    {
                        "season": season,
                        "week": week,
                        "standing_team": team,
                        "games_played": games_played,
                        "games_remaining": games_remaining,
                        "win_pct": team_win_pct,
                        "point_diff_per_game": point_diff.get(team),
                        "conference_rank_fraction": conference_ranks.get(team),
                        "division_rank_fraction": division_ranks.get(team),
                        "gap_to_conference_7th": None
                        if team_win_pct is None or cutoff is None
                        else team_win_pct - cutoff[0],
                        "gap_to_division_leader": None
                        if team_win_pct is None or division not in division_leaders
                        else team_win_pct - division_leaders[division],
                        "max_wins_over_conference_7th": None
                        if cutoff is None
                        else values["wins"] + games_remaining - cutoff[1],
                    }
                )
            for game in games_by_week.get(week, []):
                home_score = game["home_score"]
                away_score = game["away_score"]
                if home_score is None or away_score is None:
                    continue
                home = str(game["home_team"])
                away = str(game["away_team"])
                for team, scored, allowed in (
                    (home, home_score, away_score),
                    (away, away_score, home_score),
                ):
                    state[team]["games"] += 1.0
                    state[team]["points_for"] += float(scored)
                    state[team]["points_against"] += float(allowed)
                if float(home_score) > float(away_score):
                    state[home]["wins"] += 1.0
                elif float(away_score) > float(home_score):
                    state[away]["wins"] += 1.0
                else:
                    state[home]["ties"] += 1.0
                    state[away]["ties"] += 1.0
    return season_context_features_pl.DataFrame(output).with_columns(
        season_context_features_pl.col("season").cast(season_context_features_pl.Int64),
        season_context_features_pl.col("week").cast(season_context_features_pl.Int64),
        *[
            season_context_features_pl.col(column).cast(season_context_features_pl.Float64)
            for column in season_context_features__SNAPSHOT_FIELDS
        ],
    )


def season_context_features_build_season_context_features(
    base_rows: season_context_features_pl.DataFrame,
    schedules: season_context_features_pl.DataFrame,
    teams: season_context_features_pl.DataFrame,
) -> season_context_features_pl.DataFrame:
    """Append season-progress and prior-week standings features."""
    season_context_features__require_columns(
        base_rows,
        ("game_id", "season", "week", "team", "opponent_team"),
        label="base_rows",
    )
    if base_rows.select(
        season_context_features_pl.struct(season_context_features__BASE_KEYS).is_duplicated().any()
    ).item():
        raise ValueError("base_rows must have exactly one row per game_id/team")
    existing = sorted(set(season_context_features_SEASON_CONTEXT_FEATURE_COLUMNS).intersection(base_rows.columns))
    if existing:
        raise ValueError(f"season-context columns already exist: {existing}")
    seasons = [int(value) for value in base_rows.get_column("season").unique().to_list()]
    snapshots = season_context_features__standings_snapshots(schedules, teams, seasons)
    working = base_rows.with_row_index("_season_context_row_order").with_columns(
        season_context_features_pl.col("season").cast(season_context_features_pl.Int64).alias("_season_context_season"),
        season_context_features_pl.col("week").cast(season_context_features_pl.Int64).alias("_season_context_week"),
    )
    team_snapshot = snapshots.rename(
        {column: f"_season_context_team_{column}" for column in season_context_features__SNAPSHOT_FIELDS}
    ).rename({"standing_team": "_season_context_team"})
    opponent_snapshot = snapshots.rename(
        {column: f"_season_context_opponent_{column}" for column in season_context_features__SNAPSHOT_FIELDS}
    ).rename({"standing_team": "_season_context_opponent"})
    rows = working.join(
        team_snapshot,
        left_on=["_season_context_season", "_season_context_week", "team"],
        right_on=["season", "week", "_season_context_team"],
        how="left",
        validate="m:1",
        maintain_order="left",
    ).join(
        opponent_snapshot,
        left_on=["_season_context_season", "_season_context_week", "opponent_team"],
        right_on=["season", "week", "_season_context_opponent"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if (
        rows.get_column("_season_context_team_games_played").null_count()
        or rows.get_column("_season_context_opponent_games_played").null_count()
    ):
        raise ValueError("base rows contain season/week/team values absent from standings snapshots")
    regular_week_denominator = (
        season_context_features_pl.when(season_context_features_pl.col("_season_context_season") <= 2020)
        .then(season_context_features_pl.lit(16.0))
        .otherwise(season_context_features_pl.lit(17.0))
    )
    rows = rows.with_columns(
        (
            (season_context_features_pl.col("_season_context_week") - 1).cast(season_context_features_pl.Float64)
            / regular_week_denominator
        ).alias("season_context_week_fraction"),
        season_context_features_pl.col("_season_context_team_games_played").alias("season_context_team_games_played"),
        season_context_features_pl.col("_season_context_team_games_remaining").alias(
            "season_context_team_games_remaining"
        ),
        season_context_features_pl.col("_season_context_team_win_pct").alias("season_context_team_win_pct"),
        season_context_features_pl.col("_season_context_opponent_win_pct").alias("season_context_opponent_win_pct"),
        (
            season_context_features_pl.col("_season_context_team_win_pct")
            - season_context_features_pl.col("_season_context_opponent_win_pct")
        ).alias("season_context_win_pct_delta"),
        season_context_features_pl.col("_season_context_team_point_diff_per_game").alias(
            "season_context_team_point_diff_per_game"
        ),
        season_context_features_pl.col("_season_context_opponent_point_diff_per_game").alias(
            "season_context_opponent_point_diff_per_game"
        ),
        (
            season_context_features_pl.col("_season_context_team_point_diff_per_game")
            - season_context_features_pl.col("_season_context_opponent_point_diff_per_game")
        ).alias("season_context_point_diff_per_game_delta"),
        season_context_features_pl.col("_season_context_team_conference_rank_fraction").alias(
            "season_context_team_conference_rank_fraction"
        ),
        season_context_features_pl.col("_season_context_opponent_conference_rank_fraction").alias(
            "season_context_opponent_conference_rank_fraction"
        ),
        (
            season_context_features_pl.col("_season_context_team_conference_rank_fraction")
            - season_context_features_pl.col("_season_context_opponent_conference_rank_fraction")
        ).alias("season_context_conference_rank_fraction_delta"),
        season_context_features_pl.col("_season_context_team_division_rank_fraction").alias(
            "season_context_team_division_rank_fraction"
        ),
        season_context_features_pl.col("_season_context_opponent_division_rank_fraction").alias(
            "season_context_opponent_division_rank_fraction"
        ),
        (
            season_context_features_pl.col("_season_context_team_division_rank_fraction")
            - season_context_features_pl.col("_season_context_opponent_division_rank_fraction")
        ).alias("season_context_division_rank_fraction_delta"),
        season_context_features_pl.col("_season_context_team_gap_to_conference_7th").alias(
            "season_context_team_gap_to_conference_7th"
        ),
        season_context_features_pl.col("_season_context_opponent_gap_to_conference_7th").alias(
            "season_context_opponent_gap_to_conference_7th"
        ),
        season_context_features_pl.col("_season_context_team_gap_to_division_leader").alias(
            "season_context_team_gap_to_division_leader"
        ),
        season_context_features_pl.col("_season_context_opponent_gap_to_division_leader").alias(
            "season_context_opponent_gap_to_division_leader"
        ),
        season_context_features_pl.col("_season_context_team_max_wins_over_conference_7th").alias(
            "season_context_team_max_wins_over_conference_7th"
        ),
        season_context_features_pl.col("_season_context_opponent_max_wins_over_conference_7th").alias(
            "season_context_opponent_max_wins_over_conference_7th"
        ),
    )
    output = rows.sort("_season_context_row_order").select(
        *base_rows.columns, *season_context_features_SEASON_CONTEXT_FEATURE_COLUMNS
    )
    if output.height != base_rows.height or not output.select(base_rows.columns).equals(base_rows):
        raise RuntimeError("season-context builder changed source rows or existing feature values")
    if any(
        (not output.schema[column].is_numeric() for column in season_context_features_SEASON_CONTEXT_FEATURE_COLUMNS)
    ):
        raise RuntimeError("season-context features must all be numeric")
    return output


feature_sets_BASE_FEATURE_SET = "qb_offense_v0"

feature_sets_FEATURE_SET_GROUPS: dict[str, tuple[str, ...]] = {
    feature_sets_BASE_FEATURE_SET: (),
    "qb_offense_defense": ("defense",),
    "qb_offense_context": ("context",),
    "qb_offense_availability": ("availability",),
    "qb_offense_market": ("market",),
    "qb_offense_defense_context": ("defense", "context"),
    "qb_offense_defense_context_weather": ("defense", "context", "weather_forecast"),
    "qb_offense_defense_context_weather_market": (
        "defense",
        "context",
        "weather_forecast",
        "market",
    ),
    "qb_offense_defense_context_season": ("defense", "context", "season_context"),
    "qb_offense_defense_context_season_weather": (
        "defense",
        "context",
        "season_context",
        "weather_forecast",
    ),
    "qb_offense_defense_context_trend": ("defense", "context", "trend"),
    "qb_offense_defense_context_matchup": ("defense", "context", "matchup"),
    "qb_offense_defense_context_derived": ("defense", "context", "trend", "matchup"),
    "qb_offense_defense_context_market": ("defense", "context", "market"),
    "qb_offense_full_sports": ("defense", "context", "availability"),
    "qb_offense_full_market": ("defense", "context", "availability", "market"),
}


def feature_sets_available_feature_sets() -> tuple[str, ...]:
    return tuple(feature_sets_FEATURE_SET_GROUPS)


def feature_sets_base_feature_columns(config: data_DataConfig) -> list[str]:
    columns = features_qb_feature_columns(config.qb_rolling_windows) + offense_features_offense_feature_columns(
        config.offense_rolling_windows
    )
    if len(columns) != 47 or len(columns) != len(set(columns)):
        raise ValueError(f"qb_offense_v0 must contain 47 unique features, got {len(columns)}")
    return columns


def feature_sets_feature_group_columns(config: data_DataConfig, group: str) -> list[str]:
    if group == "defense":
        return defense_features_defense_feature_columns(config.defense_rolling_windows)
    if group == "context":
        return list(context_features_CONTEXT_FEATURE_COLUMNS)
    if group == "availability":
        return availability_features_availability_feature_columns()
    if group == "market":
        return list(context_features_MARKET_FEATURE_COLUMNS)
    if group == "trend":
        return list(derived_features_TREND_FEATURE_COLUMNS)
    if group == "matchup":
        return list(derived_features_MATCHUP_FEATURE_COLUMNS)
    if group == "season_context":
        return list(season_context_features_SEASON_CONTEXT_FEATURE_COLUMNS)
    if group == "weather_forecast":
        raise ValueError("Weather is not part of this reproducible blog model.")
    raise ValueError(f"Unknown feature group: {group}")


def feature_sets_feature_columns_for_set(config: data_DataConfig, feature_set: str) -> list[str]:
    try:
        groups = feature_sets_FEATURE_SET_GROUPS[feature_set]
    except KeyError as error:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose one of {list(feature_sets_available_feature_sets())}"
        ) from error
    columns = feature_sets_base_feature_columns(config)
    for group in groups:
        columns.extend(feature_sets_feature_group_columns(config, group))
    if len(columns) != len(set(columns)):
        raise ValueError(f"Feature set {feature_set!r} contains duplicate columns")
    return columns


import json as live_features_json

from dataclasses import dataclass as live_features_dataclass

from pathlib import Path as live_features_Path

import nflreadpy as live_features_nfl

import polars as live_features_pl

live_features_LIVE_TARGET_COLUMN = "remaining_passing_yards"

live_features_LIVE_PBP_COLUMNS = (
    "season",
    "season_type",
    "week",
    "game_id",
    "play_id",
    "qtr",
    "time_of_day",
    "posteam",
    "home_team",
    "away_team",
    "passer_player_id",
    "passer_player_name",
    "pass_attempt",
    "complete_pass",
    "passing_yards",
    "air_yards",
    "qb_dropback",
    "sack",
    "qb_hit",
    "cpoe",
    "epa",
    "total_home_score",
    "total_away_score",
    "game_seconds_remaining",
    "quarter_seconds_remaining",
    "play_type",
    "rush_attempt",
    "qb_kneel",
    "qb_spike",
    "interception",
    "success",
)

live_features_LIVE_FEATURE_COLUMNS = (
    "live_qb_has_dropback",
    "live_qb_attempts",
    "live_qb_completions",
    "live_qb_passing_yards",
    "live_qb_dropbacks",
    "live_qb_epa_per_dropback",
    "live_qb_cpoe",
    "live_qb_air_yards_per_attempt",
    "live_qb_ypa",
    "live_qb_sack_rate",
    "live_qb_hit_rate",
    "live_qb_explosive_complete_rate",
    "live_qb_interceptions",
    "live_offense_plays",
    "live_offense_pass_rate",
    "live_offense_epa_per_play",
    "live_offense_success_rate",
    "live_team_score",
    "live_opponent_score",
    "live_score_differential",
    "live_game_total",
    "live_game_seconds_remaining",
    "live_is_leading",
)

live_features_LIVE_USAGE_FEATURE_COLUMNS = (
    "live_qb_is_latest_team_passer",
    "live_qb_anchor_quarter_pass_plays",
    "live_qb_anchor_quarter_pass_play_share",
    "live_qb_recent_pass_play_share",
    "live_qb_seconds_since_last_pass_play",
)

live_features_LIVE_BOXSCORE_FEATURE_COLUMNS = (
    "live_qb_attempts",
    "live_qb_completions",
    "live_qb_passing_yards",
    "live_qb_interceptions",
    "live_qb_ypa",
    "live_team_score",
    "live_opponent_score",
    "live_score_differential",
    "live_game_total",
    "live_game_seconds_remaining",
    "live_is_leading",
)

live_features_LIVE_PLAYSTATS_FEATURE_COLUMNS = (
    "live_qb_has_dropback",
    "live_qb_attempts",
    "live_qb_completions",
    "live_qb_passing_yards",
    "live_qb_dropbacks",
    "live_qb_ypa",
    "live_qb_sack_rate",
    "live_qb_explosive_complete_rate",
    "live_qb_interceptions",
    "live_offense_plays",
    "live_offense_pass_rate",
    "live_team_score",
    "live_opponent_score",
    "live_score_differential",
    "live_game_total",
    "live_game_seconds_remaining",
    "live_is_leading",
) + live_features_LIVE_USAGE_FEATURE_COLUMNS

live_features_LIVE_OPPONENT_FEATURE_COLUMNS = (
    "live_opponent_offense_plays",
    "live_opponent_pass_rate",
    "live_opponent_epa_per_play",
    "live_opponent_success_rate",
    "live_opponent_passing_yards",
    "live_opponent_interceptions",
    "live_game_offense_plays",
    "live_team_play_share",
)

live_features_LIVE_TEMPO_FEATURE_COLUMNS = (
    "live_qb_q2_yards_per_dropback",
    "live_qb_q2_epa_per_dropback",
    "live_qb_q2_sack_rate",
    "live_qb_q2_minus_q1_yards_per_dropback",
    "live_qb_two_minute_dropback_share",
    "live_qb_before_two_minute_yards_per_dropback",
    "live_offense_q2_pass_rate",
    "live_offense_q2_play_share",
    "live_offense_q2_minus_q1_pass_rate",
    "live_qb_two_minute_clock_coverage",
)

live_features_LIVE_DRIVE_RECEIVER_FEATURE_COLUMNS = (
    "live_offense_drive_count",
    "live_offense_plays_per_drive",
    "live_offense_first_downs_per_drive",
    "live_offense_third_down_conversion_rate",
    "live_offense_punt_drive_rate",
    "live_offense_turnover_drive_rate",
    "live_qb_targeted_receiver_count",
    "live_qb_receiver_target_hhi",
    "live_qb_top_receiver_target_share",
    "live_qb_yac_per_completion",
)


@live_features_dataclass(frozen=True)
class live_features_LiveConfig:
    anchor_quarter: int
    decision_delay_minutes: int
    maximum_quote_age_minutes: int
    pregame_feature_set: str
    pbp_path: live_features_Path
    feature_dataset_path: live_features_Path
    candlestick_cache_dir: live_features_Path
    validation_predictions_path: live_features_Path
    validation_metrics_path: live_features_Path
    test_predictions_path: live_features_Path
    test_metrics_path: live_features_Path


def live_features_live_feature_columns(
    data_config: data_DataConfig,
    live_config: live_features_LiveConfig,
    live_feature_group: str = "base",
) -> list[str]:
    live_columns = {
        "boxscore": live_features_LIVE_BOXSCORE_FEATURE_COLUMNS,
        "playstats": live_features_LIVE_PLAYSTATS_FEATURE_COLUMNS,
    }.get(live_feature_group, live_features_LIVE_FEATURE_COLUMNS)
    columns = feature_sets_feature_columns_for_set(data_config, live_config.pregame_feature_set) + list(live_columns)
    if live_feature_group in {"usage", "opponent", "tempo", "drive_receiver"}:
        columns.extend(live_features_LIVE_USAGE_FEATURE_COLUMNS)
        if live_feature_group == "opponent":
            columns.extend(live_features_LIVE_OPPONENT_FEATURE_COLUMNS)
        elif live_feature_group == "tempo":
            columns.extend(live_features_LIVE_TEMPO_FEATURE_COLUMNS)
        elif live_feature_group == "drive_receiver":
            columns.extend(live_features_LIVE_DRIVE_RECEIVER_FEATURE_COLUMNS)
    elif live_feature_group not in {"base", "boxscore", "playstats"}:
        raise ValueError(f"unknown live feature group: {live_feature_group}")
    if len(columns) != len(set(columns)):
        raise ValueError("live feature set contains duplicate columns")
    identifier_columns = {
        "game_id",
        "team",
        "actual_qb_id",
        "actual_qb_name",
        "play_id",
    }
    leaked = sorted(identifier_columns.intersection(columns))
    if leaked:
        raise ValueError(f"identifier columns are not model features: {leaked}")
    return columns


def live_features__download_live_pbp(seasons: list[int]) -> live_features_pl.DataFrame:
    frames = []
    for season in seasons:
        frame = live_features_nfl.load_pbp([season]).select(*live_features_LIVE_PBP_COLUMNS)
        frames.append(frame)
        print(
            live_features_json.dumps({"downloaded_live_pbp_season": season, "rows": frame.height}),
            flush=True,
        )
    return live_features_pl.concat(frames, how="diagonal_relaxed")


def live_features_load_live_pbp(
    data_config: data_DataConfig,
    live_config: live_features_LiveConfig,
    refresh: bool = False,
) -> live_features_pl.DataFrame:
    if live_config.pbp_path.exists() and (not refresh):
        return live_features_pl.read_parquet(live_config.pbp_path)
    live_config.pbp_path.parent.mkdir(parents=True, exist_ok=True)
    seasons = list(range(data_config.first_eligible_season, data_config.test_season + 1))
    frame = live_features__download_live_pbp(seasons)
    frame.write_parquet(live_config.pbp_path)
    return frame


def live_features__timestamped_plays(
    pbp: live_features_pl.DataFrame, season_type: str, anchor_quarter: int
) -> live_features_pl.DataFrame:
    required = set(live_features_LIVE_PBP_COLUMNS)
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"live PBP is missing required columns: {missing}")
    return (
        pbp.filter(
            (live_features_pl.col("season_type") == season_type)
            & live_features_pl.col("qtr").is_between(1, anchor_quarter)
            & live_features_pl.col("time_of_day").is_not_null()
        )
        .with_columns(
            live_features_pl.col("qtr").cast(live_features_pl.Int32),
            live_features_pl.col("time_of_day")
            .str.to_datetime(time_zone="UTC", strict=True)
            .alias("_play_timestamp_utc"),
            *[
                live_features_pl.when(live_features_pl.col(column).is_in(["OAK", "LV"]))
                .then(
                    live_features_pl.when(live_features_pl.col("season") < 2020)
                    .then(live_features_pl.lit("OAK"))
                    .otherwise(live_features_pl.lit("LV"))
                )
                .otherwise(live_features_pl.col(column))
                .alias(column)
                for column in ("posteam", "home_team", "away_team")
            ],
        )
        .filter(live_features_pl.col("_play_timestamp_utc").is_not_null())
    )


def live_features__anchor_rows(plays: live_features_pl.DataFrame, anchor_quarter: int) -> live_features_pl.DataFrame:
    return (
        plays.filter(live_features_pl.col("qtr") == anchor_quarter)
        .sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            live_features_pl.col("_play_timestamp_utc").last().alias("live_anchor_utc"),
            live_features_pl.col("play_id").last().alias("live_anchor_play_id"),
        )
    )


def live_features__qb_aggregates(
    plays: live_features_pl.DataFrame,
) -> live_features_pl.DataFrame:
    return (
        plays.filter(live_features_pl.col("passer_player_id").is_not_null())
        .group_by("game_id", live_features_pl.col("posteam").alias("team"), "passer_player_id")
        .agg(
            live_features_pl.col("pass_attempt").fill_null(0).sum().alias("live_qb_attempts"),
            live_features_pl.col("complete_pass").fill_null(0).sum().alias("live_qb_completions"),
            live_features_pl.col("passing_yards").fill_null(0).sum().alias("live_qb_passing_yards"),
            live_features_pl.col("qb_dropback").fill_null(0).sum().alias("live_qb_dropbacks"),
            live_features_pl.col("sack").fill_null(0).sum().alias("_live_qb_sacks"),
            live_features_pl.col("qb_hit").fill_null(0).sum().alias("_live_qb_hits"),
            live_features_pl.col("epa")
            .filter(live_features_pl.col("qb_dropback") == 1)
            .mean()
            .alias("live_qb_epa_per_dropback"),
            live_features_pl.col("cpoe").filter(live_features_pl.col("pass_attempt") == 1).mean().alias("live_qb_cpoe"),
            live_features_pl.col("air_yards")
            .filter(live_features_pl.col("pass_attempt") == 1)
            .mean()
            .alias("live_qb_air_yards_per_attempt"),
            ((live_features_pl.col("complete_pass") == 1) & (live_features_pl.col("passing_yards") >= 20))
            .cast(live_features_pl.Float64)
            .sum()
            .alias("_live_qb_explosive_completions"),
            live_features_pl.col("interception").fill_null(0).sum().alias("live_qb_interceptions"),
        )
        .with_columns(
            (live_features_pl.col("live_qb_dropbacks") > 0)
            .cast(live_features_pl.Float64)
            .alias("live_qb_has_dropback"),
            live_features_pl.when(live_features_pl.col("live_qb_attempts") > 0)
            .then(live_features_pl.col("live_qb_passing_yards") / live_features_pl.col("live_qb_attempts"))
            .otherwise(0.0)
            .alias("live_qb_ypa"),
            live_features_pl.when(live_features_pl.col("live_qb_dropbacks") > 0)
            .then(live_features_pl.col("_live_qb_sacks") / live_features_pl.col("live_qb_dropbacks"))
            .otherwise(0.0)
            .alias("live_qb_sack_rate"),
            live_features_pl.when(live_features_pl.col("live_qb_dropbacks") > 0)
            .then(live_features_pl.col("_live_qb_hits") / live_features_pl.col("live_qb_dropbacks"))
            .otherwise(0.0)
            .alias("live_qb_hit_rate"),
            live_features_pl.when(live_features_pl.col("live_qb_attempts") > 0)
            .then(live_features_pl.col("_live_qb_explosive_completions") / live_features_pl.col("live_qb_attempts"))
            .otherwise(0.0)
            .alias("live_qb_explosive_complete_rate"),
        )
        .drop("_live_qb_sacks", "_live_qb_hits", "_live_qb_explosive_completions")
    )


def live_features__offense_aggregates(
    plays: live_features_pl.DataFrame,
) -> live_features_pl.DataFrame:
    eligible_play = (
        live_features_pl.col("play_type").is_in(["pass", "run"])
        & (live_features_pl.col("qb_kneel").fill_null(0) != 1)
        & (live_features_pl.col("qb_spike").fill_null(0) != 1)
    )
    return (
        plays.group_by("game_id", live_features_pl.col("posteam").alias("team"))
        .agg(
            eligible_play.cast(live_features_pl.Float64).sum().alias("live_offense_plays"),
            live_features_pl.col("qb_dropback")
            .fill_null(0)
            .filter(eligible_play)
            .sum()
            .alias("_live_offense_dropbacks"),
            live_features_pl.col("epa").filter(eligible_play).mean().alias("live_offense_epa_per_play"),
            live_features_pl.col("success").filter(eligible_play).mean().alias("live_offense_success_rate"),
        )
        .with_columns(
            live_features_pl.when(live_features_pl.col("live_offense_plays") > 0)
            .then(live_features_pl.col("_live_offense_dropbacks") / live_features_pl.col("live_offense_plays"))
            .otherwise(0.0)
            .alias("live_offense_pass_rate")
        )
        .drop("_live_offense_dropbacks")
    )


def live_features__game_state(
    plays: live_features_pl.DataFrame,
) -> live_features_pl.DataFrame:
    return (
        plays.sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            live_features_pl.col("total_home_score").drop_nulls().last().alias("_live_home_score"),
            live_features_pl.col("total_away_score").drop_nulls().last().alias("_live_away_score"),
            live_features_pl.col("game_seconds_remaining").drop_nulls().last().alias("live_game_seconds_remaining"),
        )
    )


def live_features__usage_aggregates(
    plays: live_features_pl.DataFrame, anchor_quarter: int
) -> live_features_pl.DataFrame:
    """Observable passer replacement/workload signals, not postgame injury labels."""
    passer_plays = (
        plays.filter(
            live_features_pl.col("passer_player_id").is_not_null() & (live_features_pl.col("qb_dropback") == 1)
        )
        .rename({"posteam": "team"})
        .sort(["game_id", "team", "_play_timestamp_utc", "play_id"])
    )
    team_state = passer_plays.group_by("game_id", "team", maintain_order=True).agg(
        live_features_pl.col("passer_player_id").last().alias("_latest_passer_id"),
        (live_features_pl.col("qtr") == anchor_quarter).sum().alias("_team_anchor_quarter_pass_plays"),
        live_features_pl.len().clip(upper_bound=10).alias("_team_recent_pass_plays"),
    )
    recent = (
        passer_plays.group_by("game_id", "team", maintain_order=True)
        .tail(10)
        .group_by("game_id", "team", "passer_player_id")
        .agg(live_features_pl.len().alias("_qb_recent_pass_plays"))
    )
    return (
        passer_plays.group_by("game_id", "team", "passer_player_id")
        .agg(
            (live_features_pl.col("qtr") == anchor_quarter)
            .cast(live_features_pl.Float64)
            .sum()
            .alias("live_qb_anchor_quarter_pass_plays"),
            live_features_pl.col("_play_timestamp_utc").max().alias("_qb_last_pass_play_utc"),
            live_features_pl.col("live_anchor_utc").first(),
        )
        .join(team_state, on=["game_id", "team"], how="left")
        .join(recent, on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (live_features_pl.col("passer_player_id") == live_features_pl.col("_latest_passer_id"))
            .cast(live_features_pl.Float64)
            .alias("live_qb_is_latest_team_passer"),
            live_features_pl.when(live_features_pl.col("_team_anchor_quarter_pass_plays") > 0)
            .then(
                live_features_pl.col("live_qb_anchor_quarter_pass_plays")
                / live_features_pl.col("_team_anchor_quarter_pass_plays")
            )
            .otherwise(0.0)
            .alias("live_qb_anchor_quarter_pass_play_share"),
            (
                live_features_pl.col("_qb_recent_pass_plays").fill_null(0)
                / live_features_pl.col("_team_recent_pass_plays")
            ).alias("live_qb_recent_pass_play_share"),
            (live_features_pl.col("live_anchor_utc") - live_features_pl.col("_qb_last_pass_play_utc"))
            .dt.total_seconds()
            .cast(live_features_pl.Float64)
            .alias("live_qb_seconds_since_last_pass_play"),
        )
        .select(
            "game_id",
            "team",
            "passer_player_id",
            *live_features_LIVE_USAGE_FEATURE_COLUMNS,
        )
    )


def live_features_build_live_rows(
    pregame_rows: live_features_pl.DataFrame,
    pbp: live_features_pl.DataFrame,
    data_config: data_DataConfig,
    live_config: live_features_LiveConfig,
) -> live_features_pl.DataFrame:
    """Build one timestamp-bounded row per observed starting-QB/game."""
    required_pregame = {
        "game_id",
        "season",
        "week",
        "team",
        "is_home",
        "actual_qb_id",
        "actual_qb_name",
        "official_passing_yards",
    }
    missing = sorted(required_pregame.difference(pregame_rows.columns))
    if missing:
        raise ValueError(f"pregame rows are missing required columns: {missing}")
    plays = live_features__timestamped_plays(pbp, data_config.season_type, live_config.anchor_quarter)
    anchors = live_features__anchor_rows(plays, live_config.anchor_quarter)
    bounded = plays.join(anchors, on="game_id", how="inner").filter(
        live_features_pl.col("_play_timestamp_utc") <= live_features_pl.col("live_anchor_utc")
    )
    rows = (
        pregame_rows.filter(
            live_features_pl.col("season").is_between(data_config.first_eligible_season, data_config.test_season)
            & live_features_pl.col("actual_qb_id").is_not_null()
            & live_features_pl.col("official_passing_yards").is_not_null()
        )
        .join(anchors, on="game_id", how="left")
        .join(
            live_features__qb_aggregates(bounded),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(
            live_features__offense_aggregates(bounded),
            on=["game_id", "team"],
            how="left",
        )
        .join(
            live_features__usage_aggregates(bounded, live_config.anchor_quarter),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(live_features__game_state(bounded), on="game_id", how="left")
        .with_columns(
            live_features_pl.when(live_features_pl.col("is_home") == 1)
            .then(live_features_pl.col("_live_home_score"))
            .otherwise(live_features_pl.col("_live_away_score"))
            .alias("live_team_score"),
            live_features_pl.when(live_features_pl.col("is_home") == 1)
            .then(live_features_pl.col("_live_away_score"))
            .otherwise(live_features_pl.col("_live_home_score"))
            .alias("live_opponent_score"),
            (
                live_features_pl.col("live_anchor_utc")
                + live_features_pl.duration(minutes=live_config.decision_delay_minutes)
            ).alias("live_decision_utc"),
        )
        .with_columns(
            (live_features_pl.col("live_team_score") - live_features_pl.col("live_opponent_score")).alias(
                "live_score_differential"
            ),
            (live_features_pl.col("live_team_score") + live_features_pl.col("live_opponent_score")).alias(
                "live_game_total"
            ),
        )
        .with_columns(
            (live_features_pl.col("live_score_differential") > 0)
            .cast(live_features_pl.Float64)
            .alias("live_is_leading"),
            (
                live_features_pl.col("official_passing_yards")
                - live_features_pl.col("live_qb_passing_yards").fill_null(0)
            ).alias(live_features_LIVE_TARGET_COLUMN),
            live_features_pl.col("live_anchor_utc").is_not_null().alias("live_evaluation_eligible"),
        )
        .drop("_live_home_score", "_live_away_score")
    )
    rows = rows.with_columns(
        [
            live_features_pl.col(column).cast(live_features_pl.Float64).fill_null(0.0).fill_nan(0.0)
            for column in live_features_LIVE_FEATURE_COLUMNS + live_features_LIVE_USAGE_FEATURE_COLUMNS
        ]
    )
    return rows.sort(["season", "week", "game_id", "team"])


import polars as live_flow_features_pl

live_flow_features_FLOW_COLUMNS = (
    *live_features_LIVE_OPPONENT_FEATURE_COLUMNS,
    *live_features_LIVE_TEMPO_FEATURE_COLUMNS,
)

live_flow_features_KEYS = ["game_id", "team"]


def live_flow_features__ratio(
    numerator: live_flow_features_pl.Expr, denominator: live_flow_features_pl.Expr
) -> live_flow_features_pl.Expr:
    return (
        live_flow_features_pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(None)
        .cast(live_flow_features_pl.Float64)
    )


def live_flow_features_build_live_flow_features(
    rows: live_flow_features_pl.DataFrame,
    pbp: live_flow_features_pl.DataFrame,
    live_config: live_features_LiveConfig,
    season_type: str,
) -> live_flow_features_pl.DataFrame:
    """Only consume plays at/before each already-frozen anchor; never targets."""
    if live_config.anchor_quarter != 2:
        raise ValueError("quarter-flow features are defined only for the halftime protocol")
    required = {
        *live_flow_features_KEYS,
        "opponent_team",
        "actual_qb_id",
        "live_anchor_utc",
    }
    if required - set(rows.columns):
        raise ValueError(f"missing frozen row fields: {sorted(required - set(rows.columns))}")
    if set(live_flow_features_FLOW_COLUMNS) & set(rows.columns):
        raise ValueError("flow columns already exist; preserve the original source table")
    if rows.select(live_flow_features_KEYS).is_duplicated().any():
        raise ValueError("flow source requires unique QB-game rows")
    anchors = rows.select("game_id", "live_anchor_utc").unique()
    if anchors["game_id"].is_duplicated().any():
        raise ValueError("each game must have one frozen anchor")
    plays = (
        live_features__timestamped_plays(pbp, season_type, 2)
        .join(anchors, on="game_id", how="inner")
        .filter(live_flow_features_pl.col("_play_timestamp_utc") <= live_flow_features_pl.col("live_anchor_utc"))
        .filter(live_flow_features_pl.col("posteam").is_not_null())
        .filter(
            live_flow_features_pl.col("play_type").is_in(["pass", "run"])
            & (live_flow_features_pl.col("qb_kneel").fill_null(0) != 1)
            & (live_flow_features_pl.col("qb_spike").fill_null(0) != 1)
        )
        .rename({"posteam": "team"})
    )
    q2 = live_flow_features_pl.col("qtr") == 2
    valid_clock = live_flow_features_pl.col("quarter_seconds_remaining").is_between(0, 900).fill_null(False)
    two_minute = q2 & valid_clock & (live_flow_features_pl.col("quarter_seconds_remaining") <= 120)
    before_two_minute = ~q2 | valid_clock & (live_flow_features_pl.col("quarter_seconds_remaining") > 120)
    dropback = live_flow_features_pl.col("qb_dropback").fill_null(0)
    team = (
        plays.group_by(live_flow_features_KEYS)
        .agg(
            live_flow_features_pl.len().cast(live_flow_features_pl.Float64).alias("_plays"),
            dropback.mean().alias("_pass_rate"),
            live_flow_features_pl.col("epa").mean().alias("_epa"),
            live_flow_features_pl.col("success").mean().alias("_success"),
            live_flow_features_pl.col("passing_yards").fill_null(0).sum().alias("_passing_yards"),
            live_flow_features_pl.col("interception").fill_null(0).sum().alias("_interceptions"),
            q2.cast(live_flow_features_pl.Float64).sum().alias("_q2_plays"),
            dropback.filter(q2).mean().alias("live_offense_q2_pass_rate"),
            dropback.filter(~q2).mean().alias("_q1_pass_rate"),
        )
        .with_columns(
            live_flow_features__ratio(
                live_flow_features_pl.col("_q2_plays"),
                live_flow_features_pl.col("_plays"),
            ).alias("live_offense_q2_play_share"),
            (live_flow_features_pl.col("live_offense_q2_pass_rate") - live_flow_features_pl.col("_q1_pass_rate")).alias(
                "live_offense_q2_minus_q1_pass_rate"
            ),
        )
    )
    opponent = team.select(
        "game_id",
        live_flow_features_pl.col("team").alias("opponent_team"),
        live_flow_features_pl.col("_plays").alias("live_opponent_offense_plays"),
        live_flow_features_pl.col("_pass_rate").alias("live_opponent_pass_rate"),
        live_flow_features_pl.col("_epa").alias("live_opponent_epa_per_play"),
        live_flow_features_pl.col("_success").alias("live_opponent_success_rate"),
        live_flow_features_pl.col("_passing_yards").alias("live_opponent_passing_yards"),
        live_flow_features_pl.col("_interceptions").alias("live_opponent_interceptions"),
    )
    qb = (
        plays.filter((dropback == 1) & live_flow_features_pl.col("passer_player_id").is_not_null())
        .group_by(
            *live_flow_features_KEYS,
            live_flow_features_pl.col("passer_player_id").alias("actual_qb_id"),
        )
        .agg(
            live_flow_features_pl.len().cast(live_flow_features_pl.Float64).alias("_qb_dropbacks"),
            q2.cast(live_flow_features_pl.Float64).sum().alias("_qb_q2_dropbacks"),
            (q2 & ~valid_clock).cast(live_flow_features_pl.Float64).sum().alias("_qb_bad_q2_clock"),
            two_minute.fill_null(False).cast(live_flow_features_pl.Float64).sum().alias("_qb_two_minute_dropbacks"),
            live_flow_features_pl.col("passing_yards")
            .fill_null(0)
            .filter(q2)
            .mean()
            .alias("live_qb_q2_yards_per_dropback"),
            live_flow_features_pl.col("passing_yards")
            .fill_null(0)
            .filter(~q2)
            .mean()
            .alias("_qb_q1_yards_per_dropback"),
            live_flow_features_pl.col("epa").filter(q2).mean().alias("live_qb_q2_epa_per_dropback"),
            live_flow_features_pl.col("sack").filter(q2).mean().alias("live_qb_q2_sack_rate"),
            live_flow_features_pl.col("passing_yards")
            .fill_null(0)
            .filter(before_two_minute)
            .mean()
            .alias("_qb_before_two_minute_ypdb"),
        )
        .with_columns(
            (
                live_flow_features_pl.col("live_qb_q2_yards_per_dropback")
                - live_flow_features_pl.col("_qb_q1_yards_per_dropback")
            ).alias("live_qb_q2_minus_q1_yards_per_dropback"),
            live_flow_features_pl.when(live_flow_features_pl.col("_qb_bad_q2_clock") == 0)
            .then(
                live_flow_features__ratio(
                    live_flow_features_pl.col("_qb_two_minute_dropbacks"),
                    live_flow_features_pl.col("_qb_dropbacks"),
                )
            )
            .otherwise(None)
            .alias("live_qb_two_minute_dropback_share"),
            live_flow_features_pl.when(live_flow_features_pl.col("_qb_bad_q2_clock") == 0)
            .then(live_flow_features_pl.col("_qb_before_two_minute_ypdb"))
            .otherwise(None)
            .alias("live_qb_before_two_minute_yards_per_dropback"),
            live_flow_features_pl.when(live_flow_features_pl.col("_qb_q2_dropbacks") > 0)
            .then(1.0 - live_flow_features_pl.col("_qb_bad_q2_clock") / live_flow_features_pl.col("_qb_q2_dropbacks"))
            .otherwise(1.0)
            .alias("live_qb_two_minute_clock_coverage"),
        )
        .select(
            *live_flow_features_KEYS,
            "actual_qb_id",
            *[c for c in live_features_LIVE_TEMPO_FEATURE_COLUMNS if c.startswith("live_qb_")],
        )
    )
    team_tempo = [c for c in live_features_LIVE_TEMPO_FEATURE_COLUMNS if c.startswith("live_offense_")]
    result = (
        rows.join(
            team.select(*live_flow_features_KEYS, "_plays", *team_tempo),
            on=live_flow_features_KEYS,
            how="left",
            validate="1:1",
        )
        .join(opponent, on=["game_id", "opponent_team"], how="left", validate="m:1")
        .join(
            qb,
            on=[*live_flow_features_KEYS, "actual_qb_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (live_flow_features_pl.col("_plays") + live_flow_features_pl.col("live_opponent_offense_plays")).alias(
                "live_game_offense_plays"
            )
        )
        .with_columns(
            live_flow_features__ratio(
                live_flow_features_pl.col("_plays"),
                live_flow_features_pl.col("live_game_offense_plays"),
            ).alias("live_team_play_share")
        )
        .with_columns(
            [
                live_flow_features_pl.col(c).cast(live_flow_features_pl.Float64).fill_nan(None)
                for c in live_flow_features_FLOW_COLUMNS
            ]
        )
        .sort(live_flow_features_KEYS)
    )
    if (
        "live_offense_plays" in rows.columns
        and result.filter(
            live_flow_features_pl.col("live_anchor_utc").is_not_null()
            & (live_flow_features_pl.col("_plays").fill_null(0) != live_flow_features_pl.col("live_offense_plays"))
        ).height
    ):
        raise ValueError("source halftime aggregates are stale; rebuild the base table before adding flow features")
    result = result.drop("_plays")
    if not result.select(rows.columns).equals(rows.sort(live_flow_features_KEYS)):
        raise RuntimeError("flow join changed the source rows or existing feature values")
    return result


import polars as live_q1_features_pl


def live_q1_features__normalized_timestamped_plays(
    pbp: live_q1_features_pl.DataFrame, season_type: str
) -> live_q1_features_pl.DataFrame:
    missing = sorted(set(live_features_LIVE_PBP_COLUMNS).difference(pbp.columns))
    if missing:
        raise ValueError(f"Q1 PBP is missing required columns: {missing}")
    return (
        pbp.filter(
            (live_q1_features_pl.col("season_type") == season_type)
            & live_q1_features_pl.col("qtr").is_in([1, 2])
            & live_q1_features_pl.col("time_of_day").is_not_null()
        )
        .with_columns(
            live_q1_features_pl.col("qtr").cast(live_q1_features_pl.Int32),
            live_q1_features_pl.col("time_of_day")
            .str.to_datetime(time_zone="UTC", strict=True)
            .alias("_play_timestamp_utc"),
            *[
                live_q1_features_pl.when(live_q1_features_pl.col(column).is_in(["OAK", "LV"]))
                .then(
                    live_q1_features_pl.when(live_q1_features_pl.col("season") < 2020)
                    .then(live_q1_features_pl.lit("OAK"))
                    .otherwise(live_q1_features_pl.lit("LV"))
                )
                .otherwise(live_q1_features_pl.col(column))
                .alias(column)
                for column in ("posteam", "home_team", "away_team")
            ],
        )
        .filter(live_q1_features_pl.col("_play_timestamp_utc").is_not_null())
    )


def live_q1_features_q1_boundaries(
    plays: live_q1_features_pl.DataFrame,
) -> live_q1_features_pl.DataFrame:
    """Use the earliest timestamped Q2 record, with a frozen clock guard."""
    anchors = (
        plays.filter(live_q1_features_pl.col("qtr") == 2)
        .sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            live_q1_features_pl.col("_play_timestamp_utc").first().alias("live_anchor_utc"),
            live_q1_features_pl.col("play_id").first().alias("live_anchor_play_id"),
            live_q1_features_pl.col("game_seconds_remaining").first().alias("live_anchor_game_seconds_remaining"),
        )
    )
    invalid = anchors.filter(~live_q1_features_pl.col("live_anchor_game_seconds_remaining").is_between(2400.0, 2700.0))
    if invalid.height:
        raise ValueError(f"Q1 boundary clock is outside [2400, 2700]: {invalid.head(5).to_dicts()}")
    return anchors


def live_q1_features__q1_game_state(
    plays: live_q1_features_pl.DataFrame,
) -> live_q1_features_pl.DataFrame:
    return (
        plays.sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            live_q1_features_pl.col("total_home_score").drop_nulls().last().alias("_live_home_score"),
            live_q1_features_pl.col("total_away_score").drop_nulls().last().alias("_live_away_score"),
            live_q1_features_pl.col("game_seconds_remaining").drop_nulls().last().alias("live_game_seconds_remaining"),
        )
    )


def live_q1_features__q1_usage_aggregates(
    plays: live_q1_features_pl.DataFrame,
) -> live_q1_features_pl.DataFrame:
    passer_plays = (
        plays.filter(
            live_q1_features_pl.col("passer_player_id").is_not_null() & (live_q1_features_pl.col("qb_dropback") == 1)
        )
        .rename({"posteam": "team"})
        .sort(["game_id", "team", "_play_timestamp_utc", "play_id"])
    )
    team_state = passer_plays.group_by("game_id", "team", maintain_order=True).agg(
        live_q1_features_pl.col("passer_player_id").last().alias("_latest_passer_id"),
        live_q1_features_pl.len().alias("_team_q1_pass_plays"),
        live_q1_features_pl.len().clip(upper_bound=10).alias("_team_recent_pass_plays"),
    )
    recent = (
        passer_plays.group_by("game_id", "team", maintain_order=True)
        .tail(10)
        .group_by("game_id", "team", "passer_player_id")
        .agg(live_q1_features_pl.len().alias("_qb_recent_pass_plays"))
    )
    return (
        passer_plays.group_by("game_id", "team", "passer_player_id")
        .agg(
            live_q1_features_pl.len().cast(live_q1_features_pl.Float64).alias("live_qb_anchor_quarter_pass_plays"),
            live_q1_features_pl.col("_play_timestamp_utc").max().alias("_qb_last_pass_play_utc"),
            live_q1_features_pl.col("live_anchor_utc").first(),
        )
        .join(team_state, on=["game_id", "team"], how="left")
        .join(recent, on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (live_q1_features_pl.col("passer_player_id") == live_q1_features_pl.col("_latest_passer_id"))
            .cast(live_q1_features_pl.Float64)
            .alias("live_qb_is_latest_team_passer"),
            (
                live_q1_features_pl.col("live_qb_anchor_quarter_pass_plays")
                / live_q1_features_pl.col("_team_q1_pass_plays")
            ).alias("live_qb_anchor_quarter_pass_play_share"),
            (
                live_q1_features_pl.col("_qb_recent_pass_plays").fill_null(0)
                / live_q1_features_pl.col("_team_recent_pass_plays")
            ).alias("live_qb_recent_pass_play_share"),
            (live_q1_features_pl.col("live_anchor_utc") - live_q1_features_pl.col("_qb_last_pass_play_utc"))
            .dt.total_seconds()
            .cast(live_q1_features_pl.Float64)
            .alias("live_qb_seconds_since_last_pass_play"),
        )
        .select(
            "game_id",
            "team",
            "passer_player_id",
            *live_features_LIVE_USAGE_FEATURE_COLUMNS,
        )
    )


def live_q1_features__pregame_columns(rows: live_q1_features_pl.DataFrame) -> list[str]:
    return [
        column
        for column in rows.columns
        if not column.startswith("live_") and column != live_features_LIVE_TARGET_COLUMN
    ]


def live_q1_features_build_q1_rows(
    source_rows: live_q1_features_pl.DataFrame,
    pbp: live_q1_features_pl.DataFrame,
    *,
    season_type: str = "REG",
    decision_delay_minutes: int = 2,
) -> live_q1_features_pl.DataFrame:
    """Build one Q1-boundary row per observed actual-QB/game."""
    required = {
        "game_id",
        "season",
        "week",
        "team",
        "is_home",
        "actual_qb_id",
        "actual_qb_name",
        "official_passing_yards",
    }
    missing = sorted(required.difference(source_rows.columns))
    if missing:
        raise ValueError(f"Q1 source rows are missing required columns: {missing}")
    key_columns = ["season", "week", "game_id", "team"]
    if source_rows.select(key_columns).is_duplicated().any():
        raise ValueError("Q1 source contains duplicate QB/game keys")
    plays = live_q1_features__normalized_timestamped_plays(pbp, season_type)
    anchors = live_q1_features_q1_boundaries(plays)
    q1 = (
        plays.filter(live_q1_features_pl.col("qtr") == 1)
        .join(anchors, on="game_id", how="inner")
        .filter(live_q1_features_pl.col("_play_timestamp_utc") < live_q1_features_pl.col("live_anchor_utc"))
    )
    missing_q1 = anchors.join(q1.select("game_id").unique(), on="game_id", how="anti")
    if missing_q1.height:
        raise ValueError(f"Q1 boundary has no preceding timestamped Q1 record: {missing_q1.head(5).to_dicts()}")
    invalid_passer_time = q1.filter(
        live_q1_features_pl.col("passer_player_id").is_not_null()
        & (live_q1_features_pl.col("_play_timestamp_utc") >= live_q1_features_pl.col("live_anchor_utc"))
    )
    if invalid_passer_time.height:
        examples = invalid_passer_time.select("game_id", "play_id").head(5).to_dicts()
        raise ValueError(f"Q1 passer timestamp does not precede Q2 boundary: {examples}")
    base = source_rows.select(live_q1_features__pregame_columns(source_rows))
    rows = (
        base.join(anchors, on="game_id", how="left")
        .join(
            live_features__qb_aggregates(q1),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(live_features__offense_aggregates(q1), on=["game_id", "team"], how="left")
        .join(
            live_q1_features__q1_usage_aggregates(q1),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(live_q1_features__q1_game_state(q1), on="game_id", how="left")
        .with_columns(
            live_q1_features_pl.when(live_q1_features_pl.col("is_home") == 1)
            .then(live_q1_features_pl.col("_live_home_score"))
            .otherwise(live_q1_features_pl.col("_live_away_score"))
            .alias("live_team_score"),
            live_q1_features_pl.when(live_q1_features_pl.col("is_home") == 1)
            .then(live_q1_features_pl.col("_live_away_score"))
            .otherwise(live_q1_features_pl.col("_live_home_score"))
            .alias("live_opponent_score"),
            (
                live_q1_features_pl.col("live_anchor_utc")
                + live_q1_features_pl.duration(minutes=decision_delay_minutes)
            ).alias("live_decision_utc"),
        )
        .with_columns(
            (live_q1_features_pl.col("live_team_score") - live_q1_features_pl.col("live_opponent_score")).alias(
                "live_score_differential"
            ),
            (live_q1_features_pl.col("live_team_score") + live_q1_features_pl.col("live_opponent_score")).alias(
                "live_game_total"
            ),
        )
        .with_columns(
            (live_q1_features_pl.col("live_score_differential") > 0)
            .cast(live_q1_features_pl.Float64)
            .alias("live_is_leading"),
            (
                live_q1_features_pl.col("official_passing_yards")
                - live_q1_features_pl.col("live_qb_passing_yards").fill_null(0.0)
            ).alias(live_features_LIVE_TARGET_COLUMN),
            live_q1_features_pl.col("live_anchor_utc").is_not_null().alias("live_evaluation_eligible"),
        )
        .drop("_live_home_score", "_live_away_score")
        .with_columns(
            *[
                live_q1_features_pl.col(column).cast(live_q1_features_pl.Float64).fill_null(0.0).fill_nan(0.0)
                for column in live_features_LIVE_FEATURE_COLUMNS + live_features_LIVE_USAGE_FEATURE_COLUMNS
            ]
        )
        .sort(key_columns)
    )
    return rows


import hashlib as pipeline_hashlib

import json as pipeline_json

from pathlib import Path as pipeline_Path

import nflreadpy as pipeline_nfl

import polars as pipeline_pl


def pipeline_blog_configs(cache_dir: pipeline_Path, output_dir: pipeline_Path):
    cache_dir, output_dir = (pipeline_Path(cache_dir), pipeline_Path(output_dir))
    data = data_DataConfig(
        2016,
        2018,
        2024,
        2025,
        "REG",
        60,
        cache_dir,
        output_dir / "base.parquet",
        output_dir / "pregame.parquet",
        (3, 8),
        (3, 8),
        0.2,
        0.8,
        3,
        True,
        (3, 8),
    )
    live = live_features_LiveConfig(
        2,
        2,
        5,
        "qb_offense_defense_context_season",
        cache_dir / "pbp_live_2018_2025.parquet",
        output_dir / "halftime.parquet",
        cache_dir / "kalshi",
        output_dir / "validation.parquet",
        output_dir / "validation.json",
        output_dir / "test.parquet",
        output_dir / "test.json",
    )
    return (data, live)


def pipeline_blog_feature_columns():
    """Exact ordered numeric candidates before training-only correlation pruning."""
    data, live = pipeline_blog_configs(pipeline_Path("cache"), pipeline_Path("outputs"))
    return (
        live_features_live_feature_columns(data, live, "usage")
        + [
            f"checkpoint_history_{source}_mean_last{window}"
            for window in checkpoint_history_WINDOWS
            for source in checkpoint_history_SOURCES
        ]
        + ["checkpoint_history_games_last8"]
    )


def pipeline_build_blog_features(cache_dir, output_dir, refresh=False):
    """Download public inputs and return Q1/halftime Polars tables.

    Identity is used to compute individual historical form, not as categorical
    model input, matching the selected original numeric model. Upstream data
    revisions may change outputs; input hashes are recorded with each run.
    """
    data, live = pipeline_blog_configs(cache_dir, output_dir)
    data.cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline_Path(output_dir).mkdir(parents=True, exist_ok=True)
    tables = data_load_nflverse_tables(data, refresh=refresh)
    print("Building QB form, offense, defense, and context.", flush=True)
    rows = data_build_base_qb_game_table(
        schedules=tables["schedules"],
        depth_charts=tables["depth_charts"],
        participation=tables["participation"],
        pbp_starter_plays=tables["pbp_starter_plays"],
        player_stats=tables["player_stats"],
        config=data,
    )
    rows = features_build_qb_rolling_features(
        rows,
        tables["player_stats"],
        tables["schedules"],
        data,
        qb_id_column="actual_qb_id",
    )
    rows = offense_features_build_offense_rolling_features(rows, tables["pbp_offense_plays"], tables["schedules"], data)
    rows = defense_features_build_defense_features(
        rows,
        tables["pbp_defense_plays"],
        tables["schedules"],
        season_type=data.season_type,
        windows=data.defense_rolling_windows,
    )
    rows = context_features_build_context_features(rows, tables["schedules"])
    pbp = live_features_load_live_pbp(data, live, refresh=refresh)
    halftime = live_features_build_live_rows(rows, pbp, data, live)
    q1 = live_q1_features_build_q1_rows(halftime, pbp)
    halftime = live_flow_features_build_live_flow_features(halftime, pbp, live, data.season_type)
    teams_path = data.cache_dir / "teams.parquet"
    if refresh or not teams_path.exists():
        pipeline_nfl.load_teams().write_parquet(teams_path)
    teams = pipeline_pl.read_parquet(teams_path)
    result = {}
    for name, checkpoint in (("q1", q1), ("halftime", halftime)):
        checkpoint = season_context_features_build_season_context_features(checkpoint, tables["schedules"], teams)
        checkpoint, _, _ = checkpoint_history_add_checkpoint_history(checkpoint)
        checkpoint.select(pipeline_blog_feature_columns())
        checkpoint.write_parquet(pipeline_Path(output_dir) / f"{name}_blog_features.parquet")
        result[name] = checkpoint
        print(
            f"{name}: {checkpoint.height} rows, {len(pipeline_blog_feature_columns())} candidates",
            flush=True,
        )
    manifest = {}
    for path in sorted(data.cache_dir.glob("*.parquet")):
        digest = pipeline_hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest[path.name] = digest.hexdigest()
    (pipeline_Path(output_dir) / "feature_input_hashes.json").write_text(pipeline_json.dumps(manifest, indent=2) + "\n")
    return result
