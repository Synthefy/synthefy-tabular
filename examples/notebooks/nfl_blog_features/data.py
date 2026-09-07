from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl


@dataclass(frozen=True)
class DataConfig:
    warmup_start_season: int
    first_eligible_season: int
    validation_season: int
    test_season: int
    season_type: str
    prediction_cutoff_minutes: int
    cache_dir: Path
    base_dataset_path: Path
    feature_dataset_path: Path
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


def load_config(path: Path) -> DataConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    data = raw["data"]
    features = raw["features"]
    evaluation = raw["evaluation"]
    root = path.parent
    return DataConfig(
        warmup_start_season=int(data["warmup_start_season"]),
        first_eligible_season=int(data["first_eligible_season"]),
        validation_season=int(data["validation_season"]),
        test_season=int(data["test_season"]),
        season_type=str(data["season_type"]),
        prediction_cutoff_minutes=int(data["prediction_cutoff_minutes"]),
        cache_dir=root / data["cache_dir"],
        base_dataset_path=root / data["base_dataset_path"],
        feature_dataset_path=root / data["feature_dataset_path"],
        qb_rolling_windows=tuple(int(window) for window in features["qb_rolling_windows"]),
        offense_rolling_windows=tuple(int(window) for window in features["offense_rolling_windows"]),
        neutral_wp_lower=float(features["neutral_wp_lower"]),
        neutral_wp_upper=float(features["neutral_wp_upper"]),
        neutral_max_quarter=int(features["neutral_max_quarter"]),
        include_starter_mismatch_trades=bool(evaluation["include_starter_mismatch_trades"]),
        defense_rolling_windows=tuple(int(window) for window in features["defense_rolling_windows"]),
    )


def _load_cached(
    cache_path: Path,
    loader: Callable[[list[int]], pl.DataFrame],
    seasons: list[int],
    refresh: bool,
) -> pl.DataFrame:
    if cache_path.exists() and not refresh:
        return pl.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = loader(seasons)
    frame.write_parquet(cache_path)
    return frame


def _load_pbp_starter_fields(seasons: list[int]) -> pl.DataFrame:
    fields = ("game_id", "season_type", "play_id", "posteam", "passer_player_id")
    return pl.concat([nfl.load_pbp(season).select(*fields) for season in seasons], how="diagonal_relaxed")


def _load_pbp_offense_fields(seasons: list[int]) -> pl.DataFrame:
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
    # Select each season before concatenating so the 300+ unused PBP columns
    # do not remain resident for every season at once.
    return pl.concat(
        [nfl.load_pbp(season).select(*fields) for season in seasons],
        how="diagonal_relaxed",
    )


def _load_pbp_defense_fields(seasons: list[int]) -> pl.DataFrame:
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
    return pl.concat(
        [nfl.load_pbp(season).select(*fields) for season in seasons],
        how="diagonal_relaxed",
    )


def load_nflverse_tables(config: DataConfig, refresh: bool = False) -> dict[str, pl.DataFrame]:
    season_tag = f"{config.warmup_start_season}_{config.test_season}"
    depth_tag = f"{config.first_eligible_season}_{config.test_season}"
    pbp_starter_seasons = config.depth_chart_seasons
    return {
        "schedules": _load_cached(
            config.cache_dir / f"schedules_{season_tag}.parquet",
            nfl.load_schedules,
            config.stat_seasons,
            refresh,
        ),
        "player_stats": _load_cached(
            config.cache_dir / f"player_stats_{season_tag}.parquet",
            nfl.load_player_stats,
            config.stat_seasons,
            refresh,
        ),
        # Participation is used only after games to identify the actual QB on
        # the first offensive snap. It is never included in model features.
        "participation": _load_cached(
            config.cache_dir / f"participation_{depth_tag}.parquet",
            nfl.load_participation,
            config.depth_chart_seasons,
            refresh,
        ),
        "pbp_starter_plays": _load_cached(
            config.cache_dir / f"pbp_starter_plays_{pbp_starter_seasons[0]}_{pbp_starter_seasons[-1]}.parquet",
            _load_pbp_starter_fields,
            pbp_starter_seasons,
            refresh,
        ),
        "pbp_offense_plays": _load_cached(
            config.cache_dir / f"pbp_offense_plays_{season_tag}.parquet",
            _load_pbp_offense_fields,
            config.stat_seasons,
            refresh,
        ),
        "pbp_defense_plays": _load_cached(
            config.cache_dir / f"pbp_defense_plays_{season_tag}.parquet",
            _load_pbp_defense_fields,
            config.stat_seasons,
            refresh,
        ),
        "depth_charts": _load_cached(
            config.cache_dir / f"depth_charts_{depth_tag}.parquet",
            nfl.load_depth_charts,
            config.depth_chart_seasons,
            refresh,
        ),
    }


def normalize_schedules(
    schedules: pl.DataFrame,
    season_type: str,
    first_eligible_season: int,
    test_season: int,
    cutoff_minutes: int,
) -> pl.DataFrame:
    games = (
        schedules.filter(
            (pl.col("game_type") == season_type) & pl.col("season").is_between(first_eligible_season, test_season)
        )
        .with_columns(
            pl.concat_str(["gameday", "gametime"], separator=" ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="America/New_York", strict=True)
            .dt.convert_time_zone("UTC")
            .alias("kickoff_utc")
        )
        .with_columns((pl.col("kickoff_utc") - pl.duration(minutes=cutoff_minutes)).alias("prediction_cutoff_utc"))
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
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent_team"),
        pl.lit(1, dtype=pl.Int8).alias("is_home"),
        pl.col("home_rest").alias("rest_days"),
    )
    away = games.select(
        *shared,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent_team"),
        pl.lit(0, dtype=pl.Int8).alias("is_home"),
        pl.col("away_rest").alias("rest_days"),
    )
    return pl.concat([home, away]).sort(["kickoff_utc", "game_id", "team"])


def _unique_starters(
    frame: pl.DataFrame,
    keys: list[str],
    id_column: str,
    name_column: str,
) -> pl.DataFrame:
    return (
        frame.filter(pl.col(id_column).is_not_null())
        .group_by(keys)
        .agg(
            pl.col(id_column).n_unique().alias("starter_candidate_count"),
            pl.col(id_column).first().alias("anticipated_qb_id"),
            pl.col(name_column).first().alias("anticipated_qb_name"),
        )
        .with_columns(
            pl.when(pl.col("starter_candidate_count") == 1)
            .then(pl.col("anticipated_qb_id"))
            .otherwise(None)
            .alias("anticipated_qb_id"),
            pl.when(pl.col("starter_candidate_count") == 1)
            .then(pl.col("anticipated_qb_name"))
            .otherwise(None)
            .alias("anticipated_qb_name"),
        )
    )


def select_anticipated_starters(schedule_rows: pl.DataFrame, depth_charts: pl.DataFrame) -> pl.DataFrame:
    # nflreadpy unions the legacy and timestamped schemas when both eras are
    # requested. Explicit casts also keep an empty half of that union joinable
    # when a caller supplies only one era (as the synthetic tests do).
    depth_charts = depth_charts.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("club_code").cast(pl.String),
        pl.col("depth_team").cast(pl.String),
        pl.col("position").cast(pl.String),
        pl.col("dt").cast(pl.String),
        pl.col("team").cast(pl.String),
        pl.col("pos_abb").cast(pl.String),
        pl.col("pos_rank").cast(pl.Int64),
    )
    legacy = depth_charts.filter(
        pl.col("dt").is_null() & (pl.col("position") == "QB") & (pl.col("depth_team").cast(pl.String) == "1")
    )
    legacy = _unique_starters(
        legacy,
        keys=["season", "week", "club_code"],
        id_column="gsis_id",
        name_column="full_name",
    ).rename({"club_code": "team"})
    legacy = legacy.with_columns(
        pl.lit("weekly_depth_chart").alias("starter_source"),
        pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("starter_snapshot_utc"),
        pl.lit(False).alias("starter_cutoff_verified"),
    )

    current = depth_charts.filter(
        pl.col("dt").is_not_null() & (pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1)
    ).with_columns(pl.col("dt").str.to_datetime(time_zone="UTC", strict=True).alias("starter_snapshot_utc"))
    current = _unique_starters(
        current,
        keys=["starter_snapshot_utc", "team"],
        id_column="gsis_id",
        name_column="player_name",
    ).with_columns(
        pl.lit("timestamped_depth_chart").alias("starter_source"),
        pl.lit(True).alias("starter_cutoff_verified"),
    )

    legacy_rows = schedule_rows.filter(pl.col("season") < 2025).join(
        legacy,
        on=["season", "week", "team"],
        how="left",
        validate="m:1",
    )
    current_rows = (
        schedule_rows.filter(pl.col("season") >= 2025)
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
    return pl.concat([legacy_rows, current_rows], how="diagonal_relaxed").sort(["kickoff_utc", "game_id", "team"])


def actual_starters_from_participation(participation: pl.DataFrame) -> pl.DataFrame:
    players = pl.col("offense_players").str.split(";")
    positions = pl.col("offense_positions").str.split(";")
    qb_appearances = (
        participation.filter(
            pl.col("nflverse_game_id").is_not_null()
            & pl.col("possession_team").is_not_null()
            & pl.col("play_id").is_not_null()
            & pl.col("offense_players").is_not_null()
            & pl.col("offense_positions").is_not_null()
        )
        .select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("possession_team").alias("team"),
            "play_id",
            players.alias("offense_player_ids"),
            positions.alias("offense_player_positions"),
        )
        .filter(pl.col("offense_player_ids").list.len() == pl.col("offense_player_positions").list.len())
        .explode(["offense_player_ids", "offense_player_positions"], empty_as_null=True)
        .filter(pl.col("offense_player_positions") == "QB")
        .select(
            "game_id",
            "team",
            "play_id",
            pl.col("offense_player_ids").alias("actual_qb_id"),
        )
        .unique()
    )
    first_qb_play = qb_appearances.group_by(["game_id", "team"]).agg(
        pl.col("play_id").min().alias("actual_starter_play_id")
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
            pl.col("play_id").first().alias("actual_starter_play_id"),
            pl.col("actual_qb_id").n_unique().alias("actual_starter_candidate_count"),
            pl.col("actual_qb_id").first(),
        )
        .with_columns(
            pl.when(pl.col("actual_starter_candidate_count") == 1)
            .then(pl.col("actual_qb_id"))
            .otherwise(None)
            .alias("actual_qb_id"),
            pl.lit("first_offensive_qb_participation").alias("actual_starter_source"),
        )
    )


def actual_starters_from_first_passer(pbp_starter_plays: pl.DataFrame) -> pl.DataFrame:
    return (
        pbp_starter_plays.filter(
            (pl.col("season_type") == "REG")
            & pl.col("game_id").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("play_id").is_not_null()
            & pl.col("passer_player_id").is_not_null()
        )
        .sort(["game_id", "posteam", "play_id"])
        .group_by(["game_id", "posteam"], maintain_order=True)
        .agg(
            pl.col("play_id").first().alias("actual_starter_play_id"),
            pl.col("passer_player_id").first().alias("actual_qb_id"),
        )
        .rename({"posteam": "team"})
        .with_columns(
            pl.when((pl.col("game_id").str.slice(0, 4).cast(pl.Int32) < 2020) & (pl.col("team") == "LV"))
            .then(pl.lit("OAK"))
            .otherwise(pl.col("team"))
            .alias("team"),
            pl.lit(1, dtype=pl.UInt32).alias("actual_starter_candidate_count"),
            pl.lit("first_passer_fallback").alias("actual_starter_source"),
        )
    )


def attach_postgame_outcomes(
    rows: pl.DataFrame,
    participation: pl.DataFrame,
    pbp_starter_plays: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    participation_starters = actual_starters_from_participation(participation).filter(
        pl.col("actual_qb_id").is_not_null()
    )
    fallback_starters = actual_starters_from_first_passer(pbp_starter_plays)
    actual_starters = pl.concat([participation_starters, fallback_starters], how="diagonal_relaxed").unique(
        ["game_id", "team"], keep="first", maintain_order=True
    )
    recorded_passers = (
        pbp_starter_plays.filter(pl.col("passer_player_id").is_not_null())
        .select(
            "game_id",
            pl.col("passer_player_id").alias("actual_qb_id"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("actual_recorded_pass"))
    )
    qb_outcomes = (
        player_stats.filter(pl.col("season_type") == "REG")
        .select(
            "game_id",
            pl.col("player_id").alias("actual_qb_id"),
            pl.col("player_display_name").alias("actual_qb_name"),
            pl.col("passing_yards").cast(pl.Float64).alias("official_passing_yards"),
        )
        .unique(["game_id", "actual_qb_id"])
    )
    return (
        rows.join(actual_starters, on=["game_id", "team"], how="left", validate="1:1")
        .join(qb_outcomes, on=["game_id", "actual_qb_id"], how="left", validate="m:1")
        .join(recorded_passers, on=["game_id", "actual_qb_id"], how="left", validate="m:1")
        .with_columns(pl.col("actual_recorded_pass").fill_null(False))
        .with_columns(
            (
                pl.col("actual_qb_id").is_not_null()
                & pl.col("official_passing_yards").is_null()
                & ~pl.col("actual_recorded_pass")
            ).alias("target_zero_no_pass_attempt")
        )
        .with_columns(
            pl.when(pl.col("target_zero_no_pass_attempt"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("official_passing_yards"))
            .alias("official_passing_yards")
        )
        .with_columns(
            (
                pl.col("anticipated_qb_id").is_not_null()
                & pl.col("actual_qb_id").is_not_null()
                & (pl.col("anticipated_qb_id") == pl.col("actual_qb_id"))
            ).alias("starter_matches_actual")
        )
        .with_columns(
            (pl.col("starter_matches_actual") & pl.col("official_passing_yards").is_not_null()).alias(
                "evaluation_eligible"
            )
        )
    )


def build_base_qb_game_table(
    schedules: pl.DataFrame,
    depth_charts: pl.DataFrame,
    participation: pl.DataFrame,
    pbp_starter_plays: pl.DataFrame,
    player_stats: pl.DataFrame,
    config: DataConfig,
) -> pl.DataFrame:
    schedule_rows = normalize_schedules(
        schedules=schedules,
        season_type=config.season_type,
        first_eligible_season=config.first_eligible_season,
        test_season=config.test_season,
        cutoff_minutes=config.prediction_cutoff_minutes,
    )
    rows = select_anticipated_starters(schedule_rows, depth_charts)
    rows = attach_postgame_outcomes(rows, participation, pbp_starter_plays, player_stats)
    return rows.with_columns(
        pl.when(pl.col("anticipated_qb_id").is_null())
        .then(pl.lit("missing_or_ambiguous_qb1"))
        .otherwise(pl.lit("eligible_starter_row"))
        .alias("row_status")
    )


def summarize_base_table(rows: pl.DataFrame) -> dict[str, int]:
    return {
        "rows": rows.height,
        "games": rows["game_id"].n_unique(),
        "eligible_starter_rows": rows.filter(pl.col("anticipated_qb_id").is_not_null()).height,
        "missing_or_ambiguous_qb1_rows": rows.filter(pl.col("anticipated_qb_id").is_null()).height,
        "cutoff_verified_rows": rows.filter(pl.col("starter_cutoff_verified")).height,
        "evaluation_eligible_rows": rows.filter(pl.col("evaluation_eligible")).height,
        "starter_mismatch_rows": rows.filter(
            pl.col("anticipated_qb_id").is_not_null() & ~pl.col("starter_matches_actual")
        ).height,
    }
