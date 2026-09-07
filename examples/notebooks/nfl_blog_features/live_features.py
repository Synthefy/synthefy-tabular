"""Leak-safe in-game features for NFL quarterback passing-yard forecasts.

The primary live protocol makes one decision two minutes after the final
second-quarter play recorded by nflverse.  Every aggregate in this module is
restricted to a play timestamp at or before that anchor.  The model predicts
yards remaining for the named quarterback; current yards are added back to
recover the final-yard distribution used to price Kalshi thresholds.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from .data import DataConfig, load_config
from .derived_features import MATCHUP_FEATURE_COLUMNS, TREND_FEATURE_COLUMNS, build_derived_features
from .feature_sets import feature_columns_for_set
from .features import build_qb_rolling_features, qb_feature_columns

ROOT = Path(__file__).resolve().parent
LIVE_TARGET_COLUMN = "remaining_passing_yards"

LIVE_PBP_COLUMNS = (
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

LIVE_FEATURE_COLUMNS = (
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

LIVE_USAGE_FEATURE_COLUMNS = (
    "live_qb_is_latest_team_passer",
    "live_qb_anchor_quarter_pass_plays",
    "live_qb_anchor_quarter_pass_play_share",
    "live_qb_recent_pass_play_share",
    "live_qb_seconds_since_last_pass_play",
)

LIVE_BOXSCORE_FEATURE_COLUMNS = (
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

# Basic play/box statistics only: retain usage and pressure through sacks,
# but do not require live model-derived EPA/CPOE or charted air yards/hits.
# This is an experiment contract, not a claim that a live adapter is validated.
LIVE_PLAYSTATS_FEATURE_COLUMNS = (
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
) + LIVE_USAGE_FEATURE_COLUMNS

LIVE_OPPONENT_FEATURE_COLUMNS = (
    "live_opponent_offense_plays",
    "live_opponent_pass_rate",
    "live_opponent_epa_per_play",
    "live_opponent_success_rate",
    "live_opponent_passing_yards",
    "live_opponent_interceptions",
    "live_game_offense_plays",
    "live_team_play_share",
)

LIVE_TEMPO_FEATURE_COLUMNS = (
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

LIVE_DRIVE_RECEIVER_FEATURE_COLUMNS = (
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


@dataclass(frozen=True)
class LiveConfig:
    anchor_quarter: int
    decision_delay_minutes: int
    maximum_quote_age_minutes: int
    pregame_feature_set: str
    pbp_path: Path
    feature_dataset_path: Path
    candlestick_cache_dir: Path
    validation_predictions_path: Path
    validation_metrics_path: Path
    test_predictions_path: Path
    test_metrics_path: Path


def load_live_config(path: Path) -> LiveConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)["live"]
    root = path.parent
    config = LiveConfig(
        anchor_quarter=int(raw["anchor_quarter"]),
        decision_delay_minutes=int(raw["decision_delay_minutes"]),
        maximum_quote_age_minutes=int(raw["maximum_quote_age_minutes"]),
        pregame_feature_set=str(raw["pregame_feature_set"]),
        pbp_path=root / raw["pbp_path"],
        feature_dataset_path=root / raw["feature_dataset_path"],
        candlestick_cache_dir=root / raw["candlestick_cache_dir"],
        validation_predictions_path=root / raw["validation_predictions_path"],
        validation_metrics_path=root / raw["validation_metrics_path"],
        test_predictions_path=root / raw["test_predictions_path"],
        test_metrics_path=root / raw["test_metrics_path"],
    )
    if config.anchor_quarter not in {1, 2, 3}:
        raise ValueError("live.anchor_quarter must be 1, 2, or 3")
    if config.decision_delay_minutes < 0:
        raise ValueError("live.decision_delay_minutes must be non-negative")
    if config.maximum_quote_age_minutes < 0:
        raise ValueError("live.maximum_quote_age_minutes must be non-negative")
    return config


def live_feature_columns(
    data_config: DataConfig,
    live_config: LiveConfig,
    live_feature_group: str = "base",
) -> list[str]:
    live_columns = {
        "boxscore": LIVE_BOXSCORE_FEATURE_COLUMNS,
        "playstats": LIVE_PLAYSTATS_FEATURE_COLUMNS,
    }.get(live_feature_group, LIVE_FEATURE_COLUMNS)
    columns = feature_columns_for_set(data_config, live_config.pregame_feature_set) + list(live_columns)
    if live_feature_group in {"usage", "opponent", "tempo", "drive_receiver"}:
        columns.extend(LIVE_USAGE_FEATURE_COLUMNS)
        if live_feature_group == "opponent":
            columns.extend(LIVE_OPPONENT_FEATURE_COLUMNS)
        elif live_feature_group == "tempo":
            columns.extend(LIVE_TEMPO_FEATURE_COLUMNS)
        elif live_feature_group == "drive_receiver":
            columns.extend(LIVE_DRIVE_RECEIVER_FEATURE_COLUMNS)
    elif live_feature_group not in {"base", "boxscore", "playstats"}:
        raise ValueError(f"unknown live feature group: {live_feature_group}")
    if len(columns) != len(set(columns)):
        raise ValueError("live feature set contains duplicate columns")
    identifier_columns = {"game_id", "team", "actual_qb_id", "actual_qb_name", "play_id"}
    leaked = sorted(identifier_columns.intersection(columns))
    if leaked:
        raise ValueError(f"identifier columns are not model features: {leaked}")
    return columns


def _download_live_pbp(seasons: list[int]) -> pl.DataFrame:
    frames = []
    for season in seasons:
        frame = nfl.load_pbp([season]).select(*LIVE_PBP_COLUMNS)
        frames.append(frame)
        print(json.dumps({"downloaded_live_pbp_season": season, "rows": frame.height}), flush=True)
    return pl.concat(frames, how="diagonal_relaxed")


def load_live_pbp(data_config: DataConfig, live_config: LiveConfig, refresh: bool = False) -> pl.DataFrame:
    if live_config.pbp_path.exists() and not refresh:
        return pl.read_parquet(live_config.pbp_path)
    live_config.pbp_path.parent.mkdir(parents=True, exist_ok=True)
    seasons = list(range(data_config.first_eligible_season, data_config.test_season + 1))
    frame = _download_live_pbp(seasons)
    frame.write_parquet(live_config.pbp_path)
    return frame


def _timestamped_plays(pbp: pl.DataFrame, season_type: str, anchor_quarter: int) -> pl.DataFrame:
    required = set(LIVE_PBP_COLUMNS)
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"live PBP is missing required columns: {missing}")
    return (
        pbp.filter(
            (pl.col("season_type") == season_type)
            & pl.col("qtr").is_between(1, anchor_quarter)
            & pl.col("time_of_day").is_not_null()
        )
        .with_columns(
            pl.col("qtr").cast(pl.Int32),
            pl.col("time_of_day").str.to_datetime(time_zone="UTC", strict=True).alias("_play_timestamp_utc"),
            *[
                pl.when(pl.col(column).is_in(["OAK", "LV"]))
                .then(pl.when(pl.col("season") < 2020).then(pl.lit("OAK")).otherwise(pl.lit("LV")))
                .otherwise(pl.col(column))
                .alias(column)
                for column in ("posteam", "home_team", "away_team")
            ],
        )
        .filter(pl.col("_play_timestamp_utc").is_not_null())
    )


def _anchor_rows(plays: pl.DataFrame, anchor_quarter: int) -> pl.DataFrame:
    return (
        plays.filter(pl.col("qtr") == anchor_quarter)
        .sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            pl.col("_play_timestamp_utc").last().alias("live_anchor_utc"),
            pl.col("play_id").last().alias("live_anchor_play_id"),
        )
    )


def _qb_aggregates(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.filter(pl.col("passer_player_id").is_not_null())
        .group_by("game_id", pl.col("posteam").alias("team"), "passer_player_id")
        .agg(
            pl.col("pass_attempt").fill_null(0).sum().alias("live_qb_attempts"),
            pl.col("complete_pass").fill_null(0).sum().alias("live_qb_completions"),
            pl.col("passing_yards").fill_null(0).sum().alias("live_qb_passing_yards"),
            pl.col("qb_dropback").fill_null(0).sum().alias("live_qb_dropbacks"),
            pl.col("sack").fill_null(0).sum().alias("_live_qb_sacks"),
            pl.col("qb_hit").fill_null(0).sum().alias("_live_qb_hits"),
            pl.col("epa").filter(pl.col("qb_dropback") == 1).mean().alias("live_qb_epa_per_dropback"),
            pl.col("cpoe").filter(pl.col("pass_attempt") == 1).mean().alias("live_qb_cpoe"),
            pl.col("air_yards").filter(pl.col("pass_attempt") == 1).mean().alias("live_qb_air_yards_per_attempt"),
            ((pl.col("complete_pass") == 1) & (pl.col("passing_yards") >= 20))
            .cast(pl.Float64)
            .sum()
            .alias("_live_qb_explosive_completions"),
            pl.col("interception").fill_null(0).sum().alias("live_qb_interceptions"),
        )
        .with_columns(
            (pl.col("live_qb_dropbacks") > 0).cast(pl.Float64).alias("live_qb_has_dropback"),
            pl.when(pl.col("live_qb_attempts") > 0)
            .then(pl.col("live_qb_passing_yards") / pl.col("live_qb_attempts"))
            .otherwise(0.0)
            .alias("live_qb_ypa"),
            pl.when(pl.col("live_qb_dropbacks") > 0)
            .then(pl.col("_live_qb_sacks") / pl.col("live_qb_dropbacks"))
            .otherwise(0.0)
            .alias("live_qb_sack_rate"),
            pl.when(pl.col("live_qb_dropbacks") > 0)
            .then(pl.col("_live_qb_hits") / pl.col("live_qb_dropbacks"))
            .otherwise(0.0)
            .alias("live_qb_hit_rate"),
            pl.when(pl.col("live_qb_attempts") > 0)
            .then(pl.col("_live_qb_explosive_completions") / pl.col("live_qb_attempts"))
            .otherwise(0.0)
            .alias("live_qb_explosive_complete_rate"),
        )
        .drop("_live_qb_sacks", "_live_qb_hits", "_live_qb_explosive_completions")
    )


def _offense_aggregates(plays: pl.DataFrame) -> pl.DataFrame:
    eligible_play = (
        pl.col("play_type").is_in(["pass", "run"])
        & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1)
    )
    return (
        plays.group_by("game_id", pl.col("posteam").alias("team"))
        .agg(
            eligible_play.cast(pl.Float64).sum().alias("live_offense_plays"),
            pl.col("qb_dropback").fill_null(0).filter(eligible_play).sum().alias("_live_offense_dropbacks"),
            pl.col("epa").filter(eligible_play).mean().alias("live_offense_epa_per_play"),
            pl.col("success").filter(eligible_play).mean().alias("live_offense_success_rate"),
        )
        .with_columns(
            pl.when(pl.col("live_offense_plays") > 0)
            .then(pl.col("_live_offense_dropbacks") / pl.col("live_offense_plays"))
            .otherwise(0.0)
            .alias("live_offense_pass_rate")
        )
        .drop("_live_offense_dropbacks")
    )


def _game_state(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            pl.col("total_home_score").drop_nulls().last().alias("_live_home_score"),
            pl.col("total_away_score").drop_nulls().last().alias("_live_away_score"),
            pl.col("game_seconds_remaining").drop_nulls().last().alias("live_game_seconds_remaining"),
        )
    )


def _usage_aggregates(plays: pl.DataFrame, anchor_quarter: int) -> pl.DataFrame:
    """Observable passer replacement/workload signals, not postgame injury labels."""

    passer_plays = (
        plays.filter(pl.col("passer_player_id").is_not_null() & (pl.col("qb_dropback") == 1))
        .rename({"posteam": "team"})
        .sort(["game_id", "team", "_play_timestamp_utc", "play_id"])
    )
    team_state = passer_plays.group_by("game_id", "team", maintain_order=True).agg(
        pl.col("passer_player_id").last().alias("_latest_passer_id"),
        (pl.col("qtr") == anchor_quarter).sum().alias("_team_anchor_quarter_pass_plays"),
        pl.len().clip(upper_bound=10).alias("_team_recent_pass_plays"),
    )
    recent = (
        passer_plays.group_by("game_id", "team", maintain_order=True)
        .tail(10)
        .group_by("game_id", "team", "passer_player_id")
        .agg(pl.len().alias("_qb_recent_pass_plays"))
    )
    return (
        passer_plays.group_by("game_id", "team", "passer_player_id")
        .agg(
            (pl.col("qtr") == anchor_quarter).cast(pl.Float64).sum().alias("live_qb_anchor_quarter_pass_plays"),
            pl.col("_play_timestamp_utc").max().alias("_qb_last_pass_play_utc"),
            pl.col("live_anchor_utc").first(),
        )
        .join(team_state, on=["game_id", "team"], how="left")
        .join(recent, on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (pl.col("passer_player_id") == pl.col("_latest_passer_id"))
            .cast(pl.Float64)
            .alias("live_qb_is_latest_team_passer"),
            pl.when(pl.col("_team_anchor_quarter_pass_plays") > 0)
            .then(pl.col("live_qb_anchor_quarter_pass_plays") / pl.col("_team_anchor_quarter_pass_plays"))
            .otherwise(0.0)
            .alias("live_qb_anchor_quarter_pass_play_share"),
            (pl.col("_qb_recent_pass_plays").fill_null(0) / pl.col("_team_recent_pass_plays")).alias(
                "live_qb_recent_pass_play_share"
            ),
            (pl.col("live_anchor_utc") - pl.col("_qb_last_pass_play_utc"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("live_qb_seconds_since_last_pass_play"),
        )
        .select("game_id", "team", "passer_player_id", *LIVE_USAGE_FEATURE_COLUMNS)
    )


def build_live_rows(
    pregame_rows: pl.DataFrame,
    pbp: pl.DataFrame,
    data_config: DataConfig,
    live_config: LiveConfig,
) -> pl.DataFrame:
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

    plays = _timestamped_plays(pbp, data_config.season_type, live_config.anchor_quarter)
    anchors = _anchor_rows(plays, live_config.anchor_quarter)
    bounded = plays.join(anchors, on="game_id", how="inner").filter(
        pl.col("_play_timestamp_utc") <= pl.col("live_anchor_utc")
    )

    rows = (
        pregame_rows.filter(
            pl.col("season").is_between(data_config.first_eligible_season, data_config.test_season)
            & pl.col("actual_qb_id").is_not_null()
            & pl.col("official_passing_yards").is_not_null()
        )
        .join(anchors, on="game_id", how="left")
        .join(
            _qb_aggregates(bounded),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(_offense_aggregates(bounded), on=["game_id", "team"], how="left")
        .join(
            _usage_aggregates(bounded, live_config.anchor_quarter),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(_game_state(bounded), on="game_id", how="left")
        .with_columns(
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("_live_home_score"))
            .otherwise(pl.col("_live_away_score"))
            .alias("live_team_score"),
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("_live_away_score"))
            .otherwise(pl.col("_live_home_score"))
            .alias("live_opponent_score"),
            (pl.col("live_anchor_utc") + pl.duration(minutes=live_config.decision_delay_minutes)).alias(
                "live_decision_utc"
            ),
        )
        .with_columns(
            (pl.col("live_team_score") - pl.col("live_opponent_score")).alias("live_score_differential"),
            (pl.col("live_team_score") + pl.col("live_opponent_score")).alias("live_game_total"),
        )
        .with_columns(
            (pl.col("live_score_differential") > 0).cast(pl.Float64).alias("live_is_leading"),
            (pl.col("official_passing_yards") - pl.col("live_qb_passing_yards").fill_null(0)).alias(LIVE_TARGET_COLUMN),
            pl.col("live_anchor_utc").is_not_null().alias("live_evaluation_eligible"),
        )
        .drop("_live_home_score", "_live_away_score")
    )
    rows = rows.with_columns(
        [
            pl.col(column).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            for column in LIVE_FEATURE_COLUMNS + LIVE_USAGE_FEATURE_COLUMNS
        ]
    )
    return rows.sort(["season", "week", "game_id", "team"])
