"""Leak-safe first-quarter rows for the frozen NFL timing screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import load_config
from .live_features import (
    LIVE_FEATURE_COLUMNS,
    LIVE_PBP_COLUMNS,
    LIVE_TARGET_COLUMN,
    LIVE_USAGE_FEATURE_COLUMNS,
    _offense_aggregates,
    _qb_aggregates,
    load_live_config,
)

def _normalized_timestamped_plays(pbp: pl.DataFrame, season_type: str) -> pl.DataFrame:
    missing = sorted(set(LIVE_PBP_COLUMNS).difference(pbp.columns))
    if missing:
        raise ValueError(f"Q1 PBP is missing required columns: {missing}")
    return (
        pbp.filter(
            (pl.col("season_type") == season_type)
            & pl.col("qtr").is_in([1, 2])
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


def q1_boundaries(plays: pl.DataFrame) -> pl.DataFrame:
    """Use the earliest timestamped Q2 record, with a frozen clock guard."""
    anchors = (
        plays.filter(pl.col("qtr") == 2)
        .sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            pl.col("_play_timestamp_utc").first().alias("live_anchor_utc"),
            pl.col("play_id").first().alias("live_anchor_play_id"),
            pl.col("game_seconds_remaining").first().alias("live_anchor_game_seconds_remaining"),
        )
    )
    invalid = anchors.filter(~pl.col("live_anchor_game_seconds_remaining").is_between(2400.0, 2700.0))
    if invalid.height:
        raise ValueError(f"Q1 boundary clock is outside [2400, 2700]: {invalid.head(5).to_dicts()}")
    return anchors


def _q1_game_state(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.sort(["game_id", "_play_timestamp_utc", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            pl.col("total_home_score").drop_nulls().last().alias("_live_home_score"),
            pl.col("total_away_score").drop_nulls().last().alias("_live_away_score"),
            pl.col("game_seconds_remaining").drop_nulls().last().alias("live_game_seconds_remaining"),
        )
    )


def _q1_usage_aggregates(plays: pl.DataFrame) -> pl.DataFrame:
    passer_plays = (
        plays.filter(pl.col("passer_player_id").is_not_null() & (pl.col("qb_dropback") == 1))
        .rename({"posteam": "team"})
        .sort(["game_id", "team", "_play_timestamp_utc", "play_id"])
    )
    team_state = passer_plays.group_by("game_id", "team", maintain_order=True).agg(
        pl.col("passer_player_id").last().alias("_latest_passer_id"),
        pl.len().alias("_team_q1_pass_plays"),
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
            pl.len().cast(pl.Float64).alias("live_qb_anchor_quarter_pass_plays"),
            pl.col("_play_timestamp_utc").max().alias("_qb_last_pass_play_utc"),
            pl.col("live_anchor_utc").first(),
        )
        .join(team_state, on=["game_id", "team"], how="left")
        .join(recent, on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (pl.col("passer_player_id") == pl.col("_latest_passer_id"))
            .cast(pl.Float64)
            .alias("live_qb_is_latest_team_passer"),
            (pl.col("live_qb_anchor_quarter_pass_plays") / pl.col("_team_q1_pass_plays"))
            .alias("live_qb_anchor_quarter_pass_play_share"),
            (pl.col("_qb_recent_pass_plays").fill_null(0) / pl.col("_team_recent_pass_plays"))
            .alias("live_qb_recent_pass_play_share"),
            (pl.col("live_anchor_utc") - pl.col("_qb_last_pass_play_utc"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("live_qb_seconds_since_last_pass_play"),
        )
        .select("game_id", "team", "passer_player_id", *LIVE_USAGE_FEATURE_COLUMNS)
    )


def _pregame_columns(rows: pl.DataFrame) -> list[str]:
    return [column for column in rows.columns if not column.startswith("live_") and column != LIVE_TARGET_COLUMN]


def build_q1_rows(
    source_rows: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    season_type: str = "REG",
    decision_delay_minutes: int = 2,
) -> pl.DataFrame:
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

    plays = _normalized_timestamped_plays(pbp, season_type)
    anchors = q1_boundaries(plays)
    q1 = (
        plays.filter(pl.col("qtr") == 1)
        .join(anchors, on="game_id", how="inner")
        .filter(pl.col("_play_timestamp_utc") < pl.col("live_anchor_utc"))
    )
    missing_q1 = anchors.join(q1.select("game_id").unique(), on="game_id", how="anti")
    if missing_q1.height:
        raise ValueError(f"Q1 boundary has no preceding timestamped Q1 record: {missing_q1.head(5).to_dicts()}")
    invalid_passer_time = q1.filter(
        pl.col("passer_player_id").is_not_null()
        & (pl.col("_play_timestamp_utc") >= pl.col("live_anchor_utc"))
    )
    if invalid_passer_time.height:
        examples = invalid_passer_time.select("game_id", "play_id").head(5).to_dicts()
        raise ValueError(f"Q1 passer timestamp does not precede Q2 boundary: {examples}")

    base = source_rows.select(_pregame_columns(source_rows))
    rows = (
        base.join(anchors, on="game_id", how="left")
        .join(
            _qb_aggregates(q1),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(_offense_aggregates(q1), on=["game_id", "team"], how="left")
        .join(
            _q1_usage_aggregates(q1),
            left_on=["game_id", "team", "actual_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .join(_q1_game_state(q1), on="game_id", how="left")
        .with_columns(
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("_live_home_score"))
            .otherwise(pl.col("_live_away_score"))
            .alias("live_team_score"),
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("_live_away_score"))
            .otherwise(pl.col("_live_home_score"))
            .alias("live_opponent_score"),
            (pl.col("live_anchor_utc") + pl.duration(minutes=decision_delay_minutes)).alias("live_decision_utc"),
        )
        .with_columns(
            (pl.col("live_team_score") - pl.col("live_opponent_score")).alias("live_score_differential"),
            (pl.col("live_team_score") + pl.col("live_opponent_score")).alias("live_game_total"),
        )
        .with_columns(
            (pl.col("live_score_differential") > 0).cast(pl.Float64).alias("live_is_leading"),
            (pl.col("official_passing_yards") - pl.col("live_qb_passing_yards").fill_null(0.0))
            .alias(LIVE_TARGET_COLUMN),
            pl.col("live_anchor_utc").is_not_null().alias("live_evaluation_eligible"),
        )
        .drop("_live_home_score", "_live_away_score")
        .with_columns(
            *[
                pl.col(column).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
                for column in LIVE_FEATURE_COLUMNS + LIVE_USAGE_FEATURE_COLUMNS
            ]
        )
        .sort(key_columns)
    )
    return rows
