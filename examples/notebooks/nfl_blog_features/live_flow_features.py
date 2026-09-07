"""Append opponent and quarter-flow features to immutable halftime rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .data import load_config
from .live_features import (
    LIVE_OPPONENT_FEATURE_COLUMNS,
    LIVE_TEMPO_FEATURE_COLUMNS,
    LiveConfig,
    _timestamped_plays,
    load_live_config,
)
from .season_context_features import sha256_file

ROOT = Path(__file__).resolve().parent
FLOW_COLUMNS = (*LIVE_OPPONENT_FEATURE_COLUMNS, *LIVE_TEMPO_FEATURE_COLUMNS)
KEYS = ["game_id", "team"]


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None).cast(pl.Float64)


def build_live_flow_features(
    rows: pl.DataFrame, pbp: pl.DataFrame, live_config: LiveConfig, season_type: str
) -> pl.DataFrame:
    """Only consume plays at/before each already-frozen anchor; never targets."""
    if live_config.anchor_quarter != 2:
        raise ValueError("quarter-flow features are defined only for the halftime protocol")
    required = {*KEYS, "opponent_team", "actual_qb_id", "live_anchor_utc"}
    if required - set(rows.columns):
        raise ValueError(f"missing frozen row fields: {sorted(required - set(rows.columns))}")
    if set(FLOW_COLUMNS) & set(rows.columns):
        raise ValueError("flow columns already exist; preserve the original source table")
    if rows.select(KEYS).is_duplicated().any():
        raise ValueError("flow source requires unique QB-game rows")
    anchors = rows.select("game_id", "live_anchor_utc").unique()
    if anchors["game_id"].is_duplicated().any():
        raise ValueError("each game must have one frozen anchor")
    plays = (
        _timestamped_plays(pbp, season_type, 2)
        .join(anchors, on="game_id", how="inner")
        .filter(pl.col("_play_timestamp_utc") <= pl.col("live_anchor_utc"))
        .filter(pl.col("posteam").is_not_null())
        .filter(
            pl.col("play_type").is_in(["pass", "run"])
            & (pl.col("qb_kneel").fill_null(0) != 1)
            & (pl.col("qb_spike").fill_null(0) != 1)
        )
        .rename({"posteam": "team"})
    )
    q2 = pl.col("qtr") == 2
    valid_clock = pl.col("quarter_seconds_remaining").is_between(0, 900).fill_null(False)
    two_minute = q2 & valid_clock & (pl.col("quarter_seconds_remaining") <= 120)
    before_two_minute = (~q2) | (valid_clock & (pl.col("quarter_seconds_remaining") > 120))
    dropback = pl.col("qb_dropback").fill_null(0)
    team = (
        plays.group_by(KEYS)
        .agg(
            pl.len().cast(pl.Float64).alias("_plays"),
            dropback.mean().alias("_pass_rate"),
            pl.col("epa").mean().alias("_epa"),
            pl.col("success").mean().alias("_success"),
            pl.col("passing_yards").fill_null(0).sum().alias("_passing_yards"),
            pl.col("interception").fill_null(0).sum().alias("_interceptions"),
            q2.cast(pl.Float64).sum().alias("_q2_plays"),
            dropback.filter(q2).mean().alias("live_offense_q2_pass_rate"),
            dropback.filter(~q2).mean().alias("_q1_pass_rate"),
        )
        .with_columns(
            _ratio(pl.col("_q2_plays"), pl.col("_plays")).alias("live_offense_q2_play_share"),
            (pl.col("live_offense_q2_pass_rate") - pl.col("_q1_pass_rate")).alias("live_offense_q2_minus_q1_pass_rate"),
        )
    )
    opponent = team.select(
        "game_id",
        pl.col("team").alias("opponent_team"),
        pl.col("_plays").alias("live_opponent_offense_plays"),
        pl.col("_pass_rate").alias("live_opponent_pass_rate"),
        pl.col("_epa").alias("live_opponent_epa_per_play"),
        pl.col("_success").alias("live_opponent_success_rate"),
        pl.col("_passing_yards").alias("live_opponent_passing_yards"),
        pl.col("_interceptions").alias("live_opponent_interceptions"),
    )
    qb = (
        plays.filter((dropback == 1) & pl.col("passer_player_id").is_not_null())
        .group_by(*KEYS, pl.col("passer_player_id").alias("actual_qb_id"))
        .agg(
            pl.len().cast(pl.Float64).alias("_qb_dropbacks"),
            q2.cast(pl.Float64).sum().alias("_qb_q2_dropbacks"),
            (q2 & ~valid_clock).cast(pl.Float64).sum().alias("_qb_bad_q2_clock"),
            two_minute.fill_null(False).cast(pl.Float64).sum().alias("_qb_two_minute_dropbacks"),
            pl.col("passing_yards").fill_null(0).filter(q2).mean().alias("live_qb_q2_yards_per_dropback"),
            pl.col("passing_yards").fill_null(0).filter(~q2).mean().alias("_qb_q1_yards_per_dropback"),
            pl.col("epa").filter(q2).mean().alias("live_qb_q2_epa_per_dropback"),
            pl.col("sack").filter(q2).mean().alias("live_qb_q2_sack_rate"),
            pl.col("passing_yards").fill_null(0).filter(before_two_minute).mean().alias("_qb_before_two_minute_ypdb"),
        )
        .with_columns(
            (pl.col("live_qb_q2_yards_per_dropback") - pl.col("_qb_q1_yards_per_dropback")).alias(
                "live_qb_q2_minus_q1_yards_per_dropback"
            ),
            pl.when(pl.col("_qb_bad_q2_clock") == 0)
            .then(_ratio(pl.col("_qb_two_minute_dropbacks"), pl.col("_qb_dropbacks")))
            .otherwise(None)
            .alias("live_qb_two_minute_dropback_share"),
            pl.when(pl.col("_qb_bad_q2_clock") == 0)
            .then(pl.col("_qb_before_two_minute_ypdb"))
            .otherwise(None)
            .alias("live_qb_before_two_minute_yards_per_dropback"),
            pl.when(pl.col("_qb_q2_dropbacks") > 0)
            .then(1.0 - pl.col("_qb_bad_q2_clock") / pl.col("_qb_q2_dropbacks"))
            .otherwise(1.0)
            .alias("live_qb_two_minute_clock_coverage"),
        )
        .select(*KEYS, "actual_qb_id", *[c for c in LIVE_TEMPO_FEATURE_COLUMNS if c.startswith("live_qb_")])
    )
    team_tempo = [c for c in LIVE_TEMPO_FEATURE_COLUMNS if c.startswith("live_offense_")]
    result = (
        rows.join(team.select(*KEYS, "_plays", *team_tempo), on=KEYS, how="left", validate="1:1")
        .join(opponent, on=["game_id", "opponent_team"], how="left", validate="m:1")
        .join(qb, on=[*KEYS, "actual_qb_id"], how="left", validate="1:1")
        .with_columns((pl.col("_plays") + pl.col("live_opponent_offense_plays")).alias("live_game_offense_plays"))
        .with_columns(_ratio(pl.col("_plays"), pl.col("live_game_offense_plays")).alias("live_team_play_share"))
        .with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in FLOW_COLUMNS])
        .sort(KEYS)
    )
    if (
        "live_offense_plays" in rows.columns
        and result.filter(
            pl.col("live_anchor_utc").is_not_null() & (pl.col("_plays").fill_null(0) != pl.col("live_offense_plays"))
        ).height
    ):
        raise ValueError("source halftime aggregates are stale; rebuild the base table before adding flow features")
    result = result.drop("_plays")
    if not result.select(rows.columns).equals(rows.sort(KEYS)):
        raise RuntimeError("flow join changed the source rows or existing feature values")
    return result
