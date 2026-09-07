from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_blog_features.live_features import LIVE_PBP_COLUMNS
from nfl_blog_features.live_q1_features import build_q1_rows, q1_boundaries


def _source() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_KC_JAX"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "opponent_team": ["JAX"],
            "kickoff_utc": [datetime(2025, 9, 7, 17, tzinfo=UTC)],
            "is_home": [0],
            "actual_qb_id": ["qb-kc"],
            "actual_qb_name": ["Kansas City QB"],
            "official_passing_yards": [230.0],
            "live_qb_passing_yards": [999.0],
            "remaining_passing_yards": [-769.0],
        }
    )


def _play(play_id: float, qtr: int, timestamp: str, yards: float | None, *, passer: str | None = "qb-kc"):
    is_pass = passer is not None
    values = {
        "season": 2025,
        "season_type": "REG",
        "week": 1,
        "game_id": "2025_01_KC_JAX",
        "play_id": play_id,
        "qtr": qtr,
        "time_of_day": timestamp,
        "posteam": "KC",
        "home_team": "JAX",
        "away_team": "KC",
        "passer_player_id": passer,
        "passer_player_name": "K. QB" if passer else None,
        "pass_attempt": float(is_pass),
        "complete_pass": float(is_pass),
        "passing_yards": yards,
        "air_yards": 8.0 if is_pass else None,
        "qb_dropback": float(is_pass),
        "sack": 0.0,
        "qb_hit": 0.0,
        "cpoe": 4.0 if is_pass else None,
        "epa": 0.2,
        "total_home_score": 0.0,
        "total_away_score": 3.0 if play_id >= 20 else 0.0,
        "game_seconds_remaining": 3600.0 - play_id if qtr == 1 else 2700.0 - (play_id - 30.0),
        "quarter_seconds_remaining": 900.0 - play_id if qtr == 1 else 900.0 - (play_id - 30.0),
        "play_type": "pass" if is_pass else "run",
        "rush_attempt": float(not is_pass),
        "qb_kneel": 0.0,
        "qb_spike": 0.0,
        "interception": 0.0,
        "success": 1.0,
    }
    return values


def _pbp() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _play(10, 1, "2025-09-07T17:10:00.000Z", 12.0),
            _play(20, 1, "2025-09-07T17:35:00.000Z", None, passer=None),
            _play(30, 2, "2025-09-07T17:50:00.000Z", None, passer=None),
            _play(40, 2, "2025-09-07T17:49:00.000Z", 500.0),
        ]
    ).select(*LIVE_PBP_COLUMNS)


def test_q1_boundary_uses_earliest_q2_timestamp_not_first_play_id() -> None:
    plays = _pbp().with_columns(
        pl.col("time_of_day").str.to_datetime(time_zone="UTC").alias("_play_timestamp_utc")
    )
    boundary = q1_boundaries(plays).row(0, named=True)
    assert boundary["live_anchor_play_id"] == 40
    assert boundary["live_anchor_utc"] == datetime(2025, 9, 7, 17, 49, tzinfo=UTC)


def test_q1_rows_replace_old_live_columns_and_exclude_q2_values() -> None:
    row = build_q1_rows(_source(), _pbp()).row(0, named=True)
    anchor = datetime(2025, 9, 7, 17, 49, tzinfo=UTC)
    assert row["live_anchor_utc"] == anchor
    assert row["live_decision_utc"] == anchor + timedelta(minutes=2)
    assert row["live_qb_passing_yards"] == 12.0
    assert row["remaining_passing_yards"] == 218.0
    assert row["live_team_score"] == 3.0
    assert row["live_qb_anchor_quarter_pass_plays"] == 1.0


def test_nonpasser_bad_q1_timestamp_does_not_change_play_order_state() -> None:
    mutated = _pbp().with_columns(
        pl.when(pl.col("play_id") == 20)
        .then(pl.lit("2025-09-08T17:35:00.000Z"))
        .otherwise(pl.col("time_of_day"))
        .alias("time_of_day")
    )
    observed = build_q1_rows(_source(), mutated)
    expected = build_q1_rows(_source(), _pbp().filter(pl.col("play_id") != 20))
    live = [column for column in observed.columns if column.startswith("live_")]
    assert observed.select(live).equals(expected.select(live))


def test_q1_passer_timestamp_at_or_after_boundary_is_not_observed() -> None:
    mutated = _pbp().with_columns(
        pl.when(pl.col("play_id") == 10)
        .then(pl.lit("2025-09-07T17:50:00.000Z"))
        .otherwise(pl.col("time_of_day"))
        .alias("time_of_day")
    )
    row = build_q1_rows(_source(), mutated).row(0, named=True)
    assert row["live_qb_passing_yards"] == 0.0
    assert row["remaining_passing_yards"] == 230.0


def test_q2_anchor_clock_outside_frozen_range_fails_closed() -> None:
    mutated = _pbp().with_columns(
        pl.when(pl.col("qtr") == 2)
        .then(2300.0)
        .otherwise(pl.col("game_seconds_remaining"))
        .alias("game_seconds_remaining")
    )
    with pytest.raises(ValueError, match="boundary clock"):
        build_q1_rows(_source(), mutated)


def test_q2_outcomes_do_not_change_q1_features() -> None:
    original = build_q1_rows(_source(), _pbp())
    mutated = _pbp().with_columns(
        pl.when(pl.col("qtr") == 2).then(9999.0).otherwise(pl.col("passing_yards")).alias("passing_yards"),
        pl.when(pl.col("qtr") == 2).then(99.0).otherwise(pl.col("total_home_score")).alias("total_home_score"),
    )
    changed = build_q1_rows(_source(), mutated)
    live = [column for column in original.columns if column.startswith("live_")]
    assert original.select(live).equals(changed.select(live))
