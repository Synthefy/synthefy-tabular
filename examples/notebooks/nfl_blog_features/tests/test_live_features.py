from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl_blog_features.data import DataConfig
from nfl_blog_features.live_features import (
    LIVE_FEATURE_COLUMNS,
    LIVE_PBP_COLUMNS,
    LIVE_PLAYSTATS_FEATURE_COLUMNS,
    LIVE_USAGE_FEATURE_COLUMNS,
    LiveConfig,
    build_live_rows,
    live_feature_columns,
)


def _data_config() -> DataConfig:
    return DataConfig(
        warmup_start_season=2016,
        first_eligible_season=2018,
        validation_season=2024,
        test_season=2025,
        season_type="REG",
        prediction_cutoff_minutes=60,
        cache_dir=Path("data/raw"),
        base_dataset_path=Path("outputs/base.parquet"),
        feature_dataset_path=Path("outputs/features.parquet"),
        qb_rolling_windows=(3, 8),
        offense_rolling_windows=(3, 8),
        neutral_wp_lower=0.20,
        neutral_wp_upper=0.80,
        neutral_max_quarter=3,
        include_starter_mismatch_trades=True,
    )


def _live_config() -> LiveConfig:
    return LiveConfig(
        anchor_quarter=2,
        decision_delay_minutes=2,
        maximum_quote_age_minutes=5,
        pregame_feature_set="qb_offense_v0",
        pbp_path=Path("data/raw/live.parquet"),
        feature_dataset_path=Path("outputs/live.parquet"),
        candlestick_cache_dir=Path("data/raw/candles"),
        validation_predictions_path=Path("outputs/live_2024.parquet"),
        validation_metrics_path=Path("outputs/live_2024.json"),
        test_predictions_path=Path("outputs/live_2025.parquet"),
        test_metrics_path=Path("outputs/live_2025.json"),
    )


def _pregame_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_KC_JAX"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "is_home": [0],
            "actual_qb_id": ["qb-kc"],
            "actual_qb_name": ["Kansas City QB"],
            "official_passing_yards": [230.0],
        }
    )


def _play(
    play_id: float,
    qtr: float,
    timestamp: str,
    *,
    play_type: str,
    passer_id: str | None,
    passing_yards: float | None,
    home_score: float,
    away_score: float,
) -> dict[str, object]:
    is_pass = play_type == "pass"
    return {
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
        "passer_player_id": passer_id,
        "passer_player_name": "K. Quarterback" if passer_id else None,
        "pass_attempt": 1.0 if is_pass else 0.0,
        "complete_pass": 1.0 if is_pass else 0.0,
        "passing_yards": passing_yards,
        "air_yards": 8.0 if is_pass else None,
        "qb_dropback": 1.0 if is_pass else 0.0,
        "sack": 0.0,
        "qb_hit": 0.0,
        "cpoe": 5.0 if is_pass else None,
        "epa": 0.3,
        "total_home_score": home_score,
        "total_away_score": away_score,
        "game_seconds_remaining": 3600.0 - play_id,
        "quarter_seconds_remaining": 900.0 - (play_id % 900),
        "play_type": play_type,
        "rush_attempt": 1.0 if play_type == "run" else 0.0,
        "qb_kneel": 0.0,
        "qb_spike": 0.0,
        "interception": 0.0,
        "success": 1.0,
    }


def _pbp() -> pl.DataFrame:
    rows = [
        _play(
            10.0,
            1.0,
            "2025-09-07T17:10:00.000Z",
            play_type="pass",
            passer_id="qb-kc",
            passing_yards=10.0,
            home_score=0.0,
            away_score=0.0,
        ),
        _play(
            50.0,
            2.0,
            "2025-09-07T17:50:00.000Z",
            play_type="pass",
            passer_id="qb-kc",
            passing_yards=20.0,
            home_score=7.0,
            away_score=3.0,
        ),
        _play(
            80.0,
            2.0,
            "2025-09-07T18:00:00.000Z",
            play_type="run",
            passer_id=None,
            passing_yards=None,
            home_score=7.0,
            away_score=10.0,
        ),
        _play(
            100.0,
            3.0,
            "2025-09-07T18:20:00.000Z",
            play_type="pass",
            passer_id="qb-kc",
            passing_yards=200.0,
            home_score=14.0,
            away_score=17.0,
        ),
    ]
    return pl.DataFrame(rows).select(*LIVE_PBP_COLUMNS)


def test_halftime_features_exclude_third_quarter_plays() -> None:
    rows = build_live_rows(_pregame_rows(), _pbp(), _data_config(), _live_config())
    row = rows.row(0, named=True)

    assert row["live_qb_attempts"] == 2.0
    assert row["live_qb_passing_yards"] == 30.0
    assert row["remaining_passing_yards"] == 200.0
    assert row["live_team_score"] == 10.0
    assert row["live_opponent_score"] == 7.0
    assert row["live_score_differential"] == 3.0


def test_anchor_and_decision_timestamp_use_last_second_quarter_play() -> None:
    rows = build_live_rows(_pregame_rows(), _pbp(), _data_config(), _live_config())
    row = rows.row(0, named=True)
    expected_anchor = datetime(2025, 9, 7, 18, 0, tzinfo=UTC)

    assert row["live_anchor_utc"] == expected_anchor
    assert row["live_anchor_play_id"] == 80.0
    assert row["live_decision_utc"] == expected_anchor + timedelta(minutes=2)
    assert row["live_evaluation_eligible"] is True


def test_live_features_are_numeric_and_identifiers_are_not_features() -> None:
    rows = build_live_rows(_pregame_rows(), _pbp(), _data_config(), _live_config())
    values = rows.select(*LIVE_FEATURE_COLUMNS).row(0)

    assert all(value is not None for value in values)
    assert all(isinstance(value, float) for value in values)
    assert {"game_id", "team", "actual_qb_id"}.isdisjoint(LIVE_FEATURE_COLUMNS)


def test_missing_anchor_fails_closed() -> None:
    first_quarter_only = _pbp().filter(pl.col("qtr") == 1)
    rows = build_live_rows(_pregame_rows(), first_quarter_only, _data_config(), _live_config())
    row = rows.row(0, named=True)

    assert row["live_anchor_utc"] is None
    assert row["live_decision_utc"] is None
    assert row["live_evaluation_eligible"] is False


def test_usage_group_observes_replacement_before_halftime() -> None:
    pbp = _pbp().with_columns(
        pl.when(pl.col("play_id") == 80.0)
        .then(pl.lit("qb-backup"))
        .otherwise(pl.col("passer_player_id"))
        .alias("passer_player_id"),
        pl.when(pl.col("play_id") == 80.0).then(1.0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"),
    )
    row = build_live_rows(_pregame_rows(), pbp, _data_config(), _live_config()).row(0, named=True)

    assert row["live_qb_is_latest_team_passer"] == 0.0
    assert row["live_qb_anchor_quarter_pass_plays"] == 1.0
    assert row["live_qb_anchor_quarter_pass_play_share"] == 0.5
    assert row["live_qb_recent_pass_play_share"] == 2.0 / 3.0
    assert row["live_qb_seconds_since_last_pass_play"] == 600.0


def test_historical_oak_rows_match_lv_play_codes_without_changing_game_identity() -> None:
    pregame = _pregame_rows().with_columns(
        pl.lit(2019).alias("season"), pl.lit("OAK").alias("team"), pl.lit("2019_01_DEN_OAK").alias("game_id")
    )
    plays = _pbp().with_columns(
        pl.lit(2019).alias("season"), pl.lit("LV").alias("posteam"), pl.lit("2019_01_DEN_OAK").alias("game_id")
    )
    row = build_live_rows(pregame, plays, _data_config(), _live_config()).row(0, named=True)
    assert row["team"] == "OAK" and row["game_id"] == "2019_01_DEN_OAK"
    assert row["live_offense_plays"] == 3
    assert row["live_qb_passing_yards"] == 30
    assert row["remaining_passing_yards"] == 200
    assert row["live_qb_is_latest_team_passer"] == 1


def test_playstats_removes_exactly_six_live_features_and_retains_usage() -> None:
    old = live_feature_columns(_data_config(), _live_config(), "usage")
    new = live_feature_columns(_data_config(), _live_config(), "playstats")
    removed = {
        "live_qb_epa_per_dropback",
        "live_qb_cpoe",
        "live_qb_air_yards_per_attempt",
        "live_qb_hit_rate",
        "live_offense_epa_per_play",
        "live_offense_success_rate",
    }
    assert set(old) - set(new) == removed
    assert new == [column for column in old if column not in removed]
    assert len(LIVE_PLAYSTATS_FEATURE_COLUMNS) == 22
    assert len(new) == len(set(new))
    assert set(LIVE_USAGE_FEATURE_COLUMNS).issubset(new)
    assert [column for column in old if not column.startswith("live_")] == [
        column for column in new if not column.startswith("live_")
    ]


def test_playstats_does_not_depend_on_removed_play_fields() -> None:
    original = build_live_rows(_pregame_rows(), _pbp(), _data_config(), _live_config())
    mutated = _pbp().with_columns(
        *[pl.lit(999.0).alias(column) for column in ("epa", "cpoe", "air_yards", "qb_hit", "success")]
    )
    changed = build_live_rows(_pregame_rows(), mutated, _data_config(), _live_config())
    assert changed.select(LIVE_PLAYSTATS_FEATURE_COLUMNS).equals(original.select(LIVE_PLAYSTATS_FEATURE_COLUMNS))


def test_playstats_still_excludes_post_anchor_information() -> None:
    original = build_live_rows(_pregame_rows(), _pbp(), _data_config(), _live_config())
    mutated = _pbp().with_columns(
        pl.when(pl.col("qtr") > 2).then(9999.0).otherwise(pl.col("passing_yards")).alias("passing_yards"),
        pl.when(pl.col("qtr") > 2)
        .then(pl.lit("another-qb"))
        .otherwise(pl.col("passer_player_id"))
        .alias("passer_player_id"),
    )
    changed = build_live_rows(_pregame_rows(), mutated, _data_config(), _live_config())
    assert changed.select(LIVE_PLAYSTATS_FEATURE_COLUMNS).equals(original.select(LIVE_PLAYSTATS_FEATURE_COLUMNS))


def test_playstats_does_not_change_default_or_accept_unknown_groups() -> None:
    default = live_feature_columns(_data_config(), _live_config())
    assert default == live_feature_columns(_data_config(), _live_config(), "base")
    assert default[-len(LIVE_FEATURE_COLUMNS) :] == list(LIVE_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="unknown"):
        live_feature_columns(_data_config(), _live_config(), "unknown")
