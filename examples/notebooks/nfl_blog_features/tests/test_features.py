from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl_blog_features.data import DataConfig
from nfl_blog_features.features import build_qb_rolling_features


def _config() -> DataConfig:
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


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2024_18_A_B", "2025_01_A_B", "2025_02_A_B", "2025_03_A_B"],
            "season": [2024, 2025, 2025, 2025],
            "week": [18, 1, 2, 3],
            "game_type": ["REG"] * 4,
            "gameday": ["2025-01-05", "2025-09-07", "2025-09-14", "2025-09-21"],
            "gametime": ["13:00"] * 4,
        }
    )


def _player_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["qb-1", "qb-1", "qb-1", "qb-1"],
            "game_id": ["2024_18_A_B", "2025_01_A_B", "2025_02_A_B", "2025_03_A_B"],
            "season": [2024, 2025, 2025, 2025],
            "week": [18, 1, 2, 3],
            "season_type": ["REG"] * 4,
            "position": ["QB"] * 4,
            "attempts": [30, 10, 40, 99],
            "passing_yards": [240, 100, 200, 999],
            "passing_air_yards": [210, 120, 300, 999],
            "sacks_suffered": [3, 2, 5, 9],
            "passing_epa": [6.0, 5.0, 10.0, 99.0],
            "passing_cpoe": [4.0, 2.0, 10.0, 99.0],
        }
    )


def _base_row(season: int, week: int, game_id: str, kickoff: datetime, qb_id: str = "qb-1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [game_id],
            "season": [season],
            "week": [week],
            "kickoff_utc": [kickoff],
            "team": ["A"],
            "anticipated_qb_id": [qb_id],
        }
    )


def test_current_week_is_excluded_and_rates_use_ratio_of_sums() -> None:
    base = _base_row(2025, 3, "2025_03_A_B", datetime(2025, 9, 21, 17, tzinfo=UTC))
    row = build_qb_rolling_features(base, _player_stats(), _schedules(), _config()).row(0, named=True)

    assert row["qb_passing_yards_lag1"] == 200.0
    assert row["qb_history_games_last3"] == 3.0
    assert row["qb_attempts_avg_last3"] == 80.0 / 3.0
    assert row["qb_passing_yards_avg_last3"] == 540.0 / 3.0
    assert row["qb_ypa_last3"] == 540.0 / 80.0
    assert row["qb_cpoe_last3"] == (4.0 * 30 + 2.0 * 10 + 10.0 * 40) / 80.0
    assert row["qb_days_since_previous_game"] == 7.0


def test_season_features_reset_but_trailing_features_cross_seasons() -> None:
    base = _base_row(2025, 1, "2025_01_A_B", datetime(2025, 9, 7, 17, tzinfo=UTC))
    row = build_qb_rolling_features(base, _player_stats(), _schedules(), _config()).row(0, named=True)

    assert row["qb_passing_yards_lag1"] == 240.0
    assert row["qb_ypa_last3"] == 8.0
    assert row["qb_history_games_season"] == 0.0
    assert row["qb_ypa_season"] is None


def test_rookie_keeps_missing_values_and_zero_history_counts() -> None:
    base = _base_row(
        2025,
        1,
        "2025_01_A_B",
        datetime(2025, 9, 7, 17, tzinfo=UTC),
        qb_id="rookie",
    )
    row = build_qb_rolling_features(base, _player_stats(), _schedules(), _config()).row(0, named=True)

    assert row["qb_passing_yards_lag1"] is None
    assert row["qb_history_games_last3"] == 0.0
    assert row["qb_history_games_last8"] == 0.0
    assert row["qb_history_games_season"] == 0.0


def test_live_history_uses_observed_starter_when_anticipated_starter_differs() -> None:
    base = _base_row(
        2025,
        3,
        "2025_03_A_B",
        datetime(2025, 9, 21, 17, tzinfo=UTC),
        qb_id="anticipated-but-not-playing",
    ).with_columns(pl.lit("qb-1").alias("actual_qb_id"))
    row = build_qb_rolling_features(
        base,
        _player_stats(),
        _schedules(),
        _config(),
        qb_id_column="actual_qb_id",
    ).row(0, named=True)

    assert row["anticipated_qb_id"] == "anticipated-but-not-playing"
    assert row["actual_qb_id"] == "qb-1"
    assert row["qb_passing_yards_lag1"] == 200.0
    assert row["qb_previous_game_id"] == "2025_02_A_B"
