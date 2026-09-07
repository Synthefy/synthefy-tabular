from __future__ import annotations

import polars as pl
import pytest

from nfl_blog_features.season_context_features import (
    SEASON_CONTEXT_FEATURE_COLUMNS,
    build_season_context_features,
)


def _base_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_A_B", "2025_01_A_B", "2025_02_A_C", "2025_02_A_C"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "C"],
            "opponent_team": ["B", "A", "C", "A"],
            "official_passing_yards": [200.0, 210.0, 220.0, 230.0],
        }
    )


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_A_B", "2025_01_C_D", "2025_02_A_C", "2025_02_B_D"],
            "season": [2025] * 4,
            "week": [1, 1, 2, 2],
            "game_type": ["REG"] * 4,
            "away_team": ["A", "C", "A", "B"],
            "home_team": ["B", "D", "C", "D"],
            "away_score": [24, 17, 99, 14],
            "home_score": [10, 20, 0, 21],
        }
    )


def _teams() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team_abbr": ["A", "B", "C", "D"],
            "team_conf": ["AFC", "AFC", "AFC", "AFC"],
            "team_division": ["AFC East", "AFC East", "AFC West", "AFC West"],
        }
    )


def test_week_one_has_only_schedule_known_stage_values() -> None:
    rows = build_season_context_features(_base_rows(), _schedules(), _teams())
    week_one = rows.filter(pl.col("week") == 1)

    assert week_one["season_context_week_fraction"].to_list() == [0.0, 0.0]
    assert week_one["season_context_team_games_played"].to_list() == [0.0, 0.0]
    assert week_one["season_context_team_games_remaining"].to_list() == [17.0, 17.0]
    assert week_one["season_context_team_win_pct"].null_count() == 2
    assert week_one["season_context_team_conference_rank_fraction"].null_count() == 2


def test_week_two_uses_only_completed_week_one_results() -> None:
    rows = build_season_context_features(_base_rows(), _schedules(), _teams())
    a = rows.filter((pl.col("week") == 2) & (pl.col("team") == "A")).row(0, named=True)

    assert a["season_context_team_games_played"] == 1.0
    assert a["season_context_team_games_remaining"] == 16.0
    assert a["season_context_team_win_pct"] == 1.0
    assert a["season_context_opponent_win_pct"] == 0.0
    assert a["season_context_win_pct_delta"] == 1.0
    assert a["season_context_team_point_diff_per_game"] == 14.0
    assert a["season_context_opponent_point_diff_per_game"] == -3.0
    assert a["season_context_point_diff_per_game_delta"] == 17.0


def test_current_week_scores_cannot_change_current_week_features() -> None:
    original = build_season_context_features(_base_rows(), _schedules(), _teams())
    mutated_current = _schedules().with_columns(
        pl.when(pl.col("week") == 2).then(pl.lit(-999)).otherwise(pl.col("away_score")).alias("away_score"),
        pl.when(pl.col("week") == 2).then(pl.lit(999)).otherwise(pl.col("home_score")).alias("home_score"),
    )
    changed = build_season_context_features(_base_rows(), mutated_current, _teams())

    assert original.select(SEASON_CONTEXT_FEATURE_COLUMNS).equals(changed.select(SEASON_CONTEXT_FEATURE_COLUMNS))


def test_prior_week_scores_do_change_later_features() -> None:
    original = build_season_context_features(_base_rows(), _schedules(), _teams())
    reversed_week_one = _schedules().with_columns(
        pl.when(pl.col("game_id") == "2025_01_A_B").then(pl.lit(0)).otherwise(pl.col("away_score")).alias("away_score"),
        pl.when(pl.col("game_id") == "2025_01_A_B")
        .then(pl.lit(30))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
    )
    changed = build_season_context_features(_base_rows(), reversed_week_one, _teams())

    original_a = original.filter((pl.col("week") == 2) & (pl.col("team") == "A"))
    changed_a = changed.filter((pl.col("week") == 2) & (pl.col("team") == "A"))
    assert original_a["season_context_team_win_pct"].item() == 1.0
    assert changed_a["season_context_team_win_pct"].item() == 0.0


def test_builder_preserves_source_rows_order_and_numeric_feature_contract() -> None:
    base = _base_rows()
    rows = build_season_context_features(base, _schedules(), _teams())

    assert rows.height == base.height
    assert rows.select(base.columns).equals(base)
    assert len(SEASON_CONTEXT_FEATURE_COLUMNS) == len(set(SEASON_CONTEXT_FEATURE_COLUMNS))
    assert all(rows.schema[column].is_numeric() for column in SEASON_CONTEXT_FEATURE_COLUMNS)
    assert not {"home_score", "away_score", "official_passing_yards"}.intersection(SEASON_CONTEXT_FEATURE_COLUMNS)


def test_builder_rejects_duplicate_source_keys_and_missing_metadata() -> None:
    with pytest.raises(ValueError, match="one row per game_id/team"):
        build_season_context_features(
            pl.concat([_base_rows(), _base_rows().head(1)]),
            _schedules(),
            _teams(),
        )

    with pytest.raises(ValueError, match="metadata is missing"):
        build_season_context_features(_base_rows(), _schedules(), _teams().filter(pl.col("team_abbr") != "D"))
