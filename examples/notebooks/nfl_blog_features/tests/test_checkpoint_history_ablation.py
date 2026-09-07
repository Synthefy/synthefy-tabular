from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_blog_features.checkpoint_history import add_checkpoint_history


def rows():
    start = datetime(2023, 12, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "season": 2023 if i < 2 else 2024,
                "week": 17 if i < 2 else i - 1,
                "game_id": str(i),
                "actual_qb_id": "qb",
                "kickoff_utc": start + timedelta(days=i * 7),
                "live_qb_passing_yards": float(10 * (i + 1)),
                "live_qb_attempts": float(i + 2),
                "live_qb_ypa": 5.0,
                "remaining_passing_yards": float(100 + i),
                "live_evaluation_eligible": True,
            }
            for i in range(5)
        ]
    )


def test_history_excludes_current_and_same_week_and_crosses_seasons():
    data = rows()
    result, averages, deviations = add_checkpoint_history(data)
    name = "checkpoint_history_live_qb_passing_yards_mean_last3"
    assert result[name].to_list()[:3] == [None, None, 15.0]
    assert result[name][3] == 20.0
    changed = data.with_columns(
        pl.when(pl.col("game_id") == "2")
        .then(9999.0)
        .otherwise(pl.col("remaining_passing_yards"))
        .alias("remaining_passing_yards")
    )
    altered, _, _ = add_checkpoint_history(changed)
    assert result.select(*averages, *deviations).head(3).equals(altered.select(*averages, *deviations).head(3))
    assert result["checkpoint_deviation_live_qb_passing_yards_last3"][2] == 15.0
    assert result["checkpoint_history_games_last8"][0] == 0


def test_order_invariance_and_duplicate_rejection():
    original, _, _ = add_checkpoint_history(rows())
    reverse, _, _ = add_checkpoint_history(rows().reverse())
    assert original.equals(reverse.reverse())
    with pytest.raises(ValueError, match="duplicate"):
        add_checkpoint_history(pl.concat([rows(), rows().head(1)]))


def test_ineligible_history_and_unseen_qb():
    data = rows().with_columns((pl.col("game_id") != "0").alias("live_evaluation_eligible"))
    result, _, _ = add_checkpoint_history(data)
    assert result["checkpoint_history_live_qb_passing_yards_mean_last3"][2] == 20.0
    data = data.with_columns(
        pl.when(pl.col("game_id") == "4").then(pl.lit("new")).otherwise(pl.col("actual_qb_id")).alias("actual_qb_id")
    )
    result, _, _ = add_checkpoint_history(data)
    assert result["checkpoint_history_games_last8"][4] == 0
