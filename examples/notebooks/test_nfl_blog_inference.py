"""Offline guards for the exact blog inference port."""

import numpy as np
import polars as pl

from nfl_blog_inference import (
    modeling_rows,
    pearson,
    select_columns,
    IDENTITY_COLUMNS,
    prediction_provenance,
    read_prediction_cache,
    write_prediction_cache,
)


def test_pairwise_missing_and_constant():
    assert pearson(np.array([1.0, 2.0, np.nan]), np.array([2.0, 4.0, 9.0])) == 1.0
    assert pearson(np.ones(3), np.arange(3.0)) is None


def test_pruning_ties_and_constants():
    rows = pl.DataFrame(
        {
            "remaining_passing_yards": [1.0, 2.0, 3.0, 4.0],
            "z": [1.0, 2.0, 3.0, 4.0],
            "a": [2.0, 4.0, 6.0, 8.0],
            "constant": [0.0, 0.0, 0.0, 0.0],
        }
    )
    assert select_columns(rows, ["z", "constant", "a"]) == ["a"]


def test_context_start_and_eligibility():
    rows = pl.DataFrame(
        {
            "season": [2017, 2018, 2025, 2025, 2026],
            "week": [1] * 5,
            "game_id": ["a", "b", "c", "d", "e"],
            "team": ["X"] * 5,
            "live_decision_utc": [1, 2, 3, 4, 5],
            "live_evaluation_eligible": [True, True, True, False, True],
            "remaining_passing_yards": [1.0] * 5,
            "official_passing_yards": [2.0] * 5,
        }
    )
    assert modeling_rows(rows)["game_id"].to_list() == ["b", "c"]


def test_resume_validates_inputs_and_output(tmp_path):
    query = pl.DataFrame({name: [1] for name in IDENTITY_COLUMNS}).with_columns(pl.lit(2.0).alias("feature"))
    train = query
    columns = ["feature"]
    provenance = prediction_provenance(train, query, columns, "cpu")
    prediction = query.select(IDENTITY_COLUMNS).with_columns(
        pl.lit(1).alias("context_rows"),
        pl.lit('["feature"]').alias("selected_feature_columns_json"),
        *[pl.lit(1.0).alias(name) for name in ("nori_mean", "nori_p10", "nori_median", "nori_p90")],
        pl.lit([0.1, 0.5, 0.9]).alias("nori_quantile_taus"),
        pl.lit([1.0, 2.0, 3.0]).alias("nori_quantile_values"),
    )
    path = tmp_path / "prediction.parquet"
    assert read_prediction_cache(path, provenance, query, 1, columns) is None
    write_prediction_cache(path, prediction, provenance)
    assert read_prediction_cache(path, provenance, query, 1, columns).equals(prediction)
    equivalent = prediction_provenance(
        train.with_columns(pl.lit(2.0 + 1e-12).alias("feature")),
        query.with_columns(pl.lit(2.0 - 1e-12).alias("feature")),
        columns,
        "cpu",
    )
    assert equivalent == provenance
    changed_target = prediction_provenance(
        train.with_columns(pl.lit(1.0 + 1e-12).alias("remaining_passing_yards")), query, columns, "cpu"
    )
    assert changed_target != provenance
    assert read_prediction_cache(path, equivalent, query, 1, columns).equals(prediction)
    changed = prediction_provenance(train.with_columns(pl.lit(2.0 + 1e-4).alias("feature")), query, columns, "cpu")
    assert read_prediction_cache(path, changed, query, 1, columns) is None
    assert read_prediction_cache(path, provenance, query, 2, columns) is None
    assert read_prediction_cache(path, provenance, query, 1, ["other"]) is None
    assert read_prediction_cache(path, {**provenance, "device": "cuda:0"}, query, 1, columns) is None
    path.write_bytes(b"corrupt")
    assert read_prediction_cache(path, provenance, query, 1, columns) is None
