"""Pinned, weekly expanding-context inference for the NFL blog experiment."""

from __future__ import annotations

import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from synthefy_nori import NoriRegressor

from nfl_blog_features import blog_feature_columns

REVISION = "157d6be39b5ba8809e4296d50abf3f41f3b72947"
CHECKPOINT_SHA256 = "a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323"
TARGET = "remaining_passing_yards"
IDENTITY_COLUMNS = ["game_id", "season", "week", "kickoff_utc", "live_anchor_utc", "live_decision_utc",
                    "team", "opponent_team", "actual_qb_id", "actual_qb_name", "official_passing_yards",
                    "live_qb_attempts", "live_qb_passing_yards", TARGET]


def runtime_versions():
    return {name: version(name) for name in ("synthefy-nori", "numpy", "torch", "polars", "scipy", "scikit-learn")}


def frame_sha256(frame):
    buffer = io.BytesIO()
    frame.rechunk().write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def prediction_provenance(train, query, columns, device):
    """Fingerprint all numerical inputs, their order, identities and runtime."""
    inputs = list(dict.fromkeys([*IDENTITY_COLUMNS, *columns]))
    return {
        "cache_version": 1,
        "train_sha256": frame_sha256(train.select(inputs)),
        "query_sha256": frame_sha256(query.select(inputs)),
        "selected_columns": list(columns),
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_revision": REVISION,
        "runtime_versions": runtime_versions(), "device": str(device),
        "settings": {"memory_policy": "exact", "context_cap": None, "context_start": 2018,
                     "pruning_threshold": .75, "categorical_columns": [],
                     "inference_config": "installed NoriRegressor default", "augmentations": ["yj"],
                     "yj_skew_threshold": 10.0, "quantile_collapse": "mean"},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def read_prediction_cache(path, provenance, query, train_rows, columns):
    """Return None for missing/stale/corrupt caches; never trust an unmanifested file."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("provenance") != provenance:
            return None
        if manifest.get("prediction_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            return None
        prediction = pl.read_parquet(path)
        if not prediction.select(IDENTITY_COLUMNS).equals(query.select(IDENTITY_COLUMNS)):
            return None
        if prediction["context_rows"].to_list() != [train_rows] * query.height:
            return None
        if prediction["selected_feature_columns_json"].to_list() != [json.dumps(columns)] * query.height:
            return None
        required = {"nori_mean", "nori_p10", "nori_median", "nori_p90", "nori_quantile_taus", "nori_quantile_values"}
        if not required.issubset(prediction.columns):
            return None
        return prediction
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError):
        return None


def write_prediction_cache(path, prediction, provenance):
    path = Path(path)
    prediction.write_parquet(path)
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "provenance": provenance,
        "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2))



def checkpoint_path():
    path = Path(hf_hub_download("Synthefy/Nori", "nori.pt", revision=REVISION))
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Downloaded checkpoint does not match the blog weights")
    return path


def pearson(x, y):
    """Pairwise-complete correlation, matching the original training-only selector."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None
    x, y = x[valid], y[valid]
    x, y = x - x.mean(), y - y.mean()
    xx, yy = float(np.dot(x, x)), float(np.dot(y, y))
    if xx <= 0 or yy <= 0:
        return None
    return float(np.clip(np.dot(x, y) / np.sqrt(xx * yy), -1, 1))


def select_columns(train, candidates, threshold=0.75):
    """Rank by absolute target correlation; alphabetical ties; drop correlation > .75."""
    if len(candidates) != len(set(candidates)) or not candidates:
        raise ValueError("Candidate columns must be nonempty and unique")
    target = train[TARGET].to_numpy().astype(np.float64)
    values = {c: train[c].to_numpy().astype(np.float64) for c in candidates}
    correlations = {c: pearson(values[c], target) for c in sorted(candidates)}
    scores = {c: None if r is None else abs(r) for c, r in correlations.items()}
    ranked = sorted(candidates, key=lambda c: (scores[c] is None, -(scores[c] or 0), c))
    kept = []
    for c in ranked:
        if scores[c] is None:
            continue
        redundant = False
        for other in kept:
            r = pearson(values[c], values[other])
            if r is not None and abs(r) > threshold:
                redundant = True
        if not redundant:
            kept.append(c)
    if not kept:
        raise ValueError("No nonconstant features remain")
    return kept


def modeling_rows(rows, season=2025):
    return rows.filter(
        pl.col("live_evaluation_eligible")
        & pl.col("season").is_between(2018, season)
        & pl.col(TARGET).is_not_null() & pl.col(TARGET).is_finite()
        & pl.col("official_passing_yards").is_not_null() & pl.col("official_passing_yards").is_finite()
    ).sort(["season", "week", "live_decision_utc", "game_id", "team"])


def predict_week(estimator, train, query, columns):
    x = train.select(columns).to_numpy().astype(np.float32, copy=False)
    xq = query.select(columns).to_numpy().astype(np.float32, copy=False)
    if np.isinf(x).any() or np.isinf(xq).any():
        raise ValueError("Infinite feature values")
    estimator.fit(x, train[TARGET].to_numpy().astype(np.float64, copy=False))
    distribution = estimator.predict(xq, output_type="full")
    report = getattr(estimator, "memory_report_", None) or {}
    if report.get("dropped_context_rows", 0):
        raise RuntimeError("Exact reproduction cannot drop context rows")
    bank = np.asarray(distribution["quantiles"], dtype=np.float64)
    taus = np.asarray(distribution["taus"], dtype=np.float64)
    if bank.ndim != 2:
        raise ValueError("Unexpected quantile bank dimensions")
    if bank.shape[0] != query.height and bank.shape[1] == query.height:
        bank = bank.T
    if bank.shape != (query.height, taus.size):
        raise ValueError("Unexpected quantile bank shape")
    current = query["live_qb_passing_yards"].to_numpy().astype(np.float64)
    final = bank + current[:, None]
    quantiles = np.asarray([[np.interp(t, taus, row) for t in (.1, .5, .9)] for row in final])
    return query.select(IDENTITY_COLUMNS).with_columns(
        pl.lit(train.height, dtype=pl.Int64).alias("context_rows"),
        pl.lit(len(columns), dtype=pl.Int64).alias("selected_feature_count"),
        pl.lit(json.dumps(columns)).alias("selected_feature_columns_json"),
        pl.Series("nori_mean", np.asarray(distribution["mean"], dtype=np.float64) + current),
        pl.Series("nori_p10", quantiles[:, 0]), pl.Series("nori_median", quantiles[:, 1]),
        pl.Series("nori_p90", quantiles[:, 2]),
        pl.Series("nori_quantile_taus", [taus.tolist()] * query.height),
        pl.Series("nori_quantile_values", final.tolist()),
    )


def prediction_metrics(frame):
    actual = frame["official_passing_yards"].to_numpy()
    residual = actual - frame["nori_median"].to_numpy()
    pinball = {}
    for tau, column in ((.1, "nori_p10"), (.5, "nori_median"), (.9, "nori_p90")):
        error = actual - frame[column].to_numpy()
        pinball[str(tau)] = float(np.maximum(tau * error, (tau - 1) * error).mean())
    return {"rows": frame.height, "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())), "pinball_loss": pinball,
            "mean_pinball_loss": float(np.mean(list(pinball.values()))),
            "p10_p90_coverage": float(((actual >= frame["nori_p10"].to_numpy()) &
                                      (actual <= frame["nori_p90"].to_numpy())).mean())}


def run_blog_predictions(q1_rows, ht_rows, output_dir, weeks=tuple(range(1, 19)), device="cpu", season=2025, horizons=("q1", "halftime"), resume=False):
    """Compute checkpoints without a context cap; optionally resume verified local work.

    ``weeks`` limits scoring origins, not historical context. CPU execution of
    every week may take hours; choose a single week for a first smoke run.
    ``resume=True`` only reuses matching locally computed, fingerprinted results.
    """
    weeks = tuple(weeks)
    if not weeks or len(set(weeks)) != len(weeks):
        raise ValueError("weeks must be nonempty and unique")
    if not horizons or len(set(horizons)) != len(horizons) or set(horizons) - {"q1", "halftime"}:
        raise ValueError("horizons must contain q1 and/or halftime, without duplicates")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model = NoriRegressor(model_path=str(checkpoint_path()), device=device, memory_policy="exact")
    candidates = blog_feature_columns()
    outputs = {}
    for horizon, raw in (("q1", q1_rows), ("halftime", ht_rows)):
        if horizon not in horizons:
            continue
        rows = modeling_rows(raw, season)
        predictions, selections = [], {}
        for week in sorted(weeks):
            train = rows.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
            query = rows.filter((pl.col("season") == season) & (pl.col("week") == week))
            if train.is_empty() or query.is_empty():
                raise ValueError(f"Empty context/query for {horizon} week {week}")
            columns = select_columns(train, candidates)
            print(f"{horizon} week {week}: {train.height} context, {query.height} queries, {len(columns)} features", flush=True)
            path = directory / f"{horizon}_{season}_week{week}_predictions.parquet"
            provenance = prediction_provenance(train, query, columns, device)
            prediction = read_prediction_cache(path, provenance, query, train.height, columns) if resume else None
            if prediction is None:
                prediction = predict_week(model, train, query, columns)
                write_prediction_cache(path, prediction, provenance)
            else:
                print(f"Resumed verified {horizon} week {week}", flush=True)
            predictions.append(prediction)
            selections[week] = columns
        frame = pl.concat(predictions).sort(["week", "live_decision_utc", "game_id", "team"])
        frame.write_parquet(directory / f"{horizon}_{season}_predictions.parquet")
        (directory / f"{horizon}_{season}_manifest.json").write_text(json.dumps({
            "checkpoint_revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
            "runtime_versions": {name: version(name) for name in ("synthefy-nori", "numpy", "torch", "polars")},
            "device": device,
            "context_start": 2018, "context_cap": None, "categorical_columns": [],
            "pruning_threshold": .75, "selected_columns": selections, "metrics": prediction_metrics(frame),
        }, indent=2))
        outputs[horizon] = frame
    return outputs
