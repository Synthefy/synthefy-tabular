"""Unchanged prior-checkpoint history math from the original experiment."""
from __future__ import annotations
import numpy as np
import polars as pl

WINDOWS = (3, 8)
SOURCES = (
    "live_qb_passing_yards",
    "live_qb_attempts",
    "live_qb_ypa",
    "remaining_passing_yards",
)


def add_checkpoint_history(rows: pl.DataFrame) -> tuple[pl.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Use only earlier season/weeks; never current-game or same-week labels.

    Call separately for Q1 and halftime. Cold starters have missing averages
    and a zero history count. Preserve input order and all original columns.
    """
    keys = ["season", "week", "game_id", "actual_qb_id"]
    if rows.select(keys).is_duplicated().any():
        raise ValueError("duplicate QB-game checkpoint rows")
    records = rows.to_dicts()
    histories: dict[str, list[dict]] = {}
    features: dict[int, dict] = {}
    groups: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(records):
        if row["actual_qb_id"] is None:
            raise ValueError("checkpoint history requires QB identity")
        groups.setdefault((row["season"], row["week"]), []).append(i)
    averages = tuple(f"checkpoint_history_{source}_mean_last{window}" for window in WINDOWS for source in SOURCES) + (
        "checkpoint_history_games_last8",
    )
    deviations = tuple(f"checkpoint_deviation_{source}_last{window}" for window in WINDOWS for source in SOURCES[:2])
    for period, indices in sorted(groups.items()):
        for i in indices:
            row = records[i]
            prior = histories.get(row["actual_qb_id"], [])
            # A late/rescheduled prior-week game must also precede this kickoff.
            prior = [r for r in prior if r["kickoff_utc"] < row["kickoff_utc"]]
            f = {"checkpoint_history_games_last8": float(min(len(prior), 8))}
            for window in WINDOWS:
                recent = prior[-window:]
                for source in SOURCES:
                    values = [r[source] for r in recent if r[source] is not None and np.isfinite(r[source])]
                    mean = float(np.mean(values)) if values else None
                    f[f"checkpoint_history_{source}_mean_last{window}"] = mean
                    if source in SOURCES[:2]:
                        current = row[source]
                        f[f"checkpoint_deviation_{source}_last{window}"] = (
                            float(current - mean) if current is not None and mean is not None else None
                        )
            features[i] = f
        # Update only after every query in the week has been constructed.
        for i in sorted(indices, key=lambda j: records[j]["kickoff_utc"]):
            row = records[i]
            if row.get("live_evaluation_eligible", True):
                histories.setdefault(row["actual_qb_id"], []).append(row)
    return (
        rows.hstack(
            pl.DataFrame(
                [features[i] for i in range(len(records))], schema={c: pl.Float64 for c in (*averages, *deviations)}
            )
        ),
        averages,
        deviations,
    )
