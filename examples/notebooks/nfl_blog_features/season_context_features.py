"""Prior-week NFL season and standings context features.

Every query row is computed from games in strictly earlier NFL weeks. The
conference and division ranks are standings proxies: ties are ranked by win
percentage and point differential, not by the NFL's full tiebreaker tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import polars as pl

ROOT = Path(__file__).resolve().parent

SEASON_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "season_context_week_fraction",
    "season_context_team_games_played",
    "season_context_team_games_remaining",
    "season_context_team_win_pct",
    "season_context_opponent_win_pct",
    "season_context_win_pct_delta",
    "season_context_team_point_diff_per_game",
    "season_context_opponent_point_diff_per_game",
    "season_context_point_diff_per_game_delta",
    "season_context_team_conference_rank_fraction",
    "season_context_opponent_conference_rank_fraction",
    "season_context_conference_rank_fraction_delta",
    "season_context_team_division_rank_fraction",
    "season_context_opponent_division_rank_fraction",
    "season_context_division_rank_fraction_delta",
    "season_context_team_gap_to_conference_7th",
    "season_context_opponent_gap_to_conference_7th",
    "season_context_team_gap_to_division_leader",
    "season_context_opponent_gap_to_division_leader",
    "season_context_team_max_wins_over_conference_7th",
    "season_context_opponent_max_wins_over_conference_7th",
)

_BASE_KEYS = ("game_id", "team")
_SNAPSHOT_FIELDS = (
    "games_played",
    "games_remaining",
    "win_pct",
    "point_diff_per_game",
    "conference_rank_fraction",
    "division_rank_fraction",
    "gap_to_conference_7th",
    "gap_to_division_leader",
    "max_wins_over_conference_7th",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _season_rules(season: int) -> tuple[int, int]:
    """Return regular-season weeks and scheduled games per team."""

    return (17, 16) if season <= 2020 else (18, 17)


def _average_rank_fraction(
    values: Mapping[str, tuple[float, float]],
    members: Sequence[str],
) -> dict[str, float | None]:
    """Rank descending with average ranks for exact metric ties."""

    available = [team for team in members if team in values]
    if not available:
        return {team: None for team in members}
    ordered = sorted(available, key=lambda team: (-values[team][0], -values[team][1], team))
    denominator = max(len(ordered) - 1, 1)
    result: dict[str, float | None] = {team: None for team in members}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        fraction = (average_rank - 1.0) / denominator
        for team in ordered[index:end]:
            result[team] = float(fraction)
        index = end
    return result


def _standings_snapshots(schedules: pl.DataFrame, teams: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    _require_columns(
        schedules,
        ("season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"),
        label="schedules",
    )
    _require_columns(teams, ("team_abbr", "team_conf", "team_division"), label="teams")
    metadata_rows = teams.select("team_abbr", "team_conf", "team_division").unique().to_dicts()
    if len(metadata_rows) != teams.get_column("team_abbr").n_unique():
        raise ValueError("teams metadata maps one abbreviation to multiple conference/division rows")
    metadata = {
        str(row["team_abbr"]): (str(row["team_conf"]), str(row["team_division"]))
        for row in metadata_rows
    }

    regular = schedules.filter(
        (pl.col("game_type") == "REG") & pl.col("season").is_in([int(season) for season in seasons])
    ).sort(["season", "week", "game_id"] if "game_id" in schedules.columns else ["season", "week"])
    output: list[dict[str, Any]] = []
    for season in sorted({int(value) for value in seasons}):
        season_games = regular.filter(pl.col("season") == season)
        scheduled_teams = sorted(
            set(season_games.get_column("home_team").drop_nulls().to_list())
            | set(season_games.get_column("away_team").drop_nulls().to_list())
        )
        if not scheduled_teams:
            raise ValueError(f"schedules contain no regular-season games for {season}")
        missing_metadata = sorted(set(scheduled_teams).difference(metadata))
        if missing_metadata:
            raise ValueError(f"teams metadata is missing season {season} codes: {missing_metadata}")
        regular_weeks, games_per_team = _season_rules(season)
        state = {
            team: {"games": 0.0, "wins": 0.0, "ties": 0.0, "points_for": 0.0, "points_against": 0.0}
            for team in scheduled_teams
        }
        games_by_week: dict[int, list[dict[str, Any]]] = {}
        for row in season_games.select("week", "home_team", "away_team", "home_score", "away_score").to_dicts():
            games_by_week.setdefault(int(row["week"]), []).append(row)

        for week in range(1, regular_weeks + 1):
            win_pct = {
                team: (values["wins"] + 0.5 * values["ties"]) / values["games"]
                for team, values in state.items()
                if values["games"] > 0
            }
            point_diff = {
                team: (values["points_for"] - values["points_against"]) / values["games"]
                for team, values in state.items()
                if values["games"] > 0
            }
            rank_values = {team: (win_pct[team], point_diff[team]) for team in win_pct}
            conference_ranks: dict[str, float | None] = {}
            division_ranks: dict[str, float | None] = {}
            conference_cutoffs: dict[str, tuple[float, float]] = {}
            division_leaders: dict[str, float] = {}
            for conference in sorted({metadata[team][0] for team in scheduled_teams}):
                members = [team for team in scheduled_teams if metadata[team][0] == conference]
                conference_ranks.update(_average_rank_fraction(rank_values, members))
                ordered = sorted(
                    [team for team in members if team in rank_values],
                    key=lambda team: (-rank_values[team][0], -rank_values[team][1], team),
                )
                if ordered:
                    cutoff = ordered[min(6, len(ordered) - 1)]
                    conference_cutoffs[conference] = (win_pct[cutoff], state[cutoff]["wins"])
            for division in sorted({metadata[team][1] for team in scheduled_teams}):
                members = [team for team in scheduled_teams if metadata[team][1] == division]
                division_ranks.update(_average_rank_fraction(rank_values, members))
                observed = [win_pct[team] for team in members if team in win_pct]
                if observed:
                    division_leaders[division] = max(observed)

            for team in scheduled_teams:
                conference, division = metadata[team]
                values = state[team]
                games_played = values["games"]
                games_remaining = float(max(0, games_per_team - int(games_played)))
                team_win_pct = win_pct.get(team)
                cutoff = conference_cutoffs.get(conference)
                output.append(
                    {
                        "season": season,
                        "week": week,
                        "standing_team": team,
                        "games_played": games_played,
                        "games_remaining": games_remaining,
                        "win_pct": team_win_pct,
                        "point_diff_per_game": point_diff.get(team),
                        "conference_rank_fraction": conference_ranks.get(team),
                        "division_rank_fraction": division_ranks.get(team),
                        "gap_to_conference_7th": None
                        if team_win_pct is None or cutoff is None
                        else team_win_pct - cutoff[0],
                        "gap_to_division_leader": None
                        if team_win_pct is None or division not in division_leaders
                        else team_win_pct - division_leaders[division],
                        "max_wins_over_conference_7th": None
                        if cutoff is None
                        else values["wins"] + games_remaining - cutoff[1],
                    }
                )

            # Week N outcomes enter state only after the Week N snapshot.
            for game in games_by_week.get(week, []):
                home_score = game["home_score"]
                away_score = game["away_score"]
                if home_score is None or away_score is None:
                    continue
                home = str(game["home_team"])
                away = str(game["away_team"])
                for team, scored, allowed in ((home, home_score, away_score), (away, away_score, home_score)):
                    state[team]["games"] += 1.0
                    state[team]["points_for"] += float(scored)
                    state[team]["points_against"] += float(allowed)
                if float(home_score) > float(away_score):
                    state[home]["wins"] += 1.0
                elif float(away_score) > float(home_score):
                    state[away]["wins"] += 1.0
                else:
                    state[home]["ties"] += 1.0
                    state[away]["ties"] += 1.0

    return pl.DataFrame(output).with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        *[pl.col(column).cast(pl.Float64) for column in _SNAPSHOT_FIELDS],
    )


def build_season_context_features(
    base_rows: pl.DataFrame,
    schedules: pl.DataFrame,
    teams: pl.DataFrame,
) -> pl.DataFrame:
    """Append season-progress and prior-week standings features."""

    _require_columns(base_rows, ("game_id", "season", "week", "team", "opponent_team"), label="base_rows")
    if base_rows.select(pl.struct(_BASE_KEYS).is_duplicated().any()).item():
        raise ValueError("base_rows must have exactly one row per game_id/team")
    existing = sorted(set(SEASON_CONTEXT_FEATURE_COLUMNS).intersection(base_rows.columns))
    if existing:
        raise ValueError(f"season-context columns already exist: {existing}")

    seasons = [int(value) for value in base_rows.get_column("season").unique().to_list()]
    snapshots = _standings_snapshots(schedules, teams, seasons)
    working = base_rows.with_row_index("_season_context_row_order").with_columns(
        pl.col("season").cast(pl.Int64).alias("_season_context_season"),
        pl.col("week").cast(pl.Int64).alias("_season_context_week"),
    )
    team_snapshot = snapshots.rename(
        {column: f"_season_context_team_{column}" for column in _SNAPSHOT_FIELDS}
    ).rename({"standing_team": "_season_context_team"})
    opponent_snapshot = snapshots.rename(
        {column: f"_season_context_opponent_{column}" for column in _SNAPSHOT_FIELDS}
    ).rename({"standing_team": "_season_context_opponent"})
    rows = (
        working.join(
            team_snapshot,
            left_on=["_season_context_season", "_season_context_week", "team"],
            right_on=["season", "week", "_season_context_team"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(
            opponent_snapshot,
            left_on=["_season_context_season", "_season_context_week", "opponent_team"],
            right_on=["season", "week", "_season_context_opponent"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    )
    if rows.get_column("_season_context_team_games_played").null_count() or rows.get_column(
        "_season_context_opponent_games_played"
    ).null_count():
        raise ValueError("base rows contain season/week/team values absent from standings snapshots")

    regular_week_denominator = (
        pl.when(pl.col("_season_context_season") <= 2020).then(pl.lit(16.0)).otherwise(pl.lit(17.0))
    )
    rows = rows.with_columns(
        ((pl.col("_season_context_week") - 1).cast(pl.Float64) / regular_week_denominator).alias(
            "season_context_week_fraction"
        ),
        pl.col("_season_context_team_games_played").alias("season_context_team_games_played"),
        pl.col("_season_context_team_games_remaining").alias("season_context_team_games_remaining"),
        pl.col("_season_context_team_win_pct").alias("season_context_team_win_pct"),
        pl.col("_season_context_opponent_win_pct").alias("season_context_opponent_win_pct"),
        (pl.col("_season_context_team_win_pct") - pl.col("_season_context_opponent_win_pct")).alias(
            "season_context_win_pct_delta"
        ),
        pl.col("_season_context_team_point_diff_per_game").alias("season_context_team_point_diff_per_game"),
        pl.col("_season_context_opponent_point_diff_per_game").alias(
            "season_context_opponent_point_diff_per_game"
        ),
        (
            pl.col("_season_context_team_point_diff_per_game")
            - pl.col("_season_context_opponent_point_diff_per_game")
        ).alias("season_context_point_diff_per_game_delta"),
        pl.col("_season_context_team_conference_rank_fraction").alias(
            "season_context_team_conference_rank_fraction"
        ),
        pl.col("_season_context_opponent_conference_rank_fraction").alias(
            "season_context_opponent_conference_rank_fraction"
        ),
        (
            pl.col("_season_context_team_conference_rank_fraction")
            - pl.col("_season_context_opponent_conference_rank_fraction")
        ).alias("season_context_conference_rank_fraction_delta"),
        pl.col("_season_context_team_division_rank_fraction").alias(
            "season_context_team_division_rank_fraction"
        ),
        pl.col("_season_context_opponent_division_rank_fraction").alias(
            "season_context_opponent_division_rank_fraction"
        ),
        (
            pl.col("_season_context_team_division_rank_fraction")
            - pl.col("_season_context_opponent_division_rank_fraction")
        ).alias("season_context_division_rank_fraction_delta"),
        pl.col("_season_context_team_gap_to_conference_7th").alias(
            "season_context_team_gap_to_conference_7th"
        ),
        pl.col("_season_context_opponent_gap_to_conference_7th").alias(
            "season_context_opponent_gap_to_conference_7th"
        ),
        pl.col("_season_context_team_gap_to_division_leader").alias(
            "season_context_team_gap_to_division_leader"
        ),
        pl.col("_season_context_opponent_gap_to_division_leader").alias(
            "season_context_opponent_gap_to_division_leader"
        ),
        pl.col("_season_context_team_max_wins_over_conference_7th").alias(
            "season_context_team_max_wins_over_conference_7th"
        ),
        pl.col("_season_context_opponent_max_wins_over_conference_7th").alias(
            "season_context_opponent_max_wins_over_conference_7th"
        ),
    )
    output = rows.sort("_season_context_row_order").select(*base_rows.columns, *SEASON_CONTEXT_FEATURE_COLUMNS)
    if output.height != base_rows.height or not output.select(base_rows.columns).equals(base_rows):
        raise RuntimeError("season-context builder changed source rows or existing feature values")
    if any(not output.schema[column].is_numeric() for column in SEASON_CONTEXT_FEATURE_COLUMNS):
        raise RuntimeError("season-context features must all be numeric")
    return output
