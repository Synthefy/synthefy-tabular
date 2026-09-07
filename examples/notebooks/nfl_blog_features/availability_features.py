"""Leak-safe offensive-line and pass-catcher availability features.

The nflverse injuries release contains one row per player/team/week and a
``date_modified`` timestamp through 2024. The 2025 release currently omits that
timestamp. This module therefore fails closed: a row is covered only when at
least one injury snapshot can be proven to exist at or before the prediction
cutoff. Untimestamped rows never contribute a status or a count.
"""

from __future__ import annotations

import polars as pl

_BASE_KEYS = ("game_id", "team")
_OFFENSIVE_LINE_POSITIONS = ("C", "G", "T", "OL")
_PASS_CATCHER_POSITIONS = ("WR", "TE", "RB", "FB")
_STATUS_BUCKETS = ("out", "doubtful", "questionable")
SNAP_HISTORY_GAMES = 4


def availability_feature_columns() -> list[str]:
    """Return the numeric columns produced by :func:`build_availability_features`."""

    columns = [
        "availability_report_coverage",
        "availability_relevant_player_count",
    ]
    for group in ("ol", "pass_catcher"):
        columns.extend(f"availability_{group}_{status}_count" for status in _STATUS_BUCKETS)
    return columns


def snap_weighted_availability_feature_columns() -> list[str]:
    """Return the six recent-role-weighted injury columns."""

    return [
        f"availability_{group}_{status}_snap_share"
        for group in ("ol", "pass_catcher")
        for status in _STATUS_BUCKETS
    ]


def _assert_unique_base_rows(base_rows: pl.DataFrame) -> None:
    missing = [column for column in _BASE_KEYS if column not in base_rows.columns]
    if missing:
        raise ValueError(f"base_rows is missing required columns: {missing}")
    if base_rows.select(*_BASE_KEYS).unique().height != base_rows.height:
        raise ValueError("base_rows must contain at most one row per game_id/team")


def _as_utc_timestamp(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = frame.schema[column]
    if dtype == pl.String:
        return frame.with_columns(
            pl.col(column).str.to_datetime(time_zone="UTC", strict=False).alias(column)
        )
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            return frame.with_columns(pl.col(column).dt.replace_time_zone("UTC").alias(column))
        if dtype.time_zone != "UTC":
            return frame.with_columns(pl.col(column).dt.convert_time_zone("UTC").alias(column))
        return frame
    raise ValueError(f"{column} must be a string or datetime timestamp, got {dtype}")


def select_pregame_injury_snapshots(
    base_rows: pl.DataFrame,
    injuries: pl.DataFrame,
) -> pl.DataFrame:
    """Select each player's latest injury snapshot proven available by cutoff.

    The return value includes all positions and report statuses so the caller
    can distinguish a covered team report from a report containing no relevant
    OL/pass-catcher designation. If ``date_modified`` is unavailable, an empty
    frame is returned rather than treating a season/week label as a timestamp.
    """

    _assert_unique_base_rows(base_rows)
    required_base = {
        "game_id",
        "season",
        "week",
        "team",
        "prediction_cutoff_utc",
        "anticipated_qb_id",
    }
    missing_base = sorted(required_base.difference(base_rows.columns))
    if missing_base:
        raise ValueError(f"base_rows is missing required columns: {missing_base}")

    required_injuries = {"season", "week", "team", "gsis_id", "position", "report_status"}
    missing_injuries = sorted(required_injuries.difference(injuries.columns))
    if missing_injuries:
        raise ValueError(f"injuries is missing required columns: {missing_injuries}")

    result_schema = {
        "game_id": base_rows.schema["game_id"],
        "team": base_rows.schema["team"],
        "anticipated_qb_id": base_rows.schema["anticipated_qb_id"],
        "gsis_id": injuries.schema["gsis_id"],
        "position": pl.String,
        "report_status": pl.String,
        "date_modified": pl.Datetime(time_zone="UTC"),
        "prediction_cutoff_utc": pl.Datetime(time_zone="UTC"),
    }
    if "date_modified" not in injuries.columns:
        return pl.DataFrame(schema=result_schema)

    injury_rows = injuries
    if "game_type" in injury_rows.columns:
        injury_rows = injury_rows.filter(pl.col("game_type") == "REG")
    elif "season_type" in injury_rows.columns:
        injury_rows = injury_rows.filter(pl.col("season_type") == "REG")

    injury_rows = _as_utc_timestamp(injury_rows, "date_modified").with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("gsis_id").cast(pl.String),
        pl.col("position").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("report_status").cast(pl.String).str.strip_chars().str.to_lowercase(),
    )
    prediction_rows = _as_utc_timestamp(base_rows, "prediction_cutoff_utc").select(
        "game_id",
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("anticipated_qb_id").cast(pl.String),
        "prediction_cutoff_utc",
    )

    return (
        prediction_rows.join(
            injury_rows.select(
                "season",
                "week",
                "team",
                "gsis_id",
                "position",
                "report_status",
                "date_modified",
            ),
            on=["season", "week", "team"],
            how="inner",
            validate="1:m",
        )
        .filter(
            pl.col("gsis_id").is_not_null()
            & pl.col("date_modified").is_not_null()
            & (pl.col("date_modified") <= pl.col("prediction_cutoff_utc"))
        )
        .sort("date_modified")
        .unique(["game_id", "team", "gsis_id"], keep="last", maintain_order=True)
        .select(*result_schema)
        .sort(["game_id", "team", "gsis_id"])
    )


def _empty_availability_features(base_rows: pl.DataFrame) -> pl.DataFrame:
    return base_rows.with_columns(
        pl.lit(0.0).alias("availability_report_coverage"),
        *[
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in availability_feature_columns()
            if column != "availability_report_coverage"
        ],
    )


def build_availability_features(
    base_rows: pl.DataFrame,
    injuries: pl.DataFrame,
) -> pl.DataFrame:
    """Attach leak-safe OL/pass-catcher game-status counts to base QB-games.

    Counts are null when no timestamped injury report is available for the
    team/game and zero when a timestamped report is available but contains no
    player in that bucket. Quarterbacks are excluded both by position and by
    the anticipated-QB identifier.
    """

    snapshots = select_pregame_injury_snapshots(base_rows, injuries)
    if snapshots.is_empty():
        return _empty_availability_features(base_rows)

    classified = snapshots.with_columns(
        pl.when(pl.col("position").is_in(_OFFENSIVE_LINE_POSITIONS))
        .then(pl.lit("ol"))
        .when(pl.col("position").is_in(_PASS_CATCHER_POSITIONS))
        .then(pl.lit("pass_catcher"))
        .otherwise(None)
        .alias("availability_group"),
        pl.when(pl.col("report_status").is_in(_STATUS_BUCKETS))
        .then(pl.col("report_status"))
        .otherwise(None)
        .alias("availability_status"),
    ).with_columns(
        (
            (pl.col("position") != "QB")
            & (
                pl.col("anticipated_qb_id").is_null()
                | (pl.col("gsis_id") != pl.col("anticipated_qb_id"))
            )
            & pl.col("availability_group").is_not_null()
            & pl.col("availability_status").is_not_null()
        ).alias("is_relevant_player")
    )

    aggregations: list[pl.Expr] = [
        pl.lit(1.0).first().alias("availability_report_coverage"),
        pl.col("is_relevant_player").sum().cast(pl.Float64).alias("availability_relevant_player_count"),
    ]
    for group in ("ol", "pass_catcher"):
        for status in _STATUS_BUCKETS:
            aggregations.append(
                (
                    pl.col("is_relevant_player")
                    & (pl.col("availability_group") == group)
                    & (pl.col("availability_status") == status)
                )
                .sum()
                .cast(pl.Float64)
                .alias(f"availability_{group}_{status}_count")
            )

    features = classified.group_by(*_BASE_KEYS).agg(*aggregations)
    result = base_rows.join(features, on=list(_BASE_KEYS), how="left", validate="1:1")
    count_columns = [
        column for column in availability_feature_columns() if column != "availability_report_coverage"
    ]
    return result.with_columns(
        pl.col("availability_report_coverage").fill_null(0.0),
        *[
            pl.when(pl.col("availability_report_coverage") > 0)
            .then(pl.col(column).fill_null(0.0))
            .otherwise(None)
            .cast(pl.Float64)
            .alias(column)
            for column in count_columns
        ],
    )


def _franchise(team: str) -> str:
    """Normalize the only franchise relocation in the frozen source window."""

    return "LV" if team == "OAK" else team


def _snap_source_rows(snap_counts: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id",
        "season",
        "game_type",
        "week",
        "pfr_player_id",
        "team",
        "offense_pct",
    }
    missing = sorted(required.difference(snap_counts.columns))
    if missing:
        raise ValueError(f"snap_counts is missing required columns: {missing}")
    rows = (
        snap_counts.filter(pl.col("game_type") == "REG")
        .select(*required)
        .with_columns(
            pl.col("game_id").cast(pl.String),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("pfr_player_id").cast(pl.String),
            pl.col("team").cast(pl.String).map_elements(_franchise, return_dtype=pl.String),
            pl.col("offense_pct").cast(pl.Float64),
        )
    )
    if rows.is_empty():
        raise ValueError("snap_counts has no regular-season rows")
    invalid_share = rows.filter(
        pl.col("offense_pct").is_null()
        | ~pl.col("offense_pct").is_finite()
        | ~pl.col("offense_pct").is_between(0.0, 1.0)
    )
    if invalid_share.height:
        raise ValueError("snap_counts offense_pct must be finite and between zero and one")
    keys = ["game_id", "team", "pfr_player_id"]
    if rows.select(keys).is_duplicated().any():
        raise ValueError("snap_counts contains duplicate game/team/player rows")
    return rows


def _player_crosswalk(players: pl.DataFrame) -> dict[str, str]:
    required = {"gsis_id", "pfr_id"}
    missing = sorted(required.difference(players.columns))
    if missing:
        raise ValueError(f"players is missing required columns: {missing}")
    rows = (
        players.select(
            pl.col("gsis_id").cast(pl.String),
            pl.col("pfr_id").cast(pl.String),
        )
        .filter(pl.col("gsis_id").is_not_null() & pl.col("pfr_id").is_not_null())
        .unique()
    )
    conflicts = rows.group_by("gsis_id").agg(pl.col("pfr_id").n_unique().alias("pfr_ids"))
    if conflicts.filter(pl.col("pfr_ids") != 1).height:
        raise ValueError("players contains conflicting gsis_id to pfr_id mappings")
    return dict(rows.unique("gsis_id").iter_rows())


def build_snap_weighted_availability_features(
    base_rows: pl.DataFrame,
    injuries: pl.DataFrame,
    snap_counts: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Attach the frozen four-team-game snap-weighted availability features.

    A missing player row in a known prior team game is a zero share.  A missing
    crosswalk or fewer than four prior team games fails the six weighted values
    closed to null.  Report coverage remains separate so unknown is never
    encoded as a healthy offense.
    """

    _assert_unique_base_rows(base_rows)
    weighted_columns = snap_weighted_availability_feature_columns()
    overlap = sorted(set(weighted_columns).intersection(base_rows.columns))
    if overlap:
        raise ValueError(f"base_rows already contains snap-weighted columns: {overlap}")
    snapshots = select_pregame_injury_snapshots(base_rows, injuries)
    snaps = _snap_source_rows(snap_counts)
    crosswalk = _player_crosswalk(players)

    team_games_frame = snaps.select("season", "week", "game_id", "team").unique()
    if team_games_frame.select("season", "week", "team").is_duplicated().any():
        raise ValueError("snap_counts contains multiple regular-season games for a team/week")
    team_games: dict[str, list[tuple[int, int, str]]] = {}
    for season, week, game_id, team in team_games_frame.sort(
        ["team", "season", "week", "game_id"]
    ).iter_rows():
        team_games.setdefault(str(team), []).append((int(season), int(week), str(game_id)))
    snap_share = {
        (str(game_id), str(team), str(pfr_id)): float(share)
        for game_id, team, pfr_id, share in snaps.select(
            "game_id", "team", "pfr_player_id", "offense_pct"
        ).iter_rows()
    }

    snapshot_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in snapshots.iter_rows(named=True):
        snapshot_rows.setdefault((str(row["game_id"]), str(row["team"])), []).append(row)

    records: list[dict[str, object]] = []
    for base in base_rows.select(
        "game_id",
        "season",
        "week",
        "team",
        "anticipated_qb_id",
    ).iter_rows(named=True):
        key = (str(base["game_id"]), str(base["team"]))
        observed = snapshot_rows.get(key)
        record: dict[str, object] = {
            "game_id": key[0],
            "team": key[1],
            "_computed_availability_report_coverage": 1.0 if observed else 0.0,
            **{column: None for column in weighted_columns},
        }
        if not observed:
            records.append(record)
            continue

        relevant: list[tuple[str, str, str]] = []
        for injury in observed:
            position = str(injury["position"])
            status = str(injury["report_status"])
            gsis_id = str(injury["gsis_id"])
            group = None
            if position in _OFFENSIVE_LINE_POSITIONS:
                group = "ol"
            elif position in _PASS_CATCHER_POSITIONS:
                group = "pass_catcher"
            if (
                group is not None
                and status in _STATUS_BUCKETS
                and position != "QB"
                and gsis_id != str(base["anticipated_qb_id"])
            ):
                relevant.append((gsis_id, group, status))

        if not relevant:
            record.update({column: 0.0 for column in weighted_columns})
            records.append(record)
            continue

        franchise = _franchise(str(base["team"]))
        season_week = (int(base["season"]), int(base["week"]))
        history = [
            game
            for game in team_games.get(franchise, [])
            if (game[0], game[1]) < season_week
        ][-SNAP_HISTORY_GAMES:]
        if len(history) != SNAP_HISTORY_GAMES or any(gsis_id not in crosswalk for gsis_id, _, _ in relevant):
            records.append(record)
            continue

        totals = {column: 0.0 for column in weighted_columns}
        for gsis_id, group, status in relevant:
            pfr_id = crosswalk[gsis_id]
            weight = sum(
                snap_share.get((game_id, franchise, pfr_id), 0.0)
                for _, _, game_id in history
            ) / SNAP_HISTORY_GAMES
            totals[f"availability_{group}_{status}_snap_share"] += weight
        record.update(totals)
        records.append(record)

    features = pl.DataFrame(records).with_columns(
        pl.col("_computed_availability_report_coverage").cast(pl.Float64),
        *[pl.col(column).cast(pl.Float64) for column in weighted_columns],
    )
    if "availability_report_coverage" in base_rows.columns:
        expected = base_rows.select(*_BASE_KEYS, "availability_report_coverage").sort(_BASE_KEYS)
        computed = features.select(
            *_BASE_KEYS,
            pl.col("_computed_availability_report_coverage").alias("availability_report_coverage"),
        ).sort(_BASE_KEYS)
        if not expected.equals(computed):
            raise ValueError("existing availability_report_coverage disagrees with timestamped snapshots")
        features = features.drop("_computed_availability_report_coverage")
    else:
        features = features.rename(
            {"_computed_availability_report_coverage": "availability_report_coverage"}
        )
    return base_rows.join(features, on=list(_BASE_KEYS), how="left", validate="1:1")
