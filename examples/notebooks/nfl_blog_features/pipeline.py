"""Rebuild original blog inputs from public nflverse data, without saved predictions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import nflreadpy as nfl
import polars as pl
from .data import DataConfig, build_base_qb_game_table, load_nflverse_tables
from .features import build_qb_rolling_features
from .offense_features import build_offense_rolling_features
from .defense_features import build_defense_features
from .context_features import build_context_features
from .live_features import LiveConfig, build_live_rows, load_live_pbp, live_feature_columns
from .live_q1_features import build_q1_rows
from .live_flow_features import build_live_flow_features
from .season_context_features import build_season_context_features
from .checkpoint_history import add_checkpoint_history, WINDOWS, SOURCES


def blog_configs(cache_dir: Path, output_dir: Path):
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    data = DataConfig(2016, 2018, 2024, 2025, "REG", 60, cache_dir,
        output_dir / "base.parquet", output_dir / "pregame.parquet",
        (3, 8), (3, 8), 0.2, 0.8, 3, True, (3, 8))
    live = LiveConfig(2, 2, 5, "qb_offense_defense_context_season",
        cache_dir / "pbp_live_2018_2025.parquet", output_dir / "halftime.parquet",
        cache_dir / "kalshi", output_dir / "validation.parquet",
        output_dir / "validation.json", output_dir / "test.parquet", output_dir / "test.json")
    return data, live


def blog_feature_columns():
    """Exact ordered numeric candidates before training-only correlation pruning."""
    data, live = blog_configs(Path("cache"), Path("outputs"))
    return live_feature_columns(data, live, "usage") + [
        f"checkpoint_history_{source}_mean_last{window}" for window in WINDOWS for source in SOURCES
    ] + ["checkpoint_history_games_last8"]


def build_blog_features(cache_dir, output_dir, refresh=False):
    """Download public inputs and return Q1/halftime Polars tables.

    Identity is used to compute individual historical form, not as categorical
    model input, matching the selected original numeric model. Upstream data
    revisions may change outputs; input hashes are recorded with each run.
    """
    data, live = blog_configs(cache_dir, output_dir)
    data.cache_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tables = load_nflverse_tables(data, refresh=refresh)
    print("Building QB form, offense, defense, and context.", flush=True)
    rows = build_base_qb_game_table(schedules=tables["schedules"], depth_charts=tables["depth_charts"],
        participation=tables["participation"], pbp_starter_plays=tables["pbp_starter_plays"],
        player_stats=tables["player_stats"], config=data)
    # The actual starter is observable in-game; recompute that player's past form.
    rows = build_qb_rolling_features(rows, tables["player_stats"], tables["schedules"], data,
        qb_id_column="actual_qb_id")
    rows = build_offense_rolling_features(rows, tables["pbp_offense_plays"], tables["schedules"], data)
    rows = build_defense_features(rows, tables["pbp_defense_plays"], tables["schedules"],
        season_type=data.season_type, windows=data.defense_rolling_windows)
    rows = build_context_features(rows, tables["schedules"])
    pbp = load_live_pbp(data, live, refresh=refresh)
    halftime = build_live_rows(rows, pbp, data, live)
    q1 = build_q1_rows(halftime, pbp)
    halftime = build_live_flow_features(halftime, pbp, live, data.season_type)
    teams_path = data.cache_dir / "teams.parquet"
    if refresh or not teams_path.exists():
        nfl.load_teams().write_parquet(teams_path)
    teams = pl.read_parquet(teams_path)
    result = {}
    for name, checkpoint in (("q1", q1), ("halftime", halftime)):
        checkpoint = build_season_context_features(checkpoint, tables["schedules"], teams)
        checkpoint, _, _ = add_checkpoint_history(checkpoint)
        checkpoint.select(blog_feature_columns())
        checkpoint.write_parquet(Path(output_dir) / f"{name}_blog_features.parquet")
        result[name] = checkpoint
        print(f"{name}: {checkpoint.height} rows, {len(blog_feature_columns())} candidates", flush=True)
    manifest = {}
    for path in sorted(data.cache_dir.glob("*.parquet")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest[path.name] = digest.hexdigest()
    (Path(output_dir) / "feature_input_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result
