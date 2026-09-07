"""Public-source Q1/halftime NFL passing-yard modeling baseline.

Downloads nflverse via nflreadpy; no saved predictions or private artifacts.
Retrospectively corrected play data and play timestamps are NOT a real-time
publication feed. This new baseline does not reproduce the blog's selected model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from synthefy_nori import NoriRegressor


CATEGORIES = ["actual_qb_id", "team", "opponent_team"]
LIVE = ["yards_so_far", "attempts_so_far", "dropbacks_so_far", "sacks_so_far",
        "epa_per_dropback", "cpoe", "air_yards_per_attempt", "sack_rate", "ypa",
        "offense_plays", "offense_pass_rate", "offense_epa", "score_margin"]


def _load(cache_dir, kind, season):
    path = Path(cache_dir) / f"{kind}_{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading nflverse {kind} {season}", flush=True)
        loader = nfl.load_pbp if kind == "pbp" else nfl.load_player_stats
        loader([season]).write_parquet(path)
    manifest_path = Path(cache_dir) / f"{kind}_{season}.source.json"
    manifest_path.write_text(json.dumps({"loader": f"nflreadpy.load_{kind}", "season": season,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}, indent=2))
    return pd.read_parquet(path)


def _sum(frame, col):
    return float(frame[col].fillna(0).sum())


def _mean(frame, col):
    return float(frame[col].mean()) if len(frame) else np.nan


def build_dataset(cache_dir, seasons, decision_delay_seconds=120):
    """One first-passer/team/game/checkpoint row, with earlier-week history.

    QB selection uses the first identified passer by the checkpoint, never final
    attempts. Injury/replacement outcomes remain in evaluation. Missing official
    labels are excluded, not fabricated. No final-game weather or betting lines
    are features because their historical publication times are unknown.
    """
    if decision_delay_seconds < 0:
        raise ValueError("decision_delay_seconds must be nonnegative")
    records = []
    for season in sorted(set(seasons)):
        pbp = _load(cache_dir, "pbp", season)
        stats = _load(cache_dir, "player_stats", season)
        stats = stats.loc[stats.season_type == "REG"]
        qb_ids = set(stats.loc[stats.position == "QB", "player_id"])
        stats = stats.set_index(["game_id", "player_id"])
        pbp = pbp.loc[pbp.season_type == "REG"].copy()
        pbp["_utc"] = pd.to_datetime(pbp.time_of_day, utc=True, errors="coerce")
        for game_id, game in pbp.groupby("game_id", sort=True):
            game = game.sort_values("play_id")
            for quarter, checkpoint in [(1, "q1"), (2, "halftime")]:
                quarter_rows = game.loc[game.qtr == quarter]
                anchor = quarter_rows._utc.max()
                if pd.isna(anchor):
                    continue
                seen = game.loc[game.qtr.between(1, quarter) & (game._utc <= anchor)]
                if seen.empty:
                    continue
                last = seen.iloc[-1]
                for team in [last.home_team, last.away_team]:
                    offense = seen.loc[seen.posteam == team]
                    passers = offense.loc[offense.passer_player_id.isin(qb_ids)]
                    if passers.empty:
                        continue
                    qb_id = passers.iloc[0].passer_player_id
                    if (game_id, qb_id) not in stats.index:
                        continue
                    label = stats.loc[(game_id, qb_id)]
                    if isinstance(label, pd.DataFrame):
                        raise ValueError(f"Duplicate official label: {game_id}/{qb_id}")
                    qb = offense.loc[(offense.passer_player_id == qb_id) |
                                     ((offense.rusher_player_id == qb_id) & (offense.qb_dropback == 1))]
                    drops = qb.loc[qb.qb_dropback == 1]
                    attempts = qb.loc[(qb.pass_attempt == 1) & (qb.sack != 1) & (qb.two_point_attempt != 1)]
                    valid = offense.loc[offense.play_type.isin(["pass", "run"])]
                    yards = _sum(qb.loc[qb.two_point_attempt != 1], "passing_yards")
                    home = team == last.home_team
                    rec = dict(game_id=game_id, game_date=str(last.game_date), season=int(season),
                               week=int(last.week), team=team,
                               opponent_team=last.away_team if home else last.home_team,
                               actual_qb_id=qb_id, actual_qb_name=label.player_display_name,
                               checkpoint=checkpoint, live_anchor_utc=anchor,
                               game_end_utc=game._utc.max() + pd.Timedelta(minutes=5),
                               decision_utc=anchor + pd.Timedelta(seconds=decision_delay_seconds),
                               official_passing_yards=float(label.passing_yards),
                               home=int(home), seconds_remaining=3600-quarter*900,
                               yards_so_far=yards, attempts_so_far=float(len(attempts)),
                               dropbacks_so_far=float(len(drops)), sacks_so_far=_sum(qb, "sack"),
                               epa_per_dropback=_mean(drops, "epa"), cpoe=_mean(attempts, "cpoe"),
                               air_yards_per_attempt=_mean(attempts, "air_yards"),
                               sack_rate=_sum(qb, "sack") / max(len(drops), 1),
                               ypa=yards / max(len(attempts), 1), offense_plays=float(len(valid)),
                               offense_pass_rate=_mean(valid, "qb_dropback"), offense_epa=_mean(valid, "epa"),
                               score_margin=float(last.total_home_score-last.total_away_score)*(1 if home else -1))
                    rec["remaining_yards"] = rec["official_passing_yards"] - yards
                    records.append(rec)
    rows = pd.DataFrame(records).sort_values(["season", "week", "game_id", "checkpoint"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("No timestamped checkpoint rows with official labels")
    return add_history(rows)


def add_history(rows):
    """Group matching-checkpoint histories; embargo the entire current week."""
    rows = rows.copy()
    for checkpoint, idx in rows.groupby("checkpoint").groups.items():
        part = rows.loc[idx]
        for index, row in part.iterrows():
            prior = part.loc[((part.season < row.season) | ((part.season == row.season) & (part.week < row.week)))
                             & (part.game_end_utc < row.live_anchor_utc)]
            qb = prior.loc[prior.actual_qb_id == row.actual_qb_id]
            for window in [3, 8]:
                hist = qb.tail(window)
                for col in LIVE + ["remaining_yards", "official_passing_yards"]:
                    rows.loc[index, f"qb_prior{window}_{col}"] = hist[col].mean()
            rows.loc[index, "qb_prior_games"] = len(qb)
            if not qb.empty:
                rows.loc[index, "rest_days"] = (pd.Timestamp(row.game_date) - pd.Timestamp(qb.iloc[-1].game_date)).days
            for name, key, value in [("offense", "team", row.team), ("defense", "opponent_team", row.opponent_team)]:
                hist = prior.loc[prior[key] == value].tail(8)
                for col in ["ypa", "epa_per_dropback", "sack_rate", "offense_pass_rate", "offense_plays"]:
                    rows.loc[index, f"{name}_prior8_{col}"] = hist[col].mean()
    return rows


def feature_columns(rows):
    return ["week", "home", "seconds_remaining"] + LIVE + [c for c in rows if "_prior" in c or c == "rest_days"]


def predict_weekly(rows, test_season, max_week=18, device="cpu", context_limit=512,
                   min_week=1, model="nori-6m", weekly_update=True):
    """Fresh public Nori predictions, independently per checkpoint and week."""
    if context_limit < 2:
        raise ValueError("context_limit must be >= 2")
    outputs = []
    features = feature_columns(rows)
    model_instance = NoriRegressor(model=model, device=device)
    for week in range(min_week, max_week + 1):
        for checkpoint in ["q1", "halftime"]:
            available = rows.loc[rows.checkpoint == checkpoint]
            train = available.loc[(available.season < test_season) |
                                  (weekly_update & (available.season == test_season) & (available.week < week))]
            train = train.sort_values(["season", "week", "game_id"]).tail(context_limit)
            test = available.loc[(available.season == test_season) & (available.week == week)].copy()
            if test.empty:
                continue
            train = train.loc[train.game_end_utc < test.live_anchor_utc.min()]
            if len(train) < 2:
                raise ValueError(f"Insufficient prior context for {test_season} week {week}")
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            xc = encoder.fit_transform(train[CATEGORIES].fillna("unknown"))
            xq = encoder.transform(test[CATEGORIES].fillna("unknown"))
            xtrain = np.column_stack([train[features].to_numpy(dtype=float), xc])
            xtest = np.column_stack([test[features].to_numpy(dtype=float), xq])
            print(f"Nori {test_season} W{week} {checkpoint}: {len(train)} context, {len(test)} queries", flush=True)
            model_instance.fit(xtrain, train.remaining_yards.to_numpy())
            dist = model_instance.predict(xtest, output_type="full")
            quantiles = np.asarray(dist["quantiles"]) + test.yards_so_far.to_numpy()[:, None]
            taus = np.asarray(dist["taus"])
            test["predicted_mean"] = np.asarray(dist["mean"]).reshape(-1) + test.yards_so_far.to_numpy()
            test["quantile_levels"] = [taus.tolist() for _ in range(len(test))]
            test["quantile_values"] = quantiles.tolist()
            test["context_rows"] = len(train)
            test["model"] = model
            for tau, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
                test[label] = [np.interp(tau, taus, q) for q in quantiles]
            outputs.append(test)
    if not outputs:
        raise ValueError("No query rows in requested season/week range")
    return pd.concat(outputs, ignore_index=True)


def prediction_metrics(predictions):
    results = []
    for checkpoint, part in predictions.groupby("checkpoint"):
        actual = part.official_passing_yards.to_numpy()
        error = part.predicted_mean.to_numpy() - actual
        q = np.stack(part.quantile_values)
        taus = np.asarray(part.iloc[0].quantile_levels)
        residual = actual[:, None] - q
        results.append(dict(checkpoint=checkpoint, rows=len(part), mae=float(np.abs(error).mean()),
                            rmse=float(np.sqrt((error**2).mean())),
                            pinball_loss=float(np.maximum(taus*residual, (taus-1)*residual).mean()),
                            p10_p90_coverage=float(((actual >= part.p10) & (actual <= part.p90)).mean())))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/nfl_passing_yards")
    parser.add_argument("--start-season", type=int, default=2023)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--max-week", type=int, default=1)
    parser.add_argument("--context-limit", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="nori-6m")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    rows = build_dataset(args.cache_dir, range(args.start_season, args.test_season+1))
    rows.to_parquet(Path(args.cache_dir)/"checkpoint_features.parquet", index=False)
    print(f"Built {len(rows)} checkpoint rows, {len(feature_columns(rows))+len(CATEGORIES)} features")
    if not args.features_only:
        pred = predict_weekly(rows, args.test_season, args.max_week, args.device, args.context_limit, model=args.model)
        pred.to_parquet(Path(args.cache_dir)/"predictions.parquet", index=False)
        metrics = prediction_metrics(pred)
        print(metrics.to_string(index=False))
        (Path(args.cache_dir)/"metrics.json").write_text(json.dumps(metrics.to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
