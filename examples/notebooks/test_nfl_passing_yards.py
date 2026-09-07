import json
import pytest
import requests
import nfl_passing_yards_markets as markets

"""Offline correctness checks for the public NFL notebook helpers."""
import numpy as np
import pandas as pd
from nfl_passing_yards_pipeline import LIVE, add_history, feature_columns
from nfl_passing_yards_markets import PublicKalshi, probability_over, select_strategy, summarize, taker_fee


def test_history_excludes_same_week_and_unfinished_games():
    rows = []
    for week, day, value in [(1, 1, 10), (2, 8, 20), (2, 9, 999), (3, 15, 30)]:
        rows.append(
            dict(
                season=2025,
                week=week,
                game_id=str(day),
                checkpoint="q1",
                actual_qb_id="QB",
                team="A",
                opponent_team="B",
                game_date=f"2025-09-{day:02}",
                live_anchor_utc=pd.Timestamp(f"2025-09-{day:02}", tz="UTC"),
                game_end_utc=pd.Timestamp(f"2025-09-{day:02}", tz="UTC") + pd.Timedelta(hours=3),
                remaining_yards=value,
                official_passing_yards=value,
                **{c: value for c in LIVE},
            )
        )
    frame = pd.DataFrame(rows)
    result = add_history(frame)
    assert result.loc[1, "qb_prior3_remaining_yards"] == 10
    assert result.loc[2, "qb_prior3_remaining_yards"] == 10
    frame.loc[0, "game_end_utc"] = pd.Timestamp("2025-10-01", tz="UTC")
    assert np.isnan(add_history(frame).loc[1, "qb_prior3_remaining_yards"])
    assert "official_passing_yards" not in feature_columns(result)
    assert "remaining_yards" not in feature_columns(result)


def test_q1_entry_blocks_halftime():
    rows = pd.DataFrame(
        [
            dict(
                game_id="g",
                actual_qb_name="QB",
                checkpoint=h,
                ticker="t",
                side="yes",
                model_probability=p,
                entry_price=0.2,
                fee=0.02,
                edge=p - 0.22,
                settlement_value=1,
                reason="eligible",
            )
            for h, p in [("q1", 0.5), ("halftime", 0.8)]
        ]
    )
    audit, selected = select_strategy(rows)
    assert selected.checkpoint.tolist() == ["q1"]
    assert audit.iloc[1].reason == "earlier_entry"
    report = summarize(selected)
    combined = report[(report.checkpoint == "combined") & (report.allowance == 0.05)].iloc[0]
    assert abs(combined.profit - 0.73) < 1e-8


def test_halftime_requires_matching_probability_confirmation():
    rows = pd.DataFrame(
        [
            dict(
                game_id="g",
                actual_qb_name="QB",
                checkpoint=h,
                ticker="t",
                side="yes",
                model_probability=p,
                entry_price=price,
                fee=0.02,
                edge=p - price - 0.02,
                settlement_value=0,
                reason="eligible",
            )
            for h, p, price in [("q1", 0.5, 0.45), ("halftime", 0.4, 0.2)]
        ]
    )
    assert select_strategy(rows)[1].empty
    rows.loc[1, "model_probability"] = 0.6
    rows.loc[1, "edge"] = 0.38
    assert select_strategy(rows)[1].checkpoint.tolist() == ["halftime"]


def test_probability_and_fee():
    assert probability_over([0.1, 0.5, 0.9], [100, 200, 300], 200) == 0.5
    assert taker_fee(0.21) == 0.02


def test_quote_dollar_strings_and_future_candle_exclusion():
    class FixtureClient(PublicKalshi):
        def __init__(self):
            pass

        def get(self, *args, **kwargs):
            return {
                "candlesticks": [
                    {"end_period_ts": 1000, "yes_bid": {"close": "0.1200"}, "yes_ask": {"close": "0.2100"}},
                    {"end_period_ts": 1060, "yes_bid": {"close": "0.8000"}, "yes_ask": {"close": "0.9000"}},
                ]
            }

    quote = FixtureClient().quote_at("contract", pd.Timestamp(1012, unit="s", tz="UTC"))
    assert quote["yes_bid"] == 0.12 and quote["yes_ask"] == 0.21
    assert quote["quote_age_seconds"] == 12


def test_unavailable_kalshi_preserves_forecasts(tmp_path, monkeypatch):
    forecast = tmp_path / "predictions.parquet"
    forecast.write_bytes(b"preserve me")

    def fail(*args, **kwargs):
        raise requests.HTTPError("503 unavailable")

    monkeypatch.setattr(markets, "run_backtest", fail)
    assert markets.run_backtest_if_available(None, tmp_path, tmp_path) is None
    status = json.loads((tmp_path / "kalshi_status.json").read_text())
    assert status["available"] is False and status["roi"] is None
    assert forecast.read_bytes() == b"preserve me"


def test_programming_errors_are_not_hidden(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise KeyError("bad schema")

    monkeypatch.setattr(markets, "run_backtest", fail)
    with pytest.raises(KeyError):
        markets.run_backtest_if_available(None, tmp_path, tmp_path)
