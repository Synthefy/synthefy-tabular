"""Offline checks for the original blog's exact strategy semantics."""
from datetime import datetime, timezone
import polars as pl
import pytest
import requests
import nfl_blog_strategy as strategy


def fixture():
    row = dict(game_id="2025_01_A_B", week=1, kickoff_utc="2025-09-01T00:00:00+00:00",
               live_anchor_utc="2025-09-01T00:30:00+00:00", live_decision_utc="2025-09-01T00:32:00+00:00",
               team="A", opponent_team="B", actual_qb_id="qb", actual_qb_name="Test QB",
               live_qb_passing_yards=50., official_passing_yards=250., nori_median=250.,
               nori_quantile_taus=[.1,.5,.9], nori_quantile_values=[150.,250.,350.])
    market = dict(ticker="TEST", title="Test QB records 200+ passing yards", floor_strike=199.5,
                  expected_expiration_time="2025-09-01T04:00:00+00:00",
                  created_time="2025-08-30T00:00:00+00:00", open_time="2025-08-30T00:00:00+00:00",
                  close_time="2025-09-01T04:00:00+00:00", result="yes")
    quote = dict(quote_ts=int(datetime(2025,9,1,0,32,tzinfo=timezone.utc).timestamp()), yes_bid=.4, yes_ask=.45)
    return row, market, quote


def replay(row, market, q, half=None, first_quote=None):
    return strategy.replay_blog_strategy(pl.DataFrame([row]), pl.DataFrame([half or row]), [market],
                                        {"q1":{"TEST":first_quote or q}, "halftime":{"TEST":q}})


def test_q1_priority_and_fees():
    row,m,q=fixture()
    d,r=replay(row,m,q)
    assert r["bets"]==1 and d["selected_horizon"][0]=="q1"
    assert d["entry_price"][0]==.45 and d["fee"][0]==.02
    assert r["capital"]==pytest.approx(.52)
    assert r["pnl"]==pytest.approx(.48)


def test_half_confirmation_does_not_require_q1_market():
    row,m,q=fixture()
    d,r=replay(row,m,q,first_quote={"quote_ts":None})
    assert r["bets"]==1 and d["selected_horizon"][0]=="halftime"
    assert d["probability_confirmed"][0]


def test_lower_half_probability_rejected():
    row,m,q=fixture()
    half={**row,"nori_quantile_values":[100.,200.,300.]}
    q={**q,"yes_bid":.2,"yes_ask":.25}
    d,r=replay(row,m,q,half=half,first_quote={"quote_ts":None})
    assert r["bets"]==0 and r["roi"] is None
    assert d["decision_status"][0]=="pass_probability_confirmation"


def test_q1_strike_selected_by_market_not_largest_edge():
    row,m,q=fixture()
    low={**m,"ticker":"LOW","floor_strike":99.5}
    low_q={**q,"yes_bid":.1,"yes_ask":.15}
    matched,_=strategy.match_live_markets_to_predictions([m,low],pl.DataFrame([row]))
    from types import SimpleNamespace
    d,_,_=strategy.run_live_primary_strategy(matched,pl.DataFrame([row]),{"TEST":q,"LOW":low_q},
        SimpleNamespace(contracts_per_qb_game=1,primary_min_net_edge=.10,max_spread=.10),
        SimpleNamespace(maximum_quote_age_minutes=5),.07,line_policy="market_balanced")
    assert d["ticker"][0]=="TEST"


def test_future_or_pre_anchor_quotes_rejected():
    row,m,q=fixture()
    for delta in [1,-121]:
        changed={**q,"quote_ts":q["quote_ts"]+delta}
        _,r=replay(row,m,changed)
        assert r["bets"]==0


def test_ten_cent_decimal_spread_is_inclusive():
    assert strategy.bid_ask_spread(.45,.55)==.10
    row,m,q=fixture()
    d,r=replay(row,m,{**q,"yes_bid":.45,"yes_ask":.55})
    assert r["bets"]==1


def test_market_closed_before_decision_rejected():
    row,m,q=fixture()
    m={**m,"close_time":row["live_decision_utc"]}
    _,r=replay(row,m,q)
    assert r["bets"]==0


def test_unavailable_retains_no_roi(monkeypatch,tmp_path):
    def fail(self):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(strategy.PublicKalshi,"markets",fail)
    row,_,_=fixture()
    d,r=strategy.run_kalshi_backtest(pl.DataFrame([row]),pl.DataFrame([row]),tmp_path)
    assert d.is_empty() and r["status"]=="unavailable" and r["roi"] is None


def test_software_errors_do_not_masquerade_as_network(monkeypatch,tmp_path):
    def fail(self):
        raise ValueError("bad schema")
    monkeypatch.setattr(strategy.PublicKalshi,"markets",fail)
    row,_,_=fixture()
    with pytest.raises(ValueError,match="bad schema"):
        strategy.run_kalshi_backtest(pl.DataFrame([row]),pl.DataFrame([row]),tmp_path)


def test_quote_close_uses_latest_completed_candle():
    row,m,q=fixture()
    class Client:
        def get(self,path,**params):
            assert params["end_ts"]==q["quote_ts"]
            return {"candlesticks":[
                {"end_period_ts":q["quote_ts"],"yes_bid":{"close_dollars":"0.12"},"yes_ask":{"close_dollars":"0.21"}},
                {"end_period_ts":q["quote_ts"]+60,"yes_bid":{"close_dollars":"0.90"},"yes_ask":{"close_dollars":"0.91"}}]}
    result=strategy.quote_after_anchor(Client(),{**m,**row})
    assert result["yes_ask"]==.21 and result["yes_bid"]==.12


@pytest.mark.parametrize("raw", [{"close_dollars": "0.21"}, {"close": "0.21"}, {"close": 21}])
def test_quote_price_formats(raw):
    assert strategy._close_value(raw) == .21
