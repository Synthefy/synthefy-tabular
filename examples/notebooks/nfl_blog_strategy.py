"""Original NFL blog strategy, ported without research-workspace dependencies.

Historical minute quotes are not evidence of available quantity or guaranteed fills.
"""

from __future__ import annotations
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
import numpy as np
import polars as pl
import requests
from nfl_passing_yards_markets import PublicKalshi

UTC = timezone.utc
EDGE_THRESHOLD = 0.10
NUMERIC_TOLERANCE = 1e-12
KEYS = ["game_id", "team"]
IDENTITY_COLUMNS = (
    "week",
    "game_id",
    "team",
    "opponent_team",
    "actual_qb_id",
    "actual_qb_name",
    "official_passing_yards",
)
SELECTED_COLUMNS = (
    "live_anchor_utc",
    "live_decision_utc",
    "live_qb_passing_yards",
    "nori_median",
    "ticker",
    "line",
    "side",
    "model_probability",
    "yes_bid",
    "yes_ask",
    "spread",
    "quote_ts",
    "quote_age_seconds",
    "entry_price",
    "fee",
    "expected_net_edge",
    "settlement_value",
    "capital",
    "profit",
)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return _parse_time(str(value))


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def market_player_name(title: str) -> str:
    return re.split(r"\s+records\s+\d+\+|:\s*\d+\+", title, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def kalshi_taker_fee(price: float, contracts: int, rate: float) -> float:
    raw_fee = rate * contracts * price * (1.0 - price)
    return math.ceil(raw_fee * 100.0 - 1e-12) / 100.0


def bid_ask_spread(bid: float, ask: float) -> float:
    """Respect exact decimal price cutoffs (0.55 - 0.45 is ten cents)."""
    return float(Decimal(str(ask)) - Decimal(str(bid)))


def _close_value(side: dict[str, Any] | None) -> float | None:
    if not side:
        return None
    if side.get("close_dollars") is not None:
        return float(side["close_dollars"])
    value = side.get("close")
    if value is None:
        return None
    # Fixed-point strings are dollars; legacy integer fields are cents.
    return float(value) if isinstance(value, str) else float(value) / 100.0


def _settlement_value(market: dict[str, Any]) -> float:
    value = market.get("settlement_value_dollars")
    if value is not None:
        return float(value)
    if market.get("result") == "yes":
        return 1.0
    if market.get("result") == "no":
        return 0.0
    raise ValueError(f"Market {market['ticker']} lacks a final settlement value")


def match_live_markets_to_predictions(
    markets: list[dict[str, Any]],
    predictions: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    required = {
        "game_id",
        "week",
        "kickoff_utc",
        "live_anchor_utc",
        "live_decision_utc",
        "team",
        "opponent_team",
        "actual_qb_id",
        "actual_qb_name",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"live predictions are missing market-match columns: {missing}")

    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in predictions.select(*sorted(required)).to_dicts():
        by_name.setdefault(normalize_name(str(row["actual_qb_name"])), []).append(row)

    matched: list[dict[str, Any]] = []
    missing_name = 0
    ambiguous = 0
    for market in markets:
        if market.get("floor_strike") is None or not market.get("expected_expiration_time"):
            continue
        candidates = by_name.get(normalize_name(market_player_name(str(market.get("title", "")))), [])
        expiration = _parse_time(str(market["expected_expiration_time"]))
        candidates = [
            row
            for row in candidates
            if timedelta(hours=1) <= expiration - _as_datetime(row["kickoff_utc"]) <= timedelta(hours=8)
        ]
        if not candidates:
            missing_name += 1
            continue
        if len(candidates) != 1:
            ambiguous += 1
            continue
        matched.append(
            {
                **market,
                **candidates[0],
                "model_line": float(market["floor_strike"]),
            }
        )
    return matched, {
        "input_markets": len(markets),
        "matched_markets": len(matched),
        "unmatched_market_rows": missing_name,
        "ambiguous_market_rows": ambiguous,
    }


def probability_over_line(taus: list[float], quantile_values: list[float], line: float) -> float:
    levels = np.asarray(taus, dtype=np.float64)
    values = np.asarray(quantile_values, dtype=np.float64)
    if levels.ndim != 1 or values.ndim != 1 or levels.size != values.size or levels.size < 2:
        raise ValueError("invalid predictive quantile row")
    if not np.isfinite(levels).all() or not np.isfinite(values).all():
        raise ValueError("predictive quantile row contains non-finite values")
    values = np.maximum.accumulate(values)
    probability = 1.0 - np.interp(line, values, levels, left=0.0, right=1.0)
    return float(np.clip(probability, 0.0, 1.0))


def _empty_decision(prediction: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "game_id": prediction["game_id"],
        "week": prediction["week"],
        "team": prediction["team"],
        "opponent_team": prediction["opponent_team"],
        "actual_qb_id": prediction["actual_qb_id"],
        "actual_qb_name": prediction["actual_qb_name"],
        "live_anchor_utc": prediction["live_anchor_utc"],
        "live_decision_utc": prediction["live_decision_utc"],
        "live_qb_passing_yards": prediction["live_qb_passing_yards"],
        "official_passing_yards": prediction["official_passing_yards"],
        "nori_median": prediction["nori_median"],
        "ticker": None,
        "line": None,
        "side": None,
        "model_probability": None,
        "yes_bid": None,
        "yes_ask": None,
        "spread": None,
        "quote_ts": None,
        "quote_age_seconds": None,
        "entry_price": None,
        "fee": None,
        "expected_net_edge": None,
        "settlement_value": None,
        "capital": None,
        "profit": None,
        "bet_taken": False,
        "decision_status": status,
        "realized_profit": None,
    }


def run_live_primary_strategy(
    matched_markets: list[dict[str, Any]],
    predictions: pl.DataFrame,
    quotes: dict[str, dict[str, Any]],
    strategy: SimpleNamespace,
    live_config: SimpleNamespace,
    taker_fee_rate: float,
    *,
    line_policy: str = "max_edge",
    familywise_arms: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    if line_policy not in {"max_edge", "market_balanced"}:
        raise ValueError(f"unsupported line policy: {line_policy}")
    prediction_map = {(str(row["game_id"]), str(row["actual_qb_id"])): row for row in predictions.to_dicts()}
    available_keys: set[tuple[str, str]] = set()
    quoted_keys: set[tuple[str, str]] = set()
    fresh_keys: set[tuple[str, str]] = set()
    valid_spread_keys: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    available_markets = 0
    quoted_markets = 0
    fresh_markets = 0
    for market in matched_markets:
        key = (str(market["game_id"]), str(market["actual_qb_id"]))
        prediction = prediction_map.get(key)
        if prediction is None:
            continue
        decision = _as_datetime(prediction["live_decision_utc"])
        created = _parse_time(str(market["created_time"]))
        open_time = _parse_time(str(market["open_time"]))
        close_time = _parse_time(str(market["close_time"]))
        if created > decision or open_time > decision or close_time <= decision:
            continue
        available_keys.add(key)
        available_markets += 1

        quote = quotes.get(str(market["ticker"]), {})
        quote_ts = quote.get("quote_ts")
        yes_bid = quote.get("yes_bid")
        yes_ask = quote.get("yes_ask")
        if quote_ts is None or yes_bid is None or yes_ask is None:
            continue
        quoted_keys.add(key)
        quoted_markets += 1
        quote_time = datetime.fromtimestamp(int(quote_ts), tz=UTC)
        anchor = _as_datetime(prediction["live_anchor_utc"])
        quote_age_seconds = (decision - quote_time).total_seconds()
        if (
            quote_time < anchor
            or quote_time > decision
            or quote_age_seconds > live_config.maximum_quote_age_minutes * 60
        ):
            continue
        fresh_keys.add(key)
        fresh_markets += 1

        spread = bid_ask_spread(float(yes_bid), float(yes_ask))
        if not 0.0 <= float(yes_bid) <= float(yes_ask) <= 1.0 or spread > strategy.max_spread:
            continue
        valid_spread_keys.add(key)
        probability_yes = probability_over_line(
            prediction["nori_quantile_taus"],
            prediction["nori_quantile_values"],
            float(market["model_line"]),
        )
        settlement_yes = _settlement_value(market)
        for side, probability, entry_price, settlement in (
            ("yes", probability_yes, float(yes_ask), settlement_yes),
            ("no", 1.0 - probability_yes, 1.0 - float(yes_bid), 1.0 - settlement_yes),
        ):
            fee = kalshi_taker_fee(entry_price, strategy.contracts_per_qb_game, taker_fee_rate)
            expected_net_edge = probability - entry_price - fee / strategy.contracts_per_qb_game
            capital = entry_price * strategy.contracts_per_qb_game + fee
            profit = settlement * strategy.contracts_per_qb_game - capital
            candidates.append(
                {
                    "game_id": prediction["game_id"],
                    "week": prediction["week"],
                    "team": prediction["team"],
                    "opponent_team": prediction["opponent_team"],
                    "actual_qb_id": prediction["actual_qb_id"],
                    "actual_qb_name": prediction["actual_qb_name"],
                    "live_anchor_utc": prediction["live_anchor_utc"],
                    "live_decision_utc": prediction["live_decision_utc"],
                    "live_qb_passing_yards": prediction["live_qb_passing_yards"],
                    "official_passing_yards": prediction["official_passing_yards"],
                    "nori_median": prediction["nori_median"],
                    "ticker": market["ticker"],
                    "line": float(market["model_line"]) + 0.5,
                    "side": side,
                    "model_probability": probability,
                    "yes_bid": float(yes_bid),
                    "yes_ask": float(yes_ask),
                    "spread": spread,
                    "quote_ts": int(quote_ts),
                    "quote_age_seconds": quote_age_seconds,
                    "entry_price": entry_price,
                    "fee": fee,
                    "expected_net_edge": expected_net_edge,
                    "settlement_value": settlement,
                    "capital": capital,
                    "profit": profit,
                }
            )

    def candidate_priority(row: dict[str, Any]) -> tuple:
        prefix = (row["game_id"], row["actual_qb_id"])
        if line_policy == "market_balanced":
            # Choose the strike using the market alone, before comparing sides.
            # Tied strikes use ticker order, never model edge or settlement.
            return prefix + (
                round(abs((row["yes_bid"] + row["yes_ask"]) / 2.0 - 0.5), 10),
                row["ticker"],
                -row["expected_net_edge"],
                row["side"],
            )
        return (
            row["game_id"],
            row["actual_qb_id"],
            -row["expected_net_edge"],
            row["ticker"],
            row["side"],
        )

    candidates.sort(key=candidate_priority)
    best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate["game_id"]), str(candidate["actual_qb_id"]))
        best_by_key.setdefault(key, candidate)

    decisions: list[dict[str, Any]] = []
    for key, prediction in prediction_map.items():
        if key in best_by_key:
            decision = best_by_key[key]
            decision["bet_taken"] = decision["expected_net_edge"] >= strategy.primary_min_net_edge
            decision["decision_status"] = "bet" if decision["bet_taken"] else "pass"
            decision["realized_profit"] = decision["profit"] if decision["bet_taken"] else 0.0
        elif key not in available_keys:
            decision = _empty_decision(prediction, "no_live_market")
        elif key not in quoted_keys:
            decision = _empty_decision(prediction, "no_post_anchor_quote")
        elif key not in fresh_keys:
            decision = _empty_decision(prediction, "stale_quote")
        elif key not in valid_spread_keys:
            decision = _empty_decision(prediction, "spread_too_wide")
        else:
            decision = _empty_decision(prediction, "no_eligible_candidate")
        decisions.append(decision)

    decisions.sort(key=lambda row: (row["week"], row["game_id"], row["team"]))
    selected = [decision for decision in decisions if decision["bet_taken"]]
    decision_frame = pl.DataFrame(decisions, infer_schema_length=None)
    trades = decision_frame.filter(pl.col("bet_taken"))
    return decision_frame, trades, {}


def _float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def q1_yes_probability(prediction: dict[str, Any], integer_line: float) -> float:
    return probability_over_line(
        prediction["nori_quantile_taus"],
        prediction["nori_quantile_values"],
        integer_line - 0.5,
    )


def build_confirmed_decisions(
    q1: pl.DataFrame,
    halftime: pl.DataFrame,
    q1_predictions: pl.DataFrame,
    *,
    enforce_frozen_coverage: bool = True,
) -> pl.DataFrame:
    """Apply Q1 priority and inclusive same-strike probability confirmation."""
    q1 = q1.sort(KEYS)
    halftime = halftime.sort(KEYS)
    q1_predictions = q1_predictions.sort(KEYS)
    if not q1.select(KEYS).equals(halftime.select(KEYS)) or not q1.select(KEYS).equals(q1_predictions.select(KEYS)):
        raise ValueError("probability-confirmation sources are not aligned")
    records: list[dict[str, Any]] = []
    for q1_row, half_row, prediction in zip(
        q1.to_dicts(),
        halftime.to_dicts(),
        q1_predictions.to_dicts(),
        strict=True,
    ):
        q1_eligible = bool(q1_row["bet_taken"])
        half_cost_eligible = (
            bool(half_row["bet_taken"])
            and _float(half_row["expected_net_edge"], "halftime expected edge") >= EDGE_THRESHOLD - NUMERIC_TOLERANCE
        )
        halftime_reached = not q1_eligible
        q1_side_probability = None
        halftime_side_probability = None
        probability_change = None
        probability_confirmed = None
        confirmation_ticker = None
        confirmation_line = None
        confirmation_side = None
        if q1_eligible:
            selected = q1_row
            selected_horizon = "q1"
            status = "bet"
        elif half_cost_eligible:
            confirmation_ticker = str(half_row["ticker"])
            confirmation_line = _float(half_row["line"], "halftime line")
            confirmation_side = str(half_row["side"])
            q1_probability_yes = q1_yes_probability(prediction, confirmation_line)
            if confirmation_side == "yes":
                q1_side_probability = q1_probability_yes
            elif confirmation_side == "no":
                q1_side_probability = 1.0 - q1_probability_yes
            else:
                raise ValueError("probability-confirmation source has an invalid side")
            halftime_side_probability = _float(half_row["model_probability"], "halftime side probability")
            probability_change = halftime_side_probability - q1_side_probability
            probability_confirmed = halftime_side_probability + NUMERIC_TOLERANCE >= q1_side_probability
            if probability_confirmed:
                selected = half_row
                selected_horizon = "halftime"
                status = "bet"
            else:
                selected = None
                selected_horizon = "abstain"
                status = "pass_probability_confirmation"
        else:
            selected = None
            selected_horizon = "abstain"
            status = "abstain"
        record = {column: q1_row[column] for column in IDENTITY_COLUMNS}
        record.update(
            {
                "q1_eligible": q1_eligible,
                "halftime_cost_eligible": half_cost_eligible,
                "halftime_reached": halftime_reached,
                "probability_confirmed": probability_confirmed,
                "q1_selected_side_probability": q1_side_probability,
                "halftime_selected_side_probability": halftime_side_probability,
                "selected_side_probability_change": probability_change,
                "confirmation_ticker": confirmation_ticker,
                "confirmation_line": confirmation_line,
                "confirmation_side": confirmation_side,
                "selected_horizon": selected_horizon,
                "bet_taken": selected is not None,
                "decision_status": status,
            }
        )
        for column in SELECTED_COLUMNS:
            record[column] = selected[column] if selected is not None else None
        record["realized_profit"] = selected["profit"] if selected is not None else None
        records.append(record)
    decisions = pl.DataFrame(records, infer_schema_length=None).sort(KEYS)
    return decisions


def quote_after_anchor(client, market):
    """Same anchor-to-decision minute-close request as the original experiment."""
    anchor = int(_as_datetime(market["live_anchor_utc"]).timestamp())
    decision = int(_as_datetime(market["live_decision_utc"]).timestamp())
    payload = client.get(
        f"historical/markets/{quote(market['ticker'], safe='')}/candlesticks",
        start_ts=anchor,
        end_ts=decision,
        period_interval=1,
    )
    candles = [c for c in payload["candlesticks"] if anchor <= int(c["end_period_ts"]) <= decision]
    result = dict(
        ticker=market["ticker"], anchor_ts=anchor, decision_ts=decision, quote_ts=None, yes_bid=None, yes_ask=None
    )
    if candles:
        candle = max(candles, key=lambda c: int(c["end_period_ts"]))
        result.update(
            quote_ts=int(candle["end_period_ts"]),
            yes_bid=_close_value(candle.get("yes_bid")),
            yes_ask=_close_value(candle.get("yes_ask")),
        )
    return result


def summarize(decisions, execution_cost=0.05):
    if not math.isfinite(execution_cost) or execution_cost < 0:
        raise ValueError("Execution allowance must be finite and nonnegative")
    trades = decisions.filter(pl.col("bet_taken")).to_dicts()
    capital = sum(r["capital"] + execution_cost for r in trades)
    pnl = sum(r["profit"] - execution_cost for r in trades)
    return {
        "status": "quote_based_simulation",
        "bets": len(trades),
        "games": len({r["game_id"] for r in trades}),
        "wins": sum(r["settlement_value"] == 1 for r in trades),
        "capital": capital,
        "pnl": pnl,
        "roi": pnl / capital if capital else None,
        "execution_cost": execution_cost,
        "limitation": "Recorded quotes do not establish available size or fills. 2025 was used for strategy exploration.",
    }


def replay_blog_strategy(q1, halftime, markets, quotes_by_horizon, execution_cost=0.05):
    """Original selections: Q1 balanced market, then confirmed max-edge halftime.

    Input quantiles predict final passing yards (not remaining yards). No future
    transaction price or settlement enters the candidate-ranking rule. Quotes
    are raw public responses transformed by quote_after_anchor.
    """
    if q1.is_empty() or halftime.is_empty():
        raise ValueError("Both checkpoints need predictions")
    for predictions in (q1, halftime):
        if predictions.select(KEYS).is_duplicated().any():
            raise ValueError("Duplicate QB-game prediction keys")
    decisions = {}
    coverage = {}
    live = SimpleNamespace(maximum_quote_age_minutes=5)
    for horizon, predictions in (("q1", q1), ("halftime", halftime)):
        strategy = SimpleNamespace(
            contracts_per_qb_game=1, primary_min_net_edge=0.10 if horizon == "q1" else 0.05, max_spread=0.10
        )
        matched, counts = match_live_markets_to_predictions(markets, predictions)
        decisions[horizon], _, _ = run_live_primary_strategy(
            matched,
            predictions,
            quotes_by_horizon[horizon],
            strategy,
            live,
            0.07,
            line_policy="market_balanced" if horizon == "q1" else "max_edge",
        )
        coverage[horizon] = counts
    combined = build_confirmed_decisions(decisions["q1"], decisions["halftime"], q1, enforce_frozen_coverage=False)
    combined = combined.with_columns(
        pl.when(pl.col("bet_taken")).then(pl.col("capital") + execution_cost).otherwise(None).alias("stressed_capital"),
        pl.when(pl.col("bet_taken")).then(pl.col("profit") - execution_cost).otherwise(None).alias("stressed_profit"),
    )
    report = summarize(combined, execution_cost)
    report["coverage"] = coverage
    report["checkpoints"] = {
        h: summarize(combined.filter(pl.col("selected_horizon") == h), execution_cost) for h in ("q1", "halftime")
    }
    return combined, report


def run_kalshi_backtest(q1, halftime, cache_dir, execution_cost=0.05):
    """Fetch public archives; network unavailability never invalidates forecasts.

    A failed page/quote aborts only the optional market stage. Do not report ROI
    from an incomplete download. Software/data-contract errors remain visible.
    """
    client = PublicKalshi(Path(cache_dir))
    try:
        markets = client.markets()
        quotes = {}
        for horizon, predictions in (("q1", q1), ("halftime", halftime)):
            matched, _ = match_live_markets_to_predictions(markets, predictions)
            quotes[horizon] = {}
            for index, market in enumerate(matched):
                quotes[horizon][market["ticker"]] = quote_after_anchor(client, market)
                if (index + 1) % 100 == 0:
                    print(f"Kalshi {horizon}: {index + 1}/{len(matched)} quotes", flush=True)
        return replay_blog_strategy(q1, halftime, markets, quotes, execution_cost)
    except (requests.RequestException, FileNotFoundError, TimeoutError) as error:
        return pl.DataFrame(), {
            "status": "unavailable",
            "reason": str(error),
            "roi": None,
            "note": "Predictions are retained. Rerun this optional stage when public history is available.",
        }
