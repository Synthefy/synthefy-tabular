"""Public, cached Kalshi quote-based replay. No live orders or assumed fills.

This deliberately uses public minute-candle closing quotes, not order-book depth.
Freshly computed predictions need not reproduce any previously published result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


class MarketDataUnavailable(RuntimeError):
    """Historical data cannot support a complete betting report."""


class PublicKalshi:
    """Unauthenticated GET-only client; immutable request-keyed provenance cache."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir) / "kalshi"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )))

    def get(self, endpoint, **params):
        url = f"{BASE_URL}/{endpoint}"
        identity = json.dumps([url, params], sort_keys=True)
        path = self.cache_dir / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if hashlib.sha256(json.dumps(record["payload"], sort_keys=True).encode()).hexdigest() != record["payload_sha256"]:
                raise ValueError(f"Corrupt cached payload: {path}")
            return record["payload"]
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        record = dict(url=response.url, retrieved_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                      sha256=hashlib.sha256(response.content).hexdigest(),
                      payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload=payload)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)
        return payload

    def markets(self):
        """Download all archived passing-yard markets, following every cursor."""
        result, cursor, seen = [], "", set()
        while True:
            params = {"series_ticker": "KXNFLPASSYDS", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = self.get("historical/markets", **params)
            result.extend(page["markets"])
            cursor = page.get("cursor", "")
            if not cursor:
                break
            if cursor in seen:
                raise RuntimeError("Repeated historical-market cursor; download is incomplete")
            seen.add(cursor)
        return list({m["ticker"]: m for m in result}.values())

    def quote_at(self, ticker, decision_utc, max_age_seconds=300):
        """Last completed minute candle at/before decision; never use future quotes."""
        decision = pd.Timestamp(decision_utc)
        if pd.isna(decision) or decision.tzinfo is None:
            raise ValueError("Decision timestamp must be known and timezone-aware")
        end = int(decision.timestamp())
        payload = self.get(f"historical/markets/{quote(ticker, safe='')}/candlesticks",
                           start_ts=end - max_age_seconds - 60, end_ts=end, period_interval=1)
        candles = [c for c in payload["candlesticks"]
                   if 0 <= decision.timestamp() - c["end_period_ts"] <= max_age_seconds]
        if not candles:
            return {}
        candle = max(candles, key=lambda c: c["end_period_ts"])
        def close(field):
            values = candle.get(field) or {}
            if values.get("close_dollars") is not None:
                return float(values["close_dollars"])
            value = values.get("close")
            if value is None:
                return None
            # Historical fixed-point strings are dollars; legacy integer fields are cents.
            return float(value) if isinstance(value, str) else float(value) / 100
        return dict(quote_utc=pd.Timestamp(candle["end_period_ts"], unit="s", tz="UTC").isoformat(),
                    quote_age_seconds=decision.timestamp() - candle["end_period_ts"],
                    yes_bid=close("yes_bid"), yes_ask=close("yes_ask"))


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def market_matches(market, row):
    """Require exact normalized player and original scheduled game date/team pair."""
    name = re.split(r"\s+records\s+\d+\+|:\s*\d+\+", market.get("title", ""),
                    maxsplit=1, flags=re.I)[0]
    if normalize_name(name) != normalize_name(row["actual_qb_name"]):
        return False
    day = pd.Timestamp(row["game_date"]).strftime("%y%b%d").upper()
    suffix = market.get("event_ticker", "").removeprefix("KXNFLPASSYDS-")
    aliases = {"LA": "LAR", "OAK": "LV", "SD": "LAC"}
    team = aliases.get(row["team"], row["team"])
    opponent = aliases.get(row["opponent_team"], row["opponent_team"])
    return suffix in (day + team + opponent, day + opponent + team)


def probability_over(levels, values, threshold):
    levels, values = np.asarray(levels, float), np.asarray(values, float)
    if (levels.ndim != 1 or values.shape != levels.shape or len(levels) < 2
            or not np.isfinite(levels).all() or not np.isfinite(values).all()
            or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1):
        raise ValueError("Invalid predictive quantiles")
    # Explicit piecewise-linear CDF approximation, monotone-repaired quantiles.
    return float(1 - np.interp(threshold, np.maximum.accumulate(values), levels, left=0, right=1))


def taker_fee(price):
    """Modeled one-contract general taker fee, rounded upward to cents."""
    p = Decimal(str(price))
    return float((Decimal("0.07") * p * (1 - p)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def build_candidates(predictions, client, markets=None, max_age_seconds=300, max_spread=.10):
    """Return every matched side and rejected row, including missing data reasons.

    Network errors stop the run (successful requests remain cached), rather than
    silently treating failed downloads as absent liquidity.
    """
    if predictions.empty:
        raise ValueError("No predictions supplied")
    markets = client.markets() if markets is None else markets
    rows = []
    for row in predictions.to_dict("records"):
        base = {k: row[k] for k in ["game_id", "week", "actual_qb_name", "checkpoint", "decision_utc"]}
        if pd.isna(row["decision_utc"]):
            rows.append({**base, "reason": "missing_decision_timestamp"})
            continue
        matches = [m for m in markets if market_matches(m, row) and m.get("floor_strike") is not None]
        if not matches:
            rows.append({**base, "reason": "no_matching_market"})
        for market in matches:
            ticker, threshold = market["ticker"], float(market["floor_strike"])
            if market.get("strike_type") not in ("greater", "structured") or not re.search(r"records \d+\+ passing yards", market.get("title", ""), re.I):
                rows.append({**base, "ticker": ticker, "reason": "unsupported_contract"})
                continue
            p_yes = probability_over(row["quantile_levels"], row["quantile_values"], threshold)
            snapshot = client.quote_at(ticker, row["decision_utc"], max_age_seconds)
            bid, ask = snapshot.get("yes_bid"), snapshot.get("yes_ask")
            reason = "eligible"
            if bid is None or ask is None:
                reason = "missing_quote"
            elif not 0 <= bid <= ask <= 1:
                reason = "invalid_quote"
            elif ask - bid > max_spread + 1e-9:
                reason = "wide_spread"
            settlement = market.get("settlement_value_dollars")
            if settlement is None:
                settlement = {"yes": 1, "no": 0}.get(market.get("result"))
            if settlement is not None and not 0 <= float(settlement) <= 1:
                raise ValueError(f"Invalid settlement value for {ticker}")
            for side, probability, price in [("yes", p_yes, ask), ("no", 1 - p_yes, None if bid is None else 1 - bid)]:
                fee = taker_fee(price) if price is not None and 0 < price < 1 else None
                rows.append({**base, **snapshot, "ticker": ticker, "threshold": threshold,
                             "line": math.floor(threshold) + 1, "side": side,
                             "model_probability": probability, "entry_price": price, "fee": fee,
                             "edge": probability - price - fee if fee is not None else None,
                             "settlement_value": None if settlement is None else (float(settlement) if side == "yes" else 1 - float(settlement)),
                             "reason": "nontradeable_price" if reason == "eligible" and fee is None else reason})
    return pd.DataFrame(rows)


def select_strategy(candidates, min_edge=.10):
    """Freeze Q1 first, halftime fallback; select highest edge deterministically.

    Settlement never influences selection. Only one one-contract position per
    QB/game. Halftime confirmation compares the identical ticker and side.
    """
    audit = candidates.copy()
    audit["selected"] = False
    for _, group in audit.groupby(["game_id", "actual_qb_name"], sort=False):
        previous = {}
        bought = False
        for checkpoint in ["q1", "halftime"]:
            choices = []
            for index, row in group[group.checkpoint == checkpoint].iterrows():
                key = (row.get("ticker"), row.get("side"))
                if checkpoint == "q1" and pd.notna(row.get("model_probability")):
                    previous[key] = row["model_probability"]
                if row["reason"] != "eligible":
                    continue
                reason = None
                if bought:
                    reason = "earlier_entry"
                elif row["edge"] < min_edge - 1e-12:
                    reason = "below_edge"
                elif checkpoint == "halftime" and key not in previous:
                    reason = "missing_q1_confirmation"
                elif checkpoint == "halftime" and row["model_probability"] < previous[key]:
                    reason = "probability_not_confirmed"
                if reason:
                    audit.at[index, "reason"] = reason
                else:
                    choices.append(index)
            if choices:
                winner = sorted(choices, key=lambda i: (-audit.at[i, "edge"], audit.at[i, "ticker"], audit.at[i, "side"]))[0]
                for index in choices:
                    audit.at[index, "reason"] = "selected" if index == winner else "lower_priority"
                audit.at[winner, "selected"] = True
                bought = True
    selected = audit[audit.selected].copy()
    selected["stake_contracts"] = 1
    if len(selected):
        selected["capital"] = selected.entry_price + selected.fee
        selected["profit"] = selected.settlement_value - selected.capital
        for cents in (5, 10):
            selected[f"capital_plus_{cents}c"] = selected.capital + cents / 100
            selected[f"profit_plus_{cents}c"] = selected.profit - cents / 100
    return audit, selected


def summarize(selected, allowances=(0, .05, .10)):
    """One-contract quote-based P&L, including each checkpoint and combined."""
    results = []
    for checkpoint in ["q1", "halftime", "combined"]:
        rows = selected if checkpoint == "combined" else selected[selected.checkpoint == checkpoint]
        if len(rows) and rows.settlement_value.isna().any():
            raise MarketDataUnavailable("Selected markets are unsettled: cannot report complete P&L")
        for allowance in allowances:
            capital = float((rows.entry_price + rows.fee + allowance).sum()) if len(rows) else 0
            payout = float(rows.settlement_value.sum()) if len(rows) else 0
            results.append(dict(checkpoint=checkpoint, allowance=allowance, entries=len(rows),
                                capital=capital, profit=payout-capital,
                                return_on_deployed=(payout-capital)/capital if capital else None))
    return pd.DataFrame(results)


def run_backtest(predictions, cache_dir, output_dir, **selection_options):
    """Download public data, write candidate reasons/entries/results, and return all."""
    candidates = build_candidates(predictions, PublicKalshi(cache_dir))
    audit, selected = select_strategy(candidates, **selection_options)
    report = summarize(selected)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "candidate_audit.csv", index=False)
    selected.to_csv(output_dir / "selected_entries.csv", index=False)
    report.to_csv(output_dir / "quote_based_results.csv", index=False)
    return audit, selected, report


def run_backtest_if_available(predictions, cache_dir, output_dir, **options):
    """Keep forecast outputs on API failure. Programming/cache errors still raise."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_backtest(predictions, cache_dir, output_dir, **options)
        audit, selected, report = result
        if not audit.reason.isin(['selected', 'below_edge', 'lower_priority', 'earlier_entry',
                                  'probability_not_confirmed', 'missing_q1_confirmation']).any():
            raise MarketDataUnavailable('No usable matching quotes for this prediction sample')
    except (requests.RequestException, MarketDataUnavailable) as error:
        status = {'available': False, 'reason': str(error), 'roi': None,
                  'note': 'Forecast outputs remain valid; no betting result is asserted.'}
        (output_dir / 'kalshi_status.json').write_text(json.dumps(status, indent=2))
        pd.DataFrame([status]).to_csv(output_dir / 'quote_based_results.csv', index=False)
        print('Kalshi betting results unavailable:', error)
        return None
    (output_dir / 'kalshi_status.json').write_text(json.dumps({'available': True}))
    return result
