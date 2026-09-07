from __future__ import annotations

from .availability_features import availability_feature_columns
from .context_features import CONTEXT_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from .data import DataConfig
from .defense_features import defense_feature_columns
from .derived_features import MATCHUP_FEATURE_COLUMNS, TREND_FEATURE_COLUMNS
from .features import qb_feature_columns
from .offense_features import offense_feature_columns
from .season_context_features import SEASON_CONTEXT_FEATURE_COLUMNS

BASE_FEATURE_SET = "qb_offense_v0"

FEATURE_SET_GROUPS: dict[str, tuple[str, ...]] = {
    BASE_FEATURE_SET: (),
    "qb_offense_defense": ("defense",),
    "qb_offense_context": ("context",),
    "qb_offense_availability": ("availability",),
    "qb_offense_market": ("market",),
    "qb_offense_defense_context": ("defense", "context"),
    "qb_offense_defense_context_weather": ("defense", "context", "weather_forecast"),
    "qb_offense_defense_context_weather_market": (
        "defense",
        "context",
        "weather_forecast",
        "market",
    ),
    "qb_offense_defense_context_season": ("defense", "context", "season_context"),
    "qb_offense_defense_context_season_weather": (
        "defense",
        "context",
        "season_context",
        "weather_forecast",
    ),
    "qb_offense_defense_context_trend": ("defense", "context", "trend"),
    "qb_offense_defense_context_matchup": ("defense", "context", "matchup"),
    "qb_offense_defense_context_derived": ("defense", "context", "trend", "matchup"),
    "qb_offense_defense_context_market": ("defense", "context", "market"),
    "qb_offense_full_sports": ("defense", "context", "availability"),
    "qb_offense_full_market": ("defense", "context", "availability", "market"),
}


def available_feature_sets() -> tuple[str, ...]:
    return tuple(FEATURE_SET_GROUPS)


def base_feature_columns(config: DataConfig) -> list[str]:
    columns = qb_feature_columns(config.qb_rolling_windows) + offense_feature_columns(config.offense_rolling_windows)
    if len(columns) != 47 or len(columns) != len(set(columns)):
        raise ValueError(f"qb_offense_v0 must contain 47 unique features, got {len(columns)}")
    return columns


def feature_group_columns(config: DataConfig, group: str) -> list[str]:
    if group == "defense":
        return defense_feature_columns(config.defense_rolling_windows)
    if group == "context":
        return list(CONTEXT_FEATURE_COLUMNS)
    if group == "availability":
        return availability_feature_columns()
    if group == "market":
        return list(MARKET_FEATURE_COLUMNS)
    if group == "trend":
        return list(TREND_FEATURE_COLUMNS)
    if group == "matchup":
        return list(MATCHUP_FEATURE_COLUMNS)
    if group == "season_context":
        return list(SEASON_CONTEXT_FEATURE_COLUMNS)
    if group == "weather_forecast":
        raise ValueError("Weather is not part of this reproducible blog model.")
    raise ValueError(f"Unknown feature group: {group}")


def feature_columns_for_set(config: DataConfig, feature_set: str) -> list[str]:
    try:
        groups = FEATURE_SET_GROUPS[feature_set]
    except KeyError as error:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose one of {list(available_feature_sets())}"
        ) from error

    columns = base_feature_columns(config)
    for group in groups:
        columns.extend(feature_group_columns(config, group))
    if len(columns) != len(set(columns)):
        raise ValueError(f"Feature set {feature_set!r} contains duplicate columns")
    return columns
