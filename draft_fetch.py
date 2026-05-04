import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from fetch import get_placeholder_headshot_data_uri


NCAAB_BASE_URL = "https://api.balldontlie.io/ncaab/v1"
DEFAULT_DRAFT_SEASON = 2025
_TIMEOUT_S = 8
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_key(secret_names: list[str], env_names: list[str], local_fallback: str = "") -> str | None:
    try:
        secrets_obj = st.secrets
    except Exception:
        secrets_obj = None

    if secrets_obj is not None:
        for name in secret_names:
            try:
                value = secrets_obj.get(name)
            except Exception:
                value = None
            if value:
                return str(value)

    for name in env_names:
        value = os.getenv(name)
        if value:
            return value

    return local_fallback or None


BALLDONTLIE_NCAAB_API_KEY = _load_key(
    ["BALLDONTLIE_NCAAB_API_KEY"],
    ["BALLDONTLIE_NCAAB_API_KEY"],
)


def _to_float(value) -> float | None:
    try:
        if value in (None, "", "—"):
            return None
        return float(value)
    except Exception:
        return None


def _parse_minutes_value(value) -> float | None:
    if value in (None, "", "—"):
        return None
    if isinstance(value, str) and ":" in value:
        try:
            minutes_str, seconds_str = value.split(":", 1)
            minutes = float(minutes_str)
            seconds = float(seconds_str)
            return minutes + (seconds / 60.0)
        except Exception:
            return None
    return _to_float(value)


def _normalize_rate_stat(value, games: float | None) -> float | None:
    stat = _to_float(value)
    if stat is None:
        return None
    if games and games > 0:
        return stat / games
    return stat


def _normalize_minutes(value, games: float | None) -> float | None:
    minutes = _parse_minutes_value(value)
    if minutes is None:
        return None
    if isinstance(value, str) and ":" in value:
        return minutes
    # Numeric minute fields can be mixed across sources: some are already per-game,
    # others are season totals. Treat realistic per-game minute values as already normalized.
    if minutes <= 48:
        return minutes
    if games and games > 0:
        return minutes / games
    return minutes


def _normalize_pct(value) -> float | None:
    pct = _to_float(value)
    if pct is None:
        return None
    if 0 < pct <= 1:
        return pct * 100.0
    return pct


def _load_json(name: str, default: Any) -> Any:
    path = _DATA_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def ncaab_api_ready() -> bool:
    return bool(BALLDONTLIE_NCAAB_API_KEY)


def _ncaab_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not BALLDONTLIE_NCAAB_API_KEY:
        raise RuntimeError("BALLDONTLIE_NCAAB_API_KEY is not configured.")
    response = requests.get(
        f"{NCAAB_BASE_URL}{path}",
        headers={"Authorization": BALLDONTLIE_NCAAB_API_KEY},
        params=params or {},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"data": payload}


def search_ncaab_players(query: str, per_page: int = 8) -> list[dict]:
    if not ncaab_api_ready():
        return []
    query = str(query or "").strip()
    if len(query) < 2:
        return []
    try:
        payload = _ncaab_get("/players", {"search": query, "per_page": per_page})
    except Exception:
        return []
    rows = payload.get("data") or []
    return rows if isinstance(rows, list) else []


def get_ncaab_player_season_stats(player_id: int, season: int = DEFAULT_DRAFT_SEASON) -> dict | None:
    if not ncaab_api_ready():
        return None
    try:
        payload = _ncaab_get(
            "/player_season_stats",
            {"player_ids[]": player_id, "seasons[]": season, "per_page": 100},
        )
    except Exception:
        return None
    rows = payload.get("data") or []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def lookup_ncaab_player(full_name: str, school_or_team: str | None = None) -> dict | None:
    rows = search_ncaab_players(full_name, per_page=12)
    if not rows:
        return None
    school_target = str(school_or_team or "").strip().lower()
    if school_target:
        for row in rows:
            team = row.get("team") or {}
            school = str(team.get("college") or team.get("full_name") or "").strip().lower()
            if school and school == school_target:
                return row
    query_name = str(full_name or "").strip().lower()
    for row in rows:
        candidate = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip().lower()
        if candidate == query_name:
            return row
    return rows[0]


def get_espn_ncaab_headshot_url(espn_id: str | int) -> str:
    return f"https://a.espncdn.com/combiner/i?img=/i/headshots/mens-college-basketball/players/full/{espn_id}.png&w=350&h=254"


def load_espn_prospect_media(prospect: dict) -> dict | None:
    espn_id = prospect.get("espn_id")
    if not espn_id:
        return None
    return {"espn_id": espn_id, "headshot_url": get_espn_ncaab_headshot_url(espn_id)}


def load_live_prospect_profile(player: dict, season: int = DEFAULT_DRAFT_SEASON) -> dict:
    stats_row = get_ncaab_player_season_stats(int(player.get("id")), season=season) or {}
    team = player.get("team") or {}
    raw_games = stats_row.get("games_played") or stats_row.get("gp") or stats_row.get("games")
    games = _to_float(raw_games)
    minutes = _normalize_minutes(stats_row.get("min") or stats_row.get("minutes"), games)
    turnovers = stats_row.get("turnovers") or stats_row.get("turnover")
    ppg = _normalize_rate_stat(stats_row.get("pts") or stats_row.get("points"), games)
    rpg = _normalize_rate_stat(stats_row.get("reb") or stats_row.get("rebounds"), games)
    apg = _normalize_rate_stat(stats_row.get("ast") or stats_row.get("assists"), games)
    spg = _normalize_rate_stat(stats_row.get("stl") or stats_row.get("steals"), games)
    bpg = _normalize_rate_stat(stats_row.get("blk") or stats_row.get("blocks"), games)
    tpg = _normalize_rate_stat(turnovers, games)

    # Some balldontlie NCAAB season rows appear to return minutes as a share of game
    # instead of true MPG (for example 0.9 instead of ~36). When the production is clearly
    # rotation-level but minutes are implausibly tiny, scale the share to a 40-minute game.
    if minutes is not None and minutes <= 1.5:
        production_signals = [value for value in (ppg, rpg, apg, spg, bpg) if value is not None]
        if production_signals and max(production_signals) >= 2:
            minutes = minutes * 40.0

    profile = {
        "full_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
        "school_or_team": team.get("college") or team.get("full_name") or "—",
        "league": "NCAAB",
        "position": player.get("position") or "—",
        "height": player.get("height") or player.get("height_display") or "—",
        "weight": player.get("weight") or player.get("weight_display") or "—",
        "birthdate": player.get("date_of_birth") or "",
        "country": player.get("country") or "",
        "games": games,
        "minutes": minutes,
        "ppg": ppg,
        "rpg": rpg,
        "apg": apg,
        "spg": spg,
        "bpg": bpg,
        "tpg": tpg,
        "fg_pct": _normalize_pct(stats_row.get("fg_pct")),
        "three_pct": _normalize_pct(stats_row.get("fg3_pct") or stats_row.get("three_pt_pct")),
        "ft_pct": _normalize_pct(stats_row.get("ft_pct")),
        "ts_pct": _normalize_pct(stats_row.get("ts_pct")),
        "stats_source": "balldontlie NCAAB",
        "season": season,
        "measurement_stage": "Unverified",
        "measurement_source": "Live search profile",
        "measurement_confidence": "Low",
    }
    return {k: v for k, v in profile.items() if v not in (None, "", [])}


def load_local_prospect_metadata() -> list[dict]:
    rows = _load_json("prospect_metadata.json", [])
    return rows if isinstance(rows, list) else []


def load_prospect_consensus_anchors() -> list[dict]:
    rows = _load_json("prospect_consensus_anchors.json", [])
    return rows if isinstance(rows, list) else []


def load_espn_mock_2026() -> dict:
    payload = _load_json("espn_mock_2026.json", {})
    return payload if isinstance(payload, dict) else {}


def get_placeholder_draft_headshot() -> str:
    return get_placeholder_headshot_data_uri()
