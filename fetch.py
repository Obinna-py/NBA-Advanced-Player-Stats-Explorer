# nba_app/fetch.py
import time, random, hashlib, json
from pathlib import Path
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
import requests
from nba_api.stats.endpoints import playercareerstats, leaguedashteamstats, playergamelog, playergamelogs, commonplayerinfo
from nba_api.stats.static import players as static_players
import streamlit as st
from config import BALLDONTLIE_API_KEY

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "nba_api"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_WATCHLIST_FILE = _CACHE_DIR / "watchlist.json"
_BALLDONTLIE_BASE = "https://api.balldontlie.io"
_NBA_TIMEOUT_S = 5
_BALLDONTLIE_TIMEOUT_S = 5
_CACHE_SCHEMA_VERSION = "v3"
_PLAYER_ALIASES = {
    "ace bailey": ["Airious Bailey"],
    "airious bailey": ["Ace Bailey"],
    "wemby": ["Victor Wembanyama"],
}


def _disk_cache_paths(namespace: str, key: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    base = _CACHE_DIR / namespace / digest
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".pkl"), base.with_suffix(".json")


def _read_disk_cache(namespace: str, key: str) -> tuple[pd.DataFrame | None, dict | None]:
    data_path, meta_path = _disk_cache_paths(namespace, key)
    if not data_path.exists():
        return None, None
    try:
        df = pd.read_pickle(data_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return df, meta
    except Exception:
        return None, None


def _write_disk_cache(namespace: str, key: str, df: pd.DataFrame, meta: dict | None = None) -> None:
    data_path, meta_path = _disk_cache_paths(namespace, key)
    df.to_pickle(data_path)
    payload = {"cached_at": int(time.time())}
    if meta:
        payload.update(meta)
    meta_path.write_text(json.dumps(payload))


def _with_persistent_cache(namespace: str, key: str, fetch_fn, *, tries=1, base=0.35) -> pd.DataFrame:
    cached_df, cached_meta = _read_disk_cache(namespace, key)
    try:
        df = _with_retry(fetch_fn, tries=tries, base=base)
        if isinstance(df, pd.DataFrame):
            df = df.copy()
            df.attrs["cache_source"] = "live"
            df.attrs["cached_at"] = int(time.time())
            _write_disk_cache(namespace, key, df, {"source": "live"})
        return df
    except Exception:
        if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
            cached_df = cached_df.copy()
            cached_df.attrs["cache_source"] = "stale_disk_cache"
            if cached_meta and "cached_at" in cached_meta:
                cached_df.attrs["cached_at"] = cached_meta["cached_at"]
            return cached_df
        raise

def _with_retry(fetch_fn, *, tries=4, base=1.0):
    for i in range(tries):
        try:
            return fetch_fn()
        except (ReadTimeout, ConnectionError):
            if i == tries - 1: raise
            time.sleep(base * (2 ** i) + random.random() * 0.4)


def _read_stale_cache_only(namespace: str, key: str) -> pd.DataFrame | None:
    cached_df, cached_meta = _read_disk_cache(namespace, key)
    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
        cached_df = cached_df.copy()
        cached_df.attrs["cache_source"] = "stale_disk_cache"
        if cached_meta:
            if "cached_at" in cached_meta:
                cached_df.attrs["cached_at"] = cached_meta["cached_at"]
            if "provider" in cached_meta:
                cached_df.attrs["provider"] = cached_meta["provider"]
        return cached_df
    return None


def _fetch_provider_then_cache(namespace: str, key: str, providers: list[tuple[str, callable]]) -> pd.DataFrame:
    last_error = None
    for provider_name, provider_fn in providers:
        if provider_fn is None:
            continue
        try:
            df = provider_fn()
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.attrs["cache_source"] = "live"
                df.attrs["provider"] = provider_name
                df.attrs["cached_at"] = int(time.time())
                _write_disk_cache(namespace, key, df, {"source": "live", "provider": provider_name})
                return df
        except Exception as e:
            last_error = e

    cached_df = _read_stale_cache_only(namespace, key)
    if cached_df is not None:
        return cached_df

    if last_error:
        raise last_error
    return pd.DataFrame()


def _empty_result_with_error(error: Exception | None = None) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["cache_source"] = "empty"
    if error is not None:
        df.attrs["error_type"] = type(error).__name__
        df.attrs["error_message"] = str(error)
    return df


def _season_id_from_year(year: int) -> str:
    return f"{year}-{str((year + 1) % 100).zfill(2)}"


def _current_season_start_year() -> int:
    now = pd.Timestamp.now(tz="America/New_York")
    return int(now.year if now.month >= 10 else now.year - 1)


def _balldontlie_get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY is not configured.")
    resp = requests.get(
        f"{_BALLDONTLIE_BASE}{path}",
        headers={"Authorization": BALLDONTLIE_API_KEY},
        params=params or {},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _candidate_name_variants(full_name: str) -> list[str]:
    cleaned = " ".join((full_name or "").strip().split())
    if not cleaned:
        return []

    lowered = cleaned.lower()
    variants = [cleaned]
    variants.extend(_PLAYER_ALIASES.get(lowered, []))

    parts = cleaned.split()
    if len(parts) >= 2:
        variants.extend([parts[-1], parts[0]])

    seen = set()
    ordered = []
    for variant in variants:
        key = variant.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(variant.strip())
    return ordered


def _collect_balldontlie_candidates(full_name: str) -> list[dict]:
    variants = _candidate_name_variants(full_name)
    candidates = []
    seen_ids = set()

    def add_players(players: list[dict] | None) -> None:
        for p in players or []:
            pid = p.get("id")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            candidates.append(p)

    for variant in variants:
        for path in ("/v1/players", "/v1/players/active"):
            payload = _balldontlie_get(
                path,
                params={"search": variant, "per_page": 25},
                timeout=_BALLDONTLIE_TIMEOUT_S,
            )
            add_players(payload.get("data", []) or [])

    for variant in variants:
        parts = [p for p in variant.split() if p]
        if len(parts) < 2:
            continue
        exact_first = parts[0]
        exact_last = parts[-1]
        for path in ("/v1/players", "/v1/players/active"):
            payload = _balldontlie_get(
                path,
                params={"first_name": exact_first, "last_name": exact_last, "per_page": 25},
                timeout=_BALLDONTLIE_TIMEOUT_S,
            )
            add_players(payload.get("data", []) or [])

    return candidates


def _rank_balldontlie_candidates(full_name: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    cleaned = " ".join((full_name or "").strip().split())
    normalized = cleaned.lower()
    alias_norms = {name.lower() for name in _PLAYER_ALIASES.get(normalized, [])}
    parts = cleaned.lower().split()
    first = parts[0] if parts else ""
    last = parts[-1] if parts else ""

    def score(player: dict) -> tuple[int, str, int]:
        player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        player_norm = player_name.lower()
        player_first = str(player.get("first_name", "")).lower()
        player_last = str(player.get("last_name", "")).lower()
        team_name = str((player.get("team") or {}).get("full_name") or "")

        if player_norm == normalized:
            return (0, player_name, -int(player.get("draft_year") or 0))
        if player_norm in alias_norms:
            return (1, player_name, -int(player.get("draft_year") or 0))
        if last and player_last == last and first and player_first.startswith(first[:3]):
            return (2, player_name, -int(player.get("draft_year") or 0))
        if last and player_last == last:
            return (3, player_name, -int(player.get("draft_year") or 0))
        if first and player_first == first:
            return (4, player_name, -int(player.get("draft_year") or 0))
        if normalized and normalized in player_norm:
            return (5, player_name, -int(player.get("draft_year") or 0))
        return (6, f"{player_name}|{team_name}", -int(player.get("draft_year") or 0))

    return sorted(candidates, key=score)


def _find_balldontlie_player_by_name(full_name: str) -> dict | None:
    if not full_name:
        return None
    candidates = _rank_balldontlie_candidates(full_name, _collect_balldontlie_candidates(full_name))
    return candidates[0] if candidates else None


@st.cache_data(ttl=600, show_spinner=False)
def search_balldontlie_players(full_name: str) -> list[dict]:
    if not full_name:
        return []
    data = _rank_balldontlie_candidates(full_name, _collect_balldontlie_candidates(full_name))
    results = []
    for p in data:
        team = p.get("team", {}) or {}
        results.append({
            "id": p.get("id"),
            "full_name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "first_name": p.get("first_name"),
            "last_name": p.get("last_name"),
            "position": p.get("position"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "college": p.get("college"),
            "country": p.get("country"),
            "jersey_number": p.get("jersey_number"),
            "team_id": team.get("id"),
            "team_name": team.get("full_name") or team.get("name"),
            "team_abbreviation": team.get("abbreviation"),
        })
    exact_names = {full_name.strip().lower(), *[name.lower() for name in _PLAYER_ALIASES.get(full_name.strip().lower(), [])]}
    exact = [p for p in results if p["full_name"].lower() in exact_names]
    return exact if exact else results


@st.cache_data(ttl=600, show_spinner=False)
def get_balldontlie_player(player_id: int) -> dict | None:
    payload = _balldontlie_get(f"/v1/players/{player_id}", timeout=_BALLDONTLIE_TIMEOUT_S)
    player = (payload or {}).get("data") or {}
    if not player:
        return None
    team = player.get("team", {}) or {}
    return {
        "id": player.get("id"),
        "full_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
        "first_name": player.get("first_name"),
        "last_name": player.get("last_name"),
        "position": player.get("position"),
        "height": player.get("height"),
        "weight": player.get("weight"),
        "college": player.get("college"),
        "country": player.get("country"),
        "jersey_number": player.get("jersey_number"),
        "team_id": team.get("id"),
        "team_name": team.get("full_name") or team.get("name"),
        "team_abbreviation": team.get("abbreviation"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_balldontlie_team_games(team_id: int, per_page: int = 10) -> pd.DataFrame:
    payload = _balldontlie_get(
        "/v1/games",
        params={"team_ids[]": team_id, "per_page": per_page},
        timeout=_BALLDONTLIE_TIMEOUT_S,
    )
    games = payload.get("data", []) or []
    rows = []
    for g in games:
        home = g.get("home_team", {}) or {}
        away = g.get("visitor_team", {}) or {}
        rows.append({
            "Date": g.get("date", "")[:10],
            "Season": g.get("season"),
            "Status": g.get("status"),
            "Home": home.get("abbreviation"),
            "Home Score": g.get("home_team_score"),
            "Away": away.get("abbreviation"),
            "Away Score": g.get("visitor_team_score"),
        })
    return pd.DataFrame(rows)


def _get_balldontlie_player_info(full_name: str) -> pd.DataFrame:
    player = _find_balldontlie_player_by_name(full_name)
    if not player:
        return pd.DataFrame()

    team = player.get("team", {}) or {}
    row = {
        "DISPLAY_FIRST_LAST": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
        "TEAM_ID": team.get("id"),
        "TEAM_NAME": team.get("full_name") or team.get("name"),
        "TEAM_ABBREVIATION": team.get("abbreviation"),
        "POSITION": player.get("position"),
        "HEIGHT": player.get("height"),
        "WEIGHT": player.get("weight"),
        "SCHOOL": player.get("college"),
        "COUNTRY": player.get("country"),
        "JERSEY": player.get("jersey_number"),
        "DRAFT_YEAR": player.get("draft_year"),
        "BIRTHDATE": None,
    }
    df = pd.DataFrame([row])
    df.attrs["provider"] = "balldontlie"
    return df


def _resolve_player_name_and_nba_id(player_id: int, player_name: str | None = None, player_source: str | None = None) -> tuple[str | None, int | None]:
    if player_source == "balldontlie" and player_name:
        nba_matches = static_players.find_players_by_full_name(player_name)
        exact = [p for p in nba_matches if p.get("full_name", "").lower() == player_name.lower()]
        nba_player = (exact or nba_matches or [None])[0]
        return player_name, (nba_player.get("id") if nba_player else None)

    player = static_players.find_player_by_id(player_id)
    if player:
        return player.get("full_name"), player_id

    try:
        bd_player = get_balldontlie_player(int(player_id))
    except Exception:
        bd_player = None

    if not bd_player:
        return None, None

    full_name = bd_player.get("full_name")
    if not full_name:
        return None, None

    nba_matches = static_players.find_players_by_full_name(full_name)
    exact = [p for p in nba_matches if p.get("full_name", "").lower() == full_name.lower()]
    nba_player = (exact or nba_matches or [None])[0]
    return full_name, (nba_player.get("id") if nba_player else None)


def get_nba_headshot_url(player_id: int, player_name: str | None = None, player_source: str | None = None) -> str | None:
    _, nba_player_id = _resolve_player_name_and_nba_id(player_id, player_name=player_name, player_source=player_source)
    if not nba_player_id:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{nba_player_id}.png"


def _resolve_balldontlie_player_id(player_id: int, player_name: str | None = None, player_source: str | None = None) -> int | None:
    if player_source == "balldontlie":
        try:
            return int(player_id)
        except Exception:
            return None

    full_name, _ = _resolve_player_name_and_nba_id(player_id, player_name=player_name, player_source=player_source)
    if not full_name:
        return None

    try:
        bd_player = _find_balldontlie_player_by_name(full_name)
    except Exception:
        bd_player = None

    if not bd_player:
        return None
    try:
        return int(bd_player.get("id"))
    except Exception:
        return None


def _parse_balldontlie_minutes(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        try:
            mins, secs = text.split(":", 1)
            return float(mins) + (float(secs) / 60.0)
        except Exception:
            return None
    try:
        return float(text)
    except Exception:
        return None


def _balldontlie_stat_row_to_log_row(row: dict) -> dict:
    game = row.get("game", {}) or {}
    player = row.get("player", {}) or {}
    team = row.get("team", {}) or {}

    game_date = str(game.get("date") or "")[:10]
    season_start = game.get("season")
    season_id = _season_id_from_year(int(season_start)) if season_start is not None else None

    team_id = team.get("id") or player.get("team_id")
    team_abbrev = team.get("abbreviation")
    home_team_id = game.get("home_team_id")
    visitor_team_id = game.get("visitor_team_id")
    home_score = game.get("home_team_score")
    visitor_score = game.get("visitor_team_score")

    wl = None
    try:
        if team_id == home_team_id and home_score is not None and visitor_score is not None:
            wl = "W" if float(home_score) > float(visitor_score) else "L"
        elif team_id == visitor_team_id and home_score is not None and visitor_score is not None:
            wl = "W" if float(visitor_score) > float(home_score) else "L"
    except Exception:
        wl = None

    return {
        "GAME_ID": game.get("id"),
        "GAME_DATE": game_date,
        "SEASON_ID": season_id,
        "SEASON_START": season_start,
        "MATCHUP": None,
        "WL": wl,
        "TEAM_ABBREVIATION": team_abbrev,
        "PLAYER_ID": player.get("id"),
        "MIN": _parse_balldontlie_minutes(row.get("min")),
        "PTS": row.get("pts"),
        "REB": row.get("reb"),
        "AST": row.get("ast"),
        "STL": row.get("stl"),
        "BLK": row.get("blk"),
        "TOV": row.get("turnover"),
        "FGM": row.get("fgm"),
        "FGA": row.get("fga"),
        "FG3M": row.get("fg3m"),
        "FG3A": row.get("fg3a"),
        "FTM": row.get("ftm"),
        "FTA": row.get("fta"),
        "PLUS_MINUS": row.get("plus_minus"),
    }


def _get_balldontlie_head_to_head_games(
    p1_id: int,
    p2_id: int,
    seasons: list[str],
    *,
    p1_name: str | None = None,
    p2_name: str | None = None,
    p1_source: str | None = None,
    p2_source: str | None = None,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    bd_p1 = _resolve_balldontlie_player_id(p1_id, player_name=p1_name, player_source=p1_source)
    bd_p2 = _resolve_balldontlie_player_id(p2_id, player_name=p2_name, player_source=p2_source)
    if not bd_p1 or not bd_p2 or not seasons:
        return pd.DataFrame()

    season_years = []
    for s in seasons:
        try:
            season_years.append(int(str(s)[:4]))
        except Exception:
            continue
    if not season_years:
        return pd.DataFrame()

    params = {
        "player_ids[]": [bd_p1, bd_p2],
        "seasons[]": season_years,
        "per_page": 100,
        "postseason": str(season_type == "Playoffs").lower(),
    }

    all_rows = []
    cursor = None
    while True:
        req_params = dict(params)
        if cursor is not None:
            req_params["cursor"] = cursor
        payload = _balldontlie_get("/v1/stats", params=req_params, timeout=max(_BALLDONTLIE_TIMEOUT_S, 8))
        batch = payload.get("data", []) or []
        all_rows.extend(batch)
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not batch:
            break

    if not all_rows:
        return pd.DataFrame()

    logs = pd.DataFrame([_balldontlie_stat_row_to_log_row(row) for row in all_rows])
    if logs.empty:
        return logs

    p1_logs = logs[logs["PLAYER_ID"] == bd_p1].copy()
    p2_logs = logs[logs["PLAYER_ID"] == bd_p2].copy()
    if p1_logs.empty or p2_logs.empty:
        return pd.DataFrame()

    keep = ["GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ABBREVIATION","SEASON_ID","SEASON_START",
            "MIN","PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS"]

    merged = p1_logs[[c for c in keep if c in p1_logs.columns]].merge(
        p2_logs[[c for c in keep if c in p2_logs.columns]],
        on="GAME_ID",
        suffixes=("_P1", "_P2"),
    )
    if merged.empty:
        return merged

    if "TEAM_ABBREVIATION_P1" in merged.columns and "TEAM_ABBREVIATION_P2" in merged.columns:
        merged = merged[merged["TEAM_ABBREVIATION_P1"] != merged["TEAM_ABBREVIATION_P2"]].copy()

    if "GAME_DATE_P1" in merged.columns:
        merged.rename(columns={"GAME_DATE_P1":"GAME_DATE"}, inplace=True)
    if "SEASON_ID_P1" in merged.columns:
        merged.rename(columns={"SEASON_ID_P1":"SEASON_ID"}, inplace=True)
    if "SEASON_START_P1" in merged.columns:
        merged.rename(columns={"SEASON_START_P1":"SEASON_START"}, inplace=True)
    if "WL_P1" in merged.columns:
        merged["P1_WIN"] = merged["WL_P1"].astype(str).str.upper().str.startswith("W")

    min_p1 = pd.to_numeric(merged.get("MIN_P1"), errors="coerce") if "MIN_P1" in merged.columns else None
    min_p2 = pd.to_numeric(merged.get("MIN_P2"), errors="coerce") if "MIN_P2" in merged.columns else None
    if min_p1 is not None and min_p2 is not None:
        merged = merged[(min_p1 > 0) & (min_p2 > 0)].copy()

    if "GAME_DATE" in merged.columns:
        merged = merged.sort_values("GAME_DATE")
    merged.attrs["provider"] = "balldontlie"
    return merged


def _build_balldontlie_stats_row(player: dict, season: int, base_stats: dict, adv_stats: dict, per_mode: str) -> dict:
    team = player.get("team", {}) or {}
    gp = float(base_stats.get("gp") or 0)

    def totalize(value):
        try:
            return float(value) * gp
        except Exception:
            return value

    row = {
        "SEASON_ID": _season_id_from_year(season),
        "SEASON_START": season,
        "TEAM_ID": team.get("id"),
        "TEAM_ABBREVIATION": team.get("abbreviation"),
        "GP": int(gp) if gp else 0,
        "MIN": base_stats.get("min"),
        "PTS": base_stats.get("pts"),
        "REB": base_stats.get("reb"),
        "AST": base_stats.get("ast"),
        "STL": base_stats.get("stl"),
        "BLK": base_stats.get("blk"),
        "TOV": base_stats.get("tov"),
        "FGM": base_stats.get("fgm"),
        "FGA": base_stats.get("fga"),
        "FG3M": base_stats.get("fg3m"),
        "FG3A": base_stats.get("fg3a"),
        "FTM": base_stats.get("ftm"),
        "FTA": base_stats.get("fta"),
        "OREB": base_stats.get("oreb"),
        "DREB": base_stats.get("dreb"),
        "FG_PCT": base_stats.get("fg_pct"),
        "FG3_PCT": base_stats.get("fg3_pct"),
        "FT_PCT": base_stats.get("ft_pct"),
        "FG%": (base_stats.get("fg_pct") * 100) if base_stats.get("fg_pct") is not None else None,
        "3P%": (base_stats.get("fg3_pct") * 100) if base_stats.get("fg3_pct") is not None else None,
        "FT%": (base_stats.get("ft_pct") * 100) if base_stats.get("ft_pct") is not None else None,
        "PLUS_MINUS": base_stats.get("plus_minus"),
        "USG% (true)": (adv_stats.get("usg_pct") * 100) if adv_stats.get("usg_pct") is not None else None,
        "AST%": (adv_stats.get("ast_pct") * 100) if adv_stats.get("ast_pct") is not None else None,
        "TRB%": (adv_stats.get("reb_pct") * 100) if adv_stats.get("reb_pct") is not None else None,
        "ORB%": (adv_stats.get("oreb_pct") * 100) if adv_stats.get("oreb_pct") is not None else None,
        "DRB%": (adv_stats.get("dreb_pct") * 100) if adv_stats.get("dreb_pct") is not None else None,
        "AST/TO": adv_stats.get("ast_to"),
        "TS%": (adv_stats.get("ts_pct") * 100) if adv_stats.get("ts_pct") is not None else None,
        "EFG%": (adv_stats.get("efg_pct") * 100) if adv_stats.get("efg_pct") is not None else None,
    }

    if per_mode == "Totals":
        total_cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB"]
        for col in total_cols:
            row[col] = totalize(row[col])

    try:
        fga = float(row.get("FGA")) if row.get("FGA") is not None else None
    except Exception:
        fga = None
    try:
        fta = float(row.get("FTA")) if row.get("FTA") is not None else None
    except Exception:
        fta = None
    try:
        pts = float(row.get("PTS")) if row.get("PTS") is not None else None
    except Exception:
        pts = None
    try:
        fg3a = float(row.get("FG3A")) if row.get("FG3A") is not None else None
    except Exception:
        fg3a = None
    try:
        minutes = float(row.get("MIN")) if row.get("MIN") is not None else None
    except Exception:
        minutes = None

    row["PPS"] = (pts / fga) if fga and pts is not None else None
    row["3PAr"] = (fg3a / fga) if fga and fg3a is not None else None
    row["FTr"] = (fta / fga) if fga and fta is not None else None

    def per36(stat_key: str) -> float | None:
        try:
            stat_val = float(row.get(stat_key)) if row.get(stat_key) is not None else None
        except Exception:
            stat_val = None
        if stat_val is None or minutes is None or not minutes:
            return None
        return (stat_val / minutes) * 36.0

    for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "OREB", "DREB"]:
        row[f"{stat}/36"] = per36(stat)

    return row


def _get_balldontlie_latest_stats(full_name: str, per_mode: str, all_seasons: bool = False) -> pd.DataFrame:
    player = _find_balldontlie_player_by_name(full_name)
    if not player:
        return pd.DataFrame()

    player_id = player.get("id")
    current_season = _current_season_start_year()
    start_season = max(int(player.get("draft_year") or current_season), 1996)
    seasons = list(range(start_season, current_season + 1)) if all_seasons else [current_season]
    rows = []

    for season in seasons:
        common_params = {
            "season": season,
            "season_type": "regular",
            "player_ids[]": player_id,
            "per_page": 1,
        }

        base_payload = _balldontlie_get(
            "/v1/season_averages/general",
            params={**common_params, "type": "base"},
            timeout=_BALLDONTLIE_TIMEOUT_S,
        )
        advanced_payload = _balldontlie_get(
            "/v1/season_averages/general",
            params={**common_params, "type": "advanced"},
            timeout=_BALLDONTLIE_TIMEOUT_S,
        )

        base_rows = base_payload.get("data", []) or []
        adv_rows = advanced_payload.get("data", []) or []
        if not base_rows:
            continue

        base_stats = (base_rows[0].get("stats") or {}).copy()
        adv_stats = (adv_rows[0].get("stats") or {}).copy() if adv_rows else {}
        rows.append(_build_balldontlie_stats_row(player, season, base_stats, adv_stats, per_mode))

    return pd.DataFrame(rows)


def _fetch_balldontlie_season_average_type(season: int, stat_type: str) -> pd.DataFrame:
    rows = []
    cursor = None

    while True:
        params = {
            "season": season,
            "season_type": "regular",
            "type": stat_type,
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor

        payload = _balldontlie_get(
            "/v1/season_averages/general",
            params=params,
            timeout=max(_BALLDONTLIE_TIMEOUT_S, 8),
        )
        data = payload.get("data", []) or []
        for item in data:
            player = item.get("player", {}) or {}
            stats = item.get("stats", {}) or {}
            row = {
                "PLAYER_ID": player.get("id"),
                "PLAYER_NAME": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "POSITION": player.get("position"),
                "SEASON_START": item.get("season"),
            }
            row.update(stats)
            rows.append(row)

        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not data:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["SEASON_ID"] = df["SEASON_START"].apply(_season_id_from_year)
        df["SOURCE_TYPE"] = stat_type
    return df


@st.cache_data(ttl=21600, show_spinner=False)
def get_balldontlie_league_season_averages(season: int) -> pd.DataFrame:
    try:
        base_df = _fetch_balldontlie_season_average_type(season, "base")
        adv_df = _fetch_balldontlie_season_average_type(season, "advanced")
    except Exception as e:
        return _empty_result_with_error(e)

    if base_df.empty and adv_df.empty:
        return pd.DataFrame()
    if base_df.empty:
        return adv_df
    if adv_df.empty:
        return base_df

    overlap_drop = [c for c in adv_df.columns if c in base_df.columns and c not in {"PLAYER_ID", "SEASON_ID", "SEASON_START", "PLAYER_NAME", "POSITION"}]
    merged = base_df.merge(
        adv_df.drop(columns=overlap_drop, errors="ignore"),
        on=["PLAYER_ID", "PLAYER_NAME", "POSITION", "SEASON_START", "SEASON_ID"],
        how="outer",
    )
    merged.attrs["provider"] = "balldontlie"
    merged.attrs["season"] = season
    return merged


def search_players(full_name: str) -> list[dict]:
    if not full_name:
        return []

    try:
        fallback = search_balldontlie_players(full_name)
        if fallback:
            return [{**p, "source": "balldontlie"} for p in fallback]
    except Exception:
        pass

    found = static_players.find_players_by_full_name(full_name)
    exact = [p for p in found if p["full_name"].lower() == full_name.strip().lower()]
    found = exact if exact else found
    return found


def player_to_share_token(player: dict | None) -> str | None:
    if not player or "id" not in player:
        return None
    source = player.get("source", "nba_api")
    return f"{source}:{player['id']}"


def player_from_share_token(token: str) -> dict | None:
    if not token or ":" not in token:
        return None

    source, raw_id = token.split(":", 1)
    try:
        player_id = int(raw_id)
    except Exception:
        return None

    if source == "balldontlie":
        player = get_balldontlie_player(player_id)
        if not player:
            return None
        return {**player, "source": "balldontlie"}

    player = static_players.find_player_by_id(player_id)
    if not player:
        return None
    return {**player, "source": "nba_api"}


def load_watchlist_tokens() -> list[str]:
    if not _WATCHLIST_FILE.exists():
        return []
    try:
        payload = json.loads(_WATCHLIST_FILE.read_text())
    except Exception:
        return []
    tokens = payload.get("tokens", payload if isinstance(payload, list) else [])
    if not isinstance(tokens, list):
        return []
    out = []
    seen = set()
    for token in tokens:
        token = str(token).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def save_watchlist_tokens(tokens: list[str]) -> None:
    deduped = []
    seen = set()
    for token in tokens:
        token = str(token).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    _WATCHLIST_FILE.write_text(json.dumps({"tokens": deduped}, indent=2))


def get_watchlist_players() -> list[dict]:
    players = []
    for token in load_watchlist_tokens():
        try:
            player = player_from_share_token(token)
        except Exception:
            player = None
        if player:
            players.append(player)
    return players


def add_player_to_watchlist(player: dict | None) -> list[str]:
    token = player_to_share_token(player)
    if not token:
        return load_watchlist_tokens()
    tokens = load_watchlist_tokens()
    if token not in tokens:
        tokens.append(token)
        save_watchlist_tokens(tokens)
    return tokens


def remove_player_from_watchlist(player: dict | None = None, token: str | None = None) -> list[str]:
    target = token or player_to_share_token(player)
    tokens = [t for t in load_watchlist_tokens() if t != target]
    save_watchlist_tokens(tokens)
    return tokens


@st.cache_data(ttl=120, show_spinner=False)
def check_nba_api_health(test_player_id: int = 203999, timeout: int = 8) -> dict:
    started_at = time.time()
    try:
        df = commonplayerinfo.CommonPlayerInfo(
            player_id=test_player_id,
            timeout=timeout
        ).get_data_frames()[0]
        elapsed = round(time.time() - started_at, 2)
        return {
            "ok": True,
            "status": "healthy",
            "elapsed_s": elapsed,
            "rows": 0 if df is None else len(df),
            "message": f"NBA API responded in {elapsed}s.",
        }
    except Exception as e:
        elapsed = round(time.time() - started_at, 2)
        return {
            "ok": False,
            "status": "unhealthy",
            "elapsed_s": elapsed,
            "error_type": type(e).__name__,
            "message": str(e),
        }


@st.cache_data(ttl=120, show_spinner=False)
def check_balldontlie_api_health(test_query: str = "Jokic", timeout: int = 5) -> dict:
    started_at = time.time()
    try:
        payload = _balldontlie_get(
            "/v1/players",
            params={"search": test_query, "per_page": 5},
            timeout=timeout,
        )
        data = payload.get("data", []) or []
        elapsed = round(time.time() - started_at, 2)
        return {
            "ok": True,
            "status": "healthy",
            "elapsed_s": elapsed,
            "rows": len(data),
            "message": f"balldontlie responded in {elapsed}s.",
        }
    except Exception as e:
        elapsed = round(time.time() - started_at, 2)
        return {
            "ok": False,
            "status": "unhealthy",
            "elapsed_s": elapsed,
            "error_type": type(e).__name__,
            "message": str(e),
        }

@st.cache_data(ttl=3600, show_spinner=False)
def get_player_career(player_id: int, per_mode: str = 'Totals', player_name: str | None = None, player_source: str | None = None, all_seasons: bool = False) -> pd.DataFrame:
    full_name, nba_player_id = _resolve_player_name_and_nba_id(player_id, player_name=player_name, player_source=player_source)

    def _fetch_nba():
        if not nba_player_id:
            return pd.DataFrame()
        return playercareerstats.PlayerCareerStats(
            player_id=nba_player_id, per_mode36=per_mode, timeout=_NBA_TIMEOUT_S
        ).get_data_frames()[0]

    def _fetch_balldontlie():
        if not full_name or not BALLDONTLIE_API_KEY:
            return pd.DataFrame()
        return _get_balldontlie_latest_stats(full_name, per_mode, all_seasons=all_seasons)

    try:
        return _fetch_provider_then_cache(
            "player_career",
            f"{_CACHE_SCHEMA_VERSION}:{player_id}:{per_mode}:{player_name or ''}:{player_source or ''}:{int(all_seasons)}",
            [
                ("balldontlie", _fetch_balldontlie),
                ("nba_api", _fetch_nba),
            ],
        )
    except Exception as e:
        cached_df = _read_stale_cache_only("player_career", f"{_CACHE_SCHEMA_VERSION}:{player_id}:{per_mode}:{player_name or ''}:{player_source or ''}:{int(all_seasons)}")
        return cached_df if cached_df is not None else _empty_result_with_error(e)

@st.cache_data(ttl=3600, show_spinner=False)
def get_player_info(player_id: int, player_name: str | None = None, player_source: str | None = None) -> pd.DataFrame:
    full_name, nba_player_id = _resolve_player_name_and_nba_id(player_id, player_name=player_name, player_source=player_source)

    def _fetch_nba():
        if not nba_player_id:
            return pd.DataFrame()
        return commonplayerinfo.CommonPlayerInfo(
            player_id=nba_player_id,
            timeout=_NBA_TIMEOUT_S
        ).get_data_frames()[0]

    def _fetch_balldontlie():
        if not full_name or not BALLDONTLIE_API_KEY:
            return pd.DataFrame()
        return _get_balldontlie_player_info(full_name)

    try:
        return _fetch_provider_then_cache(
            "player_info",
            f"{player_id}:{player_name or ''}:{player_source or ''}",
            [
                ("balldontlie", _fetch_balldontlie),
                ("nba_api", _fetch_nba),
            ],
        )
    except Exception as e:
        cached_df = _read_stale_cache_only("player_info", f"{player_id}:{player_name or ''}:{player_source or ''}")
        return cached_df if cached_df is not None else _empty_result_with_error(e)

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_for_season(season: str) -> pd.DataFrame:
    def _fetch():
        return leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="Totals", measure_type_detailed_defense="Base", timeout=_NBA_TIMEOUT_S
        ).get_data_frames()[0]
    df = _fetch_provider_then_cache(
        "team_totals",
        season,
        [("nba_api", _fetch)],
    )
    needed = ["TEAM_ID","TEAM_ABBREVIATION","MIN","FGA","FTA","TOV","FGM","REB","OREB","DREB","FG3M","FG3A","FTM","PTS","OPP_REB","OPP_OREB","OPP_DREB"]
    for c in needed:
        if c not in df.columns: df[c] = 0
    df["TRB"] = df["REB"]; df["SEASON_ID"] = season
    return df[["SEASON_ID"] + needed + ["TRB"]].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_many(seasons: list[str]) -> pd.DataFrame:
    seasons = sorted(set([s for s in seasons if isinstance(s, str) and s]))
    frames = [get_team_totals_for_season(s) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_player_game_logs_many(player_id: int, seasons: list[str], season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Robustly fetch a player's game logs across multiple seasons.
    Tries the faster playergamelogs endpoint per season; falls back to playergamelog.
    Normalizes columns to UPPERCASE and coerces numeric types.
    """
    import time
    import pandas as pd
    seasons = [s for s in seasons if isinstance(s, str) and s]
    frames = []

    for s in seasons:
        # 1) Try the bulk-ish endpoint first (fewer API quirks)
        def _fetch_bulk():
            return playergamelogs.PlayerGameLogs(
                season_nullable=s,
                season_type_nullable=season_type,
                player_id_nullable=str(player_id),
                timeout=60
            ).get_data_frames()[0]

        # 2) Fallback single-season endpoint
        def _fetch_simple():
            return playergamelog.PlayerGameLog(
                player_id=player_id,
                season=s,
                season_type_all_star=season_type,
                timeout=60
            ).get_data_frames()[0]

        df = pd.DataFrame()
        # Try bulk with a couple retries, then fallback
        for attempt in range(3):
            try:
                df = _fetch_bulk()
                if not df.empty:
                    break
            except Exception:
                time.sleep(0.7)  # gentle backoff
        if df is None or df.empty:
            for attempt in range(3):
                try:
                    df = _fetch_simple()
                    if not df.empty:
                        break
                except Exception:
                    time.sleep(0.7)

        if df is None or df.empty:
            # Give the API a breather; move on
            time.sleep(0.5)
            continue

        # Normalize column names
        df.columns = [str(c).upper() for c in df.columns]
        df["SEASON_ID"] = s
        try:
            df["SEASON_START"] = int(str(s)[:4])
        except Exception:
            pass

        # Unify and coerce numeric columns
        if "PLUSMINUS" in df.columns and "PLUS_MINUS" not in df.columns:
            df.rename(columns={"PLUSMINUS": "PLUS_MINUS"}, inplace=True)

        for c in ["PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS","MIN"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)
        # light delay to avoid rate limits
        time.sleep(0.35)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_head_to_head_games(
    p1_id: int,
    p2_id: int,
    seasons: list[str],
    season_type: str = "Regular Season",
    force: int = 0,
    p1_name: str | None = None,
    p2_name: str | None = None,
    p1_source: str | None = None,
    p2_source: str | None = None,
) -> pd.DataFrame:
    """
    Returns game-level head-to-head DataFrame for the two players by intersecting GAME_ID.
    'force' is a no-op used only to bust the cache when True.
    """
    import pandas as pd
    if not seasons:
        return pd.DataFrame()

    balldontlie_attempted = False
    try:
        balldontlie_attempted = True
        h2h_balldontlie = _get_balldontlie_head_to_head_games(
            p1_id, p2_id, seasons,
            p1_name=p1_name, p2_name=p2_name,
            p1_source=p1_source, p2_source=p2_source,
            season_type=season_type,
        )
        if h2h_balldontlie is not None:
            return h2h_balldontlie
    except Exception:
        pass

    if balldontlie_attempted:
        empty = pd.DataFrame()
        empty.attrs["provider"] = "balldontlie"
        return empty

    g1 = get_player_game_logs_many(p1_id, seasons, season_type=season_type)
    g2 = get_player_game_logs_many(p2_id, seasons, season_type=season_type)
    if g1.empty or g2.empty:
        return pd.DataFrame()

    keep = ["GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ABBREVIATION","SEASON_ID","SEASON_START",
            "MIN","PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS"]

    g1 = g1[[c for c in keep if c in g1.columns]].copy()
    g2 = g2[[c for c in keep if c in g2.columns]].copy()

    if "GAME_ID" not in g1.columns or "GAME_ID" not in g2.columns:
        return pd.DataFrame()

    merged = g1.merge(g2, on="GAME_ID", suffixes=("_P1", "_P2"))

    # Exclude teammate overlaps
    if "TEAM_ABBREVIATION_P1" in merged.columns and "TEAM_ABBREVIATION_P2" in merged.columns:
        merged = merged[merged["TEAM_ABBREVIATION_P1"] != merged["TEAM_ABBREVIATION_P2"]].copy()

    # Prefer P1's date/season fields
    if "GAME_DATE_P1" in merged.columns:
        merged.rename(columns={"GAME_DATE_P1":"GAME_DATE"}, inplace=True)
    if "SEASON_ID_P1" in merged.columns:
        merged.rename(columns={"SEASON_ID_P1":"SEASON_ID"}, inplace=True)
    if "SEASON_START_P1" in merged.columns:
        merged.rename(columns={"SEASON_START_P1":"SEASON_START"}, inplace=True)

    if "WL_P1" in merged.columns:
        merged["P1_WIN"] = merged["WL_P1"].astype(str).str.upper().str.startswith("W")

    out = merged.sort_values("GAME_DATE")
    out.attrs["provider"] = "nba_api"
    return out
