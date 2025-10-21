# nba_app/fetch.py
import time, random
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import playercareerstats, leaguedashteamstats, playergamelog, playergamelogs
import streamlit as st

def _with_retry(fetch_fn, *, tries=4, base=1.0):
    for i in range(tries):
        try:
            return fetch_fn()
        except (ReadTimeout, ConnectionError):
            if i == tries - 1: raise
            time.sleep(base * (2 ** i) + random.random() * 0.4)

@st.cache_data(ttl=3600, show_spinner=False)
def get_player_career(player_id: int, per_mode: str = 'Totals') -> pd.DataFrame:
    def _fetch():
        return playercareerstats.PlayerCareerStats(
            player_id=player_id, per_mode36=per_mode, timeout=60
        ).get_data_frames()[0]
    return _with_retry(_fetch, tries=4, base=1.0)

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_for_season(season: str) -> pd.DataFrame:
    def _fetch():
        return leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="Totals", measure_type_detailed_defense="Base", timeout=60
        ).get_data_frames()[0]
    df = _with_retry(_fetch, tries=4, base=1.0)
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
def get_head_to_head_games(p1_id: int, p2_id: int, seasons: list[str], season_type: str = "Regular Season", force: int = 0) -> pd.DataFrame:
    """
    Returns game-level head-to-head DataFrame for the two players by intersecting GAME_ID.
    'force' is a no-op used only to bust the cache when True.
    """
    import pandas as pd
    if not seasons:
        return pd.DataFrame()

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

    return merged.sort_values("GAME_DATE")
