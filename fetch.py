# nba_app/fetch.py
import time, random
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import playercareerstats, leaguedashteamstats, playergamelog
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
    Fetches a player's game logs for multiple seasons (joins them).
    seasons: list like ["2018-19","2019-20", ...]
    season_type: "Regular Season" or "Playoffs"
    """
    seasons = [s for s in seasons if isinstance(s, str) and s]
    frames = []
    for s in seasons:
        def _fetch():
            return playergamelog.PlayerGameLog(
                player_id=player_id,
                season=s,
                season_type_all_star=season_type,
                timeout=60
            ).get_data_frames()[0]
        try:
            df = _with_retry(_fetch, tries=4, base=1.0)
            if not df.empty:
                # 1) Normalize columns to ALL CAPS to avoid merge key errors (Game_ID vs GAME_ID, etc.)
                df.columns = [str(c).upper() for c in df.columns]

                # 2) Make sure we have SEASON_ID and SEASON_START
                df["SEASON_ID"] = s
                # Add SEASON_START as int (e.g., 2019 from "2019-20")
                try:
                    df["SEASON_START"] = int(str(s)[:4])
                except Exception:
                    pass

                # 3) Unify common stat column names that sometimes vary
                rename_map = {
                    "PLUSMINUS": "PLUS_MINUS",  # nba_api sometimes uses PLUSMINUS
                    "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",  # keep as is (explicitness)
                }
                df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

                # 4) Coerce numeric stat columns
                for c in ["PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS","MIN"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                frames.append(df)

        except Exception:
            # skip a broken season
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["SEASON_START"] = out["SEASON_ID"].astype(str).str[:4].astype(int)
        # standardize a few expected columns
        for c in ["PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS","MIN"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_head_to_head_games(p1_id: int, p2_id: int, seasons: list[str], season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Returns game-level head-to-head DataFrame for the two players by intersecting GAME_ID.
    Columns are suffixed with _P1 / _P2 for symmetry.
    """
    if not seasons:
        return pd.DataFrame()

    g1 = get_player_game_logs_many(p1_id, seasons, season_type=season_type)
    g2 = get_player_game_logs_many(p2_id, seasons, season_type=season_type)
    if g1.empty or g2.empty:
        return pd.DataFrame()

    keep = ["GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ABBREVIATION","SEASON_ID","SEASON_START",
        "MIN","PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS"]

    # Select only columns that exist (after we uppercased names in get_player_game_logs_many)
    g1 = g1[[c for c in keep if c in g1.columns]].copy()
    g2 = g2[[c for c in keep if c in g2.columns]].copy()

    # If either side doesn't have GAME_ID after normalization, there's nothing to merge on
    if "GAME_ID" not in g1.columns or "GAME_ID" not in g2.columns:
        return pd.DataFrame()

    merged = g1.merge(g2, on="GAME_ID", suffixes=("_P1", "_P2"))

    # keep only games where they were on opposing teams (avoid teammate overlaps)
    if "TEAM_ABBREVIATION_P1" in merged.columns and "TEAM_ABBREVIATION_P2" in merged.columns:
        merged = merged[merged["TEAM_ABBREVIATION_P1"] != merged["TEAM_ABBREVIATION_P2"]].copy()

    # prefer P1's dates and season fields
    if "GAME_DATE_P1" in merged.columns:
        merged.rename(columns={"GAME_DATE_P1":"GAME_DATE"}, inplace=True)
    if "SEASON_ID_P1" in merged.columns:
        merged.rename(columns={"SEASON_ID_P1":"SEASON_ID"}, inplace=True)
    if "SEASON_START_P1" in merged.columns:
        merged.rename(columns={"SEASON_START_P1":"SEASON_START"}, inplace=True)

    # convenience flags
    if "WL_P1" in merged.columns:
        merged["P1_WIN"] = merged["WL_P1"].astype(str).str.upper().str.startswith("W")

    return merged.sort_values("GAME_DATE")

