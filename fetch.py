# nba_app/fetch.py
import time, random
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import playercareerstats, leaguedashteamstats
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
    needed = ["TEAM_ID","TEAM_ABBREVIATION","MIN","FGA","FTA","TOV","FGM","REB","OREB","DREB","OPP_REB","OPP_OREB","OPP_DREB"]
    for c in needed:
        if c not in df.columns: df[c] = 0
    df["TRB"] = df["REB"]; df["SEASON_ID"] = season
    return df[["SEASON_ID"] + needed + ["TRB"]].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_many(seasons: list[str]) -> pd.DataFrame:
    seasons = sorted(set([s for s in seasons if isinstance(s, str) and s]))
    frames = [get_team_totals_for_season(s) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
