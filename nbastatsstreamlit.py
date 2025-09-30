# -*- coding: utf-8 -*-
# NBA Advanced Player Stats Explorer — polished UI + session-state selections

from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo, leaguedashteamstats
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import google.generativeai as genai

# ============ CONFIG ============
st.set_page_config(page_title="NBA Advanced Player Stats Explorer", layout="wide")

# Safely load Gemini API key
GEMINI_API_KEY =  "AIzaSyC0APdZnTD2G3GtPUBmWwB-XhNauX4TqUo"
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
else:
    model = None  # AI features disabled if no key

@st.cache_data(ttl=3600)
def get_team_totals_for_season(season: str) -> pd.DataFrame:
    """
    Fetch team totals for a given season (team-level box totals).
    We normalize column names and guarantee presence of fields used by compute_full_advanced_stats.
    """
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    # columns that we actually use downstream; include safe fallbacks
    needed = [
        "TEAM_ID", "MIN", "FGA", "FTA", "TOV", "FGM",
        "REB", "OREB", "DREB",
        # Opponent rebound fields are not always present in this endpoint;
        # we include them if they exist; else we fill with 0 and the code will fall back.
        "OPP_REB", "OPP_OREB", "OPP_DREB",
    ]

    # Ensure columns exist; fill missing with 0
    for col in needed:
        if col not in df.columns:
            df[col] = 0

    # No need to rename REB -> TRB here; keep NBA column names consistent
    # If you want TRB, derive it explicitly later.
    out = df[needed].copy()

    # Also compute explicit TRB (team total rebounds) if you prefer that name downstream
    if "TRB" not in out.columns:
        out["TRB"] = out["REB"]  # NBA calls it REB; alias as TRB for convenience

    return out


# ============ HELPERS ============
def compute_full_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: TS%, eFG%, PPS, 3PAr, FTr, per-36 stats,
          team-based USG% (true), AST%,
          and rebound rates ORB%/DRB%/TRB% (team + opponent context where available).

    Works on the career dataframe from PlayerCareerStats (per_mode36='PerGame' or 'Totals').
    """
    out = df.copy()

    # ---------- Shooting & efficiency (no team context)
    if 'FG_PCT' in out.columns: out['FG%'] = out['FG_PCT'] * 100
    if 'FG3_PCT' in out.columns: out['3P%'] = out['FG3_PCT'] * 100
    if 'FT_PCT' in out.columns: out['FT%'] = out['FT_PCT'] * 100

    out['TS%'] = out.apply(
        lambda r: (r.get('PTS', 0) / (2 * (r.get('FGA', 0) + 0.44 * r.get('FTA', 0))) * 100)
        if (r.get('FGA', 0) + 0.44 * r.get('FTA', 0)) > 0 else np.nan, axis=1
    )
    out['EFG%'] = out.apply(
        lambda r: ((r.get('FGM', 0) + 0.5 * r.get('FG3M', 0)) / r.get('FGA', 0) * 100)
        if r.get('FGA', 0) > 0 else np.nan, axis=1
    )
    out['PPS']  = out.apply(lambda r: (r.get('PTS', 0) / r.get('FGA', 0)) if r.get('FGA', 0) > 0 else np.nan, axis=1)
    out['3PAr'] = out.apply(lambda r: (r.get('FG3A', 0) / r.get('FGA', 0)) if r.get('FGA', 0) > 0 else np.nan, axis=1)
    out['FTr']  = out.apply(lambda r: (r.get('FTA', 0) / r.get('FGA', 0)) if r.get('FGA', 0) > 0 else np.nan, axis=1)
    out['AST/TO'] = out.apply(lambda r: (r.get('AST', 0) / r.get('TOV', 0)) if r.get('TOV', 0) > 0 else np.nan, axis=1)

    # ---------- Per-36 rates
    def per36(r, col):
        mins = r.get('MIN', 0)
        return (r.get(col, 0) / mins) * 36 if mins and mins > 0 else np.nan

    for stat in ['PTS','REB','AST','STL','BLK','TOV','FGM','FGA','FG3M','OREB','DREB']:
        out[f'{stat}/36'] = out.apply(lambda r, c=stat: per36(r, c), axis=1)

    # ---------- Team-context rates (true USG%, AST%, ORB%, DRB%, TRB%)
    def _team_rates(row):
        season = row.get('SEASON_ID')
        team_id = row.get('TEAM_ID')
        mp     = row.get('MIN', 0)

        # Skip rows without a concrete team (e.g., "TOT") or no minutes
        if not season or not team_id or pd.isna(team_id) or team_id == 0 or not mp:
            return pd.Series({'USG% (true)': np.nan, 'AST%': np.nan, 'ORB%': np.nan, 'DRB%': np.nan, 'TRB%': np.nan})

        teams = get_team_totals_for_season(season)
        team  = teams[teams['TEAM_ID'] == team_id]
        if team.empty:
            return pd.Series({'USG% (true)': np.nan, 'AST%': np.nan, 'ORB%': np.nan, 'DRB%': np.nan, 'TRB%': np.nan})

        tMIN   = float(team.iloc[0]['MIN']   or 0.0)
        tFGA   = float(team.iloc[0]['FGA']   or 0.0)
        tFTA   = float(team.iloc[0]['FTA']   or 0.0)
        tTOV   = float(team.iloc[0]['TOV']   or 0.0)
        tFGM   = float(team.iloc[0]['FGM']   or 0.0)
        tTRB   = float(team.iloc[0]['TRB']   or team.iloc[0].get("REB", 0) or 0.0)
        tOREB  = float(team.iloc[0]['OREB']  or 0.0)
        tDREB  = float(team.iloc[0]['DREB']  or 0.0)
        oTRB   = float(team.iloc[0]['OPP_REB']  or 0.0)
        oOREB  = float(team.iloc[0]['OPP_OREB'] or 0.0)
        oDREB  = float(team.iloc[0]['OPP_DREB'] or 0.0)

        if tMIN <= 0:
            return pd.Series({'USG% (true)': np.nan, 'AST%': np.nan, 'ORB%': np.nan, 'DRB%': np.nan, 'TRB%': np.nan})

        # --- True Usage%
        denom_usg = (tFGA + 0.44 * tFTA + tTOV)
        if denom_usg > 0 and mp > 0:
            usg_true = 100.0 * ((row.get('FGA', 0) + 0.44 * row.get('FTA', 0) + row.get('TOV', 0)) * (tMIN / 5.0)) / (mp * denom_usg)
        else:
            usg_true = np.nan

        # --- AST% (classic):
        # 100 * AST / [ (MIN/TeamMIN) * TeamFGM - FGM ]
        denom_ast = ((mp / tMIN) * tFGM) - row.get('FGM', 0)
        ast_pct = 100.0 * row.get('AST', 0) / denom_ast if denom_ast not in [0, np.nan] else np.nan

        # --- Rebound rates (need team + opponent rebounds; if opponent missing, fall back to team-only approx)
        # ORB% = 100 * ORB * (TeamMIN/5) / ( MIN * (TeamORB + OppDRB) )
        # DRB% = 100 * DRB * (TeamMIN/5) / ( MIN * (TeamDRB + OppORB) )
        # TRB% = 100 * TRB * (TeamMIN/5) / ( MIN * (TeamTRB + OppTRB) )
        # Fallback approx if opponent columns are missing: use team-only denominators.
        opp_dreb = oDREB if oDREB > 0 else np.nan
        opp_oreb = oOREB if oOREB > 0 else np.nan
        opp_trb  = oTRB  if oTRB  > 0 else np.nan

        # ORB%
        denom_orb = (tOREB + (opp_dreb if not np.isnan(opp_dreb) else 0.0))
        orb_pct = 100.0 * row.get('OREB', 0) * (tMIN / 5.0) / (mp * denom_orb) if (mp > 0 and denom_orb > 0) else np.nan

        # DRB%
        denom_drb = (tDREB + (opp_oreb if not np.isnan(opp_oreb) else 0.0))
        drb_pct = 100.0 * row.get('DREB', 0) * (tMIN / 5.0) / (mp * denom_drb) if (mp > 0 and denom_drb > 0) else np.nan

        # TRB%
        denom_trb = (tTRB + (opp_trb if not np.isnan(opp_trb) else 0.0))
        trb_pct = 100.0 * row.get('REB', 0) * (tMIN / 5.0) / (mp * denom_trb) if (mp > 0 and denom_trb > 0) else np.nan

        return pd.Series({
            'USG% (true)': usg_true,
            'AST%': ast_pct,
            'ORB%': orb_pct,
            'DRB%': drb_pct,
            'TRB%': trb_pct
        })

    rates = out.apply(_team_rates, axis=1)
    out = pd.concat([out, rates], axis=1)

    # ---------- Cleanup
    float_cols = out.select_dtypes(include=['float', 'float64', 'float32']).columns
    out[float_cols] = out[float_cols].round(2)
    out = out.drop(columns=['FG_PCT', 'FG3_PCT', 'FT_PCT'], errors='ignore')

    return out


def generate_player_summary(player_name: str, stats_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    if stats_df.empty:
        return f"No available stats for {player_name}."

    lines = [f"📊 Full season-by-season stats for **{player_name}**:\n"]
    for i in range(len(stats_df)):
        s = stats_df.iloc[i]
        a = adv_df.iloc[i]
        lines.append("---")
        lines.append(f"### Season {s['SEASON_ID']} ({s['TEAM_ABBREVIATION']})")
        lines.append(f"- **PPG:** {s['PTS']:.1f}, **RPG:** {s['REB']:.1f}, **APG:** {s['AST']:.1f}")
        lines.append(f"- **SPG:** {s['STL']:.1f}, **BPG:** {s['BLK']:.1f}, **TPG:** {s['TOV']:.1f}")
        lines.append(f"- **Games Played:** {s['GP']}, **Minutes/Game:** {s['MIN']:.1f}")
        if 'FG%' in s:
            lines.append(f"- **FG%:** {s['FG%']:.1f}%, **3P%:** {s['3P%']:.1f}%, **FT%:** {s['FT%']:.1f}%")
        lines.append(f"- **TS%:** {a.get('TS%', np.nan):.2f}%, **EFG%:** {a.get('EFG%', np.nan):.2f}%, **PPS:** {a.get('PPS', np.nan):.2f}")
        lines.append(f"- **USG% (true):** {a.get('USG% (true)', np.nan):.2f}%")
        lines.append(
            f"- **PTS/36:** {a.get('PTS/36', np.nan):.2f}, **REB/36:** {a.get('REB/36', np.nan):.2f}, "
            f"**AST/36:** {a.get('AST/36', np.nan):.2f}, **STL/36:** {a.get('STL/36', np.nan):.2f}, "
            f"**BLK/36:** {a.get('BLK/36', np.nan):.2f}, **TOV/36:** {a.get('TOV/36', np.nan):.2f}"
        )
        lines.append(f"- **AST/TO Ratio:** {a.get('AST/TO', np.nan):.2f}\n")
        lines.append(f"- **3PAr:** {a.get('3PAr', np.nan):.2f}, **FTr:** {a.get('FTr', np.nan):.2f}\n")
        lines.append(f"- **FGM/36:** {a.get('FGM/36', np.nan):.2f}, **FGA/36:** {a.get('FGA/36', np.nan):.2f}, **3PM/36:** {a.get('FG3M/36', np.nan):.2f}\n")
        lines.append(f"- **OREB/36:** {a.get('OREB/36', np.nan):.2f}, **DREB/36:** {a.get('DREB/36', np.nan):.2f}\n")
        lines.append(f"- **AST%:** {a.get('AST%', np.nan):.2f}%, **ORB%:** {a.get('ORB%', np.nan):.2f}%, **DRB%:** {a.get('DRB%', np.nan):.2f}%, **TRB%:** {a.get('TRB%', np.nan):.2f}%\n")
    return "\n".join(lines)


def age_from_birthdate(iso_dt: str) -> int:
    birthdate = datetime.strptime(iso_dt.split('T')[0], "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


# ============ LOGOS (same as your original) ============
college_logos = {
    "Duke": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Duke_Blue_Devils_basketball_mark.svg/300px-Duke_Blue_Devils_basketball_mark.svg.png",
    "North Carolina": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/North_Carolina_Tar_Heels_logo.svg/500px-North_Carolina_Tar_Heels_logo.svg.png",
    "Kentucky": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Kentucky_Wildcats_logo.svg/300px-Kentucky_Wildcats_logo.svg.png",
    "Kansas": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Kansas_Jayhawks_1946_logo.svg/400px-Kansas_Jayhawks_1946_logo.svg.png",
    "UCLA": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/UCLA_Bruins_primary_logo.svg/400px-UCLA_Bruins_primary_logo.svg.png",
    "Ohio State": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Ohio_State_Buckeyes_logo.svg/400px-Ohio_State_Buckeyes_logo.svg.png",
    "Texas": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Texas_Longhorns_logo.svg/450px-Texas_Longhorns_logo.svg.png",
    "Southern California": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/USC_Trojans_logo.svg/244px-USC_Trojans_logo.svg.png",
    "Michigan": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Michigan_Wolverines_logo.svg/300px-Michigan_Wolverines_logo.svg.png",
    "Arizona": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Arizona_Wildcats_logo.svg/300px-Arizona_Wildcats_logo.svg.png",
    "Arkansas": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Arkansas_wordmark_2014.png/500px-Arkansas_wordmark_2014.png",
    "Metropolitans 92": "https://upload.wikimedia.org/wikipedia/en/2/2a/Metropolitans_92_logo.png",
    "NBA G League Ignite": "https://upload.wikimedia.org/wikipedia/en/thumb/8/88/NBA_G_League_Ignite_logo_%282022%29.svg/400px-NBA_G_League_Ignite_logo_%282022%29.svg.png",
    "St. Vincent-St. Mary HS (OH)": "https://upload.wikimedia.org/wikipedia/en/b/be/St._Vincent-St._Mary_High_School_logo.png",
    "Murray State": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Murray_State_Racers_wordmark.svg/500px-Murray_State_Racers_wordmark.svg.png",
    "Oklahoma": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Oklahoma_Sooners_logo.svg/250px-Oklahoma_Sooners_logo.svg.png",
    "Oklahoma State": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Oklahoma_State_University_system_logo.svg/450px-Oklahoma_State_University_system_logo.svg.png",
    "Iowa State": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Iowa_State_Cyclones_logo.svg/300px-Iowa_State_Cyclones_logo.svg.png",
    "Florida": "https://upload.wikimedia.org/wikipedia/en/thumb/9/99/Florida_Gators_men%27s_basketball_logo.svg/400px-Florida_Gators_men%27s_basketball_logo.svg.png",
    "Florida State": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Florida_State_Athletics_wordmark.svg/500px-Florida_State_Athletics_wordmark.svg.png",
    "Stanford": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Stanford_Cardinal_logo.svg/200px-Stanford_Cardinal_logo.svg.png",
    "Indiana": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Indiana_Hoosiers_logo.svg/250px-Indiana_Hoosiers_logo.svg.png",
    "TCU": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/TCU_Horned_Frogs_logo.svg/350px-TCU_Horned_Frogs_logo.svg.png",
    "Louisiana State": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/LSU_Athletics_logo.svg/400px-LSU_Athletics_logo.svg.png",
    "Oregon": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Oregon_Ducks_logo.svg/250px-Oregon_Ducks_logo.svg.png",
    "Auburn": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Auburn_Tigers_logo.svg/300px-Auburn_Tigers_logo.svg.png",
    "Vanderbilt": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Vanderbilt_Athletics_logo.svg/330px-Vanderbilt_Athletics_logo.svg.png",
    "Villanova": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Villanova_Wildcats_logo.svg/300px-Villanova_Wildcats_logo.svg.png",
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/b/be/Real_Madrid_Baloncesto.png",
    "Alabama": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Alabama_Crimson_Tide_logo.svg/300px-Alabama_Crimson_Tide_logo.svg.png",
    "FC Barcelona": "https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/410px-FC_Barcelona_%28crest%29.svg.png",
    "Conneticut": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Connecticut_Huskies_wordmark.svg/500px-Connecticut_Huskies_wordmark.svg.png",
    "Marquette": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Marquette_Golden_Eagles_logo.svg/300px-Marquette_Golden_Eagles_logo.svg.png",
    "Syracuse": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Syracuse_Orange_logo.svg/200px-Syracuse_Orange_logo.svg.png",
    "Baylor": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Baylor_Athletics_logo.svg/300px-Baylor_Athletics_logo.svg.png",
    "Creighton": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Creighton_athletics_wordmark_2013.png/500px-Creighton_athletics_wordmark_2013.png",
    "Purdue": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Purdue_Boilermakers_logo.svg/300px-Purdue_Boilermakers_logo.svg.png",
    "Memphis": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Memphis_Tigers_primary_wordmark.svg/500px-Memphis_Tigers_primary_wordmark.svg.png",
    "Illinois": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Illinois_Fighting_Illini_logo.svg/200px-Illinois_Fighting_Illini_logo.svg.png",
    "Seton Hall": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Seton_Hall_Pirates_wordmark.svg/250px-Seton_Hall_Pirates_wordmark.svg.png",
    "Louisville": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Louisville_Wordmark_%282023%29.svg/500px-Louisville_Wordmark_%282023%29.svg.png",
    "Michigan State": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Michigan_State_Spartans_wordmark.svg/400px-Michigan_State_Spartans_wordmark.svg.png",
    "Wake Forest": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Wake_Forest_University_Athletic_logo.svg/300px-Wake_Forest_University_Athletic_logo.svg.png",
    "Notre Dame": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Nd_athletics_gold_logo_2015.svg/300px-Nd_athletics_gold_logo_2015.svg.png",
    "Georgia": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Georgia_Athletics_logo.svg/400px-Georgia_Athletics_logo.svg.png",
    "Missouri": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Mizzou_Athletics_wordmark.svg/500px-Mizzou_Athletics_wordmark.svg.png",
    "Georgetown": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Georgetown_Hoyas_logo.svg/300px-Georgetown_Hoyas_logo.svg.png",
    "Texas A&M": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Texas_A%26M_University_logo.svg/300px-Texas_A%26M_University_logo.svg.png",
    "NBA Global Academy": "https://cdn.nba.com/logos/microsites/nba-academy.svg",
    "Gonzaga": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Gonzaga_Bulldogs_wordmark.svg/450px-Gonzaga_Bulldogs_wordmark.svg.png",
    "Iowa": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Iowa_Hawkeyes_wordmark.svg/400px-Iowa_Hawkeyes_wordmark.svg.png",
    "Providence": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Providence_wordmark1_2002.png/500px-Providence_wordmark1_2002.png",
    "Georgia Tech": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Georgia_Tech_Yellow_Jackets_logo.svg/350px-Georgia_Tech_Yellow_Jackets_logo.svg.png",
    "Maryland": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Maryland_Terrapins_logo.svg/250px-Maryland_Terrapins_logo.svg.png",
    "Wisconsin": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Wisconsin_Badgers_logo_basketball_red.svg/400px-Wisconsin_Badgers_logo_basketball_red.svg.png",
    "UNLV": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/UNLV_Rebels_wordmark.svg/500px-UNLV_Rebels_wordmark.svg.png",
    "Brigham Young": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/BYU_Stretch_Y_Logo.png/500px-BYU_Stretch_Y_Logo.png",
    "Houston": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Houston_Cougars_primary_logo.svg/300px-Houston_Cougars_primary_logo.svg.png",
    "Colorado": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Colorado_Buffaloes_wordmark_black.svg/400px-Colorado_Buffaloes_wordmark_black.svg.png",
    "Washington": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Washington_Huskies_logo.svg/300px-Washington_Huskies_logo.svg.png",
    "Utah": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Utah_Utes_primary_logo.svg/300px-Utah_Utes_primary_logo.svg.png",
    "Boise State": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Boise_State_Broncos_wordmark.svg/500px-Boise_State_Broncos_wordmark.svg.png",
    "Xavier": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Xavier_wordmark-basketball-fc-lt.svg/400px-Xavier_wordmark-basketball-fc-lt.svg.png",
    "California": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/California_Golden_Bears_logo.svg/250px-California_Golden_Bears_logo.svg.png",
    "Arizona State": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Arizona_State_Athletics_wordmark.svg/500px-Arizona_State_Athletics_wordmark.svg.png",
    "Pittsburgh": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Pitt_Panthers_wordmark.svg/350px-Pitt_Panthers_wordmark.svg.png",
    "St John's": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/St_johns_wordmark_red_2015.png/500px-St_johns_wordmark_red_2015.png",
    "Davidson": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Davidson_Wildcats_logo.png/250px-Davidson_Wildcats_logo.png",
}

# ============ SIDEBAR ============
with st.sidebar:
    st.header("🔍 Search Player")
    name = st.text_input("Enter an NBA player's name")
    search_clicked = st.button("Search")

# Session-state keys for first player
for key, default in [
    ("matches", []), ("player", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Handle first-player search
if search_clicked:
    found = players.find_players_by_full_name(name) if name else []
    exact = [p for p in found if p['full_name'].lower() == (name or "").lower()]
    found = exact if exact else found
    if not found:
        st.session_state["matches"] = []
        st.session_state["player"] = None
        st.sidebar.error("❌ No players found. Check spelling.")
    elif len(found) == 1:
        st.session_state["player"] = found[0]
        st.session_state["matches"] = []
    else:
        st.session_state["matches"] = found
        st.session_state["player"] = None  # wait for selection

# Show selection UI if multiple matches for first player
if st.session_state["matches"]:
    st.write("Multiple players found with that name:")
    options = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["matches"]}
    choice = st.radio(
        "Select a player:",
        ["⬇️ Pick a player"] + list(options.keys()),
        index=0,
        key="player_selection_radio"
    )
    if choice != "⬇️ Pick a player":
        st.session_state["player"] = options[choice]
        st.session_state["matches"] = []  # clear so text disappears

# ============ MAIN CONTENT ============
if st.session_state["player"]:
    player = st.session_state["player"]
    # Pull player info
    info = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]
    team_id = info.loc[0, 'TEAM_ID']
    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png"
    team_logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

    st.title("🏀 NBA Advanced Player Stats Explorer")

    # Tabs
    tab_info, tab_stats, tab_compare = st.tabs(["📋 Player Info", "📊 Stats", "🤝 Compare Players"])

    with tab_info:
        st.subheader("Player Info")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(headshot_url, width=220)
            st.image(team_logo_url, width=120)
        with c2:
            age = age_from_birthdate(info.loc[0, 'BIRTHDATE'])
            st.markdown(f"### {player['full_name']}")
            st.write(f"**Age:** {age}")
            st.write(f"**Height:** {info.loc[0, 'HEIGHT']}")
            st.write(f"**Weight:** {info.loc[0, 'WEIGHT']} lbs")
            st.write(f"**Position:** {info.loc[0, 'POSITION']}")
            college = info.loc[0, 'SCHOOL']
            st.write(f"**College:** {college}")
            # College + team logos side by side
            if college:
                c3, c4 = st.columns([1, 1])
                with c3:
                    if college in college_logos:
                        st.image(college_logos[college], width=120)
                with c4:
                    st.image(team_logo_url, width=120)

    with tab_stats:
        st.subheader("Most Recent Season Stats")
        raw = playercareerstats.PlayerCareerStats(player_id=player['id'], per_mode36='PerGame').get_data_frames()[0]
        adv = compute_full_advanced_stats(raw)

        # Latest season metrics (use last row; NBA API seasons are chronological)
        if not adv.empty:
            latest = adv.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PPG", f"{latest.get('PTS', np.nan):.1f}")
            m2.metric("RPG", f"{latest.get('REB', np.nan):.1f}")
            m3.metric("APG", f"{latest.get('AST', np.nan):.1f}")
            m4.metric("TS%", f"{latest.get('TS%', np.nan):.1f}%")

        st.dataframe(adv, use_container_width=True)

        # AI Assistant
        with st.expander("🧠 Ask the AI Assistant about this player"):
            if model:
                q = st.text_input("Ask something about this player:", key="ai_question")
                if q:
                    summary = generate_player_summary(player['full_name'], adv, adv)
                    prompt = (
                        f"You are an expert NBA analyst. Here is the stat summary for {player['full_name']}:\n\n"
                        f"{summary}\n\nQuestion: {q}\n\n"
                        f"Note: USG% and PER are rough estimates here."
                    )
                    with st.spinner("Analyzing..."):
                        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                        st.markdown("### 🧠 AI Analysis")
                        st.write(resp.text if hasattr(resp, "text") else "No response.")
            else:
                st.info("Add your Gemini API key to enable AI analysis.")

    with tab_compare:
        st.subheader("Compare Players")

        # Session-state for second player flow
        for key, default in [
            ("other_matches", []),
            ("other_player", None),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        other_name = st.text_input("Enter another player's name to compare:", key="other_name_input")
        search2 = st.button("Search Second Player")

        if search2:
            om = players.find_players_by_full_name(other_name) if other_name else []
            exact2 = [p for p in om if p['full_name'].lower() == (other_name or "").lower()]
            om = exact2 if exact2 else om
            if not om:
                st.session_state["other_matches"] = []
                st.session_state["other_player"] = None
                st.error("❌ No second player found. Check spelling.")
            elif len(om) == 1:
                st.session_state["other_player"] = om[0]
                st.session_state["other_matches"] = []
            else:
                st.session_state["other_matches"] = om
                st.session_state["other_player"] = None

        # If multiple matches for second player, show radio to pick
        if st.session_state["other_matches"]:
            st.write("Multiple players found with that name:")
            options2 = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["other_matches"]}
            pick2 = st.radio(
                "Select a player:",
                ["⬇️ Pick a player"] + list(options2.keys()),
                index=0,
                key="other_player_selection_radio"
            )
            if pick2 != "⬇️ Pick a player":
                st.session_state["other_player"] = options2[pick2]
                st.session_state["other_matches"] = []  # clear so text disappears

        other_player = st.session_state["other_player"]

        if other_player:
            st.success(f"Comparing **{player['full_name']}** vs **{other_player['full_name']}**")

            # Quick cards
            cL, cR = st.columns(2)
            # Left card
            with cL:
                info1 = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]
                st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png", width=180)
                st.markdown(f"**{player['full_name']}**")
                st.caption(f"{info1.loc[0, 'TEAM_NAME']} • {info1.loc[0, 'POSITION']}")
            # Right card
            with cR:
                info2 = commonplayerinfo.CommonPlayerInfo(player_id=other_player['id']).get_data_frames()[0]
                st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{other_player['id']}.png", width=180)
                st.markdown(f"**{other_player['full_name']}**")
                st.caption(f"{info2.loc[0, 'TEAM_NAME']} • {info2.loc[0, 'POSITION']}")

            # Pull careers + advanced
            raw1 = playercareerstats.PlayerCareerStats(player_id=player['id'], per_mode36='PerGame').get_data_frames()[0]
            raw2 = playercareerstats.PlayerCareerStats(player_id=other_player['id'], per_mode36='PerGame').get_data_frames()[0]
            adv1 = compute_full_advanced_stats(raw1)
            adv2 = compute_full_advanced_stats(raw2)

            # Add SEASON_START for aligning
            adv1['SEASON_START'] = adv1['SEASON_ID'].str[:4].astype(int)
            adv2['SEASON_START'] = adv2['SEASON_ID'].str[:4].astype(int)

            # Shared stats to compare (exclude IDs)
            excluded = {'SEASON_ID', 'PLAYER_ID', 'TEAM_ID', 'LEAGUE_ID', 'TEAM_ABBREVIATION'}
            shared_stats = sorted([c for c in adv1.columns if c in adv2.columns and c not in excluded and pd.api.types.is_numeric_dtype(adv1[c])])

            # Choose stat
            stat_choice = st.selectbox("📊 Choose a stat to compare:", shared_stats, index=shared_stats.index('PTS') if 'PTS' in shared_stats else 0)

            # Merge overlapping seasons
            common = adv1[['SEASON_START', 'SEASON_ID', stat_choice]].merge(
                adv2[['SEASON_START', 'SEASON_ID', stat_choice]],
                on='SEASON_START', suffixes=(f"_{player['full_name']}", f"_{other_player['full_name']}")
            )

            if common.empty:
                st.warning("No overlapping seasons to compare.")
            else:
                # Interactive line chart with Plotly
                fig_df = pd.DataFrame({
                    "Season": common[f"SEASON_ID_{player['full_name']}"],
                    player['full_name']: common[f"{stat_choice}_{player['full_name']}"],
                    other_player['full_name']: common[f"{stat_choice}_{other_player['full_name']}"],
                })
                fig = px.line(fig_df, x="Season", y=[player['full_name'], other_player['full_name']], markers=True,
                              title=f"{stat_choice} — Overlapping Seasons")
                fig.update_layout(xaxis_title="Season", yaxis_title=stat_choice, legend_title="Player")
                st.plotly_chart(fig, use_container_width=True)

            # Side-by-side advanced tables
            st.subheader("📊 Advanced Stats (All Seasons)")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"**{player['full_name']}**")
                st.dataframe(adv1.drop(columns=['SEASON_START']), use_container_width=True)
            with t2:
                st.markdown(f"**{other_player['full_name']}**")
                st.dataframe(adv2.drop(columns=['SEASON_START']), use_container_width=True)

            # AI compare
            with st.expander("🧠 Ask the AI Assistant about these players"):
                if model:
                    q2 = st.text_input("Ask something about these players:", key="ai_compare_question")
                    if q2:
                        sum1 = generate_player_summary(player['full_name'], adv1, adv1)
                        sum2 = generate_player_summary(other_player['full_name'], adv2, adv2)
                        prompt2 = (
                            f"You are an expert NBA analyst. Compare these two players.\n\n"
                            f"Player 1: {player['full_name']}\n{sum1}\n\n"
                            f"Player 2: {other_player['full_name']}\n{sum2}\n\n"
                            f"Question: {q2}\n\n"
                            f"Note: Some advanced metrics are estimates."
                        )
                        with st.spinner("Analyzing..."):
                            resp2 = model.generate_content(prompt2, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                            st.markdown("### 🧠 AI Analysis")
                            st.write(resp2.text if hasattr(resp2, "text") else "No response.")
                else:
                    st.info("Add your Gemini API key to enable AI analysis.")

else:
    st.info("Use the sidebar to search for a player.")
