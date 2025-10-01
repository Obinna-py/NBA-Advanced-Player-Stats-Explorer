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

# extra imports for robustness
import time, random
from requests.exceptions import ReadTimeout, ConnectionError

# ============ CONFIG ============
st.set_page_config(page_title="NBA Advanced Player Stats Explorer", layout="wide")

# Safely load Gemini API key
GEMINI_API_KEY = "AIzaSyC0APdZnTD2G3GtPUBmWwB-XhNauX4TqUo"  # consider env var for production
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
else:
    model = None  # AI features disabled if no key

# ==================== NETWORK HARDENING ====================
def _with_retry(fetch_fn, *, tries=4, base=1.0):
    """Exponential backoff with jitter for nba_api calls."""
    for i in range(tries):
        try:
            return fetch_fn()
        except (ReadTimeout, ConnectionError):
            if i == tries - 1:
                raise
            sleep_s = base * (2 ** i) + random.random() * 0.4
            time.sleep(sleep_s)

@st.cache_data(ttl=3600, show_spinner=False)
def get_player_career(player_id: int, per_mode: str = 'Totals') -> pd.DataFrame:
    def _fetch():
        return playercareerstats.PlayerCareerStats(
            player_id=player_id,
            per_mode36=per_mode,
            timeout=60
        ).get_data_frames()[0]
    return _with_retry(_fetch, tries=4, base=1.0)

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_for_season(season: str) -> pd.DataFrame:
    """
    Fetch team totals for a given season (team-level box totals).
    One network call per season; normalized columns; cached.
    """
    def _fetch():
        return leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="Totals",
            measure_type_detailed_defense="Base",
            timeout=60,
            proxy=None
        ).get_data_frames()[0]

    df = _with_retry(_fetch, tries=4, base=1.0)

    needed = ["TEAM_ID","TEAM_ABBREVIATION","MIN","FGA","FTA","TOV","FGM","REB","OREB","DREB",
              "OPP_REB","OPP_OREB","OPP_DREB"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
    df["TRB"] = df["REB"]
    df["SEASON_ID"] = season
    return df[["SEASON_ID"] + needed + ["TRB"]].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_totals_many(seasons: list[str]) -> pd.DataFrame:
    """Batch fetch team totals for all needed seasons once."""
    seasons = sorted(set([s for s in seasons if isinstance(s, str) and s]))
    frames = [get_team_totals_for_season(s) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ============ HELPERS ============
def compute_full_advanced_stats(player_df_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized advanced stats. Input must be PER-MODE = Totals (not PerGame).
    Adds TS%, EFG%, PPS, 3PAr, FTr, per-36, USG% (true), AST%, ORB%/DRB%/TRB%.
    Uses a merge with team totals (one fetch per season).
    """
    if player_df_totals is None or player_df_totals.empty:
        return player_df_totals

    out = player_df_totals.copy()

    # ---- Shooting/efficiency (vectorized)
    if 'FG_PCT' in out.columns: out['FG%']  = out['FG_PCT'] * 100
    if 'FG3_PCT' in out.columns: out['3P%'] = out['FG3_PCT'] * 100
    if 'FT_PCT' in out.columns: out['FT%']  = out['FT_PCT'] * 100

    denom_ts = (out.get('FGA',0) + 0.44 * out.get('FTA',0))
    out['TS%']  = np.where(denom_ts>0, out.get('PTS',0) / (2*denom_ts) * 100, np.nan)
    out['EFG%'] = np.where(out.get('FGA',0)>0, (out.get('FGM',0) + 0.5*out.get('FG3M',0)) / out.get('FGA',0) * 100, np.nan)
    out['PPS']  = np.where(out.get('FGA',0)>0, out.get('PTS',0) / out.get('FGA',0), np.nan)
    out['3PAr'] = np.where(out.get('FGA',0)>0, out.get('FG3A',0) / out.get('FGA',0), np.nan)
    out['FTr']  = np.where(out.get('FGA',0)>0, out.get('FTA',0)  / out.get('FGA',0), np.nan)
    out['AST/TO'] = np.where(out.get('TOV',0)>0, out.get('AST',0)/out.get('TOV',0), np.nan)

    # ---- Per-36 (vectorized on totals)
    mins = out.get('MIN', 0)
    def per36(col): return np.where(mins>0, out.get(col,0) / mins * 36, np.nan)
    for stat in ['PTS','REB','AST','STL','BLK','TOV','FGM','FGA','FG3M','OREB','DREB']:
        out[f'{stat}/36'] = per36(stat)

    # ---- Team context merge (single pass)
    seasons_needed = out['SEASON_ID'].dropna().unique().tolist()
    team_totals = get_team_totals_many(seasons_needed)

    # Merge on season+team; "TOT" rows (TEAM_ID==0) won't match, will yield NaNs for team-context rates
    out = out.merge(
        team_totals.add_prefix('TEAM_'),
        left_on=['SEASON_ID','TEAM_ID'],
        right_on=['TEAM_SEASON_ID','TEAM_TEAM_ID'],
        how='left'
    )

    # Protect from missing merges
    tMIN = out.get('TEAM_MIN', 0).astype(float)
    tFGA = out.get('TEAM_FGA', 0).astype(float)
    tFTA = out.get('TEAM_FTA', 0).astype(float)
    tTOV = out.get('TEAM_TOV', 0).astype(float)
    tFGM = out.get('TEAM_FGM', 0).astype(float)
    tTRB = np.where(out.get('TEAM_TRB', np.nan).isna(), out.get('TEAM_REB', 0), out.get('TEAM_TRB', 0)).astype(float)
    tOREB = out.get('TEAM_OREB', 0).astype(float)
    tDREB = out.get('TEAM_DREB', 0).astype(float)
    oTRB  = out.get('TEAM_OPP_REB', 0).astype(float)
    oOREB = out.get('TEAM_OPP_OREB', 0).astype(float)
    oDREB = out.get('TEAM_OPP_DREB', 0).astype(float)

    mp = out.get('MIN', 0).astype(float)

    # USG% (true)
    denom_usg = (tFGA + 0.44*tFTA + tTOV)
    num_usg   = (out.get('FGA',0) + 0.44*out.get('FTA',0) + out.get('TOV',0)) * np.where(tMIN>0, tMIN/5.0, 0)
    out['USG% (true)'] = np.where((denom_usg>0) & (mp>0), 100.0 * num_usg / (mp * denom_usg), np.nan)

    # AST% = 100 * AST / ( (MIN/TeamMIN)*TeamFGM - FGM )
    denom_ast = (np.where(tMIN>0, mp/tMIN, 0) * tFGM) - out.get('FGM',0)
    out['AST%'] = np.where(denom_ast != 0, 100.0 * out.get('AST',0) / denom_ast, np.nan)

    # Rebound rates
    denom_orb = tOREB + np.where(oDREB>0, oDREB, 0.0)
    denom_drb = tDREB + np.where(oOREB>0, oOREB, 0.0)
    denom_trb = tTRB  + np.where(oTRB >0, oTRB,  0.0)
    scale = np.where((mp>0) & (tMIN>0), (tMIN/5.0) / mp, np.nan)

    out['ORB%'] = np.where(denom_orb>0, 100.0 * out.get('OREB',0) * scale / denom_orb, np.nan)
    out['DRB%'] = np.where(denom_drb>0, 100.0 * out.get('DREB',0) * scale / denom_drb, np.nan)
    out['TRB%'] = np.where(denom_trb>0, 100.0 * out.get('REB',0)  * scale / denom_trb, np.nan)

    # Cleanup
    out = out.drop(columns=[c for c in ['FG_PCT','FG3_PCT','FT_PCT','TEAM_SEASON_ID','TEAM_TEAM_ID'] if c in out.columns])
    float_cols = out.select_dtypes(include=['float', 'float64', 'float32']).columns
    out[float_cols] = out[float_cols].round(2)
    out = out[[c for c in out.columns if not c.startswith("TEAM_")]]
    return out

def generate_player_summary(player_name: str, stats_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    if stats_df is None or stats_df.empty:
        return f"No available stats for {player_name}."

    lines = [f"📊 Full season-by-season stats for **{player_name}**:\n"]
    for i in range(len(stats_df)):
        s = stats_df.iloc[i]
        a = adv_df.iloc[i] if (adv_df is not None and i < len(adv_df)) else {}
        lines.append("---")
        lines.append(f"### Season {s['SEASON_ID']} ({s['TEAM_ABBREVIATION']})")
        lines.append(f"- **PPG:** {s.get('PTS', np.nan):.1f}, **RPG:** {s.get('REB', np.nan):.1f}, **APG:** {s.get('AST', np.nan):.1f}")
        lines.append(f"- **SPG:** {s.get('STL', np.nan):.1f}, **BPG:** {s.get('BLK', np.nan):.1f}, **TPG:** {s.get('TOV', np.nan):.1f}")
        lines.append(f"- **Games Played:** {s.get('GP', np.nan)}, **Minutes/Game:** {s.get('MIN', np.nan):.1f}")
        if 'FG_PCT' in s:
            lines.append(f"- **FG%:** {s.get('FG_PCT', np.nan)*100:.1f}%, **3P%:** {s.get('FG3_PCT', np.nan)*100:.1f}%, **FT%:** {s.get('FT_PCT', np.nan)*100:.1f}%")
        lines.append(f"- **TS%:** {a.get('TS%', np.nan):.2f}%, **EFG%:** {a.get('EFG%', np.nan):.2f}%, **PPS:** {a.get('PPS', np.nan):.2f}")
        lines.append(f"- **USG% (true):** {a.get('USG% (true)', np.nan):.2f}%")
        lines.append(
            f"- **PTS/36:** {a.get('PTS/36', np.nan):.2f}, **REB/36:** {a.get('REB/36', np.nan):.2f}, "
            f"**AST/36:** {a.get('AST/36', np.nan):.2f}, **STL/36:** {a.get('STL/36', np.nan):.2f}, "
            f"**BLK/36:** {a.get('BLK/36', np.nan):.2f}, **TOV/36:** {a.get('TOV/36', np.nan):.2f}"
        )
        lines.append(f"- **AST/TO Ratio:** {a.get('AST/TO', np.nan):.2f}")
        lines.append(f"- **3PAr:** {a.get('3PAr', np.nan):.2f}, **FTr:** {a.get('FTr', np.nan):.2f}")
        lines.append(f"- **ORB%:** {a.get('ORB%', np.nan):.2f}%, **DRB%:** {a.get('DRB%', np.nan):.2f}%, **TRB%:** {a.get('TRB%', np.nan):.2f}%")
        lines.append(f"- **AST%:** {a.get('AST%', np.nan):.2f}%")
    return "\n".join(lines)

def age_from_birthdate(iso_dt: str) -> int:
    birthdate = datetime.strptime(iso_dt.split('T')[0], "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

# ---------- Question Ideas helpers ----------
def _compact_player_context(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    g = lambda k, d=np.nan: float(row[k]) if k in row and pd.notna(row[k]) else d
    return {
        "season": row.get("SEASON_ID", "Unknown"),
        "team": row.get("TEAM_ABBREVIATION", "UNK"),
        "ppg": g("PTS"),
        "rpg": g("REB"),
        "apg": g("AST"),
        "ts": g("TS%"),
        "efg": g("EFG%"),
        "usg": g("USG% (true)") if "USG% (true)" in row else np.nan,
        "mpg": g("MIN"),
    }

# ---------- Question Ideas (stat-answerable) ----------
def _seed_eval_questions(player_name: str, ctx: dict) -> list[str]:
    """Stat-based, answerable by the data we already have (PPG/RPG/APG, %, TS, eFG, AST%, USG, per-36, etc)."""
    season = ctx.get("season", "latest")
    ts = ctx.get("ts", np.nan); ppg = ctx.get("ppg", np.nan); usg = ctx.get("usg", np.nan)

    base = [
        f"What was {player_name}'s best season by TS%?",
        f"In {season}, is {player_name} more scorer or playmaker (PPG vs APG)?",
        f"Is {player_name} an efficient scorer (TS% vs league avg ~57%)?",
        f"Has {player_name}'s usage outpaced efficiency (USG vs TS%)?",
        f"Is {player_name} a good passer (AST% and AST/TO trend)?",
        f"Did {player_name} peak already? Which season looks like the peak?",
        f"How does {player_name}'s eFG% trend over the last 3 seasons?",
        f"Is {player_name} a volume 3PT shooter (3PAr) or selective?",
        f"Does {player_name} draw fouls (FTr) at a high rate?",
        f"Is {player_name} turnover-prone (TOV and AST/TO trend)?",
        f"What is {player_name}'s best scoring rate (PTS/36) season?",
        f"Did rebounding improve or decline (ORB%/DRB%/TRB% trend)?",
        f"How does {player_name}'s shooting split FG% / 3P% / FT% evolve?",
        f"Is {player_name} above average at the rim (proxy via eFG% change)?",
        f"Did {player_name} change role after a team switch (USG/AST% jump)?",
        f"Who was {player_name} statistically similar to in best season?",
        f"Is clutch proxy improving (TS% change year over year)?",
        f"Does per-36 scoring align with actual minutes (MP vs PTS/36)?",
        f"Is {player_name} more on-ball or off-ball (USG vs 3PAr)?",
        f"What season combined peak scoring and playmaking (PTS/36 + AST%)?",
        f"Is {player_name} a good rebounder for position (TRB% vs role expectation)?",
        f"Which season shows best two-way profile (TS% plus STL/BLK per-36)?",
        f"Is shot selection improving (PPS trend)?",
        f"Does {player_name} elevate teammates (AST% trend at stable minutes)?",
        f"How does efficiency change with workload (USG bins vs TS%)?",
        f"What was {player_name}'s worst season and what slipped most?",
        f"Is free-throw shooting stable (FT% volatility)?",
        f"Are threes adding value (3PAr up with steady eFG%)?",
        f"What’s {player_name}'s prime window based on the metrics?",
        f"What single skill upgrade would add most value (based on gaps)?",
    ]

    # Small conditional inserts so ideas feel personalized
    if pd.notna(ts) and ts >= 60:
        base.insert(0, f"Is {player_name} an elite efficiency scorer (TS% ≥ {ts:.1f}%)?")
    if pd.notna(ppg) and ppg >= 25:
        base.insert(1, f"Is {player_name}'s volume ({ppg:.1f} PPG) supported by strong TS%?")
    if pd.notna(usg) and usg >= 28:
        base.insert(2, f"Has high usage (~{usg:.1f}%) hurt or helped efficiency over time?")

    # keep concise (~30) and unique
    out, seen = [], set()
    for q in base:
        k = q.lower()
        if k in seen: 
            continue
        seen.add(k); out.append(q)
    return out[:30]

def _ai_question_ideas(player_name: str, ctx: dict, model=None, topic_hint: str="") -> list[str]:
    """Use model if available, but constrain to stat-answerable evaluation questions only."""
    seeds = _seed_eval_questions(player_name, ctx)
    if not model:
        return seeds

    prompt = (
        "Generate 20 short, evaluative questions about ONE NBA player that can be answered "
        "strictly from these metrics: PPG, RPG, APG, FG%, 3P%, FT%, TS%, eFG%, PPS, 3PAr, FTr, "
        "USG% (true), AST%, TRB%/ORB%/DRB%, per-36 stats, TOV, AST/TO, and season-to-season trends. "
        "No tactics, no play diagrams, no team scheme questions. Keep under 14 words each. "
        "Do NOT ask for on/off or lineup data. Return as a numbered list.\n\n"
        f"Player: {player_name}\nContext: {ctx}\nFocus hint: {topic_hint or 'none'}"
    )
    try:
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 256, "temperature": 0.4})
        text = getattr(resp, "text", "") or ""
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        items = []
        for l in lines:
            items.append(l.split(" ", 1)[1] if l[:2].isdigit() and " " in l else l.lstrip("0123456789.) ").strip())
        # fallback to seeds if the model drifts off brief
        filtered = [i for i in items if i and len(i) <= 80 and "coverage" not in i.lower() and "lineup" not in i.lower()]
        if not filtered:
            return seeds
        # dedupe and cap
        seen, out = set(), []
        for i in filtered + seeds:
            k = i.lower()
            if k in seen: 
                continue
            seen.add(k); out.append(i)
        return out[:20]
    except Exception:
        return seeds

@st.cache_data(ttl=3600, show_spinner=False)
def cached_ai_question_ideas(player_name: str, ctx: dict, topic_hint: str, use_model: bool) -> list[str]:
    return _ai_question_ideas(player_name, ctx, model if use_model else None, topic_hint)

def abbrev(text: str, max_len: int = 40) -> str:
    return text if len(text) <= max_len else text[:max_len-1] + "…"

# ---------- Comparison Question Ideas (stat-answerable) ----------
def _seed_compare_questions(p1: str, p2: str, c1: dict, c2: dict) -> list[str]:
    """Short evaluative questions answerable from PPG/RPG/APG, TS%, eFG%, USG% (true), AST%, TRB% and per-36 trends."""
    def fmt(v): 
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    q = []
    q += [f"Who had the better peak season by TS%: {p1} or {p2}?",
          f"Who is the better passer (AST% and AST/TO): {p1} or {p2}?",
          f"Who scores more efficiently (TS%, eFG%): {p1} or {p2}?",
          f"Whose usage is higher and does it match efficiency?",
          f"Who is the better rebounder for position (TRB%)?",
          f"Which season was each player’s statistical peak?",
          f"Who improved more year over year (TS%/PPS trend)?",
          f"Who is more turnover-prone (TOV, AST/TO)?",
          f"Who gets to the line more (FTr)?",
          f"Who takes more threes (3PAr) and is it efficient?",
          f"Who contributes more per-36 on scoring and playmaking?",
          f"Who has the stronger two-way season (TS% with STL/BLK per-36)?",
          f"Whose efficiency holds at higher usage?",
          f"Who is closer to their prime based on recent trends?",
          f"What skill would most raise each player’s value?"]
    # tiny personalizations
    for label, ctx in [(p1, c1), (p2, c2)]:
        ts = fmt(ctx.get("ts")); usg = fmt(ctx.get("usg")); ppg = fmt(ctx.get("ppg"))
        if ts and ts >= 60: q.insert(0, f"Is {label} sustaining elite TS% (≥{ts:.1f}%) versus the other?")
        if usg and usg >= 28: q.insert(1, f"Does {label}'s high usage (~{usg:.1f}%) outpace the other’s efficiency?")
        if ppg and ppg >= 25: q.insert(2, f"Is {label}'s volume ({ppg:.1f} PPG) more efficient than {p1 if label==p2 else p2}?")
    # dedupe & cap
    out, seen = [], set()
    for s in q:
        k = s.lower()
        if k in seen: continue
        seen.add(k); out.append(s)
    return out[:24]

def _ai_compare_question_ideas(p1: str, p2: str, c1: dict, c2: dict, model=None, topic_hint: str="") -> list[str]:
    seeds = _seed_compare_questions(p1, p2, c1, c2)
    if not model:
        return seeds
    prompt = (
        "Generate 18 short comparison questions about TWO NBA players that can be answered strictly "
        "from per-season stats we already have: PPG, RPG, APG, FG%, 3P%, FT%, TS%, eFG%, PPS, 3PAr, FTr, "
        "USG% (true), AST%, TRB%/ORB%/DRB%, TOV, AST/TO, per-36, and simple trends. "
        "No tactics, no lineup/on-off data. ≤ 14 words each. Numbered list.\n\n"
        f"Player A: {p1}  Context: {c1}\n"
        f"Player B: {p2}  Context: {c2}\n"
        f"Focus hint: {topic_hint or 'none'}"
    )
    try:
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 256, "temperature": 0.4})
        text = getattr(resp, "text", "") or ""
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        items = []
        for l in lines:
            items.append(l.split(" ", 1)[1] if l[:2].isdigit() and " " in l else l.lstrip("0123456789.) ").strip())
        filtered = [i for i in items if i and len(i) <= 80 and all(bad not in i.lower() for bad in ["coverage","scheme","lineup","on/off","play type"])]
        if not filtered:
            return seeds
        seen, out = set(), []
        for i in filtered + seeds:
            k = i.lower()
            if k in seen: continue
            seen.add(k); out.append(i)
        return out[:20]
    except Exception:
        return seeds

@st.cache_data(ttl=3600, show_spinner=False)
def cached_ai_compare_question_ideas(p1: str, p2: str, c1: dict, c2: dict, topic_hint: str, use_model: bool) -> list[str]:
    return _ai_compare_question_ideas(p1, p2, c1, c2, model if use_model else None, topic_hint)


# ============ LOGOS ============
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
for key, default in [("matches", []), ("player", None)]:
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

        # UI: speed mode
        speed_mode = st.toggle(
            "Compute advanced for ALL seasons (slower)",
            value=False,
            help="Off = latest season only (fast). On = all seasons (slower)."
        )

        # Fetch PerGame (for display) and Totals (for advanced)
        raw_pergame = get_player_career(player['id'], per_mode='PerGame')
        raw_totals  = get_player_career(player['id'], per_mode='Totals')

        if raw_pergame is None or raw_pergame.empty:
            st.warning("No PerGame data available for this player right now.")
        if raw_totals is None or raw_totals.empty:
            st.warning("No Totals data available for this player right now.")

        # FAST PATH: only latest season for advanced calcs by default
        if raw_totals is not None and not raw_totals.empty:
            if speed_mode:
                adv_source = raw_totals
            else:
                latest_season = raw_totals['SEASON_ID'].iloc[-1]
                adv_source = raw_totals[raw_totals['SEASON_ID'] == latest_season].copy()

            with st.spinner("Computing advanced metrics…"):
                adv = compute_full_advanced_stats(adv_source)
        else:
            adv = pd.DataFrame()

        # Latest metrics (PerGame where possible; TS% from adv)
        latest_src = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
        if latest_src is not None and not latest_src.empty:
            latest = latest_src.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PPG", f"{latest.get('PTS', np.nan):.1f}")
            m2.metric("RPG", f"{latest.get('REB', np.nan):.1f}")
            m3.metric("APG", f"{latest.get('AST', np.nan):.1f}")
            ts_val = adv.iloc[-1]['TS%'] if not adv.empty and 'TS%' in adv.columns else np.nan
            m4.metric("TS%", f"{ts_val:.1f}%")

        # ---- 💡 Question Ideas (clickable chips -> fill AI textbox)
        with st.expander("💡 Question Ideas for this player", expanded=False):
            preset = st.radio(
                "Quick presets",
                ["Overview", "Scoring & Efficiency", "Playmaking & TOs", "Rebounding & Defense", "Peak & Trends"],
                horizontal=True,
                key="idea_preset",
            )

            topic_map = {
                "Overview": "balanced overview; mix of efficiency, usage, passing, rebounding, trends",
                "Scoring & Efficiency": "PPG, TS%, eFG%, PPS, shot selection, 3PAr, FTr, per-36 scoring",
                "Playmaking & TOs": "AST%, AST/TO, assist trends, turnover rate, usage vs passing load",
                "Rebounding & Defense": "TRB%, ORB%, DRB%, STL/36, BLK/36, defensive impact proxies",
                "Peak & Trends": "best season, worst season, year-over-year changes, prime window indicators",
            }
            topic_default = topic_map.get(preset, "")
            topic = st.text_input("Optional focus (refines suggestions):", value=topic_default, key="idea_focus")

            ctx = _compact_player_context(adv if not adv.empty else raw_pergame)
            ideas = cached_ai_question_ideas(player['full_name'], ctx, topic, use_model=(model is not None))

            st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
            cols_per_row = 2  # wider buttons to avoid vertical text
            for i in range(0, len(ideas), cols_per_row):
                row = ideas[i:i+cols_per_row]
                cols = st.columns(len(row))
                for c, idea in zip(cols, row):
                    short = abbrev(idea, 32)
                    with c:
                        if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"idea_btn_{i}_{short}"):
                            st.session_state["ai_question"] = idea
                            st.rerun()

        # Advanced table(s)
        if adv is not None and not adv.empty:
            st.dataframe(adv, use_container_width=True)
            if not speed_mode:
                st.info("Showing latest season advanced metrics. Turn on “ALL seasons” above to compute the full career (slower).")

        # AI Assistant
        with st.expander("🧠 Ask the AI Assistant about this player"):
            if model:
                q = st.text_input("Ask something about this player:", key="ai_question")
                if q:
                    pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                    adv_for_summary     = adv if adv is not None else pd.DataFrame()
                    summary = generate_player_summary(player['full_name'], pergame_for_summary, adv_for_summary)
                    prompt = (
                        f"You are an expert NBA analyst. Here is the stat summary for {player['full_name']}:\n\n"
                        f"{summary}\n\nQuestion: {q}\n\n"
                        f"Note: Some advanced metrics are estimates from team-context formulas."
                    )
                    with st.spinner("Analyzing…"):
                        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                        st.markdown("### 🧠 AI Analysis")
                        st.write(resp.text if hasattr(resp, "text") else "No response.")
            else:
                st.info("Add your Gemini API key to enable AI analysis.")

    with tab_compare:
        st.subheader("Compare Players")

        # Session-state for second player flow
        for key, default in [("other_matches", []), ("other_player", None)]:
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
            p1 = player['full_name']; p2 = other_player['full_name']
            st.success(f"Comparing **{p1}** vs **{p2}**")

            # Quick cards
            cL, cR = st.columns(2)
            with cL:
                info1 = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]
                st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png", width=180)
                st.markdown(f"**{p1}**")
                st.caption(f"{info1.loc[0, 'TEAM_NAME']} • {info1.loc[0, 'POSITION']}")
            with cR:
                info2 = commonplayerinfo.CommonPlayerInfo(player_id=other_player['id']).get_data_frames()[0]
                st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{other_player['id']}.png", width=180)
                st.markdown(f"**{p2}**")
                st.caption(f"{info2.loc[0, 'TEAM_NAME']} • {info2.loc[0, 'POSITION']}")

            # ===== SPEED TOGGLE for Advanced in Compare =====
            comp_speed_mode = st.toggle(
                "Compute advanced for ALL seasons in comparison (slower)",
                value=False,
                help="Off = latest season only (fast). On = all seasons (slower).",
                key="compare_speed_toggle"
            )

            # Pull careers (PerGame for charting; Totals for advanced)
            raw1_pg = get_player_career(player['id'], per_mode='PerGame')
            raw2_pg = get_player_career(other_player['id'], per_mode='PerGame')
            raw1_t  = get_player_career(player['id'], per_mode='Totals')
            raw2_t  = get_player_career(other_player['id'], per_mode='Totals')

            # Choose advanced source: latest only (fast) or all seasons
            if comp_speed_mode:
                adv1_src, adv2_src = raw1_t, raw2_t
            else:
                latest1 = raw1_t['SEASON_ID'].iloc[-1] if not raw1_t.empty else None
                latest2 = raw2_t['SEASON_ID'].iloc[-1] if not raw2_t.empty else None
                adv1_src = raw1_t[raw1_t['SEASON_ID'] == latest1] if latest1 else raw1_t
                adv2_src = raw2_t[raw2_t['SEASON_ID'] == latest2] if latest2 else raw2_t

            with st.spinner("Computing comparison advanced metrics…"):
                adv1 = compute_full_advanced_stats(adv1_src) if adv1_src is not None and not adv1_src.empty else pd.DataFrame()
                adv2 = compute_full_advanced_stats(adv2_src) if adv2_src is not None and not adv2_src.empty else pd.DataFrame()

            # Build compact contexts for question ideas
            ctx1_src = adv1 if (isinstance(adv1, pd.DataFrame) and not adv1.empty) else raw1_pg
            ctx2_src = adv2 if (isinstance(adv2, pd.DataFrame) and not adv2.empty) else raw2_pg
            c1 = _compact_player_context(ctx1_src)
            c2 = _compact_player_context(ctx2_src)

            # ===== Comparison Question Ideas (chips) =====
            with st.expander("💡 Comparison Question Ideas for these players", expanded=False):
                preset = st.radio(
                    "Quick presets",
                    ["Overview", "Scoring & Efficiency", "Playmaking & TOs", "Rebounding & Defense", "Peak & Trends"],
                    horizontal=True,
                    key="compare_idea_preset",
                )
                topic_map = {
                    "Overview": "balanced; efficiency, usage, passing, rebounding, trends",
                    "Scoring & Efficiency": "TS%, eFG%, PPS, 3PAr, FTr, PTS/36, shooting splits",
                    "Playmaking & TOs": "AST%, AST/TO, TOV trend, usage vs passing load",
                    "Rebounding & Defense": "TRB%, ORB%, DRB%, STL/36, BLK/36",
                    "Peak & Trends": "best/worst seasons, YoY changes, prime window",
                }
                topic_default = topic_map.get(preset, "")
                topic = st.text_input("Optional focus (refines suggestions):", value=topic_default, key="compare_idea_focus")

                ideas_cmp = cached_ai_compare_question_ideas(p1, p2, c1, c2, topic, use_model=(model is not None))
                st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
                cols_per_row = 2
                for i in range(0, len(ideas_cmp), cols_per_row):
                    row = ideas_cmp[i:i+cols_per_row]
                    cols = st.columns(len(row))
                    for c, idea in zip(cols, row):
                        short = abbrev(idea, 40)
                        with c:
                            if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"cmp_idea_btn_{i}_{short}"):
                                st.session_state["ai_compare_question"] = idea
                                st.rerun()

            # ===== Overlapping seasons chart (PerGame for readability) =====
            if not raw1_pg.empty: raw1_pg['SEASON_START'] = raw1_pg['SEASON_ID'].str[:4].astype(int)
            if not raw2_pg.empty: raw2_pg['SEASON_START'] = raw2_pg['SEASON_ID'].str[:4].astype(int)

            source_for_shared = raw1_pg if not raw1_pg.empty else adv1
            other_for_shared  = raw2_pg if not raw2_pg.empty else adv2
            excluded = {'SEASON_ID', 'PLAYER_ID', 'TEAM_ID', 'LEAGUE_ID', 'TEAM_ABBREVIATION'}
            shared_stats = sorted([c for c in source_for_shared.columns if c in other_for_shared.columns and c not in excluded and pd.api.types.is_numeric_dtype(source_for_shared[c])])

            default_idx = shared_stats.index('PTS') if 'PTS' in shared_stats else 0 if shared_stats else 0
            stat_choice = st.selectbox("📊 Choose a stat to compare:", shared_stats or ['PTS'], index=default_idx, key="cmp_stat_choice")

            if not raw1_pg.empty and not raw2_pg.empty:
                common = raw1_pg[['SEASON_START', 'SEASON_ID', stat_choice]].merge(
                    raw2_pg[['SEASON_START', 'SEASON_ID', stat_choice]],
                    on='SEASON_START', suffixes=(f"_{p1}", f"_{p2}")
                )
            else:
                common = pd.DataFrame()

            if common.empty:
                st.warning("No overlapping seasons to compare.")
            else:
                fig_df = pd.DataFrame({
                    "Season": common[f"SEASON_ID_{p1}"],
                    p1: common[f"{stat_choice}_{p1}"],
                    p2: common[f"{stat_choice}_{p2}"],
                })
                fig = px.line(fig_df, x="Season", y=[p1, p2], markers=True,
                            title=f"{stat_choice} — Overlapping Seasons")
                fig.update_layout(xaxis_title="Season", yaxis_title=stat_choice, legend_title="Player")
                st.plotly_chart(fig, use_container_width=True)

            # ===== Side-by-side advanced tables =====
            st.subheader("📊 Advanced Stats")
            if comp_speed_mode:
                st.caption("All seasons (slower).")
            else:
                st.caption("Latest season only (fast). Toggle above for full career.")

            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"**{p1}**")
                st.dataframe(adv1, use_container_width=True)
            with t2:
                st.markdown(f"**{p2}**")
                st.dataframe(adv2, use_container_width=True)

            # ===== AI compare (uses question box + chip ideas) =====
            with st.expander("🧠 Ask the AI Assistant about these players"):
                if model:
                    q2 = st.text_input("Ask something about these players:", key="ai_compare_question")
                    if q2:
                        sum1 = generate_player_summary(p1, raw1_pg if not raw1_pg.empty else adv1, adv1)
                        sum2 = generate_player_summary(p2, raw2_pg if not raw2_pg.empty else adv2, adv2)
                        prompt2 = (
                            "You are an expert NBA analyst. Compare the two players strictly using the provided season tables. "
                            "Lean on TS%, eFG%, PPS, 3PAr, FTr, USG% (true), AST%, TRB% and per-36 trends. "
                            "If data is missing for a metric, acknowledge it and use available proxies.\n\n"
                            f"Player 1: {p1}\n{sum1}\n\n"
                            f"Player 2: {p2}\n{sum2}\n\n"
                            f"Question: {q2}"
                        )
                        with st.spinner("Analyzing…"):
                            resp2 = model.generate_content(prompt2, generation_config={"max_output_tokens": 2048, "temperature": 0.6})
                            st.markdown("### 🧠 AI Analysis")
                            st.write(resp2.text if hasattr(resp2, "text") else "No response.")
                else:
                    st.info("Add your Gemini API key to enable AI analysis.")

else:
    st.info("Use the sidebar to search for a player.")
