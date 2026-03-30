# nba_app/metrics.py
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from fetch import get_team_totals_many, get_balldontlie_league_season_averages

# ---------------------------
# Numeric helpers / formatting
# ---------------------------
def _to_num(x, nd=None):
    try:
        v = pd.to_numeric(x)
        return float(v)
    except Exception:
        return nd

def _fmt_num(x, ndashes=False, digits=1):
    x = _to_num(x)
    if x is None or pd.isna(x):
        return "—" if ndashes else ""
    return f"{round(x, digits):.{digits}f}"

def _fmt_pct(x, ndashes=False, digits=2, already_pct=True):
    """Format percentage-like values.
    If already_pct=True, we assume x is 0–100; otherwise we scale by 100 first.
    """
    x = _to_num(x)
    if x is None or pd.isna(x):
        return "—" if ndashes else ""
    val = x if already_pct else (x * 100.0)
    return f"{round(val, digits):.{digits}f}%"

def _safe_div(num, den):
    num = _to_num(num, 0.0)
    den = _to_num(den, 0.0)
    return (num / den) if den and den != 0 else np.nan

def _ensure_season_start(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and "SEASON_ID" in df.columns:
        if "SEASON_START" not in df.columns:
            df = df.copy()
            df["SEASON_START"] = df["SEASON_ID"].astype(str).str[:4].astype(int)
    return df


def _fill_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    for src, dst in [('FG_PCT', 'FG%'), ('FG3_PCT', '3P%'), ('FT_PCT', 'FT%')]:
        if src in out.columns and dst not in out.columns:
            out[dst] = pd.to_numeric(out[src], errors='coerce') * 100

    fga = pd.to_numeric(out['FGA'], errors='coerce') if 'FGA' in out.columns else pd.Series(np.nan, index=out.index)
    fta = pd.to_numeric(out['FTA'], errors='coerce') if 'FTA' in out.columns else pd.Series(np.nan, index=out.index)
    pts = pd.to_numeric(out['PTS'], errors='coerce') if 'PTS' in out.columns else pd.Series(np.nan, index=out.index)
    fg3a = pd.to_numeric(out['FG3A'], errors='coerce') if 'FG3A' in out.columns else pd.Series(np.nan, index=out.index)
    mins = pd.to_numeric(out['MIN'], errors='coerce') if 'MIN' in out.columns else pd.Series(np.nan, index=out.index)

    if 'PPS' not in out.columns:
        out['PPS'] = np.where(fga > 0, pts / fga, np.nan)
    if '3PAr' not in out.columns:
        out['3PAr'] = np.where(fga > 0, fg3a / fga, np.nan)
    if 'FTr' not in out.columns:
        out['FTr'] = np.where(fga > 0, fta / fga, np.nan)

    for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
        per36_col = f'{stat}/36'
        if per36_col in out.columns or stat not in out.columns:
            continue
        stat_vals = pd.to_numeric(out[stat], errors='coerce')
        out[per36_col] = np.where(mins > 0, stat_vals / mins * 36.0, np.nan)

    return out


# ---------------------------
# Advanced metrics
# ---------------------------
def compute_full_advanced_stats(player_df_totals: pd.DataFrame) -> pd.DataFrame:
    if player_df_totals is None or player_df_totals.empty:
        return player_df_totals
    out = player_df_totals.copy()

    # Shooting splits (percent form 0–100)
    if 'FG_PCT' in out.columns:  out['FG%']  = out['FG_PCT']*100
    if 'FG3_PCT' in out.columns: out['3P%'] = out['FG3_PCT']*100
    if 'FT_PCT' in out.columns:  out['FT%']  = out['FT_PCT']*100

    # TS / eFG / shots profile
    denom_ts = (out.get('FGA',0) + 0.44*out.get('FTA',0))
    out['TS%']  = np.where(denom_ts>0, out.get('PTS',0)/(2*denom_ts)*100, np.nan)
    out['EFG%'] = np.where(out.get('FGA',0)>0, (out.get('FGM',0)+0.5*out.get('FG3M',0))/out.get('FGA',0)*100, np.nan)
    out['PPS']  = np.where(out.get('FGA',0)>0, out.get('PTS',0)/out.get('FGA',0), np.nan)
    out['3PAr'] = np.where(out.get('FGA',0)>0, out.get('FG3A',0)/out.get('FGA',0), np.nan)
    out['FTr']  = np.where(out.get('FGA',0)>0, out.get('FTA',0)/out.get('FGA',0), np.nan)
    out['AST/TO'] = np.where(out.get('TOV',0)>0, out.get('AST',0)/out.get('TOV',0), np.nan)

    # Per-36 suite
    mins = out.get('MIN',0)
    def per36(col): return np.where(mins>0, out.get(col,0)/mins*36, np.nan)
    for stat in ['PTS','REB','AST','STL','BLK','TOV','FGM','FGA','FG3M','OREB','DREB']:
        out[f'{stat}/36'] = per36(stat)

    # Team context merge (for rate stats)
    seasons_needed = out['SEASON_ID'].dropna().unique().tolist()
    team_totals = get_team_totals_many(seasons_needed)
    out = out.merge(team_totals.add_prefix('TEAM_'),
                    left_on=['SEASON_ID','TEAM_ID'],
                    right_on=['TEAM_SEASON_ID','TEAM_TEAM_ID'],
                    how='left')

    # Team and opponent context
    tMIN = out.get('TEAM_MIN',0).astype(float)
    tFGA = out.get('TEAM_FGA',0).astype(float)
    tFTA = out.get('TEAM_FTA',0).astype(float)
    tTOV = out.get('TEAM_TOV',0).astype(float)
    tFGM = out.get('TEAM_FGM',0).astype(float)
    tTRB = np.where(out.get('TEAM_TRB',np.nan).isna(), out.get('TEAM_REB',0), out.get('TEAM_TRB',0)).astype(float)
    tOREB= out.get('TEAM_OREB',0).astype(float)
    tDREB= out.get('TEAM_DREB',0).astype(float)
    oTRB = out.get('TEAM_OPP_REB',0).astype(float)
    oOREB= out.get('TEAM_OPP_OREB',0).astype(float)
    oDREB= out.get('TEAM_OPP_DREB',0).astype(float)
    mp   = out.get('MIN',0).astype(float)

    # Usage (true)
    denom_usg = (tFGA + 0.44*tFTA + tTOV)
    num_usg   = (out.get('FGA',0) + 0.44*out.get('FTA',0) + out.get('TOV',0)) * np.where(tMIN>0, tMIN/5.0, 0)
    out['USG% (true)'] = np.where((denom_usg>0) & (mp>0), 100.0 * num_usg / (mp*denom_usg), np.nan)

    # AST%
    denom_ast = (np.where(tMIN>0, mp/tMIN, 0) * tFGM) - out.get('FGM',0)
    out['AST%'] = np.where(denom_ast != 0, 100.0 * out.get('AST',0)/denom_ast, np.nan)

    # Rebound % (OREB/DRB/TRB)
    denom_orb = tOREB + np.where(oDREB>0, oDREB, 0.0)
    denom_drb = tDREB + np.where(oOREB>0, oOREB, 0.0)
    denom_trb = tTRB  + np.where(oTRB >0, oTRB,  0.0)
    scale = np.where((mp>0) & (tMIN>0), (tMIN/5.0)/mp, np.nan)

    out['ORB%'] = np.where(denom_orb>0, 100.0 * out.get('OREB',0)*scale / denom_orb, np.nan)
    out['DRB%'] = np.where(denom_drb>0, 100.0 * out.get('DREB',0)*scale / denom_drb, np.nan)
    out['TRB%'] = np.where(denom_trb>0, 100.0 * out.get('REB',0) *scale / denom_trb, np.nan)

    # Cleanup
    out = out.drop(columns=[c for c in ['FG_PCT','FG3_PCT','FT_PCT','TEAM_SEASON_ID','TEAM_TEAM_ID'] if c in out.columns])
    float_cols = out.select_dtypes(include=['float','float64','float32']).columns
    out[float_cols] = out[float_cols].round(2)
    out = out[[c for c in out.columns if not c.startswith("TEAM_")]]
    out = _ensure_season_start(out)
    return out


# ---------------------------
# Per-game merge
# ---------------------------
def add_per_game_columns(adv_df: pd.DataFrame, per_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge per-game columns (PPG, RPG, APG, TPG, SPG, BPG, MPG, GP) into the advanced table.
    Also computes 3PA/G and 2PA/G (from FG3A and FGA, which are per-game in PerGame endpoint).
    Works for both team rows and TOT rows. Uses whatever join keys exist in both frames.
    """
    if adv_df is None or adv_df.empty or per_game_df is None or per_game_df.empty:
        return adv_df

    # --- 1) Columns we want from the PerGame frame (define BEFORE use!)
    keep_pg = [
        'SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP',
        'PTS', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'MIN',
        'FG3A', 'FGA',   # needed to derive 3PA/G and 2PA/G
    ]

    # Subset the per-game dataframe
    pg = per_game_df[[c for c in keep_pg if c in per_game_df.columns]].copy()

    # --- 2) Rename per-game fields to friendly labels
    rename_map = {
        'PTS': 'PPG',
        'REB': 'RPG',
        'AST': 'APG',
        'TOV': 'TPG',
        'STL': 'SPG',
        'BLK': 'BPG',
        'MIN': 'MPG',
    }
    for src, dst in rename_map.items():
        if src in pg.columns and dst not in pg.columns:
            pg[dst] = pd.to_numeric(pg[src], errors='coerce')

    # --- 3) Create 3PA/G and 2PA/G (PerGame endpoint already provides per-game FGA/FG3A)
    if 'FG3A' in pg.columns:
        pg['3PA/G'] = pd.to_numeric(pg['FG3A'], errors='coerce')
    if 'FGA' in pg.columns and 'FG3A' in pg.columns:
        pg['2PA/G'] = pd.to_numeric(pg['FGA'], errors='coerce') - pd.to_numeric(pg['FG3A'], errors='coerce')

    # --- 4) Build join keys that exist in both dataframes (prefer SEASON_ID + TEAM_ID)
    join_keys = []
    for k in ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']:
        if k in adv_df.columns and k in pg.columns:
            join_keys.append(k)

    if 'SEASON_ID' not in join_keys:
        # Can't safely merge without season; just return adv as-is
        return adv_df

    merged = adv_df.merge(pg, on=join_keys, how='left')

    # --- 5) Round per-game columns nicely
    for c in ['PPG', 'RPG', 'APG', 'TPG', 'SPG', 'BPG', 'MPG', '3PA/G', '2PA/G']:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').round(1)

    if 'GP' in merged.columns:
        merged['GP'] = pd.to_numeric(merged['GP'], errors='coerce').fillna(0).astype(int)

    merged = _ensure_season_start(merged)
    return merged


# ---------------------------
# AI summary (robust)
# ---------------------------
def generate_player_summary(player_name: str, per_game_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    """
    Build a rich, season-by-season summary for the AI.

    EXPECTS per_game_df from nba_api with per_mode='PerGame' (PTS/REB/AST/MIN are per-game).
    - We first rename PerGame columns to PPG/RPG/APG/MPG/etc.
    - We only compute per-game from totals if those renamed fields are missing.
    - We left-join advanced metrics by SEASON_ID (+ team key when available), with a
      fallback join on SEASON_ID only so we never lose seasons.
    """
    if per_game_df is None or per_game_df.empty:
        return f"No available stats for {player_name}."

    # Start from a copy and RENAME per-game fields right away (critical to avoid dividing by GP again)
    pg = _fill_derived_metrics(per_game_df.copy())

    rename_map = {
        'PTS': 'PPG',
        'REB': 'RPG',
        'AST': 'APG',
        'TOV': 'TPG',
        'STL': 'SPG',
        'BLK': 'BPG',
        'MIN': 'MPG',  # PerGame MIN is already minutes per game
    }
    for src, dst in rename_map.items():
        if src in pg.columns and dst not in pg.columns:
            pg[dst] = pd.to_numeric(pg[src], errors='coerce')

    # Ensure expected identity columns exist
    for col in ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP']:
        if col not in pg.columns:
            pg[col] = np.nan

    # Fallback ONLY IF per-game fields are missing (handles the rare case you pass a Totals frame)
    gp_series = pd.to_numeric(pg['GP'], errors='coerce')
    def _fill_pg_from_totals(total_col, out_col):
        if out_col in pg.columns and pg[out_col].notna().any():
            return  # already have per-game values — don't divide again
        if total_col in per_game_df.columns:
            with np.errstate(invalid='ignore', divide='ignore'):
                pg[out_col] = np.where((gp_series > 0) & pd.notna(per_game_df[total_col]),
                                       pd.to_numeric(per_game_df[total_col], errors='coerce')/gp_series,
                                       np.nan)

    _fill_pg_from_totals('PTS', 'PPG')
    _fill_pg_from_totals('REB', 'RPG')
    _fill_pg_from_totals('AST', 'APG')
    _fill_pg_from_totals('TOV', 'TPG')
    _fill_pg_from_totals('STL', 'SPG')
    _fill_pg_from_totals('BLK', 'BPG')
    _fill_pg_from_totals('MIN', 'MPG')

    # Round the per-game display fields nicely
    for c in ['PPG','RPG','APG','SPG','BPG','TPG','MPG']:
        if c in pg.columns:
            pg[c] = pd.to_numeric(pg[c], errors='coerce').round(1)

    # Merge advanced metrics
    adv = _fill_derived_metrics(adv_df.copy()) if adv_df is not None else pd.DataFrame()
    if adv is None or adv.empty:
        merged = pg.copy()
    else:
        keys = ['SEASON_ID']
        if 'TEAM_ID' in pg.columns and 'TEAM_ID' in adv.columns:
            keys.append('TEAM_ID')
        elif 'TEAM_ABBREVIATION' in pg.columns and 'TEAM_ABBREVIATION' in adv.columns:
            keys.append('TEAM_ABBREVIATION')

        merged = pg.merge(adv, on=keys, how='left', suffixes=('', '_adv'))

        # If most advanced fields are missing due to team-key mismatch, fallback to SEASON_ID only
        if merged.isna().all(axis=1).mean() > 0.5:
            merged = pg.merge(adv, on=['SEASON_ID'], how='left', suffixes=('', '_adv'))

    # Order by season
    def _ensure_season_start(df):
        if df is not None and not df.empty and "SEASON_ID" in df.columns:
            if "SEASON_START" not in df.columns:
                df = df.copy()
                df["SEASON_START"] = df["SEASON_ID"].astype(str).str[:4].astype(int)
        return df

    merged = _ensure_season_start(merged)
    if 'SEASON_START' in merged.columns:
        merged = merged.sort_values(['SEASON_START', 'TEAM_ABBREVIATION'], kind='mergesort')
    else:
        merged = merged.sort_values('SEASON_ID', kind='mergesort')

    # Small helpers
    def _to_num(x):
        try:
            return float(pd.to_numeric(x))
        except Exception:
            return np.nan

    def _fmt_num(x, digits=1):
        x = _to_num(x)
        return (f"{round(x, digits):.{digits}f}" if pd.notna(x) else "—")

    def _fmt_pct(x, digits=2, already_pct=True):
        x = _to_num(x)
        if pd.isna(x):
            return "—"
        return f"{round(x if already_pct else x*100.0, digits):.{digits}f}%"

    # Build lines
    lines = [f"📊 Full season-by-season stats for **{player_name}**:\n"]
    for _, s in merged.iterrows():
        season = s.get('SEASON_ID', 'Unknown')
        team   = s.get('TEAM_ABBREVIATION', 'UNK')

        ppg = _fmt_num(s.get('PPG'), 1)
        rpg = _fmt_num(s.get('RPG'), 1)
        apg = _fmt_num(s.get('APG'), 1)
        spg = _fmt_num(s.get('SPG'), 1)
        bpg = _fmt_num(s.get('BPG'), 1)
        tpg = _fmt_num(s.get('TPG'), 1)
        mpg = _fmt_num(s.get('MPG'), 1)
        gp  = s.get('GP'); gp = int(gp) if pd.notna(gp) else "—"

        ts   = _fmt_pct(s.get('TS%'), 2, already_pct=True)
        efg  = _fmt_pct(s.get('EFG%'), 2, already_pct=True)
        pps  = _fmt_num(s.get('PPS'), 2)
        tpar = _fmt_num(s.get('3PAr'), 2)
        ftr  = _fmt_num(s.get('FTr'), 2)
        usg  = _fmt_pct(s.get('USG% (true)'), 2, already_pct=True)
        astp = _fmt_pct(s.get('AST%'), 2, already_pct=True)
        trbp = _fmt_pct(s.get('TRB%'), 2, already_pct=True)
        orbp = _fmt_pct(s.get('ORB%'), 2, already_pct=True)
        drbp = _fmt_pct(s.get('DRB%'), 2, already_pct=True)
        ast_to = _fmt_num(s.get('AST/TO'), 2)

        fg   = _fmt_pct(s.get('FG%'), 2, already_pct=True)
        tp   = _fmt_pct(s.get('3P%'), 2, already_pct=True)
        ft   = _fmt_pct(s.get('FT%'), 2, already_pct=True)

        pts36 = _fmt_num(s.get('PTS/36'), 2)
        reb36 = _fmt_num(s.get('REB/36'), 2)
        ast36 = _fmt_num(s.get('AST/36'), 2)
        stl36 = _fmt_num(s.get('STL/36'), 2)
        blk36 = _fmt_num(s.get('BLK/36'), 2)
        tov36 = _fmt_num(s.get('TOV/36'), 2)

        lines += [
            "---",
            f"### Season {season} ({team})",
            f"- **PPG:** {ppg}, **RPG:** {rpg}, **APG:** {apg}",
            f"- **SPG:** {spg}, **BPG:** {bpg}, **TPG:** {tpg}",
            f"- **Games Played:** {gp}, **Minutes/Game:** {mpg}",
            f"- **FG% / 3P% / FT%:** {fg} / {tp} / {ft}",
            f"- **TS%:** {ts}, **eFG%:** {efg}, **PPS:** {pps}",
            f"- **3PAr:** {tpar}, **FTr:** {ftr}, **AST/TO:** {ast_to}",
            f"- **USG% (true):** {usg}, **AST%:** {astp}, **TRB%:** {trbp}, **ORB%:** {orbp}, **DRB%:** {drbp}",
            f"- **Per-36:** PTS {pts36}, REB {reb36}, AST {ast36}, STL {stl36}, BLK {blk36}, TOV {tov36}",
        ]

    return "\n".join(lines)


def build_ai_stat_packet(player_name: str, per_game_df: pd.DataFrame, adv_df: pd.DataFrame) -> dict:
    """
    Build a compact structured payload for the LLM.
    Uses the latest season plus a short all-seasons table when available.
    Omits keys whose values are missing instead of sending placeholders.
    """
    pg = _fill_derived_metrics(per_game_df.copy()) if per_game_df is not None else pd.DataFrame()
    adv = _fill_derived_metrics(adv_df.copy()) if adv_df is not None else pd.DataFrame()

    if (pg is None or pg.empty) and (adv is None or adv.empty):
        return {
            "player_name": player_name,
            "latest_season": {},
            "season_rows": [],
            "available_metric_count": 0,
        }
    if pg is None or pg.empty:
        pg = adv.copy()

    rename_map = {
        'PTS': 'PPG',
        'REB': 'RPG',
        'AST': 'APG',
        'TOV': 'TPG',
        'STL': 'SPG',
        'BLK': 'BPG',
        'MIN': 'MPG',
    }
    for src, dst in rename_map.items():
        if src in pg.columns and dst not in pg.columns:
            pg[dst] = pd.to_numeric(pg[src], errors='coerce')

    keys = ['SEASON_ID']
    if 'TEAM_ID' in pg.columns and 'TEAM_ID' in adv.columns:
        keys.append('TEAM_ID')
    elif 'TEAM_ABBREVIATION' in pg.columns and 'TEAM_ABBREVIATION' in adv.columns:
        keys.append('TEAM_ABBREVIATION')

    merged = pg.merge(adv, on=keys, how='left', suffixes=('', '_adv')) if not adv.empty else pg
    merged = _ensure_season_start(merged)
    if 'SEASON_START' in merged.columns:
        merged = merged.sort_values(['SEASON_START', 'TEAM_ABBREVIATION'], kind='mergesort')

    preferred_fields = [
        'SEASON_ID', 'TEAM_ABBREVIATION', 'GP',
        'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'MPG',
        'FG%', '3P%', 'FT%',
        'TS%', 'EFG%', 'PPS', '3PAr', 'FTr',
        'USG% (true)', 'AST%', 'TRB%', 'ORB%', 'DRB%', 'AST/TO',
        'PTS/36', 'REB/36', 'AST/36', 'STL/36', 'BLK/36', 'TOV/36',
    ]

    def clean_value(v):
        num = _to_num(v)
        if num is None or pd.isna(num):
            if isinstance(v, str) and v and v != "—":
                return v
            return None
        return round(num, 2)

    season_rows = []
    for _, row in merged.iterrows():
        item = {}
        for field in preferred_fields:
            if field not in merged.columns:
                continue
            cleaned = clean_value(row.get(field))
            if cleaned is None:
                continue
            item[field] = cleaned
        if item:
            season_rows.append(item)

    latest = season_rows[-1] if season_rows else {}
    return {
        "player_name": player_name,
        "latest_season": latest,
        "season_rows": season_rows,
        "available_metric_count": len(latest),
    }


# ---------------------------
# Context for chips and tables
# ---------------------------
def compact_player_context(df: pd.DataFrame) -> dict:
    """
    Build a tiny, per-game-oriented summary for the latest season row.
    Prefers PPG/RPG/APG/... columns; falls back to totals/GP if needed.
    """
    if df is None or df.empty:
        return {}

    df = _ensure_season_start(df)
    df = df.sort_values('SEASON_START') if 'SEASON_START' in df.columns else df
    row = df.iloc[-1]

    def _num(val, default=np.nan):
        try:
            v = float(val)
            return v
        except Exception:
            return default

    def _get(name: str, default=np.nan):
        return _num(row[name], default) if name in row and pd.notna(row[name]) else default

    gp = _get("GP", np.nan)

    # Helper to get per-game with fallback from totals
    def _per_game(primary_pg_col: str, total_col: str):
        if primary_pg_col in row and pd.notna(row[primary_pg_col]):
            return _num(row[primary_pg_col], np.nan)
        tot = _get(total_col, np.nan)
        if pd.notna(tot) and pd.notna(gp) and gp > 0:
            return tot / gp
        return np.nan

    # Per-game metrics (prefer explicit PPG/RPG/APG etc.)
    ppg = _per_game("PPG", "PTS")
    rpg = _per_game("RPG", "REB")
    apg = _per_game("APG", "AST")
    spg = _per_game("SPG", "STL")
    bpg = _per_game("BPG", "BLK")
    tpg = _per_game("TPG", "TOV")  # turnovers per game

    # Efficiency / rates (already per possession/attempt)
    ts  = _get("TS%")
    efg = _get("EFG%")
    usg = _get("USG% (true)")
    ast_pct = _get("AST%")
    trb_pct = _get("TRB%")

    # Light sanity clamps to avoid garbage getting into prompts
    def _clamp(x, lo, hi):
        return x if (pd.notna(x) and lo <= x <= hi) else np.nan

    ppg = _clamp(ppg, 0, 60)
    rpg = _clamp(rpg, 0, 25)
    apg = _clamp(apg, 0, 20)
    spg = _clamp(spg, 0, 6)
    bpg = _clamp(bpg, 0, 6)
    tpg = _clamp(tpg, 0, 8)

    return {
        "season": row.get("SEASON_ID", "Unknown"),
        "team": row.get("TEAM_ABBREVIATION", "UNK"),
        "gp": gp,
        "ppg": ppg,
        "rpg": rpg,
        "apg": apg,
        "spg": spg,
        "bpg": bpg,
        "tpg": tpg,
        "ts": ts,
        "efg": efg,
        "usg": usg,
        "ast_pct": ast_pct,
        "trb_pct": trb_pct,
        "mpg": _per_game("MPG", "MIN"),
    }


_DISPLAY_PRIORITY = [
    # identity/context
    "SEASON_ID", "TEAM_ABBREVIATION",
    # per-game
    "GP", "MPG", "PPG", "RPG", "APG", "SPG", "BPG", "TPG",
    "3PA/G", "2PA/G",
    # shooting splits (if present)
    "FG%", "3P%", "FT%",
    # core advanced (summary)
    "TS%", "EFG%", "PPS", "3PAr", "FTr", "USG% (true)", "AST%", "TRB%", "ORB%", "DRB%", "AST/TO",
    # per-36
    "PTS/36", "REB/36", "AST/36", "STL/36", "BLK/36", "TOV/36", "FGM/36", "FGA/36", "FG3M/36", "OREB/36", "DREB/36",
    
]

# Hide these IDs & internal columns from tables
_HIDDEN_EXACT = {"PLAYER_ID", "TEAM_ID", "LEAGUE_ID", "SEASON_START", "CFID", "CFPARAMS"}
_HIDDEN_PREFIXES = ("TEAM_",)  # e.g., merged team context columns

def order_columns_for_display(df):
    """Return a reordered list of columns with high-priority fields first."""
    if df is None or df.empty:
        return []
    priority = [c for c in _DISPLAY_PRIORITY if c in df.columns]
    rest = [c for c in df.columns if c not in priority]
    return priority + rest

def metric_public_cols(df):
    """Filter out internal IDs/team_* and return the display-ordered list."""
    if df is None or df.empty:
        return []
    visible = [c for c in df.columns
               if c not in _HIDDEN_EXACT and not any(c.startswith(pfx) for pfx in _HIDDEN_PREFIXES)]
    # keep original df but show in ordered, filtered column set
    ordered = [c for c in order_columns_for_display(df) if c in visible]
    # make sure we don't drop any remaining visible columns that weren't in the priority list
    tail = [c for c in visible if c not in ordered]
    return ordered + tail

# ---------------------------
# League shooting baselines (per season)
# ---------------------------
def compute_league_shooting_table(seasons: list[str]) -> pd.DataFrame:
    """
    Compute weighted league shooting metrics for each season in `seasons`.
    Returns columns: SEASON_ID, SEASON_START, FG%, 3P%, FT%, TS%, EFG% (all 0–100 scale).
    """
    import numpy as np
    import pandas as pd

    if not seasons:
        return pd.DataFrame()

    league = get_team_totals_many(seasons)
    if league is None or league.empty:
        return pd.DataFrame()

    grp = league.groupby("SEASON_ID", as_index=False).agg({
        "FGM": "sum", "FGA": "sum",
        "FG3M": "sum", "FG3A": "sum",
        "FTM": "sum", "FTA": "sum",
        "PTS": "sum",
    })
    grp["SEASON_START"] = grp["SEASON_ID"].astype(str).str[:4].astype(int)

    # Percentages (0–100)
    grp["FG%"]  = np.where(grp["FGA"]>0, 100.0*grp["FGM"]/grp["FGA"], np.nan)
    grp["3P%"]  = np.where(grp["FG3A"]>0, 100.0*grp["FG3M"]/grp["FG3A"], np.nan)
    grp["FT%"]  = np.where(grp["FTA"]>0, 100.0*grp["FTM"]/grp["FTA"], np.nan)
    grp["TS%"]  = np.where((grp["FGA"]+0.44*grp["FTA"])>0,
                           100.0*grp["PTS"]/(2.0*(grp["FGA"]+0.44*grp["FTA"])),
                           np.nan)
    grp["EFG%"] = np.where(grp["FGA"]>0,
                           100.0*(grp["FGM"] + 0.5*grp["FG3M"])/grp["FGA"],
                           np.nan)

    return grp[["SEASON_ID","SEASON_START","FG%","3P%","FT%","TS%","EFG%"]].copy()

def build_ai_phase_table(adv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a compact season table for the LLM.
    Must include SEASON_ID and key metrics.
    """
    if adv_df is None or adv_df.empty:
        return pd.DataFrame()

    # Import lazily here to avoid a module-level circular import.
    from ui_compare import _make_readable_stats_table

    # Make it readable first (your existing function)
    nice, _, _ = _make_readable_stats_table(adv_df)

    keep = [c for c in [
        "Season", "Team",
        "MIN/G", "PTS/G", "REB/G", "AST/G", "STL/G", "BLK/G", "TOV/G",
        "TS%", "eFG%", "USG%", "AST%", "TRB%", "ORB%", "DRB%", "AST/TO",
        "PTS/36", "REB/36", "AST/36", "STL/36", "BLK/36", "TOV/36"
    ] if c in nice.columns]

    out = nice[keep].copy()

    # Ensure Season is not blank
    if "Season" in out.columns:
        out["Season"] = out["Season"].astype(str).replace({"—": "", "nan": "", "None": ""})

    return out


def compute_player_percentile_context(player_name: str, season_id: str, adv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact ranking / percentile table for the player's latest season.
    Uses league-wide balldontlie season averages and existing rank fields when available.
    """
    if adv_df is None or adv_df.empty or not season_id:
        return pd.DataFrame()

    try:
        season_start = int(str(season_id)[:4])
    except Exception:
        return pd.DataFrame()

    league_df = get_balldontlie_league_season_averages(season_start)
    if league_df is None or league_df.empty:
        return pd.DataFrame()

    player_rows = league_df[league_df["PLAYER_NAME"].astype(str).str.lower() == str(player_name).lower()].copy()
    if player_rows.empty:
        return pd.DataFrame()

    player_row = player_rows.iloc[0]
    total_players = int(league_df["PLAYER_ID"].nunique()) if "PLAYER_ID" in league_df.columns else int(len(league_df))

    latest = adv_df.iloc[-1]
    gp = pd.to_numeric(latest.get("GP"), errors="coerce")

    def _player_display_value(label: str):
        if label in latest.index and pd.notna(latest.get(label)):
            return latest.get(label)
        if label == "PPG":
            pts = pd.to_numeric(latest.get("PTS"), errors="coerce")
            return (pts / gp) if pd.notna(pts) and pd.notna(gp) and gp > 0 else np.nan
        if label == "RPG":
            reb = pd.to_numeric(latest.get("REB"), errors="coerce")
            return (reb / gp) if pd.notna(reb) and pd.notna(gp) and gp > 0 else np.nan
        if label == "APG":
            ast = pd.to_numeric(latest.get("AST"), errors="coerce")
            return (ast / gp) if pd.notna(ast) and pd.notna(gp) and gp > 0 else np.nan
        if label == "BLK/G":
            blk = pd.to_numeric(latest.get("BLK"), errors="coerce")
            return (blk / gp) if pd.notna(blk) and pd.notna(gp) and gp > 0 else np.nan
        if label == "STL/G":
            stl = pd.to_numeric(latest.get("STL"), errors="coerce")
            return (stl / gp) if pd.notna(stl) and pd.notna(gp) and gp > 0 else np.nan
        return np.nan

    metrics = [
        ("PPG", "pts", "pts_rank", True, "Scoring"),
        ("RPG", "reb", "reb_rank", True, "Rebounding"),
        ("APG", "ast", "ast_rank", True, "Playmaking"),
        ("TS%", "ts_pct", "ts_pct_rank", False, "Efficiency"),
        ("eFG%", "efg_pct", "efg_pct_rank", False, "Efficiency"),
        ("USG%", "usg_pct", "usg_pct_rank", False, "Role"),
        ("AST%", "ast_pct", "ast_pct_rank", False, "Playmaking"),
        ("TRB%", "reb_pct", "reb_pct_rank", False, "Rebounding"),
        ("ORB%", "oreb_pct", "oreb_pct_rank", False, "Rebounding"),
        ("DRB%", "dreb_pct", "dreb_pct_rank", False, "Rebounding"),
        ("AST/TO", "ast_to", "ast_to_rank", True, "Playmaking"),
        ("3P%", "fg3_pct", "fg3_pct_rank", False, "Shooting"),
        ("FT%", "ft_pct", "ft_pct_rank", False, "Shooting"),
        ("BLK/G", "blk", "blk_rank", True, "Defense"),
        ("STL/G", "stl", "stl_rank", True, "Defense"),
    ]

    rows = []
    for label, league_col, rank_col, use_adv_value_directly, group in metrics:
        if label not in latest.index and league_col not in player_row.index:
            continue

        value = _player_display_value(label)
        if pd.isna(value) and league_col in player_row.index:
            value = player_row.get(league_col)
            if pd.notna(value) and not use_adv_value_directly and label.endswith("%"):
                value = float(value) * 100.0

        rank_val = pd.to_numeric(player_row.get(rank_col), errors="coerce") if rank_col in player_row.index else np.nan
        if pd.isna(rank_val) or not total_players:
            continue

        rank_int = int(rank_val)
        percentile = round(((total_players - rank_int) / max(total_players - 1, 1)) * 100.0, 1)
        rows.append({
            "Metric": label,
            "Value": value,
            "Rank": rank_int,
            "Percentile": percentile,
            "Of": total_players,
            "Category": group,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    sort_order = {
        "Scoring": 0,
        "Efficiency": 1,
        "Playmaking": 2,
        "Rebounding": 3,
        "Defense": 4,
        "Role": 5,
        "Shooting": 6,
    }
    out["__order"] = out["Category"].map(sort_order).fillna(99)
    out = out.sort_values(["__order", "Percentile"], ascending=[True, False]).drop(columns="__order")
    return out.reset_index(drop=True)


def detect_player_archetype(player_name: str, adv_df: pd.DataFrame, percentile_df: pd.DataFrame | None = None) -> dict:
    """
    Rules-based archetype detection using latest-season production and percentile context.
    Returns a primary archetype, optional secondary archetype, confidence, and stat evidence.
    """
    if adv_df is None or adv_df.empty:
        return {}

    latest = adv_df.iloc[-1]
    percentile_df = percentile_df if isinstance(percentile_df, pd.DataFrame) else pd.DataFrame()

    def num(key, default=np.nan):
        return pd.to_numeric(latest.get(key), errors="coerce") if key in latest.index else default

    def pct(metric: str, default=50.0):
        if percentile_df.empty or "Metric" not in percentile_df.columns:
            return default
        rows = percentile_df[percentile_df["Metric"] == metric]
        if rows.empty:
            return default
        return pd.to_numeric(rows.iloc[0]["Percentile"], errors="coerce")

    position = str(latest.get("POSITION", latest.get("Pos", "")) or "").upper()
    gp = num("GP")

    def per_game(preferred: str, total: str):
        preferred_val = num(preferred)
        if pd.notna(preferred_val):
            return preferred_val
        total_val = num(total)
        if pd.notna(total_val) and pd.notna(gp) and gp > 0:
            return total_val / gp
        return np.nan

    ppg = per_game("PPG", "PTS")
    rpg = per_game("RPG", "REB")
    apg = per_game("APG", "AST")
    spg = per_game("SPG", "STL")
    bpg = per_game("BPG", "BLK")
    ts = num("TS%")
    efg = num("EFG%")
    usg = num("USG% (true)")
    ast_pct = num("AST%")
    trb_pct = num("TRB%")
    orb_pct = num("ORB%")
    drb_pct = num("DRB%")
    three_pa = num("3PA/G")
    three_pct = num("3P%")
    ft_rate = num("FTr")

    archetypes = {
        "Heliocentric Creator": {"score": 0, "evidence": []},
        "Point Center / Offensive Hub": {"score": 0, "evidence": []},
        "Unicorn Big": {"score": 0, "evidence": []},
        "Stretch Big": {"score": 0, "evidence": []},
        "Rim-Protecting Big": {"score": 0, "evidence": []},
        "Two-Way Wing": {"score": 0, "evidence": []},
        "Three-Level Scorer": {"score": 0, "evidence": []},
        "Interior Finisher": {"score": 0, "evidence": []},
        "Secondary Playmaker": {"score": 0, "evidence": []},
        "Shot-Creating Wing": {"score": 0, "evidence": []},
        "3-and-D Wing": {"score": 0, "evidence": []},
        "Combo Guard Scorer": {"score": 0, "evidence": []},
        "Floor-Spacing Big": {"score": 0, "evidence": []},
        "Rim-Running Big": {"score": 0, "evidence": []},
        "Glass-Cleaning Big": {"score": 0, "evidence": []},
    }

    def boost(name: str, amount: float, reason: str):
        archetypes[name]["score"] += amount
        archetypes[name]["evidence"].append(reason)

    if pd.notna(usg) and usg >= 28:
        boost("Heliocentric Creator", 2.0, f"High usage load ({usg:.1f}%).")
        boost("Three-Level Scorer", 1.0, f"Carries real scoring volume ({usg:.1f}% usage).")
        boost("Shot-Creating Wing", 0.8, f"Offense runs through self-created volume ({usg:.1f}% usage).")
        boost("Combo Guard Scorer", 0.8, f"Carries a major scoring burden ({usg:.1f}% usage).")
    if pd.notna(ast_pct) and ast_pct >= 30:
        boost("Heliocentric Creator", 2.5, f"Elite playmaking share ({ast_pct:.1f} AST%).")
        boost("Point Center / Offensive Hub", 2.0, f"Runs offense through creation ({ast_pct:.1f} AST%).")
        boost("Secondary Playmaker", 0.8, f"Strong table-setting profile ({ast_pct:.1f} AST%).")
    if pd.notna(apg) and apg >= 7:
        boost("Heliocentric Creator", 1.8, f"Top-tier passing volume ({apg:.1f} APG).")
        boost("Combo Guard Scorer", 0.8, f"Guard-like creation burden ({apg:.1f} APG).")
    elif pd.notna(apg) and apg >= 4.5:
        boost("Secondary Playmaker", 1.8, f"Meaningful creation role ({apg:.1f} APG).")

    if "C" in position or ("F" in position and pd.notna(rpg) and rpg >= 7):
        boost("Unicorn Big", 0.8, "Frontcourt size/role profile.")
        boost("Point Center / Offensive Hub", 0.8, "Big-man positional profile.")
        boost("Stretch Big", 0.8, "Frontcourt size/role profile.")
        boost("Rim-Protecting Big", 0.8, "Frontcourt defensive role.")
        boost("Interior Finisher", 0.8, "Frontcourt interior role.")
        boost("Floor-Spacing Big", 0.5, "Frontcourt profile with spacing upside.")
        boost("Rim-Running Big", 0.5, "Frontcourt interior finisher profile.")
        boost("Glass-Cleaning Big", 0.5, "Frontcourt rebounding profile.")

    if pd.notna(three_pa) and three_pa >= 4.5 and pd.notna(three_pct) and three_pct >= 34:
        boost("Unicorn Big", 1.6, f"Rare big-man perimeter volume ({three_pa:.1f} 3PA/G, {three_pct:.1f}% 3P).")
        boost("Stretch Big", 2.0, f"Real floor-spacing volume ({three_pa:.1f} 3PA/G, {three_pct:.1f}% 3P).")
        boost("Three-Level Scorer", 1.3, f"Strong perimeter scoring profile ({three_pa:.1f} 3PA/G).")
        boost("Floor-Spacing Big", 2.2, f"Big-man shooting gravity ({three_pa:.1f} 3PA/G, {three_pct:.1f}% 3P).")
        boost("3-and-D Wing", 1.0, f"High-volume catch-and-shoot threat ({three_pa:.1f} 3PA/G).")
        boost("Shot-Creating Wing", 0.8, f"Perimeter volume supports shot-creation role ({three_pa:.1f} 3PA/G).")

    if pd.notna(bpg) and bpg >= 2.0:
        boost("Unicorn Big", 1.8, f"Big-man rim protection plus mobility ({bpg:.1f} BLK/G).")
        boost("Rim-Protecting Big", 2.8, f"High rim protection output ({bpg:.1f} BLK/G).")
        boost("Two-Way Wing", 0.8, f"Impact shot blocking ({bpg:.1f} BLK/G).")
        boost("3-and-D Wing", 0.6, f"Strong defensive event production ({bpg:.1f} BLK/G).")
    elif pd.notna(bpg) and bpg >= 1.0:
        boost("Rim-Protecting Big", 1.4, f"Solid rim protection ({bpg:.1f} BLK/G).")

    if pd.notna(trb_pct) and trb_pct >= 15:
        boost("Rim-Protecting Big", 1.2, f"Strong rebounding share ({trb_pct:.1f} TRB%).")
        boost("Point Center / Offensive Hub", 1.0, f"Controls possessions on the glass ({trb_pct:.1f} TRB%).")
        boost("Glass-Cleaning Big", 2.2, f"Dominates the glass ({trb_pct:.1f} TRB%).")
    if pd.notna(orb_pct) and orb_pct >= 8:
        boost("Interior Finisher", 1.8, f"Creates extra possessions inside ({orb_pct:.1f} ORB%).")
        boost("Rim-Running Big", 1.8, f"Pressure on the rim shows up on the offensive glass ({orb_pct:.1f} ORB%).")
    if pd.notna(ts) and ts >= 60:
        boost("Unicorn Big", 0.9, f"Pairs unusual skill with real efficiency ({ts:.1f} TS%).")
        boost("Interior Finisher", 1.4, f"Finishes efficiently ({ts:.1f} TS%).")
        boost("Three-Level Scorer", 1.2, f"Scoring efficiency is strong ({ts:.1f} TS%).")
        boost("Rim-Running Big", 1.2, f"High-efficiency finishing profile ({ts:.1f} TS%).")
    if pd.notna(ppg) and ppg >= 24:
        boost("Unicorn Big", 1.1, f"Star-level scoring from a big profile ({ppg:.1f} PPG).")
        boost("Three-Level Scorer", 2.0, f"High scoring volume ({ppg:.1f} PPG).")
        boost("Shot-Creating Wing", 1.6, f"Produces star-level wing scoring volume ({ppg:.1f} PPG).")
        boost("Combo Guard Scorer", 1.6, f"Produces star-level guard scoring volume ({ppg:.1f} PPG).")
    elif pd.notna(ppg) and ppg >= 18:
        boost("Three-Level Scorer", 1.0, f"Reliable scoring role ({ppg:.1f} PPG).")

    if pd.notna(spg) and spg >= 1.2 and pd.notna(bpg) and bpg >= 0.8:
        boost("Two-Way Wing", 2.0, f"Stocks profile supports two-way impact ({spg:.1f} STL, {bpg:.1f} BLK).")
        boost("3-and-D Wing", 1.8, f"Defensive playmaking supports 3-and-D role ({spg:.1f} STL, {bpg:.1f} BLK).")
    if pd.notna(three_pa) and three_pa >= 5 and pd.notna(three_pct) and three_pct >= 35:
        boost("Two-Way Wing", 1.0, f"Adds real perimeter spacing ({three_pa:.1f} 3PA/G).")
        boost("3-and-D Wing", 1.8, f"Volume plus accuracy from deep fits 3-and-D mold ({three_pa:.1f} 3PA/G, {three_pct:.1f}% 3P).")

    if pd.notna(ast_pct) and ast_pct >= 20 and pd.notna(usg) and usg < 28:
        boost("Secondary Playmaker", 1.4, f"Creates offense without extreme usage ({ast_pct:.1f} AST%, {usg:.1f} USG%).")
        boost("Shot-Creating Wing", 0.8, f"Can bend the defense and create for others ({ast_pct:.1f} AST%).")
        boost("Combo Guard Scorer", 0.8, f"Can score and create without carrying the full offense ({ast_pct:.1f} AST%).")
    if ("C" in position or "F" in position) and pd.notna(ast_pct) and ast_pct >= 16:
        boost("Unicorn Big", 1.3, f"Unusual playmaking for a frontcourt player ({ast_pct:.1f} AST%).")
    if pd.notna(ft_rate) and ft_rate >= 0.28:
        boost("Interior Finisher", 0.8, f"Gets to the line at a healthy rate ({ft_rate:.2f} FTr).")
        boost("Three-Level Scorer", 0.5, f"Pressure on defenses shows up in foul drawing ({ft_rate:.2f} FTr).")
        boost("Shot-Creating Wing", 0.8, f"Creates contact and rim pressure ({ft_rate:.2f} FTr).")
        boost("Combo Guard Scorer", 0.8, f"Attacks enough to draw fouls ({ft_rate:.2f} FTr).")

    if pct("APG") >= 95 or pct("AST%") >= 95:
        boost("Heliocentric Creator", 1.0, f"League-elite playmaking percentile ({max(pct('APG'), pct('AST%')):.1f}).")
        boost("Point Center / Offensive Hub", 0.8, f"Rare passing for role ({max(pct('APG'), pct('AST%')):.1f} percentile).")
    if ("C" in position or "F" in position) and (pct("AST%") >= 85 or pct("APG") >= 85):
        boost("Unicorn Big", 1.0, f"Rare frontcourt playmaking percentile ({max(pct('APG'), pct('AST%')):.1f}).")
    if pct("RPG") >= 95 or pct("TRB%") >= 95:
        boost("Rim-Protecting Big", 0.8, f"Elite glass work ({max(pct('RPG'), pct('TRB%')):.1f} percentile rebounding).")
        boost("Point Center / Offensive Hub", 0.6, f"Owns the glass ({max(pct('RPG'), pct('TRB%')):.1f} percentile rebounding).")
        boost("Glass-Cleaning Big", 1.2, f"Elite rebounding percentile ({max(pct('RPG'), pct('TRB%')):.1f}).")
    if pct("BLK/G") >= 95:
        boost("Unicorn Big", 1.0, f"Rare shot-blocking percentile for a skilled frontcourt player ({pct('BLK/G'):.1f}).")
        boost("Rim-Protecting Big", 1.0, f"Elite shot-blocking percentile ({pct('BLK/G'):.1f}).")
        boost("3-and-D Wing", 0.5, f"Rare defensive event percentile ({pct('BLK/G'):.1f}).")
    if pct("TS%") >= 90 and pct("PPG") >= 90:
        boost("Unicorn Big", 0.8, "Combines big-man size with star scoring efficiency and volume.")
        boost("Three-Level Scorer", 1.0, f"Combines scoring volume and efficiency at elite percentile levels.")
        boost("Shot-Creating Wing", 0.8, "Looks like a top-end scoring engine by efficiency and volume.")
        boost("Combo Guard Scorer", 0.8, "Looks like a top-end scoring guard by efficiency and volume.")

    if "G" in position and pd.notna(apg) and apg >= 5 and pd.notna(ppg) and ppg >= 18:
        boost("Combo Guard Scorer", 1.6, f"Blends scoring and guard creation ({ppg:.1f} PPG, {apg:.1f} APG).")
    if ("F" in position or "G" in position) and pd.notna(ppg) and ppg >= 20 and pd.notna(three_pa) and three_pa >= 4:
        boost("Shot-Creating Wing", 1.6, f"Wing-sized shot creation with perimeter volume ({ppg:.1f} PPG, {three_pa:.1f} 3PA/G).")
    if ("F" in position or "G" in position) and pd.notna(three_pa) and three_pa >= 5 and pd.notna(spg) and spg >= 1.0:
        boost("3-and-D Wing", 1.2, f"Spacing plus event defense profile ({three_pa:.1f} 3PA/G, {spg:.1f} STL/G).")
    if "C" in position and pd.notna(three_pa) and three_pa >= 3.5:
        boost("Unicorn Big", 1.0, f"Center-sized player with real floor-spacing volume ({three_pa:.1f} 3PA/G).")
        boost("Floor-Spacing Big", 1.6, f"Center who stretches defenses ({three_pa:.1f} 3PA/G).")
    if "C" in position and pd.notna(orb_pct) and orb_pct >= 9 and pd.notna(ts) and ts >= 58:
        boost("Rim-Running Big", 1.8, f"Rim pressure and efficient interior finishing ({orb_pct:.1f} ORB%, {ts:.1f} TS%).")
    if ("C" in position or ("F" in position and pd.notna(rpg) and rpg >= 7)) and pd.notna(three_pa) and three_pa >= 3.5 and pd.notna(bpg) and bpg >= 1.5:
        boost("Unicorn Big", 2.0, f"Blends frontcourt shooting and rim protection in a rare way ({three_pa:.1f} 3PA/G, {bpg:.1f} BLK/G).")

    archetype_descriptions = {
        "Heliocentric Creator": "The offense runs through this player as the main scorer and creator.",
        "Point Center / Offensive Hub": "A big who acts like an offensive engine through passing, touches, and decision-making.",
        "Unicorn Big": "A rare frontcourt player who blends size with perimeter skill, unusual creation, and/or rim protection in a way most bigs cannot.",
        "Stretch Big": "A frontcourt player who adds real three-point volume and scoring gravity.",
        "Floor-Spacing Big": "A big whose main offensive value includes pulling defenders out with shooting.",
        "Rim-Protecting Big": "A big whose defensive identity is built around shot blocking and interior coverage.",
        "Glass-Cleaning Big": "A frontcourt player who wins possessions with elite rebounding.",
        "Rim-Running Big": "A big who pressures the rim, finishes efficiently, and creates value inside.",
        "Two-Way Wing": "A wing who contributes on both ends with scoring utility and defensive activity.",
        "3-and-D Wing": "A wing whose value comes from perimeter shooting plus disruptive defense.",
        "Shot-Creating Wing": "A wing who can generate offense for himself and punish defenses with scoring volume.",
        "Three-Level Scorer": "A player who can put up points efficiently across different areas of the floor.",
        "Interior Finisher": "A player whose scoring comes heavily from pressure at the rim and efficient inside finishing.",
        "Secondary Playmaker": "A player who may not run the whole offense but still creates a meaningful amount for teammates.",
        "Combo Guard Scorer": "A scoring guard who can both create shots and handle a solid share of playmaking.",
    }

    ranked = sorted(
        [{"name": name, **payload} for name, payload in archetypes.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
    primary = ranked[0]
    secondary = ranked[1] if len(ranked) > 1 and ranked[1]["score"] >= 2.5 else None

    confidence = min(0.95, max(0.45, 0.45 + (primary["score"] / 10.0)))
    return {
        "player_name": player_name,
        "primary": primary["name"],
        "secondary": secondary["name"] if secondary else None,
        "primary_description": archetype_descriptions.get(primary["name"], ""),
        "secondary_description": archetype_descriptions.get(secondary["name"], "") if secondary else None,
        "confidence": round(confidence, 2),
        "evidence": primary["evidence"][:4],
        "style_scores": [
            {"Archetype": item["name"], "Score": round(item["score"], 2)}
            for item in ranked if item["score"] > 0
        ],
    }


def find_similar_players(player_name: str, season_id: str, adv_df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """
    Find statistically similar players for the latest season using league-wide balldontlie season averages.
    Returns a compact table with a similarity score and a few anchor metrics.
    """
    if adv_df is None or adv_df.empty or not season_id:
        return pd.DataFrame()

    try:
        season_start = int(str(season_id)[:4])
    except Exception:
        return pd.DataFrame()

    league_df = get_balldontlie_league_season_averages(season_start)
    if league_df is None or league_df.empty:
        return pd.DataFrame()

    latest = adv_df.iloc[-1]
    gp = pd.to_numeric(latest.get("GP"), errors="coerce")

    def latest_per_game(preferred: str, total: str):
        preferred_val = pd.to_numeric(latest.get(preferred), errors="coerce") if preferred in latest.index else np.nan
        if pd.notna(preferred_val):
            return preferred_val
        total_val = pd.to_numeric(latest.get(total), errors="coerce") if total in latest.index else np.nan
        if pd.notna(total_val) and pd.notna(gp) and gp > 0:
            return total_val / gp
        return np.nan

    target = {
        "pts": latest_per_game("PPG", "PTS"),
        "reb": latest_per_game("RPG", "REB"),
        "ast": latest_per_game("APG", "AST"),
        "stl": latest_per_game("SPG", "STL"),
        "blk": latest_per_game("BPG", "BLK"),
        "fg3a": pd.to_numeric(latest.get("3PA/G"), errors="coerce"),
        "fg3_pct": (pd.to_numeric(latest.get("3P%"), errors="coerce") / 100.0) if "3P%" in latest.index else np.nan,
        "ts_pct": (pd.to_numeric(latest.get("TS%"), errors="coerce") / 100.0) if "TS%" in latest.index else np.nan,
        "usg_pct": (pd.to_numeric(latest.get("USG% (true)"), errors="coerce") / 100.0) if "USG% (true)" in latest.index else np.nan,
        "ast_pct": (pd.to_numeric(latest.get("AST%"), errors="coerce") / 100.0) if "AST%" in latest.index else np.nan,
        "reb_pct": (pd.to_numeric(latest.get("TRB%"), errors="coerce") / 100.0) if "TRB%" in latest.index else np.nan,
        "oreb_pct": (pd.to_numeric(latest.get("ORB%"), errors="coerce") / 100.0) if "ORB%" in latest.index else np.nan,
        "dreb_pct": (pd.to_numeric(latest.get("DRB%"), errors="coerce") / 100.0) if "DRB%" in latest.index else np.nan,
        "ast_to": pd.to_numeric(latest.get("AST/TO"), errors="coerce"),
    }

    feature_weights = {
        "pts": 1.2,
        "reb": 1.0,
        "ast": 1.0,
        "stl": 0.7,
        "blk": 0.9,
        "fg3a": 0.9,
        "fg3_pct": 0.8,
        "ts_pct": 1.0,
        "usg_pct": 1.1,
        "ast_pct": 1.0,
        "reb_pct": 0.8,
        "oreb_pct": 0.6,
        "dreb_pct": 0.6,
        "ast_to": 0.7,
    }

    feature_cols = [c for c in feature_weights if c in league_df.columns]
    usable_features = [c for c in feature_cols if pd.notna(target.get(c))]
    if len(usable_features) < 5:
        return pd.DataFrame()

    comp = league_df.copy()
    comp = comp[comp["PLAYER_NAME"].astype(str).str.lower() != str(player_name).lower()].copy()
    if comp.empty:
        return pd.DataFrame()

    min_gp = max(15, int(gp * 0.4)) if pd.notna(gp) and gp > 0 else 15
    if "gp" in comp.columns:
        comp = comp[pd.to_numeric(comp["gp"], errors="coerce") >= min_gp].copy()
    if comp.empty:
        return pd.DataFrame()

    def zscore(series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        std = vals.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (vals - vals.mean()) / std

    z_cols = {}
    for col in usable_features:
        z = zscore(comp[col])
        z_cols[col] = z
        comp[f"{col}__z"] = z

    target_z = {}
    for col in usable_features:
        vals = pd.to_numeric(comp[col], errors="coerce")
        std = vals.std(ddof=0)
        mean = vals.mean()
        if pd.isna(std) or std == 0:
            target_z[col] = 0.0
        else:
            target_z[col] = (float(target[col]) - mean) / std

    distance = np.zeros(len(comp), dtype=float)
    for col in usable_features:
        weight = feature_weights[col]
        distance += weight * np.square(comp[f"{col}__z"] - target_z[col])
    comp["similarity_distance"] = np.sqrt(distance)

    position = str(latest.get("POSITION", latest.get("Pos", "")) or "").upper()
    if position:
        comp_position = comp["POSITION"].astype(str).str.upper()

        def position_family(pos: str) -> str:
            if "C" in pos:
                return "big"
            if "G" in pos and "F" not in pos:
                return "guard"
            if "F" in pos:
                return "wing"
            return "other"

        target_family = position_family(position)
        candidate_family = comp_position.apply(position_family)
        strict_family = comp[candidate_family == target_family].copy()
        if len(strict_family) >= max(limit + 2, 8):
            comp = strict_family
            comp_position = comp["POSITION"].astype(str).str.upper()
            candidate_family = comp_position.apply(position_family)
        family_penalty = np.where(candidate_family == target_family, 0.0, 0.55)
        exact_penalty = np.where(comp_position == position, 0.0, 0.08)
        comp["similarity_distance"] = comp["similarity_distance"] + family_penalty + exact_penalty

    comp = comp.sort_values("similarity_distance").head(limit).copy()
    if comp.empty:
        return pd.DataFrame()

    min_dist = comp["similarity_distance"].min()
    comp["Similarity"] = (100 - (comp["similarity_distance"] - min_dist) * 18).clip(lower=55, upper=99)

    out = pd.DataFrame({
        "Player": comp["PLAYER_NAME"],
        "Position": comp["POSITION"],
        "Similarity": comp["Similarity"].round(1),
        "PPG": pd.to_numeric(comp.get("pts"), errors="coerce").round(1),
        "RPG": pd.to_numeric(comp.get("reb"), errors="coerce").round(1),
        "APG": pd.to_numeric(comp.get("ast"), errors="coerce").round(1),
        "TS%": (pd.to_numeric(comp.get("ts_pct"), errors="coerce") * 100.0).round(1),
        "3P%": (pd.to_numeric(comp.get("fg3_pct"), errors="coerce") * 100.0).round(1),
        "USG%": (pd.to_numeric(comp.get("usg_pct"), errors="coerce") * 100.0).round(1),
    })

    reason_cols = ["pts", "reb", "ast", "ts_pct", "usg_pct", "blk", "fg3a"]
    reasons = []
    for _, row in comp.iterrows():
        diffs = []
        labels = {
            "pts": "scoring",
            "reb": "rebounding",
            "ast": "playmaking",
            "ts_pct": "efficiency",
            "usg_pct": "usage",
            "blk": "rim protection",
            "fg3a": "three-point volume",
        }
        for col in reason_cols:
            if col not in usable_features:
                continue
            league_val = pd.to_numeric(row.get(col), errors="coerce")
            target_val = pd.to_numeric(target.get(col), errors="coerce")
            if pd.isna(league_val) or pd.isna(target_val):
                continue
            if col == "blk" and float(target_val) < 1.0:
                continue
            if col == "fg3a" and float(target_val) < 2.5:
                continue
            vals = pd.to_numeric(league_df[col], errors="coerce")
            std = vals.std(ddof=0)
            normalized_gap = abs(float(league_val) - float(target_val)) / max(float(std) if pd.notna(std) and std > 0 else 1.0, 1e-6)
            diffs.append((normalized_gap, labels[col]))
        diffs.sort(key=lambda x: x[0])
        reasons.append(", ".join([label for _, label in diffs[:3]]) if diffs else "overall statistical profile")
    out["Why Similar"] = reasons
    return out.reset_index(drop=True)


_NL_METRIC_DEFS = {
    "scoring": {"league_col": "pts", "rank_col": "pts_rank", "label": "PPG", "weight": 1.0},
    "rebounding": {"league_col": "reb", "rank_col": "reb_rank", "label": "RPG", "weight": 0.9},
    "playmaking": {"league_col": "ast_pct", "rank_col": "ast_pct_rank", "label": "AST%", "weight": 1.0},
    "rim_protection": {"league_col": "blk", "rank_col": "blk_rank", "label": "BLK/G", "weight": 1.1},
    "steals": {"league_col": "stl", "rank_col": "stl_rank", "label": "STL/G", "weight": 0.8},
    "efficiency": {"league_col": "ts_pct", "rank_col": "ts_pct_rank", "label": "TS%", "weight": 1.0},
    "shooting": {"league_col": "fg3_pct", "rank_col": "fg3_pct_rank", "label": "3P%", "weight": 0.9},
    "three_point_volume": {"league_col": "fg3a", "rank_col": "fg3a_rank", "label": "3PA/G", "weight": 0.8},
    "usage": {"league_col": "usg_pct", "rank_col": "usg_pct_rank", "label": "USG%", "weight": 0.9},
    "ball_security": {"league_col": "ast_to", "rank_col": "ast_to_rank", "label": "AST/TO", "weight": 0.7},
}

_NL_UNSUPPORTED_HINTS = {
    "young": "age filters",
    "rookie": "age / experience filters",
    "older": "age filters",
    "veteran": "age / experience filters",
    "improving": "multi-season trend filters",
    "declining": "multi-season trend filters",
    "playoff": "playoff-only filters",
    "clutch": "clutch splits",
}


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _position_family(pos: str) -> str:
    pos = str(pos or "").upper()
    if "C" in pos:
        return "big"
    if "G" in pos and "F" not in pos:
        return "guard"
    if "F" in pos:
        return "wing"
    return "other"


def _fallback_nl_query_parse(query: str) -> dict:
    q = str(query or "").lower()
    metric_aliases = {
        "scoring": ["score", "scorer", "points", "ppg", "bucket"],
        "rebounding": ["rebound", "glass", "boards", "rpg"],
        "playmaking": ["playmaking", "passing", "creator", "create", "assist", "ast"],
        "rim_protection": ["rim protection", "rim-protection", "shot blocking", "shot-blocking", "blocks", "blocker", "blk"],
        "steals": ["steals", "stl", "disruption", "disruptive"],
        "efficiency": ["efficient", "efficiency", "true shooting", "ts%", "ts", "efg", "eFG"],
        "shooting": ["shooter", "shooting", "three-point", "3-point", "3pt", "3p%", "spacing"],
        "three_point_volume": ["3pa", "three-point volume", "high-volume shooting", "high volume shooting", "volume from deep"],
        "usage": ["usage", "offensive load", "heliocentric"],
        "ball_security": ["turnover", "ball security", "assist-to-turnover", "ast/to"],
    }

    metrics = [key for key, aliases in metric_aliases.items() if any(alias in q for alias in aliases)]
    if not metrics:
        metrics = ["scoring", "efficiency"]

    position_family = None
    if any(term in q for term in ["big", "bigs", "center", "centers", "frontcourt", "frontcourt player"]):
        position_family = "big"
    elif any(term in q for term in ["guard", "guards", "backcourt"]):
        position_family = "guard"
    elif any(term in q for term in ["wing", "wings", "forward", "forwards"]):
        position_family = "wing"

    min_percentile = 85 if any(term in q for term in ["elite", "best", "top", "high-end"]) else 75 if any(term in q for term in ["great", "strong"]) else 65 if any(term in q for term in ["good", "solid"]) else None
    unsupported = [label for term, label in _NL_UNSUPPORTED_HINTS.items() if term in q]

    return {
        "position_family": position_family,
        "metric_keys": metrics[:4],
        "min_percentile": min_percentile,
        "unsupported_terms": unsupported,
        "force_center": False,
        "summary": "Latest-season player search from plain English.",
    }


def _apply_nl_query_overrides(query: str, parsed: dict) -> dict:
    q = str(query or "").lower().strip()
    out = dict(parsed or {})

    if any(term in q for term in ["3 and d", "3-and-d", "3&d", "3nd", "3 n d"]):
        out["position_family"] = "wing"
        out["metric_keys"] = ["shooting", "three_point_volume", "steals"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 75)
        out["summary"] = "Searching for wing-sized 3-and-D profiles."

    if any(term in q for term in ["stretch 5", "stretch five", "stretch-five"]):
        out["position_family"] = "big"
        out["force_center"] = True
        out["metric_keys"] = ["shooting", "three_point_volume", "efficiency"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 70)
        out["summary"] = "Searching for center-sized stretch-five profiles."

    if any(term in q for term in ["heliocentric guard", "offensive engine guard", "main offensive engine guard"]):
        out["position_family"] = "guard"
        out["metric_keys"] = ["usage", "playmaking", "scoring", "efficiency"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 78)
        out["summary"] = "Searching for guard offensive engines with scoring and creation load."

    if any(term in q for term in ["rim-running big", "rim runner", "rim-running center", "rim running big"]):
        out["position_family"] = "big"
        out["force_center"] = True
        out["metric_keys"] = ["efficiency", "rebounding", "rim_protection"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 72)
        out["summary"] = "Searching for rim-running bigs who finish, rebound, and protect the rim."

    if any(term in q for term in ["shot-creating wing", "shot creator wing", "scoring wing", "shot creating wing"]):
        out["position_family"] = "wing"
        out["metric_keys"] = ["scoring", "usage", "playmaking", "efficiency"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 75)
        out["summary"] = "Searching for wing scorers who can create offense."

    if any(term in q for term in ["point center", "point-centre", "point centre", "offensive hub big", "playmaking center"]):
        out["position_family"] = "big"
        out["force_center"] = True
        out["metric_keys"] = ["playmaking", "rebounding", "efficiency"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 75)
        out["summary"] = "Searching for centers who function like offensive hubs."

    if any(term in q for term in ["floor-spacing big", "floor spacing big", "spacing big", "stretch big"]):
        out["position_family"] = "big"
        out["metric_keys"] = ["shooting", "three_point_volume", "efficiency"]
        out["min_percentile"] = max(int(out.get("min_percentile") or 0), 70)
        out["summary"] = "Searching for bigs who stretch the floor with real shooting value."

    return out


def _ai_parse_nl_query(query: str, model=None) -> dict | None:
    if model is None:
        return None
    prompt = (
        "Convert this NBA player search into compact JSON for a latest-season stat finder.\n"
        "Allowed position_family values: big, wing, guard, null.\n"
        "Allowed metric_keys values: scoring, rebounding, playmaking, rim_protection, steals, efficiency, shooting, three_point_volume, usage, ball_security.\n"
        "Return JSON only with keys: position_family, metric_keys, min_percentile, unsupported_terms, summary.\n"
        "If the user asks for trends, age, rookies, playoffs, or clutch, include that in unsupported_terms.\n"
        f"Query: {query}"
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 220,
                "response_mime_type": "application/json",
            },
        )
        parsed = _extract_json_object(getattr(resp, "text", "") or "")
        if not parsed:
            return None
        parsed["metric_keys"] = [m for m in parsed.get("metric_keys", []) if m in _NL_METRIC_DEFS]
        if not parsed.get("metric_keys"):
            parsed["metric_keys"] = ["scoring", "efficiency"]
        if parsed.get("position_family") not in {"big", "wing", "guard", None}:
            parsed["position_family"] = None
        return parsed
    except Exception:
        return None


def _metric_percentile_from_row(row: pd.Series, league_df: pd.DataFrame, metric_key: str) -> float | None:
    metric = _NL_METRIC_DEFS[metric_key]
    total = int(league_df["PLAYER_ID"].nunique()) if "PLAYER_ID" in league_df.columns else int(len(league_df))
    rank_val = pd.to_numeric(row.get(metric["rank_col"]), errors="coerce")
    if pd.notna(rank_val) and total > 1:
        rank_int = int(rank_val)
        return round(((total - rank_int) / max(total - 1, 1)) * 100.0, 1)

    series = pd.to_numeric(league_df.get(metric["league_col"]), errors="coerce")
    value = pd.to_numeric(row.get(metric["league_col"]), errors="coerce")
    if pd.isna(value) or series.dropna().empty:
        return None
    return round(float((series <= value).mean() * 100.0), 1)


def _metric_is_eligible(row: pd.Series, metric_key: str) -> bool:
    pts = pd.to_numeric(row.get("pts"), errors="coerce")
    reb = pd.to_numeric(row.get("reb"), errors="coerce")
    ast = pd.to_numeric(row.get("ast"), errors="coerce")
    blk = pd.to_numeric(row.get("blk"), errors="coerce")
    stl = pd.to_numeric(row.get("stl"), errors="coerce")
    gp = pd.to_numeric(row.get("gp"), errors="coerce")
    mpg = pd.to_numeric(row.get("min"), errors="coerce")
    fg3a = pd.to_numeric(row.get("fg3a"), errors="coerce")
    usg = pd.to_numeric(row.get("usg_pct"), errors="coerce")
    ast_pct = pd.to_numeric(row.get("ast_pct"), errors="coerce")

    if metric_key == "shooting":
        return pd.notna(fg3a) and fg3a >= 3.5
    if metric_key == "three_point_volume":
        return pd.notna(fg3a) and fg3a >= 4.0
    if metric_key == "efficiency":
        return ((pd.notna(pts) and pts >= 12) or (pd.notna(usg) and usg >= 0.18)) and ((pd.notna(mpg) and mpg >= 18) or (pd.notna(gp) and gp >= 20))
    if metric_key == "rim_protection":
        return pd.notna(blk) and blk >= 0.8 and pd.notna(mpg) and mpg >= 18
    if metric_key == "steals":
        return pd.notna(stl) and stl >= 0.8 and pd.notna(mpg) and mpg >= 18
    if metric_key == "playmaking":
        return (pd.notna(ast) and ast >= 3.0) or (pd.notna(ast_pct) and ast_pct >= 0.18)
    if metric_key == "rebounding":
        return (pd.notna(reb) and reb >= 5.0) or (pd.notna(mpg) and mpg >= 20)
    if metric_key == "scoring":
        return pd.notna(pts) and pts >= 10
    if metric_key == "usage":
        return pd.notna(usg) and usg >= 0.18
    if metric_key == "ball_security":
        return (pd.notna(ast) and ast >= 3.0) or (pd.notna(ast_pct) and ast_pct >= 0.18)
    return True


@st.cache_data(ttl=1800, show_spinner=False)
def find_players_by_natural_language(query: str, season: int | None, limit: int, use_model: bool, _model=None) -> tuple[pd.DataFrame, dict]:
    if not query or not str(query).strip():
        return pd.DataFrame(), {"message": "Enter a plain-English prompt to search the latest-season player pool."}

    if season is None:
        now = pd.Timestamp.now(tz="America/New_York")
        season = int(now.year if now.month >= 10 else now.year - 1)

    league_df = get_balldontlie_league_season_averages(int(season))
    if league_df is None or league_df.empty:
        return pd.DataFrame(), {"message": "League season data could not be loaded right now."}

    parsed = _ai_parse_nl_query(query, _model if use_model else None) or _fallback_nl_query_parse(query)
    parsed = _apply_nl_query_overrides(query, parsed)
    metric_keys = parsed.get("metric_keys") or ["scoring", "efficiency"]
    position_family = parsed.get("position_family")
    min_percentile = parsed.get("min_percentile")
    force_center = bool(parsed.get("force_center"))

    comp = league_df.copy()
    if "gp" in comp.columns:
        comp = comp[pd.to_numeric(comp["gp"], errors="coerce") >= 12].copy()
    if "min" in comp.columns:
        comp = comp[pd.to_numeric(comp["min"], errors="coerce") >= 12].copy()
    if position_family:
        comp = comp[comp["POSITION"].apply(_position_family) == position_family].copy()
    if force_center:
        comp = comp[comp["POSITION"].astype(str).str.upper().str.contains("C", na=False)].copy()
    if comp.empty:
        return pd.DataFrame(), {
            "message": "No players matched that position filter in the latest-season pool.",
            "summary": parsed.get("summary", ""),
            "unsupported_terms": parsed.get("unsupported_terms", []),
        }

    strict = comp.copy()
    if min_percentile is not None:
        for metric_key in metric_keys:
            strict = strict[strict.apply(lambda row: _metric_is_eligible(row, metric_key), axis=1)].copy()
            if strict.empty:
                break
            strict = strict[
                strict.apply(
                    lambda row: (_metric_percentile_from_row(row, comp, metric_key) or 0) >= float(min_percentile),
                    axis=1,
                )
            ].copy()
            if strict.empty:
                break

    candidates = strict if not strict.empty else comp.copy()
    scores = []
    reasons = []
    for _, row in candidates.iterrows():
        weighted = 0.0
        weight_total = 0.0
        row_reasons = []
        for metric_key in metric_keys:
            if not _metric_is_eligible(row, metric_key):
                continue
            pct = _metric_percentile_from_row(row, comp, metric_key)
            if pct is None:
                continue
            metric = _NL_METRIC_DEFS[metric_key]
            weight = metric["weight"]
            if metric_key == "shooting":
                fg3a = pd.to_numeric(row.get("fg3a"), errors="coerce")
                volume_bonus = min(max((float(fg3a) - 3.5) / 4.5, 0.0), 1.0) if pd.notna(fg3a) else 0.0
                pct = pct * (0.75 + 0.25 * volume_bonus)
            elif metric_key == "efficiency":
                pts = pd.to_numeric(row.get("pts"), errors="coerce")
                usg = pd.to_numeric(row.get("usg_pct"), errors="coerce")
                load_bonus = max(
                    min(((float(pts) - 12.0) / 18.0), 1.0) if pd.notna(pts) else 0.0,
                    min(((float(usg) - 0.18) / 0.14), 1.0) if pd.notna(usg) else 0.0,
                )
                pct = pct * (0.8 + 0.2 * max(load_bonus, 0.0))
            weighted += pct * weight
            weight_total += weight
            if pct >= 80:
                row_reasons.append(f"{metric['label']} ({pct:.0f}th percentile)")
        score = (weighted / weight_total) if weight_total else 0.0
        scores.append(score)
        reasons.append(", ".join(row_reasons[:3]) if row_reasons else "overall profile fit")

    candidates = candidates.copy()
    candidates["Match Score"] = scores
    candidates["Why It Matched"] = reasons
    candidates = candidates[candidates["Match Score"] > 0].copy()
    candidates = candidates.sort_values("Match Score", ascending=False).head(limit).copy()
    if candidates.empty:
        return pd.DataFrame(), {
            "message": "No players matched that search right now.",
            "summary": parsed.get("summary", ""),
            "unsupported_terms": parsed.get("unsupported_terms", []),
        }

    out = pd.DataFrame({
        "Player": candidates["PLAYER_NAME"],
        "Position": candidates["POSITION"],
        "Match": candidates["Match Score"].round(1),
        "PPG": pd.to_numeric(candidates.get("pts"), errors="coerce").round(1),
        "RPG": pd.to_numeric(candidates.get("reb"), errors="coerce").round(1),
        "APG": pd.to_numeric(candidates.get("ast"), errors="coerce").round(1),
        "TS%": (pd.to_numeric(candidates.get("ts_pct"), errors="coerce") * 100.0).round(1),
        "3P%": (pd.to_numeric(candidates.get("fg3_pct"), errors="coerce") * 100.0).round(1),
        "BLK/G": pd.to_numeric(candidates.get("blk"), errors="coerce").round(1),
        "STL/G": pd.to_numeric(candidates.get("stl"), errors="coerce").round(1),
        "USG%": (pd.to_numeric(candidates.get("usg_pct"), errors="coerce") * 100.0).round(1),
        "Why It Matched": candidates["Why It Matched"],
        "Player Token": candidates["PLAYER_ID"].apply(lambda x: f"balldontlie:{int(x)}" if pd.notna(x) else None),
    })

    meta = {
        "message": "",
        "summary": parsed.get("summary", "Latest-season player search from plain English."),
        "unsupported_terms": parsed.get("unsupported_terms", []),
        "used_position_filter": position_family,
        "metric_labels": [_NL_METRIC_DEFS[key]["label"] for key in metric_keys],
        "strict_match_count": int(len(strict)) if min_percentile is not None else int(len(candidates)),
        "relaxed": bool(min_percentile is not None and strict.empty),
        "season": season,
    }
    return out.reset_index(drop=True), meta
