# nba_app/metrics.py
import numpy as np
import pandas as pd
from fetch import get_team_totals_many

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
    Works for both team rows and TOT rows. Uses whatever join keys exist in both frames.
    """
    if adv_df is None or adv_df.empty or per_game_df is None or per_game_df.empty:
        return adv_df

    # Columns we want from per-game
    keep_pg = ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'PTS', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'MIN']
    pg = per_game_df[[c for c in keep_pg if c in per_game_df.columns]].copy()

    # Rename to per-game labels
    rename_map = {
        'PTS': 'PPG',
        'REB': 'RPG',
        'AST': 'APG',
        'TOV': 'TPG',
        'STL': 'SPG',
        'BLK': 'BPG',
        'MIN': 'MPG',
    }
    pg = pg.rename(columns=rename_map)

    # Build join keys that exist in both dataframes (prefer SEASON_ID + TEAM_ID)
    join_keys = []
    for k in ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']:
        if k in adv_df.columns and k in pg.columns:
            join_keys.append(k)
    if 'SEASON_ID' not in join_keys:
        # Can't safely merge without season; just return adv as-is
        return adv_df

    merged = adv_df.merge(pg, on=join_keys, how='left')

    # Round per-game columns nicely
    for c in ['PPG', 'RPG', 'APG', 'TPG', 'SPG', 'BPG', 'MPG']:
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
    pg = per_game_df.copy()

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
    adv = adv_df.copy() if adv_df is not None else pd.DataFrame()
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
    # core advanced (summary)
    "TS%", "EFG%", "PPS", "3PAr", "FTr", "USG% (true)", "AST%", "TRB%", "ORB%", "DRB%", "AST/TO",
    # per-36
    "PTS/36", "REB/36", "AST/36", "STL/36", "BLK/36", "TOV/36", "FGM/36", "FGA/36", "FG3M/36", "OREB/36", "DREB/36",
    # shooting splits (if present)
    "FG%", "3P%", "FT%",
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
