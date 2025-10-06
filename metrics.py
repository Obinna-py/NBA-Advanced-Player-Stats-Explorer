# nba_app/metrics.py
import numpy as np
import pandas as pd
from fetch import get_team_totals_many

def compute_full_advanced_stats(player_df_totals: pd.DataFrame) -> pd.DataFrame:
    if player_df_totals is None or player_df_totals.empty:
        return player_df_totals
    out = player_df_totals.copy()

    if 'FG_PCT' in out.columns:  out['FG%']  = out['FG_PCT']*100
    if 'FG3_PCT' in out.columns: out['3P%'] = out['FG3_PCT']*100
    if 'FT_PCT' in out.columns:  out['FT%']  = out['FT_PCT']*100

    denom_ts = (out.get('FGA',0) + 0.44*out.get('FTA',0))
    out['TS%']  = np.where(denom_ts>0, out.get('PTS',0)/(2*denom_ts)*100, np.nan)
    out['EFG%'] = np.where(out.get('FGA',0)>0, (out.get('FGM',0)+0.5*out.get('FG3M',0))/out.get('FGA',0)*100, np.nan)
    out['PPS']  = np.where(out.get('FGA',0)>0, out.get('PTS',0)/out.get('FGA',0), np.nan)
    out['3PAr'] = np.where(out.get('FGA',0)>0, out.get('FG3A',0)/out.get('FGA',0), np.nan)
    out['FTr']  = np.where(out.get('FGA',0)>0, out.get('FTA',0)/out.get('FGA',0), np.nan)
    out['AST/TO'] = np.where(out.get('TOV',0)>0, out.get('AST',0)/out.get('TOV',0), np.nan)

    mins = out.get('MIN',0)
    def per36(col): return np.where(mins>0, out.get(col,0)/mins*36, np.nan)
    for stat in ['PTS','REB','AST','STL','BLK','TOV','FGM','FGA','FG3M','OREB','DREB']:
        out[f'{stat}/36'] = per36(stat)

    seasons_needed = out['SEASON_ID'].dropna().unique().tolist()
    team_totals = get_team_totals_many(seasons_needed)
    out = out.merge(team_totals.add_prefix('TEAM_'),
                    left_on=['SEASON_ID','TEAM_ID'],
                    right_on=['TEAM_SEASON_ID','TEAM_TEAM_ID'],
                    how='left')

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

    denom_usg = (tFGA + 0.44*tFTA + tTOV)
    num_usg   = (out.get('FGA',0) + 0.44*out.get('FTA',0) + out.get('TOV',0)) * np.where(tMIN>0, tMIN/5.0, 0)
    out['USG% (true)'] = np.where((denom_usg>0) & (mp>0), 100.0 * num_usg / (mp*denom_usg), np.nan)

    denom_ast = (np.where(tMIN>0, mp/tMIN, 0) * tFGM) - out.get('FGM',0)
    out['AST%'] = np.where(denom_ast != 0, 100.0 * out.get('AST',0)/denom_ast, np.nan)

    denom_orb = tOREB + np.where(oDREB>0, oDREB, 0.0)
    denom_drb = tDREB + np.where(oOREB>0, oOREB, 0.0)
    denom_trb = tTRB  + np.where(oTRB >0, oTRB,  0.0)
    scale = np.where((mp>0) & (tMIN>0), (tMIN/5.0)/mp, np.nan)

    out['ORB%'] = np.where(denom_orb>0, 100.0 * out.get('OREB',0)*scale / denom_orb, np.nan)
    out['DRB%'] = np.where(denom_drb>0, 100.0 * out.get('DREB',0)*scale / denom_drb, np.nan)
    out['TRB%'] = np.where(denom_trb>0, 100.0 * out.get('REB',0) *scale / denom_trb, np.nan)

    out = out.drop(columns=[c for c in ['FG_PCT','FG3_PCT','FT_PCT','TEAM_SEASON_ID','TEAM_TEAM_ID'] if c in out.columns])
    float_cols = out.select_dtypes(include=['float','float64','float32']).columns
    out[float_cols] = out[float_cols].round(2)
    out = out[[c for c in out.columns if not c.startswith("TEAM_")]]
    return out

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

    return merged


def generate_player_summary(player_name: str, stats_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    if stats_df is None or stats_df.empty:
        return f"No available stats for {player_name}."
    lines = [f"📊 Full season-by-season stats for **{player_name}**:\n"]
    for i in range(len(stats_df)):
        s = stats_df.iloc[i]
        a = adv_df.iloc[i] if (adv_df is not None and i < len(adv_df)) else {}
        lines += [
            "---",
            f"### Season {s['SEASON_ID']} ({s['TEAM_ABBREVIATION']})",
            f"- **PPG:** {s.get('PTS', np.nan):.1f}, **RPG:** {s.get('REB', np.nan):.1f}, **APG:** {s.get('AST', np.nan):.1f}",
            f"- **SPG:** {s.get('STL', np.nan):.1f}, **BPG:** {s.get('BLK', np.nan):.1f}, **TPG:** {s.get('TOV', np.nan):.1f}",
            f"- **Games Played:** {s.get('GP', np.nan)}, **Minutes/Game:** {s.get('MIN', np.nan):.1f}",
            f"- **TS%:** {a.get('TS%', np.nan):.2f}%, **EFG%:** {a.get('EFG%', np.nan):.2f}%, **PPS:** {a.get('PPS', np.nan):.2f}",
            f"- **USG% (true):** {a.get('USG% (true)', np.nan):.2f}%",
            f"- **PTS/36:** {a.get('PTS/36', np.nan):.2f}, **REB/36:** {a.get('REB/36', np.nan):.2f}, **AST/36:** {a.get('AST/36', np.nan):.2f}",
            f"- **AST/TO:** {a.get('AST/TO', np.nan):.2f}, **3PAr:** {a.get('3PAr', np.nan):.2f}, **FTr:** {a.get('FTr', np.nan):.2f}",
            f"- **ORB%:** {a.get('ORB%', np.nan):.2f}%, **DRB%:** {a.get('DRB%', np.nan):.2f}%, **TRB%:** {a.get('TRB%', np.nan):.2f}%, **AST%:** {a.get('AST%', np.nan):.2f}%",
        ]
    return "\n".join(lines)

def compact_player_context(df: pd.DataFrame) -> dict:
    if df is None or df.empty: return {}
    row = df.iloc[-1]
    g = lambda k, d=np.nan: float(row[k]) if k in row and pd.notna(row[k]) else d
    return {
        "season": row.get("SEASON_ID", "Unknown"),
        "team": row.get("TEAM_ABBREVIATION", "UNK"),
        "ppg": g("PTS"), "rpg": g("REB"), "apg": g("AST"),
        "ts": g("TS%"), "efg": g("EFG%"),
        "usg": g("USG% (true)") if "USG% (true)" in row else np.nan,
        "mpg": g("MIN"),
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