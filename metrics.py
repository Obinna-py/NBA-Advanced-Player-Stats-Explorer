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
