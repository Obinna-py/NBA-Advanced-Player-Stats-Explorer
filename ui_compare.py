# ui_compare.py
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
from config import ai_generate_text

# --- project imports
from fetch import get_player_career, get_head_to_head_games, get_player_info, search_players, get_nba_headshot_url
from metrics import (
    compute_full_advanced_stats,
    add_per_game_columns,
    generate_player_summary,
    compact_player_context,
    compute_player_percentile_context,
    metric_public_cols,          # returns a filtered/ordered list of columns for display
    order_columns_for_display,
    compute_league_shooting_table    # preferred display ordering (PPG/RPG/APG → TS% → etc.)
)
from ideas import cached_ai_compare_question_ideas
from utils import abbrev, make_anchor, smooth_scroll_to
from datetime import datetime


# -------------------------
# Small helpers
# -------------------------
def _friendly_ai_error_message(error: Exception) -> str:
    text = str(error or "").lower()
    if any(term in text for term in ["quota", "resourceexhausted", "resource exhausted", "rate limit", "429"]):
        return "AI is temporarily unavailable because the current OpenAI quota or rate limit has been reached. Please try again a little later."
    if any(term in text for term in ["api key", "permission", "unauthorized", "403"]):
        return "AI is unavailable right now because the OpenAI connection or permissions need attention."
    return "AI is unavailable right now. Please try again in a moment."


def render_html_table(
    df,
    *,
    rename_map=None,
    number_cols=None,
    percent_cols=None,
    date_cols=None,
    max_height_px=420,
):
    import pandas as pd
    if df is None or df.empty:
        st.info("No data to display.")
        return

    df2 = df.copy()
    if rename_map:
        df2.rename(columns=rename_map, inplace=True)

    number_cols  = [c for c in (number_cols  or []) if c in df2.columns]
    percent_cols = [c for c in (percent_cols or []) if c in df2.columns]
    date_cols    = [c for c in (date_cols    or []) if c in df2.columns]

    # format
    for c in date_cols:
        try:
            df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    for c in number_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").round(1)
    for c in percent_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").round(2)

    # build HTML with pandas Styler (then inject our own CSS)
    
    fmt = {}

    # Explicit lists win
    for c in (number_cols or []):
        fmt[c] = "{:.1f}"
    for c in (percent_cols or []):
        fmt[c] = "{:.2f}%"

    # Auto-apply to any other numeric columns not covered above
    auto_numeric = [c for c in df2.select_dtypes(include="number").columns if c not in fmt]
    for c in auto_numeric:
        fmt[c] = "{:.1f}"

    # Build HTML with pandas Styler
    styler = (
        df2.style
        .hide(axis="index")
        .format(fmt, na_rep="—")  # <- critical: formatting happens here
        .set_table_attributes('class="nice-table"')
    )

    html = styler.to_html()



    st.markdown(f"""
<style>
.nice-table {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  overflow: hidden;
  font-size: 0.92rem;
}}
.nice-table thead th {{
  background: rgba(255,255,255,0.06);
  text-align: left;
  font-weight: 600;
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.nice-table tbody td {{
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.nice-table tbody tr:nth-child(even) td {{ background: rgba(255,255,255,0.03); }}
.nice-table tbody tr:hover td {{ background: rgba(255,255,255,0.06); }}
.nice-table-wrapper {{
  max-height: {max_height_px}px;
  overflow: auto;
  border-radius: 12px;
}}
/* right-align obvious numeric columns */
.nice-table td:has(.num), .nice-table th:has(.num) {{ text-align: right; }}
</style>
<div class="nice-table-wrapper">
{html}
</div>
""", unsafe_allow_html=True)




# light CSS polish (rounded corners, zebra stripes, compact font)
st.markdown("""
<style>
/* round corners & subtle border */
[data-testid="stDataFrame"] div[role="grid"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}
/* header tweaks */
[data-testid="stDataFrame"] thead tr th {
  font-weight: 600;
  letter-spacing: .2px;
}
/* zebra rows */
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
  background-color: rgba(255,255,255,0.03);
}
/* slightly smaller cell font for density */
[data-testid="stDataFrame"] div[role="gridcell"] {
  font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# --- make the stats table human-friendly -------------------------------------
def _make_readable_stats_table(df: pd.DataFrame):
    """
    Take the wide advanced/per-game dataframe and return:
      - a cleaned dataframe with readable column names
      - lists of number_cols and percent_cols for render_html_table()
    """
    if df is None or df.empty:
        return df, [], []

    # 1) Start from your public/order helper
    cols = [c for c in metric_public_cols(df)]

    # 2) Remove backend/noisy columns (suffix merges, ids, etc.)
    drop_if_suffix = ("_x", "_y")
    cols = [
        c for c in cols
        if not c.endswith(drop_if_suffix) and c not in {
            "PLAYER_AGE", "CFID", "CFPARAMS",
            "FG_PCT", "FG3_PCT", "FT_PCT",
            "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
            "PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_3PT",
            "PCT_PTS_MIDRANGE_2PT", "PCT_PTS_PAINT",
        }
    ]
    nice = df[cols].copy()

    # 3) Rename to readable labels
    rename = {
        "SEASON_ID": "Season",
        "TEAM_ABBREVIATION": "Team",

        # per-game (more explicit)
        "MPG": "MIN/G",
        "PPG": "PTS/G",
        "RPG": "REB/G",
        "APG": "AST/G",
        "SPG": "STL/G",
        "BPG": "BLK/G",
        "TPG": "TOV/G",

        # rates/efficiency
        "USG% (true)": "USG%",
        "EFG%": "eFG%",
        "TS%": "TS%",
        "PPS": "Pts/Shot",
        "3PAr": "3PA Rate",
        "FTr": "FT Rate",

        # attempts per game you added
        "3PA/G": "3PA/G",
        "2PA/G": "2PA/G",

        # season totals
        "FGM": "FGM",
        "FG3M": "3PM",
        "FTM": "FTM",
        "FTA": "FTA",
        "OREB": "OREB",
        "DREB": "DREB",
        "PLUS_MINUS": "+/-",
    }
    nice.rename(columns=rename, inplace=True)

    # 4) Preferred order (only keep those that actually exist)
    preferred = [
        "Season", "Team",
        "MIN/G", "PTS/G", "REB/G", "AST/G", "STL/G", "BLK/G", "TOV/G",
        "3PA/G", "2PA/G",
        "FG%", "3P%", "FT%", "TS%", "eFG%",
        "Pts/Shot", "3PA Rate", "FT Rate",
        "USG%", "AST%", "TRB%", "ORB%", "DRB%", "AST/TO",
        "FGM", "3PM", "FTM", "FTA", "OREB", "DREB", "+/-",
        "PTS/36", "REB/36", "AST/36", "STL/36", "BLK/36", "TOV/36",
        "FGM/36", "FGA/36", "FG3M/36", "OREB/36", "DREB/36",
    ]
    ordered = [c for c in preferred if c in nice.columns] + [c for c in nice.columns if c not in preferred]
    nice = nice[ordered]

    # 5) Tell the HTML renderer how to format things
    percent_cols = [c for c in ["FG%", "3P%", "FT%", "TS%", "eFG%", "USG%", "AST%", "TRB%", "ORB%", "DRB%"] if c in nice.columns]
    # Everything numeric that isn’t a percent → one decimal
    number_cols = [c for c in nice.columns if c not in percent_cols and pd.api.types.is_numeric_dtype(nice[c])]

    # Totals should read like basketball box stats, not high-precision float outputs.
    integer_like_cols = [c for c in ["GP", "FGM", "3PM", "FTM", "FTA", "OREB", "DREB", "+/-"] if c in nice.columns]
    for c in integer_like_cols:
        nice[c] = pd.to_numeric(nice[c], errors="coerce").round(1)

    return nice, number_cols, percent_cols


def _nzdf(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return df if it's a DataFrame, else empty DataFrame."""
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _add_season_start(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure SEASON_START exists (int) derived from SEASON_ID like '2019-20'."""
    if df is not None and not df.empty and "SEASON_ID" in df.columns:
        if "SEASON_START" not in df.columns:
            df = df.copy()
            df["SEASON_START"] = df["SEASON_ID"].astype(str).str[:4].astype(int)
    return df

# --- ERA / AXIS HELPERS ---
def _career_index(df: pd.DataFrame) -> pd.DataFrame:
    """Adds CAREER_YEAR = 1..N by SEASON_START order."""
    if df is None or df.empty or "SEASON_START" not in df.columns:
        return df
    out = df.copy().sort_values("SEASON_START").reset_index(drop=True)
    out["CAREER_YEAR"] = np.arange(1, len(out) + 1)
    return out

def _get_birth_year(player_id: int) -> int | None:
    """Year-only (approx) for season-level age alignment."""
    try:
        info = get_player_info(player_id)
        bdate = str(info.loc[0, "BIRTHDATE"]).split("T")[0]
        return int(bdate[:4])
    except Exception:
        return None

def _add_age_column(df: pd.DataFrame, birth_year: int | None) -> pd.DataFrame:
    """Adds AGE_APPROX = SEASON_START - birth_year."""
    if df is None or df.empty or "SEASON_START" not in df.columns or birth_year is None:
        return df
    out = df.copy()
    out["AGE_APPROX"] = out["SEASON_START"].astype(int) - int(birth_year)
    return out

def _align_for_chart(src1: pd.DataFrame, src2: pd.DataFrame, p1: str, p2: str, stat: str, mode: str) -> pd.DataFrame:
    """
    Returns tidy frame with x-axis depending on alignment mode:
      - "Calendar (overlap only)" -> x = SEASON_ID (season labels)
      - "Career year"             -> x = CAREER_YEAR (1..N)
      - "Age"                     -> x = AGE_APPROX (years)
    """
    need_cols = {"SEASON_START", "SEASON_ID", stat}
    if src1 is None or src2 is None or src1.empty or src2.empty:
        return pd.DataFrame()
    if not need_cols.issubset(src1.columns) or not need_cols.issubset(src2.columns):
        return pd.DataFrame()

    if mode == "Career year":
        s1 = _career_index(src1); s2 = _career_index(src2); on = "CAREER_YEAR"
    elif mode == "Age":
        on = "AGE_APPROX"
        if on not in src1.columns or on not in src2.columns:
            return pd.DataFrame()
        s1, s2 = src1, src2
    else:
        s1, s2 = src1, src2; on = "SEASON_START"

    keep1 = [on, "SEASON_ID", stat]
    keep2 = [on, "SEASON_ID", stat]
    merged = s1[keep1].merge(s2[keep2], on=on, suffixes=(f"_{p1}", f"_{p2}"))
    if merged.empty:
        return merged

    if mode == "Calendar (overlap only)":
        merged["X_AXIS"] = merged[f"SEASON_ID_{p1}"]
    else:
        merged["X_AXIS"] = merged[on].astype(int)

    return pd.DataFrame({
        "X": merged["X_AXIS"],
        p1: pd.to_numeric(merged[f"{stat}_{p1}"], errors="coerce"),
        p2: pd.to_numeric(merged[f"{stat}_{p2}"], errors="coerce"),
    })


def _find_player_by_name(name: str):
    if not name:
        return None
    found = search_players(name)
    if not found:
        return None
    exact = [p for p in found if p["full_name"].lower() == name.lower()]
    return (exact or found)[0]


def _pick_shared_stats_for_dropdown(src1: pd.DataFrame, src2: pd.DataFrame) -> list[str]:
    """
    Build & order the selectable stats for the compare dropdown.
    Uses order_columns_for_display() when possible and hides internals like SEASON_START.
    """
    if src1 is None or src2 is None or src1.empty or src2.empty:
        return []

    excluded = {
        "SEASON_ID", "PLAYER_ID", "TEAM_ID", "LEAGUE_ID",
        "TEAM_ABBREVIATION", "SEASON_START"
    }

    numeric1 = {c for c in src1.columns if pd.api.types.is_numeric_dtype(src1[c])}
    numeric2 = {c for c in src2.columns if pd.api.types.is_numeric_dtype(src2[c])}
    candidates = list((numeric1 & numeric2) - excluded)

    if not candidates:
        return []

    # Prefer your global display ordering if available
    try:
        ordered_pref = [c for c in order_columns_for_display(src1) if c in candidates]
        # add any remaining candidates at the end
        ordered_pref += [c for c in candidates if c not in ordered_pref]
        return ordered_pref
    except Exception:
        # Fallback simple priority
        priority = [
            "PPG", "RPG", "APG", "SPG", "BPG", "TPG",
            "TS%", "EFG%", "PPS", "3PAr", "FTr",
            "USG% (true)", "AST%", "TRB%", "ORB%", "DRB%",
            "PTS/36", "REB/36", "AST/36", "STL/36", "BLK/36", "TOV/36",
            "FG%", "3P%", "FT%"
        ]
        ordered = [c for c in priority if c in candidates] + [c for c in candidates if c not in priority]
        return ordered


def _build_overlap_for_chart(src1: pd.DataFrame, src2: pd.DataFrame, p1: str, p2: str, stat: str) -> pd.DataFrame:
    """
    Build overlapping seasons dataframe for charting the selected stat.
    Assumes SEASON_START exists on both sources.
    """
    if src1 is None or src2 is None or src1.empty or src2.empty:
        return pd.DataFrame()
    if "SEASON_START" not in src1.columns or "SEASON_START" not in src2.columns:
        return pd.DataFrame()

    cols_needed = {"SEASON_START", "SEASON_ID", stat}
    if not cols_needed.issubset(src1.columns) or not cols_needed.issubset(src2.columns):
        return pd.DataFrame()

    common = src1[list(cols_needed)].merge(
        src2[list(cols_needed)],
        on="SEASON_START",
        suffixes=(f"_{p1}", f"_{p2}")
    )
    return common

def _ensure_ctx_dict(ctx_obj):
    # ideas._seed_compare_questions expects a dict with .get(...)
    if isinstance(ctx_obj, dict):
        return ctx_obj
    if ctx_obj is None:
        return {}

    # pandas objects
    try:
        if hasattr(ctx_obj, "to_dict"):
            return ctx_obj.to_dict()
    except Exception:
        pass

    # JSON string? try to parse to dict
    if isinstance(ctx_obj, str):
        s = ctx_obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
                return {"items": parsed}
            except Exception:
                pass
        return {"text": ctx_obj}

    # Last resort
    try:
        return dict(ctx_obj)
    except Exception:
        return {"text": str(ctx_obj)}


def _player_key(player: dict) -> str:
    return f"{player.get('source', 'nba_api')}:{player.get('id')}"


def _dedupe_players(players: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for player in players:
        key = _player_key(player)
        if key in seen:
            continue
        seen.add(key)
        out.append(player)
    return out


def _pick_shared_stats_for_many(frames: list[pd.DataFrame]) -> list[str]:
    usable = [df for df in frames if df is not None and not df.empty]
    if len(usable) < 2:
        return []

    excluded = {
        "SEASON_ID", "PLAYER_ID", "TEAM_ID", "LEAGUE_ID",
        "TEAM_ABBREVIATION", "SEASON_START", "CAREER_YEAR", "AGE_APPROX"
    }
    shared = None
    for df in usable:
        numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
        shared = numeric if shared is None else (shared & numeric)

    candidates = list((shared or set()) - excluded)
    if not candidates:
        return []

    ordered = [c for c in order_columns_for_display(usable[0]) if c in candidates]
    ordered += [c for c in candidates if c not in ordered]
    return ordered


def _seed_multi_compare_questions(player_names: list[str]) -> list[str]:
    joined = ", ".join(player_names)
    return [
        f"Who is the best overall offensive player among {joined}?",
        f"Which of {joined} scores most efficiently?",
        f"Who has the best playmaking profile among {joined}?",
        f"Which player has the best scoring versus usage balance among {joined}?",
        f"Who is the strongest rebounder and interior presence among {joined}?",
        f"Which of {joined} has the best all-around statistical case?",
        f"Who looks best by recent trend among {joined}?",
        f"How would you rank {joined} based on these stats?",
    ]


def _build_multi_aligned_df(player_frames: list[dict], stat: str, align_mode: str) -> pd.DataFrame:
    rows = []
    for item in player_frames:
        df = item["chart_src"]
        if df is None or df.empty or stat not in df.columns:
            continue

        working = df.copy()
        if align_mode == "Career year":
            working = _career_index(working)
            x_col = "CAREER_YEAR"
        elif align_mode == "Age":
            x_col = "AGE_APPROX"
        else:
            x_col = "SEASON_ID"

        if x_col not in working.columns:
            continue

        part = pd.DataFrame({
            "X": working[x_col],
            "Value": pd.to_numeric(working[stat], errors="coerce"),
            "Player": item["name"],
        }).dropna(subset=["X", "Value"])
        rows.append(part)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    if series.dropna().empty:
        return None
    return float(series.mean())


def _safe_pct(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return (num / den) * 100.0


def _h2h_matchup_insights(h2h: pd.DataFrame, p1: str, p2: str) -> dict:
    if h2h is None or h2h.empty:
        return {}

    p1_pts = _safe_mean(h2h, "PTS_P1")
    p2_pts = _safe_mean(h2h, "PTS_P2")
    p1_reb = _safe_mean(h2h, "REB_P1")
    p2_reb = _safe_mean(h2h, "REB_P2")
    p1_ast = _safe_mean(h2h, "AST_P1")
    p2_ast = _safe_mean(h2h, "AST_P2")
    p1_blk = _safe_mean(h2h, "BLK_P1")
    p2_blk = _safe_mean(h2h, "BLK_P2")
    p1_stl = _safe_mean(h2h, "STL_P1")
    p2_stl = _safe_mean(h2h, "STL_P2")
    p1_tov = _safe_mean(h2h, "TOV_P1")
    p2_tov = _safe_mean(h2h, "TOV_P2")

    p1_fgm = _safe_mean(h2h, "FGM_P1")
    p2_fgm = _safe_mean(h2h, "FGM_P2")
    p1_fga = _safe_mean(h2h, "FGA_P1")
    p2_fga = _safe_mean(h2h, "FGA_P2")
    p1_fg3m = _safe_mean(h2h, "FG3M_P1")
    p2_fg3m = _safe_mean(h2h, "FG3M_P2")
    p1_fg3a = _safe_mean(h2h, "FG3A_P1")
    p2_fg3a = _safe_mean(h2h, "FG3A_P2")
    p1_ftm = _safe_mean(h2h, "FTM_P1")
    p2_ftm = _safe_mean(h2h, "FTM_P2")
    p1_fta = _safe_mean(h2h, "FTA_P1")
    p2_fta = _safe_mean(h2h, "FTA_P2")

    p1_fg = _safe_pct(p1_fgm, p1_fga)
    p2_fg = _safe_pct(p2_fgm, p2_fga)
    p1_3p = _safe_pct(p1_fg3m, p1_fg3a)
    p2_3p = _safe_pct(p2_fg3m, p2_fg3a)
    p1_ft = _safe_pct(p1_ftm, p1_fta)
    p2_ft = _safe_pct(p2_ftm, p2_fta)

    p1_ts = _safe_pct(p1_pts, 2 * ((p1_fga or 0) + 0.44 * (p1_fta or 0)))
    p2_ts = _safe_pct(p2_pts, 2 * ((p2_fga or 0) + 0.44 * (p2_fta or 0)))

    categories = []

    def add_edge(label: str, v1: float | None, v2: float | None, higher_is_better: bool = True, fmt: str = "{:.1f}"):
        if v1 is None or v2 is None:
            return
        diff = v1 - v2
        if not higher_is_better:
            diff = -diff
        if abs(diff) < 1e-9:
            winner = "Even"
            detail = f"Both are basically even in {label.lower()}."
        else:
            winner = p1 if diff > 0 else p2
            display_gap = (v1 - v2) if higher_is_better else (v2 - v1)
            detail = f"{winner} has the edge in {label.lower()} ({fmt.format(v1)} vs {fmt.format(v2)})."
            if not higher_is_better:
                detail = f"{winner} protects the ball better in {label.lower()} ({fmt.format(v1)} vs {fmt.format(v2)} turnovers)."
        categories.append({"Category": label, "Winner": winner, "Detail": detail})

    add_edge("Scoring", p1_pts, p2_pts)
    add_edge("Rebounding", p1_reb, p2_reb)
    add_edge("Playmaking", p1_ast, p2_ast)
    add_edge("Efficiency", p1_ts, p2_ts, fmt="{:.1f}%")
    add_edge("Rim Protection", p1_blk, p2_blk)
    add_edge("Disruption", p1_stl, p2_stl)
    add_edge("Ball Security", p1_tov, p2_tov, higher_is_better=False)

    sorted_edges = [c for c in categories if c["Winner"] != "Even"]
    summary = []
    if sorted_edges:
        summary.append(sorted_edges[0]["Detail"])
    if len(sorted_edges) > 1:
        summary.append(sorted_edges[1]["Detail"])
    if p1_fg is not None and p2_fg is not None:
        fg_winner = p1 if p1_fg > p2_fg else p2
        summary.append(f"{fg_winner} has the better field-goal mark in these matchups ({p1_fg:.1f}% vs {p2_fg:.1f}%).")

    return {
        "edges": categories,
        "summary": summary[:3],
        "snapshot": {
            p1: {"PTS": p1_pts, "REB": p1_reb, "AST": p1_ast, "TS%": p1_ts, "FG%": p1_fg, "3P%": p1_3p, "FT%": p1_ft},
            p2: {"PTS": p2_pts, "REB": p2_reb, "AST": p2_ast, "TS%": p2_ts, "FG%": p2_fg, "3P%": p2_3p, "FT%": p2_ft},
        },
    }


def _safe_latest_value(df: pd.DataFrame, col: str) -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    value = pd.to_numeric(df.iloc[-1].get(col), errors="coerce")
    return None if pd.isna(value) else float(value)


def _latest_player_value(item: dict, col: str) -> float | None:
    for frame_key in ("adv", "chart_src", "raw_pg", "raw_t"):
        frame = item.get(frame_key)
        val = _safe_latest_value(frame, col) if isinstance(frame, pd.DataFrame) else None
        if val is not None:
            return val
    return None


def _derive_two_point_profile(item: dict) -> tuple[float | None, float | None]:
    fg_pct = _latest_player_value(item, "FG%")
    three_pct = _latest_player_value(item, "3P%")
    three_pa = _latest_player_value(item, "3PA/G")
    two_pa = _latest_player_value(item, "2PA/G")

    if fg_pct is None or three_pct is None or three_pa is None or two_pa is None:
        return None, two_pa

    total_fga_pg = two_pa + three_pa
    if total_fga_pg <= 0 or two_pa <= 0:
        return None, two_pa

    fgm_pg = (fg_pct / 100.0) * total_fga_pg
    fg3m_pg = (three_pct / 100.0) * three_pa
    twopm_pg = fgm_pg - fg3m_pg
    two_pct = (twopm_pg / two_pa) * 100.0 if two_pa > 0 else None

    if two_pct is None:
        return None, two_pa

    return max(min(two_pct, 100.0), 0.0), two_pa


def _metric_winner(player_frames: list[dict], col: str, label: str, *, fmt: str = "{:.1f}") -> dict | None:
    rows = []
    for item in player_frames:
        val = _latest_player_value(item, col)
        if val is None:
            continue
        rows.append((item["name"], val))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_value = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_value - runner_up[1] if runner_up is not None else None
    why = f"{winner_name} leads this group in {col} at {fmt.format(winner_value)}"
    if runner_up is not None:
        why += f", ahead of {runner_up[0]} ({fmt.format(runner_up[1])})."
    else:
        why += "."
    return {
        "label": label,
        "winner": winner_name,
        "value": winner_value,
        "display": fmt.format(winner_value),
        "margin": margin,
        "why": why,
    }


def _defensive_winner(player_frames: list[dict]) -> dict | None:
    rows = []
    for item in player_frames:
        blk = _latest_player_value(item, "BLK/G") or 0.0
        stl = _latest_player_value(item, "STL/G") or 0.0
        trb = _latest_player_value(item, "TRB%") or 0.0
        drb = _latest_player_value(item, "DRB%") or 0.0
        score = (blk * 2.2) + (stl * 1.8) + (trb * 0.12) + (drb * 0.08)
        rows.append((item["name"], score))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_value = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_value - runner_up[1] if runner_up is not None else None
    why = f"{winner_name} grades out best from the blended defensive score"
    if runner_up is not None:
        why += f", beating {runner_up[0]} by {margin:.1f}."
    else:
        why += "."
    why += " It weighs steals, blocks, total rebound %, and defensive rebound %."
    return {
        "label": "Best Defender",
        "winner": winner_name,
        "value": winner_value,
        "display": f"{winner_value:.1f}",
        "margin": margin,
        "why": why,
    }


def _shooting_winner(player_frames: list[dict]) -> dict | None:
    rows = []
    for item in player_frames:
        three_pct = _latest_player_value(item, "3P%")
        three_pa = _latest_player_value(item, "3PA/G")
        ft_pct = _latest_player_value(item, "FT%") or 0.0
        ts = _latest_player_value(item, "TS%") or 0.0
        two_pct, two_pa = _derive_two_point_profile(item)
        if three_pct is None or three_pa is None:
            continue

        if two_pct is None:
            continue

        volume_factor = min(max(three_pa, 0.0), 8.0) / 8.0
        two_volume_factor = min(max(two_pa, 0.0), 12.0) / 12.0
        score = (
            (three_pct * (0.32 + 0.18 * volume_factor)) +
            (two_pct * (0.28 + 0.17 * two_volume_factor)) +
            (three_pa * 1.7) +
            (two_pa * 0.7) +
            (ft_pct * 0.18) +
            (ts * 0.12)
        )
        rows.append((item["name"], score, three_pct, three_pa, two_pct, two_pa, ft_pct, ts))

    if not rows:
        return _metric_winner(player_frames, "3P%", "Best Shooter", fmt="{:.1f}%")

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_score, winner_pct, winner_pa, winner_two_pct, winner_two_pa, winner_ft_pct, winner_ts = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_score - runner_up[1] if runner_up is not None else None
    why = (
        f"{winner_name} wins on an all-around shooting score built from 2P%, 2PA volume, 3P%, 3PA volume, FT%, and TS%. "
        f"They're at {winner_two_pct:.1f}% on twos and {winner_pct:.1f}% from three on {winner_pa:.1f} threes per game"
    )
    why += f", with {winner_ft_pct:.1f}% from the line and {winner_ts:.1f}% TS."
    if runner_up is not None:
        why += f" That puts them ahead of {runner_up[0]} in the full shooting profile."

    return {
        "label": "Best Shooter",
        "winner": winner_name,
        "value": winner_score,
        "display": f"{winner_two_pct:.1f}% 2P | {winner_pct:.1f}% 3P",
        "margin": margin,
        "why": why,
    }


def _three_point_winner(player_frames: list[dict]) -> dict | None:
    rows = []
    for item in player_frames:
        three_pct = _latest_player_value(item, "3P%")
        three_pa = _latest_player_value(item, "3PA/G")
        if three_pct is None or three_pa is None:
            continue

        volume_factor = min(max(three_pa, 0.0), 9.0) / 9.0
        score = (three_pct * (0.65 + 0.35 * volume_factor)) + (three_pa * 2.6)
        rows.append((item["name"], score, three_pct, three_pa))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_score, winner_pct, winner_pa = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_score - runner_up[1] if runner_up is not None else None
    why = f"{winner_name} wins the 3-point shooting card with {winner_pct:.1f}% from deep on {winner_pa:.1f} attempts per game."
    if runner_up is not None:
        why += f" That beats {runner_up[0]} once volume and accuracy are weighed together."

    return {
        "label": "Best 3-Point Shooter",
        "winner": winner_name,
        "value": winner_score,
        "display": f"{winner_pct:.1f}% | {winner_pa:.1f} 3PA/G",
        "margin": margin,
        "why": why,
    }


def _two_point_winner(player_frames: list[dict]) -> dict | None:
    rows = []
    for item in player_frames:
        two_pct, two_pa = _derive_two_point_profile(item)
        ts = _latest_player_value(item, "TS%") or 0.0
        midrange_pts_share = _latest_player_value(item, "PCT_PTS_MIDRANGE_2PT") or 0.0
        pts_2pt_share = _latest_player_value(item, "PCT_PTS_2PT") or 0.0
        if two_pct is None or two_pa is None:
            continue

        if two_pa <= 0:
            continue
        score = (
            (two_pct * (0.60 + 0.20 * min(two_pa, 12.0) / 12.0)) +
            (two_pa * 1.2) +
            (midrange_pts_share * 0.55) +
            (pts_2pt_share * 0.18) +
            (ts * 0.06)
        )
        rows.append((item["name"], score, two_pct, two_pa, midrange_pts_share))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_score, winner_pct, winner_pa, winner_mid_share = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_score - runner_up[1] if runner_up is not None else None
    label = "Best Midrange / 2-Point Shooter"
    why = f"{winner_name} wins this card with {winner_pct:.1f}% on {winner_pa:.1f} two-point attempts per game."
    if winner_mid_share > 0:
        why += f" balldontlie also shows {winner_mid_share:.1f}% of their points coming from midrange twos, so this verdict leans toward real midrange involvement."
    else:
        why += " This uses 2-point efficiency and volume because no direct midrange share was available for that row."
    if runner_up is not None:
        why += f" It puts them ahead of {runner_up[0]} once 2-point efficiency, volume, and shot profile are combined."

    return {
        "label": label,
        "winner": winner_name,
        "value": winner_score,
        "display": f"{winner_pct:.1f}% | {winner_pa:.1f} 2PA/G",
        "margin": margin,
        "why": why,
    }


def _render_comparison_verdict_cards(player_frames: list[dict]) -> None:
    verdicts = [
        _metric_winner(player_frames, "PPG", "Best Scorer"),
        _metric_winner(player_frames, "APG", "Best Playmaker"),
        _metric_winner(player_frames, "RPG", "Best Rebounder"),
        _metric_winner(player_frames, "TS%", "Most Efficient", fmt="{:.1f}%"),
        _shooting_winner(player_frames),
        _three_point_winner(player_frames),
        _two_point_winner(player_frames),
        _defensive_winner(player_frames),
        _metric_winner(player_frames, "BLK/G", "Best Rim Protector"),
    ]
    verdicts = [v for v in verdicts if v is not None]
    if not verdicts:
        return

    st.markdown("### 🏁 Comparison Verdict Cards")
    st.caption("Quick winners from the latest loaded season view.")
    cols = st.columns(min(3, len(verdicts)))
    for idx, verdict in enumerate(verdicts):
        with cols[idx % len(cols)]:
            delta = f"+{verdict['margin']:.1f}" if verdict["margin"] is not None else None
            if delta and "%" in verdict["display"]:
                delta += " pts"
            st.metric(verdict["label"], verdict["winner"], delta=delta)
            st.caption(f"{verdict['display']} in the current compare view")
            if verdict.get("why"):
                st.caption(verdict["why"])


def _render_percentile_compare_view(player_frames: list[dict]) -> None:
    rows = []
    focus_metrics = ["PPG", "RPG", "APG", "TS%", "3P%", "USG%", "AST%", "TRB%", "BLK/G", "STL/G"]

    for item in player_frames:
        adv = item["adv"]
        if adv is None or adv.empty or "SEASON_ID" not in adv.columns:
            continue
        latest_season_id = str(adv.iloc[-1].get("SEASON_ID", ""))
        pct_df = compute_player_percentile_context(item["name"], latest_season_id, adv)
        if pct_df is None or pct_df.empty:
            continue

        pct_df = pct_df[pct_df["Metric"].isin(focus_metrics)].copy()
        for _, row in pct_df.iterrows():
            pct = pd.to_numeric(row.get("Percentile"), errors="coerce")
            if pd.isna(pct):
                continue
            rows.append({
                "Player": item["name"],
                "Metric": row["Metric"],
                "Percentile": float(pct),
                "Value": row.get("Value"),
                "Rank / Field": f"{int(row['Rank'])} / {int(row['Of'])}" if pd.notna(row.get("Rank")) and pd.notna(row.get("Of")) else "—",
            })

    if not rows:
        return

    percentile_df = pd.DataFrame(rows)
    st.markdown("### 📈 Percentile Compare View")
    st.caption("Latest-season percentile context so the comparison is easier to read than raw stats alone.")

    pivot = percentile_df.pivot(index="Metric", columns="Player", values="Percentile").reset_index()
    player_cols = [c for c in pivot.columns if c != "Metric"]
    render_html_table(pivot, number_cols=player_cols, max_height_px=320)

    with st.expander("Percentile details", expanded=False):
        render_html_table(
            percentile_df[["Player", "Metric", "Value", "Percentile", "Rank / Field"]],
            number_cols=["Value"],
            percent_cols=["Percentile"],
            max_height_px=360,
        )


# -------------------------
# Main render function
# -------------------------
def render_compare_tab(primary_player: dict, model=None):
    st.subheader("Compare Players")

    if "compare_players" not in st.session_state:
        st.session_state["compare_players"] = []

    st.session_state["compare_players"] = [
        p for p in st.session_state["compare_players"]
        if _player_key(p) != _player_key(primary_player)
    ]
    st.session_state["compare_players"] = _dedupe_players(st.session_state["compare_players"])

    add_name = st.text_input("Add another player's name to compare:", key="compare_add_name")
    add_col, clear_col = st.columns(2)
    with add_col:
        if st.button("Add Player", use_container_width=True):
            found = _find_player_by_name(add_name)
            if not found:
                st.error("No player found for that search.")
            elif _player_key(found) == _player_key(primary_player):
                st.info("That player is already the main player in this view.")
            else:
                st.session_state["compare_players"] = _dedupe_players(
                    st.session_state["compare_players"] + [found]
                )[:4]
                st.rerun()
    with clear_col:
        if st.button("Clear Comparison Pool", use_container_width=True):
            st.session_state["compare_players"] = []
            st.rerun()

    compare_pool = _dedupe_players([primary_player] + st.session_state["compare_players"])[:5]
    st.caption("Compare up to 5 players. Head-to-head stays available when exactly 2 players are selected.")
    st.info("Share this comparison by copying the browser URL. The selected players and active compare view stay in the link.")

    if len(compare_pool) < 2:
        st.info("Add at least one more player to start the comparison.")
        return

    st.success("Comparing " + " vs ".join([f"**{p['full_name']}**" for p in compare_pool]))

    st.markdown("**Selected players**")
    remove_cols = st.columns(len(compare_pool))
    for idx, player in enumerate(compare_pool):
        with remove_cols[idx]:
            st.caption(player["full_name"])
            if idx == 0:
                st.caption("Primary player")
            else:
                if st.button("Remove", key=f"remove_compare_{_player_key(player)}", use_container_width=True):
                    st.session_state["compare_players"] = [
                        p for p in st.session_state["compare_players"]
                        if _player_key(p) != _player_key(player)
                    ]
                    st.rerun()

    card_cols = st.columns(len(compare_pool))
    for col, player in zip(card_cols, compare_pool):
        with col:
            headshot = get_nba_headshot_url(
                player["id"],
                player_name=player.get("full_name"),
                player_source=player.get("source"),
            )
            if headshot:
                st.image(headshot, width=150)
            st.markdown(f"**{player['full_name']}**")
            try:
                info = get_player_info(
                    player["id"],
                    player_name=player.get("full_name"),
                    player_source=player.get("source"),
                )
                st.caption(f"{info.loc[0, 'TEAM_NAME']} • {info.loc[0, 'POSITION']}")
            except Exception:
                st.caption("Player bio unavailable right now")

    comp_speed_mode = st.toggle(
        "Compute advanced for ALL seasons in comparison (slower)",
        value=False,
        help="Off = chart from per-game (fast, multi-season). On = advanced for all seasons (slower).",
        key="compare_speed_toggle"
    )

    player_frames = []
    with st.spinner("Loading comparison data…"):
        for player in compare_pool:
            raw_pg = _nzdf(get_player_career(
                player["id"], per_mode="PerGame",
                player_name=player.get("full_name"),
                player_source=player.get("source"),
                all_seasons=comp_speed_mode,
            ))
            raw_t = _nzdf(get_player_career(
                player["id"], per_mode="Totals",
                player_name=player.get("full_name"),
                player_source=player.get("source"),
                all_seasons=comp_speed_mode,
            ))

            if comp_speed_mode:
                adv_src = raw_t
            else:
                latest = raw_t["SEASON_ID"].iloc[-1] if not raw_t.empty else None
                adv_src = raw_t[raw_t["SEASON_ID"] == latest] if latest else raw_t

            if not adv_src.empty and adv_src.attrs.get("provider") == "balldontlie":
                adv = adv_src.copy()
            else:
                adv = compute_full_advanced_stats(adv_src) if not adv_src.empty else pd.DataFrame()

            adv = add_per_game_columns(adv, raw_pg)
            adv = _add_season_start(adv)
            chart_src = adv.copy() if comp_speed_mode else _add_season_start(raw_pg.copy())
            ctx_src = adv if not adv.empty else raw_pg
            ctx = _ensure_ctx_dict(compact_player_context(ctx_src) if ctx_src is not None and not ctx_src.empty else {})
            ctx = {str(k).lower(): v for k, v in ctx.items()}

            player_frames.append({
                "player": player,
                "name": player["full_name"],
                "raw_pg": raw_pg,
                "raw_t": raw_t,
                "adv": adv,
                "chart_src": chart_src,
                "ctx": ctx,
            })

    align_mode = st.radio(
        "Align seasons by",
        ["Calendar (overlap only)", "Career year", "Age"],
        horizontal=True,
        key="cmp_align_mode"
    )
    era_toggle = st.checkbox("Era-adjust (stat+ vs league avg = 100)", value=False, key="era_adjust_toggle")

    if align_mode == "Age":
        for item in player_frames:
            birth_year = _get_birth_year(item["player"]["id"])
            item["chart_src"] = _add_age_column(item["chart_src"], birth_year)

    year_sets = [
        set(item["chart_src"]["SEASON_START"].unique().tolist())
        for item in player_frames
        if item["chart_src"] is not None and not item["chart_src"].empty and "SEASON_START" in item["chart_src"].columns
    ]
    common_years = sorted(set.intersection(*year_sets)) if year_sets else []

    if align_mode == "Calendar (overlap only)":
        if not common_years:
            st.warning("No overlapping calendar seasons across the selected players. Try 'Career year' or 'Age' alignment.")
        elif len(common_years) == 1:
            y = common_years[0]
            st.caption(f"Only one overlapping season ({y}-{str(y+1)[-2:]}).")
            for item in player_frames:
                if "SEASON_START" in item["chart_src"].columns:
                    item["chart_src"] = item["chart_src"].loc[item["chart_src"]["SEASON_START"].eq(y)].copy()
        else:
            with st.expander("🔧 Filter seasons to compare (calendar)", expanded=False):
                yr_lo, yr_hi = st.slider(
                    "Season start year range",
                    min_value=min(common_years),
                    max_value=max(common_years),
                    value=(min(common_years), max(common_years)),
                    step=1,
                    key="cmp_season_range_slider",
                )
            for item in player_frames:
                if "SEASON_START" in item["chart_src"].columns:
                    item["chart_src"] = item["chart_src"].loc[
                        item["chart_src"]["SEASON_START"].between(yr_lo, yr_hi)
                    ].copy()

    shared_stats = _pick_shared_stats_for_many([item["chart_src"] for item in player_frames])
    if not shared_stats:
        st.warning("No shared numeric stats available to compare.")
        st.stop()

    default_name = "PPG" if "PPG" in shared_stats else ("PTS" if "PTS" in shared_stats else shared_stats[0])
    stat_choice = st.selectbox(
        "📊 Choose a stat to compare:",
        shared_stats,
        index=shared_stats.index(default_name) if default_name in shared_stats else 0,
        key="cmp_stat_choice",
    )

    supported_plus = {"TS%","EFG%","FG%","3P%","FT%"}
    stat_for_align = stat_choice
    label_suffix = ""
    if era_toggle and stat_choice in supported_plus:
        all_seasons = sorted({
            int(season)
            for item in player_frames
            for season in (item["chart_src"]["SEASON_START"].unique().tolist() if "SEASON_START" in item["chart_src"].columns else [])
        })
        league_tbl = compute_league_shooting_table([f"{y}-{str(y+1)[-2:]}" for y in all_seasons])
        for item in player_frames:
            item["chart_src"] = item["chart_src"].merge(
                league_tbl[["SEASON_START", stat_choice]].rename(columns={stat_choice: "LEAGUE_BASE"}),
                on="SEASON_START",
                how="left",
            )
            item["chart_src"][f"{stat_choice}+"] = np.where(
                item["chart_src"]["LEAGUE_BASE"] > 0,
                100.0 * pd.to_numeric(item["chart_src"][stat_choice], errors="coerce") / item["chart_src"]["LEAGUE_BASE"],
                np.nan,
            )
            item["chart_src"].drop(columns=["LEAGUE_BASE"], errors="ignore", inplace=True)
        stat_for_align = f"{stat_choice}+"
        label_suffix = "+"
    elif era_toggle and stat_choice not in supported_plus:
        st.info("Era-adjust currently supports TS%, EFG%, FG%, 3P%, FT%. Showing raw values.")

    aligned = _build_multi_aligned_df(player_frames, stat_for_align, align_mode)
    if aligned.empty:
        st.warning("No aligned values are available for that comparison view right now.")
    else:
        x_label = {"Calendar (overlap only)": "Season", "Career year": "Career Year", "Age": "Age (approx)"}[align_mode]
        title = f"{stat_choice}{label_suffix} — {align_mode}"
        fig = px.line(aligned, x="X", y="Value", color="Player", markers=True, title=title)
        fig.update_layout(xaxis_title=x_label, yaxis_title=f"{stat_choice}{label_suffix}", legend_title="Player")
        st.plotly_chart(fig, use_container_width=True)

    _render_comparison_verdict_cards(player_frames)
    _render_percentile_compare_view(player_frames)

    if len(player_frames) == 2:
        st.markdown("## ⚔️ Head-to-Head")
        h2h_season_type = st.radio(
            "Games to include",
            ["Regular Season", "Playoffs"],
            horizontal=True,
            key="h2h_season_type"
        )
        pf1, pf2 = player_frames
        years1 = set(pf1["raw_pg"]["SEASON_START"].unique().tolist()) if not pf1["raw_pg"].empty and "SEASON_START" in pf1["raw_pg"].columns else set()
        years2 = set(pf2["raw_pg"]["SEASON_START"].unique().tolist()) if not pf2["raw_pg"].empty and "SEASON_START" in pf2["raw_pg"].columns else set()
        h2h_years = sorted(years1 & years2)

        if not h2h_years:
            st.info("These players never shared an NBA season, so there are no head-to-head games.")
        else:
            if len(h2h_years) == 1:
                yr_lo, yr_hi = h2h_years[0], h2h_years[0]
            else:
                with st.expander("🔧 Filter head-to-head by season range (calendar)", expanded=False):
                    yr_lo, yr_hi = st.slider(
                        "Season start year range",
                        min_value=min(h2h_years),
                        max_value=max(h2h_years),
                        value=(min(h2h_years), max(h2h_years)),
                        step=1,
                        key="h2h_year_range",
                    )
            season_ids = [f"{y}-{str(y+1)[-2:]}" for y in h2h_years if y >= yr_lo and y <= yr_hi]
            force_refetch = st.checkbox("🔁 Force re-fetch (bypass cache; slower)", value=False, key="h2h_force")
            with st.spinner("Loading head-to-head…"):
                h2h = get_head_to_head_games(
                    pf1["player"]["id"], pf2["player"]["id"],
                    seasons=season_ids,
                    season_type=h2h_season_type,
                    force=int(force_refetch),
                    p1_name=pf1["player"].get("full_name"),
                    p2_name=pf2["player"].get("full_name"),
                    p1_source=pf1["player"].get("source"),
                    p2_source=pf2["player"].get("source"),
                )
            if h2h.empty:
                provider = h2h.attrs.get("provider")
                if provider == "balldontlie":
                    st.info("No head-to-head games were returned from balldontlie for the chosen range / season type.")
                else:
                    st.info("No head-to-head games found in the chosen range / season type.")
            else:
                p1 = pf1["name"]
                p2 = pf2["name"]
                insights = _h2h_matchup_insights(h2h, p1, p2)
                def _avg(col):
                    return float(pd.to_numeric(h2h[col], errors="coerce").mean()) if col in h2h.columns else None
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Games", len(h2h))
                with colB:
                    if "P1_WIN" in h2h.columns:
                        wins = int(h2h["P1_WIN"].sum())
                        st.metric(f"{p1} W-L", f"{wins}-{len(h2h)-wins}")
                with colC:
                    if "PTS_P1" in h2h.columns and "PTS_P2" in h2h.columns:
                        p1_pts = round(_avg("PTS_P1") or 0, 1)
                        p2_pts = round(_avg("PTS_P2") or 0, 1)
                        st.metric("PPG (H2H)", f"{p1_pts} vs {p2_pts}", delta=f"{(p1_pts - p2_pts):+.1f}")

                if insights:
                    st.markdown("### Matchup Insights")
                    summary = insights.get("summary", [])
                    if summary:
                        for line in summary:
                            st.write(f"- {line}")

                    edge_df = pd.DataFrame(insights.get("edges", []))
                    if not edge_df.empty:
                        render_html_table(
                            edge_df[["Category", "Winner", "Detail"]],
                            max_height_px=260,
                        )

                    snapshot = insights.get("snapshot", {})
                    if snapshot:
                        snap_rows = []
                        for player_name, stats in snapshot.items():
                            row = {"Player": player_name}
                            row.update(stats)
                            snap_rows.append(row)
                        snapshot_df = pd.DataFrame(snap_rows)
                        render_html_table(
                            snapshot_df,
                            number_cols=["PTS", "REB", "AST"],
                            percent_cols=["TS%", "FG%", "3P%", "FT%"],
                            max_height_px=220,
                        )

                chart_keys = ["PTS","REB","AST","STL","BLK","TOV"]
                plot_df = pd.DataFrame({
                    "Stat": chart_keys,
                    p1: [round(_avg(f"{k}_P1") or 0.0, 2) for k in chart_keys],
                    p2: [round(_avg(f"{k}_P2") or 0.0, 2) for k in chart_keys],
                })
                fig = px.bar(plot_df.melt(id_vars=["Stat"], var_name="Player", value_name="Value"),
                             x="Stat", y="Value", color="Player", barmode="group",
                             title=f"Head-to-Head Averages — {h2h_season_type}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Head-to-head is available when exactly 2 players are selected.")

    st.subheader("📊 Advanced Stats")
    st.caption("All seasons (slower)." if comp_speed_mode else "Latest season only (fast). Toggle above for full career.")
    tabs = st.tabs([item["name"] for item in player_frames])
    for tab, item in zip(tabs, player_frames):
        with tab:
            show_df = item["adv"][metric_public_cols(item["adv"])] if not item["adv"].empty else item["adv"]
            show_df, nums, pcts = _make_readable_stats_table(show_df)
            render_html_table(show_df, number_cols=nums, percent_cols=pcts, max_height_px=420)

    with st.expander("💡 Comparison Question Ideas for these players", expanded=False):
        topic_map = {
            "Overview": "balanced; efficiency, usage, passing, rebounding, trends",
            "Scoring & Efficiency": "TS%, eFG%, PPS, 3PAr, FTr, PTS/36, shooting splits",
            "Playmaking & TOs": "AST%, AST/TO, TOV trend, usage vs passing load",
            "Rebounding & Defense": "TRB%, ORB%, DRB%, STL/36, BLK/36",
            "Peak & Trends": "best/worst seasons, YoY changes, prime window",
        }
        preset = st.radio("Quick presets", list(topic_map.keys()), horizontal=True, key="compare_idea_preset")
        topic = st.text_input("Optional focus (refines suggestions):", value=topic_map[preset], key="compare_idea_focus")
        if len(player_frames) == 2:
            p1 = player_frames[0]["name"]
            p2 = player_frames[1]["name"]
            c1 = player_frames[0]["ctx"]
            c2 = player_frames[1]["ctx"]
            try:
                ideas_cmp = cached_ai_compare_question_ideas(p1, p2, c1, c2, topic, use_model=(model is not None), _model=model)
            except Exception as e:
                st.warning(f"Ideas generator hiccup: {e}")
                ideas_cmp = []
        else:
            ideas_cmp = _seed_multi_compare_questions([item["name"] for item in player_frames])
        st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
        for i in range(0, len(ideas_cmp), 2):
            cols = st.columns(min(2, len(ideas_cmp) - i))
            for col, idea in zip(cols, ideas_cmp[i:i+2]):
                short = abbrev(idea, 40)
                with col:
                    if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"cmp_idea_btn_{i}_{short}"):
                        st.session_state["ai_compare_question"] = idea
                        smooth_scroll_to(make_anchor("compare_ai_anchor"))
                        st.rerun()

    anchor = make_anchor("compare_ai_anchor")
    with st.expander("🧠 Ask the AI Assistant about these players", expanded=False):
        if model:
            q2 = st.text_input("Ask something about these players:", key="ai_compare_question")
            if q2:
                summaries = []
                for item in player_frames:
                    summary = generate_player_summary(
                        item["name"],
                        item["raw_pg"] if not item["raw_pg"].empty else item["adv"],
                        item["adv"],
                    )
                    summaries.append(f"{item['name']}:\n{summary}")
                prompt2 = (
                    "You are an expert NBA analyst. Compare the selected players strictly using the provided season tables. "
                    "Give a fuller answer, use specific stats to support claims, and when helpful rank the players for the question being asked. "
                    "Lean on TS%, eFG%, PPS, 3PAr, FTr, USG% (true), AST%, TRB%, per-game output, and per-36 trends. "
                    "If data is missing for a metric, acknowledge it and use available proxies.\n\n"
                    + "\n\n".join(summaries)
                    + f"\n\nQuestion: {q2}"
                )
                with st.spinner("Analyzing…"):
                    try:
                        text = ai_generate_text(model, prompt2, max_output_tokens=4096, temperature=0.6)
                        st.markdown("### 🧠 AI Analysis")
                        st.write(text or "No response.")
                    except Exception as e:
                        st.warning(_friendly_ai_error_message(e))
                        st.caption(f"Details: {type(e).__name__}")
        else:
            st.info("Add your OpenAI API key to enable AI analysis.")
