# ui_compare.py
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
import html
import re
from config import ai_generate_text, AI_SETUP_ERROR
try:
    from streamlit_searchbox import st_searchbox
except Exception:
    st_searchbox = None

# --- project imports
from fetch import get_player_career, get_head_to_head_games, get_player_info, get_player_birthdate, search_players, get_nba_headshot_url, get_placeholder_headshot_data_uri, get_team_record_for_season
from metrics import (
    compute_full_advanced_stats,
    add_per_game_columns,
    build_ai_phase_table,
    generate_player_summary,
    compact_player_context,
    compute_player_percentile_context,
    detect_player_archetype,
    metric_public_cols,          # returns a filtered/ordered list of columns for display
    order_columns_for_display,
    compute_league_shooting_table    # preferred display ordering (PPG/RPG/APG → TS% → etc.)
)
from ideas import cached_ai_compare_question_ideas, ai_detect_career_phases
from utils import abbrev, make_anchor, smooth_scroll_to
from datetime import datetime


_COMPARE_SEARCHBOX_STYLE_OVERRIDES = {
    "searchbox": {
        "control": {
            "backgroundColor": "#111827",
            "borderColor": "rgba(255,255,255,0.12)",
            "borderRadius": 12,
            "minHeight": 46,
            "boxShadow": "none",
        },
        "input": {"color": "#f9fafb", "fontSize": 15},
        "placeholder": {"color": "rgba(255,255,255,0.45)", "fontSize": 14},
        "singleValue": {"color": "#f9fafb", "fontSize": 15, "fontWeight": 500},
        "menu": {
            "backgroundColor": "#0f172a",
            "borderRadius": 12,
            "overflow": "hidden",
            "border": "1px solid rgba(255,255,255,0.08)",
            "boxShadow": "0 18px 40px rgba(0,0,0,0.35)",
        },
        "menuList": {"backgroundColor": "#0f172a", "paddingTop": 6, "paddingBottom": 6},
        "option": {"fontSize": 14, "paddingTop": 10, "paddingBottom": 10, "paddingLeft": 12, "paddingRight": 12},
        "noOptionsMessage": {"color": "rgba(255,255,255,0.55)", "fontSize": 13},
    },
    "dropdown": {"fill": "#9ca3af", "width": 22, "height": 22, "rotate": True},
    "clear": {"width": 18, "height": 18, "icon": "cross", "clearable": "always"},
}


STAT_TOOLTIPS = {
    "Age": "Age: the player's approximate age during that season window.",
    "AGE": "Age: the player's approximate age during that season window.",
    "PPG": "Points Per Game: the average number of points a player scores each game.",
    "RPG": "Rebounds Per Game: the average number of rebounds a player grabs each game.",
    "APG": "Assists Per Game: the average number of assists a player records each game.",
    "SPG": "Steals Per Game: the average number of steals a player records each game.",
    "BPG": "Blocks Per Game: the average number of shots a player blocks each game.",
    "STL/G": "Steals Per Game: the average number of steals a player records each game.",
    "BLK/G": "Blocks Per Game: the average number of shots a player blocks each game.",
    "TOV/G": "Turnovers Per Game: the average number of turnovers a player commits each game.",
    "MIN/G": "Minutes Per Game: the average number of minutes a player plays each game.",
    "FG%": "Field Goal Percentage: the share of all field-goal attempts a player makes.",
    "3P%": "Three-Point Percentage: the share of three-point attempts a player makes.",
    "FT%": "Free Throw Percentage: the share of free throws a player makes.",
    "2P%": "Two-Point Percentage: the share of two-point attempts a player makes.",
    "2PA/G": "Two-Point Attempts Per Game: how many two-point shots a player takes each game.",
    "3PA/G": "Three-Point Attempts Per Game: how many threes a player takes each game.",
    "FGM": "Field Goals Made: total made field goals in the selected view.",
    "3PM": "Three-Point Field Goals Made: total made threes in the selected view.",
    "FTM": "Free Throws Made: total made free throws in the selected view.",
    "FTA": "Free Throw Attempts: total free throw attempts in the selected view.",
    "TS%": "True Shooting Percentage: scoring efficiency that combines twos, threes, and free throws into one number.",
    "eFG%": "Effective Field Goal Percentage: field-goal efficiency that gives extra credit for made threes.",
    "USG%": "Usage Percentage: estimate of how much of the offense a player personally finishes while on the floor.",
    "AST%": "Assist Percentage: estimate of the share of teammate field goals a player assisted while on the court.",
    "TRB%": "Total Rebound Percentage: estimate of the share of all available rebounds a player grabbed while on the floor.",
    "ORB%": "Offensive Rebound Percentage: estimate of the share of available offensive rebounds a player grabbed.",
    "DRB%": "Defensive Rebound Percentage: estimate of the share of available defensive rebounds a player grabbed.",
    "AST/TO": "Assist-to-Turnover Ratio: how often a player creates assists relative to how often they turn it over.",
    "3PAr": "Three-Point Attempt Rate: the share of a player's field-goal attempts that come from three.",
    "FTr": "Free Throw Rate: how often a player gets to the foul line relative to field-goal attempts.",
    "PPS": "Points Per Shot: a quick efficiency read on how many points a player produces per field-goal attempt.",
    "Pts/Shot": "Points Per Shot: a quick efficiency read on how many points a player produces per field-goal attempt.",
    "PTS": "Points: total points in the selected view.",
    "REB": "Rebounds: total rebounds in the selected view.",
    "AST": "Assists: total assists in the selected view.",
    "STL": "Steals: total steals in the selected view.",
    "BLK": "Blocks: total blocks in the selected view.",
    "TOV": "Turnovers: total turnovers in the selected view.",
    "GP": "Games Played: number of games included in the selected view.",
    "Season": "The NBA season shown in the row.",
}


def _tooltip_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _label_with_tooltip(label: str) -> str:
    tooltip = STAT_TOOLTIPS.get(label)
    if not tooltip:
        return html.escape(label)
    return (
        f'<span class="stat-tooltip" title="{_tooltip_attr(tooltip)}">'
        f"{html.escape(label)}</span>"
    )


_STAT_TERM_PATTERN = re.compile(
    "|".join(sorted((re.escape(k) for k in STAT_TOOLTIPS.keys()), key=len, reverse=True))
)


def annotate_stat_terms(text: str) -> str:
    escaped = html.escape(str(text))
    for stat in sorted(STAT_TOOLTIPS.keys(), key=len, reverse=True):
        escaped = escaped.replace(html.escape(stat), _label_with_tooltip(stat))
    return escaped


def render_stat_text(text: str, *, small: bool = False) -> None:
    cls = "stat-inline-text stat-inline-text--small" if small else "stat-inline-text"
    st.markdown(
        f"""
        <style>
        .stat-inline-text {{
          color: rgba(255,255,255,0.86);
          line-height: 1.55;
          font-size: 0.97rem;
        }}
        .stat-inline-text--small {{
          color: rgba(255,255,255,0.68);
          font-size: 0.85rem;
        }}
        .stat-inline-text .stat-tooltip {{
          cursor: help;
          text-decoration: underline dotted rgba(255,255,255,0.35);
          text-underline-offset: 3px;
        }}
        </style>
        <div class="{cls}">{annotate_stat_terms(text)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_card(label: str, winner: str, delta: str | None) -> None:
    delta_html = f'<div class="verdict-card-delta">{html.escape(delta)}</div>' if delta else ""
    st.markdown(
        f"""
        <style>
        .verdict-card {{
          padding: 2px 0 0 0;
        }}
        .verdict-card-label {{
          color: rgba(255,255,255,0.68);
          font-size: 0.82rem;
          font-weight: 600;
          margin-bottom: 8px;
        }}
        .verdict-card-winner {{
          color: #f9fafb;
          font-size: 1.18rem;
          font-weight: 700;
          line-height: 1.2;
          word-break: break-word;
          overflow-wrap: anywhere;
          margin-bottom: 8px;
        }}
        .verdict-card-delta {{
          display: inline-block;
          background: rgba(34, 197, 94, 0.14);
          color: #86efac;
          border-radius: 999px;
          padding: 2px 8px;
          font-size: 0.8rem;
          font-weight: 700;
          margin-bottom: 6px;
        }}
        </style>
        <div class="verdict-card">
          <div class="verdict-card-label">{html.escape(label)}</div>
          <div class="verdict-card-winner">{html.escape(winner)}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_sticky_ai_rail_css(anchor_class: str) -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:has(.{anchor_class}) {{
          position: fixed;
          top: 0;
          right: 0;
          width: min(360px, 30vw);
          height: 100vh;
          overflow-y: auto;
          background: #262730;
          border-left: 1px solid rgba(255,255,255,0.08);
          box-shadow: -18px 0 32px rgba(0,0,0,0.22);
          padding: 5.25rem 1rem 1rem 1rem;
          z-index: 40;
        }}
        .{anchor_class} {{
          display: none;
        }}
        div[data-testid="column"]:has(.{anchor_class}) h3,
        div[data-testid="column"]:has(.{anchor_class}) p,
        div[data-testid="column"]:has(.{anchor_class}) label {{
          color: #f9fafb;
        }}
        div[data-testid="column"]:has(.{anchor_class}) .stButton > button {{
          width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_headshot_image(headshot_url: str | None, width: int, alt_text: str) -> None:
    fallback_url = get_placeholder_headshot_data_uri()
    source = headshot_url or fallback_url
    escaped_alt = html.escape(alt_text or "Player headshot")
    st.markdown(
        (
            f'<img src="{source}" alt="{escaped_alt}" '
            f'onerror="this.onerror=null;this.src=\'{fallback_url}\';" '
            f'style="width:{width}px;max-width:100%;aspect-ratio:1/1;object-fit:cover;'
            'border-radius:18px;background:#111827;border:1px solid rgba(255,255,255,0.08);" />'
        ),
        unsafe_allow_html=True,
    )


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


def _player_phase_state_key(player: dict) -> str:
    return f"{player.get('source', 'nba_api')}:{player.get('id')}"


def _weighted_average(df: pd.DataFrame, col: str, weights: pd.Series) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _summarize_stat_slice(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    weights = pd.to_numeric(df.get("GP"), errors="coerce") if "GP" in df.columns else pd.Series(1.0, index=df.index)
    weights = weights.fillna(1.0)

    summary = {
        "PPG": _weighted_average(df, "PPG", weights),
        "RPG": _weighted_average(df, "RPG", weights),
        "APG": _weighted_average(df, "APG", weights),
        "TS%": _weighted_average(df, "TS%", weights),
    }
    return pd.Series(summary)


def _validate_phase_output(phases: dict, seasons_available: list[str]) -> dict:
    if not phases or not seasons_available:
        return {}

    available_set = set(seasons_available)

    def clean_list(xs):
        xs = [x for x in (xs or []) if isinstance(x, str)]
        xs = [x for x in xs if x in available_set]
        ordered = [s for s in seasons_available if s in xs]
        out = []
        for s in ordered:
            if s not in out:
                out.append(s)
        return out

    phases["early"] = clean_list(phases.get("early"))
    phases["prime"] = clean_list(phases.get("prime"))
    phases["late"] = clean_list(phases.get("late"))

    used = set(phases["early"] + phases["prime"] + phases["late"])
    missing = [s for s in seasons_available if s not in used]
    if missing:
        if phases["prime"]:
            first_prime = seasons_available.index(phases["prime"][0])
            last_prime = seasons_available.index(phases["prime"][-1])
            for s in missing:
                idx = seasons_available.index(s)
                if idx < first_prime:
                    phases["early"].append(s)
                elif idx > last_prime:
                    phases["late"].append(s)
                else:
                    phases["prime"].append(s)
        phases["early"] = clean_list(phases["early"])
        phases["prime"] = clean_list(phases["prime"])
        phases["late"] = clean_list(phases["late"])

    peak = phases.get("peak_season")
    if peak not in available_set:
        phases["peak_season"] = phases["prime"][-1] if phases["prime"] else seasons_available[len(seasons_available) // 2]

    try:
        phases["confidence"] = float(phases.get("confidence", 0.5))
    except Exception:
        phases["confidence"] = 0.5
    phases["confidence"] = max(0.0, min(1.0, phases["confidence"]))
    return phases


def _slice_compare_scope(df: pd.DataFrame, scope_label: str, phases: dict | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_label == "Latest Season":
        return df.tail(1).copy()
    if scope_label == "Full Career":
        return df.copy()
    if not phases:
        return df.copy()

    if scope_label == "Peak Season":
        peak_season = str(phases.get("peak_season", "")).strip()
        if not peak_season or "SEASON_ID" not in df.columns:
            return pd.DataFrame()
        return df[df["SEASON_ID"].astype(str) == peak_season].copy()

    phase_key_map = {
        "Early Career": "early",
        "Prime": "prime",
        "Late Career": "late",
    }
    phase_key = phase_key_map.get(scope_label)
    season_list = [str(s) for s in (phases.get(phase_key, []) if phase_key else []) if s]
    if not season_list or "SEASON_ID" not in df.columns:
        return pd.DataFrame()
    return df[df["SEASON_ID"].astype(str).isin(season_list)].copy()


def render_html_table(
    df,
    *,
    rename_map=None,
    number_cols=None,
    percent_cols=None,
    date_cols=None,
    tooltip_map=None,
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
    tooltip_map = tooltip_map or STAT_TOOLTIPS
    for col in df2.columns:
        tooltip = tooltip_map.get(str(col))
        if not tooltip:
            continue
        html = html.replace(
            f">{col}</th>",
            f'>{_label_with_tooltip(str(col))}</th>',
            1,
        )



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
.nice-table .stat-tooltip {{
  cursor: help;
  text-decoration: underline dotted rgba(255,255,255,0.35);
  text-underline-offset: 3px;
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
        "TEAM_RECORD": "Team Record",
        "AGE_APPROX": "Age",

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
        "Season", "Age", "Team", "Team Record",
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

def _get_birth_year(player_id: int, player_name: str | None = None, player_source: str | None = None) -> int | None:
    """Year-only (approx) for season-level age alignment."""
    try:
        birthdate = get_player_birthdate(player_id, player_name=player_name, player_source=player_source)
        if not birthdate:
            return None
        bdate = str(birthdate).split("T")[0]
        return int(bdate[:4])
    except Exception:
        return None

def _add_age_column(df: pd.DataFrame, birth_year: int | None) -> pd.DataFrame:
    """Adds AGE_APPROX = SEASON_START - birth_year."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "SEASON_START" not in out.columns:
        out["AGE_APPROX"] = np.nan
        return out
    if birth_year is None:
        out["AGE_APPROX"] = np.nan
        return out
    out["AGE_APPROX"] = out["SEASON_START"].astype(int) - int(birth_year)
    return out


def _add_team_record_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "TEAM_RECORD" not in out.columns:
        out["TEAM_RECORD"] = pd.NA
    if "TEAM_W" not in out.columns:
        out["TEAM_W"] = pd.NA
    if "TEAM_L" not in out.columns:
        out["TEAM_L"] = pd.NA
    if "TEAM_WIN_PCT" not in out.columns:
        out["TEAM_WIN_PCT"] = pd.NA

    for idx, row in out.iterrows():
        season = row.get("SEASON_ID")
        team_id = row.get("TEAM_ID")
        team_abbrev = row.get("TEAM_ABBREVIATION")
        if not season:
            continue
        record = get_team_record_for_season(str(season), team_id=team_id, team_abbreviation=team_abbrev)
        if not record:
            continue
        out.at[idx, "TEAM_RECORD"] = record.get("record")
        out.at[idx, "TEAM_W"] = record.get("wins")
        out.at[idx, "TEAM_L"] = record.get("losses")
        out.at[idx, "TEAM_WIN_PCT"] = record.get("win_pct")
    return out


def _current_age_for_player(player_id: int, player_name: str | None = None, player_source: str | None = None) -> int | None:
    try:
        birthdate = get_player_birthdate(player_id, player_name=player_name, player_source=player_source)
        if not birthdate:
            return None
        bdate = str(birthdate).split("T")[0]
        birthdate = datetime.strptime(bdate, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except Exception:
        return None

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


def _compare_player_suggestions(searchterm: str) -> list[tuple[str, dict]]:
    if not searchterm or len(searchterm.strip()) < 2:
        return []
    found = search_players(searchterm.strip())[:8]
    out = []
    for player in found:
        meta = []
        if player.get("position"):
            meta.append(player["position"])
        if player.get("team_name"):
            meta.append(player["team_name"])
        meta_text = f" ({' • '.join(meta)})" if meta else ""
        out.append((f"{player['full_name']}{meta_text}", player))
    return out


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


def _build_era_compare_df(player_frames: list[dict], stat_choice: str, align_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_seasons = sorted({
        int(season)
        for item in player_frames
        for season in (
            item["chart_src"]["SEASON_START"].unique().tolist()
            if item["chart_src"] is not None and not item["chart_src"].empty and "SEASON_START" in item["chart_src"].columns
            else []
        )
    })
    if not all_seasons:
        return pd.DataFrame(), pd.DataFrame()

    league_tbl = compute_league_shooting_table([f"{y}-{str(y+1)[-2:]}" for y in all_seasons])
    if league_tbl is None or league_tbl.empty or stat_choice not in league_tbl.columns:
        return pd.DataFrame(), pd.DataFrame()

    adjusted_frames = []
    snapshot_rows = []
    plus_col = f"{stat_choice}+"
    for item in player_frames:
        chart_src = item["chart_src"]
        if chart_src is None or chart_src.empty or stat_choice not in chart_src.columns or "SEASON_START" not in chart_src.columns:
            continue
        working = chart_src.copy()
        working = working.merge(
            league_tbl[["SEASON_START", stat_choice]].rename(columns={stat_choice: "LEAGUE_BASE"}),
            on="SEASON_START",
            how="left",
        )
        working[plus_col] = np.where(
            pd.to_numeric(working["LEAGUE_BASE"], errors="coerce") > 0,
            100.0 * pd.to_numeric(working[stat_choice], errors="coerce") / pd.to_numeric(working["LEAGUE_BASE"], errors="coerce"),
            np.nan,
        )
        working.drop(columns=["LEAGUE_BASE"], errors="ignore", inplace=True)
        adjusted_frames.append({**item, "chart_src": working})

        avg_plus = _safe_mean(working, plus_col)
        latest_plus = pd.to_numeric(working[plus_col], errors="coerce").dropna().iloc[-1] if not working.empty and pd.to_numeric(working[plus_col], errors="coerce").dropna().size else None
        raw_avg = _safe_mean(working, stat_choice)
        snapshot_rows.append({
            "Player": item["name"],
            f"{stat_choice}+": avg_plus,
            f"Latest {stat_choice}+": latest_plus,
            f"Raw {stat_choice}": raw_avg,
        })

    aligned = _build_multi_aligned_df(adjusted_frames, plus_col, align_mode) if adjusted_frames else pd.DataFrame()
    snapshot_df = pd.DataFrame(snapshot_rows)
    if not snapshot_df.empty and f"{stat_choice}+" in snapshot_df.columns:
        snapshot_df[f"{stat_choice}+"] = pd.to_numeric(snapshot_df[f"{stat_choice}+"], errors="coerce")
        snapshot_df = snapshot_df.sort_values(by=f"{stat_choice}+", ascending=False, na_position="last").reset_index(drop=True)
    return aligned, snapshot_df


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
    alias_map = {
        "BLK/G": ["BLK/G", "BPG", "BLK"],
        "STL/G": ["STL/G", "SPG", "STL"],
        "TOV/G": ["TOV/G", "TPG", "TOV"],
        "MIN/G": ["MIN/G", "MPG", "MIN"],
    }
    candidate_cols = alias_map.get(col, [col])

    summary_df = item.get("summary_df")
    if isinstance(summary_df, pd.DataFrame):
        for candidate in candidate_cols:
            val = _safe_latest_value(summary_df, candidate)
            if val is not None:
                return val
    for frame_key in ("adv", "chart_src", "raw_pg", "raw_t"):
        frame = item.get(frame_key)
        if isinstance(frame, pd.DataFrame):
            for candidate in candidate_cols:
                val = _safe_latest_value(frame, candidate)
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
    if len(rows) > 1:
        trailing = ", ".join([f"{name} ({fmt.format(value)})" for name, value in rows[1:]])
        why += f", ahead of {trailing}."
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
        score = (blk * 7.0) + (stl * 5.0) + (trb * 0.01) + (drb * 0.01)
        rows.append((item["name"], score, blk, stl, trb, drb))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    winner_name, winner_value, winner_blk, winner_stl, winner_trb, winner_drb = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = winner_value - runner_up[1] if runner_up is not None else None
    why = (
        f"{winner_name} grades out best from the blended defensive score with "
        f"{winner_blk:.1f} BLK/G, {winner_stl:.1f} STL/G, {winner_trb:.1f} TRB%, and {winner_drb:.1f} DRB%."
    )
    if len(rows) > 1:
        trailing = ", ".join(
            [
                f"{name} ({blk:.1f} BLK/G, {stl:.1f} STL/G, {trb:.1f} TRB%, {drb:.1f} DRB%)"
                for name, _, blk, stl, trb, drb in rows[1:]
            ]
        )
        why += f" That puts them ahead of {trailing}."
    else:
        why += "."
    why += " The formula now leans overwhelmingly on steals and blocks, with rebound percentages only used as a light tiebreaker."
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
    if len(rows) > 1:
        trailing = ", ".join([f"{name} ({score:.1f})" for name, score, *_ in rows[1:]])
        why += f" That puts them ahead of {trailing} in the full shooting profile."

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
    if len(rows) > 1:
        trailing = ", ".join([f"{name} ({pct:.1f}% on {pa:.1f} 3PA/G)" for name, _, pct, pa in rows[1:]])
        why += f" That beats {trailing} once volume and accuracy are weighed together."

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
    if len(rows) > 1:
        trailing = ", ".join([f"{name} ({pct:.1f}% on {pa:.1f} 2PA/G)" for name, _, pct, pa, *_ in rows[1:]])
        why += f" It puts them ahead of {trailing} once 2-point efficiency, volume, and shot profile are combined."

    return {
        "label": label,
        "winner": winner_name,
        "value": winner_score,
        "display": f"{winner_pct:.1f}% | {winner_pa:.1f} 2PA/G",
        "margin": margin,
        "why": why,
    }


def _render_comparison_verdict_cards(player_frames: list[dict], scope_label: str) -> None:
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
    render_stat_text(
        f"Quick winners from the selected compare view ({scope_label}), evaluated across all {len(player_frames)} selected players.",
        small=True,
    )
    st.markdown(
        """
        <style>
        .verdict-card {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.95rem 1rem 0.85rem 1rem;
            min-height: 148px;
            margin-bottom: 0.9rem;
        }
        .verdict-card-label {
            color: rgba(255,255,255,0.72);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
            line-height: 1.2;
        }
        .verdict-card-winner {
            color: #f9fafb;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.25;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            margin-bottom: 0.55rem;
        }
        .verdict-card-delta {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.12);
            color: #86efac;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(min(3, len(verdicts)))
    for idx, verdict in enumerate(verdicts):
        with cols[idx % len(cols)]:
            delta = f"+{verdict['margin']:.1f}" if verdict["margin"] is not None else None
            if delta and "%" in verdict["display"]:
                delta += " pts"
            label_html = html.escape(verdict["label"])
            winner_html = html.escape(verdict["winner"])
            delta_html = f'<div class="verdict-card-delta">↑ {html.escape(delta)}</div>' if delta else ""
            st.markdown(
                f"""
                <div class="verdict-card">
                    <div class="verdict-card-label">{label_html}</div>
                    <div class="verdict-card-winner">{winner_html}</div>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_stat_text(f"{verdict['display']} in the current compare view", small=True)
            if verdict.get("why"):
                render_stat_text(verdict["why"], small=True)


def _render_percentile_compare_view(player_frames: list[dict], scope_label: str) -> None:
    if scope_label != "Latest Season":
        st.markdown("### 📈 Percentile Compare View")
        st.info("Percentile compare is currently shown for Latest Season only, since league percentile context is season-based.")
        return

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
    render_stat_text("Latest-season percentile context so the comparison is easier to read than raw stats alone.", small=True)

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


def _render_visual_overlap_charts(player_frames: list[dict], scope_label: str) -> None:
    st.markdown("### 🎯 Visual Overlap Charts")
    render_stat_text(
        f"A quick visual overlap snapshot for the selected compare view ({scope_label}).",
        small=True,
    )

    radar_metrics = ["PPG", "RPG", "APG", "TS%", "3P%", "BLK/G", "STL/G"]
    radar_rows = []
    for item in player_frames:
        row = {"Player": item["name"]}
        has_any = False
        for metric in radar_metrics:
            value = _latest_player_value(item, metric)
            if value is None or pd.isna(value):
                row[metric] = np.nan
            else:
                row[metric] = float(value)
                has_any = True
        if has_any:
            radar_rows.append(row)

    if not radar_rows:
        st.info("No shared stat profile is available to build overlap charts right now.")
        return

    radar_df = pd.DataFrame(radar_rows)
    normalized = radar_df.copy()
    for metric in radar_metrics:
        col = pd.to_numeric(normalized[metric], errors="coerce")
        valid = col.dropna()
        if valid.empty:
            normalized[metric] = 0.0
            continue
        min_val = valid.min()
        max_val = valid.max()
        if max_val == min_val:
            normalized[metric] = 100.0
        else:
            normalized[metric] = ((col - min_val) / (max_val - min_val) * 100.0).fillna(0.0)

    top_row = normalized.melt(id_vars=["Player"], var_name="Metric", value_name="Score")
    bottom_metrics = ["PPG", "RPG", "APG", "TS%", "3P%"]
    bottom_df = radar_df[["Player"] + bottom_metrics].melt(id_vars=["Player"], var_name="Metric", value_name="Value")

    top_col, bottom_col = st.columns([1.15, 1], gap="large")
    with top_col:
        radar_fig = go.Figure()
        for player in normalized["Player"]:
            player_slice = top_row[top_row["Player"] == player]
            theta = player_slice["Metric"].tolist()
            r = player_slice["Score"].tolist()
            if theta:
                theta = theta + [theta[0]]
                r = r + [r[0]]
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill="toself",
                    name=player,
                    opacity=0.5,
                )
            )
        radar_fig.update_layout(
            title="Player DNA Overlap",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=30, r=30, t=50, b=30),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with bottom_col:
        overlap_fig = px.bar(
            bottom_df,
            x="Metric",
            y="Value",
            color="Player",
            barmode="group",
            title="Core Stat Snapshot",
        )
        overlap_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title="Player")
        st.plotly_chart(overlap_fig, use_container_width=True)


def _render_strengths_weaknesses_matrix(player_frames: list[dict], scope_label: str) -> None:
    categories = [
        ("Scoring", "PPG", 75.0, 35.0),
        ("Efficiency", "TS%", 75.0, 40.0),
        ("Playmaking", "APG", 75.0, 40.0),
        ("Rebounding", "RPG", 75.0, 35.0),
        ("Shooting", "3P%", 70.0, 40.0),
        ("Defense", "BLK/G", 75.0, 35.0),
    ]
    rows = []
    for item in player_frames:
        latest_season_id = str(item["adv"].iloc[-1].get("SEASON_ID", "")) if item.get("adv") is not None and not item["adv"].empty else ""
        pct_df = compute_player_percentile_context(item["name"], latest_season_id, item["adv"]) if latest_season_id else pd.DataFrame()
        pct_lookup = {}
        if pct_df is not None and not pct_df.empty:
            for _, row in pct_df.iterrows():
                metric = row.get("Metric")
                pct = pd.to_numeric(row.get("Percentile"), errors="coerce")
                if metric and pd.notna(pct):
                    pct_lookup[str(metric)] = float(pct)

        strengths = []
        neutral = []
        weaknesses = []
        for label, metric, strong_cutoff, weak_cutoff in categories:
            pct_val = pct_lookup.get(metric)
            if pct_val is None:
                continue
            metric_value = _latest_player_value(item, metric)
            detail = f"{label} ({metric}: {metric_value:.1f})" if metric_value is not None and not pd.isna(metric_value) else label
            if pct_val >= strong_cutoff:
                strengths.append(detail)
            elif pct_val <= weak_cutoff:
                weaknesses.append(detail)
            else:
                neutral.append(detail)

        rows.append({
            "Player": item["name"],
            "Strengths": " | ".join(strengths) if strengths else "—",
            "Neutral": " | ".join(neutral[:4]) if neutral else "—",
            "Weaknesses": " | ".join(weaknesses) if weaknesses else "—",
        })

    if not rows:
        return

    st.markdown("### 🧱 Strengths and Weaknesses Matrix")
    render_stat_text(
        f"A fast read on where each player clearly wins, holds steady, or lags in the selected compare view ({scope_label}).",
        small=True,
    )
    matrix_df = pd.DataFrame(rows)
    render_html_table(matrix_df, max_height_px=340)


def _render_shot_profile_compare(player_frames: list[dict], scope_label: str) -> None:
    rows = []
    for item in player_frames:
        rows.append({
            "Player": item["name"],
            "3PA/G": _latest_player_value(item, "3PA/G"),
            "2PA/G": _latest_player_value(item, "2PA/G"),
            "3P%": _latest_player_value(item, "3P%"),
            "FG%": _latest_player_value(item, "FG%"),
            "FT Rate": _latest_player_value(item, "FTr"),
            "Pts/Shot": _latest_player_value(item, "Pts/Shot") or _latest_player_value(item, "PPS"),
            "TS%": _latest_player_value(item, "TS%"),
            "USG%": _latest_player_value(item, "USG%"),
        })

    shot_df = pd.DataFrame(rows)
    if shot_df.empty:
        return

    st.markdown("### 🎯 Shot Profile / Offensive Style Compare")
    render_stat_text(
        f"How these players get their offense in the selected compare view ({scope_label}).",
        small=True,
    )

    display_df = shot_df.copy()
    render_html_table(
        display_df,
        number_cols=["3PA/G", "2PA/G", "FT Rate", "Pts/Shot"],
        percent_cols=["3P%", "FG%", "TS%", "USG%"],
        max_height_px=280,
    )

    plot_df = shot_df.melt(
        id_vars=["Player"],
        value_vars=["3PA/G", "2PA/G", "FT Rate", "Pts/Shot", "TS%", "USG%"],
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Value"])
    if not plot_df.empty:
        fig = px.bar(
            plot_df,
            x="Metric",
            y="Value",
            color="Player",
            barmode="group",
            title="Offensive Style Snapshot",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title="Player")
        st.plotly_chart(fig, use_container_width=True)


def _render_archetype_compare_view(player_frames: list[dict], scope_label: str) -> None:
    archetypes = []
    for item in player_frames:
        adv = item.get("adv")
        if adv is None or adv.empty:
            continue
        latest_season_id = str(adv.iloc[-1].get("SEASON_ID", ""))
        percentile_df = compute_player_percentile_context(item["name"], latest_season_id, adv) if latest_season_id else pd.DataFrame()
        archetype = detect_player_archetype(item["name"], adv, percentile_df)
        if archetype:
            archetypes.append((item["name"], archetype))

    if not archetypes:
        return

    st.markdown("### 🧩 Archetype Compare")
    render_stat_text(
        f"Role and style identity for the selected compare view ({scope_label}).",
        small=True,
    )
    cols = st.columns(min(3, len(archetypes)))
    for idx, (name, archetype) in enumerate(archetypes):
        with cols[idx % len(cols)]:
            primary = archetype.get("primary", "—")
            secondary = archetype.get("secondary") or "—"
            style_tags = ", ".join(archetype.get("style_tags", [])) or "—"
            impact_tags = ", ".join(archetype.get("impact_tags", [])) or "—"
            primary_desc = archetype.get("primary_description") or ""
            secondary_desc = archetype.get("secondary_description") or ""
            confidence = archetype.get("confidence", 0.0)
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(f"**Primary:** {primary}")
                if primary_desc:
                    st.caption(primary_desc)
                st.write(f"**Secondary:** {secondary}")
                if secondary_desc:
                    st.caption(secondary_desc)
                st.write("**Style Tags:** " + style_tags)
                st.write("**Impact Tags:** " + impact_tags)
                st.caption(f"Confidence: {confidence:.2f}")
            for line in archetype.get("evidence", [])[:2]:
                st.caption(line)


def _build_compare_ai_summaries(player_frames: list[dict]) -> list[str]:
    summaries = []
    for item in player_frames:
        age = _current_age_for_player(
            item["player"]["id"],
            player_name=item["player"].get("full_name"),
            player_source=item["player"].get("source"),
        )
        summary = generate_player_summary(
            item["name"],
            item["raw_pg"] if not item["raw_pg"].empty else item["adv"],
            item["adv"],
        )
        age_line = f"Current age: {age}.\n" if age is not None else ""
        summaries.append(f"{item['name']}:\n{age_line}{summary}")
    return summaries


def _compare_ai_chat_key(player_frames: list[dict]) -> str:
    names = sorted(item["name"] for item in player_frames)
    return "|".join(names)


def _build_full_career_ai_summaries(player_frames: list[dict]) -> list[str]:
    summaries = []
    for item in player_frames:
        age = _current_age_for_player(
            item["player"]["id"],
            player_name=item["player"].get("full_name"),
            player_source=item["player"].get("source"),
        )
        raw_src = item["full_raw_pg"] if item.get("full_raw_pg") is not None and not item["full_raw_pg"].empty else item["full_adv"]
        adv_src = item["full_adv"] if item.get("full_adv") is not None and not item["full_adv"].empty else item["adv"]
        summary = generate_player_summary(
            item["name"],
            raw_src,
            adv_src,
        )
        age_line = f"Current age: {age}.\n" if age is not None else ""
        summaries.append(f"{item['name']}:\n{age_line}{summary}")
    return summaries


def _compare_scouting_report_prompt(player_frames: list[dict], scope_label: str) -> str:
    names = ", ".join(item["name"] for item in player_frames)
    exact_count = len(player_frames)
    summaries = _build_compare_ai_summaries(player_frames)
    return (
        "You are an expert NBA scout and analyst. Build a side-by-side scouting report for the selected players "
        f"using the selected compare view ({scope_label}). Use only the provided player stat summaries. "
        f"You must only discuss these {exact_count} players: {names}. Do not mention, rank, compare, or reference any other NBA players. "
        "For each player, write concise markdown with these exact subheads: Offensive Identity, Shooting Profile, "
        "Playmaking, Defense, Rebounding / Physicality, Weaknesses, Best Team Fit. Use specific stats in every player section. "
        "After the player-by-player sections, add a short 'Best At A Glance' section with quick winners for scoring, playmaking, "
        "shooting, defense, and overall offensive engine. Keep it sharp and readable.\n\n"
        + "\n\n".join(summaries)
        + f"\n\nPlayers: {names}"
    )


def _build_around_prompt(player_frames: list[dict], scope_label: str, lens: str) -> str:
    summaries = _build_full_career_ai_summaries(player_frames)
    names = [item["name"] for item in player_frames]
    joined_names = ", ".join(names)
    exact_count = len(names)
    lens_instruction = {
        "Long-Term Franchise Pick": "prioritize long-term build-around value, sustainable offensive creation, scalable defense, and franchise ceiling",
        "Win-Now Contender": "prioritize immediate high-level impact for a contender right now",
        "Balanced": "balance current value with long-term build-around value",
    }.get(lens, "balance current value with long-term build-around value")
    return (
        "You are an expert NBA team-building analyst. Decide who you would rather build around from this player group. "
        f"Ignore the page's temporary comparison slice and use each player's full career body of work for this decision. {lens_instruction}. "
        f"You must rank exactly these {exact_count} selected players and nobody else: {joined_names}. "
        "Do not mention or rank any players outside this selected group. "
        "Return exactly one numbered ranking entry per selected player, then a Bottom Line section. "
        "Then provide a short verdict for each player with concrete stat reasons. "
        "End with a 'Bottom Line' paragraph naming the best single choice and why. "
        "If age or contract info is missing, say so briefly and lean on the provided stats and archetype-style implications instead. "
        "If you cannot support a claim from the provided player summaries, do not invent extra players or extra data.\n\n"
        + "\n\n".join(summaries)
        + f"\n\nLens: {lens}"
    )


def _compare_what_changed_prompt(player_frames: list[dict]) -> str:
    blocks = []
    for item in player_frames:
        full_adv = item.get("full_adv", pd.DataFrame())
        full_raw_pg = item.get("full_raw_pg", pd.DataFrame())
        if full_adv is None or full_adv.empty:
            continue
        phases = item.get("phases") or {}
        early = ", ".join(phases.get("early", [])) or "—"
        prime = ", ".join(phases.get("prime", [])) or "—"
        late = ", ".join(phases.get("late", [])) or "—"
        peak = phases.get("peak_season") or "—"
        summary = generate_player_summary(
            item["name"],
            full_raw_pg if full_raw_pg is not None and not full_raw_pg.empty else full_adv,
            full_adv,
        )
        blocks.append(
            f"{item['name']}\n"
            f"Career phases:\n"
            f"- Early Career: {early}\n"
            f"- Prime: {prime}\n"
            f"- Late Career: {late}\n"
            f"- Peak Season: {peak}\n\n"
            f"Full-career season summary:\n{summary}"
        )

    joined_names = ", ".join(item["name"] for item in player_frames)
    return (
        "You are an expert NBA development analyst. Compare how these players changed over the course of their careers "
        "using only the provided full-career stats and AI-labeled career phases.\n"
        "This analysis should use full career context even if the visible compare tab is on a smaller view like Latest Season or Prime.\n"
        "Write in markdown with these exact sections: Each Player's Arc, Biggest Growth Areas, Biggest Declines or Shifts, "
        "How Their Evolutions Differ, Most Complete Development Arc, Bottom Line.\n"
        "Use specific stats and phase windows throughout. Focus on role changes, scoring burden, shooting growth, playmaking evolution, "
        "defensive changes, and how their peak versions differ from their early and late-career identities.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nPlayers: {joined_names}"
    )


def _compare_debate_prompt(player_frames: list[dict], scope_label: str, lens: str, custom_focus: str = "") -> str:
    greatest_lenses = {
        "Greatest Overall Career",
        "Highest Peak",
        "Best Prime",
        "Best Legacy / Resume",
        "Most Complete All-Time Player",
    }
    use_full_career = lens in greatest_lenses
    summaries = _build_full_career_ai_summaries(player_frames) if use_full_career else _build_compare_ai_summaries(player_frames)
    names = [item["name"] for item in player_frames]
    joined_names = ", ".join(names)
    exact_count = len(names)
    lens_instruction = {
        "Best Overall Player": "argue who is the strongest all-around player in this selected compare view",
        "Playoff Series": "argue who you would trust most in a playoff series in this selected compare view",
        "Best Offensive Engine": "argue who is the strongest offensive engine in this selected compare view",
        "Best Defensive Piece": "argue who brings the strongest defensive value in this selected compare view",
        "Best Second Star": "argue who scales best as a second star next to another elite player in this selected compare view",
        "Greatest Overall Career": "argue who has the greatest overall NBA career among the selected players",
        "Highest Peak": "argue who reached the highest peak at his best",
        "Best Prime": "argue who sustained the strongest prime across multiple seasons",
        "Best Legacy / Resume": "argue who has the strongest all-time legacy and career resume",
        "Most Complete All-Time Player": "argue who combines peak, longevity, versatility, and overall greatness the best",
    }.get(lens, "argue which player has the strongest case in this selected compare view")
    custom_line = f"Additional debate framing from the user: {custom_focus.strip()}.\n" if custom_focus.strip() else ""
    scope_line = (
        "Use the selected players only and lean on their full career body of work, including peak, prime, longevity, role, and historical stature. "
        if use_full_career
        else "Use only the selected players and only the provided stat summaries from the selected compare view. "
    )
    section_names = (
        "Opening Case, Greatest Case For Each Player, Legacy Counter-Arguments, Greatest Debate Verdict, Bottom Line"
        if use_full_career
        else "Opening Case, Best Case For Each Player, Strongest Counter-Arguments, Debate Verdict, Bottom Line"
    )
    detail_instructions = (
        "In 'Greatest Case For Each Player', give every selected player their own subheading and 2-4 concrete points using the provided career summaries. "
        "In 'Legacy Counter-Arguments', challenge each selected player with 1-3 realistic limitations or gaps. "
        "In 'Greatest Debate Verdict', choose a winner and explain why that player wins this all-time lens. "
        "Keep it sharp, historically aware, and specific."
        if use_full_career
        else
        "In 'Best Case For Each Player', give every selected player their own subheading and 2-4 concrete stat-backed points. "
        "In 'Strongest Counter-Arguments', challenge each selected player with 1-3 stat-backed weaknesses or limitations. "
        "In 'Debate Verdict', choose a winner and explain why they win this exact debate lens. "
        "Keep it readable, direct, and specific."
    )
    return (
        "You are moderating a sharp NBA debate. "
        f"You must only discuss these {exact_count} players: {joined_names}. Do not mention or compare any other NBA players. "
        + scope_line +
        f"{lens_instruction}. "
        f"Write in markdown with these exact sections: {section_names}. "
        + detail_instructions + "\n\n"
        + custom_line
        + "\n\n".join(summaries)
        + f"\n\nPlayers: {joined_names}\nDebate lens: {lens}\nCompare view: {scope_label if not use_full_career else 'Full Career / All-Time Lens'}"
    )


def _compare_debate_chat_key(player_frames: list[dict], scope_label: str, lens: str) -> str:
    names = sorted(item["name"] for item in player_frames)
    greatest_lenses = {
        "Greatest Overall Career",
        "Highest Peak",
        "Best Prime",
        "Best Legacy / Resume",
        "Most Complete All-Time Player",
    }
    scope_key = "full-career-all-time" if lens in greatest_lenses else scope_label
    return f"{scope_key}|{lens}|{'|'.join(names)}"


def render_compare_scouting_report_page() -> None:
    st.markdown("## 🧠 Side-by-Side AI Scouting Report")
    compare_names = st.session_state.get("compare_report_players") or []
    compare_view = st.session_state.get("compare_report_view") or "Compare"
    if compare_names:
        st.caption(f"Players: {', '.join(compare_names)}")
    st.caption(f"Report scope: {compare_view}")

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Compare", key="back_to_compare_from_report", use_container_width=True):
            st.session_state["compare_report_mode"] = None
            st.session_state["requested_active_view"] = "🤝 Compare Players"
            st.rerun()

    report = st.session_state.get("compare_scouting_report_output")
    if report:
        st.markdown(report)
    else:
        st.info("No scouting report is available right now. Generate one from the Compare tab first.")


def render_compare_what_changed_page() -> None:
    st.markdown("## 🔄 What Changed? (Compare Mode)")
    compare_names = st.session_state.get("compare_what_changed_players") or []
    if compare_names:
        st.caption(f"Players: {', '.join(compare_names)}")
    st.caption("Analysis scope: Full career evolution")

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Compare", key="back_to_compare_from_what_changed", use_container_width=True):
            st.session_state["compare_report_mode"] = None
            st.session_state["requested_active_view"] = "🤝 Compare Players"
            st.rerun()

    report = st.session_state.get("compare_what_changed_output")
    if report:
        st.markdown(report)
    else:
        st.info("No compare evolution report is available right now. Generate one from the Compare tab first.")


def render_compare_debate_page(model) -> None:
    st.markdown("## 🥊 AI Debate Mode")
    compare_names = st.session_state.get("compare_debate_players") or []
    compare_view = st.session_state.get("compare_debate_view") or "Compare"
    compare_lens = st.session_state.get("compare_debate_lens") or "Best Overall Player"
    if compare_names:
        st.caption(f"Players: {', '.join(compare_names)}")
    st.caption(f"Debate scope: {compare_view}")
    st.caption(f"Debate lens: {compare_lens}")

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Compare", key="back_to_compare_from_debate", use_container_width=True):
            st.session_state["compare_report_mode"] = None
            st.session_state["requested_active_view"] = "🤝 Compare Players"
            st.rerun()

    if not model:
        st.info("AI is unavailable right now.")
        return

    chat_key = st.session_state.get("compare_debate_chat_key")
    system_prompt = st.session_state.get("compare_debate_system_prompt")
    chat_store = st.session_state.setdefault("compare_debate_conversations", {})
    history = chat_store.get(chat_key, []) if chat_key else []

    if st.button("Clear Debate Conversation", key=f"clear_compare_debate_{chat_key or 'empty'}", use_container_width=False):
        if chat_key:
            chat_store[chat_key] = []
        st.session_state["compare_debate_output"] = None
        st.rerun()

    if history:
        for message in history:
            role = "You" if message["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {message['content']}")
    elif st.session_state.get("compare_debate_output"):
        st.markdown(st.session_state["compare_debate_output"])
    else:
        st.info("No debate is available right now. Generate one from the Compare tab first.")

    if not chat_key or not system_prompt:
        return

    with st.form(key=f"compare_debate_followup_form_{chat_key}", clear_on_submit=True):
        q = st.text_input(
            "Ask a follow-up about this debate:",
            key=f"compare_debate_followup_input_{chat_key}",
        )
        submitted = st.form_submit_button("Send Follow-Up", use_container_width=True)
    if submitted and q:
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": q}]
        with st.spinner("Continuing the debate…"):
            try:
                text = ai_generate_text(model, messages=messages, max_output_tokens=4096, temperature=0.7)
                updated_history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": text or "No response."}]
                chat_store[chat_key] = updated_history[-12:]
                st.rerun()
            except Exception as e:
                st.warning(_friendly_ai_error_message(e))
                st.caption(f"Details: {type(e).__name__}")


def _compare_ai_system_prompt(player_frames: list[dict]) -> str:
    summaries = []
    for item in player_frames:
        summary = generate_player_summary(
            item["name"],
            item["raw_pg"] if not item["raw_pg"].empty else item["adv"],
            item["adv"],
        )
        summaries.append(f"{item['name']}:\n{summary}")
    return (
        "You are an expert NBA analyst. Compare the selected players strictly using the provided season tables. "
        "Give a fuller answer, use specific stats to support claims, and when helpful rank the players for the question being asked. "
        "Lean on TS%, eFG%, PPS, 3PAr, FTr, USG% (true), AST%, TRB%, per-game output, and per-36 trends. "
        "If data is missing for a metric, acknowledge it and use available proxies.\n\n"
        + "\n\n".join(summaries)
    )


def render_compare_ai_chat_page(model) -> None:
    compare_names = st.session_state.get("compare_ai_chat_players") or []
    chat_key = st.session_state.get("compare_ai_chat_key")
    st.markdown("## 💬 AI Compare Chat")
    if compare_names:
        st.caption(f"Players: {', '.join(compare_names)}")

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Compare", key="back_to_compare_from_chat", use_container_width=True):
            st.session_state["compare_report_mode"] = None
            st.session_state["requested_active_view"] = "🤝 Compare Players"
            st.rerun()

    if not model:
        st.info("AI is unavailable right now.")
        return

    if not chat_key:
        st.info("No compare AI chat is available yet. Start one from the Compare tab first.")
        return

    chat_store = st.session_state.setdefault("compare_ai_conversations", {})
    history = chat_store.get(chat_key, [])
    system_prompt = st.session_state.get("compare_ai_system_prompt")

    if st.button("Clear Conversation", key=f"clear_compare_ai_page_{chat_key}", use_container_width=False):
        chat_store[chat_key] = []
        st.rerun()

    if history:
        for message in history:
            role = "You" if message["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {message['content']}")
    else:
        st.caption("Start the conversation below.")

    with st.form(key=f"compare_ai_page_form_{chat_key}", clear_on_submit=True):
        q = st.text_input("Ask a follow-up about these players:", key=f"compare_ai_page_input_{chat_key}")
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and q and system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": q}]
        with st.spinner("Analyzing…"):
            try:
                text = ai_generate_text(model, messages=messages, max_output_tokens=4096, temperature=0.6)
                updated_history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": text or "No response."}]
                chat_store[chat_key] = updated_history[-10:]
                st.rerun()
            except Exception as e:
                st.warning(_friendly_ai_error_message(e))
                st.caption(f"Details: {type(e).__name__}")


# -------------------------
# Main render function
# -------------------------
def render_compare_tab(primary_player: dict, model=None):
    st.subheader("Compare Players")
    max_total_players = 5
    max_added_players = max_total_players - 1

    if "compare_players" not in st.session_state:
        st.session_state["compare_players"] = []

    st.session_state["compare_players"] = [
        p for p in st.session_state["compare_players"]
        if _player_key(p) != _player_key(primary_player)
    ]
    st.session_state["compare_players"] = _dedupe_players(st.session_state["compare_players"])[:max_added_players]
    compare_pool = _dedupe_players([primary_player] + st.session_state["compare_players"])[:max_total_players]
    compare_pool_is_full = len(compare_pool) >= max_total_players

    if st_searchbox is not None:
        if compare_pool_is_full:
            st.caption("Add another player's name to compare")
            st.warning(f"You've reached the max comparison size of {max_total_players} total players. Remove a player to add someone new.")
            st.session_state.pop("compare_player_searchbox", None)
            st.session_state.pop("_last_compare_search_pick", None)
        else:
            selected_compare_player = st_searchbox(
                _compare_player_suggestions,
                label="Add another player's name to compare",
                placeholder="Start typing a player name...",
                key="compare_player_searchbox",
                clear_on_submit=False,
                edit_after_submit="option",
                style_overrides=_COMPARE_SEARCHBOX_STYLE_OVERRIDES,
            )
            if isinstance(selected_compare_player, dict) and selected_compare_player.get("id") is not None:
                selected_key = _player_key(selected_compare_player)
                last_handled_key = st.session_state.get("_last_compare_search_pick")
                if selected_key == last_handled_key:
                    pass
                elif selected_key == _player_key(primary_player):
                    st.session_state["_last_compare_search_pick"] = selected_key
                    st.info("That player is already the main player in this view.")
                else:
                    current_keys = {_player_key(p) for p in st.session_state["compare_players"]}
                    st.session_state["_last_compare_search_pick"] = selected_key
                    if selected_key not in current_keys:
                        st.session_state["compare_players"] = _dedupe_players(
                            st.session_state["compare_players"] + [selected_compare_player]
                        )[:max_added_players]
                        st.rerun()
        clear_col = st.columns(1)[0]
    else:
        add_name = st.text_input("Add another player's name to compare:", key="compare_add_name")
        add_col, clear_col = st.columns(2)
        with add_col:
            if st.button("Add Player", use_container_width=True):
                found = _find_player_by_name(add_name)
                if compare_pool_is_full:
                    st.warning(f"You've reached the max comparison size of {max_total_players} total players. Remove a player to add someone new.")
                elif not found:
                    st.error("No player found for that search.")
                elif _player_key(found) == _player_key(primary_player):
                    st.info("That player is already the main player in this view.")
                else:
                    st.session_state["compare_players"] = _dedupe_players(
                        st.session_state["compare_players"] + [found]
                    )[:max_added_players]
                    st.rerun()
    with clear_col:
        if st.button("Clear Comparison Pool", use_container_width=True):
            st.session_state["compare_players"] = []
            st.rerun()

    compare_pool = _dedupe_players([primary_player] + st.session_state["compare_players"])[:max_total_players]
    st.caption(f"Compare up to {max_total_players} total players ({max_added_players} added plus the primary player). Head-to-head stays available when exactly 2 players are selected.")
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
            _render_headshot_image(headshot, 150, player.get("full_name", "Player"))
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

    player_frames = []
    phase_store = st.session_state.setdefault("career_phases_by_player", {})
    with st.spinner("Loading comparison data…"):
        for player in compare_pool:
            raw_pg = _nzdf(get_player_career(
                player["id"], per_mode="PerGame",
                player_name=player.get("full_name"),
                player_source=player.get("source"),
                all_seasons=True,
            ))
            raw_t = _nzdf(get_player_career(
                player["id"], per_mode="Totals",
                player_name=player.get("full_name"),
                player_source=player.get("source"),
                all_seasons=True,
            ))

            if not raw_t.empty and raw_t.attrs.get("provider") == "balldontlie":
                full_adv = raw_t.copy()
            else:
                full_adv = compute_full_advanced_stats(raw_t) if not raw_t.empty else pd.DataFrame()

            full_adv = add_per_game_columns(full_adv, raw_pg)
            full_adv = _add_season_start(full_adv)

            phase_key = _player_phase_state_key(player)
            phases = phase_store.get(phase_key)
            if model and not phases and full_adv is not None and not full_adv.empty:
                phase_df = build_ai_phase_table(full_adv)
                seasons = phase_df["Season"].dropna().astype(str).tolist() if "Season" in phase_df.columns else []
                if seasons:
                    try:
                        phases = ai_detect_career_phases(
                            player["full_name"],
                            phase_df.to_csv(index=False),
                            use_model=True,
                            _model=model,
                        )
                        phases = _validate_phase_output(phases, seasons)
                        phase_store[phase_key] = phases
                    except Exception:
                        phases = None

            ctx_src = full_adv if not full_adv.empty else raw_pg
            ctx = _ensure_ctx_dict(compact_player_context(ctx_src) if ctx_src is not None and not ctx_src.empty else {})
            ctx = {str(k).lower(): v for k, v in ctx.items()}

            player_frames.append({
                "player": player,
                "name": player["full_name"],
                "full_raw_pg": _add_season_start(raw_pg.copy()) if raw_pg is not None and not raw_pg.empty else pd.DataFrame(),
                "full_raw_t": _add_season_start(raw_t.copy()) if raw_t is not None and not raw_t.empty else pd.DataFrame(),
                "full_adv": full_adv,
                "phases": phases,
                "ctx": ctx,
            })

    compare_view_options = ["Latest Season"]
    if all(item.get("phases") and item["phases"].get("peak_season") for item in player_frames):
        compare_view_options.append("Peak Season")
    phase_option_specs = [
        ("Prime", "prime"),
        ("Early Career", "early"),
        ("Late Career", "late"),
    ]
    for label, phase_key in phase_option_specs:
        if all(item.get("phases") and item["phases"].get(phase_key) for item in player_frames):
            compare_view_options.append(label)
    compare_view_options.append("Full Career")
    preferred_order = ["Latest Season", "Peak Season", "Prime", "Full Career", "Early Career", "Late Career"]
    compare_view_options = [label for label in preferred_order if label in compare_view_options]

    selected_compare_view = st.selectbox(
        "Comparison view",
        compare_view_options,
        index=0,
        key="compare_summary_view",
    )
    compare_view_captions = {
        "Latest Season": "Comparing each player's latest available season.",
        "Peak Season": "Comparing each player's AI-labeled peak season.",
        "Full Career": "Comparing full-career windows, with verdicts using games-weighted averages across each player's full body of work.",
        "Early Career": "Comparing each player's AI-labeled early-career seasons.",
        "Prime": "Comparing each player's AI-labeled prime seasons.",
        "Late Career": "Comparing each player's AI-labeled late-career seasons.",
    }
    render_stat_text(compare_view_captions.get(selected_compare_view, ""), small=True)

    scoped_frames = []
    for item in player_frames:
        scoped_raw_pg = _slice_compare_scope(item["full_raw_pg"], selected_compare_view, item.get("phases"))
        scoped_raw_t = _slice_compare_scope(item["full_raw_t"], selected_compare_view, item.get("phases"))
        scoped_adv = _slice_compare_scope(item["full_adv"], selected_compare_view, item.get("phases"))
        birth_year = _get_birth_year(
            item["player"]["id"],
            player_name=item["player"].get("full_name"),
            player_source=item["player"].get("source"),
        )
        scoped_raw_pg = _add_age_column(_add_season_start(scoped_raw_pg), birth_year)
        scoped_raw_t = _add_age_column(_add_season_start(scoped_raw_t), birth_year)
        scoped_adv = _add_age_column(_add_season_start(scoped_adv), birth_year)
        scoped_raw_pg = _add_team_record_column(scoped_raw_pg)
        scoped_raw_t = _add_team_record_column(scoped_raw_t)
        scoped_adv = _add_team_record_column(scoped_adv)
        chart_src = scoped_adv.copy() if scoped_adv is not None and not scoped_adv.empty else scoped_raw_pg.copy()
        ctx_src = scoped_adv if scoped_adv is not None and not scoped_adv.empty else scoped_raw_pg
        summary_source = scoped_adv if scoped_adv is not None and not scoped_adv.empty else scoped_raw_pg
        summary_row = _summarize_stat_slice(summary_source)
        summary_df = pd.DataFrame([summary_row]) if not summary_row.empty else pd.DataFrame()
        scoped_frames.append({
            **item,
            "raw_pg": scoped_raw_pg,
            "raw_t": scoped_raw_t,
            "adv": scoped_adv,
            "summary_df": summary_df,
            "chart_src": _add_season_start(chart_src) if chart_src is not None and not chart_src.empty else pd.DataFrame(),
            "ctx": _ensure_ctx_dict(compact_player_context(ctx_src) if ctx_src is not None and not ctx_src.empty else {}),
        })
    player_frames = scoped_frames
    compare_signature = selected_compare_view + "|" + "|".join(item["name"] for item in player_frames)
    if st.session_state.get("compare_ai_output_signature") != compare_signature:
        st.session_state.pop("compare_scouting_report_output", None)
        st.session_state.pop("compare_build_around_output", None)
        st.session_state.pop("compare_what_changed_output", None)
        st.session_state.pop("compare_debate_output", None)
        st.session_state.pop("compare_debate_chat_key", None)
        st.session_state.pop("compare_debate_system_prompt", None)
        st.session_state["compare_ai_output_signature"] = compare_signature

    if "show_compare_ai_rail" not in st.session_state:
        st.session_state["show_compare_ai_rail"] = True
    rail_toggle_col_left, rail_toggle_col_right = st.columns([4.5, 1.2])
    with rail_toggle_col_right:
        if st.button(
            "Hide AI sidebar" if st.session_state["show_compare_ai_rail"] else "Show AI sidebar",
            key="toggle_compare_ai_rail",
            use_container_width=True,
        ):
            st.session_state["show_compare_ai_rail"] = not st.session_state["show_compare_ai_rail"]
            st.rerun()

    if st.session_state["show_compare_ai_rail"]:
        _inject_sticky_ai_rail_css("sticky-compare-ai-rail")
        main_col, ai_col = st.columns([3.2, 1.35], gap="large")
    else:
        main_col = st.container()
        ai_col = None

    with main_col:
        st.subheader("📊 Advanced Stats")
        render_stat_text(f"Showing the selected compare view: {selected_compare_view}.", small=True)
        tabs = st.tabs([item["name"] for item in player_frames])
        for tab, item in zip(tabs, player_frames):
            with tab:
                show_df = item["adv"][metric_public_cols(item["adv"])] if not item["adv"].empty else item["adv"]
                show_df, nums, pcts = _make_readable_stats_table(show_df)
                render_html_table(show_df, number_cols=nums, percent_cols=pcts, max_height_px=420)

        _render_archetype_compare_view(player_frames, selected_compare_view)

        align_mode = st.radio(
            "Align seasons by",
            ["Calendar (overlap only)", "Career year", "Age"],
            horizontal=True,
            key="cmp_align_mode"
        )
        era_toggle = st.checkbox("Era-adjust (stat+ vs league avg = 100)", value=False, key="era_adjust_toggle")

        if align_mode == "Age":
            for item in player_frames:
                birth_year = _get_birth_year(
                    item["player"]["id"],
                    player_name=item["player"].get("full_name"),
                    player_source=item["player"].get("source"),
                )
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

        st.markdown("### 🕰️ Era Compare")
        render_stat_text(
            "Compare players against the league they actually played in. A score of 100 means league average for that season, "
            "while numbers above 100 mean the player beat his era on that metric.",
            small=True,
        )
        era_metric = st.selectbox(
            "Era-adjusted metric",
            ["TS%", "EFG%", "FG%", "3P%", "FT%"],
            index=0,
            key="era_compare_metric",
        )
        era_aligned, era_snapshot = _build_era_compare_df(player_frames, era_metric, align_mode)
        if era_aligned.empty:
            st.info("Era compare is unavailable for this view right now.")
        else:
            era_x_label = {"Calendar (overlap only)": "Season", "Career year": "Career Year", "Age": "Age (approx)"}[align_mode]
            era_fig = px.line(
                era_aligned,
                x="X",
                y="Value",
                color="Player",
                markers=True,
                title=f"{era_metric}+ Era Compare",
            )
            era_fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.5)", annotation_text="League average")
            era_fig.update_layout(
                xaxis_title=era_x_label,
                yaxis_title=f"{era_metric}+",
                legend_title="Player",
            )
            st.plotly_chart(era_fig, use_container_width=True)
            if not era_snapshot.empty:
                render_html_table(
                    era_snapshot,
                    number_cols=[f"{era_metric}+", f"Latest {era_metric}+", f"Raw {era_metric}"],
                    max_height_px=260,
                )

        _render_visual_overlap_charts(player_frames, selected_compare_view)
        _render_strengths_weaknesses_matrix(player_frames, selected_compare_view)
        _render_shot_profile_compare(player_frames, selected_compare_view)
        _render_comparison_verdict_cards(player_frames, selected_compare_view)
        _render_percentile_compare_view(player_frames, selected_compare_view)

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

    if ai_col is not None:
        with ai_col:
            st.markdown('<div class="sticky-compare-ai-rail"></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Tools")
            st.caption("Open or close each panel as needed.")

            with st.expander("Side-by-Side AI Scouting Report", expanded=False):
                if model:
                    st.caption("Generate a structured scouting report for every selected player in this compare view.")
                    if st.button("Generate Scouting Report", key="generate_compare_scouting_report", use_container_width=True):
                        prompt = _compare_scouting_report_prompt(player_frames, selected_compare_view)
                        with st.spinner("Building scouting report…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.55)
                                st.session_state["compare_scouting_report_output"] = text or "No response."
                                st.session_state["compare_report_players"] = [item["name"] for item in player_frames]
                                st.session_state["compare_report_view"] = selected_compare_view
                                st.session_state["compare_report_mode"] = "scouting"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("compare_scouting_report_output"):
                        st.caption("A scouting report is already available.")
                        if st.button("Open Report Page", key="open_compare_scouting_report_page", use_container_width=True):
                            st.session_state["compare_report_mode"] = "scouting"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Who Would You Rather Build Around?", expanded=False):
                if model:
                    lens = st.selectbox(
                        "Build-around lens",
                        ["Balanced", "Long-Term Franchise Pick", "Win-Now Contender"],
                        key="compare_build_around_lens",
                    )
                    if st.button("Run Build-Around Verdict", key="run_build_around_verdict", use_container_width=True):
                        prompt = _build_around_prompt(player_frames, selected_compare_view, lens)
                        with st.spinner("Weighing the build-around case…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.55)
                                st.session_state["compare_build_around_output"] = text or "No response."
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("compare_build_around_output"):
                        st.markdown(st.session_state["compare_build_around_output"])
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("What Changed?", expanded=False):
                if model:
                    st.caption("Compare how the selected players evolved across early career, prime, late career, and peak season using full-career context.")
                    if st.button("Generate Evolution Compare", key="generate_compare_what_changed", use_container_width=True):
                        prompt = _compare_what_changed_prompt(player_frames)
                        with st.spinner("Tracing each player's evolution…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.6)
                                st.session_state["compare_what_changed_output"] = text or "No response."
                                st.session_state["compare_what_changed_players"] = [item["name"] for item in player_frames]
                                st.session_state["compare_report_mode"] = "what-changed"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("compare_what_changed_output"):
                        st.caption("A compare evolution report is already available.")
                        if st.button("Open What Changed Page", key="open_compare_what_changed_page", use_container_width=True):
                            st.session_state["compare_report_mode"] = "what-changed"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Debate Mode", expanded=False):
                if model:
                    debate_lens = st.selectbox(
                        "Debate lens",
                        [
                            "Best Overall Player",
                            "Playoff Series",
                            "Best Offensive Engine",
                            "Best Defensive Piece",
                            "Best Second Star",
                            "Greatest Overall Career",
                            "Highest Peak",
                            "Best Prime",
                            "Best Legacy / Resume",
                            "Most Complete All-Time Player",
                        ],
                        key="compare_debate_lens_select",
                    )
                    debate_focus = st.text_input(
                        "Optional custom framing",
                        value="",
                        placeholder="Example: Make the case for who is harder to scheme against in the playoffs",
                        key="compare_debate_focus",
                    )
                    if st.button("Generate Debate", key="generate_compare_debate", use_container_width=True):
                        prompt = _compare_debate_prompt(player_frames, selected_compare_view, debate_lens, debate_focus)
                        debate_key = _compare_debate_chat_key(player_frames, selected_compare_view, debate_lens)
                        with st.spinner("Building the debate…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.7)
                                st.session_state["compare_debate_output"] = text or "No response."
                                st.session_state["compare_debate_players"] = [item["name"] for item in player_frames]
                                st.session_state["compare_debate_view"] = selected_compare_view
                                st.session_state["compare_debate_lens"] = debate_lens
                                st.session_state["compare_debate_chat_key"] = debate_key
                                st.session_state["compare_debate_system_prompt"] = prompt
                                st.session_state.setdefault("compare_debate_conversations", {})[debate_key] = [
                                    {"role": "assistant", "content": text or "No response."}
                                ]
                                st.session_state["compare_report_mode"] = "debate"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("compare_debate_output"):
                        st.caption("A debate page is already available.")
                        if st.button("Open Debate Page", key="open_compare_debate_page", use_container_width=True):
                            st.session_state["compare_report_mode"] = "debate"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Comparison Question Ideas", expanded=False):
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
                for i, idea in enumerate(ideas_cmp):
                    short = abbrev(idea, 44)
                    if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"cmp_idea_btn_{i}_{short}"):
                        st.session_state["ai_compare_question"] = idea
                        smooth_scroll_to(make_anchor("compare_ai_anchor"))
                        st.rerun()

            anchor = make_anchor("compare_ai_anchor")
            with st.expander("Ask the AI Assistant", expanded=True):
                if model:
                    chat_store = st.session_state.setdefault("compare_ai_conversations", {})
                    chat_key = _compare_ai_chat_key(player_frames)
                    history = chat_store.get(chat_key, [])
                    system_prompt = _compare_ai_system_prompt(player_frames)
                    st.session_state["compare_ai_system_prompt"] = system_prompt
                    st.session_state["compare_ai_chat_key"] = chat_key
                    st.session_state["compare_ai_chat_players"] = [item["name"] for item in player_frames]

                    if history:
                        st.caption("Continue the compare conversation on its own page.")
                        if st.button("Open AI Chat Page", key=f"open_compare_ai_chat_{chat_key}", use_container_width=True):
                            st.session_state["compare_report_mode"] = "chat"
                            st.rerun()
                    else:
                        with st.form(key=f"compare_ai_form_{chat_key}", clear_on_submit=True):
                            q2 = st.text_input("Ask something about these players:", key=f"ai_compare_question_{chat_key}")
                            submitted = st.form_submit_button("Open AI Chat", use_container_width=True)
                        if submitted and q2:
                            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q2}]
                            with st.spinner("Analyzing…"):
                                try:
                                    text = ai_generate_text(model, messages=messages, max_output_tokens=4096, temperature=0.6)
                                    chat_store[chat_key] = [{"role": "user", "content": q2}, {"role": "assistant", "content": text or "No response."}]
                                    st.session_state["compare_report_mode"] = "chat"
                                    st.rerun()
                                except Exception as e:
                                    st.warning(_friendly_ai_error_message(e))
                                    st.caption(f"Details: {type(e).__name__}")
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")
