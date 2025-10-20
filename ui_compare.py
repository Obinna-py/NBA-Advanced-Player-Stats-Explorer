# ui_compare.py
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players as nba_players

# --- project imports
from fetch import get_player_career, get_head_to_head_games
from metrics import (
    compute_full_advanced_stats,
    add_per_game_columns,
    generate_player_summary,
    compact_player_context,
    metric_public_cols,          # returns a filtered/ordered list of columns for display
    order_columns_for_display,
    compute_league_shooting_table    # preferred display ordering (PPG/RPG/APG → TS% → etc.)
)
from ideas import cached_ai_compare_question_ideas
from utils import abbrev, make_anchor, smooth_scroll_to


# -------------------------
# Small helpers
# -------------------------
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
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
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
    found = nba_players.find_players_by_full_name(name)
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


# -------------------------
# Main render function
# -------------------------
def render_compare_tab(primary_player: dict, model=None):
    st.subheader("Compare Players")

    # ---- Second player state & search
    if "other_player" not in st.session_state:
        st.session_state["other_player"] = None

    other_name = st.text_input("Enter another player's name to compare:", key="other_name_input")
    if st.button("Search Second Player"):
        op = _find_player_by_name(other_name)
        if not op:
            st.session_state["other_player"] = None
            st.error("❌ No second player found. Check spelling.")
        else:
            st.session_state["other_player"] = op

    other_player = st.session_state["other_player"]
    if not other_player:
        st.info("Search and select a second player to start the comparison.")
        return

    p1 = primary_player["full_name"]; p2 = other_player["full_name"]
    st.success(f"Comparing **{p1}** vs **{p2}**")

    # --- Top info cards
    cL, cR = st.columns(2)
    with cL:
        info1 = commonplayerinfo.CommonPlayerInfo(player_id=primary_player['id']).get_data_frames()[0]
        st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{primary_player['id']}.png", width=180)
        st.markdown(f"**{p1}**")
        st.caption(f"{info1.loc[0, 'TEAM_NAME']} • {info1.loc[0, 'POSITION']}")
    with cR:
        info2 = commonplayerinfo.CommonPlayerInfo(player_id=other_player['id']).get_data_frames()[0]
        st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{other_player['id']}.png", width=180)
        st.markdown(f"**{p2}**")
        st.caption(f"{info2.loc[0, 'TEAM_NAME']} • {info2.loc[0, 'POSITION']}")

    # --- Performance toggle
    comp_speed_mode = st.toggle(
        "Compute advanced for ALL seasons in comparison (slower)",
        value=False,
        help="Off = chart from per-game (fast, multi-season). On = advanced for all seasons (slower).",
        key="compare_speed_toggle"
    )

    # --- Load careers
    raw1_pg = _nzdf(get_player_career(primary_player['id'], per_mode='PerGame'))
    raw2_pg = _nzdf(get_player_career(other_player['id'],  per_mode='PerGame'))
    raw1_t  = _nzdf(get_player_career(primary_player['id'], per_mode='Totals'))
    raw2_t  = _nzdf(get_player_career(other_player['id'],  per_mode='Totals'))

    # --- Compute advanced (latest or all seasons)
    if comp_speed_mode:
        adv1_src, adv2_src = raw1_t, raw2_t
    else:
        latest1 = raw1_t['SEASON_ID'].iloc[-1] if not raw1_t.empty else None
        latest2 = raw2_t['SEASON_ID'].iloc[-1] if not raw2_t.empty else None
        adv1_src = raw1_t[raw1_t['SEASON_ID'] == latest1] if latest1 else raw1_t
        adv2_src = raw2_t[raw2_t['SEASON_ID'] == latest2] if latest2 else raw2_t

    with st.spinner("Computing comparison advanced metrics…"):
        adv1 = compute_full_advanced_stats(adv1_src) if not adv1_src.empty else pd.DataFrame()
        adv2 = compute_full_advanced_stats(adv2_src) if not adv2_src.empty else pd.DataFrame()

    # --- Add per-game aliases & SEASON_START to advanced frames
    adv1 = add_per_game_columns(adv1, raw1_pg)
    adv2 = add_per_game_columns(adv2, raw2_pg)
    adv1 = _add_season_start(adv1)
    adv2 = _add_season_start(adv2)

    # --- Question ideas (chips)
    # --- Question ideas (chips)
    # --- Question ideas (chips)
    ctx1_src = adv1 if not adv1.empty else raw1_pg
    ctx2_src = adv2 if not adv2.empty else raw2_pg

    # Safely turn the player context into dictionaries
    def _ensure_ctx_dict(ctx_obj):
        import json
        if isinstance(ctx_obj, dict):
            return ctx_obj
        if ctx_obj is None:
            return {}
        if hasattr(ctx_obj, "to_dict"):
            try:
                return ctx_obj.to_dict()
            except Exception:
                pass
        if isinstance(ctx_obj, str):
            try:
                parsed = json.loads(ctx_obj)
                return parsed if isinstance(parsed, dict) else {"text": ctx_obj}
            except Exception:
                return {"text": ctx_obj}
        try:
            return dict(ctx_obj)
        except Exception:
            return {"text": str(ctx_obj)}

    def _as_ctx_dict(x):
        d = _ensure_ctx_dict(x)
        return {str(k).lower(): v for k, v in d.items()}  # normalize lowercase keys

    c1_raw = compact_player_context(ctx1_src) if ctx1_src is not None and not ctx1_src.empty else {}
    c2_raw = compact_player_context(ctx2_src) if ctx2_src is not None and not ctx2_src.empty else {}
    c1 = _as_ctx_dict(c1_raw)
    c2 = _as_ctx_dict(c2_raw)






    # --- Choose which sources to chart from
    if comp_speed_mode:
        chart_src1, chart_src2 = adv1.copy(), adv2.copy()
    else:
        chart_src1, chart_src2 = raw1_pg.copy(), raw2_pg.copy()

    # --- Choose sources for chart (already defined above) ---
    chart_src1 = _add_season_start(chart_src1)
    chart_src2 = _add_season_start(chart_src2)

    # Alignment mode
    align_mode = st.radio(
        "Align seasons by",
        ["Calendar (overlap only)", "Career year", "Age"],
        horizontal=True,
        key="cmp_align_mode"
    )

    # Era-adjust toggle (stat+ vs league avg = 100)
    era_toggle = st.checkbox("Era-adjust (stat+ vs league avg = 100)", value=False, key="era_adjust_toggle")

    # If Age, add approximate age columns
    if align_mode == "Age":
        by1 = _get_birth_year(primary_player['id'])
        by2 = _get_birth_year(other_player['id'])
        chart_src1 = _add_age_column(chart_src1, by1)
        chart_src2 = _add_age_column(chart_src2, by2)

    years1 = set(chart_src1["SEASON_START"].unique().tolist()) if "SEASON_START" in chart_src1.columns else set()
    years2 = set(chart_src2["SEASON_START"].unique().tolist()) if "SEASON_START" in chart_src2.columns else set()
    common_years = sorted(years1 & years2)

    # Calendar-only filter UI
    if align_mode == "Calendar (overlap only)":
        if len(common_years) == 0:
            st.warning("No overlapping calendar seasons — try 'Career year' or 'Age' alignment.")
        elif len(common_years) == 1:
            y = common_years[0]
            st.caption(f"Only one overlapping season ({y}-{str(y+1)[-2:]}). Season filter disabled.")
            st.session_state.pop("cmp_season_range_slider", None)
            mask1 = chart_src1["SEASON_START"].eq(y)
            mask2 = chart_src2["SEASON_START"].eq(y)
            chart_src1 = chart_src1.loc[mask1].copy()
            chart_src2 = chart_src2.loc[mask2].copy()
        else:
            min_year, max_year = min(common_years), max(common_years)
            with st.expander("🔧 Filter seasons to compare (calendar)", expanded=False):
                yr_lo, yr_hi = st.slider(
                    "Season start year range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    step=1,
                    key="cmp_season_range_slider",
                )
            mask1 = chart_src1["SEASON_START"].between(yr_lo, yr_hi)
            mask2 = chart_src2["SEASON_START"].between(yr_lo, yr_hi)
            chart_src1 = chart_src1.loc[mask1].copy()
            chart_src2 = chart_src2.loc[mask2].copy()
            if chart_src1.empty or chart_src2.empty:
                st.warning("No data remains after applying the season filter.")

    # Shared stat selection
    shared_stats = _pick_shared_stats_for_dropdown(chart_src1, chart_src2)
    if not shared_stats:
        st.warning("No shared numeric stats available to compare.")
        st.stop()

    default_name = "PPG" if "PPG" in shared_stats else ("PTS" if "PTS" in shared_stats else shared_stats[0])
    try:
        default_idx = shared_stats.index(default_name)
    except ValueError:
        default_idx = 0
    stat_choice = st.selectbox("📊 Choose a stat to compare:", shared_stats, index=default_idx, key="cmp_stat_choice")

    # Era-adjust (stat+) for shooting percentages
    supported_plus = {"TS%","EFG%","FG%","3P%","FT%"}
    stat_for_align = stat_choice
    label_suffix = ""
    if era_toggle and stat_choice in supported_plus:
        all_seasons = sorted(set(list(years1 | years2)))
        season_ids = [f"{y}-{str(y+1)[-2:]}" for y in all_seasons]
        league_tbl = compute_league_shooting_table(season_ids)
        # Merge baseline to each player's seasons
        chart_src1 = chart_src1.merge(
            league_tbl[["SEASON_START", stat_choice]].rename(columns={stat_choice: "LEAGUE_BASE"}),
            on="SEASON_START", how="left"
        )
        chart_src2 = chart_src2.merge(
            league_tbl[["SEASON_START", stat_choice]].rename(columns={stat_choice: "LEAGUE_BASE"}),
            on="SEASON_START", how="left"
        )
        # Compute STAT+: 100 = league average that season
        chart_src1[f"{stat_choice}+"] = np.where(chart_src1["LEAGUE_BASE"]>0,
            100.0*pd.to_numeric(chart_src1[stat_choice], errors="coerce")/chart_src1["LEAGUE_BASE"], np.nan)
        chart_src2[f"{stat_choice}+"] = np.where(chart_src2["LEAGUE_BASE"]>0,
            100.0*pd.to_numeric(chart_src2[stat_choice], errors="coerce")/chart_src2["LEAGUE_BASE"], np.nan)
        stat_for_align = f"{stat_choice}+"
        label_suffix = "+"
        # Clean temp columns
        chart_src1.drop(columns=["LEAGUE_BASE"], errors="ignore", inplace=True)
        chart_src2.drop(columns=["LEAGUE_BASE"], errors="ignore", inplace=True)
    elif era_toggle and stat_choice not in supported_plus:
        st.info("Era-adjust currently supports TS%, EFG%, FG%, 3P%, FT%. Showing raw values.")

    # Build aligned series & chart
    aligned = _align_for_chart(chart_src1, chart_src2, p1, p2, stat_for_align, align_mode)
    if aligned.empty:
        if align_mode != "Calendar (overlap only)":
            st.warning(f"No overlapping values found under '{align_mode}' alignment.")
        else:
            st.warning("No overlapping seasons to compare.")
    else:
        x_label = {"Calendar (overlap only)": "Season", "Career year": "Career Year", "Age": "Age (approx)"}[align_mode]
        title = f"{stat_choice}{label_suffix} — {align_mode}"
        fig = px.line(aligned, x="X", y=[p1, p2], markers=True, title=title)
        fig.update_layout(xaxis_title=x_label, yaxis_title=f"{stat_choice}{label_suffix}", legend_title="Player")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Head-to-Head (calendar-based, only when both actually played)
    # ---------------------------
    st.markdown("## 🤝 Head-to-Head")

    # Choose season type
    h2h_season_type = st.radio(
        "Games to include",
        ["Regular Season", "Playoffs"],
        horizontal=True,
        key="h2h_season_type"
    )

    # Determine candidate seasons (calendar overlap only — head-to-head needs real games)
    p1_years = set(chart_src1["SEASON_START"].unique().tolist()) if "SEASON_START" in chart_src1.columns else set()
    p2_years = set(chart_src2["SEASON_START"].unique().tolist()) if "SEASON_START" in chart_src2.columns else set()
    h2h_years = sorted(p1_years & p2_years)

    if len(h2h_years) == 0:
        st.info("These players never shared an NBA season, so there are no head-to-head games.")
    else:
        if len(h2h_years) == 1:
            yr_lo, yr_hi = h2h_years[0], h2h_years[0]
            st.caption(f"Only one shared season: {yr_lo}-{str(yr_lo+1)[-2:]}")
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

        # Build list of season IDs from the year range
        def _yr_to_id(y: int) -> str:
            return f"{y}-{str(y+1)[-2:]}"
        season_ids = [_yr_to_id(y) for y in h2h_years if (y >= yr_lo and y <= yr_hi)]

        # Fetch head-to-head game list
        p1_id = primary_player["id"]
        p2_id = other_player["id"]
        p1_name = p1
        p2_name = p2

        with st.spinner("Loading head-to-head…"):
            h2h = get_head_to_head_games(p1_id, p2_id, seasons=season_ids, season_type=h2h_season_type)

        if h2h.empty:
            st.info("No head-to-head games found in the chosen range / season type.")
        else:
            # Summary block
            gp = len(h2h)
            p1_wins = int(h2h["P1_WIN"].sum()) if "P1_WIN" in h2h.columns else None
            p2_wins = gp - p1_wins if p1_wins is not None else None

            def _avg(col):
                return float(pd.to_numeric(h2h[col], errors="coerce").mean()) if col in h2h.columns else None

            # Per-game averages
            # P1 columns end with _P1; P2 columns end with _P2
            avg_stats = ["PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS","MIN"]
            row = []
            for sname in avg_stats:
                v1 = _avg(f"{sname}_P1")
                v2 = _avg(f"{sname}_P2")
                row.append((sname, v1, v2))

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Games", gp)
            with colB:
                if p1_wins is not None:
                    st.metric(f"{p1_name} W-L", f"{p1_wins}-{p2_wins}")
            with colC:
                if "PTS_P1" in h2h.columns and "PTS_P2" in h2h.columns:
                    p1_pts = round(_avg("PTS_P1") or 0, 1)
                    p2_pts = round(_avg("PTS_P2") or 0, 1)
                    st.metric("PPG (H2H)", f"{p1_pts} vs {p2_pts}", delta=f"{(p1_pts - p2_pts):+.1f}")

            # Quick bar chart of key averages
            chart_keys = ["PTS","REB","AST","STL","BLK","TOV"]
            plot_df = pd.DataFrame({
                "Stat": chart_keys,
                p1_name: [round(_avg(f"{k}_P1") or 0.0, 2) for k in chart_keys],
                p2_name: [round(_avg(f"{k}_P2") or 0.0, 2) for k in chart_keys],
            })
            fig = px.bar(plot_df.melt(id_vars=["Stat"], var_name="Player", value_name="Value"),
                        x="Stat", y="Value", color="Player", barmode="group",
                        title=f"Head-to-Head Averages — {h2h_season_type}")
            st.plotly_chart(fig, use_container_width=True)

            # Games table (compact)
            show_cols = []
            base_cols = ["GAME_DATE","SEASON_ID","TEAM_ABBREVIATION_P1","TEAM_ABBREVIATION_P2","WL_P1"]
            for c in base_cols:
                if c in h2h.columns: show_cols.append(c)
            for k in ["MIN","PTS","REB","AST","STL","BLK","TOV","PLUS_MINUS"]:
                c1, c2 = f"{k}_P1", f"{k}_P2"
                if c1 in h2h.columns and c2 in h2h.columns:
                    show_cols.extend([c1, c2])

            st.dataframe(
                h2h[show_cols].rename(columns={
                    "TEAM_ABBREVIATION_P1": f"Team {p1_name}",
                    "TEAM_ABBREVIATION_P2": f"Team {p2_name}",
                    "WL_P1": f"{p1_name} W/L",
                    "GAME_DATE": "Date",
                    "SEASON_ID": "Season",
                    "MIN_P1": f"MIN {p1_name}", "MIN_P2": f"MIN {p2_name}",
                    "PTS_P1": f"PTS {p1_name}", "PTS_P2": f"PTS {p2_name}",
                    "REB_P1": f"REB {p1_name}", "REB_P2": f"REB {p2_name}",
                    "AST_P1": f"AST {p1_name}", "AST_P2": f"AST {p2_name}",
                    "STL_P1": f"STL {p1_name}", "STL_P2": f"STL {p2_name}",
                    "BLK_P1": f"BLK {p1_name}", "BLK_P2": f"BLK {p2_name}",
                    "TOV_P1": f"TOV {p1_name}", "TOV_P2": f"TOV {p2_name}",
                    "PLUS_MINUS_P1": f"+/- {p1_name}", "PLUS_MINUS_P2": f"+/- {p2_name}",
                }),
                use_container_width=True,
                hide_index=True
            )


    # --- Side-by-side advanced tables
    st.subheader("📊 Advanced Stats")
    if comp_speed_mode:
        st.caption("All seasons (slower).")
    else:
        st.caption("Latest season only (fast). Toggle above for full career.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**{p1}**")
        st.dataframe(adv1[metric_public_cols(adv1)] if not adv1.empty else adv1, use_container_width=True)
    with t2:
        st.markdown(f"**{p2}**")
        st.dataframe(adv2[metric_public_cols(adv2)] if not adv2.empty else adv2, use_container_width=True)
    
    with st.expander("💡 Comparison Question Ideas for these players", expanded=False):
        topic_map = {
            "Overview": "balanced; efficiency, usage, passing, rebounding, trends",
            "Scoring & Efficiency": "TS%, eFG%, PPS, 3PAr, FTr, PTS/36, shooting splits",
            "Playmaking & TOs": "AST%, AST/TO, TOV trend, usage vs passing load",
            "Rebounding & Defense": "TRB%, ORB%, DRB%, STL/36, BLK/36",
            "Peak & Trends": "best/worst seasons, YoY changes, prime window",
        }
        preset = st.radio(
            "Quick presets",
            list(topic_map.keys()),
            horizontal=True,
            key="compare_idea_preset",
        )
        topic = st.text_input("Optional focus (refines suggestions):", value=topic_map[preset], key="compare_idea_focus")

        try:
            ideas_cmp = cached_ai_compare_question_ideas(p1, p2, c1, c2, topic, use_model=(model is not None))
        except Exception as e:
            st.warning(f"Ideas generator hiccup: {e}")
            ideas_cmp = []

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

    # --- AI compare
    anchor = make_anchor("compare_ai_anchor")
    with st.expander("🧠 Ask the AI Assistant about these players", expanded=False):
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
                    try:
                        resp = model.generate_content(prompt2, generation_config={"max_output_tokens": 4096, "temperature": 0.6})
                        st.markdown("### 🧠 AI Analysis")
                        st.write(resp.text if hasattr(resp, "text") else "No response.")
                    except Exception as e:
                        st.error(f"AI error: {e}")
        else:
            st.info("Add your Gemini API key to enable AI analysis.")
