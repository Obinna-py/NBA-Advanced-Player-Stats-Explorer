# ui_player.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import html
from datetime import datetime
from config import ai_generate_text, AI_SETUP_ERROR
from logos import college_logos
from fetch import get_player_career, get_player_info, get_player_birthdate, get_balldontlie_player, get_balldontlie_team_games, get_nba_headshot_url, get_balldontlie_league_season_averages
from metrics import compute_full_advanced_stats, generate_player_summary, compact_player_context, add_per_game_columns, metric_public_cols, build_ai_phase_table, build_ai_stat_packet, compute_player_percentile_context, detect_player_archetype, find_similar_players
from ideas import cached_ai_question_ideas, presets, ai_detect_career_phases
from utils import abbrev, public_cols
from ui_compare import render_html_table, _make_readable_stats_table, STAT_TOOLTIPS, _label_with_tooltip, render_stat_text, _inject_sticky_ai_rail_css


def _friendly_ai_error_message(error: Exception) -> str:
    text = str(error or "").lower()
    if any(term in text for term in ["quota", "resourceexhausted", "resource exhausted", "rate limit", "429"]):
        return "AI is temporarily unavailable because the current OpenAI quota or rate limit has been reached. Please try again a little later."
    if any(term in text for term in ["api key", "permission", "unauthorized", "403"]):
        return "AI is unavailable right now because the OpenAI connection or permissions need attention."
    return "AI is unavailable right now. Please try again in a moment."


def _player_scouting_report_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    return (
        "You are an expert NBA scout and analyst. Build a clean player scouting report using only the provided stats.\n"
        "Write in markdown with these exact sections: Offensive Identity, Shooting Profile, Playmaking, Defense, "
        "Rebounding / Physicality, Weaknesses, Best Team Fit, Bottom Line.\n"
        "Use specific stats throughout. Keep it detailed but readable for a serious basketball fan. "
        "Do not mention any other players unless the provided stats directly require a comparison.\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_ai_system_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    return (
        "You are an expert NBA analyst writing for a curious basketball fan.\n\n"
        "Answer questions using ONLY the stats provided below.\n"
        "Write a fuller analysis, not a short answer.\n"
        "Give 3-5 solid paragraphs unless the question is extremely narrow.\n"
        "Be direct, but explain your reasoning clearly and tie claims to specific stats.\n"
        "If some metrics are unavailable, do not refuse the question. Use the available metrics, explain what they show, and mention a missing stat only if it materially limits precision.\n"
        "Do not claim a stat is missing if it appears in the structured stat packet.\n"
        "Prefer the structured stat packet first, then use the season summary for extra context.\n"
        "When useful, compare efficiency, volume, playmaking load, rebounding, and trends across seasons.\n"
        "End with a short bottom-line takeaway sentence.\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n\n"
        "Note: Some advanced metrics are estimates from team-context formulas."
    )


def render_player_scouting_report_page() -> None:
    st.markdown("## 🧠 AI Player Scouting Report")
    player_name = st.session_state.get("player_report_player_name") or "Selected Player"
    st.caption(player_name)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_report", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_scouting_report_output")
    if report:
        st.markdown(report)
    else:
        st.info("No scouting report is available right now. Generate one from the player's Stats page first.")


def render_player_ai_chat_page(model) -> None:
    player_name = st.session_state.get("player_ai_chat_player_name") or "Selected Player"
    chat_key = st.session_state.get("player_ai_chat_key")
    st.markdown("## 💬 AI Player Chat")
    st.caption(player_name)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_chat", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    if not model:
        st.info("AI is unavailable right now.")
        return

    if not chat_key:
        st.info("No AI chat is available yet. Start one from the player's Stats page first.")
        return

    chat_store = st.session_state.setdefault("player_ai_conversations", {})
    history = chat_store.get(chat_key, [])
    system_prompt = st.session_state.get("player_ai_system_prompt")

    if st.button("Clear Conversation", key=f"clear_player_ai_page_{chat_key}", use_container_width=False):
        chat_store[chat_key] = []
        st.rerun()

    if history:
        for message in history:
            role = "You" if message["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {message['content']}")
    else:
        st.caption("Start the conversation below.")

    with st.form(key=f"player_ai_page_form_{chat_key}", clear_on_submit=True):
        q = st.text_input("Ask a follow-up about this player:", key=f"player_ai_page_input_{chat_key}")
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and q and system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": q}]
        with st.spinner("Analyzing…"):
            try:
                text = ai_generate_text(model, messages=messages, max_output_tokens=3072, temperature=0.7)
                updated_history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": text or "No response."}]
                chat_store[chat_key] = updated_history[-10:]
                st.rerun()
            except Exception as e:
                st.warning(_friendly_ai_error_message(e))
                st.caption(f"Details: {type(e).__name__}")


def _age_from_birthdate(iso_dt: str) -> int:
    birthdate = datetime.strptime(iso_dt.split('T')[0], "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


def _player_ai_chat_key(player: dict) -> str:
    return f"{player.get('source','unknown')}:{player.get('id','unknown')}:{player.get('full_name','')}"


def _birth_year_for_player(player: dict) -> int | None:
    try:
        birthdate = get_player_birthdate(player["id"], player_name=player.get("full_name"), player_source=player.get("source"))
        if not birthdate:
            return None
        bdate = str(birthdate).split("T")[0]
        return int(bdate[:4])
    except Exception:
        return None


def _add_season_start(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "SEASON_ID" not in df.columns:
        return df
    out = df.copy()
    if "SEASON_START" not in out.columns:
        out["SEASON_START"] = out["SEASON_ID"].astype(str).str[:4].astype(int)
    return out


def _add_age_column(df: pd.DataFrame, birth_year: int | None) -> pd.DataFrame:
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

def info_tab(player):
    if player.get("source") == "balldontlie":
        balldontlie_info_tab(player)
        return

    try:
        info = get_player_info(player['id'], player_name=player.get("full_name"), player_source=player.get("source"))
    except Exception as e:
        st.subheader("Player Info")
        st.warning(
            "Player bio info could not be loaded from the live providers right now."
        )
        st.caption(f"Details: {type(e).__name__}")
        return

    if info is None or info.empty:
        st.subheader("Player Info")
        st.warning("No player bio information was returned from the NBA stats service.")
        return

    if info.attrs.get("cache_source") == "stale_disk_cache":
        st.caption("Showing cached player info because both live providers were unavailable.")
    elif info.attrs.get("provider") == "balldontlie":
        st.caption("Showing player info from balldontlie.")
    elif info.attrs.get("provider") == "nba_api":
        st.caption("Showing player info from NBA stats.")

    team_id = info.loc[0, 'TEAM_ID']
    headshot_url = get_nba_headshot_url(player['id'], player_name=player.get("full_name"), player_source=player.get("source"))
    team_logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

    st.subheader("Player Info")
    c1, c2 = st.columns([1, 2])
    with c1:
        if headshot_url:
            st.image(headshot_url, width=220)
        if pd.notna(team_id) and info.attrs.get("provider") != "balldontlie":
            st.image(team_logo_url, width=120)
    with c2:
        st.markdown(f"### {player['full_name']}")
        birthdate = info.loc[0, 'BIRTHDATE'] if 'BIRTHDATE' in info.columns else None
        if pd.notna(birthdate) and birthdate:
            age = _age_from_birthdate(birthdate)
            st.write(f"**Age:** {age}")
        st.write(f"**Height:** {info.loc[0, 'HEIGHT']}")
        weight = info.loc[0, 'WEIGHT']
        st.write(f"**Weight:** {weight} lbs" if pd.notna(weight) and weight else "**Weight:** —")
        st.write(f"**Position:** {info.loc[0, 'POSITION']}")
        college = info.loc[0, 'SCHOOL']
        st.write(f"**College:** {college}")
        if college and college in college_logos:
            st.image(college_logos[college], width=120)


def balldontlie_info_tab(player):
    st.subheader("Player Info")
    # The search payload already contains enough fields to render the player card.
    # Avoid blocking the page on a second live request unless key fields are missing.
    needs_refresh = not any(player.get(k) for k in ["team_name", "position", "height", "college", "country"])
    fresh = None
    if needs_refresh:
        try:
            fresh = get_balldontlie_player(int(player["id"]))
        except Exception:
            fresh = None

    player = fresh or player
    st.caption("Showing player info from balldontlie.")
    if player.get("source"):
        st.caption(f"Selected player source: {player.get('source')}")

    headshot_url = get_nba_headshot_url(player["id"], player_name=player.get("full_name"), player_source=player.get("source"))
    c1, c2 = st.columns([1, 2])
    with c1:
        if headshot_url:
            st.image(headshot_url, width=220)
    with c2:
        st.markdown(f"### {player.get('full_name', 'Unknown Player')}")
        st.write(f"**Team:** {player.get('team_name') or '—'}")
        st.write(f"**Position:** {player.get('position') or '—'}")
        st.write(f"**Height:** {player.get('height') or '—'}")
        weight = player.get("weight")
        st.write(f"**Weight:** {weight} lbs" if weight else "**Weight:** —")
        st.write(f"**Jersey:** {player.get('jersey_number') or '—'}")
        st.write(f"**College:** {player.get('college') or '—'}")
        st.write(f"**Country:** {player.get('country') or '—'}")


def balldontlie_games_tab(player):
    st.subheader("Recent Team Games")
    st.caption("Using balldontlie free-tier games data for the player's current team.")

    team_id = player.get("team_id")
    if not team_id:
        st.warning("No team is available for this player.")
        return

    try:
        games = get_balldontlie_team_games(int(team_id), per_page=10)
    except Exception as e:
        st.warning("Could not load team games from balldontlie right now.")
        st.caption(f"{type(e).__name__}: {e}")
        return

    if games.empty:
        st.info("No recent team games were returned.")
        return

    st.write(f"**Team:** {player.get('team_name') or player.get('team_abbreviation') or '—'}")
    st.dataframe(games, use_container_width=True, hide_index=True)


def _position_family_label(position: str) -> str:
    pos = str(position or "").upper()
    if "C" in pos:
        return "Big"
    if "G" in pos and "F" not in pos:
        return "Guard"
    if "F" in pos:
        return "Wing"
    return "Other"


def _render_hover_stat_cards(cards: list[tuple[str, str]]) -> None:
    if not cards:
        return

    st.markdown(
        """
        <style>
        .hover-stat-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 14px 16px;
          min-height: 110px;
        }
        .hover-stat-label {
          color: rgba(255,255,255,0.72);
          font-size: 0.85rem;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .hover-stat-value {
          color: #f9fafb;
          font-size: 2rem;
          font-weight: 700;
          line-height: 1.1;
        }
        .hover-stat-label .stat-tooltip {
          cursor: help;
          text-decoration: underline dotted rgba(255,255,255,0.35);
          text-underline-offset: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(cards))
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="hover-stat-card">
                  <div class="hover-stat-label">{_label_with_tooltip(label)}</div>
                  <div class="hover-stat-value">{html.escape(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_player_storytelling_dashboard(player_name: str, adv: pd.DataFrame, percentile_df: pd.DataFrame) -> None:
    if adv is None or adv.empty:
        return

    st.markdown("### 🎛️ Player Dashboard")
    st.caption("A quick visual story of this player's production, style, and league context.")

    left, right = st.columns(2)

    with left:
        if "SEASON_ID" in adv.columns and len(adv) >= 2:
            trend_df = adv.copy()
            trend_df["Season"] = trend_df["SEASON_ID"].astype(str)
            trend_df = trend_df.sort_values("Season")
            chart_rows = []
            metric_map = {
                "PPG": "Scoring",
                "RPG": "Rebounding",
                "APG": "Playmaking",
                "TS%": "Efficiency",
            }
            for col, label in metric_map.items():
                if col not in trend_df.columns:
                    continue
                for _, row in trend_df.iterrows():
                    val = pd.to_numeric(row.get(col), errors="coerce")
                    if pd.isna(val):
                        continue
                    chart_rows.append({"Season": row["Season"], "Value": val, "Metric": label})
            if chart_rows:
                trend_plot = pd.DataFrame(chart_rows)
                fig = px.line(
                    trend_plot,
                    x="Season",
                    y="Value",
                    color="Metric",
                    markers=True,
                    title="Career Arc",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough seasonal trend data to draw a career arc yet.")
        else:
            st.info("Turn on “ALL seasons” to unlock the career arc chart.")

    with right:
        if percentile_df is not None and not percentile_df.empty:
            radar_metrics = ["PPG", "TS%", "APG", "TRB%", "BLK/G", "3P%"]
            radar_rows = percentile_df[percentile_df["Metric"].isin(radar_metrics)].copy()
            if not radar_rows.empty:
                ordered = []
                for metric in radar_metrics:
                    match = radar_rows[radar_rows["Metric"] == metric]
                    if not match.empty:
                        ordered.append((metric, float(match.iloc[0]["Percentile"])))
                if ordered:
                    labels = [item[0] for item in ordered]
                    values = [item[1] for item in ordered]
                    labels.append(labels[0])
                    values.append(values[0])
                    radar = go.Figure()
                    radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels,
                        fill="toself",
                        name=player_name,
                        line=dict(color="#f59e0b"),
                    ))
                    radar.update_layout(
                        title="Player DNA",
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        height=360,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(radar, use_container_width=True)
                else:
                    st.info("Not enough percentile data to draw the player DNA chart.")
            else:
                st.info("Not enough percentile data to draw the player DNA chart.")
        else:
            st.info("Percentile context is needed to build the player DNA chart.")

    try:
        season_id = str(adv.iloc[-1].get("SEASON_ID", ""))
        season_start = int(season_id[:4])
    except Exception:
        season_start = None

    if season_start:
        league_df = get_balldontlie_league_season_averages(season_start)
    else:
        league_df = pd.DataFrame()

    if league_df is not None and not league_df.empty:
        scatter = league_df.copy()
        scatter["USG"] = pd.to_numeric(scatter.get("usg_pct"), errors="coerce") * 100.0
        scatter["TS"] = pd.to_numeric(scatter.get("ts_pct"), errors="coerce") * 100.0
        scatter["PPG"] = pd.to_numeric(scatter.get("pts"), errors="coerce")
        scatter["Position Family"] = scatter["POSITION"].apply(_position_family_label)
        scatter = scatter.dropna(subset=["USG", "TS", "PPG"])
        scatter = scatter[scatter["PPG"] >= 8].copy()

        latest_name = str(player_name).lower()
        player_row = scatter[scatter["PLAYER_NAME"].astype(str).str.lower() == latest_name].copy()

        if not scatter.empty:
            fig = px.scatter(
                scatter,
                x="USG",
                y="TS",
                size="PPG",
                color="Position Family",
                hover_name="PLAYER_NAME",
                title="League Context: Usage vs Efficiency",
                labels={"USG": "USG%", "TS": "TS%"},
                opacity=0.45,
            )
            if not player_row.empty:
                fig.add_trace(go.Scatter(
                    x=player_row["USG"],
                    y=player_row["TS"],
                    mode="markers+text",
                    text=[player_name],
                    textposition="top center",
                    marker=dict(size=18, color="#f59e0b", line=dict(color="white", width=1.5)),
                    name=player_name,
                ))
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

def validate_phase_output(phases: dict, seasons_available: list[str]) -> dict:
    """
    - Remove seasons not in data
    - Ensure ordering matches seasons_available order
    - Ensure no duplicates
    - Ensure every season is assigned (optional)
    """
    if not phases or not seasons_available:
        return {}

    available_set = set(seasons_available)

    def clean_list(xs):
        xs = [x for x in (xs or []) if isinstance(x, str)]
        xs = [x for x in xs if x in available_set]
        # keep order as in seasons_available
        ordered = [s for s in seasons_available if s in xs]
        # unique
        out = []
        for s in ordered:
            if s not in out:
                out.append(s)
        return out

    phases["early"] = clean_list(phases.get("early"))
    phases["prime"] = clean_list(phases.get("prime"))
    phases["late"]  = clean_list(phases.get("late"))

    # Optionally auto-fill missing seasons (so every season belongs somewhere)
    used = set(phases["early"] + phases["prime"] + phases["late"])
    missing = [s for s in seasons_available if s not in used]

    if missing:
        # Put missing seasons into the closest bucket by position in timeline
        # Simple: before prime -> early, after prime -> late
        if phases["prime"]:
            first_prime = seasons_available.index(phases["prime"][0])
            last_prime  = seasons_available.index(phases["prime"][-1])
            for s in missing:
                idx = seasons_available.index(s)
                if idx < first_prime:
                    phases["early"].append(s)
                elif idx > last_prime:
                    phases["late"].append(s)
                else:
                    phases["prime"].append(s)

        # re-order again
        phases["early"] = clean_list(phases["early"])
        phases["prime"] = clean_list(phases["prime"])
        phases["late"]  = clean_list(phases["late"])

    # Peak season sanity
    peak = phases.get("peak_season")
    if peak not in available_set:
        # fallback: use last prime season or middle season
        if phases["prime"]:
            phases["peak_season"] = phases["prime"][-1]
        else:
            phases["peak_season"] = seasons_available[len(seasons_available)//2]

    # Confidence clamp
    try:
        phases["confidence"] = float(phases.get("confidence", 0.5))
    except:
        phases["confidence"] = 0.5
    phases["confidence"] = max(0.0, min(1.0, phases["confidence"]))

    return phases


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


def _build_summary_view_options(player: dict, adv: pd.DataFrame) -> tuple[dict[str, dict], dict[str, str]]:
    options = {}
    captions = {}
    if adv is None or adv.empty:
        return options, captions

    latest_df = adv.tail(1).copy()
    options["Latest Season"] = {
        "summary": latest_df.iloc[-1],
        "table_df": latest_df,
    }
    latest_season = str(adv.iloc[-1].get("SEASON_ID", "latest season"))
    captions["Latest Season"] = f"Using the player's latest available season ({latest_season})."

    options["Full Career"] = {
        "summary": _summarize_stat_slice(adv),
        "table_df": adv.copy(),
    }
    captions["Full Career"] = "Showing the player's full career, with the headline cards using a games-weighted average."

    phase_store = st.session_state.get("career_phases_by_player", {})
    phases = phase_store.get(_player_phase_state_key(player))
    if not phases:
        return options, captions

    phase_labels = [
        ("Early Career", phases.get("early", [])),
        ("Prime", phases.get("prime", [])),
        ("Late Career", phases.get("late", [])),
    ]
    for label, season_list in phase_labels:
        season_list = [str(s) for s in season_list if s]
        if not season_list:
            continue
        subset = adv[adv["SEASON_ID"].astype(str).isin(season_list)].copy() if "SEASON_ID" in adv.columns else pd.DataFrame()
        if subset.empty:
            continue
        options[label] = {
            "summary": _summarize_stat_slice(subset),
            "table_df": subset,
        }
        captions[label] = f"Showing these seasons: {', '.join(season_list)}. The headline cards use a games-weighted average for that window."

    return options, captions


def stats_tab(player, model):
    st.subheader("Player Stats")

    raw_pergame = get_player_career(
        player['id'],
        per_mode='PerGame',
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        all_seasons=True,
    )
    raw_totals = get_player_career(
        player['id'],
        per_mode='Totals',
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        all_seasons=True,
    )

    if raw_pergame.empty and raw_totals.empty:
        st.warning("No stats could be loaded from balldontlie, NBA stats, or local cache right now.")
        err_type = raw_totals.attrs.get("error_type") or raw_pergame.attrs.get("error_type")
        if err_type:
            st.caption(f"Last fetch error: {err_type}")
        return

    career_sources = {raw_pergame.attrs.get("cache_source"), raw_totals.attrs.get("cache_source")}
    if "stale_disk_cache" in career_sources:
        st.caption("Showing cached career stats because both live providers were unavailable.")
    elif raw_pergame.attrs.get("provider") == "balldontlie" or raw_totals.attrs.get("provider") == "balldontlie":
        st.caption("Showing stats from balldontlie.")
    elif raw_pergame.attrs.get("provider") == "nba_api" or raw_totals.attrs.get("provider") == "nba_api":
        st.caption("Showing stats from NBA stats.")

    if raw_totals is not None and not raw_totals.empty:
        if raw_totals.attrs.get("provider") == "balldontlie":
            adv = raw_totals.copy()
        else:
            with st.spinner("Computing advanced metrics…"):
                adv = compute_full_advanced_stats(raw_totals)
    else:
        adv = pd.DataFrame()
    
    adv = add_per_game_columns(adv, raw_pergame)

    full_adv = adv

    phase_player_key = _player_phase_state_key(player)
    phase_store = st.session_state.setdefault("career_phases_by_player", {})
    if model and phase_player_key not in phase_store and full_adv is not None and not full_adv.empty:
        phase_df = build_ai_phase_table(full_adv)
        seasons = phase_df["Season"].dropna().astype(str).tolist() if "Season" in phase_df.columns else []
        if seasons:
            phase_table_text = phase_df.to_csv(index=False)
            try:
                with st.spinner("Preparing career phases…"):
                    phases = ai_detect_career_phases(
                        player["full_name"],
                        phase_table_text,
                        use_model=True,
                        _model=model,
                    )
                phases = validate_phase_output(phases, seasons)
                phase_store[phase_player_key] = phases
            except Exception:
                pass

    summary_source = full_adv if full_adv is not None and not full_adv.empty else adv
    summary_views, summary_captions = _build_summary_view_options(player, summary_source)
    selected_summary_label = None
    selected_summary = None
    display_adv = adv
    if summary_views:
        selected_summary_label = st.selectbox(
            "Summary view",
            list(summary_views.keys()),
            index=0,
            key=f"summary_view_{_player_phase_state_key(player)}",
        )
        selected_summary = summary_views[selected_summary_label]["summary"]
        display_adv = summary_views[selected_summary_label]["table_df"]

    latest_src = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
    if latest_src is not None and not latest_src.empty:
        latest = selected_summary if selected_summary is not None else latest_src.iloc[-1]
        headshot_url = get_nba_headshot_url(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        hero_col, stats_col = st.columns([1, 3])
        with hero_col:
            if headshot_url:
                st.image(headshot_url, width=170)
            st.caption(player.get("full_name", ""))
        with stats_col:
            if selected_summary_label and selected_summary_label in summary_captions:
                render_stat_text(summary_captions[selected_summary_label], small=True)
            ppg_val = pd.to_numeric(latest.get("PPG", latest.get("PTS", np.nan)), errors="coerce")
            rpg_val = pd.to_numeric(latest.get("RPG", latest.get("REB", np.nan)), errors="coerce")
            apg_val = pd.to_numeric(latest.get("APG", latest.get("AST", np.nan)), errors="coerce")
            ts_val = latest.get('TS%', np.nan)
            ts_num = pd.to_numeric(ts_val, errors="coerce")
            _render_hover_stat_cards([
                ("PPG", f"{ppg_val:.1f}" if pd.notna(ppg_val) else "—"),
                ("RPG", f"{rpg_val:.1f}" if pd.notna(rpg_val) else "—"),
                ("APG", f"{apg_val:.1f}" if pd.notna(apg_val) else "—"),
                ("TS%", f"{ts_num:.1f}%" if pd.notna(ts_num) else "—"),
            ])

    if "show_player_ai_rail" not in st.session_state:
        st.session_state["show_player_ai_rail"] = True
    rail_toggle_col_left, rail_toggle_col_right = st.columns([4.5, 1.2])
    with rail_toggle_col_right:
        if st.button(
            "Hide AI sidebar" if st.session_state["show_player_ai_rail"] else "Show AI sidebar",
            key="toggle_player_ai_rail",
            use_container_width=True,
        ):
            st.session_state["show_player_ai_rail"] = not st.session_state["show_player_ai_rail"]
            st.rerun()

    if st.session_state["show_player_ai_rail"]:
        _inject_sticky_ai_rail_css("sticky-player-ai-rail")
        main_col, ai_col = st.columns([3.2, 1.35], gap="large")
    else:
        main_col = st.container()
        ai_col = None

    with main_col:
        if display_adv is not None and not display_adv.empty:
            birth_year = _birth_year_for_player(player)
            display_adv = _add_age_column(_add_season_start(display_adv), birth_year)
            stats_df, number_cols, percent_cols = _make_readable_stats_table(display_adv)

            render_html_table(
                stats_df,
                number_cols=number_cols,
                percent_cols=["Percentile"],
                max_height_px=520
            )

            latest_season_id = str(display_adv.iloc[-1].get("SEASON_ID", "")) if "SEASON_ID" in display_adv.columns else ""
            percentile_df = compute_player_percentile_context(player["full_name"], latest_season_id, display_adv)
            _render_player_storytelling_dashboard(player["full_name"], display_adv, percentile_df)
            if not percentile_df.empty:
                st.markdown("### 📈 Percentile & Ranking Context")
                render_stat_text("Latest-season context versus the league distribution from balldontlie season averages.", small=True)

                top_rows = percentile_df.head(6).copy()
                summary_cols = st.columns(min(3, len(top_rows)))
                for col, (_, row) in zip(summary_cols, top_rows.iterrows()):
                    with col:
                        st.metric(
                            row["Metric"],
                            f"{row['Percentile']:.1f} pct",
                            delta=f"Rank {int(row['Rank'])} / {int(row['Of'])}",
                        )

                display_df = percentile_df.copy()
                display_df["Percentile"] = pd.to_numeric(display_df["Percentile"], errors="coerce")
                display_df["Rank / Field"] = display_df.apply(
                    lambda r: f"{int(r['Rank'])} / {int(r['Of'])}" if pd.notna(r["Rank"]) and pd.notna(r["Of"]) else "—",
                    axis=1,
                )
                display_df = display_df[["Category", "Metric", "Value", "Percentile", "Rank / Field"]]
                render_html_table(
                    display_df,
                    number_cols=["Value"],
                    percent_cols=["Percentile"],
                    max_height_px=360,
                )

            archetype = detect_player_archetype(player["full_name"], display_adv, percentile_df)
            if archetype:
                st.markdown("### 🧩 Role Archetype")
                c1, c2 = st.columns([1.4, 1])
                with c1:
                    st.metric("Primary Role", archetype["primary"])
                    if archetype.get("primary_description"):
                        st.caption(archetype["primary_description"])
                    if archetype.get("secondary"):
                        st.caption(f"Secondary role: {archetype['secondary']}")
                        if archetype.get("secondary_description"):
                            st.caption(archetype["secondary_description"])
                    if archetype.get("style_tags"):
                        st.write("**Style Tags:** " + ", ".join(archetype["style_tags"]))
                    if archetype.get("impact_tags"):
                        st.write("**Impact Tags:** " + ", ".join(archetype["impact_tags"]))
                    st.caption(f"Confidence: {archetype['confidence']:.2f}")
                    for line in archetype.get("evidence", []):
                        st.write(f"- {line}")
                    tag_descriptions = archetype.get("tag_descriptions", {})
                    if tag_descriptions:
                        with st.expander("What these tags mean", expanded=False):
                            for tag in archetype.get("style_tags", []) + archetype.get("impact_tags", []):
                                desc = tag_descriptions.get(tag)
                                if desc:
                                    st.write(f"**{tag}:** {desc}")
                with c2:
                    scores = pd.DataFrame(archetype.get("style_scores", []))
                    if not scores.empty:
                        render_html_table(
                            scores,
                            number_cols=["Score"],
                            max_height_px=260,
                        )

            similar_df = find_similar_players(player["full_name"], latest_season_id, display_adv, limit=6)
            if not similar_df.empty:
                st.markdown("### 🧬 Similar Player Finder")
                render_stat_text("Closest latest-season statistical matches from the league-wide balldontlie season-average pool.", small=True)
                top_match = similar_df.iloc[0]
                render_stat_text(
                    f"Closest match right now: {top_match['Player']} "
                    f"({top_match['Similarity']:.1f} similarity) based on {top_match['Why Similar']}.",
                )
                render_html_table(
                    similar_df,
                    number_cols=["Similarity", "PPG", "RPG", "APG"],
                    percent_cols=["TS%", "3P%", "USG%"],
                    max_height_px=360,
                )

            render_stat_text("Hover over stat labels in the cards, tables, and stat blurbs to see quick explanations.", small=True)

    if ai_col is not None:
        with ai_col:
            st.markdown('<div class="sticky-player-ai-rail"></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Tools")
            st.caption("Open or close each panel as needed.")

            with st.expander("Career Phase Detection", expanded=False):
                if full_adv is not None and not full_adv.empty and model:
                    phase_df = build_ai_phase_table(full_adv if full_adv is not None and not full_adv.empty else adv)
                    seasons = phase_df["Season"].dropna().astype(str).tolist() if "Season" in phase_df.columns else []

                    if not seasons:
                        st.warning("No Season values available for phase detection.")
                    else:
                        phase_table_text = phase_df.to_csv(index=False)
                        run_ai = st.button("Run AI Career Phase Detection", use_container_width=True)

                        if run_ai:
                            with st.spinner("AI is labeling early/prime/late…"):
                                try:
                                    phases = ai_detect_career_phases(
                                        player["full_name"],
                                        phase_table_text,
                                        use_model=(model is not None),
                                        _model=model,
                                    )
                                    phases = validate_phase_output(phases, seasons)
                                    phase_store = st.session_state.setdefault("career_phases_by_player", {})
                                    phase_store[_player_phase_state_key(player)] = phases
                                    st.session_state["career_phases"] = phases
                                except Exception as e:
                                    st.warning(_friendly_ai_error_message(e))
                                    st.caption(f"Details: {type(e).__name__}")

                        phase_store = st.session_state.get("career_phases_by_player", {})
                        phases = phase_store.get(_player_phase_state_key(player)) or st.session_state.get("career_phases")
                        if phases:
                            st.success(
                                f"Peak season: **{phases['peak_season']}** • Confidence: **{phases['confidence']:.2f}**"
                            )
                            st.write(
                                f"**Early:** {', '.join(phases['early']) if phases['early'] else '—'}\n\n"
                                f"**Prime:** {', '.join(phases['prime']) if phases['prime'] else '—'}\n\n"
                                f"**Late:** {', '.join(phases['late']) if phases['late'] else '—'}"
                            )

                            rs = phases.get("reasoning_short", {})
                            with st.expander("Why the AI chose these phases", expanded=False):
                                st.write(f"**Early:** {rs.get('early','—')}")
                                st.write(f"**Prime:** {rs.get('prime','—')}")
                                st.write(f"**Late:** {rs.get('late','—')}")
                else:
                    st.info("Career phase detection becomes available when AI and seasonal data are available.")

            with st.expander("Question Ideas", expanded=False):
                choices, topic_map = presets()
                preset = st.radio("Quick presets", choices, horizontal=True, key="idea_preset")
                topic_default = topic_map.get(preset, "")
                topic = st.text_input("Optional focus (refines suggestions):", value=topic_default, key="idea_focus")
                ctx = compact_player_context(adv if not adv.empty else raw_pergame)
                ideas = cached_ai_question_ideas(player['full_name'], ctx, topic, use_model=(model is not None), _model=model)
                st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
                for i, idea in enumerate(ideas):
                    short = abbrev(idea, 40)
                    if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"idea_btn_{i}_{short}"):
                        st.session_state["ai_question"] = idea
                        st.rerun()

            with st.expander("AI Scouting Report", expanded=False):
                if model:
                    st.caption("Generate a dedicated scouting report page for this player.")
                    if st.button("Generate Scouting Report", key="generate_player_scouting_report", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = adv if adv is not None else pd.DataFrame()
                        prompt = _player_scouting_report_prompt(player["full_name"], pergame_for_summary, adv_for_summary)
                        with st.spinner("Building scouting report…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.6)
                                st.session_state["player_scouting_report_output"] = text or "No response."
                                st.session_state["player_report_player_name"] = player["full_name"]
                                st.session_state["player_report_mode"] = "scouting"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_scouting_report_output"):
                        st.caption("A scouting report is already available.")
                        if st.button("Open Report Page", key="open_player_scouting_report_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "scouting"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Ask the AI Assistant", expanded=True):
                if model:
                    chat_store = st.session_state.setdefault("player_ai_conversations", {})
                    chat_key = _player_ai_chat_key(player)
                    history = chat_store.get(chat_key, [])
                    pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                    adv_for_summary = adv if adv is not None else pd.DataFrame()
                    system_prompt = _player_ai_system_prompt(player['full_name'], pergame_for_summary, adv_for_summary)
                    st.session_state["player_ai_system_prompt"] = system_prompt
                    st.session_state["player_ai_chat_key"] = chat_key
                    st.session_state["player_ai_chat_player_name"] = player["full_name"]

                    if history:
                        st.caption("Continue the player conversation on its own page.")
                        if st.button("Open AI Chat Page", key=f"open_player_ai_chat_{chat_key}", use_container_width=True):
                            st.session_state["player_report_mode"] = "chat"
                            st.rerun()
                    else:
                        with st.form(key=f"player_ai_form_{chat_key}", clear_on_submit=True):
                            q = st.text_input("Ask something about this player:", key=f"ai_question_{chat_key}")
                            submitted = st.form_submit_button("Open AI Chat", use_container_width=True)
                        if submitted and q:
                            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
                            with st.spinner("Analyzing…"):
                                try:
                                    text = ai_generate_text(model, messages=messages, max_output_tokens=3072, temperature=0.7)
                                    chat_store[chat_key] = [{"role": "user", "content": q}, {"role": "assistant", "content": text or "No response."}]
                                    st.session_state["player_report_mode"] = "chat"
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
