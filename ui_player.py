# ui_player.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import html
import unicodedata
from datetime import datetime
from config import ai_generate_text, AI_SETUP_ERROR
from logos import college_logos
from fetch import get_player_career, get_player_info, get_player_birthdate, get_balldontlie_player, get_balldontlie_team_games, get_nba_headshot_url, get_placeholder_headshot_data_uri, get_balldontlie_league_season_averages, get_balldontlie_player_contracts, get_balldontlie_player_contract_aggregates, get_team_record_for_season, get_balldontlie_player_game_logs, get_balldontlie_team_players
from metrics import compute_full_advanced_stats, generate_player_summary, compact_player_context, add_per_game_columns, metric_public_cols, build_ai_phase_table, build_ai_stat_packet, compute_player_percentile_context, detect_player_archetype, find_similar_players, compute_impact_index
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


def _fmt_money(value) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "—"
    return f"${num:,.0f}"


def _contract_snapshot(contract_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> dict:
    latest_contract = contract_df.sort_values("season").iloc[-1] if contract_df is not None and not contract_df.empty else pd.Series(dtype=object)
    latest_agg = aggregate_df.sort_values("start_year").iloc[-1] if aggregate_df is not None and not aggregate_df.empty else pd.Series(dtype=object)
    return {
        "season": latest_contract.get("season"),
        "cap_hit": latest_contract.get("cap_hit"),
        "base_salary": latest_contract.get("base_salary"),
        "total_cash": latest_contract.get("total_cash"),
        "rank": latest_contract.get("rank"),
        "team_name": latest_contract.get("team_name"),
        "average_salary": latest_agg.get("average_salary"),
        "total_value": latest_agg.get("total_value"),
        "contract_years": latest_agg.get("contract_years"),
        "total_guaranteed": latest_agg.get("total_guaranteed"),
        "contract_status": latest_agg.get("contract_status"),
        "contract_type": latest_agg.get("contract_type"),
    }


def _render_contract_snapshot_section(contract_snapshot: dict) -> None:
    st.markdown("### 💸 Contract Snapshot")
    years = pd.to_numeric(contract_snapshot.get("contract_years"), errors="coerce")
    _render_hover_stat_cards(
        [
            ("Cap Hit", _fmt_money(contract_snapshot.get("cap_hit"))),
            ("Avg Salary", _fmt_money(contract_snapshot.get("average_salary"))),
            ("Contract Years", str(int(years)) if pd.notna(years) else "—"),
            ("Guaranteed", _fmt_money(contract_snapshot.get("total_guaranteed"))),
        ],
        columns_per_row=2,
    )
    contract_caption = []
    if contract_snapshot.get("season"):
        contract_caption.append(f"Season: {contract_snapshot.get('season')}")
    if contract_snapshot.get("contract_type"):
        contract_caption.append(f"Type: {contract_snapshot.get('contract_type')}")
    if contract_snapshot.get("contract_status"):
        contract_caption.append(f"Status: {contract_snapshot.get('contract_status')}")
    if contract_caption:
        render_stat_text(" • ".join(contract_caption), small=True)


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


def _player_scouting_report_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
    return (
        "You are an expert NBA scout and analyst. Build a clean player scouting report using only the provided stats.\n"
        "Write in markdown with these exact sections: Offensive Identity, Shooting Profile, Playmaking, Defense, "
        "Rebounding / Physicality, Weaknesses, Best Team Fit, Bottom Line.\n"
        "Use specific stats throughout. Keep it detailed but readable for a serious basketball fan. "
        "Do not mention any other players unless the provided stats directly require a comparison.\n\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_ai_system_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
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
        f"Current team record: {latest_team_record}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n\n"
        "Note: Some advanced metrics are estimates from team-context formulas."
    )


def _player_team_fit_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame, fit_context: str) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
    focus = fit_context.strip() or "General NBA contender fit"
    return (
        "You are an expert NBA roster-building analyst. Evaluate this player's team fit using only the provided stats.\n"
        "Write in markdown with these exact sections: Best Offensive Fit, Best Defensive Fit, Ideal Teammate Types, "
        "Potential Fit Concerns, Best Roles, Bottom Line.\n"
        "Use specific stats throughout. Focus on lineup fit, scalability, usage fit, spacing, playmaking fit, and defensive scheme fit.\n"
        "Do not mention random players unless the fit context directly calls for it.\n\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Fit context:\n{focus}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_what_changed_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame, phases: dict) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
    early = ", ".join(phases.get("early", [])) or "—"
    prime = ", ".join(phases.get("prime", [])) or "—"
    late = ", ".join(phases.get("late", [])) or "—"
    peak = phases.get("peak_season", "—")
    return (
        "You are an expert NBA development analyst. Explain how this player's game changed over the course of his career "
        "using only the provided stats and career-phase labels.\n"
        "Write in markdown with these exact sections: Early Career Identity, Prime Evolution, Late-Career Changes, "
        "What Improved, What Declined or Shifted, Bottom Line.\n"
        "Use specific stats and trends throughout. Focus on role, efficiency, shooting profile, playmaking, defense, and usage changes.\n"
        "Do not mention other players unless directly required by the provided stats.\n\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Career phases:\nEarly Career: {early}\nPrime: {prime}\nLate Career: {late}\nPeak Season: {peak}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_role_recommendation_prompt(
    player_name: str,
    pergame_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    role_summary: dict,
) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
    primary = role_summary.get("primary", "—")
    secondary = role_summary.get("secondary", "—")
    style_tags = ", ".join(role_summary.get("style_tags", []) or []) or "—"
    impact_tags = ", ".join(role_summary.get("impact_tags", []) or []) or "—"
    return (
        "You are an expert NBA role and roster analyst. Recommend this player's best NBA role using only the provided stats and archetype signals.\n"
        "Write in markdown with these exact sections: Best Current Role, Best Long-Term Role, Offensive Responsibilities, "
        "Defensive Responsibilities, Ideal Usage Level, Best Supporting Cast, Role Risks, Bottom Line.\n"
        "Be specific about whether the player projects best as a primary engine, secondary creator, off-ball scorer, connector, "
        "rim protector, floor spacer, defensive stopper, or other concrete NBA role. Use stats throughout.\n\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Archetype summary:\nPrimary: {primary}\nSecondary: {secondary}\nStyle Tags: {style_tags}\nImpact Tags: {impact_tags}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_contract_value_prompt(
    player_name: str,
    pergame_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    contract_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    age_value,
) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_team_record = "—"
    if adv_df is not None and not adv_df.empty and "TEAM_RECORD" in adv_df.columns:
        latest_team_record = adv_df.iloc[-1].get("TEAM_RECORD") or "—"
    contract_snapshot = _contract_snapshot(contract_df, aggregate_df)
    contract_lines = [
        f"Current contract season: {contract_snapshot.get('season') or '—'}",
        f"Cap hit: {_fmt_money(contract_snapshot.get('cap_hit'))}",
        f"Base salary: {_fmt_money(contract_snapshot.get('base_salary'))}",
        f"Total cash: {_fmt_money(contract_snapshot.get('total_cash'))}",
        f"Average salary: {_fmt_money(contract_snapshot.get('average_salary'))}",
        f"Total value: {_fmt_money(contract_snapshot.get('total_value'))}",
        f"Contract years: {contract_snapshot.get('contract_years') or '—'}",
        f"Total guaranteed: {_fmt_money(contract_snapshot.get('total_guaranteed'))}",
        f"Contract type: {contract_snapshot.get('contract_type') or '—'}",
        f"Contract status: {contract_snapshot.get('contract_status') or '—'}",
        f"Current age: {age_value if age_value is not None else '—'}",
    ]
    return (
        "You are an expert NBA salary-cap and player-value analyst. Decide whether this player looks overpaid, fairly paid, or underpaid "
        "based on the provided stats, career context, age, and contract snapshot.\n"
        "Write in markdown with these exact sections: Current Contract Snapshot, Performance vs Pay, Why He Might Be Underpaid, "
        "Why He Might Be Overpaid, Fair Value Range, Final Verdict.\n"
        "Use specific stats and salary numbers throughout. Keep the tone balanced and analytical, not hot-take driven.\n\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Contract snapshot:\n" + "\n".join(contract_lines) + "\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_story_mode_prompt(
    player_name: str,
    pergame_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    archetype_profile: dict | None = None,
    phases: dict | None = None,
) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_ctx = compact_player_context(adv_df if adv_df is not None and not adv_df.empty else pergame_df)
    latest_team_record = latest_ctx.get("team_record") or "Unavailable"
    age_value = latest_ctx.get("age")
    age_text = str(int(age_value)) if isinstance(age_value, (int, float)) and not pd.isna(age_value) else "Unavailable"
    archetype_profile = archetype_profile or {}
    archetype_lines = [
        f"Primary role: {archetype_profile.get('primary') or '—'}",
        f"Secondary role: {archetype_profile.get('secondary') or '—'}",
        f"Style tags: {', '.join(archetype_profile.get('style_tags', [])) if archetype_profile.get('style_tags') else '—'}",
        f"Impact tags: {', '.join(archetype_profile.get('impact_tags', [])) if archetype_profile.get('impact_tags') else '—'}",
    ]
    phase_lines = [
        f"Early career: {', '.join(phases.get('early', [])) if phases else '—'}",
        f"Prime: {', '.join(phases.get('prime', [])) if phases else '—'}",
        f"Late career: {', '.join(phases.get('late', [])) if phases else '—'}",
        f"Peak season: {phases.get('peak_season') if phases else '—'}",
    ]
    return (
        "You are an expert NBA storyteller and analyst. Write a compelling, readable player story that helps a fan quickly understand "
        "who this player is, how he plays, how his career has evolved, and why he matters.\n"
        "Write in markdown with these exact sections: Who He Is, How He Plays, Career Arc, Why He Matters, What Makes Him Unique, Bottom Line.\n"
        "Ground the story in the provided stats and avoid generic filler. Use plain basketball language first, then bring in the stats to support it.\n\n"
        f"Current age: {age_text}\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Archetype profile:\n" + "\n".join(archetype_lines) + "\n\n"
        f"Career phase context:\n" + "\n".join(phase_lines) + "\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_franchise_ranker_prompt(
    player_name: str,
    franchise_name: str,
    pergame_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    archetype_profile: dict | None = None,
) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    latest_ctx = compact_player_context(adv_df if adv_df is not None and not adv_df.empty else pergame_df)
    latest_team_record = latest_ctx.get("team_record") or "Unavailable"
    age_value = latest_ctx.get("age")
    age_text = str(int(age_value)) if isinstance(age_value, (int, float)) and not pd.isna(age_value) else "Unavailable"
    archetype_profile = archetype_profile or {}
    archetype_lines = [
        f"Primary role: {archetype_profile.get('primary') or '—'}",
        f"Secondary role: {archetype_profile.get('secondary') or '—'}",
        f"Style tags: {', '.join(archetype_profile.get('style_tags', [])) if archetype_profile.get('style_tags') else '—'}",
        f"Impact tags: {', '.join(archetype_profile.get('impact_tags', [])) if archetype_profile.get('impact_tags') else '—'}",
    ]
    return (
        "You are an expert NBA franchise historian and analyst. Rank where this player belongs in the history of the specified franchise. "
        "Use the provided player stats and career context plus your basketball knowledge of franchise history, but stay measured and avoid fake certainty.\n"
        "Write in markdown with these exact sections: Franchise Context, Historical Tier, Case For A Higher Rank, Case For A Lower Rank, Estimated Franchise Ranking, Bottom Line.\n"
        "In Estimated Franchise Ranking, give a realistic range like 'Top 3', 'Top 5', 'Top 10', or 'Outside Top 10' and explain why. "
        "Mention the player's role, longevity, peak, and franchise-specific impact. Keep it readable and stat-backed.\n\n"
        f"Franchise: {franchise_name}\n"
        f"Current age: {age_text}\n"
        f"Current team record: {latest_team_record}\n\n"
        f"Archetype profile:\n" + "\n".join(archetype_lines) + "\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
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


def render_player_story_mode_page() -> None:
    st.markdown("## 📖 Player Story Mode")
    player_name = st.session_state.get("player_story_player_name") or "Selected Player"
    story_summary = st.session_state.get("player_story_summary") or "Player story unavailable"
    st.caption(player_name)
    st.caption(story_summary)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_story_mode", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_story_output")
    if report:
        st.markdown(report)
    else:
        st.info("No player story is available right now. Generate one from the player's Stats page first.")


def render_player_franchise_ranker_page() -> None:
    st.markdown("## 🏛️ Franchise Ranker")
    player_name = st.session_state.get("player_franchise_ranker_player_name") or "Selected Player"
    franchise_summary = st.session_state.get("player_franchise_ranker_summary") or "Franchise context unavailable"
    st.caption(player_name)
    st.caption(franchise_summary)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_franchise_ranker", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_franchise_ranker_output")
    if report:
        st.markdown(report)
    else:
        st.info("No franchise ranking is available right now. Generate one from the player's Stats page first.")


def render_player_what_changed_page() -> None:
    st.markdown("## 🔄 What Changed?")
    player_name = st.session_state.get("player_what_changed_player_name") or "Selected Player"
    phase_summary = st.session_state.get("player_what_changed_phase_summary") or "Career phases unavailable"
    st.caption(player_name)
    st.caption(phase_summary)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_what_changed", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_what_changed_output")
    if report:
        st.markdown(report)
    else:
        st.info("No 'What Changed?' report is available right now. Generate one from the player's Stats page first.")


def render_player_role_recommendation_page() -> None:
    st.markdown("## 🎯 AI Role Recommendation")
    player_name = st.session_state.get("player_role_player_name") or "Selected Player"
    archetype_summary = st.session_state.get("player_role_archetype_summary") or "Archetype context unavailable"
    st.caption(player_name)
    st.caption(archetype_summary)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_role_recommendation", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_role_output")
    if report:
        st.markdown(report)
    else:
        st.info("No role recommendation is available right now. Generate one from the player's Stats page first.")


def render_player_team_fit_page() -> None:
    st.markdown("## 🧩 AI Team Fit Analyzer")
    player_name = st.session_state.get("player_team_fit_player_name") or "Selected Player"
    fit_context = st.session_state.get("player_team_fit_context") or "General NBA contender fit"
    st.caption(player_name)
    st.caption(f"Fit context: {fit_context}")

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Stats", key="back_to_stats_from_team_fit", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "📊 Stats"
            st.rerun()

    report = st.session_state.get("player_team_fit_output")
    if report:
        st.markdown(report)
    else:
        st.info("No team fit analysis is available right now. Generate one from the player's Stats page first.")


def render_player_contract_value_page() -> None:
    st.markdown("## 💸 Contract Value Analyzer")
    player_name = st.session_state.get("player_contract_value_player_name") or "Selected Player"
    contract_summary = st.session_state.get("player_contract_value_summary") or "Contract snapshot unavailable"
    st.caption(player_name)
    st.caption(contract_summary)

    top_left, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Value & Props", key="back_to_stats_from_contract_value", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "💸 Value & Props"
            st.rerun()

    report = st.session_state.get("player_contract_value_output")
    if report:
        st.markdown(report)
    else:
        st.info("No contract value analysis is available right now. Generate one from the player's Value & Props page first.")


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
        _render_headshot_image(headshot_url, 220, player.get("full_name", "Player"))
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
        _render_headshot_image(headshot_url, 220, player.get("full_name", "Player"))
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


def _normalize_name_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip().lower()


def _cluster_archetype_label(row: pd.Series) -> str:
    family = _position_family_label(row.get("POSITION"))
    ppg = pd.to_numeric(row.get("PPG"), errors="coerce")
    rpg = pd.to_numeric(row.get("RPG"), errors="coerce")
    apg = pd.to_numeric(row.get("APG"), errors="coerce")
    spg = pd.to_numeric(row.get("STL"), errors="coerce")
    bpg = pd.to_numeric(row.get("BLK"), errors="coerce")
    three_pa = pd.to_numeric(row.get("3PA"), errors="coerce")
    three_pct = pd.to_numeric(row.get("3P%"), errors="coerce")
    usg = pd.to_numeric(row.get("USG"), errors="coerce")
    ast_pct = pd.to_numeric(row.get("AST_PCT"), errors="coerce")
    reb_pct = pd.to_numeric(row.get("REB_PCT"), errors="coerce")
    ftr = pd.to_numeric(row.get("FTr"), errors="coerce")

    is_big = family == "Big"
    is_guard = family == "Guard"
    is_wing = family == "Wing"

    if is_big and pd.notna(ast_pct) and ast_pct >= 24 and pd.notna(apg) and apg >= 5:
        return "Playmaking Big"
    if is_big and ((pd.notna(bpg) and bpg >= 1.8) or (pd.notna(reb_pct) and reb_pct >= 16 and pd.notna(bpg) and bpg >= 1.2)):
        return "Rim-Protecting Big"
    if is_big and pd.notna(three_pa) and three_pa >= 4 and pd.notna(three_pct) and three_pct >= 34:
        return "Stretch Big"
    if is_big and pd.notna(rpg) and rpg >= 9 and pd.notna(ppg) and ppg >= 16:
        return "Interior Big"
    if (is_guard or is_wing) and pd.notna(usg) and usg >= 29 and pd.notna(ast_pct) and ast_pct >= 24:
        return "Offensive Engine"
    if (is_guard or is_wing) and pd.notna(ppg) and ppg >= 22 and pd.notna(usg) and usg >= 27:
        return "Shot Creator"
    if (is_guard or is_wing) and pd.notna(three_pa) and three_pa >= 6 and pd.notna(three_pct) and three_pct >= 37 and pd.notna(usg) and usg <= 24:
        return "Movement Shooter"
    if is_wing and pd.notna(three_pa) and three_pa >= 4 and pd.notna(three_pct) and three_pct >= 35 and ((pd.notna(spg) and spg >= 1.0) or (pd.notna(bpg) and bpg >= 0.8)):
        return "3-and-D Wing"
    if (is_guard or is_wing) and pd.notna(ftr) and ftr >= 0.28 and pd.notna(ppg) and ppg >= 18:
        return "Slashing Creator"
    if (is_guard or is_wing) and pd.notna(apg) and apg >= 5 and pd.notna(ast_pct) and ast_pct >= 20:
        return "Secondary Creator"
    if is_big:
        return "Balanced Big"
    if is_wing:
        return "Balanced Wing"
    if is_guard:
        return "Balanced Guard"
    return "Utility Role"


def _render_hover_stat_cards(cards: list[tuple[str, str]], columns_per_row: int | None = None) -> None:
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
          font-size: clamp(1.3rem, 2vw, 2rem);
          font-weight: 700;
          line-height: 1.1;
          overflow-wrap: anywhere;
          word-break: break-word;
        }
        .hover-stat-value.compact {
          font-size: clamp(1.05rem, 1.5vw, 1.45rem);
          line-height: 1.2;
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
    per_row = max(1, columns_per_row or len(cards))
    for start in range(0, len(cards), per_row):
        row_cards = cards[start:start + per_row]
        cols = st.columns(len(row_cards))
        for col, (label, value) in zip(cols, row_cards):
            value_text = "" if value is None else str(value)
            compact_class = " compact" if len(value_text) >= 10 else ""
            with col:
                st.markdown(
                    f"""
                    <div class="hover-stat-card">
                      <div class="hover-stat-label">{_label_with_tooltip(label)}</div>
                      <div class="hover-stat-value{compact_class}">{html.escape(value_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_impact_summary_cards(score: float, strongest_driver: str, strongest_value: float | None, tier: str) -> None:
    st.markdown(
        """
        <style>
        .impact-summary-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 14px 16px;
          min-height: 118px;
        }
        .impact-summary-label {
          color: rgba(255,255,255,0.72);
          font-size: 0.84rem;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .impact-summary-value {
          color: #f9fafb;
          font-size: 2rem;
          font-weight: 700;
          line-height: 1.08;
          word-break: break-word;
        }
        .impact-summary-value.is-tier {
          font-size: 1.08rem;
          line-height: 1.28;
          font-weight: 650;
          white-space: normal;
        }
        .impact-summary-delta {
          margin-top: 8px;
          color: #86efac;
          font-size: 0.82rem;
          font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    items = [
        ("Impact Index", f"{score:.1f}", None, False),
        (
            "Strongest Driver",
            strongest_driver or "—",
            f"{strongest_value:.1f}" if strongest_value is not None and pd.notna(strongest_value) else None,
            False,
        ),
        ("Tier", tier or "—", None, True),
    ]
    for col, (label, value, delta, is_tier) in zip(cols, items):
        with col:
            tier_class = " is-tier" if is_tier else ""
            delta_html = f'<div class="impact-summary-delta">↑ {html.escape(delta)}</div>' if delta else ""
            st.markdown(
                f"""
                <div class="impact-summary-card">
                  <div class="impact-summary-label">{html.escape(label)}</div>
                  <div class="impact-summary-value{tier_class}">{html.escape(value)}</div>
                  {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


_PROP_METRIC_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "3PM": "FG3M",
    "Blocks": "BLK",
    "Steals": "STL",
    "PRA": "PRA",
}


def _prop_thresholds(metric_label: str, season_avg: float | None) -> list[float]:
    defaults = {
        "Points": [15, 20, 25, 30, 35],
        "Rebounds": [5, 8, 10, 12, 15],
        "Assists": [4, 6, 8, 10, 12],
        "3PM": [1, 2, 3, 4, 5],
        "Blocks": [1, 2, 3, 4],
        "Steals": [1, 2, 3, 4],
        "PRA": [20, 25, 30, 35, 40, 45],
    }
    base = defaults.get(metric_label, [10, 15, 20])
    if season_avg is None or pd.isna(season_avg):
        return base
    near = round(float(season_avg) / 5.0) * 5.0 if metric_label in {"Points", "PRA"} else round(float(season_avg))
    if near > 0 and near not in base:
        base.append(near)
    return sorted(set(base))


def _prop_consistency_label(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Unknown"
    if score >= 78:
        return "Very Stable"
    if score >= 62:
        return "Fairly Stable"
    if score >= 45:
        return "Moderate Swing"
    return "Highly Volatile"


def _prop_volatility_label(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Unknown"
    if score >= 55:
        return "High Variance"
    if score >= 35:
        return "Noticeable Variance"
    if score >= 18:
        return "Manageable Variance"
    return "Low Variance"


def _render_trend_summary_cards(trend_label: str, last3_delta: float | None, last5_delta: float | None, last10_delta: float | None) -> None:
    st.markdown(
        """
        <style>
        .trend-summary-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 14px 16px;
          min-height: 112px;
        }
        .trend-summary-label {
          color: rgba(255,255,255,0.72);
          font-size: 0.84rem;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .trend-summary-value {
          color: #f9fafb;
          font-size: clamp(1.3rem, 2vw, 1.9rem);
          font-weight: 700;
          line-height: 1.12;
          overflow-wrap: anywhere;
          word-break: break-word;
        }
        .trend-summary-value.is-text {
          font-size: clamp(1.0rem, 1.5vw, 1.35rem);
          line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    items = [
        ("Trend", trend_label or "—", True),
        ("Last 3 vs Season", "—" if pd.isna(last3_delta) else f"{'+' if last3_delta > 0 else ''}{last3_delta:.1f}", False),
        ("Last 5 vs Season", "—" if pd.isna(last5_delta) else f"{'+' if last5_delta > 0 else ''}{last5_delta:.1f}", False),
        ("Last 10 vs Season", "—" if pd.isna(last10_delta) else f"{'+' if last10_delta > 0 else ''}{last10_delta:.1f}", False),
    ]
    cols = st.columns(len(items))
    for col, (label, value, is_text) in zip(cols, items):
        extra_class = " is-text" if is_text else ""
        with col:
            st.markdown(
                f"""
                <div class="trend-summary-card">
                  <div class="trend-summary-label">{html.escape(label)}</div>
                  <div class="trend-summary-value{extra_class}">{html.escape(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _recent_trend_label(last_5_delta: float | None, last_10_delta: float | None) -> str:
    if last_5_delta is None or pd.isna(last_5_delta):
        return "No clear trend"
    if last_5_delta >= 4:
        return "Red Hot"
    if last_5_delta >= 2:
        return "Heating Up"
    if last_5_delta <= -4:
        return "Ice Cold"
    if last_5_delta <= -2:
        return "Cooling Off"
    if last_10_delta is not None and pd.notna(last_10_delta) and abs(last_10_delta) >= 2:
        return "Slow Shift"
    return "Steady"


def _render_without_teammate_split(player: dict, latest_season_id: str, logs: pd.DataFrame, metric_label: str, stat_col: str, prop_line: float) -> None:
    player_team_id = player.get("team_id")
    if player_team_id is None:
        bd_player = get_balldontlie_player(player["id"])
        player_team_id = (bd_player or {}).get("team_id")
    if player_team_id is None:
        return

    teammates = [
        t for t in get_balldontlie_team_players(int(player_team_id))
        if t.get("id") and str(t.get("id")) != str(player.get("id"))
    ]
    if not teammates:
        return

    teammate_options = sorted(
        teammates,
        key=lambda t: ((t.get("position") or "Z"), t.get("full_name") or ""),
    )
    teammate_lookup = {t["full_name"]: t for t in teammate_options if t.get("full_name")}

    with st.expander("Usage Without Key Teammates", expanded=False):
        teammate_name = st.selectbox(
            "Select teammate to split by",
            list(teammate_lookup.keys()),
            key=f"without_teammate_{player.get('id')}_{metric_label}",
        )
        teammate = teammate_lookup.get(teammate_name)
        if not teammate:
            return

        teammate_logs = get_balldontlie_player_game_logs(
            teammate["id"],
            [latest_season_id],
            player_name=teammate.get("full_name"),
            player_source="balldontlie",
            season_type="Regular Season",
        )
        if teammate_logs is None or teammate_logs.empty or "GAME_ID" not in teammate_logs.columns:
            st.info("That teammate's game logs are not available right now.")
            return

        stat_logs = logs.copy()
        stat_logs["Value"] = pd.to_numeric(stat_logs.get(stat_col), errors="coerce")
        stat_logs = stat_logs.dropna(subset=["GAME_ID", "Value"])
        if stat_logs.empty:
            return

        teammate_game_ids = set(pd.to_numeric(teammate_logs.get("GAME_ID"), errors="coerce").dropna().astype(int).tolist())
        stat_logs["GAME_ID_INT"] = pd.to_numeric(stat_logs["GAME_ID"], errors="coerce")
        with_teammate = stat_logs[stat_logs["GAME_ID_INT"].isin(teammate_game_ids)].copy()
        without_teammate = stat_logs[~stat_logs["GAME_ID_INT"].isin(teammate_game_ids)].copy()

        with_avg = float(with_teammate["Value"].mean()) if not with_teammate.empty else np.nan
        without_avg = float(without_teammate["Value"].mean()) if not without_teammate.empty else np.nan
        with_over = float((with_teammate["Value"] > prop_line).mean() * 100.0) if not with_teammate.empty else np.nan
        without_over = float((without_teammate["Value"] > prop_line).mean() * 100.0) if not without_teammate.empty else np.nan
        with_fga = float(pd.to_numeric(with_teammate.get("FGA"), errors="coerce").mean()) if not with_teammate.empty and "FGA" in with_teammate.columns else np.nan
        without_fga = float(pd.to_numeric(without_teammate.get("FGA"), errors="coerce").mean()) if not without_teammate.empty and "FGA" in without_teammate.columns else np.nan
        with_min = float(pd.to_numeric(with_teammate.get("MIN"), errors="coerce").mean()) if not with_teammate.empty and "MIN" in with_teammate.columns else np.nan
        without_min = float(pd.to_numeric(without_teammate.get("MIN"), errors="coerce").mean()) if not without_teammate.empty and "MIN" in without_teammate.columns else np.nan

        render_stat_text(f"Splitting {metric_label} by games with vs without {teammate_name}.", small=True)
        _render_hover_stat_cards(
            [
                ("With Avg", f"{with_avg:.1f}" if pd.notna(with_avg) else "—"),
                ("Without Avg", f"{without_avg:.1f}" if pd.notna(without_avg) else "—"),
                (f"Over {prop_line:.1f} With", f"{with_over:.1f}%" if pd.notna(with_over) else "—"),
                (f"Over {prop_line:.1f} Without", f"{without_over:.1f}%" if pd.notna(without_over) else "—"),
                ("Games Split", f"{len(with_teammate)} / {len(without_teammate)}"),
            ],
            columns_per_row=2,
        )

        context_parts = []
        if pd.notna(with_fga) and pd.notna(without_fga):
            fga_delta = without_fga - with_fga
            context_parts.append(f"FGA shifts by {fga_delta:+.1f} without {teammate_name}")
        if pd.notna(with_min) and pd.notna(without_min):
            min_delta = without_min - with_min
            context_parts.append(f"minutes shift by {min_delta:+.1f}")
        if context_parts:
            render_stat_text(". ".join(context_parts) + ".", small=True)
        render_stat_text(
            "This split uses balldontlie game logs: 'with' means the teammate logged stats in that game, and 'without' means they did not appear in the stat feed for that game.",
            small=True,
        )


def _render_prop_context_dashboard(player: dict, latest_season_id: str) -> None:
    if not latest_season_id:
        return

    logs = get_balldontlie_player_game_logs(
        player["id"],
        [latest_season_id],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        season_type="Regular Season",
    )
    if logs is None or logs.empty:
        st.markdown("### 🎯 Prop Research")
        st.info("Recent player game logs were not available from balldontlie for this season yet.")
        return

    logs = logs.copy()
    for col in ["PTS", "REB", "AST", "FG3M", "BLK", "STL", "MIN"]:
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors="coerce")
    logs["PRA"] = logs[["PTS", "REB", "AST"]].sum(axis=1, min_count=3)
    if "GAME_DATE" in logs.columns:
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        logs = logs.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    st.markdown("### 🎯 Prop Research")
    render_stat_text("A research-first prop context view from latest-season balldontlie game logs. Use this to study trend, hit rates, consistency, and volatility before making any call.", small=True)

    metric_label = st.selectbox(
        "Prop stat",
        list(_PROP_METRIC_MAP.keys()),
        index=0,
        key=f"prop_metric_{player.get('id')}",
    )
    stat_col = _PROP_METRIC_MAP[metric_label]
    stat_series = pd.to_numeric(logs.get(stat_col), errors="coerce")
    valid = stat_series.dropna()
    if valid.empty:
        st.info("That prop stat is not available in the returned game logs.")
        return

    season_avg = float(valid.mean())
    last5_avg = float(valid.head(5).mean()) if len(valid) >= 1 else np.nan
    last10_avg = float(valid.head(10).mean()) if len(valid) >= 1 else np.nan
    season_high = float(valid.max())
    season_low = float(valid.min())
    std_dev = float(valid.std(ddof=0)) if len(valid) > 1 else 0.0
    cv = (std_dev / season_avg) if season_avg and season_avg > 0 else np.nan
    volatility_score = min(max((cv or 0.0) * 100.0, 0.0), 100.0) if pd.notna(cv) else np.nan
    consistency_score = 100.0 - volatility_score if pd.notna(volatility_score) else np.nan
    median_value = float(valid.median()) if len(valid) else np.nan
    hit_band = float(((valid >= season_avg - std_dev) & (valid <= season_avg + std_dev)).mean() * 100.0) if len(valid) else np.nan
    consistency_label = _prop_consistency_label(consistency_score)
    volatility_label = _prop_volatility_label(volatility_score)

    summary_cols = st.columns(5)
    summary_values = [
        ("Season Avg", season_avg),
        ("Last 5 Avg", last5_avg),
        ("Last 10 Avg", last10_avg),
        ("Season High", season_high),
        ("Season Low", season_low),
    ]
    for col, (label, value) in zip(summary_cols, summary_values):
        with col:
            st.metric(label, f"{value:.1f}" if pd.notna(value) else "—")

    context_cols = st.columns(2)
    with context_cols[0]:
        st.metric("Consistency Score", f"{consistency_score:.1f}" if pd.notna(consistency_score) else "—")
        render_stat_text(
            f"{consistency_label}. Higher is steadier, based on how tightly the game-by-game results stay around the season average.",
            small=True,
        )
    with context_cols[1]:
        st.metric("Volatility Score", f"{volatility_score:.1f}" if pd.notna(volatility_score) else "—")
        render_stat_text(
            f"{volatility_label}. Higher means wider game-to-game swings, which usually means more risk on a prop line.",
            small=True,
        )

    spread_cols = st.columns(3)
    spread_values = [
        ("Median", median_value),
        ("Std Dev", std_dev),
        ("Within 1 Std Dev", hit_band),
    ]
    for col, (label, value) in zip(spread_cols, spread_values):
        with col:
            if label == "Within 1 Std Dev":
                st.metric(label, f"{value:.1f}%" if pd.notna(value) else "—")
            else:
                st.metric(label, f"{value:.1f}" if pd.notna(value) else "—")
    render_stat_text(
        "Median helps show the typical outcome, standard deviation measures game-to-game spread, and the 1-standard-deviation band shows how often the player stays in their normal range.",
        small=True,
    )

    thresholds = _prop_thresholds(metric_label, season_avg)
    st.markdown("#### Over / Under Hit Rates")
    line_choices = sorted({float(x) for x in thresholds} | {round(float(season_avg), 1)}) if pd.notna(season_avg) else [float(x) for x in thresholds]
    default_line = round(float(season_avg), 1) if pd.notna(season_avg) else float(line_choices[0])
    line_col, custom_line_col = st.columns([1.3, 1])
    with line_col:
        selected_line = st.selectbox(
            "Quick line",
            line_choices,
            index=min(range(len(line_choices)), key=lambda i: abs(line_choices[i] - default_line)),
            key=f"prop_quick_line_{player.get('id')}_{metric_label}",
            format_func=lambda x: f"{x:.1f}",
        )
    with custom_line_col:
        prop_line = st.number_input(
            "Or set a custom line",
            min_value=0.0,
            value=float(selected_line),
            step=0.5,
            key=f"prop_custom_line_{player.get('id')}_{metric_label}",
        )

    recent_5 = valid.head(5)
    recent_10 = valid.head(10)
    recent_3 = valid.head(3)
    last3_avg = float(recent_3.mean()) if len(recent_3) else np.nan
    last5_delta = float(last5_avg - season_avg) if pd.notna(last5_avg) and pd.notna(season_avg) else np.nan
    last10_delta = float(last10_avg - season_avg) if pd.notna(last10_avg) and pd.notna(season_avg) else np.nan
    last3_delta = float(last3_avg - season_avg) if pd.notna(last3_avg) and pd.notna(season_avg) else np.nan
    trend_label = _recent_trend_label(last5_delta, last10_delta)
    if len(recent_5):
        above_avg_count = int((recent_5 > season_avg).sum()) if pd.notna(season_avg) else 0
        below_avg_count = int((recent_5 < season_avg).sum()) if pd.notna(season_avg) else 0
    else:
        above_avg_count = 0
        below_avg_count = 0

    st.markdown("#### Recent Trend Analyzer")
    _render_trend_summary_cards(trend_label, last3_delta, last5_delta, last10_delta)
    render_stat_text(
        f"Last 5 sample: {above_avg_count} games above the season average and {below_avg_count} below it. "
        "Positive deltas mean the player is trending above their normal season level.",
        small=True,
    )

    _render_without_teammate_split(player, latest_season_id, logs, metric_label, stat_col, float(prop_line))

    def _rate(series: pd.Series, mode: str) -> float:
        if len(series) == 0:
            return np.nan
        if mode == "over":
            return float((series > prop_line).mean() * 100.0)
        if mode == "under":
            return float((series < prop_line).mean() * 100.0)
        return float((series == prop_line).mean() * 100.0)

    ou_cols = st.columns(5)
    over_season = _rate(valid, "over")
    over_recent = _rate(recent_10, "over")
    under_season = _rate(valid, "under")
    under_recent = _rate(recent_10, "under")
    push_season = _rate(valid, "push")
    ou_values = [
        (f"Season Over {prop_line:.1f}", over_season),
        (f"Last 10 Over {prop_line:.1f}", over_recent),
        (f"Season Under {prop_line:.1f}", under_season),
        (f"Last 10 Under {prop_line:.1f}", under_recent),
        ("Season Push", push_season),
    ]
    for col, (label, value) in zip(ou_cols, ou_values):
        with col:
            st.metric(label, f"{value:.1f}%" if pd.notna(value) else "—")

    render_stat_text(
        "Over is counted when the player clears the line, under when they finish below it, and push when they land exactly on it.",
        small=True,
    )

    st.markdown("#### Common Threshold Hit Rates")
    hit_rows = []
    for threshold in thresholds:
        season_hit = float((valid >= threshold).mean() * 100.0) if len(valid) else np.nan
        recent_hit = float((recent_10 >= threshold).mean() * 100.0) if len(recent_10) else np.nan
        recent5_hit = float((recent_5 >= threshold).mean() * 100.0) if len(recent_5) else np.nan
        hit_rows.append({
            "Line": f"{threshold}+",
            "Season Hit Rate": season_hit,
            "Last 10 Hit Rate": recent_hit,
            "Last 5 Hit Rate": recent5_hit,
        })
    hit_df = pd.DataFrame(hit_rows)
    render_html_table(hit_df, number_cols=[], percent_cols=["Season Hit Rate", "Last 10 Hit Rate", "Last 5 Hit Rate"], max_height_px=260)

    chart_df = logs.copy()
    chart_df["Value"] = pd.to_numeric(chart_df.get(stat_col), errors="coerce")
    chart_df = chart_df.dropna(subset=["Value"])
    if not chart_df.empty:
        chart_df = chart_df.sort_values("GAME_DATE")
        chart_df["5-Game Avg"] = chart_df["Value"].rolling(5, min_periods=1).mean()
        fig = px.line(
            chart_df,
            x="GAME_DATE",
            y=["Value", "5-Game Avg"],
            title=f"{metric_label} Trend",
            labels={"GAME_DATE": "Game Date", "value": metric_label, "variable": "Series"},
        )
        fig.add_hline(y=season_avg, line_dash="dash", line_color="rgba(255,255,255,0.5)", annotation_text="Season average")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

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

        cluster = league_df.copy()
        cluster["PPG"] = pd.to_numeric(cluster.get("pts"), errors="coerce")
        cluster["RPG"] = pd.to_numeric(cluster.get("reb"), errors="coerce")
        cluster["APG"] = pd.to_numeric(cluster.get("ast"), errors="coerce")
        cluster["STL"] = pd.to_numeric(cluster.get("stl"), errors="coerce")
        cluster["BLK"] = pd.to_numeric(cluster.get("blk"), errors="coerce")
        cluster["3PA"] = pd.to_numeric(cluster.get("fg3a"), errors="coerce")
        cluster["3P%"] = pd.to_numeric(cluster.get("fg3_pct"), errors="coerce") * 100.0
        cluster["USG"] = pd.to_numeric(cluster.get("usg_pct"), errors="coerce") * 100.0
        cluster["AST_PCT"] = pd.to_numeric(cluster.get("ast_pct"), errors="coerce") * 100.0
        cluster["REB_PCT"] = pd.to_numeric(cluster.get("reb_pct"), errors="coerce") * 100.0
        cluster["TS"] = pd.to_numeric(cluster.get("ts_pct"), errors="coerce") * 100.0
        cluster["GP"] = pd.to_numeric(cluster.get("gp"), errors="coerce")
        cluster["MIN"] = pd.to_numeric(cluster.get("min"), errors="coerce")
        cluster["FGA"] = pd.to_numeric(cluster.get("fga"), errors="coerce")
        cluster["FTA"] = pd.to_numeric(cluster.get("fta"), errors="coerce")
        cluster["FTr"] = np.where(cluster["FGA"] > 0, cluster["FTA"] / cluster["FGA"], np.nan)
        cluster["Position Family"] = cluster["POSITION"].apply(_position_family_label)
        cluster = cluster.dropna(subset=["PPG", "RPG", "APG", "USG", "AST_PCT", "REB_PCT"])
        cluster = cluster[(cluster["GP"] >= 12) & (cluster["MIN"] >= 14) & (cluster["PPG"] >= 6)].copy()

        if not cluster.empty:
            cluster["Creation Load"] = (cluster["USG"] * 0.62) + (cluster["AST_PCT"] * 0.38)
            cluster["Interior Impact"] = (
                cluster["REB_PCT"] * 1.1
                + cluster["BLK"].fillna(0) * 10.0
                + np.where(cluster["Position Family"] == "Big", 7.0, 0.0)
                - cluster["3PA"].fillna(0) * 0.8
            )
            cluster["Cluster"] = cluster.apply(_cluster_archetype_label, axis=1)
            cluster["__name_key"] = cluster["PLAYER_NAME"].apply(_normalize_name_key)
            selected_key = _normalize_name_key(player_name)
            player_cluster_row = cluster[cluster["__name_key"] == selected_key].copy()

            st.markdown("#### 🧬 Archetype Cluster Map")
            st.caption("See where this player sits among league archetypes by creation load and interior impact. Color groups are stat-profile clusters from the latest balldontlie season pool.")

            fig = px.scatter(
                cluster,
                x="Creation Load",
                y="Interior Impact",
                size="PPG",
                color="Cluster",
                hover_name="PLAYER_NAME",
                hover_data={
                    "Position Family": True,
                    "PPG": ":.1f",
                    "RPG": ":.1f",
                    "APG": ":.1f",
                    "TS": ":.1f",
                    "Creation Load": ":.1f",
                    "Interior Impact": ":.1f",
                },
                title="Latest-Season Archetype Clusters",
                labels={"Creation Load": "Creation Load", "Interior Impact": "Interior Impact"},
                opacity=0.58,
            )
            if not player_cluster_row.empty:
                fig.add_trace(go.Scatter(
                    x=player_cluster_row["Creation Load"],
                    y=player_cluster_row["Interior Impact"],
                    mode="markers+text",
                    text=[player_name],
                    textposition="top center",
                    marker=dict(size=20, color="#f59e0b", line=dict(color="white", width=1.6), symbol="star"),
                    name=player_name,
                ))
                cluster_name = player_cluster_row.iloc[0].get("Cluster") or "—"
                peers = (
                    cluster[cluster["Cluster"] == cluster_name]
                    .sort_values("PPG", ascending=False)["PLAYER_NAME"]
                    .head(6)
                    .tolist()
                )
                if peers:
                    st.caption(f"Current cluster: **{cluster_name}**. Nearby archetype examples: {', '.join(peers)}.")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
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

    peak_season = str(phases.get("peak_season", "")).strip()
    if peak_season and "SEASON_ID" in adv.columns:
        peak_subset = adv[adv["SEASON_ID"].astype(str) == peak_season].copy()
        if not peak_subset.empty:
            options["Peak Season"] = {
                "summary": peak_subset.iloc[-1],
                "table_df": peak_subset,
            }
            captions["Peak Season"] = f"Showing the AI-labeled peak season ({peak_season})."

    phase_labels = [
        ("Prime", phases.get("prime", [])),
        ("Early Career", phases.get("early", [])),
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

    ordered_labels = ["Latest Season", "Peak Season", "Prime", "Full Career", "Early Career", "Late Career"]
    ordered_options = {label: options[label] for label in ordered_labels if label in options}
    ordered_captions = {label: captions[label] for label in ordered_labels if label in captions}
    return ordered_options, ordered_captions


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
    adv = _add_team_record_column(adv)

    full_adv = adv

    phase_player_key = _player_phase_state_key(player)
    if st.session_state.get("player_ai_output_signature") != phase_player_key:
        st.session_state.pop("player_story_output", None)
        st.session_state.pop("player_franchise_ranker_output", None)
        st.session_state.pop("player_scouting_report_output", None)
        st.session_state.pop("player_what_changed_output", None)
        st.session_state.pop("player_role_output", None)
        st.session_state.pop("player_team_fit_output", None)
        st.session_state.pop("player_contract_value_output", None)
        st.session_state.pop("player_story_player_name", None)
        st.session_state.pop("player_story_summary", None)
        st.session_state.pop("player_franchise_ranker_player_name", None)
        st.session_state.pop("player_franchise_ranker_summary", None)
        st.session_state.pop("player_report_player_name", None)
        st.session_state.pop("player_what_changed_player_name", None)
        st.session_state.pop("player_role_player_name", None)
        st.session_state.pop("player_team_fit_player_name", None)
        st.session_state.pop("player_contract_value_player_name", None)
        st.session_state.pop("player_what_changed_phase_summary", None)
        st.session_state.pop("player_role_archetype_summary", None)
        st.session_state.pop("player_team_fit_context", None)
        st.session_state.pop("player_contract_value_summary", None)
        st.session_state["player_ai_output_signature"] = phase_player_key
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
        contract_df = get_balldontlie_player_contracts(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        contract_agg_df = get_balldontlie_player_contract_aggregates(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        contract_snapshot = _contract_snapshot(contract_df, contract_agg_df)
        headshot_url = get_nba_headshot_url(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        hero_col, stats_col = st.columns([1, 3])
        with hero_col:
            _render_headshot_image(headshot_url, 170, player.get("full_name", "Player"))
            st.caption(player.get("full_name", ""))
        with stats_col:
            if selected_summary_label and selected_summary_label in summary_captions:
                render_stat_text(summary_captions[selected_summary_label], small=True)
            latest_team_record = latest.get("TEAM_RECORD")
            if pd.notna(latest_team_record) and latest_team_record:
                render_stat_text(f"Current team record: {latest_team_record}", small=True)
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
            impact_index = compute_impact_index(player["full_name"], display_adv)
            if impact_index:
                st.markdown("### 🧠 Impact Index")
                render_stat_text("A blended score of scoring, efficiency, playmaking, defense, rebounding, and team winning for the selected career window.", small=True)
                breakdown_df = impact_index.get("breakdown", pd.DataFrame())
                if not breakdown_df.empty:
                    top_component = breakdown_df.sort_values("Score", ascending=False).iloc[0]
                    strongest_driver = str(top_component["Component"])
                    strongest_value = pd.to_numeric(top_component["Score"], errors="coerce")
                else:
                    strongest_driver = "—"
                    strongest_value = None
                _render_impact_summary_cards(
                    impact_index["score"],
                    strongest_driver,
                    strongest_value,
                    impact_index.get("tier", "—"),
                )
                if not breakdown_df.empty:
                    render_html_table(
                        breakdown_df[["Component", "Score", "Weight", "Why"]],
                        number_cols=["Score", "Weight"],
                        max_height_px=320,
                    )
                season_scores = impact_index.get("season_scores", pd.DataFrame())
                if season_scores is not None and len(season_scores) >= 2:
                    fig = px.line(
                        season_scores,
                        x="Season",
                        y="Impact Score",
                        markers=True,
                        title="Impact Index Across the Selected Window",
                    )
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

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

            with st.expander("Player Story Mode", expanded=False):
                if model:
                    role_profile = archetype if isinstance(archetype, dict) else {}
                    phase_store = st.session_state.get("career_phases_by_player", {})
                    phases = phase_store.get(_player_phase_state_key(player))
                    story_summary = (
                        f"Primary: {role_profile.get('primary', '—')} • "
                        f"Style: {', '.join(role_profile.get('style_tags', [])[:2]) if role_profile.get('style_tags') else '—'} • "
                        f"Peak: {phases.get('peak_season', '—') if phases else '—'}"
                    )
                    st.caption("Generate a dedicated page that tells this player's story in plain basketball language backed by the stats.")
                    if st.button("Generate Player Story", key="generate_player_story_mode", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = full_adv if full_adv is not None and not full_adv.empty else adv
                        prompt = _player_story_mode_prompt(
                            player["full_name"],
                            pergame_for_summary,
                            adv_for_summary,
                            role_profile,
                            phases,
                        )
                        with st.spinner("Writing player story…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_story_output"] = text or "No response."
                                st.session_state["player_story_player_name"] = player["full_name"]
                                st.session_state["player_story_summary"] = story_summary
                                st.session_state["player_report_mode"] = "story-mode"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_story_output"):
                        st.caption("A player story is already available.")
                        if st.button("Open Player Story Page", key="open_player_story_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "story-mode"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

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

            with st.expander("Franchise Ranker", expanded=False):
                if model:
                    default_franchise = player.get("team_name") or player.get("team_abbreviation") or ""
                    franchise_name = st.text_input(
                        "Franchise context",
                        value=default_franchise,
                        placeholder="Example: Los Angeles Lakers",
                        key="player_franchise_ranker_input",
                    )
                    role_profile = archetype if isinstance(archetype, dict) else {}
                    st.caption("Estimate where this player belongs in the history of a franchise, with stats and historical context.")
                    if st.button("Generate Franchise Ranking", key="generate_player_franchise_ranker", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = full_adv if full_adv is not None and not full_adv.empty else adv
                        resolved_franchise = (franchise_name or default_franchise or "This Franchise").strip()
                        prompt = _player_franchise_ranker_prompt(
                            player["full_name"],
                            resolved_franchise,
                            pergame_for_summary,
                            adv_for_summary,
                            role_profile,
                        )
                        with st.spinner("Ranking franchise legacy…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_franchise_ranker_output"] = text or "No response."
                                st.session_state["player_franchise_ranker_player_name"] = player["full_name"]
                                st.session_state["player_franchise_ranker_summary"] = f"Franchise: {resolved_franchise}"
                                st.session_state["player_report_mode"] = "franchise-ranker"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_franchise_ranker_output"):
                        st.caption("A franchise ranking is already available.")
                        if st.button("Open Franchise Ranking Page", key="open_player_franchise_ranker_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "franchise-ranker"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("What Changed?", expanded=False):
                if model:
                    phase_store = st.session_state.get("career_phases_by_player", {})
                    phases = phase_store.get(_player_phase_state_key(player))
                    if phases and full_adv is not None and not full_adv.empty:
                        phase_summary = (
                            f"Early: {', '.join(phases.get('early', [])) or '—'} • "
                            f"Prime: {', '.join(phases.get('prime', [])) or '—'} • "
                            f"Late: {', '.join(phases.get('late', [])) or '—'}"
                        )
                        st.caption("Generate a dedicated page explaining how this player's game changed from early career to prime to late career.")
                        if st.button("Generate What Changed Report", key="generate_player_what_changed", use_container_width=True):
                            pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                            adv_for_summary = full_adv if full_adv is not None and not full_adv.empty else adv
                            prompt = _player_what_changed_prompt(player["full_name"], pergame_for_summary, adv_for_summary, phases)
                            with st.spinner("Analyzing career changes…"):
                                try:
                                    text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                    st.session_state["player_what_changed_output"] = text or "No response."
                                    st.session_state["player_what_changed_player_name"] = player["full_name"]
                                    st.session_state["player_what_changed_phase_summary"] = phase_summary
                                    st.session_state["player_report_mode"] = "what-changed"
                                    st.rerun()
                                except Exception as e:
                                    st.warning(_friendly_ai_error_message(e))
                                    st.caption(f"Details: {type(e).__name__}")
                        if st.session_state.get("player_what_changed_output"):
                            st.caption("A What Changed report is already available.")
                            if st.button("Open What Changed Page", key="open_player_what_changed_page", use_container_width=True):
                                st.session_state["player_report_mode"] = "what-changed"
                                st.rerun()
                    else:
                        st.info("What Changed becomes available once career phases and multi-season stats are available.")
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Role Recommendation", expanded=False):
                if model:
                    role_profile = archetype if isinstance(archetype, dict) else {}
                    archetype_summary = (
                        f"Primary: {role_profile.get('primary', '—')} • "
                        f"Secondary: {role_profile.get('secondary', '—')} • "
                        f"Style: {', '.join(role_profile.get('style_tags', [])[:3]) if role_profile.get('style_tags') else '—'}"
                    )
                    st.caption("Generate a dedicated page recommending this player's best NBA role and supporting context.")
                    if st.button("Generate Role Recommendation", key="generate_player_role_recommendation", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = adv if adv is not None else pd.DataFrame()
                        prompt = _player_role_recommendation_prompt(
                            player["full_name"],
                            pergame_for_summary,
                            adv_for_summary,
                            role_profile,
                        )
                        with st.spinner("Recommending role…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_role_output"] = text or "No response."
                                st.session_state["player_role_player_name"] = player["full_name"]
                                st.session_state["player_role_archetype_summary"] = archetype_summary
                                st.session_state["player_report_mode"] = "role-recommendation"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_role_output"):
                        st.caption("A role recommendation is already available.")
                        if st.button("Open Role Recommendation Page", key="open_player_role_recommendation_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "role-recommendation"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Team Fit Analyzer", expanded=False):
                if model:
                    fit_context = st.text_input(
                        "Team / teammate context",
                        value="General NBA contender fit",
                        placeholder="Example: Next to Luka Doncic on a contender",
                        key="player_team_fit_context_input",
                    )
                    st.caption("Generate a dedicated page breaking down this player's best team fits and lineup context.")
                    if st.button("Generate Team Fit Analysis", key="generate_player_team_fit", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = adv if adv is not None else pd.DataFrame()
                        prompt = _player_team_fit_prompt(player["full_name"], pergame_for_summary, adv_for_summary, fit_context)
                        with st.spinner("Analyzing team fit…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_team_fit_output"] = text or "No response."
                                st.session_state["player_team_fit_player_name"] = player["full_name"]
                                st.session_state["player_team_fit_context"] = fit_context.strip() or "General NBA contender fit"
                                st.session_state["player_report_mode"] = "team-fit"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_team_fit_output"):
                        st.caption("A team fit analysis is already available.")
                        if st.button("Open Team Fit Page", key="open_player_team_fit_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "team-fit"
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


def value_tab(player, model):
    st.subheader("Value & Props")

    raw_pergame = get_player_career(
        player["id"],
        per_mode="PerGame",
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        all_seasons=True,
    )
    raw_totals = get_player_career(
        player["id"],
        per_mode="Totals",
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        all_seasons=True,
    )
    if (raw_pergame is None or raw_pergame.empty) and (raw_totals is None or raw_totals.empty):
        st.info("No player data is available right now.")
        return

    if raw_totals is not None and not raw_totals.empty:
        if raw_totals.attrs.get("provider") == "balldontlie":
            adv = raw_totals.copy()
        else:
            adv = compute_full_advanced_stats(raw_totals)
    else:
        adv = pd.DataFrame()

    adv = add_per_game_columns(adv, raw_pergame)
    adv = _add_team_record_column(adv)

    contract_df = get_balldontlie_player_contracts(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )
    contract_agg_df = get_balldontlie_player_contract_aggregates(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )
    contract_snapshot = _contract_snapshot(contract_df, contract_agg_df)
    headshot_url = get_nba_headshot_url(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )

    latest = adv.iloc[-1] if adv is not None and not adv.empty else (raw_pergame.iloc[-1] if raw_pergame is not None and not raw_pergame.empty else pd.Series(dtype=object))
    hero_col, stats_col = st.columns([1, 3])
    with hero_col:
        _render_headshot_image(headshot_url, 170, player.get("full_name", "Player"))
        st.caption(player.get("full_name", ""))
    with stats_col:
        render_stat_text("Contract context, value analysis, and prop research in one focused workspace.", small=True)
        latest_team_record = latest.get("TEAM_RECORD")
        if pd.notna(latest_team_record) and latest_team_record:
            render_stat_text(f"Current team record: {latest_team_record}", small=True)
        age_value = _current_age_for_player(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        cards = [
            ("Age", str(age_value) if age_value is not None else "—"),
            ("Cap Hit", _fmt_money(contract_snapshot.get("cap_hit"))),
            ("Avg Salary", _fmt_money(contract_snapshot.get("average_salary"))),
            ("Guaranteed", _fmt_money(contract_snapshot.get("total_guaranteed"))),
        ]
        _render_hover_stat_cards(cards)

    if "show_value_ai_rail" not in st.session_state:
        st.session_state["show_value_ai_rail"] = True
    rail_toggle_col_left, rail_toggle_col_right = st.columns([4.5, 1.2])
    with rail_toggle_col_right:
        if st.button(
            "Hide AI sidebar" if st.session_state["show_value_ai_rail"] else "Show AI sidebar",
            key="toggle_value_ai_rail",
            use_container_width=True,
        ):
            st.session_state["show_value_ai_rail"] = not st.session_state["show_value_ai_rail"]
            st.rerun()

    if st.session_state["show_value_ai_rail"]:
        _inject_sticky_ai_rail_css("sticky-value-ai-rail")
        main_col, ai_col = st.columns([3.2, 1.35], gap="large")
    else:
        main_col = st.container()
        ai_col = None

    with main_col:
        if contract_df is not None and not contract_df.empty:
            _render_contract_snapshot_section(contract_snapshot)
        else:
            st.info("Contract data is unavailable for this player right now.")

        latest_season_id = str(adv.iloc[-1].get("SEASON_ID", "")) if adv is not None and not adv.empty and "SEASON_ID" in adv.columns else ""
        _render_prop_context_dashboard(player, latest_season_id)

    if ai_col is not None:
        with ai_col:
            st.markdown('<div class="sticky-value-ai-rail"></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Tools")
            st.caption("Focused tools for contract and value context.")

            with st.expander("Contract Value Analyzer", expanded=False):
                if model and contract_df is not None and not contract_df.empty:
                    birthdate = get_player_birthdate(
                        player["id"],
                        player_name=player.get("full_name"),
                        player_source=player.get("source"),
                    )
                    age_value = _age_from_birthdate(birthdate) if birthdate else None
                    snapshot = _contract_snapshot(contract_df, contract_agg_df)
                    contract_summary = (
                        f"Cap Hit: {_fmt_money(snapshot.get('cap_hit'))} • "
                        f"Avg Salary: {_fmt_money(snapshot.get('average_salary'))} • "
                        f"Years: {int(pd.to_numeric(snapshot.get('contract_years'), errors='coerce')) if pd.notna(pd.to_numeric(snapshot.get('contract_years'), errors='coerce')) else '—'}"
                    )
                    st.caption("Generate a dedicated page judging whether the player's contract looks overpaid, fair, or underpaid.")
                    if st.button("Generate Contract Value Analysis", key="generate_player_contract_value_value_tab", use_container_width=True):
                        pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                        adv_for_summary = adv if adv is not None and not adv.empty else pd.DataFrame()
                        prompt = _player_contract_value_prompt(
                            player["full_name"],
                            pergame_for_summary,
                            adv_for_summary,
                            contract_df,
                            contract_agg_df,
                            age_value,
                        )
                        with st.spinner("Analyzing contract value…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.6)
                                st.session_state["player_contract_value_output"] = text or "No response."
                                st.session_state["player_contract_value_player_name"] = player["full_name"]
                                st.session_state["player_contract_value_summary"] = contract_summary
                                st.session_state["player_report_mode"] = "contract-value"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_contract_value_output"):
                        st.caption("A contract value analysis is already available.")
                        if st.button("Open Contract Value Page", key="open_player_contract_value_page_value_tab", use_container_width=True):
                            st.session_state["player_report_mode"] = "contract-value"
                            st.rerun()
                else:
                    if not model:
                        if AI_SETUP_ERROR:
                            st.info("AI is unavailable in this deployment right now.")
                            st.caption(f"Setup details: {AI_SETUP_ERROR}")
                        else:
                            st.info("Add your OpenAI API key to enable AI analysis.")
                    else:
                        st.info("Contract data is unavailable for this player right now.")
