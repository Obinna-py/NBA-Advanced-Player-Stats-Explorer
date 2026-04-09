import numpy as np
import pandas as pd
import streamlit as st

from config import ai_generate_text, AI_SETUP_ERROR
from fetch import (
    get_player_career,
    get_player_birthdate,
    get_nba_headshot_url,
    get_balldontlie_player_game_logs,
    get_balldontlie_league_season_averages,
)
from metrics import compute_full_advanced_stats, add_per_game_columns, build_ai_stat_packet, generate_player_summary
from ui_player import (
    _add_team_record_column,
    _render_headshot_image,
    _render_hover_stat_cards,
    _inject_sticky_ai_rail_css,
    _friendly_ai_error_message,
    _age_from_birthdate,
)
from ui_compare import render_stat_text, render_html_table


def _fantasy_points_value(row: pd.Series) -> float | None:
    if row is None or row.empty:
        return None
    ppg = pd.to_numeric(row.get("PPG", row.get("PTS")), errors="coerce")
    rpg = pd.to_numeric(row.get("RPG", row.get("REB")), errors="coerce")
    apg = pd.to_numeric(row.get("APG", row.get("AST")), errors="coerce")
    spg = pd.to_numeric(row.get("SPG", row.get("STL")), errors="coerce")
    bpg = pd.to_numeric(row.get("BPG", row.get("BLK")), errors="coerce")
    tpg = pd.to_numeric(row.get("TPG", row.get("TOV")), errors="coerce")
    pieces = [ppg, rpg, apg, spg, bpg, tpg]
    if all(pd.isna(v) for v in pieces):
        return None
    return float(
        np.nan_to_num(ppg) * 1.0
        + np.nan_to_num(rpg) * 1.2
        + np.nan_to_num(apg) * 1.5
        + np.nan_to_num(spg) * 3.0
        + np.nan_to_num(bpg) * 3.0
        - np.nan_to_num(tpg) * 1.0
    )


def _fantasy_tier(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    if value >= 48:
        return "First-Round Fantasy Anchor"
    if value >= 42:
        return "Early-Round Fantasy Star"
    if value >= 36:
        return "Strong Starter"
    if value >= 30:
        return "Mid-Round Core Piece"
    if value >= 24:
        return "Back-End Starter"
    return "Streamer / Matchup Play"


def _norm_percentile(series: pd.Series, value: float | None, *, higher_is_better: bool = True) -> float | None:
    if value is None or pd.isna(value):
        return None
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if nums.empty:
        return None
    if higher_is_better:
        return float((nums <= value).mean() * 100.0)
    return float((nums >= value).mean() * 100.0)


def _league_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series(dtype="float64")


def _fantasy_category_profile(latest: pd.Series, latest_season_id: str) -> pd.DataFrame:
    if latest is None or latest.empty or not latest_season_id:
        return pd.DataFrame()
    try:
        season_start = int(str(latest_season_id)[:4])
    except Exception:
        return pd.DataFrame()
    league_df = get_balldontlie_league_season_averages(season_start)
    if league_df is None or league_df.empty:
        return pd.DataFrame()

    rows = []
    category_specs = [
        ("Points", pd.to_numeric(latest.get("PPG", latest.get("PTS")), errors="coerce"), _league_col(league_df, "pts"), True),
        ("Rebounds", pd.to_numeric(latest.get("RPG", latest.get("REB")), errors="coerce"), _league_col(league_df, "reb"), True),
        ("Assists", pd.to_numeric(latest.get("APG", latest.get("AST")), errors="coerce"), _league_col(league_df, "ast"), True),
        ("Steals", pd.to_numeric(latest.get("SPG", latest.get("STL")), errors="coerce"), _league_col(league_df, "stl"), True),
        ("Blocks", pd.to_numeric(latest.get("BPG", latest.get("BLK")), errors="coerce"), _league_col(league_df, "blk"), True),
        ("3PM", pd.to_numeric(latest.get("FG3M", np.nan), errors="coerce"), _league_col(league_df, "fg3m"), True),
        ("FG%", pd.to_numeric(latest.get("FG%"), errors="coerce"), _league_col(league_df, "fg_pct") * 100.0, True),
        ("FT%", pd.to_numeric(latest.get("FT%"), errors="coerce"), _league_col(league_df, "ft_pct") * 100.0, True),
        ("Turnovers", pd.to_numeric(latest.get("TPG", latest.get("TOV")), errors="coerce"), _league_col(league_df, "turnover"), False),
    ]
    for label, value, league_series, higher_is_better in category_specs:
        pct = _norm_percentile(league_series, value, higher_is_better=higher_is_better)
        if pct is None:
            continue
        if pct >= 75:
            impact = "Strength"
        elif pct <= 35:
            impact = "Weakness"
        else:
            impact = "Neutral"
        rows.append({"Category": label, "Value": value, "Percentile": pct, "Impact": impact})
    out = pd.DataFrame(rows).sort_values(["Impact", "Percentile"], ascending=[True, False])
    return out


def _fantasy_category_tags(profile_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if profile_df is None or profile_df.empty:
        return [], []
    strengths = profile_df[profile_df["Impact"] == "Strength"].sort_values("Percentile", ascending=False)["Category"].tolist()
    weaknesses = profile_df[profile_df["Impact"] == "Weakness"].sort_values("Percentile", ascending=True)["Category"].tolist()
    return strengths[:4], weaknesses[:3]


def _buy_sell_hold_decision(latest: pd.Series, logs: pd.DataFrame, profile_df: pd.DataFrame) -> dict:
    if latest is None or latest.empty:
        return {"label": "Hold", "reason": "Not enough data to make a stronger call."}
    valid_pts = pd.to_numeric(logs.get("PTS"), errors="coerce").dropna() if logs is not None and not logs.empty else pd.Series(dtype="float64")
    last5 = float(valid_pts.head(5).mean()) if len(valid_pts) else np.nan
    last10 = float(valid_pts.head(10).mean()) if len(valid_pts) else np.nan
    season_ppg = pd.to_numeric(latest.get("PPG", latest.get("PTS")), errors="coerce")
    consistency = np.nan
    if len(valid_pts) > 1 and pd.notna(season_ppg) and season_ppg > 0:
        consistency = 100.0 - min(max(float(valid_pts.std(ddof=0) / season_ppg) * 100.0, 0.0), 100.0)
    strengths, weaknesses = _fantasy_category_tags(profile_df)
    if pd.notna(last5) and pd.notna(season_ppg) and last5 <= season_ppg - 2.0 and strengths:
        return {"label": "Buy Low", "reason": f"Recent scoring is down versus the season baseline, but the category base is still strong in {', '.join(strengths[:2])}."}
    if pd.notna(last5) and pd.notna(season_ppg) and last5 >= season_ppg + 3.0 and (pd.isna(consistency) or consistency < 58):
        return {"label": "Sell High", "reason": "The recent scoring spike is well above the season baseline, and the profile still has enough volatility that this may be a peak sell window."}
    if weaknesses and pd.notna(consistency) and consistency < 50:
        return {"label": "Matchup-Dependent Hold", "reason": f"The production swings more than you want, and the main fantasy pain points remain {', '.join(weaknesses[:2])}."}
    return {"label": "Hold", "reason": "The current value looks close to the season baseline, so the safest move is to hold rather than chase a short-term swing."}


def _rest_of_season_summary(latest: pd.Series, logs: pd.DataFrame, profile_df: pd.DataFrame, age_value: int | None) -> dict:
    valid_pts = pd.to_numeric(logs.get("PTS"), errors="coerce").dropna() if logs is not None and not logs.empty else pd.Series(dtype="float64")
    last10 = float(valid_pts.head(10).mean()) if len(valid_pts) else np.nan
    season_ppg = pd.to_numeric(latest.get("PPG", latest.get("PTS")), errors="coerce")
    mpg = pd.to_numeric(latest.get("MPG", latest.get("MIN")), errors="coerce")
    team_record = latest.get("TEAM_RECORD")
    strengths, weaknesses = _fantasy_category_tags(profile_df)
    if pd.notna(last10) and pd.notna(season_ppg) and last10 >= season_ppg + 2 and pd.notna(mpg) and mpg >= 30:
        label = "Rising"
        outlook = "The role and recent production both point upward right now."
    elif pd.notna(mpg) and mpg < 24:
        label = "Fragile"
        outlook = "Minutes are still too soft to fully trust week to week."
    else:
        label = "Stable"
        outlook = "The role looks stable enough that the current season baseline is still the cleanest guide."
    bullets = []
    if strengths:
        bullets.append(f"Best category help: {', '.join(strengths[:3])}.")
    if weaknesses:
        bullets.append(f"Main category risk: {', '.join(weaknesses[:2])}.")
    if pd.notna(mpg):
        bullets.append(f"Current minutes load: {mpg:.1f} MPG.")
    if age_value is not None:
        bullets.append(f"Age context: {age_value}.")
    if team_record and str(team_record).strip():
        bullets.append(f"Team record: {team_record}.")
    return {"label": label, "summary": outlook, "bullets": bullets}


def _fantasy_snapshot_prompt(player_name: str, latest: pd.Series, profile_df: pd.DataFrame, points_value: float | None, bs_hold: dict, ros: dict) -> str:
    lines = [f"Player: {player_name}"]
    if latest is not None and not latest.empty:
        lines.extend([
            f"Season: {latest.get('SEASON_ID', '—')}",
            f"Team record: {latest.get('TEAM_RECORD', '—')}",
            f"PPG: {latest.get('PPG', latest.get('PTS', '—'))}",
            f"RPG: {latest.get('RPG', latest.get('REB', '—'))}",
            f"APG: {latest.get('APG', latest.get('AST', '—'))}",
            f"SPG: {latest.get('SPG', latest.get('STL', '—'))}",
            f"BPG: {latest.get('BPG', latest.get('BLK', '—'))}",
            f"TPG: {latest.get('TPG', latest.get('TOV', '—'))}",
            f"FG%: {latest.get('FG%', '—')}",
            f"FT%: {latest.get('FT%', '—')}",
            f"3PM: {latest.get('FG3M', '—')}",
            f"Fantasy points value: {points_value if points_value is not None else '—'}",
            f"Quick buy/sell/hold read: {bs_hold.get('label')}: {bs_hold.get('reason')}",
            f"Rest of season label: {ros.get('label')}: {ros.get('summary')}",
        ])
    if profile_df is not None and not profile_df.empty:
        lines.append("Category profile:")
        for _, row in profile_df.iterrows():
            lines.append(f"- {row['Category']}: {row['Percentile']:.1f} percentile ({row['Impact']})")
    return "\n".join(lines)


def _player_fantasy_value_prompt(player_name: str, per_game_df: pd.DataFrame, adv_df: pd.DataFrame, latest: pd.Series, profile_df: pd.DataFrame, points_value: float | None, bs_hold: dict, ros: dict) -> str:
    stat_packet = build_ai_stat_packet(player_name, per_game_df, adv_df)
    summary = generate_player_summary(player_name, per_game_df, adv_df)
    snapshot = _fantasy_snapshot_prompt(player_name, latest, profile_df, points_value, bs_hold, ros)
    return (
        "You are an expert fantasy basketball analyst. Evaluate this player for redraft fantasy leagues using the provided stats only.\n"
        "Write in markdown with these exact sections: Fantasy Snapshot, Points League View, Category League View, Buy Low Or Sell High?, Main Risks, Bottom Line.\n"
        "Give a clean recommendation: Buy Low, Sell High, Hold, or Matchup-Dependent Hold. Stay practical and readable.\n\n"
        f"Snapshot:\n{snapshot}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_rest_of_season_prompt(player_name: str, per_game_df: pd.DataFrame, adv_df: pd.DataFrame, latest: pd.Series, profile_df: pd.DataFrame, points_value: float | None, ros: dict) -> str:
    stat_packet = build_ai_stat_packet(player_name, per_game_df, adv_df)
    summary = generate_player_summary(player_name, per_game_df, adv_df)
    snapshot = _fantasy_snapshot_prompt(player_name, latest, profile_df, points_value, {"label": "—", "reason": "—"}, ros)
    return (
        "You are an expert fantasy basketball analyst. Project the player's rest-of-season fantasy outlook in redraft leagues.\n"
        "Write in markdown with these exact sections: Current Fantasy Role, Rest-Of-Season Upside, Rest-Of-Season Risks, Best League Fits, Bottom Line.\n"
        "Focus on role stability, category help, volatility, and whether the player looks like a rising, stable, or risky fantasy asset.\n\n"
        f"Snapshot:\n{snapshot}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def render_player_fantasy_value_page() -> None:
    st.markdown("## 🧩 Buy Low / Sell High / Hold")
    player_name = st.session_state.get("player_fantasy_value_player_name") or "Selected Player"
    summary = st.session_state.get("player_fantasy_value_summary") or "Fantasy context unavailable"
    st.caption(player_name)
    st.caption(summary)
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Fantasy", key="back_to_fantasy_from_value", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "🧩 Fantasy"
            st.rerun()
    report = st.session_state.get("player_fantasy_value_output")
    if report:
        st.markdown(report)
    else:
        st.info("No fantasy value analysis is available right now. Generate one from the Fantasy page first.")


def render_player_rest_of_season_page() -> None:
    st.markdown("## 📈 Rest-of-Season Analyzer")
    player_name = st.session_state.get("player_ros_player_name") or "Selected Player"
    summary = st.session_state.get("player_ros_summary") or "Fantasy context unavailable"
    st.caption(player_name)
    st.caption(summary)
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to Fantasy", key="back_to_fantasy_from_ros", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_active_view"] = "🧩 Fantasy"
            st.rerun()
    report = st.session_state.get("player_ros_output")
    if report:
        st.markdown(report)
    else:
        st.info("No rest-of-season analysis is available right now. Generate one from the Fantasy page first.")


def fantasy_tab(player: dict, model) -> None:
    st.subheader("Fantasy")
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
        adv = raw_totals.copy() if raw_totals.attrs.get("provider") == "balldontlie" else compute_full_advanced_stats(raw_totals)
    else:
        adv = pd.DataFrame()
    adv = add_per_game_columns(adv, raw_pergame)
    adv = _add_team_record_column(adv)
    if adv is None or adv.empty:
        st.info("Fantasy analysis needs at least one season of player stats.")
        return

    player_key = f"{player.get('source', 'nba_api')}:{player.get('id')}"
    if st.session_state.get("player_fantasy_output_signature") != player_key:
        for key in [
            "player_fantasy_value_output",
            "player_fantasy_value_player_name",
            "player_fantasy_value_summary",
            "player_ros_output",
            "player_ros_player_name",
            "player_ros_summary",
        ]:
            st.session_state.pop(key, None)
        st.session_state["player_fantasy_output_signature"] = player_key

    latest = adv.iloc[-1]
    latest_season_id = str(latest.get("SEASON_ID", ""))
    latest_logs = get_balldontlie_player_game_logs(
        player["id"],
        [latest_season_id],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
        season_type="Regular Season",
    )
    profile_df = _fantasy_category_profile(latest, latest_season_id)
    strengths, weaknesses = _fantasy_category_tags(profile_df)
    points_value = _fantasy_points_value(latest)
    points_tier = _fantasy_tier(points_value)
    birthdate = get_player_birthdate(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )
    age_value = _age_from_birthdate(birthdate) if birthdate else None
    bs_hold = _buy_sell_hold_decision(latest, latest_logs if latest_logs is not None else pd.DataFrame(), profile_df)
    ros = _rest_of_season_summary(latest, latest_logs if latest_logs is not None else pd.DataFrame(), profile_df, age_value)
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
        render_stat_text("Fantasy snapshot, category impact, and actionable roster guidance in one workspace.", small=True)
        latest_team_record = latest.get("TEAM_RECORD")
        if pd.notna(latest_team_record) and str(latest_team_record).strip():
            render_stat_text(f"Current team record: {latest.get('TEAM_RECORD')}", small=True)
        _render_hover_stat_cards(
            [
                ("Age", str(age_value) if age_value is not None else "—"),
                ("Fantasy Pts", f"{points_value:.1f}" if points_value is not None else "—"),
                ("Tier", points_tier),
                ("ROS Outlook", ros.get("label", "—")),
            ],
            columns_per_row=2,
        )

    if "show_fantasy_ai_rail" not in st.session_state:
        st.session_state["show_fantasy_ai_rail"] = True
    _, rail_toggle_col = st.columns([4.5, 1.2])
    with rail_toggle_col:
        if st.button(
            "Hide AI sidebar" if st.session_state["show_fantasy_ai_rail"] else "Show AI sidebar",
            key="toggle_fantasy_ai_rail",
            use_container_width=True,
        ):
            st.session_state["show_fantasy_ai_rail"] = not st.session_state["show_fantasy_ai_rail"]
            st.rerun()

    if st.session_state["show_fantasy_ai_rail"]:
        _inject_sticky_ai_rail_css("sticky-fantasy-ai-rail")
        main_col, ai_col = st.columns([3.2, 1.35], gap="large")
    else:
        main_col = st.container()
        ai_col = None

    with main_col:
        st.markdown("### 🧾 Fantasy Snapshot")
        summary_cols = st.columns(4)
        snapshot_items = [
            ("Points League Value", f"{points_value:.1f}" if points_value is not None else "—"),
            ("Buy / Sell / Hold", bs_hold.get("label", "—")),
            ("Best Category", strengths[0] if strengths else "—"),
            ("Biggest Risk", weaknesses[0] if weaknesses else "—"),
        ]
        for col, (label, value) in zip(summary_cols, snapshot_items):
            with col:
                st.metric(label, value)
        render_stat_text(bs_hold.get("reason", "No fantasy recommendation available."), small=True)

        st.markdown("### 📊 Category Profile")
        if profile_df is not None and not profile_df.empty:
            render_stat_text(
                f"Main category strengths: {', '.join(strengths) if strengths else '—'}. "
                f"Main category weaknesses: {', '.join(weaknesses) if weaknesses else '—'}.",
                small=True,
            )
            display_df = profile_df.copy()
            render_html_table(display_df, number_cols=["Value"], percent_cols=["Percentile"], max_height_px=360)
        else:
            st.info("Category profile is unavailable right now.")

        st.markdown("### 🔄 Buy Low / Sell High / Hold")
        buy_cols = st.columns([1.2, 3])
        with buy_cols[0]:
            st.metric("Verdict", bs_hold.get("label", "—"))
        with buy_cols[1]:
            render_stat_text(bs_hold.get("reason", "No clear fantasy verdict."), small=False)

        st.markdown("### 📈 Rest-of-Season Outlook")
        ros_cols = st.columns([1.2, 3])
        with ros_cols[0]:
            st.metric("ROS Label", ros.get("label", "—"))
        with ros_cols[1]:
            render_stat_text(ros.get("summary", "No rest-of-season summary available."), small=False)
            for bullet in ros.get("bullets", []):
                render_stat_text(f"- {bullet}", small=True)

    if ai_col is not None:
        with ai_col:
            st.markdown('<div class="sticky-fantasy-ai-rail"></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 Fantasy AI")
            st.caption("Dedicated fantasy tools and deeper recommendation pages.")

            with st.expander("Buy Low / Sell High / Hold Analyzer", expanded=False):
                if model:
                    st.caption("Generate a dedicated fantasy value page for this player.")
                    if st.button("Generate Fantasy Value Analysis", key="generate_fantasy_value_analysis", use_container_width=True):
                        prompt = _player_fantasy_value_prompt(
                            player["full_name"],
                            raw_pergame if raw_pergame is not None and not raw_pergame.empty else adv,
                            adv,
                            latest,
                            profile_df,
                            points_value,
                            bs_hold,
                            ros,
                        )
                        with st.spinner("Building fantasy value analysis…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_fantasy_value_output"] = text or "No response."
                                st.session_state["player_fantasy_value_player_name"] = player["full_name"]
                                points_text = f"{points_value:.1f}" if points_value is not None and not pd.isna(points_value) else "—"
                                st.session_state["player_fantasy_value_summary"] = f"Points value: {points_text} • Verdict: {bs_hold.get('label', '—')}"
                                st.session_state["player_report_mode"] = "fantasy-value"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_fantasy_value_output"):
                        st.caption("A fantasy value analysis is already available.")
                        if st.button("Open Fantasy Value Page", key="open_fantasy_value_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "fantasy-value"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")

            with st.expander("Rest-of-Season Analyzer", expanded=False):
                if model:
                    st.caption("Generate a dedicated rest-of-season fantasy outlook page.")
                    if st.button("Generate ROS Analysis", key="generate_fantasy_ros_analysis", use_container_width=True):
                        prompt = _player_rest_of_season_prompt(
                            player["full_name"],
                            raw_pergame if raw_pergame is not None and not raw_pergame.empty else adv,
                            adv,
                            latest,
                            profile_df,
                            points_value,
                            ros,
                        )
                        with st.spinner("Projecting rest of season…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_ros_output"] = text or "No response."
                                st.session_state["player_ros_player_name"] = player["full_name"]
                                st.session_state["player_ros_summary"] = f"ROS label: {ros.get('label', '—')}"
                                st.session_state["player_report_mode"] = "fantasy-ros"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_ros_output"):
                        st.caption("A rest-of-season analysis is already available.")
                        if st.button("Open ROS Page", key="open_fantasy_ros_page", use_container_width=True):
                            st.session_state["player_report_mode"] = "fantasy-ros"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")
