import pandas as pd
import streamlit as st

from config import ai_generate_text, AI_SETUP_ERROR
from fetch import (
    get_player_career,
    get_player_birthdate,
    get_nba_headshot_url,
    get_balldontlie_player_contracts,
    get_balldontlie_player_contract_aggregates,
)
from metrics import compute_full_advanced_stats, add_per_game_columns
from ui_player import (
    _contract_snapshot,
    _add_team_record_column,
    _render_headshot_image,
    _render_hover_stat_cards,
    _render_contract_snapshot_section,
    _render_prop_context_dashboard,
    _inject_sticky_ai_rail_css,
    _friendly_ai_error_message,
    _player_contract_value_prompt,
    _age_from_birthdate,
    _fmt_money,
)
from ui_compare import render_stat_text


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
        birthdate = get_player_birthdate(
            player["id"],
            player_name=player.get("full_name"),
            player_source=player.get("source"),
        )
        age_value = _age_from_birthdate(birthdate) if birthdate else None
        cards = [
            ("Age", str(age_value) if age_value is not None else "—"),
            ("Cap Hit", _fmt_money(contract_snapshot.get("cap_hit"))),
            ("Avg Salary", _fmt_money(contract_snapshot.get("average_salary"))),
            ("Guaranteed", _fmt_money(contract_snapshot.get("total_guaranteed"))),
        ]
        _render_hover_stat_cards(cards, columns_per_row=2)

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
                    snapshot = _contract_snapshot(contract_df, contract_agg_df)
                    contract_years = pd.to_numeric(snapshot.get("contract_years"), errors="coerce")
                    contract_summary = (
                        f"Cap Hit: {_fmt_money(snapshot.get('cap_hit'))} • "
                        f"Avg Salary: {_fmt_money(snapshot.get('average_salary'))} • "
                        f"Years: {int(contract_years) if pd.notna(contract_years) else '—'}"
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
