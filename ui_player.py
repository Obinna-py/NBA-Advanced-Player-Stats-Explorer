# ui_player.py
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from nba_api.stats.endpoints import commonplayerinfo
from logos import college_logos
from fetch import get_player_career
from metrics import compute_full_advanced_stats, generate_player_summary, compact_player_context, add_per_game_columns, metric_public_cols
from ideas import cached_ai_question_ideas, presets
from utils import abbrev, public_cols
from ui_compare import render_html_table, _make_readable_stats_table


def _age_from_birthdate(iso_dt: str) -> int:
    birthdate = datetime.strptime(iso_dt.split('T')[0], "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def info_tab(player):
    info = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]
    team_id = info.loc[0, 'TEAM_ID']
    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png"
    team_logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

    st.subheader("Player Info")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(headshot_url, width=220)
        st.image(team_logo_url, width=120)
    with c2:
        age = _age_from_birthdate(info.loc[0, 'BIRTHDATE'])
        st.markdown(f"### {player['full_name']}")
        st.write(f"**Age:** {age}")
        st.write(f"**Height:** {info.loc[0, 'HEIGHT']}")
        st.write(f"**Weight:** {info.loc[0, 'WEIGHT']} lbs")
        st.write(f"**Position:** {info.loc[0, 'POSITION']}")
        college = info.loc[0, 'SCHOOL']
        st.write(f"**College:** {college}")
        if college and college in college_logos:
            st.image(college_logos[college], width=120)

def stats_tab(player, model):
    st.subheader("Most Recent Season Stats")

    speed_mode = st.toggle(
        "Compute advanced for ALL seasons (slower)",
        value=False,
        help="Off = latest season only (fast). On = all seasons (slower)."
    )

    raw_pergame = get_player_career(player['id'], per_mode='PerGame')
    raw_totals  = get_player_career(player['id'], per_mode='Totals')

    if raw_totals is not None and not raw_totals.empty:
        if speed_mode:
            adv_source = raw_totals
        else:
            latest_season = raw_totals['SEASON_ID'].iloc[-1]
            adv_source = raw_totals[raw_totals['SEASON_ID'] == latest_season].copy()
        with st.spinner("Computing advanced metrics…"):
            adv = compute_full_advanced_stats(adv_source)
    else:
        adv = pd.DataFrame()
    
    adv = add_per_game_columns(adv, raw_pergame)

    latest_src = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
    if latest_src is not None and not latest_src.empty:
        latest = latest_src.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PPG", f"{latest.get('PTS', np.nan):.1f}")
        m2.metric("RPG", f"{latest.get('REB', np.nan):.1f}")
        m3.metric("APG", f"{latest.get('AST', np.nan):.1f}")
        ts_val = adv.iloc[-1]['TS%'] if not adv.empty and 'TS%' in adv.columns else np.nan
        m4.metric("TS%", f"{ts_val:.1f}%")


    if adv is not None and not adv.empty:
        cols = public_cols(adv)
        stats_df, number_cols, percent_cols = _make_readable_stats_table(adv)

        render_html_table(
            stats_df,
            number_cols=number_cols,
            percent_cols=percent_cols,
            max_height_px=520
        )

        if not speed_mode:
            st.info(
                "Showing latest season advanced metrics. "
                "Turn on “ALL seasons” above to compute the full career (slower)."
            )
            
        
    with st.expander("💡 Question Ideas for this player", expanded=False):
        choices, topic_map = presets()
        preset = st.radio("Quick presets", choices, horizontal=True, key="idea_preset")
        topic_default = topic_map.get(preset, "")
        topic = st.text_input("Optional focus (refines suggestions):", value=topic_default, key="idea_focus")
        ctx = compact_player_context(adv if not adv.empty else raw_pergame)
        # 👇 changed: pass `_model=model` instead of `model=model`
        ideas = cached_ai_question_ideas(player['full_name'], ctx, topic, use_model=(model is not None), _model=model)
        st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
        cols_per_row = 2
        for i in range(0, len(ideas), cols_per_row):
            row = ideas[i:i+cols_per_row]
            cols = st.columns(len(row))
            for c, idea in zip(cols, row):
                short = abbrev(idea, 32)
                with c:
                    if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"idea_btn_{i}_{short}"):
                        st.session_state["ai_question"] = idea
                        st.rerun()

    with st.expander("🧠 Ask the AI Assistant about this player"):
        if model:
            q = st.text_input("Ask something about this player:", key="ai_question")
            if q:
                pergame_for_summary = raw_pergame if (raw_pergame is not None and not raw_pergame.empty) else adv
                adv_for_summary     = adv if adv is not None else pd.DataFrame()
                summary = generate_player_summary(player['full_name'], pergame_for_summary, adv_for_summary)
                prompt = (
                    f"You are an expert NBA analyst. Here is the stat summary for {player['full_name']}:\n\n"
                    f"{summary}\n\nQuestion: {q}\n\n"
                    f"Note: Some advanced metrics are estimates from team-context formulas."
                )
                with st.spinner("Analyzing…"):
                    resp = model.generate_content(prompt, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                    st.markdown("### 🧠 AI Analysis")
                    st.write(resp.text if hasattr(resp, "text") else "No response.")
        else:
            st.info("Add your Gemini API key to enable AI analysis.")
