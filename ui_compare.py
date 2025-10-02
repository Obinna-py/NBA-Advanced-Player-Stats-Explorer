# ui_compare.py
import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import commonplayerinfo
import plotly.express as px
from fetch import get_player_career
from metrics import compute_full_advanced_stats, generate_player_summary, compact_player_context
from ideas import cached_ai_compare_question_ideas, presets
from utils import abbrev, public_cols, make_anchor, smooth_scroll_to

def render_compare_tab(player, model):
    st.subheader("Compare Players")

    for key, default in [("other_matches", []), ("other_player", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    other_name = st.text_input("Enter another player's name to compare:", key="other_name_input")
    search2 = st.button("Search Second Player")

    from nba_api.stats.static import players as pstatic
    if search2:
        om = pstatic.find_players_by_full_name(other_name) if other_name else []
        exact2 = [p for p in om if p['full_name'].lower() == (other_name or "").lower()]
        om = exact2 if exact2 else om
        if not om:
            st.session_state["other_matches"] = []
            st.session_state["other_player"] = None
            st.error("❌ No second player found. Check spelling.")
        elif len(om) == 1:
            st.session_state["other_player"] = om[0]
            st.session_state["other_matches"] = []
        else:
            st.session_state["other_matches"] = om
            st.session_state["other_player"] = None

    if st.session_state["other_matches"]:
        st.write("Multiple players found with that name:")
        options2 = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["other_matches"]}
        pick2 = st.radio("Select a player:", ["⬇️ Pick a player"] + list(options2.keys()), index=0, key="other_player_selection_radio")
        if pick2 != "⬇️ Pick a player":
            st.session_state["other_player"] = options2[pick2]
            st.session_state["other_matches"] = []

    other_player = st.session_state["other_player"]
    if not other_player:
        return

    p1 = player['full_name']; p2 = other_player['full_name']
    st.success(f"Comparing **{p1}** vs **{p2}**")

    cL, cR = st.columns(2)
    with cL:
        info1 = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]
        st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png", width=180)
        st.markdown(f"**{p1}**")
        st.caption(f"{info1.loc[0, 'TEAM_NAME']} • {info1.loc[0, 'POSITION']}")
    with cR:
        info2 = commonplayerinfo.CommonPlayerInfo(player_id=other_player['id']).get_data_frames()[0]
        st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{other_player['id']}.png", width=180)
        st.markdown(f"**{p2}**")
        st.caption(f"{info2.loc[0, 'TEAM_NAME']} • {info2.loc[0, 'POSITION']}")

    comp_speed_mode = st.toggle(
        "Compute advanced for ALL seasons in comparison (slower)",
        value=False,
        help="Off = latest season only (fast). On = all seasons (slower).",
        key="compare_speed_toggle"
    )

    raw1_pg = get_player_career(player['id'], per_mode='PerGame')
    raw2_pg = get_player_career(other_player['id'], per_mode='PerGame')
    raw1_t  = get_player_career(player['id'], per_mode='Totals')
    raw2_t  = get_player_career(other_player['id'], per_mode='Totals')

    if comp_speed_mode:
        adv1_src, adv2_src = raw1_t, raw2_t
    else:
        latest1 = raw1_t['SEASON_ID'].iloc[-1] if not raw1_t.empty else None
        latest2 = raw2_t['SEASON_ID'].iloc[-1] if not raw2_t.empty else None
        adv1_src = raw1_t[raw1_t['SEASON_ID'] == latest1] if latest1 else raw1_t
        adv2_src = raw2_t[raw2_t['SEASON_ID'] == latest2] if latest2 else raw2_t

    with st.spinner("Computing comparison advanced metrics…"):
        adv1 = compute_full_advanced_stats(adv1_src) if adv1_src is not None and not adv1_src.empty else pd.DataFrame()
        adv2 = compute_full_advanced_stats(adv2_src) if adv2_src is not None and not adv2_src.empty else pd.DataFrame()

    ctx1_src = adv1 if (isinstance(adv1, pd.DataFrame) and not adv1.empty) else raw1_pg
    ctx2_src = adv2 if (isinstance(adv2, pd.DataFrame) and not adv2.empty) else raw2_pg
    c1 = compact_player_context(ctx1_src)
    c2 = compact_player_context(ctx2_src)

    with st.expander("💡 Comparison Question Ideas for these players", expanded=False):
        choices, topic_map = presets()
        preset = st.radio("Quick presets", choices, horizontal=True, key="compare_idea_preset")
        topic_default = topic_map.get(preset, "")
        topic = st.text_input("Optional focus (refines suggestions):", value=topic_default, key="compare_idea_focus")

        # 👇 changed: pass `_model=model` instead of `model=model`
        ideas_cmp = cached_ai_compare_question_ideas(p1, p2, c1, c2, topic, use_model=(model is not None), _model=model)
        st.caption("Stat-based, evaluative prompts. Click to drop one into the box below.")
        cols_per_row = 2
        for i in range(0, len(ideas_cmp), cols_per_row):
            row = ideas_cmp[i:i+cols_per_row]
            cols = st.columns(len(row))
            for c, idea in zip(cols, row):
                short = abbrev(idea, 40)
                with c:
                    if st.button(f"💭 {short}", help=idea, use_container_width=True, key=f"cmp_idea_btn_{i}_{short}"):
                        st.session_state["ai_compare_question"] = idea
                        st.session_state["jump_to_compare_ai"] = True
                        st.rerun()

    if not raw1_pg.empty: raw1_pg['SEASON_START'] = raw1_pg['SEASON_ID'].str[:4].astype(int)
    if not raw2_pg.empty: raw2_pg['SEASON_START'] = raw2_pg['SEASON_ID'].str[:4].astype(int)

    source_for_shared = raw1_pg if not raw1_pg.empty else adv1
    other_for_shared  = raw2_pg if not raw2_pg.empty else adv2
    excluded = {'SEASON_ID', 'PLAYER_ID', 'TEAM_ID', 'LEAGUE_ID', 'TEAM_ABBREVIATION'}
    shared_stats = sorted([c for c in source_for_shared.columns if c in other_for_shared.columns and c not in excluded and pd.api.types.is_numeric_dtype(source_for_shared[c])])
    default_idx = shared_stats.index('PTS') if 'PTS' in shared_stats else 0 if shared_stats else 0
    stat_choice = st.selectbox("📊 Choose a stat to compare:", shared_stats or ['PTS'], index=default_idx, key="cmp_stat_choice")

    if not raw1_pg.empty and not raw2_pg.empty:
        common = raw1_pg[['SEASON_START', 'SEASON_ID', stat_choice]].merge(
            raw2_pg[['SEASON_START', 'SEASON_ID', stat_choice]],
            on='SEASON_START', suffixes=(f"_{p1}", f"_{p2}")
        )
    else:
        common = pd.DataFrame()

    if common.empty:
        st.warning("No overlapping seasons to compare.")
    else:
        fig_df = pd.DataFrame({
            "Season": common[f"SEASON_ID_{p1}"],
            p1: common[f"{stat_choice}_{p1}"],
            p2: common[f"{stat_choice}_{p2}"],
        })
        fig = px.line(fig_df, x="Season", y=[p1, p2], markers=True, title=f"{stat_choice} — Overlapping Seasons")
        fig.update_layout(xaxis_title="Season", yaxis_title=stat_choice, legend_title="Player")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Advanced Stats")
    if comp_speed_mode:
        st.caption("All seasons (slower).")
    else:
        st.caption("Latest season only (fast). Toggle above for full career.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**{p1}**")
        if not adv1.empty:
            st.dataframe(adv1[public_cols(adv1)], use_container_width=True)
        else:
            st.info("No advanced data.")
    with t2:
        st.markdown(f"**{p2}**")
        if not adv2.empty:
            st.dataframe(adv2[public_cols(adv2)], use_container_width=True)
        else:
            st.info("No advanced data.")

    # Anchor and scroll for AI box
    make_anchor("compare-ai-anchor")
    with st.expander("🧠 Ask the AI Assistant about these players", expanded=st.session_state.get("jump_to_compare_ai", False)):
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
                    resp2 = model.generate_content(prompt2, generation_config={"max_output_tokens": 2048, "temperature": 0.6})
                    st.markdown("### 🧠 AI Analysis")
                    st.write(resp2.text if hasattr(resp2, "text") else "No response.")
        else:
            st.info("Add your Gemini API key to enable AI analysis.")

    if st.session_state.get("jump_to_compare_ai"):
        smooth_scroll_to("compare-ai-anchor")
        st.session_state["jump_to_compare_ai"] = False
