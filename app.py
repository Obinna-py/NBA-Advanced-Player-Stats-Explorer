# nba_app/app.py
import streamlit as st
from nba_api.stats.static import players
from config import ensure_page_config, model
from ui_player import info_tab, stats_tab
from ui_compare import render_compare_tab

ensure_page_config()

with st.sidebar:
    st.header("🔍 Search Player")
    name = st.text_input("Enter an NBA player's name")
    search_clicked = st.button("Search")

# session state
for key, default in [("matches", []), ("player", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if search_clicked:
    found = players.find_players_by_full_name(name) if name else []
    exact = [p for p in found if p['full_name'].lower() == (name or "").lower()]
    found = exact if exact else found
    if not found:
        st.session_state["matches"] = []
        st.session_state["player"] = None
        st.sidebar.error("❌ This player is not available in official NBA stats. Rookie data usually appears after the season begins.")
    elif len(found) == 1:
        st.session_state["player"] = found[0]
        st.session_state["matches"] = []
    else:
        st.session_state["matches"] = found
        st.session_state["player"] = None

if st.session_state["matches"]:
    st.write("Multiple players found with that name:")
    opts = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["matches"]}
    choice = st.radio("Select a player:", ["⬇️ Pick a player"] + list(opts.keys()), index=0, key="player_selection_radio")
    if choice != "⬇️ Pick a player":
        st.session_state["player"] = opts[choice]
        st.session_state["matches"] = []

if st.session_state["player"]:
    tab_info, tab_stats, tab_compare = st.tabs(["📋 Player Info", "📊 Stats", "🤝 Compare Players"])
    with tab_info:
        info_tab(st.session_state["player"])
    with tab_stats:
        stats_tab(st.session_state["player"], model)
    with tab_compare:
        render_compare_tab(st.session_state["player"], model)
else:
    st.info("Use the sidebar to search for a player.")
