# nba_app/app.py
import streamlit as st
from config import ensure_page_config, model
from fetch import check_nba_api_health, check_balldontlie_api_health, search_players
from ui_player import info_tab, stats_tab
from ui_compare import render_compare_tab

ensure_page_config()

with st.sidebar:
    st.header("🔍 Search Player")
    name = st.text_input("Enter an NBA player's name")
    search_clicked = st.button("Search")
    st.divider()
    st.subheader("API Status")
    run_nba_check = st.button("Check NBA API", use_container_width=True)
    run_balldontlie_check = st.button("Check balldontlie API", use_container_width=True)

    if run_nba_check:
        st.session_state["nba_api_health"] = check_nba_api_health()
    if run_balldontlie_check:
        st.session_state["balldontlie_api_health"] = check_balldontlie_api_health()

    nba_health = st.session_state.get("nba_api_health")
    if nba_health:
        st.caption("NBA stats")
        if nba_health.get("ok"):
            st.success(nba_health.get("message", "NBA API is reachable."))
        else:
            st.warning("NBA API looks slow right now.")
            st.caption("The app now prefers balldontlie first, then NBA stats, then local cache.")

    balldontlie_health = st.session_state.get("balldontlie_api_health")
    if balldontlie_health:
        st.caption("balldontlie")
        if balldontlie_health.get("ok"):
            st.success(balldontlie_health.get("message", "balldontlie API is reachable."))
        else:
            st.warning("balldontlie looks slow or unavailable right now.")
            st.caption(f"{balldontlie_health.get('error_type', 'Error')}: {balldontlie_health.get('message', 'Unknown error')}")

# session state
for key, default in [("matches", []), ("player", None), ("search_feedback", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if search_clicked:
    try:
        found = search_players(name) if name else []
    except Exception as e:
        found = []
        st.session_state["matches"] = []
        st.session_state["player"] = None
        st.sidebar.error("Could not search for that player right now.")
        st.sidebar.caption(f"{type(e).__name__}: {e}")

    if name and not found:
        st.session_state["matches"] = []
        st.session_state["player"] = None
        st.session_state["search_feedback"] = {
            "kind": "error",
            "message": (
                f'No player was returned for "{name}". Try a fuller spelling, '
                'for example "Victor Wembanyama" instead of "Wemby".'
            ),
        }
    elif len(found) == 1:
        st.session_state["player"] = found[0]
        st.session_state["matches"] = []
        st.session_state["active_view"] = "📋 Player Info"
        st.session_state["search_feedback"] = None
    else:
        st.session_state["matches"] = found
        st.session_state["player"] = None
        st.session_state["search_feedback"] = None

feedback = st.session_state.get("search_feedback")
if feedback:
    if feedback.get("kind") == "error":
        st.error(feedback.get("message", "No player was returned for that search."))
    else:
        st.info(feedback.get("message", ""))

if st.session_state["matches"]:
    st.write("Multiple players found with that name:")
    opts = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["matches"]}
    choice = st.radio("Select a player:", ["⬇️ Pick a player"] + list(opts.keys()), index=0, key="player_selection_radio")
    if choice != "⬇️ Pick a player":
        st.session_state["player"] = opts[choice]
        st.session_state["matches"] = []
        st.session_state["active_view"] = "📋 Player Info"

if st.session_state["player"]:
    health = st.session_state.get("nba_api_health")
    if health and not health.get("ok"):
        st.warning(
            "NBA stats looks slow right now. The app is using balldontlie as the primary live provider."
        )

    view = st.radio(
        "View",
        ["📋 Player Info", "📊 Stats", "🤝 Compare Players"],
        horizontal=True,
        key="active_view",
        label_visibility="collapsed",
    )
    if view == "📋 Player Info":
        info_tab(st.session_state["player"])
    elif view == "📊 Stats":
        stats_tab(st.session_state["player"], model)
    else:
        render_compare_tab(st.session_state["player"], model)
else:
    st.info("Use the sidebar to search for a player.")
