# nba_app/app.py
import streamlit as st
from config import ensure_page_config, model
from fetch import (
    check_nba_api_health,
    check_balldontlie_api_health,
    search_players,
    player_from_share_token,
    player_to_share_token,
    get_watchlist_players,
    add_player_to_watchlist,
    remove_player_from_watchlist,
)
from metrics import find_players_by_natural_language
from ui_player import info_tab, stats_tab
from ui_compare import render_compare_tab

ensure_page_config()


_VIEW_TO_TOKEN = {
    "📋 Player Info": "info",
    "📊 Stats": "stats",
    "🤝 Compare Players": "compare",
}
_TOKEN_TO_VIEW = {token: label for label, token in _VIEW_TO_TOKEN.items()}


def _load_share_state_from_url() -> None:
    if st.session_state.get("_share_state_loaded"):
        return

    params = st.query_params
    player = player_from_share_token(params.get("player"))
    compare_tokens = str(params.get("compare", "")).strip()
    compare_players = []
    if compare_tokens:
        for token in compare_tokens.split(","):
            resolved = player_from_share_token(token.strip())
            if resolved:
                compare_players.append(resolved)

    if player:
        st.session_state["player"] = player
    if compare_players:
        primary_key = player_to_share_token(player)
        st.session_state["compare_players"] = [
            p for p in compare_players
            if player_to_share_token(p) != primary_key
        ]

    view_token = str(params.get("view", "")).strip().lower()
    if view_token in _TOKEN_TO_VIEW:
        st.session_state["active_view"] = _TOKEN_TO_VIEW[view_token]

    st.session_state["_share_state_loaded"] = True


def _sync_share_state_to_url() -> None:
    payload = {}

    player = st.session_state.get("player")
    player_token = player_to_share_token(player)
    if player_token:
        payload["player"] = player_token

    active_view = st.session_state.get("active_view")
    if active_view in _VIEW_TO_TOKEN:
        payload["view"] = _VIEW_TO_TOKEN[active_view]

    compare_players = st.session_state.get("compare_players", []) or []
    compare_tokens = [
        token
        for token in (player_to_share_token(p) for p in compare_players)
        if token and token != player_token
    ]
    if compare_tokens:
        payload["compare"] = ",".join(compare_tokens)

    current = {k: str(v) for k, v in st.query_params.items()}
    if current != payload:
        st.query_params.clear()
        for key, value in payload.items():
            st.query_params[key] = value

with st.sidebar:
    st.header("🔍 Search Player")
    name = st.text_input("Enter an NBA player's name")
    search_clicked = st.button("Search")
    st.divider()
    st.subheader("Natural Language Search")
    nl_query = st.text_area(
        "Describe the kind of player you want",
        key="nl_query",
        height=90,
        placeholder="Example: elite rim-protecting bigs with scoring punch",
    )
    nl_search_clicked = st.button("Find Players", use_container_width=True)
    st.divider()
    st.subheader("Watchlist")
    watchlist_players = get_watchlist_players()
    current_player = st.session_state.get("player")
    current_token = player_to_share_token(current_player)
    watchlist_tokens = {player_to_share_token(p) for p in watchlist_players}

    if current_player:
        if current_token in watchlist_tokens:
            if st.button("Remove Current Player", use_container_width=True):
                remove_player_from_watchlist(current_player)
                st.rerun()
        else:
            if st.button("Save Current Player", use_container_width=True):
                add_player_to_watchlist(current_player)
                st.rerun()

    if watchlist_players:
        watch_options = {player["full_name"]: player for player in watchlist_players}
        selected_watch = st.selectbox(
            "Saved players",
            ["Select a saved player"] + list(watch_options.keys()),
            index=0,
            key="watchlist_select",
        )
        open_col, remove_col = st.columns(2)
        with open_col:
            if selected_watch != "Select a saved player" and st.button("Open", use_container_width=True, key="watchlist_open"):
                st.session_state["player"] = watch_options[selected_watch]
                st.session_state["matches"] = []
                st.session_state["compare_players"] = []
                st.session_state["nl_results"] = None
                st.session_state["nl_meta"] = None
                st.session_state["active_view"] = "📋 Player Info"
                st.rerun()
        with remove_col:
            if selected_watch != "Select a saved player" and st.button("Remove", use_container_width=True, key="watchlist_remove"):
                remove_player_from_watchlist(watch_options[selected_watch])
                st.rerun()
    else:
        st.caption("Save players here for quick access later.")
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
for key, default in [
    ("matches", []),
    ("player", None),
    ("search_feedback", None),
    ("compare_players", []),
    ("nl_results", None),
    ("nl_meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "📋 Player Info"

_load_share_state_from_url()

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
        st.session_state["compare_players"] = []
        st.session_state["nl_results"] = None
        st.session_state["nl_meta"] = None
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
        st.session_state["compare_players"] = []
        st.session_state["nl_results"] = None
        st.session_state["nl_meta"] = None
        st.session_state["active_view"] = "📋 Player Info"
        st.session_state["search_feedback"] = None
    else:
        st.session_state["matches"] = found
        st.session_state["player"] = None
        st.session_state["search_feedback"] = None

if nl_search_clicked:
    results_df, meta = find_players_by_natural_language(
        nl_query,
        season=None,
        limit=10,
        use_model=bool(model),
        _model=model,
    )
    st.session_state["nl_results"] = results_df
    st.session_state["nl_meta"] = meta
    if results_df is not None and not results_df.empty:
        st.session_state["player"] = None
        st.session_state["matches"] = []
        st.session_state["search_feedback"] = None

feedback = st.session_state.get("search_feedback")
if feedback:
    if feedback.get("kind") == "error":
        st.error(feedback.get("message", "No player was returned for that search."))
    else:
        st.info(feedback.get("message", ""))

nl_results = st.session_state.get("nl_results")
nl_meta = st.session_state.get("nl_meta") or {}
if nl_results is not None and not st.session_state.get("player"):
    st.subheader("Natural Language Player Search")
    if nl_meta.get("summary"):
        st.caption(
            f"{nl_meta['summary']} Searching the {nl_meta.get('season', 'latest')} regular season player pool."
        )
    if nl_meta.get("message"):
        st.info(nl_meta["message"])
    if nl_meta.get("unsupported_terms"):
        unsupported = ", ".join(sorted(set(nl_meta["unsupported_terms"])))
        st.caption(f"Not fully handled yet in this version: {unsupported}.")
    if nl_meta.get("relaxed"):
        st.warning("No exact matches cleared every requested threshold, so these are the closest latest-season fits.")

    if isinstance(nl_results, type(None)) or nl_results.empty:
        st.info("Use the sidebar prompt to search the league by stat profile.")
    else:
        display_cols = [c for c in nl_results.columns if c != "Player Token"]
        st.dataframe(nl_results[display_cols], use_container_width=True, hide_index=True)

        open_options = {
            f"{row['Player']} ({row['Position']})": row["Player Token"]
            for _, row in nl_results.iterrows()
            if row.get("Player Token")
        }
        if open_options:
            open_choice = st.selectbox(
                "Open a player from these results",
                ["Select a player"] + list(open_options.keys()),
                index=0,
                key="nl_open_choice",
            )
            if open_choice != "Select a player" and st.button("Open Player Page", key="nl_open_player"):
                selected = player_from_share_token(open_options[open_choice])
                if selected:
                    st.session_state["player"] = selected
                    st.session_state["matches"] = []
                    st.session_state["compare_players"] = []
                    st.session_state["active_view"] = "📊 Stats"
                    st.session_state["search_feedback"] = None
                    st.session_state["nl_results"] = None
                    st.session_state["nl_meta"] = None
                    st.rerun()

if st.session_state["matches"]:
    st.write("Multiple players found with that name:")
    opts = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["matches"]}
    choice = st.radio("Select a player:", ["⬇️ Pick a player"] + list(opts.keys()), index=0, key="player_selection_radio")
    if choice != "⬇️ Pick a player":
        st.session_state["player"] = opts[choice]
        st.session_state["matches"] = []
        st.session_state["compare_players"] = []
        st.session_state["nl_results"] = None
        st.session_state["nl_meta"] = None
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

_sync_share_state_to_url()
