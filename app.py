# nba_app/app.py
import streamlit as st
from config import ensure_page_config, model, AI_SETUP_ERROR
try:
    from streamlit_searchbox import st_searchbox
except Exception:
    st_searchbox = None

from fetch import (
    check_balldontlie_api_health,
    search_players,
    player_from_share_token,
    player_to_share_token,
    get_watchlist_players,
    add_player_to_watchlist,
    remove_player_from_watchlist,
)
from metrics import (
    find_players_by_natural_language,
    find_players_by_custom_filters,
    custom_stat_finder_metric_labels,
    custom_stat_finder_role_labels,
    custom_stat_finder_tag_labels,
)
from ui_player import (
    info_tab,
    stats_tab,
    render_player_story_mode_page,
    render_player_franchise_ranker_page,
    render_player_scouting_report_page,
    render_player_ai_chat_page,
    render_player_team_fit_page,
    render_player_what_changed_page,
    render_player_role_recommendation_page,
    render_player_contract_value_page,
)
from ui_value import value_tab
from ui_fantasy import (
    fantasy_tab,
    render_player_fantasy_value_page,
    render_player_rest_of_season_page,
)
from ui_gm import (
    render_gm_workspace,
    render_player_trade_value_page,
    render_player_co_star_page,
    render_player_roster_fit_page,
    render_player_trade_package_page,
)
from ui_compare import (
    render_compare_tab,
    render_compare_scouting_report_page,
    render_compare_what_changed_page,
    render_compare_ai_chat_page,
    render_compare_debate_page,
)
from ui_draft import render_draft_workspace

ensure_page_config()


def _inject_modern_basketball_os_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 26%),
                linear-gradient(180deg, var(--background-color) 0%, var(--background-color) 100%);
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        [data-testid="stSidebar"] > div:first-child {
            background:
                radial-gradient(circle at top, rgba(245, 158, 11, 0.08), transparent 18%),
                linear-gradient(180deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 340px;
        }
        .os-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background:
                radial-gradient(circle at 0% 0%, rgba(245, 158, 11, 0.14), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(59, 130, 246, 0.16), transparent 28%),
                linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
            border-radius: 24px;
            padding: 1.25rem 1.35rem 1.15rem 1.35rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.22);
        }
        .os-hero-title {
            color: var(--text-color);
            font-size: clamp(1.55rem, 1.6vw, 2rem);
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.45rem;
        }
        .os-hero-subtitle {
            color: var(--text-color);
            opacity: 0.78;
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 72ch;
        }
        .os-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .os-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.42rem 0.78rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: var(--secondary-background-color);
            color: var(--text-color);
            font-size: 0.83rem;
            font-weight: 600;
        }
        .os-chip.is-accent {
            background: rgba(245, 158, 11, 0.14);
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.22);
        }
        .os-chip.is-info {
            background: rgba(59, 130, 246, 0.14);
            color: #93c5fd;
            border-color: rgba(59, 130, 246, 0.22);
        }
        div[data-testid="stVerticalBlock"] > div:has(> .os-sidebar-section) {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            background: linear-gradient(180deg, var(--secondary-background-color), var(--background-color));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            padding: 0.9rem 0.9rem 0.45rem 0.9rem;
            margin-bottom: 0.9rem;
        }
        .os-sidebar-brand {
            position: relative;
            overflow: hidden;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.14), transparent 28%),
                radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.16), transparent 30%),
                linear-gradient(145deg, var(--secondary-background-color), var(--background-color));
            box-shadow: 0 14px 35px rgba(2, 6, 23, 0.18);
        }
        .os-sidebar-brand::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 28%);
            pointer-events: none;
        }
        .os-sidebar-kicker {
            color: rgba(226, 232, 240, 0.72);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .os-sidebar-title {
            color: var(--text-color);
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }
        .os-sidebar-subtitle {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .os-search-shell {
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            padding: 0.95rem 1rem 0.85rem 1rem;
            margin-bottom: 0.2rem;
            border: 1px solid rgba(245, 158, 11, 0.16);
            background:
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 30%),
                linear-gradient(160deg, var(--secondary-background-color), var(--background-color));
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.22);
        }
        .os-search-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent 30%);
            pointer-events: none;
        }
        .os-search-kicker {
            color: rgba(251, 191, 36, 0.86);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .os-search-title {
            color: var(--text-color);
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.28rem;
        }
        .os-search-subtitle {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.82rem;
            line-height: 1.5;
        }
        .os-sidebar-section {
            color: rgba(226, 232, 240, 0.84);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
            opacity: 0.9;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.95rem 0 1rem 0;
            border-color: rgba(148, 163, 184, 0.12);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] .stCaptionContainer {
            color: rgba(226, 232, 240, 0.62);
        }
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {
            border-radius: 14px !important;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 14px;
            font-weight: 700;
            letter-spacing: -0.01em;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: none;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(245, 158, 11, 0.35);
        }
        div[role="radiogroup"] {
            gap: 0.35rem;
            padding: 0.3rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.64);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        div[role="radiogroup"] label {
            border-radius: 12px !important;
            padding: 0.28rem 0.72rem !important;
            background: transparent;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.18), rgba(59, 130, 246, 0.18));
            border: 1px solid rgba(248, 250, 252, 0.1);
        }
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        @media (max-width: 1100px) {
            [data-testid="stSidebar"] {
                min-width: 290px;
                max-width: 310px;
            }
            .os-hero {
                border-radius: 20px;
            }
        }
        @media (max-width: 980px) {
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1.5rem;
            }
            .os-hero {
                padding: 1rem 1rem 0.95rem 1rem;
                border-radius: 20px;
            }
            .os-hero-title {
                font-size: 1.35rem;
            }
            .os-search-shell,
            .os-sidebar-brand {
                padding: 0.9rem 0.9rem 0.85rem 0.9rem;
                border-radius: 18px;
            }
        }
        @media (max-width: 760px) {
            [data-testid="stSidebar"] {
                min-width: 100%;
                max-width: 100%;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 0 !important;
            }
            .os-chip-row {
                gap: 0.45rem;
            }
            .os-chip {
                width: 100%;
                justify-content: center;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_modern_os_hero(section: str, selected_player: dict | None, gm_player: dict | None) -> None:
    if section == "🏗 GM / Team Building":
        title = "Modern Basketball OS: GM Lab"
        subtitle = (
            "A front-office workspace for multi-team trade design, asset valuation, cap-context thinking, "
            "and roster-building decisions that feel closer to a real NBA war room."
        )
        chips = [
            ('Mode: Team Building', "accent"),
            (f"Focal Player: {(gm_player or {}).get('full_name', 'None')}", "info"),
            ("Asset-Based Trade Builder", None),
        ]
    elif section == "🎓 NBA Draft":
        title = "Modern Basketball OS: Prospect Lab"
        subtitle = (
            "A draft workspace for prospect search, AI translation, consensus context, ESPN mock tracking, "
            "and player-by-player NBA projection work."
        )
        chips = [
            ('Mode: NBA Draft', "accent"),
            ("Search Any NCAA Player", "info"),
            ("AI + Consensus + ESPN Mock", None),
        ]
    else:
        title = "Modern Basketball OS"
        subtitle = (
            "A premium basketball intelligence workspace for player evaluation, compare analysis, fantasy, "
            "value and props, and AI-assisted scouting built into one unified product."
        )
        chips = [
            ('Mode: Player Explorer', "accent"),
            (f"Current Player: {(selected_player or {}).get('full_name', 'None selected')}", "info"),
            ("AI + Stats + Decision Tools", None),
        ]

    chip_html = []
    for text, variant in chips:
        cls = "os-chip"
        if variant == "accent":
            cls += " is-accent"
        elif variant == "info":
            cls += " is-info"
        chip_html.append(f'<span class="{cls}">{text}</span>')

    st.markdown(
        f"""
        <div class="os-hero">
            <div class="os-hero-title">{title}</div>
            <div class="os-hero-subtitle">{subtitle}</div>
            <div class="os-chip-row">
                {''.join(chip_html)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if AI_SETUP_ERROR:
    st.warning("AI is not fully configured in this deployment right now.")
    st.caption(f"Setup details: {AI_SETUP_ERROR}")

_inject_modern_basketball_os_css()


_VIEW_TO_TOKEN = {
    "📋 Player Info": "info",
    "📊 Stats": "stats",
    "🧩 Fantasy": "fantasy",
    "💸 Value & Props": "value",
    "🤝 Compare Players": "compare",
}
_TOKEN_TO_VIEW = {token: label for label, token in _VIEW_TO_TOKEN.items()}
_APP_SECTIONS = ["🏀 Player Explorer", "🏗 GM / Team Building", "🎓 NBA Draft"]
_NL_ARCHETYPE_OPTIONS = [
    "Custom search",
    "3 and D",
    "stretch 5",
    "heliocentric guard",
    "rim-running big",
    "shot-creating wing",
    "point center",
    "floor-spacing big",
    "catch and shoot specialist",
    "movement shooter",
    "slashing guard",
    "drive and kick creator",
    "primary playmaker",
    "scoring playmaker",
    "point of attack defender",
    "two-way guard",
    "point forward",
    "all-around wing",
    "connector wing",
    "stretch rim protector",
    "unicorn big",
]
_CUSTOM_STAT_METRIC_OPTIONS = ["None"] + custom_stat_finder_metric_labels()
_CUSTOM_ROLE_OPTIONS = custom_stat_finder_role_labels()
_CUSTOM_TAG_OPTIONS = custom_stat_finder_tag_labels()

_SEARCHBOX_STYLE_OVERRIDES = {
    "searchbox": {
        "control": {
            "backgroundColor": "rgba(10,16,30,0.64)",
            "borderColor": "rgba(148,163,184,0.08)",
            "borderRadius": 18,
            "minHeight": 50,
            "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.02)",
        },
        "input": {
            "color": "#f8fafc",
            "fontSize": 15,
            "fontWeight": 600,
            "fontFamily": "Inter, SF Pro Display, Segoe UI, sans-serif",
        },
        "placeholder": {
            "color": "rgba(226,232,240,0.38)",
            "fontSize": 14,
            "fontFamily": "Inter, SF Pro Display, Segoe UI, sans-serif",
            "fontStyle": "normal",
        },
        "singleValue": {
            "color": "#f8fafc",
            "fontSize": 15,
            "fontWeight": 700,
            "fontFamily": "Inter, SF Pro Display, Segoe UI, sans-serif",
        },
        "menu": {
            "backgroundColor": "#0b1220",
            "borderRadius": 16,
            "overflow": "hidden",
            "border": "1px solid rgba(148,163,184,0.12)",
            "boxShadow": "0 24px 48px rgba(2,6,23,0.42)",
        },
        "menuList": {
            "backgroundColor": "#0b1220",
            "paddingTop": 6,
            "paddingBottom": 6,
        },
        "option": {
            "fontSize": 14,
            "fontFamily": "Inter, SF Pro Display, Segoe UI, sans-serif",
            "fontWeight": 600,
            "letterSpacing": "-0.01em",
            "paddingTop": 11,
            "paddingBottom": 11,
            "paddingLeft": 14,
            "paddingRight": 14,
        },
        "noOptionsMessage": {
            "color": "rgba(255,255,255,0.55)",
            "fontSize": 13,
            "fontFamily": "Inter, SF Pro Display, Segoe UI, sans-serif",
            "fontStyle": "normal",
        },
    },
    "dropdown": {
        "fill": "rgba(226,232,240,0.72)",
        "width": 20,
        "height": 20,
        "rotate": True,
    },
    "clear": {
        "width": 18,
        "height": 18,
        "icon": "cross",
        "clearable": "always",
    },
}


def _open_player(selected_player: dict) -> None:
    st.session_state["player"] = selected_player
    st.session_state["matches"] = []
    st.session_state["compare_players"] = []
    st.session_state["nl_results"] = None
    st.session_state["nl_meta"] = None
    st.session_state["custom_results"] = None
    st.session_state["custom_meta"] = None
    st.session_state["active_view"] = "📊 Stats"
    st.session_state["search_feedback"] = None
    st.session_state["player_report_mode"] = None


def _open_gm_player(selected_player: dict) -> None:
    st.session_state["gm_player"] = selected_player
    st.session_state["player_report_mode"] = None
    st.session_state["app_section"] = "🏗 GM / Team Building"


def _player_search_suggestions(searchterm: str) -> list[tuple[str, dict]]:
    if not searchterm or len(searchterm.strip()) < 2:
        return []
    try:
        results = search_players(searchterm.strip())[:8]
    except Exception:
        return []

    options = []
    for player in results:
        meta = []
        if player.get("position"):
            meta.append(player["position"])
        if player.get("team_name"):
            meta.append(player["team_name"])
        meta_text = f" ({' • '.join(meta)})" if meta else ""
        options.append((f"{player['full_name']}{meta_text}", player))
    return options


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

    report_mode = str(params.get("report", "")).strip().lower()
    if report_mode == "compare-scouting":
        st.session_state["compare_report_mode"] = "scouting"
    elif report_mode == "compare-chat":
        st.session_state["compare_report_mode"] = "chat"
    elif report_mode == "compare-debate":
        st.session_state["compare_report_mode"] = "debate"
    elif report_mode == "player-scouting":
        st.session_state["player_report_mode"] = "scouting"
    elif report_mode == "player-chat":
        st.session_state["player_report_mode"] = "chat"
    elif report_mode == "player-story-mode":
        st.session_state["player_report_mode"] = "story-mode"
    elif report_mode == "player-franchise-ranker":
        st.session_state["player_report_mode"] = "franchise-ranker"
    elif report_mode == "player-team-fit":
        st.session_state["player_report_mode"] = "team-fit"
    elif report_mode == "player-what-changed":
        st.session_state["player_report_mode"] = "what-changed"
    elif report_mode == "player-role-recommendation":
        st.session_state["player_report_mode"] = "role-recommendation"
    elif report_mode == "player-contract-value":
        st.session_state["player_report_mode"] = "contract-value"
    elif report_mode == "player-fantasy-value":
        st.session_state["player_report_mode"] = "fantasy-value"
    elif report_mode == "player-fantasy-ros":
        st.session_state["player_report_mode"] = "fantasy-ros"
    elif report_mode == "player-trade-value":
        st.session_state["player_report_mode"] = "trade-value"
    elif report_mode == "player-co-star":
        st.session_state["player_report_mode"] = "co-star"
    elif report_mode == "player-roster-fit":
        st.session_state["player_report_mode"] = "roster-fit"
    elif report_mode == "player-trade-package":
        st.session_state["player_report_mode"] = "trade-package"

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

    if st.session_state.get("compare_report_mode"):
        compare_mode = st.session_state.get("compare_report_mode")
        payload["report"] = {
            "chat": "compare-chat",
            "scouting": "compare-scouting",
            "debate": "compare-debate",
        }.get(compare_mode, "compare-scouting")
    elif st.session_state.get("player_report_mode"):
        player_mode = st.session_state.get("player_report_mode")
        payload["report"] = {
            "chat": "player-chat",
            "story-mode": "player-story-mode",
            "franchise-ranker": "player-franchise-ranker",
            "scouting": "player-scouting",
            "team-fit": "player-team-fit",
            "what-changed": "player-what-changed",
            "role-recommendation": "player-role-recommendation",
            "contract-value": "player-contract-value",
            "fantasy-value": "player-fantasy-value",
            "fantasy-ros": "player-fantasy-ros",
            "trade-value": "player-trade-value",
            "co-star": "player-co-star",
            "roster-fit": "player-roster-fit",
            "trade-package": "player-trade-package",
        }.get(player_mode, "player-scouting")

    current = {k: str(v) for k, v in st.query_params.items()}
    if current != payload:
        st.query_params.clear()
        for key, value in payload.items():
            st.query_params[key] = value

# session state
for key, default in [
    ("matches", []),
    ("player", None),
    ("gm_player", None),
    ("search_feedback", None),
    ("compare_players", []),
    ("nl_results", None),
    ("nl_meta", None),
    ("custom_results", None),
    ("custom_meta", None),
    ("compare_report_mode", None),
    ("player_report_mode", None),
    ("requested_active_view", None),
    ("requested_app_section", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "📊 Stats"
if "app_section" not in st.session_state:
    st.session_state["app_section"] = "🏀 Player Explorer"
if "custom_stat_rule_count" not in st.session_state:
    st.session_state["custom_stat_rule_count"] = 3

_load_share_state_from_url()

if st.session_state.get("requested_active_view"):
    st.session_state["active_view"] = st.session_state.pop("requested_active_view")
if st.session_state.get("requested_app_section"):
    st.session_state["app_section"] = st.session_state.pop("requested_app_section")
if st.session_state.get("active_view") not in _VIEW_TO_TOKEN:
    st.session_state["active_view"] = "📊 Stats"

app_section = st.radio(
    "App section",
    _APP_SECTIONS,
    horizontal=True,
    key="app_section",
    label_visibility="collapsed",
)

with st.sidebar:
    st.markdown(
        """
        <div class="os-sidebar-brand">
            <div class="os-sidebar-kicker">Modern Basketball OS</div>
            <div class="os-sidebar-title">NBA Intelligence Workspace</div>
            <div class="os-sidebar-subtitle">Search, screen, compare, and simulate roster decisions from one polished control rail.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if app_section == "🏀 Player Explorer":
        st.markdown('<div class="os-sidebar-section">Player Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="os-search-shell">
                <div class="os-search-kicker">Featured Search</div>
                <div class="os-search-title">Find Any Player</div>
                <div class="os-search-subtitle">Jump straight into player analysis with live league-wide autocomplete.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st_searchbox is not None:
            selected_player = st_searchbox(
                _player_search_suggestions,
                label="",
                placeholder="Type a player...",
                key="player_searchbox",
                clear_on_submit=False,
                edit_after_submit="option",
                style_overrides=_SEARCHBOX_STYLE_OVERRIDES,
            )
            if isinstance(selected_player, dict) and selected_player.get("id") is not None:
                current = st.session_state.get("player")
                if not current or player_to_share_token(current) != player_to_share_token(selected_player):
                    _open_player(selected_player)
                    st.rerun()
            search_clicked = False
            name = ""
        else:
            name = st.text_input("", key="player_search_name", placeholder="Type a player...")
            live_suggestions = []
            if name and len(name.strip()) >= 2:
                try:
                    live_suggestions = search_players(name.strip())[:8]
                except Exception:
                    live_suggestions = []

            if live_suggestions:
                st.caption("Live matches")
                for idx, player in enumerate(live_suggestions[:6]):
                    label = player["full_name"]
                    meta = []
                    if player.get("position"):
                        meta.append(player["position"])
                    if player.get("team_name"):
                        meta.append(player["team_name"])
                    meta_text = f" ({' • '.join(meta)})" if meta else ""
                    if st.button(f"{label}{meta_text}", key=f"live_match_{idx}", use_container_width=True):
                        _open_player(player)
                        st.rerun()

            search_clicked = st.button("Search")
        st.divider()
        st.markdown('<div class="os-sidebar-section">Search Tools</div>', unsafe_allow_html=True)
        st.subheader("Natural Language Search")
        nl_preset = st.selectbox(
            "Choose an archetype / prompt",
            _NL_ARCHETYPE_OPTIONS,
            index=0,
            key="nl_preset",
        )
        if nl_preset != "Custom search" and st.session_state.get("nl_query") != nl_preset:
            st.session_state["nl_query"] = nl_preset
        nl_query = st.text_area(
            "Describe the kind of player you want",
            key="nl_query",
            height=90,
            placeholder="Example: elite rim-protecting bigs with scoring punch",
        )
        nl_search_clicked = st.button("Find Players", use_container_width=True)
        st.divider()
        st.markdown('<div class="os-sidebar-section">Custom Screener</div>', unsafe_allow_html=True)
        st.subheader("Custom Stat Finder")
        rule_controls_left, rule_controls_right = st.columns(2)
        with rule_controls_left:
            if st.button("Add rule", use_container_width=True, key="custom_stat_add_rule"):
                st.session_state["custom_stat_rule_count"] = min(int(st.session_state.get("custom_stat_rule_count", 3)) + 1, 10)
                st.rerun()
        with rule_controls_right:
            if st.button("Remove rule", use_container_width=True, key="custom_stat_remove_rule"):
                st.session_state["custom_stat_rule_count"] = max(int(st.session_state.get("custom_stat_rule_count", 3)) - 1, 1)
                st.rerun()
        with st.form("custom_stat_finder_form"):
            cs_position = st.selectbox(
                "Position family",
                ["Any", "Guard", "Wing", "Big"],
                index=0,
                key="custom_stat_position",
            )
            cs_role = st.selectbox(
                "Primary role filter",
                _CUSTOM_ROLE_OPTIONS,
                index=0,
                key="custom_stat_role_filter",
            )
            cs_tag = st.selectbox(
                "Archetype / tag filter",
                _CUSTOM_TAG_OPTIONS,
                index=0,
                key="custom_stat_tag_filter",
            )
            cs_percentile_metric = st.selectbox(
                "Percentile gate metric",
                ["None"] + custom_stat_finder_metric_labels(),
                index=0,
                key="custom_stat_percentile_metric",
            )
            cs_min_percentile = st.slider(
                "Minimum percentile",
                min_value=50,
                max_value=99,
                value=75,
                step=1,
                key="custom_stat_min_percentile",
            )
            cs_min_gp = st.slider("Min games", min_value=5, max_value=82, value=20, step=1, key="custom_stat_min_gp")
            cs_min_mpg = st.slider("Min minutes per game", min_value=5.0, max_value=36.0, value=15.0, step=1.0, key="custom_stat_min_mpg")
            st.caption(f"Active rules: {int(st.session_state.get('custom_stat_rule_count', 3))}")
            for idx in range(int(st.session_state.get("custom_stat_rule_count", 3))):
                st.caption(f"Rule {idx + 1}")
                metric_col, op_col, value_col = st.columns([1.5, 0.9, 1.1])
                with metric_col:
                    st.selectbox(
                        "Metric",
                        _CUSTOM_STAT_METRIC_OPTIONS,
                        index=0,
                        key=f"custom_stat_metric_{idx}",
                        label_visibility="collapsed",
                    )
                with op_col:
                    st.selectbox(
                        "Op",
                        [">=", "<="],
                        index=0,
                        key=f"custom_stat_op_{idx}",
                        label_visibility="collapsed",
                    )
                with value_col:
                    st.number_input(
                        "Value",
                        value=0.0,
                        step=0.5,
                        key=f"custom_stat_value_{idx}",
                        label_visibility="collapsed",
                    )
            cs_sort_by = st.selectbox("Sort by", custom_stat_finder_metric_labels(), index=0, key="custom_stat_sort_by")
            cs_sort_dir = st.selectbox("Sort direction", ["Highest first", "Lowest first"], index=0, key="custom_stat_sort_dir")
            cs_limit = st.slider("Result limit", min_value=5, max_value=40, value=20, step=1, key="custom_stat_limit")
            custom_search_clicked = st.form_submit_button("Run Custom Finder", use_container_width=True)
        st.divider()
        st.markdown('<div class="os-sidebar-section">Saved Context</div>', unsafe_allow_html=True)
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
                    st.session_state["active_view"] = "📊 Stats"
                    st.rerun()
            with remove_col:
                if selected_watch != "Select a saved player" and st.button("Remove", use_container_width=True, key="watchlist_remove"):
                    remove_player_from_watchlist(watch_options[selected_watch])
                    st.rerun()
        else:
            st.caption("Save players here for quick access later.")
        st.divider()
        st.markdown('<div class="os-sidebar-section">System</div>', unsafe_allow_html=True)
        st.subheader("API Status")
        run_balldontlie_check = st.button("Check balldontlie API", use_container_width=True)

        if run_balldontlie_check:
            st.session_state["balldontlie_api_health"] = check_balldontlie_api_health()

        balldontlie_health = st.session_state.get("balldontlie_api_health")
        if balldontlie_health:
            st.caption("balldontlie")
            if balldontlie_health.get("ok"):
                st.success(balldontlie_health.get("message", "balldontlie API is reachable."))
            else:
                st.warning("balldontlie looks slow or unavailable right now.")
                st.caption(f"{balldontlie_health.get('error_type', 'Error')}: {balldontlie_health.get('message', 'Unknown error')}")
    elif app_section == "🏗 GM / Team Building":
        st.markdown('<div class="os-sidebar-section">GM Workspace</div>', unsafe_allow_html=True)
        st.header("🏗 GM Workspace")
        st.caption("Set a focal player if you want player-specific trade value, co-star, and roster-fit analysis.")
        if st_searchbox is not None:
            selected_gm_player = st_searchbox(
                _player_search_suggestions,
                label="",
                placeholder="Type a player...",
                key="gm_player_searchbox",
                clear_on_submit=False,
                edit_after_submit="option",
                style_overrides=_SEARCHBOX_STYLE_OVERRIDES,
            )
            if isinstance(selected_gm_player, dict) and selected_gm_player.get("id") is not None:
                current = st.session_state.get("gm_player")
                if not current or player_to_share_token(current) != player_to_share_token(selected_gm_player):
                    _open_gm_player(selected_gm_player)
                    st.rerun()
        else:
            gm_name = st.text_input("", key="gm_player_search_name", placeholder="Type a player...")
            gm_live_suggestions = []
            if gm_name and len(gm_name.strip()) >= 2:
                try:
                    gm_live_suggestions = search_players(gm_name.strip())[:8]
                except Exception:
                    gm_live_suggestions = []
            if gm_live_suggestions:
                st.caption("Live matches")
                for idx, player in enumerate(gm_live_suggestions[:6]):
                    label = player["full_name"]
                    meta = []
                    if player.get("position"):
                        meta.append(player["position"])
                    if player.get("team_name"):
                        meta.append(player["team_name"])
                    meta_text = f" ({' • '.join(meta)})" if meta else ""
                    if st.button(f"{label}{meta_text}", key=f"gm_live_match_{idx}", use_container_width=True):
                        _open_gm_player(player)
                        st.rerun()
        if st.session_state.get("gm_player"):
            st.caption(f"Current focal player: {st.session_state['gm_player'].get('full_name', 'Selected Player')}")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Use in Player Explorer", use_container_width=True, key="gm_open_in_player_explorer"):
                    _open_player(st.session_state["gm_player"])
                    st.session_state["app_section"] = "🏀 Player Explorer"
                    st.rerun()
            with col_b:
                if st.button("Clear Focal Player", use_container_width=True, key="gm_clear_focal_player"):
                    st.session_state["gm_player"] = None
                    st.session_state["player_report_mode"] = None
                    st.rerun()
    else:
        st.markdown('<div class="os-sidebar-section">Draft Expansion</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="os-search-shell">
                <div class="os-search-kicker">Prospect Lab</div>
                <div class="os-search-title">NBA Draft Foundation</div>
                <div class="os-search-subtitle">Search any current NCAA player, track ESPN’s latest mock, and use AI as the default translation layer for how a prospect projects.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Add your `BALLDONTLIE_NCAAB_API_KEY` to secrets or env vars if live NCAAB search is not already active.")

search_clicked = locals().get("search_clicked", False)
nl_search_clicked = locals().get("nl_search_clicked", False)
custom_search_clicked = locals().get("custom_search_clicked", False)
name = locals().get("name", "")
nl_query = locals().get("nl_query", "")

if app_section == "🏀 Player Explorer" and search_clicked:
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
        st.session_state["custom_results"] = None
        st.session_state["custom_meta"] = None
        st.session_state["search_feedback"] = {
            "kind": "error",
            "message": (
                f'No player was returned for "{name}". Try a fuller spelling, '
                'for example "Victor Wembanyama" instead of "Wemby".'
            ),
        }
    elif len(found) == 1:
        _open_player(found[0])
    else:
        st.session_state["matches"] = found
        st.session_state["player"] = None
        st.session_state["search_feedback"] = None

if app_section == "🏀 Player Explorer" and nl_search_clicked:
    results_df, meta = find_players_by_natural_language(
        nl_query,
        season=None,
        limit=10,
        use_model=bool(model),
        _model=model,
    )
    st.session_state["nl_results"] = results_df
    st.session_state["nl_meta"] = meta
    st.session_state["custom_results"] = None
    st.session_state["custom_meta"] = None
    if results_df is not None and not results_df.empty:
        st.session_state["player"] = None
        st.session_state["matches"] = []
        st.session_state["search_feedback"] = None

if app_section == "🏀 Player Explorer" and custom_search_clicked:
    custom_filters = []
    for idx in range(int(st.session_state.get("custom_stat_rule_count", 3))):
        metric = st.session_state.get(f"custom_stat_metric_{idx}")
        op = st.session_state.get(f"custom_stat_op_{idx}", ">=")
        value = st.session_state.get(f"custom_stat_value_{idx}", 0.0)
        if metric and metric != "None":
            custom_filters.append({"metric": metric, "op": op, "value": value})

    position_family = {
        "Any": None,
        "Guard": "guard",
        "Wing": "wing",
        "Big": "big",
    }.get(st.session_state.get("custom_stat_position", "Any"))

    results_df, meta = find_players_by_custom_filters(
        season=None,
        filters=custom_filters,
        position_family=position_family,
        percentile_metric=None if st.session_state.get("custom_stat_percentile_metric", "None") == "None" else st.session_state.get("custom_stat_percentile_metric"),
        min_percentile=int(st.session_state.get("custom_stat_min_percentile", 75)) if st.session_state.get("custom_stat_percentile_metric", "None") != "None" else None,
        primary_role=None if st.session_state.get("custom_stat_role_filter", "Any") == "Any" else st.session_state.get("custom_stat_role_filter"),
        archetype_tag=None if st.session_state.get("custom_stat_tag_filter", "Any") == "Any" else st.session_state.get("custom_stat_tag_filter"),
        min_gp=int(st.session_state.get("custom_stat_min_gp", 20)),
        min_mpg=float(st.session_state.get("custom_stat_min_mpg", 15.0)),
        sort_by=st.session_state.get("custom_stat_sort_by", "PPG"),
        sort_desc=(st.session_state.get("custom_stat_sort_dir", "Highest first") == "Highest first"),
        limit=int(st.session_state.get("custom_stat_limit", 20)),
    )
    st.session_state["custom_results"] = results_df
    st.session_state["custom_meta"] = meta
    if results_df is not None and not results_df.empty:
        st.session_state["player"] = None
        st.session_state["matches"] = []
        st.session_state["search_feedback"] = None
        st.session_state["nl_results"] = None
        st.session_state["nl_meta"] = None

feedback = st.session_state.get("search_feedback")
if app_section == "🏀 Player Explorer" and feedback:
    if feedback.get("kind") == "error":
        st.error(feedback.get("message", "No player was returned for that search."))
    else:
        st.info(feedback.get("message", ""))

nl_results = st.session_state.get("nl_results")
nl_meta = st.session_state.get("nl_meta") or {}
if app_section == "🏀 Player Explorer" and nl_results is not None and not st.session_state.get("player"):
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
                    st.session_state["custom_results"] = None
                    st.session_state["custom_meta"] = None
                    st.rerun()

custom_results = st.session_state.get("custom_results")
custom_meta = st.session_state.get("custom_meta") or {}
if app_section == "🏀 Player Explorer" and custom_results is not None and not st.session_state.get("player"):
    st.subheader("Custom Stat Finder")
    if custom_meta.get("summary"):
        st.caption(
            f"{custom_meta['summary']} Searching the {custom_meta.get('season', 'latest')} regular season player pool."
        )
    if custom_meta.get("filter_summary"):
        st.caption(f"Filters: {custom_meta['filter_summary']}")
    if custom_meta.get("message"):
        st.info(custom_meta["message"])

    if isinstance(custom_results, type(None)) or custom_results.empty:
        st.info("Use the sidebar screener to find players by custom stat rules.")
    else:
        display_cols = [c for c in custom_results.columns if c != "Player Token"]
        st.dataframe(custom_results[display_cols], use_container_width=True, hide_index=True)

        open_options = {
            f"{row['Player']} ({row['Position']})": row["Player Token"]
            for _, row in custom_results.iterrows()
            if row.get("Player Token")
        }
        if open_options:
            open_choice = st.selectbox(
                "Open a player from custom stat finder",
                ["Select a player"] + list(open_options.keys()),
                index=0,
                key="custom_open_choice",
            )
            if open_choice != "Select a player" and st.button("Open Player Page", key="custom_open_player"):
                selected = player_from_share_token(open_options[open_choice])
                if selected:
                    st.session_state["player"] = selected
                    st.session_state["matches"] = []
                    st.session_state["compare_players"] = []
                    st.session_state["active_view"] = "📊 Stats"
                    st.session_state["search_feedback"] = None
                    st.session_state["custom_results"] = None
                    st.session_state["custom_meta"] = None
                    st.rerun()

if app_section == "🏀 Player Explorer" and st.session_state["matches"]:
    st.write("Multiple players found with that name:")
    opts = {f"{p['full_name']} (ID: {p['id']})": p for p in st.session_state["matches"]}
    choice = st.radio("Select a player:", ["⬇️ Pick a player"] + list(opts.keys()), index=0, key="player_selection_radio")
    if choice != "⬇️ Pick a player":
        st.session_state["player"] = opts[choice]
        st.session_state["matches"] = []
        st.session_state["compare_players"] = []
        st.session_state["nl_results"] = None
        st.session_state["nl_meta"] = None
        st.session_state["active_view"] = "📊 Stats"

if app_section == "🏗 GM / Team Building":
    _render_modern_os_hero(app_section, st.session_state.get("player"), st.session_state.get("gm_player"))
    if st.session_state.get("player_report_mode") == "trade-value":
        render_player_trade_value_page()
        _sync_share_state_to_url()
        st.stop()
    if st.session_state.get("player_report_mode") == "co-star":
        render_player_co_star_page()
        _sync_share_state_to_url()
        st.stop()
    if st.session_state.get("player_report_mode") == "roster-fit":
        render_player_roster_fit_page()
        _sync_share_state_to_url()
        st.stop()
    if st.session_state.get("player_report_mode") == "trade-package":
        render_player_trade_package_page()
        _sync_share_state_to_url()
        st.stop()
    render_gm_workspace(model)
elif app_section == "🎓 NBA Draft":
    _render_modern_os_hero(app_section, st.session_state.get("player"), st.session_state.get("gm_player"))
    render_draft_workspace(model)
elif st.session_state["player"]:
    _render_modern_os_hero(app_section, st.session_state.get("player"), st.session_state.get("gm_player"))
    if (
        st.session_state.get("active_view") == "🤝 Compare Players"
        and st.session_state.get("compare_report_mode") == "scouting"
    ):
        render_compare_scouting_report_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "🤝 Compare Players"
        and st.session_state.get("compare_report_mode") == "what-changed"
    ):
        render_compare_what_changed_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "🤝 Compare Players"
        and st.session_state.get("compare_report_mode") == "chat"
    ):
        render_compare_ai_chat_page(model)
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "🤝 Compare Players"
        and st.session_state.get("compare_report_mode") == "debate"
    ):
        render_compare_debate_page(model)
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "story-mode"
    ):
        render_player_story_mode_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "franchise-ranker"
    ):
        render_player_franchise_ranker_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "scouting"
    ):
        render_player_scouting_report_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "chat"
    ):
        render_player_ai_chat_page(model)
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "team-fit"
    ):
        render_player_team_fit_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "what-changed"
    ):
        render_player_what_changed_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "📊 Stats"
        and st.session_state.get("player_report_mode") == "role-recommendation"
    ):
        render_player_role_recommendation_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "🧩 Fantasy"
        and st.session_state.get("player_report_mode") == "fantasy-value"
    ):
        render_player_fantasy_value_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "🧩 Fantasy"
        and st.session_state.get("player_report_mode") == "fantasy-ros"
    ):
        render_player_rest_of_season_page()
        _sync_share_state_to_url()
        st.stop()
    if (
        st.session_state.get("active_view") == "💸 Value & Props"
        and st.session_state.get("player_report_mode") == "contract-value"
    ):
        render_player_contract_value_page()
        _sync_share_state_to_url()
        st.stop()
    view = st.radio(
        "View",
        ["📋 Player Info", "📊 Stats", "🧩 Fantasy", "💸 Value & Props", "🤝 Compare Players"],
        horizontal=True,
        key="active_view",
        label_visibility="collapsed",
    )
    if view == "📋 Player Info":
        info_tab(st.session_state["player"])
    elif view == "📊 Stats":
        stats_tab(st.session_state["player"], model)
    elif view == "🧩 Fantasy":
        fantasy_tab(st.session_state["player"], model)
    elif view == "💸 Value & Props":
        value_tab(st.session_state["player"], model)
    else:
        render_compare_tab(st.session_state["player"], model)
else:
    if app_section == "🏀 Player Explorer":
        _render_modern_os_hero(app_section, st.session_state.get("player"), st.session_state.get("gm_player"))
        st.info("Use the sidebar to search for a player.")

_sync_share_state_to_url()
