import html
import json

import streamlit as st

from config import ai_generate_text, model as default_model
from draft_fetch import (
    DEFAULT_DRAFT_SEASON,
    get_placeholder_draft_headshot,
    load_espn_mock_2026,
    load_espn_prospect_media,
    load_live_prospect_profile,
    load_local_prospect_metadata,
    load_prospect_consensus_anchors,
    lookup_ncaab_player,
    ncaab_api_ready,
    search_ncaab_players,
)
from draft_metrics import (
    build_simple_college_profile,
    build_advanced_college_profile,
    build_outcome_bands,
    build_strengths_weaknesses_matrix,
    build_translation_confidence,
    build_generated_prospect_profile,
    build_searchable_draft_universe,
    draft_search_label,
    filter_prospects,
    find_consensus_anchor,
    find_local_prospect,
    merge_consensus_anchor,
    merge_espn_prospect_media,
    merge_live_prospect_data,
    recruiting_star_badge,
)


def _inject_draft_css() -> None:
    st.markdown(
        """
        <style>
        .draft-hero, .draft-card, .draft-nav-shell {
          border-radius: 22px;
          border: 1px solid rgba(148, 163, 184, 0.14);
          background: linear-gradient(180deg, var(--secondary-background-color), var(--background-color));
          box-shadow: 0 18px 44px rgba(2, 6, 23, 0.16);
        }
        .draft-hero {
          padding: 1.25rem 1.3rem;
          margin-bottom: 1rem;
          background:
            radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%),
            radial-gradient(circle at bottom right, rgba(245,158,11,0.10), transparent 22%),
            linear-gradient(180deg, var(--secondary-background-color), var(--background-color));
        }
        .draft-card, .draft-nav-shell {
          padding: 1rem 1.05rem;
          margin-bottom: 1rem;
        }
        .draft-hero-kicker, .draft-section-kicker, .draft-kv-label {
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-size: 0.7rem;
          font-weight: 800;
          color: rgba(148,163,184,0.84);
        }
        .draft-hero-title {
          color: var(--text-color);
          font-size: clamp(1.7rem, 2vw, 2.2rem);
          font-weight: 800;
          letter-spacing: -0.03em;
          margin: 0.35rem 0 0.45rem 0;
        }
        .draft-hero-copy, .draft-note, .draft-nav-copy {
          color: var(--text-color);
          opacity: 0.76;
          line-height: 1.55;
        }
        .draft-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.9rem;
        }
        .draft-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          padding: 0.42rem 0.78rem;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(15,23,42,0.52);
          color: var(--text-color);
          font-size: 0.84rem;
          font-weight: 700;
        }
        .draft-chip.is-accent {
          background: rgba(59,130,246,0.16);
          border-color: rgba(59,130,246,0.28);
          color: #a5c8ff;
        }
        .draft-chip.is-warning {
          background: rgba(245,158,11,0.12);
          border-color: rgba(245,158,11,0.24);
          color: #fcd34d;
        }
        .draft-grid-2 {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.95rem;
        }
        .draft-snapshot-card {
          border-radius: 18px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(15,23,42,0.38);
          padding: 0.9rem;
        }
        .draft-snapshot-value {
          color: var(--text-color);
          font-size: 1.24rem;
          font-weight: 800;
          letter-spacing: -0.03em;
          margin-top: 0.28rem;
        }
        .draft-snapshot-note {
          color: var(--text-color);
          opacity: 0.78;
          line-height: 1.5;
        }
        .draft-matrix-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.8rem;
          margin-top: 0.7rem;
        }
        .draft-matrix-card {
          border-radius: 16px;
          border: 1px solid rgba(148,163,184,0.10);
          background: rgba(15,23,42,0.32);
          padding: 0.82rem 0.88rem;
        }
        .draft-matrix-top {
          display: flex;
          justify-content: space-between;
          gap: 0.9rem;
          align-items: baseline;
          margin-bottom: 0.25rem;
        }
        .draft-matrix-name {
          color: var(--text-color);
          font-size: 0.96rem;
          font-weight: 800;
          letter-spacing: -0.02em;
        }
        .draft-matrix-score {
          color: #cbd5e1;
          font-size: 0.82rem;
          font-weight: 700;
          white-space: nowrap;
        }
        .draft-matrix-signal {
          color: var(--text-color);
          opacity: 0.72;
          font-size: 0.83rem;
          line-height: 1.5;
        }
        .draft-emphasis {
          display: inline-flex;
          margin-top: 0.45rem;
          padding: 0.22rem 0.55rem;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          border: 1px solid rgba(148,163,184,0.12);
        }
        .draft-emphasis.is-strength {
          color: #93c5fd;
          background: rgba(59,130,246,0.14);
          border-color: rgba(59,130,246,0.22);
        }
        .draft-emphasis.is-swing {
          color: #fcd34d;
          background: rgba(245,158,11,0.12);
          border-color: rgba(245,158,11,0.22);
        }
        .draft-emphasis.is-weakness {
          color: #fca5a5;
          background: rgba(239,68,68,0.12);
          border-color: rgba(239,68,68,0.22);
        }
        .draft-kv-table {
          display: grid;
          gap: 0.55rem;
          margin-top: 0.6rem;
        }
        .draft-kv-row {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          padding-bottom: 0.55rem;
          border-bottom: 1px solid rgba(148,163,184,0.08);
        }
        .draft-kv-value {
          color: var(--text-color);
          font-weight: 700;
          text-align: right;
        }
        @media (max-width: 900px) {
          .draft-grid-2 {
            grid-template-columns: 1fr;
          }
          .draft-matrix-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_headshot(headshot_url: str | None, alt_text: str) -> None:
    fallback_url = get_placeholder_draft_headshot()
    source = headshot_url or fallback_url
    escaped_alt = html.escape(alt_text or "Prospect headshot")
    st.markdown(
        (
            f'<img src="{source}" alt="{escaped_alt}" '
            f'onerror="this.onerror=null;this.src=\'{fallback_url}\';" '
            'style="width:220px;max-width:100%;aspect-ratio:1/1;object-fit:cover;'
            'border-radius:20px;border:1px solid rgba(148,163,184,0.14);background:rgba(15,23,42,0.44);" />'
        ),
        unsafe_allow_html=True,
    )


def _render_key_value_table(rows: list[tuple[str, str]]) -> None:
    st.markdown('<div class="draft-kv-table">', unsafe_allow_html=True)
    for label, value in rows:
        st.markdown(
            f"""
            <div class="draft-kv-row">
                <div class="draft-kv-label">{label}</div>
                <div class="draft-kv-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _open_draft_profile(prospect: dict) -> None:
    st.session_state["draft_selected_profile"] = dict(prospect)
    st.session_state["draft_pending_workspace_view"] = "Prospect Profiles"
    st.session_state["draft_live_player_search"] = str(prospect.get("full_name") or "")


def _draft_ai_prompt(prospect: dict, advanced_rows: list[tuple[str, str]]) -> str:
    context = {
        "full_name": prospect.get("full_name"),
        "school_or_team": prospect.get("school_or_team"),
        "position": prospect.get("position"),
        "draft_class": prospect.get("draft_class"),
        "consensus_range": prospect.get("consensus_range") or prospect.get("projected_range"),
        "big_board_tier": prospect.get("big_board_tier"),
        "pedigree_tier": prospect.get("pedigree_tier"),
        "stats": {
            "ppg": prospect.get("ppg"),
            "rpg": prospect.get("rpg"),
            "apg": prospect.get("apg"),
            "spg": prospect.get("spg"),
            "bpg": prospect.get("bpg"),
            "fg_pct": prospect.get("fg_pct"),
            "three_pct": prospect.get("three_pct"),
            "ft_pct": prospect.get("ft_pct"),
            "ts_pct": prospect.get("ts_pct"),
        },
        "advanced_profile": dict(advanced_rows),
    }
    return (
        "You are an NBA draft scouting assistant. Return compact JSON with keys "
        "nba_archetype, best_case_outcome, median_outcome, floor_risk, swing_skill. "
        "Use the structured profile below. Be specific and basketball-native, not generic. "
        "If consensus_range or big_board_tier exists, respect that market framing in your language.\n\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )


def _build_draft_ai_translation(prospect: dict, ai_client) -> dict | None:
    if ai_client is None:
        return None
    try:
        advanced_rows = build_advanced_college_profile(prospect)
        raw = ai_generate_text(ai_client, prompt=_draft_ai_prompt(prospect, advanced_rows), json_mode=True, max_output_tokens=500, temperature=0.35)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def _resolve_selected_prospect(searchable_universe: list[dict]) -> tuple[dict | None, str | None]:
    selected = st.session_state.pop("draft_selected_profile", None)
    if selected:
        return selected, None

    st.caption("Search any current NCAA player or a curated 2026 draft name.")
    live_query = st.text_input(
        "Unified prospect search",
        placeholder="Example: Caleb Wilson, AJ Dybantsa, North Carolina...",
        key="draft_live_player_search",
        label_visibility="collapsed",
    )
    if len(live_query.strip()) < 2:
        return None, None

    local_matches = [
        row for row in searchable_universe
        if live_query.strip().lower() in str(row.get("full_name", "")).lower()
        or live_query.strip().lower() in str(row.get("school_or_team", "")).lower()
    ][:12]
    live_matches = search_ncaab_players(live_query, per_page=8) if ncaab_api_ready() else []

    options: dict[str, tuple[str, dict]] = {}
    for row in local_matches:
        options[f"{draft_search_label(row)} • Seeded"] = ("local", row)
    for row in live_matches:
        label = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
        school = ((row.get("team") or {}).get("college") or (row.get("team") or {}).get("full_name") or "—")
        options[f"{label} ({row.get('position') or '—'} • {school}) • Live NCAAB"] = ("live", row)

    if not options:
        st.info("No prospects matched that search yet.")
        return None, None

    st.caption("Live matches")
    option_labels = list(options.keys())[:8]
    for idx, label in enumerate(option_labels):
        selected_kind, payload = options[label]
        if st.button(label, key=f"draft_live_match_{idx}", use_container_width=True):
            if selected_kind == "local":
                return dict(payload), None
            st.session_state["draft_search_result"] = label
            return None, label
    return None, None


def _render_overview_page(prospects: list[dict], draft_class_options: list[str]) -> None:
    class_counts = {
        draft_class: len([p for p in prospects if str(p.get("draft_class", "")).strip() == draft_class])
        for draft_class in draft_class_options if draft_class != "All"
    }
    st.markdown('<div class="draft-card">', unsafe_allow_html=True)
    st.markdown('<div class="draft-section-kicker">Draft home</div>', unsafe_allow_html=True)
    st.markdown("### Multi-page prospect workspace")
    st.markdown(
        '<div class="draft-note">Use the draft pages like a real operating surface: keep profiles separate from mocks, and keep the archive separate from the active class.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="draft-chip-row">', unsafe_allow_html=True)
    for draft_class, count in class_counts.items():
        chip_class = "draft-chip is-accent" if draft_class == str(DEFAULT_DRAFT_SEASON + 1) else "draft-chip"
        st.markdown(f'<span class="{chip_class}">Class {draft_class}: {count} prospects</span>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="draft-card"><div class="draft-section-kicker">Current build</div><h3>What this workspace can already do</h3><div class="draft-note">Search any current NCAA player, anchor draft context with a curated class layer, show ESPN mock picks on-site, and use AI as the main translation engine for how a player projects.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="draft-card"><div class="draft-section-kicker">Next build queue</div><h3>What comes next</h3><div class="draft-note">Prospect compare, deeper college percentiles, team-fit pages, and stronger scouting-note coverage across the wider class.</div></div>', unsafe_allow_html=True)


def _render_profiles_page(prospects: list[dict], ai_client) -> None:
    local_rows = load_local_prospect_metadata()
    anchors = load_prospect_consensus_anchors()
    searchable_universe = build_searchable_draft_universe(local_rows, anchors)

    preselected, live_choice = _resolve_selected_prospect(searchable_universe)
    resolved = None
    live_sync_message = None
    if preselected:
        resolved = dict(preselected)
        live_player = lookup_ncaab_player(resolved.get("full_name", ""), resolved.get("school_or_team"))
        live_profile = load_live_prospect_profile(live_player, season=DEFAULT_DRAFT_SEASON) if live_player else None
        if live_profile:
            resolved = merge_live_prospect_data(resolved, live_profile)
            live_sync_message = f"Live NCAAB season sync active via balldontlie for {DEFAULT_DRAFT_SEASON}-{str(DEFAULT_DRAFT_SEASON + 1)[-2:]}."
    elif live_choice:
        selected = st.session_state.get("draft_search_result")
        row = next(
            (
                r
                for r in search_ncaab_players(st.session_state.get("draft_live_player_search", ""), per_page=8)
                if f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() in str(selected)
            ),
            None,
        )
        if row:
            live_profile = load_live_prospect_profile(row, season=DEFAULT_DRAFT_SEASON)
            local_match = find_local_prospect(searchable_universe, live_profile.get("full_name", ""), live_profile.get("school_or_team"))
            resolved = merge_live_prospect_data(local_match or build_generated_prospect_profile(live_profile), live_profile)
            anchor = find_consensus_anchor(anchors, resolved.get("full_name", ""), resolved.get("school_or_team"))
            resolved = merge_consensus_anchor(resolved, anchor)
            live_sync_message = f"Live NCAAB season sync active via balldontlie for {DEFAULT_DRAFT_SEASON}-{str(DEFAULT_DRAFT_SEASON + 1)[-2:]}."

    if not resolved:
        st.info("Search for a prospect to open the draft profile.")
        return

    espn_media = load_espn_prospect_media(resolved)
    resolved = merge_espn_prospect_media(resolved, espn_media)

    ai_key = f"{resolved.get('full_name')}|{resolved.get('school_or_team')}|{resolved.get('draft_class')}"
    st.session_state.setdefault("draft_ai_translations", {})
    st.session_state.setdefault("draft_ai_translation_errors", {})
    if ai_key not in st.session_state["draft_ai_translations"] and ai_key not in st.session_state["draft_ai_translation_errors"]:
        ai_payload = _build_draft_ai_translation(resolved, ai_client)
        if ai_payload:
            st.session_state["draft_ai_translations"][ai_key] = ai_payload
        else:
            st.session_state["draft_ai_translation_errors"][ai_key] = "AI unavailable"
    ai_translation = st.session_state["draft_ai_translations"].get(ai_key)

    if ai_translation:
        for field in ["nba_archetype", "best_case_outcome", "median_outcome", "floor_risk", "swing_skill"]:
            if ai_translation.get(field):
                resolved[field] = ai_translation[field]

    strengths_matrix = build_strengths_weaknesses_matrix(resolved)
    translation_confidence = build_translation_confidence(resolved)
    outcome_bands = build_outcome_bands(resolved)

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        dossier_left, dossier_right = st.columns([0.75, 1.15], gap="medium")
        with dossier_left:
            st.markdown('<div class="draft-section-kicker">Prospect dossier</div>', unsafe_allow_html=True)
            _render_headshot(resolved.get("headshot_url"), resolved.get("full_name", "Prospect"))
        with dossier_right:
            badge = recruiting_star_badge(resolved)
            title = resolved.get("full_name", "Prospect")
            if badge:
                title = f"{title} {badge[0]}"
            st.markdown(f"## {title}")
            if badge:
                st.caption(badge[1])
            st.markdown(
                f"### {resolved.get('school_or_team', '—')} • {resolved.get('league', 'NCAAB')} • {resolved.get('position', '—')}"
            )
            chip_bits = [
                f'<span class="draft-chip is-accent">Stats source: {resolved.get("stats_source", "local metadata")}</span>',
                f'<span class="draft-chip">Season: {resolved.get("season", DEFAULT_DRAFT_SEASON)}</span>',
                f'<span class="draft-chip">Headshot: {"ESPN fallback" if resolved.get("espn_id") else "placeholder"}</span>',
            ]
            if resolved.get("espn_id"):
                chip_bits.append(f'<span class="draft-chip">ESPN ID: {resolved.get("espn_id")}</span>')
            st.markdown(f'<div class="draft-chip-row">{"".join(chip_bits)}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="draft-note" style="margin-top:0.75rem;">Measurement stage: <strong>{resolved.get("measurement_stage", "Unverified")}</strong> • Source: <strong>{resolved.get("measurement_source", "Live search profile")}</strong> • Confidence: <strong>{resolved.get("measurement_confidence", "Low")}</strong></div>',
                unsafe_allow_html=True,
            )
            if live_sync_message:
                st.caption(live_sync_message)

        cards = [
            {"kicker": "Projected range", "value": resolved.get("projected_range", "Board TBD"), "note": resolved.get("consensus_source", "Current draft framing")},
            {"kicker": "Pedigree tier", "value": resolved.get("pedigree_tier", "Live search profile"), "note": "How the draft market currently frames the talent"},
            {"kicker": "NBA archetype", "value": resolved.get("nba_archetype", "Draft translation pending"), "note": "How the profile translates"},
            {"kicker": "Swing skill", "value": resolved.get("swing_skill", "Draft translation pending"), "note": "The development lever that changes the ceiling"},
        ]
        st.markdown('<div class="draft-grid-2">', unsafe_allow_html=True)
        for card in cards:
            st.markdown(
                f"""
                <div class="draft-snapshot-card">
                    <div class="draft-section-kicker">{card['kicker']}</div>
                    <div class="draft-snapshot-value">{card['value']}</div>
                    <div class="draft-snapshot-note">{card['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Season snapshot</div>', unsafe_allow_html=True)
        _render_key_value_table(build_simple_college_profile(resolved))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Profile table</div>', unsafe_allow_html=True)
        _render_key_value_table(
            [
                ("Height", str(resolved.get("height", "—"))),
                ("Weight", str(resolved.get("weight", "—"))),
                ("Wingspan", str(resolved.get("wingspan", "—"))),
                ("Standing Reach", str(resolved.get("standing_reach", "—"))),
                ("Class", str(resolved.get("class_year", "—"))),
                ("Country", str(resolved.get("country", "—"))),
                ("Recruiting Stars", recruiting_star_badge(resolved)[0] if recruiting_star_badge(resolved) else "—"),
                ("Recruiting Source", str(resolved.get("recruiting_source", "—"))),
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="draft-section-kicker">NBA translation</div>', unsafe_allow_html=True)
        st.markdown("## How this profile projects")
        translation_cards = [
            ("Best-case outcome", resolved.get("best_case_outcome", "AI draft translation pending"), "Optimistic NBA path if the swing skill hits"),
            ("Median outcome", resolved.get("median_outcome", "AI draft translation pending"), "Most realistic current translation"),
            ("Floor risk", resolved.get("floor_risk", "AI draft translation pending"), "What could hold the profile back"),
            ("Draft tier", resolved.get("big_board_tier", "Unranked / Live Search"), "Your current scouting bucket"),
        ]
        for title, value, note in translation_cards:
            st.markdown(
                f"""
                <div class="draft-snapshot-card" style="margin-top:0.7rem;">
                    <div class="draft-section-kicker">{title}</div>
                    <div class="draft-snapshot-value">{value}</div>
                    <div class="draft-snapshot-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Translation confidence</div>', unsafe_allow_html=True)
        _render_key_value_table(translation_confidence)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Strengths / weaknesses matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="draft-matrix-grid">', unsafe_allow_html=True)
        for item in strengths_matrix:
            emphasis_class = (
                "is-strength" if item["emphasis"] == "Strength"
                else "is-weakness" if item["emphasis"] == "Weakness"
                else "is-swing"
            )
            st.markdown(
                f"""
                <div class="draft-matrix-card">
                    <div class="draft-matrix-top">
                        <div class="draft-matrix-name">{item["category"]}</div>
                        <div class="draft-matrix-score">{item["label"]}</div>
                    </div>
                    <div class="draft-matrix-signal">{item["signal"]}</div>
                    <span class="draft-emphasis {emphasis_class}">{item["emphasis"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Outcome bands</div>', unsafe_allow_html=True)
        _render_key_value_table(outcome_bands)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">Advanced college profile</div>', unsafe_allow_html=True)
        _render_key_value_table(build_advanced_college_profile(resolved))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="draft-card">', unsafe_allow_html=True)
        st.markdown('<div class="draft-section-kicker">AI draft translation</div>', unsafe_allow_html=True)
        st.markdown("### Primary scouting read")
        if st.button("Refresh AI Draft Translation", key=f"draft_ai_translation_{ai_key}", use_container_width=True):
            ai_payload = _build_draft_ai_translation(resolved, ai_client)
            if ai_payload:
                st.session_state["draft_ai_translations"][ai_key] = ai_payload
                st.session_state["draft_ai_translation_errors"].pop(ai_key, None)
                st.rerun()
            st.warning("AI translation could not be refreshed right now.")
        st.markdown(
            '<div class="draft-note">AI is the default narrative layer here. Consensus anchors still ground the board/range so the language stays tethered to the draft market.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_mock_page(prospects: list[dict]) -> None:
    espn_mock = load_espn_mock_2026()
    if not espn_mock:
        st.info("No ESPN mock draft source file is loaded yet.")
        return
    st.markdown(
        """
        <div class="draft-chip-row" style="margin-bottom:0.8rem;">
            <span class="draft-chip is-accent">Source: %s</span>
            <span class="draft-chip">%s</span>
            <span class="draft-chip">%s</span>
        </div>
        """
        % (
            espn_mock.get("source_name", "ESPN"),
            f"Published: {espn_mock.get('published_date', '—')}",
            espn_mock.get("round_scope", "Mock board"),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="draft-note"><strong>{espn_mock.get("title", "Latest ESPN mock draft")}</strong><br/>{espn_mock.get("status_note", "")}</div>
        """,
        unsafe_allow_html=True,
    )
    if espn_mock.get("url"):
        st.markdown(f"[Open latest ESPN mock draft]({espn_mock.get('url')})")
    if espn_mock.get("order_url"):
        st.markdown(f"[Open current ESPN draft order]({espn_mock.get('order_url')})")

    picks = list(espn_mock.get("picks") or [])
    if not picks:
        st.info("The ESPN mock source is loaded, but the picks array is still empty.")
        return
    for row in picks:
        player_name = str(row.get("player", "Prospect"))
        school = str(row.get("school", "—"))
        linked_profile = find_local_prospect(prospects, full_name=player_name, school_or_team=school)
        top, action = st.columns([0.84, 0.16])
        with top:
            st.markdown(
                f"""
                <div class="draft-snapshot-card" style="margin-top:0.7rem;">
                    <div class="draft-section-kicker">Pick {row.get("pick", "—")} • {row.get("team", "TBD")}</div>
                    <div class="draft-snapshot-value">{player_name}</div>
                    <div class="draft-snapshot-note">{school} • {row.get("range", "ESPN mock")} • {row.get("archetype", "Scouting context pending")}</div>
                    <div class="draft-note" style="margin-top:0.35rem;">{row.get("reason", "Imported from the latest ESPN mock draft source.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action:
            st.write("")
            st.write("")
            if linked_profile:
                if st.button("Open Profile", key=f"espn_mock_open_{row.get('pick')}"):
                    _open_draft_profile(linked_profile)
                    st.rerun()
            else:
                st.caption("Profile link pending")


def _render_archive_page(prospects: list[dict], draft_class_filter: str) -> None:
    archive_classes = sorted(
        {str(p.get("draft_class", "")).strip() for p in prospects if p.get("draft_class") and str(p.get("draft_class")) != str(DEFAULT_DRAFT_SEASON + 1)},
        reverse=True,
    )
    archive_target = draft_class_filter if draft_class_filter != "All" and draft_class_filter in archive_classes else (archive_classes[0] if archive_classes else "")
    archive_rows = [p for p in prospects if str(p.get("draft_class", "")).strip() == str(archive_target).strip()]
    st.markdown('<div class="draft-card">', unsafe_allow_html=True)
    st.markdown('<div class="draft-section-kicker">Draft archive</div>', unsafe_allow_html=True)
    st.markdown(f"### {archive_target or 'No archive class selected'} historical class")
    if archive_rows:
        _render_key_value_table(
            [
                (
                    row.get("full_name", "Prospect"),
                    f"Pick {row.get('draft_pick', '—')} • {row.get('drafted_team', '—')} • {row.get('nba_archetype', '—')}",
                )
                for row in archive_rows
            ]
        )
    else:
        st.info("No archived prospects are loaded for this class yet.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_draft_workspace(ai_client=None) -> None:
    _inject_draft_css()
    pending_workspace_view = st.session_state.pop("draft_pending_workspace_view", None)
    if pending_workspace_view:
        st.session_state["draft_workspace_view"] = pending_workspace_view

    local_rows = load_local_prospect_metadata()
    anchors = load_prospect_consensus_anchors()
    prospects = build_searchable_draft_universe(local_rows, anchors)
    active_class = str(DEFAULT_DRAFT_SEASON + 1)
    draft_class_options = ["All"] + sorted({str(p.get("draft_class", "Unknown")) for p in prospects if p.get("draft_class")}, reverse=True)
    class_options = ["All"] + sorted({str(p.get("class_year", "Unknown")) for p in prospects if p.get("class_year")})
    active_anchor_count = len([p for p in prospects if str(p.get("draft_class", "")).strip() == active_class])

    st.markdown(
        """
        <div class="draft-hero">
            <div class="draft-hero-kicker">Prospect Lab</div>
            <div class="draft-hero-title">Build the NBA Draft layer on top of your current NBA intelligence stack.</div>
            <div class="draft-hero-copy">
                Search any current NCAA player, anchor the class with curated prospect context, keep ESPN's latest mock in-app,
                and use AI as the default translation layer for how each profile projects to the NBA.
            </div>
            <div class="draft-chip-row">
                <span class="draft-chip is-accent">NCAAB API: %s</span>
                <span class="draft-chip">%s active-class anchors</span>
                <span class="draft-chip is-warning">AI translation is the default scouting voice</span>
            </div>
        </div>
        """
        % ("Ready" if ncaab_api_ready() else "Waiting for key", active_anchor_count),
        unsafe_allow_html=True,
    )

    controls_left, controls_middle, controls_right = st.columns([1.0, 0.75, 0.9])
    with controls_left:
        search_query = st.text_input("Prospect search", placeholder="Type a prospect, school, or league...", key="draft_search")
    with controls_middle:
        draft_class_filter = st.selectbox("Draft class", draft_class_options if draft_class_options else ["All"], index=1 if len(draft_class_options) > 1 else 0, key="draft_class_filter")
    with controls_right:
        class_filter = st.selectbox("Class filter", class_options if class_options else ["All"], key="draft_academic_class_filter")

    filtered = filter_prospects(prospects, search_query, draft_class_filter, class_filter)

    st.markdown('<div class="draft-nav-shell">', unsafe_allow_html=True)
    st.markdown('<div class="draft-section-kicker">Workspace view</div>', unsafe_allow_html=True)
    st.markdown('<div class="draft-nav-copy">Split the draft product into separate working surfaces so profiles, mocks, and archived classes each have their own space.</div>', unsafe_allow_html=True)
    draft_view = st.radio(
        "Draft workspace page",
        ["Overview", "Prospect Profiles", "Mock Draft", "Archive"],
        horizontal=True,
        label_visibility="collapsed",
        key="draft_workspace_view",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    client = ai_client if ai_client is not None else default_model
    if draft_view == "Overview":
        _render_overview_page(prospects, draft_class_options)
    elif draft_view == "Prospect Profiles":
        _render_profiles_page(filtered, client)
    elif draft_view == "Mock Draft":
        _render_mock_page(prospects)
    else:
        _render_archive_page(prospects, draft_class_filter)
