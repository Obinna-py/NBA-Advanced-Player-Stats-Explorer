import math
import re
from copy import deepcopy

from draft_fetch import DEFAULT_DRAFT_SEASON


def _norm(text: str | None) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _to_float(value) -> float | None:
    try:
        if value in (None, "", "—"):
            return None
        return float(value)
    except Exception:
        return None


def filter_prospects(prospects: list[dict], search_query: str, draft_class_filter: str, class_filter: str) -> list[dict]:
    rows = list(prospects)
    if draft_class_filter and draft_class_filter != "All":
        rows = [row for row in rows if str(row.get("draft_class", "")).strip() == str(draft_class_filter).strip()]
    if class_filter and class_filter != "All":
        rows = [row for row in rows if str(row.get("class_year", "")).strip() == str(class_filter).strip()]
    if search_query.strip():
        q = search_query.strip().lower()
        rows = [
            row
            for row in rows
            if q in str(row.get("full_name", "")).lower()
            or q in str(row.get("school_or_team", "")).lower()
            or q in str(row.get("league", "")).lower()
        ]
    return rows


def draft_search_label(prospect: dict) -> str:
    name = prospect.get("full_name", "Prospect")
    school = prospect.get("school_or_team", "—")
    position = prospect.get("position", "—")
    return f"{name} ({position} • {school})"


def _range_from_rank(rank: int) -> str:
    if rank <= 1:
        return "No. 1 overall"
    if rank <= 3:
        return "Top 3"
    if rank <= 5:
        return "Top 5"
    if rank <= 10:
        return "Top 10"
    if rank <= 14:
        return "Lottery"
    if rank <= 20:
        return "Mid-first round"
    if rank <= 30:
        return "Late first round"
    return "Board watchlist"


def _tier_from_rank(rank: int) -> str:
    if rank <= 3:
        return "Tier 1: franchise-level top-three talent"
    if rank <= 8:
        return "Tier 2: top-tier lottery core piece"
    if rank <= 14:
        return "Tier 3: lottery upside swing"
    if rank <= 30:
        return "Tier 4: first-round value band"
    return "Tier 5: board watchlist"


def _pedigree_from_rank(rank: int) -> str:
    if rank <= 5:
        return "Elite top-tier freshman"
    if rank <= 15:
        return "Top-ranked freshman"
    if rank <= 30:
        return "Strong national recruit"
    return "Consensus board prospect"


def _derived_recruiting_stars(rank: int) -> int:
    if rank <= 25:
        return 5
    if rank <= 75:
        return 4
    return 3


def enrich_consensus_anchor(anchor: dict) -> dict:
    row = deepcopy(anchor)
    rank = int(_to_float(row.get("consensus_rank")) or 999)
    if rank < 999:
        row.setdefault("consensus_range", _range_from_rank(rank))
        row.setdefault("projected_range", row.get("consensus_range") or _range_from_rank(rank))
        row.setdefault("big_board_tier", _tier_from_rank(rank))
        row.setdefault("pedigree_tier", _pedigree_from_rank(rank))
        row.setdefault("consensus_source", "Consensus anchor")
        row.setdefault("board_anchor_note", "Structured consensus anchor for the current draft market.")
        row.setdefault("recruiting_stars", _derived_recruiting_stars(rank))
        row.setdefault("recruiting_source", "Derived recruiting tier from consensus anchor")
        row.setdefault("recruiting_class", 2025)
    row.setdefault("league", "NCAAB")
    return row


def merge_consensus_anchor(prospect: dict, anchor: dict | None) -> dict:
    if not anchor:
        return dict(prospect)
    merged = dict(prospect)
    for key, value in enrich_consensus_anchor(anchor).items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def find_local_prospect(prospects: list[dict], full_name: str, school_or_team: str | None = None) -> dict | None:
    target_name = _norm(full_name)
    target_school = _norm(school_or_team)
    for row in prospects:
        if _norm(row.get("full_name")) != target_name:
            continue
        if target_school:
            school = _norm(row.get("school_or_team"))
            if school and school != target_school:
                continue
        return dict(row)
    return None


def find_consensus_anchor(anchors: list[dict], full_name: str, school_or_team: str | None = None) -> dict | None:
    return find_local_prospect(anchors, full_name=full_name, school_or_team=school_or_team)


def build_anchor_seed_profile(anchor: dict) -> dict:
    row = enrich_consensus_anchor(anchor)
    row.setdefault("stats_source", "consensus anchor")
    row.setdefault("measurement_stage", "Unverified")
    row.setdefault("measurement_source", "Consensus anchor")
    row.setdefault("measurement_confidence", "Low")
    return row


def build_generated_prospect_profile(live_profile: dict) -> dict:
    row = dict(live_profile)
    position = str(row.get("position", "")).upper()
    ppg = _to_float(row.get("ppg")) or 0.0
    apg = _to_float(row.get("apg")) or 0.0
    rpg = _to_float(row.get("rpg")) or 0.0
    three_pct = _to_float(row.get("three_pct")) or 0.0

    if "G" in position and apg >= 4.5:
        archetype = "Lead guard creator with live dribble passing"
        swing = "Pull-up shooting and turnover control"
    elif "C" in position or ("F" in position and rpg >= 8):
        archetype = "Modern frontcourt prospect with interior utility"
        swing = "Defensive positioning and NBA spacing value"
    else:
        archetype = "Toolsy live-search prospect with upside depending on role translation"
        swing = "NBA translatability still needs a board-level scouting read"

    if ppg >= 17 and three_pct >= 35:
        best_case = "High-end offensive piece whose scoring tools can scale into real first-unit value."
    elif ppg >= 14:
        best_case = "Two-way starting-caliber prospect if the swing skill sharpens."
    else:
        best_case = "Rotation-caliber prospect with room to outperform the current statistical baseline."

    if "G" in position:
        median = "Rotation guard who can support primary creators and provide value through pace, passing, and secondary scoring."
        floor = "If the efficiency and physical translation lag, he may struggle to create enough separation against NBA size."
    elif "C" in position:
        median = "Rotation big who stays on the floor through size, rebounding, and basic offensive functionality."
        floor = "If the defensive reads and mobility do not translate, the role can become matchup-dependent."
    else:
        median = "Rotation forward who scores off cuts, transition, and connective offense."
        floor = "If the jumper stalls and the physical tools do not pop at the next level, the role gets narrower."

    row.setdefault("projected_range", "Board TBD")
    row.setdefault("big_board_tier", "Unranked / Live Search")
    row.setdefault("nba_archetype", archetype)
    row.setdefault("best_case_outcome", best_case)
    row.setdefault("median_outcome", median)
    row.setdefault("floor_risk", floor)
    row.setdefault("swing_skill", swing)
    row.setdefault("pedigree_tier", "Live search profile")
    return row


def merge_live_prospect_data(base: dict, live_profile: dict | None) -> dict:
    merged = dict(base)
    if not live_profile:
        return merged
    preserved_source = merged.get("stats_source")
    for key, value in live_profile.items():
        if value not in (None, "", []):
            merged[key] = value
    if preserved_source and not live_profile.get("stats_source"):
        merged["stats_source"] = preserved_source
    return merged


def merge_espn_prospect_media(base: dict, espn_profile: dict | None) -> dict:
    merged = dict(base)
    if not espn_profile:
        return merged
    for key, value in espn_profile.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def build_searchable_draft_universe(local_rows: list[dict], anchors: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for row in local_rows:
        merged = merge_consensus_anchor(row, find_consensus_anchor(anchors, row.get("full_name", ""), row.get("school_or_team")))
        key = f"{_norm(merged.get('full_name'))}|{_norm(merged.get('school_or_team'))}"
        rows.append(merged)
        seen.add(key)
    for anchor in anchors:
        merged = build_anchor_seed_profile(anchor)
        key = f"{_norm(merged.get('full_name'))}|{_norm(merged.get('school_or_team'))}"
        if key in seen:
            continue
        rows.append(merged)
        seen.add(key)
    rows.sort(key=lambda row: (_to_float(row.get("consensus_rank")) or 9999, row.get("full_name", "")))
    return rows


def recruiting_star_badge(prospect: dict) -> tuple[str, str] | None:
    stars = int(_to_float(prospect.get("recruiting_stars")) or 0)
    if stars <= 0:
        return None
    source = str(prospect.get("recruiting_source") or "Recruiting consensus").strip()
    recruiting_class = prospect.get("recruiting_class")
    source_line = source if not recruiting_class else f"{source} • Class {recruiting_class}"
    return ("★" * stars, source_line)


def build_advanced_college_profile(prospect: dict) -> list[tuple[str, str]]:
    minutes = _to_float(prospect.get("minutes")) or 0.0
    scale = (40.0 / minutes) if minutes else None

    def per40(value_key: str) -> str:
        value = _to_float(prospect.get(value_key))
        if value is None or scale is None:
            return "—"
        return f"{value * scale:.1f}"

    apg = _to_float(prospect.get("apg"))
    tpg = _to_float(prospect.get("tpg"))
    ast_to = "—"
    if apg is not None and tpg not in (None, 0):
        ast_to = f"{apg / tpg:.2f}"

    three_pct = _to_float(prospect.get("three_pct"))
    ts_pct = _to_float(prospect.get("ts_pct"))
    efficiency_band = "—"
    if ts_pct is not None:
        if ts_pct >= 60:
            efficiency_band = "Strong"
        elif ts_pct >= 55:
            efficiency_band = "Solid"
        else:
            efficiency_band = "Developing"

    creation_load = "—"
    if apg is not None and (_to_float(prospect.get("ppg")) or 0) >= 16:
        creation_load = "High" if apg >= 4 else "Medium"

    return [
        ("PTS / 40", per40("ppg")),
        ("REB / 40", per40("rpg")),
        ("AST / 40", per40("apg")),
        ("Stocks / 40", "—" if scale is None else f"{((_to_float(prospect.get('spg')) or 0) + (_to_float(prospect.get('bpg')) or 0)) * scale:.1f}"),
        ("3PA / 40", "—"),
        ("FTA / 40", "—"),
        ("AST / TO", ast_to),
        ("3PA Rate", "—"),
        ("FT Rate", "—"),
        ("Usage Proxy", "Lead load" if (_to_float(prospect.get("ppg")) or 0) >= 18 else "Support load"),
        ("Efficiency Band", efficiency_band),
        ("Creation Load", creation_load),
    ]


def build_simple_college_profile(prospect: dict) -> list[tuple[str, str]]:
    def fmt(value, suffix: str = "", decimals: int = 1) -> str:
        number = _to_float(value)
        if number is None:
            return "—"
        return f"{number:.{decimals}f}{suffix}"

    return [
        ("Games", fmt(prospect.get("games"), decimals=0)),
        ("MIN", fmt(prospect.get("minutes"))),
        ("PPG", fmt(prospect.get("ppg"))),
        ("RPG", fmt(prospect.get("rpg"))),
        ("APG", fmt(prospect.get("apg"))),
        ("SPG", fmt(prospect.get("spg"))),
        ("BPG", fmt(prospect.get("bpg"))),
        ("FG%", fmt(prospect.get("fg_pct"), suffix="%")),
        ("3P%", fmt(prospect.get("three_pct"), suffix="%")),
        ("FT%", fmt(prospect.get("ft_pct"), suffix="%")),
        ("TS%", fmt(prospect.get("ts_pct"), suffix="%")),
    ]


def _score_to_label(score: float) -> str:
    if score >= 4.4:
        return "Elite"
    if score >= 3.6:
        return "Strong"
    if score >= 2.8:
        return "Solid"
    if score >= 2.0:
        return "Developing"
    return "Question Mark"


def _format_score(score: float) -> str:
    return f"{_score_to_label(score)} ({score:.1f}/5)"


def build_strengths_weaknesses_matrix(prospect: dict) -> list[dict]:
    ppg = _to_float(prospect.get("ppg")) or 0.0
    rpg = _to_float(prospect.get("rpg")) or 0.0
    apg = _to_float(prospect.get("apg")) or 0.0
    spg = _to_float(prospect.get("spg")) or 0.0
    bpg = _to_float(prospect.get("bpg")) or 0.0
    fg_pct = _to_float(prospect.get("fg_pct")) or 0.0
    three_pct = _to_float(prospect.get("three_pct")) or 0.0
    ft_pct = _to_float(prospect.get("ft_pct")) or 0.0
    ts_pct = _to_float(prospect.get("ts_pct")) or 0.0
    position = str(prospect.get("position", "")).upper()

    categories = [
        {
            "category": "Shot Creation",
            "score": min(5.0, 1.5 + (ppg / 7.0) + (apg / 6.0)),
            "signal": "Creation burden and self-generated offense",
        },
        {
            "category": "Playmaking",
            "score": min(5.0, 1.2 + (apg / 2.0)),
            "signal": "Passing volume and connective decision-making",
        },
        {
            "category": "Shooting",
            "score": min(5.0, 1.0 + (three_pct / 12.0) + (ft_pct / 30.0)),
            "signal": "Current spacing value and touch indicators",
        },
        {
            "category": "Interior / Rim Pressure",
            "score": min(5.0, 1.0 + (fg_pct / 18.0) + (ppg / 10.0)),
            "signal": "Finishing efficiency and pressure on the paint",
        },
        {
            "category": "Rebounding",
            "score": min(5.0, 1.0 + (rpg / 2.1)),
            "signal": "Glass impact and physical board presence",
        },
        {
            "category": "Defensive Activity",
            "score": min(5.0, 1.0 + ((spg + bpg) / 0.9)),
            "signal": "Stocks production and defensive disruption",
        },
    ]

    if "G" in position:
        categories[1]["score"] = min(5.0, categories[1]["score"] + 0.4)
        categories[4]["score"] = max(1.0, categories[4]["score"] - 0.3)
    elif "C" in position:
        categories[4]["score"] = min(5.0, categories[4]["score"] + 0.5)
        categories[5]["score"] = min(5.0, categories[5]["score"] + 0.4)
    elif "F" in position:
        categories[0]["score"] = min(5.0, categories[0]["score"] + 0.15)
        categories[5]["score"] = min(5.0, categories[5]["score"] + 0.15)

    rows = []
    for row in categories:
        score = max(1.0, min(5.0, row["score"]))
        if score >= 3.6:
            emphasis = "Strength"
        elif score <= 2.2:
            emphasis = "Weakness"
        else:
            emphasis = "Swing"
        rows.append(
            {
                "category": row["category"],
                "score": score,
                "label": _format_score(score),
                "signal": row["signal"],
                "emphasis": emphasis,
            }
        )
    return rows


def build_translation_confidence(prospect: dict) -> list[tuple[str, str]]:
    three_pct = _to_float(prospect.get("three_pct")) or 0.0
    ft_pct = _to_float(prospect.get("ft_pct")) or 0.0
    ts_pct = _to_float(prospect.get("ts_pct")) or 0.0
    apg = _to_float(prospect.get("apg")) or 0.0
    stocks = (_to_float(prospect.get("spg")) or 0.0) + (_to_float(prospect.get("bpg")) or 0.0)
    ppg = _to_float(prospect.get("ppg")) or 0.0
    consensus_rank = _to_float(prospect.get("consensus_rank")) or 60.0

    role_score = min(5.0, 1.6 + (ppg / 10.0) + (apg / 8.0) + (stocks / 4.0))
    shooting_score = min(5.0, 1.0 + (three_pct / 14.0) + (ft_pct / 35.0))
    defense_score = min(5.0, 1.2 + (stocks / 1.2))
    upside_score = min(5.0, 1.0 + max(0.0, (25.0 - consensus_rank) / 7.0) + (ppg / 14.0))
    overall = min(5.0, (role_score + shooting_score + defense_score + upside_score) / 4.0)

    return [
        ("Overall translation confidence", _format_score(overall)),
        ("Role confidence", _format_score(role_score)),
        ("Shooting confidence", _format_score(shooting_score)),
        ("Defensive confidence", _format_score(defense_score)),
        ("Star-upside confidence", _format_score(upside_score)),
    ]


def build_outcome_bands(prospect: dict) -> list[tuple[str, str]]:
    archetype = str(prospect.get("nba_archetype", "Prospect")).strip()
    best_case = str(prospect.get("best_case_outcome", "Top-end projection still needs scouting context.")).strip()
    median = str(prospect.get("median_outcome", "Median projection still needs scouting context.")).strip()
    floor = str(prospect.get("floor_risk", "Floor case still needs scouting context.")).strip()
    swing = str(prospect.get("swing_skill", "Key swing skill still needs scouting context.")).strip()

    star_path = f"If the swing skill hits at a high level, this {archetype.lower()} can become a real top-end lineup driver. {best_case}"
    starter_path = f"The clean starter outcome is the most stable lane if the current strengths scale. {median}"
    rotation_path = f"Even without a major breakout, the profile can still hold rotation value if the role is simplified and optimized around strengths."
    miss_path = f"The miss path happens if the translatable skills narrow instead of broadening. {floor}"

    return [
        ("Star path", star_path),
        ("Starter path", starter_path),
        ("Rotation path", rotation_path),
        ("Miss path", miss_path),
        ("Outcome hinge", swing),
    ]
