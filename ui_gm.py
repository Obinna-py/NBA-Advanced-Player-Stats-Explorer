import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from config import ai_generate_text, AI_SETUP_ERROR
try:
    from streamlit_searchbox import st_searchbox
except Exception:
    st_searchbox = None

from fetch import (
    get_player_career,
    get_player_birthdate,
    get_nba_headshot_url,
    get_balldontlie_player_contracts,
    get_balldontlie_player_contract_aggregates,
    search_players,
)
from metrics import (
    compute_full_advanced_stats,
    add_per_game_columns,
    build_ai_stat_packet,
    generate_player_summary,
    detect_player_archetype,
    compute_player_percentile_context,
    compute_impact_index,
)
from ui_compare import render_stat_text
from ui_player import (
    _add_team_record_column,
    _render_headshot_image,
    _render_hover_stat_cards,
    _inject_sticky_ai_rail_css,
    _friendly_ai_error_message,
    _age_from_birthdate,
    _contract_snapshot,
    _fmt_money,
)


_GM_SEARCHBOX_STYLE_OVERRIDES = {
    "searchbox": {
        "control": {
            "backgroundColor": "#111827",
            "borderColor": "rgba(255,255,255,0.12)",
            "borderRadius": 12,
            "minHeight": 46,
            "boxShadow": "none",
        },
        "input": {"color": "#f9fafb", "fontSize": 15},
        "placeholder": {"color": "rgba(255,255,255,0.45)", "fontSize": 14},
        "singleValue": {"color": "#f9fafb", "fontSize": 15, "fontWeight": 500},
        "menu": {
            "backgroundColor": "#0f172a",
            "borderRadius": 12,
            "overflow": "hidden",
            "border": "1px solid rgba(255,255,255,0.08)",
            "boxShadow": "0 18px 40px rgba(0,0,0,0.35)",
        },
        "menuList": {"backgroundColor": "#0f172a", "paddingTop": 6, "paddingBottom": 6},
        "option": {"fontSize": 14, "paddingTop": 10, "paddingBottom": 10, "paddingLeft": 12, "paddingRight": 12},
        "noOptionsMessage": {"color": "rgba(255,255,255,0.55)", "fontSize": 13},
    },
    "dropdown": {"fill": "#9ca3af", "width": 22, "height": 22, "rotate": True},
    "clear": {"width": 18, "height": 18, "icon": "cross", "clearable": "always"},
}

_NBA_TEAM_OPTIONS = [
    "Atlanta Hawks (ATL)",
    "Boston Celtics (BOS)",
    "Brooklyn Nets (BKN)",
    "Charlotte Hornets (CHA)",
    "Chicago Bulls (CHI)",
    "Cleveland Cavaliers (CLE)",
    "Dallas Mavericks (DAL)",
    "Denver Nuggets (DEN)",
    "Detroit Pistons (DET)",
    "Golden State Warriors (GSW)",
    "Houston Rockets (HOU)",
    "Indiana Pacers (IND)",
    "LA Clippers (LAC)",
    "Los Angeles Lakers (LAL)",
    "Memphis Grizzlies (MEM)",
    "Miami Heat (MIA)",
    "Milwaukee Bucks (MIL)",
    "Minnesota Timberwolves (MIN)",
    "New Orleans Pelicans (NOP)",
    "New York Knicks (NYK)",
    "Oklahoma City Thunder (OKC)",
    "Orlando Magic (ORL)",
    "Philadelphia 76ers (PHI)",
    "Phoenix Suns (PHX)",
    "Portland Trail Blazers (POR)",
    "Sacramento Kings (SAC)",
    "San Antonio Spurs (SAS)",
    "Toronto Raptors (TOR)",
    "Utah Jazz (UTA)",
    "Washington Wizards (WAS)",
]


def _load_player_gm_data(player: dict) -> dict:
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
    if raw_totals is not None and not raw_totals.empty:
        if raw_totals.attrs.get("provider") == "balldontlie":
            adv = raw_totals.copy()
        else:
            adv = compute_full_advanced_stats(raw_totals)
    else:
        adv = pd.DataFrame()
    adv = add_per_game_columns(adv, raw_pergame)
    adv = _add_team_record_column(adv)
    latest = adv.iloc[-1] if adv is not None and not adv.empty else pd.Series(dtype=object)
    latest_season_id = str(latest.get("SEASON_ID") or "")
    birthdate = get_player_birthdate(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )
    age_value = _age_from_birthdate(birthdate) if birthdate else None
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
    percentile_df = compute_player_percentile_context(player.get("full_name", "Player"), latest_season_id, adv)
    archetype = detect_player_archetype(player.get("full_name", "Player"), adv, percentile_df)
    impact = compute_impact_index(player.get("full_name", "Player"), adv)
    return {
        "pergame": raw_pergame if raw_pergame is not None else pd.DataFrame(),
        "adv": adv if adv is not None else pd.DataFrame(),
        "latest": latest,
        "birthdate": birthdate,
        "age": age_value,
        "contract_df": contract_df if contract_df is not None else pd.DataFrame(),
        "contract_agg_df": contract_agg_df if contract_agg_df is not None else pd.DataFrame(),
        "contract_snapshot": contract_snapshot,
        "percentile_df": percentile_df if percentile_df is not None else pd.DataFrame(),
        "archetype": archetype or {},
        "impact": impact or {},
    }


def _safe_num(value) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(num) else float(num)


def _cap_hit_millions(contract_snapshot: dict) -> float | None:
    cap_hit = _safe_num(contract_snapshot.get("cap_hit"))
    return None if cap_hit is None else cap_hit / 1_000_000.0


def _surplus_value_score(impact_payload: dict, contract_snapshot: dict, age_value: int | None) -> float | None:
    impact_score = _safe_num(impact_payload.get("score"))
    cap_hit_m = _cap_hit_millions(contract_snapshot)
    if impact_score is None:
        return None
    cap_penalty = 0.0 if cap_hit_m is None else max(cap_hit_m - 15.0, 0.0) * 0.8
    age_bonus = 8.0 if age_value is not None and age_value <= 24 else 3.0 if age_value is not None and age_value <= 28 else -2.0 if age_value is not None and age_value >= 32 else 0.0
    return float(max(impact_score + age_bonus - cap_penalty, 0.0))


def _asset_classification(impact_payload: dict, contract_snapshot: dict, age_value: int | None) -> tuple[str, str]:
    surplus = _surplus_value_score(impact_payload, contract_snapshot, age_value)
    if surplus is None:
        return "Unknown Asset", "Not enough production context is available to classify the asset cleanly."
    cap_hit_m = _cap_hit_millions(contract_snapshot) or 0.0
    if surplus >= 72 and (age_value is None or age_value <= 27):
        return "Franchise Cornerstone", "The production, age curve, and contract profile still read like a player you build around rather than shop."
    if surplus >= 60:
        return "Blue-Chip Core Asset", "This looks like a premium roster piece with strong trade market gravity and real on-court value."
    if surplus >= 50:
        return "Positive Starter Asset", "The player still projects as a net-positive trade asset because the impact outweighs the contract burden."
    if surplus >= 40 and cap_hit_m <= 40:
        return "Neutral Starter Asset", "The production and salary are close enough that teams could value him differently depending on timeline."
    if surplus >= 34:
        return "Context-Dependent Asset", "The value depends heavily on team situation, role, and whether the contract fits the timeline."
    return "Negative Salary Asset", "The contract burden is starting to outweigh the clean on-court surplus value."


def _timeline_fit_label(age_value: int | None, contract_snapshot: dict, impact_payload: dict) -> tuple[str, str]:
    years = _safe_num(contract_snapshot.get("contract_years")) or 0
    impact_score = _safe_num(impact_payload.get("score")) or 0
    if age_value is not None and age_value <= 24 and impact_score >= 55:
        return "Long-Term Build Piece", "Young enough to fit a rebuild and already good enough to matter to a serious long-range core."
    if age_value is not None and age_value <= 28 and years >= 2:
        return "Two-Timeline Core Piece", "Good enough to help now while still fitting a medium-term roster plan."
    if age_value is not None and age_value >= 31 and impact_score >= 60:
        return "Win-Now Piece", "The profile makes the most sense for contenders trying to maximize the next couple of seasons."
    return "Bridge Starter", "More useful as part of a good structure than as the centerpiece of a long-term plan."


def _contender_rebuild_lens(latest: pd.Series, impact_payload: dict, contract_snapshot: dict, age_value: int | None) -> dict:
    impact_score = _safe_num(impact_payload.get("score")) or 0.0
    team_win_pct = _safe_num(latest.get("TEAM_WIN_PCT")) or 0.0
    cap_hit_m = _cap_hit_millions(contract_snapshot) or 0.0
    contender = impact_score * 0.75 + team_win_pct * 25.0 - max(cap_hit_m - 45.0, 0.0) * 0.4
    rebuild = impact_score * 0.45 + (12.0 if age_value is not None and age_value <= 24 else 5.0 if age_value is not None and age_value <= 28 else -4.0)
    contender_label = "Contender-Friendly" if contender >= 58 else "Situational Contender Fit" if contender >= 46 else "More Valuable Outside True Contender Context"
    rebuild_label = "Rebuild-Friendly" if rebuild >= 42 else "Bridge Asset" if rebuild >= 32 else "Short-Term Piece"
    return {
        "contender_score": max(contender, 0.0),
        "rebuild_score": max(rebuild, 0.0),
        "contender_label": contender_label,
        "rebuild_label": rebuild_label,
    }


def _best_co_star_profile(archetype: dict, latest: pd.Series) -> dict:
    primary = str(archetype.get("primary") or "")
    style_tags = [str(tag) for tag in archetype.get("style_tags", [])]
    impact_tags = [str(tag) for tag in archetype.get("impact_tags", [])]
    ppg = _safe_num(latest.get("PPG", latest.get("PTS")))
    apg = _safe_num(latest.get("APG", latest.get("AST")))
    bpg = _safe_num(latest.get("BPG", latest.get("BLK")))
    if primary == "Playmaker" or apg is not None and apg >= 7:
        return {
            "title": "Two-Way Finishing Partner",
            "offense": "An athletic scorer who can finish, space enough to survive off the ball, and punish tilted defenses without needing the whole offense handed to him.",
            "defense": "A strong point-of-attack or wing defender who covers perimeter assignments so the offense-driving star is not carrying both ends.",
            "bad_fit": "Another heliocentric ball-dominant creator without defensive insulation.",
        }
    if "Rim Pressure" in impact_tags or (ppg is not None and ppg >= 24 and apg is not None and apg < 5):
        return {
            "title": "Playmaking Spacer",
            "offense": "A high-IQ secondary playmaker who can shoot, keep the ball moving, and feed the downhill pressure without killing spacing.",
            "defense": "A switchable forward or backline helper who gives lineup flexibility.",
            "bad_fit": "A cramped non-shooting frontcourt that shrinks the lane.",
        }
    if bpg is not None and bpg >= 2.0:
        return {
            "title": "Volume Guard Co-Star",
            "offense": "A pull-up guard or dynamic wing scorer who can create late-clock offense so the big is not asked to self-create every possession.",
            "defense": "Someone who can also survive the point-of-attack burden, letting the rim protector stay impactful near the basket.",
            "bad_fit": "Another big who needs the same paint space and does not add high-level guard creation.",
        }
    return {
        "title": "Two-Way Connector",
        "offense": "A co-star who can score without monopolizing the offense and still make quick decisions as a passer and spacer.",
        "defense": "A versatile defender who can absorb the hardest matchup assignments.",
        "bad_fit": "A stylistic duplicate who duplicates strengths but fails to patch the weak spots.",
    }


def _gm_snapshot_summary(player_name: str, latest: pd.Series, impact_payload: dict, archetype: dict, asset_label: str, timeline_label: str, lens: dict) -> str:
    team_record = latest.get("TEAM_RECORD") or "—"
    primary = archetype.get("primary") or "—"
    style = ", ".join(archetype.get("style_tags", [])[:3]) or "—"
    return (
        f"{player_name} currently reads as a {asset_label.lower()} with a {primary.lower()} core identity. "
        f"The strongest style tags are {style}, the current team record is {team_record}, "
        f"the timeline fit reads as {timeline_label.lower()}, and the contender/rebuild split is "
        f"{lens['contender_label'].lower()} vs {lens['rebuild_label'].lower()}."
    )


def _player_trade_value_prompt(
    player_name: str,
    pergame_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    contract_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    age_value: int | None,
    archetype: dict,
    impact_payload: dict,
    asset_label: str,
    timeline_label: str,
) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    snapshot = _contract_snapshot(contract_df, aggregate_df)
    return (
        "You are an expert NBA front-office analyst. Evaluate this player strictly as a trade asset.\n"
        "Write in markdown with these exact sections: Asset Classification, Contract + Production Value, Trade Market View, Best Team Contexts, Biggest Value Risks, Bottom Line.\n"
        "Blend age, contract, production, role, portability, and roster-building logic. Be practical like a front office, not hot-take driven.\n\n"
        f"Current age: {age_value if age_value is not None else '—'}\n"
        f"GM summary labels: Asset={asset_label}; Timeline Fit={timeline_label}\n"
        f"Archetype: primary={archetype.get('primary') or '—'}; secondary={archetype.get('secondary') or '—'}; "
        f"style tags={', '.join(archetype.get('style_tags', [])) or '—'}; impact tags={', '.join(archetype.get('impact_tags', [])) or '—'}\n"
        f"Impact Index score: {impact_payload.get('score', '—')}\n"
        f"Strongest driver: {impact_payload.get('strongest_driver', '—')}\n"
        f"Contract snapshot: cap hit={_fmt_money(snapshot.get('cap_hit'))}; average salary={_fmt_money(snapshot.get('average_salary'))}; "
        f"contract years={snapshot.get('contract_years') or '—'}; guaranteed={_fmt_money(snapshot.get('total_guaranteed'))}; status={snapshot.get('contract_status') or '—'}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_co_star_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame, archetype: dict, co_star: dict) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    return (
        "You are an expert NBA roster-construction analyst. Identify the best co-star profile for this player.\n"
        "Write in markdown with these exact sections: Ideal Co-Star Profile, Best Offensive Partner, Best Defensive Partner, Team-Building Rules, Bad Fits, Bottom Line.\n"
        "Keep it specific to roster construction, spacing, creation burden, defensive coverage, and playoff scaling.\n\n"
        f"Suggested co-star anchor: {co_star.get('title')}\n"
        f"Suggested offense fit: {co_star.get('offense')}\n"
        f"Suggested defense fit: {co_star.get('defense')}\n"
        f"Bad fit warning: {co_star.get('bad_fit')}\n"
        f"Archetype: primary={archetype.get('primary') or '—'}; secondary={archetype.get('secondary') or '—'}; "
        f"style tags={', '.join(archetype.get('style_tags', [])) or '—'}; impact tags={', '.join(archetype.get('impact_tags', [])) or '—'}\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _player_roster_fit_prompt(player_name: str, pergame_df: pd.DataFrame, adv_df: pd.DataFrame, asset_label: str, timeline_label: str, lens: dict) -> str:
    summary = generate_player_summary(player_name, pergame_df, adv_df)
    stat_packet = build_ai_stat_packet(player_name, pergame_df, adv_df)
    return (
        "You are an expert NBA GM strategist. Explain how this player fits into roster construction and team-building decisions.\n"
        "Write in markdown with these exact sections: Contender View, Rebuild View, Best Supporting Cast, Role On A Good Team, Main Roster Risks, Bottom Line.\n"
        "Use the stats and labels provided. Make the answer feel like front-office planning, not just scouting.\n\n"
        f"Asset label: {asset_label}\n"
        f"Timeline fit: {timeline_label}\n"
        f"Contender lens: {lens.get('contender_label')} ({lens.get('contender_score'):.1f})\n"
        f"Rebuild lens: {lens.get('rebuild_label')} ({lens.get('rebuild_score'):.1f})\n\n"
        f"Structured stat packet:\n{stat_packet}\n\n"
        f"Season summary:\n{summary}\n"
    )


def _trade_package_prompt(core_player: dict, trade_teams: list[dict], lens: str) -> str:
    def _asset_line(asset: dict) -> str:
        if asset.get("asset_type") == "pick":
            return (
                f"- Draft Pick: {asset.get('year')} {asset.get('team') or 'Team'} "
                f"{asset.get('round', '1st')}{' swap' if asset.get('is_swap') else ''}; "
                f"protection={asset.get('protection') or 'Unprotected'}; projected range={asset.get('projected_range') or 'Unknown'}; "
                f"estimated asset value={_draft_pick_value_score(asset):.1f}; "
                f"destination={asset.get('to_team') or 'Unassigned'}"
            )
        if asset.get("asset_type") == "other":
            return (
                f"- {asset.get('label') or 'Non-player asset'}: kind={asset.get('kind') or 'Other'}; "
                f"value tier={asset.get('value_tier') or '—'}; notes={asset.get('notes') or '—'}; "
                f"estimated asset value={_other_asset_value_score(asset):.1f}; "
                f"destination={asset.get('to_team') or 'Unassigned'}"
            )
        player_dict = asset
        gm_data = _load_player_gm_data(player_dict)
        latest = gm_data["latest"]
        impact_score = _safe_num(gm_data["impact"].get("score"))
        contract_snapshot = gm_data["contract_snapshot"]
        asset_label, _ = _asset_classification(gm_data["impact"], contract_snapshot, gm_data["age"])
        return (
            f"- {player_dict.get('full_name', 'Player')}: "
            f"Age {gm_data['age'] if gm_data['age'] is not None else '—'}, "
            f"PPG {_safe_num(latest.get('PPG', latest.get('PTS'))) if latest is not None else '—'}, "
            f"APG {_safe_num(latest.get('APG', latest.get('AST'))) if latest is not None else '—'}, "
            f"RPG {_safe_num(latest.get('RPG', latest.get('REB'))) if latest is not None else '—'}, "
            f"Impact Index {impact_score if impact_score is not None else '—'}, "
            f"Cap Hit {_fmt_money(contract_snapshot.get('cap_hit'))}, "
            f"Asset {asset_label}, "
            f"destination={asset.get('to_team') or 'Unassigned'}"
        )

    team_blocks = []
    for team in trade_teams:
        assets = team.get("assets", [])
        asset_lines = "\n".join(_asset_line(asset) for asset in assets) or "- No outgoing assets selected."
        team_blocks.append(
            "\n".join(
                [
                    f"Team: {team.get('team') or 'Unassigned'}",
                    f"Direction: {team.get('direction') or '—'}",
                    f"Cap situation: {team.get('cap') or '—'}",
                    f"Main roster need: {team.get('need') or '—'}",
                    "Outgoing assets:",
                    asset_lines,
                ]
            )
        )
    return (
        "You are an expert NBA GM evaluating a trade package.\n"
        "Write in markdown with these exact sections: Trade Framing, Team-By-Team Value, Best Asset In The Deal, Short-Term Winner, Long-Term Winner, Cap / Flexibility Impact, Risk Check, Bottom Line.\n"
        "Judge the trade through the selected team-building lens. Handle multi-team trades naturally, and explain what each team is buying and giving up like a real front office.\n\n"
        f"Core page player: {core_player.get('full_name', 'Player')}\n"
        f"Lens: {lens}\n\n"
        "Teams in the deal:\n"
        + "\n\n".join(team_blocks)
        + "\n"
    )


def _player_token(player: dict) -> str:
    return f"{player.get('source', '')}:{player.get('id', '')}"


def _asset_token(asset: dict) -> str:
    asset_type = asset.get("asset_type")
    if asset_type == "pick":
        return ":".join(
            [
                "pick",
                str(asset.get("team") or ""),
                str(asset.get("year") or ""),
                str(asset.get("round") or ""),
                str(asset.get("protection") or ""),
                str(asset.get("projected_range") or ""),
                "swap" if asset.get("is_swap") else "own",
            ]
        )
    if asset_type == "other":
        return ":".join(
            [
                "other",
                str(asset.get("kind") or ""),
                str(asset.get("label") or ""),
                str(asset.get("notes") or ""),
                str(asset.get("value_tier") or ""),
            ]
        )
    return _player_token(asset)


def _gm_player_signature(player: dict | None) -> str:
    if not player:
        return "none"
    return _player_token(player) + f":{player.get('full_name', '')}"


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


def _parse_team_abbrev(team_label: str) -> str:
    if "(" in str(team_label) and ")" in str(team_label):
        return str(team_label).split("(")[-1].split(")")[0].strip().upper()
    return ""


def _parse_team_name(team_label: str) -> str:
    return str(team_label).split("(")[0].strip()


def _normalize_team_key(value: str) -> str:
    return str(value or "").strip().lower()


def _team_matches_player(team_label: str, player: dict) -> bool:
    team_name = _normalize_team_key(_parse_team_name(team_label))
    team_abbrev = _normalize_team_key(_parse_team_abbrev(team_label))
    player_team_name = _normalize_team_key(player.get("team_name") or "")
    player_team_abbrev = _normalize_team_key(player.get("team_abbreviation") or "")
    return bool(team_name and player_team_name == team_name) or bool(team_abbrev and player_team_abbrev == team_abbrev)


def _team_search_suggestions(searchterm: str, team_label: str) -> list[tuple[str, dict]]:
    results = _player_search_suggestions(searchterm)
    return [(label, player) for label, player in results if _team_matches_player(team_label, player)]


def _suggest_team_label_for_player(player: dict | None) -> str | None:
    if not player:
        return None
    player_team_name = _normalize_team_key(player.get("team_name") or "")
    player_team_abbrev = _normalize_team_key(player.get("team_abbreviation") or "")
    for option in _NBA_TEAM_OPTIONS:
        if _normalize_team_key(_parse_team_name(option)) == player_team_name:
            return option
        if _normalize_team_key(_parse_team_abbrev(option)) == player_team_abbrev:
            return option
    return None


def _append_unique_player(players: list[dict], candidate: dict) -> list[dict]:
    candidate_token = _asset_token(candidate)
    if any(_asset_token(player) == candidate_token for player in players):
        return players
    return players + [candidate]


def _draft_pick_value_score(asset: dict) -> float:
    year = int(asset.get("year") or datetime.today().year)
    round_label = str(asset.get("round") or "1st")
    protection = str(asset.get("protection") or "Unprotected")
    projected_range = str(asset.get("projected_range") or "Mid")
    current_year = datetime.today().year
    distance_penalty = max(year - current_year, 0) * 1.8
    base = 72.0 if round_label == "1st" else 24.0
    range_bonus = {
        "Top 3": 18.0,
        "Top 5": 15.0,
        "Lottery": 12.0,
        "Mid 1st": 8.0,
        "Late 1st": 3.0,
        "Early 2nd": 2.0,
        "Mid 2nd": 0.0,
        "Late 2nd": -3.0,
    }.get(projected_range, 0.0)
    protection_penalty = {
        "Unprotected": 0.0,
        "Top-3 Protected": 4.0,
        "Top-5 Protected": 6.0,
        "Top-8 Protected": 8.0,
        "Top-10 Protected": 10.0,
        "Lottery Protected": 12.0,
        "Heavily Protected": 16.0,
    }.get(protection, 8.0)
    swap_modifier = -10.0 if asset.get("is_swap") else 0.0
    return max(base + range_bonus - protection_penalty - distance_penalty + swap_modifier, 2.0)


def _other_asset_value_score(asset: dict) -> float:
    kind = str(asset.get("kind") or "Other")
    value_tier = int(asset.get("value_tier") or 3)
    base = {
        "Expiring Contract": 18.0,
        "Cap Relief": 20.0,
        "Trade Exception": 14.0,
        "Draft Rights": 22.0,
        "Cash Flexibility": 12.0,
    }.get(kind, 15.0)
    return max(base + (value_tier - 3) * 5.0, 2.0)


def _asset_display_name(asset: dict) -> str:
    asset_type = asset.get("asset_type")
    if asset_type == "pick":
        return (
            f"{asset.get('year')} {asset.get('team') or 'Team'} {asset.get('round', '1st')}"
            + (" swap" if asset.get("is_swap") else "")
        )
    if asset_type == "other":
        return asset.get("label") or asset.get("kind") or "Other Asset"
    return asset.get("full_name", "Player")


def _destination_options(trade_teams: list[dict], current_index: int) -> list[str]:
    return [team.get("team") for idx, team in enumerate(trade_teams) if idx != current_index and team.get("team")]


def _trade_team_template(index: int, player: dict | None = None) -> dict:
    suggested = _suggest_team_label_for_player(player) if index == 0 else None
    return {
        "team": suggested or (_NBA_TEAM_OPTIONS[index] if index < len(_NBA_TEAM_OPTIONS) else _NBA_TEAM_OPTIONS[0]),
        "direction": "Contender" if index == 0 else "Rebuild",
        "cap": "Neutral",
        "need": "Shot Creation" if index == 0 else "Future Assets",
        "assets": [dict(player, asset_type="player", from_team=suggested or "", to_team="")] if index == 0 and player else [],
    }


def _render_trade_side_selector(
    label: str,
    state_key: str,
    search_key: str,
    fallback_query_key: str,
    fallback_choice_key: str,
) -> list[dict]:
    players = list(st.session_state.get(state_key, []))
    st.markdown(f"**{label}**")

    selected_player = None
    if st_searchbox is not None:
        selected_player = st_searchbox(
            _player_search_suggestions,
            label=f"Search {label.lower()}",
            placeholder="Start typing a player name...",
            key=search_key,
            clear_on_submit=False,
            edit_after_submit="option",
            style_overrides=_GM_SEARCHBOX_STYLE_OVERRIDES,
        )
    else:
        query = st.text_input(
            f"Search {label.lower()}",
            key=fallback_query_key,
            placeholder="Type at least 2 characters...",
        )
        results = _player_search_suggestions(query) if query else []
        if results:
            options = ["Select a player"] + [text for text, _ in results]
            choice = st.selectbox("Matches", options, index=0, key=fallback_choice_key, label_visibility="collapsed")
            if choice != "Select a player":
                selected_player = next((player for text, player in results if text == choice), None)

    if isinstance(selected_player, dict) and selected_player.get("id") is not None:
        new_players = _append_unique_player(players, selected_player)
        if len(new_players) != len(players):
            st.session_state[state_key] = new_players
            st.rerun()

    players = list(st.session_state.get(state_key, players))
    if players:
        for idx, side_player in enumerate(players):
            row_left, row_right = st.columns([3.2, 1.2])
            with row_left:
                if side_player.get("asset_type") == "pick":
                    meta = [
                        side_player.get("protection") or "Unprotected",
                        side_player.get("projected_range") or "Unknown range",
                        f"Value {_draft_pick_value_score(side_player):.1f}",
                    ]
                    st.caption(f"{_asset_display_name(side_player)} ({' • '.join(meta)})")
                elif side_player.get("asset_type") == "other":
                    meta = [
                        side_player.get("kind") or "Other",
                        f"Tier {side_player.get('value_tier') or '—'}",
                        f"Value {_other_asset_value_score(side_player):.1f}",
                    ]
                    if side_player.get("notes"):
                        meta.append(side_player.get("notes"))
                    st.caption(f"{_asset_display_name(side_player)} ({' • '.join(meta)})")
                else:
                    meta = []
                    if side_player.get("position"):
                        meta.append(side_player["position"])
                    if side_player.get("team_name"):
                        meta.append(side_player["team_name"])
                    st.caption(
                        f"{side_player.get('full_name', 'Player')}"
                        + (f" ({' • '.join(meta)})" if meta else "")
                    )
            with row_right:
                if st.button("Remove", key=f"{state_key}_remove_{idx}", use_container_width=True):
                    updated = list(players)
                    updated.pop(idx)
                    st.session_state[state_key] = updated
                    st.rerun()
    else:
        st.caption("No assets selected yet.")

    with st.expander(f"Add draft pick to {label}", expanded=False):
        year_options = list(range(datetime.today().year, datetime.today().year + 8))
        team = st.text_input("Owning team", key=f"{state_key}_pick_team", placeholder="Example: LAL")
        pick_year = st.selectbox("Draft year", year_options, index=1 if len(year_options) > 1 else 0, key=f"{state_key}_pick_year")
        round_label = st.selectbox("Round", ["1st", "2nd"], index=0, key=f"{state_key}_pick_round")
        protection = st.selectbox(
            "Protection",
            ["Unprotected", "Top-3 Protected", "Top-5 Protected", "Top-8 Protected", "Top-10 Protected", "Lottery Protected", "Heavily Protected"],
            index=0,
            key=f"{state_key}_pick_protection",
        )
        projected_range = st.selectbox(
            "Projected range",
            ["Top 3", "Top 5", "Lottery", "Mid 1st", "Late 1st", "Early 2nd", "Mid 2nd", "Late 2nd"],
            index=3,
            key=f"{state_key}_pick_range",
        )
        is_swap = st.checkbox("This is a pick swap", key=f"{state_key}_pick_swap")
        if st.button("Add Draft Pick", key=f"{state_key}_pick_add", use_container_width=True):
            pick_asset = {
                "asset_type": "pick",
                "team": (team or "Team").strip().upper(),
                "year": int(pick_year),
                "round": round_label,
                "protection": protection,
                "projected_range": projected_range,
                "is_swap": bool(is_swap),
            }
            st.session_state[state_key] = _append_unique_player(players, pick_asset)
            st.rerun()

    with st.expander(f"Add non-player asset to {label}", expanded=False):
        kind = st.selectbox(
            "Asset type",
            ["Expiring Contract", "Cap Relief", "Trade Exception", "Draft Rights", "Cash Flexibility"],
            index=0,
            key=f"{state_key}_other_kind",
        )
        label_text = st.text_input("Label", key=f"{state_key}_other_label", placeholder="Example: 2027 cap relief slot")
        notes = st.text_input("Notes", key=f"{state_key}_other_notes", placeholder="Optional notes")
        value_tier = st.slider("Value tier", min_value=1, max_value=5, value=3, step=1, key=f"{state_key}_other_value")
        if st.button("Add Non-Player Asset", key=f"{state_key}_other_add", use_container_width=True):
            other_asset = {
                "asset_type": "other",
                "kind": kind,
                "label": (label_text or kind).strip(),
                "notes": (notes or "").strip(),
                "value_tier": int(value_tier),
            }
            st.session_state[state_key] = _append_unique_player(players, other_asset)
            st.rerun()
    return players


def _render_trade_team_panel(
    trade_teams: list[dict],
    index: int,
    state_key: str,
    model,
    core_player: dict | None,
) -> None:
    team = dict(trade_teams[index])
    st.markdown(f"### Team {index + 1}")
    team["team"] = st.selectbox(
        "Team",
        _NBA_TEAM_OPTIONS,
        index=_NBA_TEAM_OPTIONS.index(team.get("team")) if team.get("team") in _NBA_TEAM_OPTIONS else min(index, len(_NBA_TEAM_OPTIONS) - 1),
        key=f"{state_key}_team_label_{index}",
    )
    config_cols = st.columns(3)
    with config_cols[0]:
        team["direction"] = st.selectbox(
            "Direction",
            ["Contender", "Playoff Team", "Retool", "Rebuild"],
            index=["Contender", "Playoff Team", "Retool", "Rebuild"].index(team.get("direction")) if team.get("direction") in ["Contender", "Playoff Team", "Retool", "Rebuild"] else 0,
            key=f"{state_key}_direction_{index}",
        )
    with config_cols[1]:
        team["cap"] = st.selectbox(
            "Cap",
            ["Tight", "Neutral", "Flexible"],
            index=["Tight", "Neutral", "Flexible"].index(team.get("cap")) if team.get("cap") in ["Tight", "Neutral", "Flexible"] else 1,
            key=f"{state_key}_cap_{index}",
        )
    with config_cols[2]:
        team["need"] = st.selectbox(
            "Need",
            ["Shot Creation", "Rim Protection", "Spacing", "Wings", "Defense", "Depth", "Future Assets"],
            index=["Shot Creation", "Rim Protection", "Spacing", "Wings", "Defense", "Depth", "Future Assets"].index(team.get("need")) if team.get("need") in ["Shot Creation", "Rim Protection", "Spacing", "Wings", "Defense", "Depth", "Future Assets"] else 0,
            key=f"{state_key}_need_{index}",
        )

    destination_opts = _destination_options(trade_teams, index)
    player_search_key = f"{state_key}_player_search_{index}"
    selected_player = None
    if st_searchbox is not None:
        selected_player = st_searchbox(
            lambda q, team_label=team["team"]: _team_search_suggestions(q, team_label),
            label="Add player asset",
            placeholder="Start typing a player from this team...",
            key=player_search_key,
            clear_on_submit=False,
            edit_after_submit="option",
            style_overrides=_GM_SEARCHBOX_STYLE_OVERRIDES,
        )
    else:
        query = st.text_input("Add player asset", key=f"{state_key}_player_query_{index}", placeholder="Type at least 2 characters...")
        matches = _team_search_suggestions(query, team["team"]) if query else []
        if matches:
            options = ["Select a player"] + [text for text, _ in matches]
            choice = st.selectbox("Matches", options, index=0, key=f"{state_key}_player_choice_{index}", label_visibility="collapsed")
            if choice != "Select a player":
                selected_player = next((player for text, player in matches if text == choice), None)
    if destination_opts:
        dest_for_player = st.selectbox(
            "Send player asset to",
            destination_opts,
            index=0,
            key=f"{state_key}_player_dest_{index}",
        )
    else:
        dest_for_player = ""
        st.caption("Add another team to create a destination.")
    if isinstance(selected_player, dict) and selected_player.get("id") is not None and dest_for_player:
        player_asset = dict(selected_player)
        player_asset["asset_type"] = "player"
        player_asset["from_team"] = team["team"]
        player_asset["to_team"] = dest_for_player
        existing_assets = list(team.get("assets", []))
        team["assets"] = _append_unique_player(existing_assets, player_asset)
        trade_teams[index] = team
        st.session_state[state_key] = trade_teams
        st.rerun()

    with st.expander(f"Add draft pick from {team['team']}", expanded=False):
        year_options = list(range(datetime.today().year, datetime.today().year + 8))
        pick_year = st.selectbox("Draft year", year_options, index=1 if len(year_options) > 1 else 0, key=f"{state_key}_pick_year_{index}")
        round_label = st.selectbox("Round", ["1st", "2nd"], index=0, key=f"{state_key}_pick_round_{index}")
        protection = st.selectbox(
            "Protection",
            ["Unprotected", "Top-3 Protected", "Top-5 Protected", "Top-8 Protected", "Top-10 Protected", "Lottery Protected", "Heavily Protected"],
            index=0,
            key=f"{state_key}_pick_protection_{index}",
        )
        projected_range = st.selectbox(
            "Projected range",
            ["Top 3", "Top 5", "Lottery", "Mid 1st", "Late 1st", "Early 2nd", "Mid 2nd", "Late 2nd"],
            index=3,
            key=f"{state_key}_pick_range_{index}",
        )
        is_swap = st.checkbox("This is a pick swap", key=f"{state_key}_pick_swap_{index}")
        dest_for_pick = st.selectbox(
            "Send pick to",
            destination_opts or [""],
            index=0,
            key=f"{state_key}_pick_dest_{index}",
            disabled=not destination_opts,
        )
        if st.button("Add Draft Pick", key=f"{state_key}_pick_add_{index}", use_container_width=True, disabled=not destination_opts):
            pick_asset = {
                "asset_type": "pick",
                "from_team": team["team"],
                "to_team": dest_for_pick,
                "team": _parse_team_abbrev(team["team"]) or _parse_team_name(team["team"]),
                "year": int(pick_year),
                "round": round_label,
                "protection": protection,
                "projected_range": projected_range,
                "is_swap": bool(is_swap),
            }
            team["assets"] = _append_unique_player(list(team.get("assets", [])), pick_asset)
            trade_teams[index] = team
            st.session_state[state_key] = trade_teams
            st.rerun()

    with st.expander(f"Add non-player asset from {team['team']}", expanded=False):
        kind = st.selectbox(
            "Asset type",
            ["Expiring Contract", "Cap Relief", "Trade Exception", "Draft Rights", "Cash Flexibility"],
            index=0,
            key=f"{state_key}_other_kind_{index}",
        )
        label_text = st.text_input("Label", key=f"{state_key}_other_label_{index}", placeholder="Example: 2027 cap relief slot")
        notes = st.text_input("Notes", key=f"{state_key}_other_notes_{index}", placeholder="Optional notes")
        value_tier = st.slider("Value tier", min_value=1, max_value=5, value=3, step=1, key=f"{state_key}_other_value_{index}")
        dest_for_other = st.selectbox(
            "Send non-player asset to",
            destination_opts or [""],
            index=0,
            key=f"{state_key}_other_dest_{index}",
            disabled=not destination_opts,
        )
        if st.button("Add Non-Player Asset", key=f"{state_key}_other_add_{index}", use_container_width=True, disabled=not destination_opts):
            other_asset = {
                "asset_type": "other",
                "from_team": team["team"],
                "to_team": dest_for_other,
                "kind": kind,
                "label": (label_text or kind).strip(),
                "notes": (notes or "").strip(),
                "value_tier": int(value_tier),
            }
            team["assets"] = _append_unique_player(list(team.get("assets", [])), other_asset)
            trade_teams[index] = team
            st.session_state[state_key] = trade_teams
            st.rerun()

    st.markdown("**Outgoing assets**")
    if team.get("assets"):
        for asset_idx, asset in enumerate(team["assets"]):
            row_left, row_right = st.columns([3.2, 1.2])
            with row_left:
                st.caption(f"{_asset_display_name(asset)} → {asset.get('to_team') or 'Unassigned'}")
            with row_right:
                if st.button("Remove", key=f"{state_key}_remove_asset_{index}_{asset_idx}", use_container_width=True):
                    updated_assets = list(team["assets"])
                    updated_assets.pop(asset_idx)
                    team["assets"] = updated_assets
                    trade_teams[index] = team
                    st.session_state[state_key] = trade_teams
                    st.rerun()
    else:
        st.caption("No outgoing assets selected yet.")

    trade_teams[index] = team


def _clear_gm_outputs() -> None:
    for key in [
        "player_trade_value_output",
        "player_trade_value_player_name",
        "player_trade_value_summary",
        "player_co_star_output",
        "player_co_star_player_name",
        "player_co_star_summary",
        "player_roster_fit_output",
        "player_roster_fit_player_name",
        "player_roster_fit_summary",
        "player_trade_package_output",
        "player_trade_package_player_name",
        "player_trade_package_summary",
    ]:
        st.session_state.pop(key, None)
    if st.session_state.get("player_report_mode") in {"trade-value", "co-star", "roster-fit", "trade-package"}:
        st.session_state["player_report_mode"] = None


def _render_trade_package_builder(core_player: dict | None, model, key_prefix: str) -> None:
    core_token = _player_token(core_player) if core_player else "gm_workspace"
    team_count_key = f"{key_prefix}_team_count_{core_token}"
    teams_state_key = f"{key_prefix}_trade_teams_{core_token}"
    if team_count_key not in st.session_state:
        st.session_state[team_count_key] = 3 if core_player else 2
    if teams_state_key not in st.session_state:
        st.session_state[teams_state_key] = [
            _trade_team_template(idx, core_player if idx == 0 else None)
            for idx in range(int(st.session_state[team_count_key]))
        ]
    if len(st.session_state[teams_state_key]) != int(st.session_state[team_count_key]):
        current_teams = list(st.session_state[teams_state_key])
        target_count = int(st.session_state[team_count_key])
        if len(current_teams) < target_count:
            for idx in range(len(current_teams), target_count):
                current_teams.append(_trade_team_template(idx, core_player if idx == 0 else None))
        else:
            current_teams = current_teams[:target_count]
        st.session_state[teams_state_key] = current_teams

    trade_lens = st.selectbox(
        "Trade lens",
        ["Balanced", "Win-Now", "Rebuild", "Salary Relief", "Star Hunt"],
        index=0,
        key=f"{key_prefix}_trade_lens_{core_token}",
    )
    count_cols = st.columns([1.2, 1.2, 3.6])
    with count_cols[0]:
        if st.button("Add Team", key=f"{key_prefix}_add_team_{core_token}", use_container_width=True, disabled=int(st.session_state[team_count_key]) >= 4):
            st.session_state[team_count_key] = min(int(st.session_state[team_count_key]) + 1, 4)
            st.rerun()
    with count_cols[1]:
        if st.button("Remove Team", key=f"{key_prefix}_remove_team_{core_token}", use_container_width=True, disabled=int(st.session_state[team_count_key]) <= 2):
            st.session_state[team_count_key] = max(int(st.session_state[team_count_key]) - 1, 2)
            st.rerun()
    with count_cols[2]:
        st.caption(f"Active teams: {int(st.session_state[team_count_key])}")

    trade_teams = list(st.session_state[teams_state_key])
    for idx in range(len(trade_teams)):
        _render_trade_team_panel(trade_teams, idx, teams_state_key, model, core_player)
        if idx < len(trade_teams) - 1:
            st.divider()
    st.session_state[teams_state_key] = trade_teams

    total_assets = sum(len(team.get("assets", [])) for team in trade_teams)
    valid_teams = [team for team in trade_teams if team.get("team")]
    if total_assets:
        if model and len(valid_teams) >= 2 and total_assets >= 2:
            if st.button("Run Trade Package Simulator", key=f"{key_prefix}_run_trade_package_{core_token}", use_container_width=True):
                prompt = _trade_package_prompt(core_player or {"full_name": "GM Workspace"}, trade_teams, trade_lens)
                with st.spinner("Simulating trade package…"):
                    try:
                        text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                        st.session_state["player_trade_package_output"] = text or "No response."
                        st.session_state["player_trade_package_player_name"] = (core_player or {}).get("full_name", "GM Workspace")
                        st.session_state["player_trade_package_summary"] = (
                            f"Lens: {trade_lens} • Teams: {len(valid_teams)} • Assets: {total_assets}"
                        )
                        st.session_state["player_report_mode"] = "trade-package"
                        st.rerun()
                    except Exception as e:
                        st.warning(_friendly_ai_error_message(e))
                        st.caption(f"Details: {type(e).__name__}")
        else:
            st.caption("Set up at least two teams and add at least two assets before running the trade package simulator.")


def render_player_trade_value_page() -> None:
    st.markdown("## 💼 Trade Value Analyzer")
    st.caption(st.session_state.get("player_trade_value_player_name") or "Selected Player")
    st.caption(st.session_state.get("player_trade_value_summary") or "Trade value context unavailable")
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to GM", key="back_to_gm_from_trade_value", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_app_section"] = "🏗 GM / Team Building"
            st.rerun()
    output = st.session_state.get("player_trade_value_output")
    if output:
        st.markdown(output)
    else:
        st.info("No trade value report is available right now. Generate one from the GM page first.")


def render_player_co_star_page() -> None:
    st.markdown("## 🤝 Best Co-Star Finder")
    st.caption(st.session_state.get("player_co_star_player_name") or "Selected Player")
    st.caption(st.session_state.get("player_co_star_summary") or "Co-star fit context unavailable")
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to GM", key="back_to_gm_from_co_star", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_app_section"] = "🏗 GM / Team Building"
            st.rerun()
    output = st.session_state.get("player_co_star_output")
    if output:
        st.markdown(output)
    else:
        st.info("No co-star analysis is available right now. Generate one from the GM page first.")


def render_player_roster_fit_page() -> None:
    st.markdown("## 🧱 Roster Fit Analyzer")
    st.caption(st.session_state.get("player_roster_fit_player_name") or "Selected Player")
    st.caption(st.session_state.get("player_roster_fit_summary") or "Roster fit context unavailable")
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to GM", key="back_to_gm_from_roster_fit", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_app_section"] = "🏗 GM / Team Building"
            st.rerun()
    output = st.session_state.get("player_roster_fit_output")
    if output:
        st.markdown(output)
    else:
        st.info("No roster fit report is available right now. Generate one from the GM page first.")


def render_player_trade_package_page() -> None:
    st.markdown("## 🔁 Trade Package Simulator")
    st.caption(st.session_state.get("player_trade_package_player_name") or "Selected Player")
    st.caption(st.session_state.get("player_trade_package_summary") or "Trade package context unavailable")
    _, top_right = st.columns([4.5, 1.2])
    with top_right:
        if st.button("Back to GM", key="back_to_gm_from_trade_package", use_container_width=True):
            st.session_state["player_report_mode"] = None
            st.session_state["requested_app_section"] = "🏗 GM / Team Building"
            st.rerun()
    output = st.session_state.get("player_trade_package_output")
    if output:
        st.markdown(output)
    else:
        st.info("No trade package report is available right now. Generate one from the GM page first.")


def gm_tab(player: dict, model, *, show_title: bool = True) -> None:
    if show_title:
        st.subheader("GM / Team Building")
    gm_data = _load_player_gm_data(player)
    adv = gm_data["adv"]
    if adv is None or adv.empty:
        st.info("No player data is available right now.")
        return

    latest = gm_data["latest"]
    archetype = gm_data["archetype"]
    impact = gm_data["impact"]
    contract_snapshot = gm_data["contract_snapshot"]
    age_value = gm_data["age"]
    asset_label, asset_reason = _asset_classification(impact, contract_snapshot, age_value)
    timeline_label, timeline_reason = _timeline_fit_label(age_value, contract_snapshot, impact)
    lens = _contender_rebuild_lens(latest, impact, contract_snapshot, age_value)
    co_star = _best_co_star_profile(archetype, latest)
    gm_summary = _gm_snapshot_summary(player.get("full_name", "Player"), latest, impact, archetype, asset_label, timeline_label, lens)
    headshot_url = get_nba_headshot_url(
        player["id"],
        player_name=player.get("full_name"),
        player_source=player.get("source"),
    )

    hero_col, summary_col = st.columns([1, 3])
    with hero_col:
        _render_headshot_image(headshot_url, 170, player.get("full_name", "Player"))
        st.caption(player.get("full_name", ""))
    with summary_col:
        render_stat_text("Front-office tools for asset value, roster construction, and trade planning.", small=True)
        latest_team_record = latest.get("TEAM_RECORD")
        if pd.notna(latest_team_record) and latest_team_record:
            render_stat_text(f"Current team record: {latest_team_record}", small=True)
        _render_hover_stat_cards(
            [
                ("Age", str(age_value) if age_value is not None else "—"),
                ("Asset Class", asset_label),
                ("Timeline Fit", timeline_label),
                ("Impact Index", f"{_safe_num(impact.get('score')):.1f}" if _safe_num(impact.get("score")) is not None else "—"),
            ],
            columns_per_row=2,
        )

    if "show_gm_ai_rail" not in st.session_state:
        st.session_state["show_gm_ai_rail"] = True
    rail_toggle_col_left, rail_toggle_col_right = st.columns([4.5, 1.2])
    with rail_toggle_col_right:
        if st.button(
            "Hide AI sidebar" if st.session_state["show_gm_ai_rail"] else "Show AI sidebar",
            key="toggle_gm_ai_rail",
            use_container_width=True,
        ):
            st.session_state["show_gm_ai_rail"] = not st.session_state["show_gm_ai_rail"]
            st.rerun()

    if st.session_state["show_gm_ai_rail"]:
        _inject_sticky_ai_rail_css("sticky-gm-ai-rail")
        main_col, ai_col = st.columns([3.2, 1.35], gap="large")
    else:
        main_col = st.container()
        ai_col = None

    with main_col:
        st.markdown("### 🧭 GM Snapshot")
        render_stat_text(gm_summary, small=True)
        _render_hover_stat_cards(
            [
                ("Cap Hit", _fmt_money(contract_snapshot.get("cap_hit"))),
                ("Avg Salary", _fmt_money(contract_snapshot.get("average_salary"))),
                ("Trade Value", f"{_surplus_value_score(impact, contract_snapshot, age_value):.1f}" if _surplus_value_score(impact, contract_snapshot, age_value) is not None else "—"),
                ("Strongest Driver", str(impact.get("strongest_driver") or "—")),
            ],
            columns_per_row=2,
        )
        st.caption(asset_reason)
        st.caption(timeline_reason)

        st.markdown("### 💰 Contract + Production Value")
        cap_hit_m = _cap_hit_millions(contract_snapshot)
        impact_score = _safe_num(impact.get("score"))
        cost_efficiency = None if cap_hit_m in (None, 0) or impact_score is None else impact_score / cap_hit_m
        _render_hover_stat_cards(
            [
                ("Production Tier", impact.get("tier") or "—"),
                ("Cost Efficiency", f"{cost_efficiency:.2f}" if cost_efficiency is not None else "—"),
                ("Contract Status", str(contract_snapshot.get("contract_status") or "—")),
                ("Guaranteed", _fmt_money(contract_snapshot.get("total_guaranteed"))),
            ],
            columns_per_row=2,
        )
        render_stat_text(
            "This section combines Impact Index, salary load, guaranteed money, and age curve to estimate whether the contract still behaves like a positive roster asset.",
            small=True,
        )

        st.markdown("### 🏁 Contender vs Rebuild Lens")
        _render_hover_stat_cards(
            [
                ("Contender Score", f"{lens['contender_score']:.1f}"),
                ("Rebuild Score", f"{lens['rebuild_score']:.1f}"),
                ("Contender Read", lens["contender_label"]),
                ("Rebuild Read", lens["rebuild_label"]),
            ],
            columns_per_row=2,
        )
        render_stat_text(
            "Contender score rewards ready-made impact and playoff portability. Rebuild score rewards youth, timeline flexibility, and long-term value retention.",
            small=True,
        )

        st.markdown("### 🤝 Best Co-Star Finder")
        _render_hover_stat_cards(
            [
                ("Ideal Co-Star", co_star["title"]),
                ("Primary Role", str(archetype.get("primary") or "—")),
            ],
            columns_per_row=2,
        )
        render_stat_text(f"Best offensive partner: {co_star['offense']}")
        render_stat_text(f"Best defensive partner: {co_star['defense']}")
        render_stat_text(f"Bad fit warning: {co_star['bad_fit']}", small=True)

        st.markdown("### 🧱 Roster Fit Snapshot")
        role_text = archetype.get("primary") or "—"
        style_text = ", ".join(archetype.get("style_tags", [])[:4]) or "—"
        impact_tags = ", ".join(archetype.get("impact_tags", [])[:3]) or "—"
        render_stat_text(f"Best role on a good team: {role_text}.")
        render_stat_text(f"Style identity: {style_text}.")
        render_stat_text(f"Impact tags shaping roster decisions: {impact_tags}.", small=True)

        st.markdown("### 🔁 Trade Package Simulator")
        _render_trade_package_builder(player, model, "gm_player")

    if ai_col is not None:
        with ai_col:
            st.markdown('<div class="sticky-gm-ai-rail"></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 GM Tools")
            st.caption("Decision-focused tools for front-office context.")
            with st.expander("Trade Value Analyzer", expanded=False):
                if model:
                    if st.button("Open Trade Value Analysis", key=f"gm_trade_value_button_{current_token}", use_container_width=True):
                        prompt = _player_trade_value_prompt(
                            player.get("full_name", "Player"),
                            gm_data["pergame"],
                            adv,
                            gm_data["contract_df"],
                            gm_data["contract_agg_df"],
                            age_value,
                            archetype,
                            impact,
                            asset_label,
                            timeline_label,
                        )
                        with st.spinner("Evaluating trade value…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.6)
                                st.session_state["player_trade_value_output"] = text or "No response."
                                st.session_state["player_trade_value_player_name"] = player.get("full_name", "Selected Player")
                                st.session_state["player_trade_value_summary"] = f"{asset_label} • {timeline_label}"
                                st.session_state["player_report_mode"] = "trade-value"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
                    if st.session_state.get("player_trade_value_output"):
                        if st.button("Open Existing Trade Value Page", key=f"gm_open_trade_value_{current_token}", use_container_width=True):
                            st.session_state["player_report_mode"] = "trade-value"
                            st.rerun()
                else:
                    if AI_SETUP_ERROR:
                        st.info("AI is unavailable in this deployment right now.")
                        st.caption(f"Setup details: {AI_SETUP_ERROR}")
                    else:
                        st.info("Add your OpenAI API key to enable AI analysis.")
            with st.expander("Best Co-Star Finder", expanded=False):
                render_stat_text(f"Ideal co-star profile: {co_star['title']}", small=True)
                if model:
                    if st.button("Open Co-Star Report", key=f"gm_co_star_button_{current_token}", use_container_width=True):
                        prompt = _player_co_star_prompt(
                            player.get("full_name", "Player"),
                            gm_data["pergame"],
                            adv,
                            archetype,
                            co_star,
                        )
                        with st.spinner("Finding best co-star fit…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_co_star_output"] = text or "No response."
                                st.session_state["player_co_star_player_name"] = player.get("full_name", "Selected Player")
                                st.session_state["player_co_star_summary"] = co_star["title"]
                                st.session_state["player_report_mode"] = "co-star"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")
            with st.expander("Roster Fit Analyzer", expanded=False):
                render_stat_text(
                    f"Contender lens: {lens['contender_label']} • Rebuild lens: {lens['rebuild_label']}",
                    small=True,
                )
                if model:
                    if st.button("Open Roster Fit Report", key=f"gm_roster_fit_button_{current_token}", use_container_width=True):
                        prompt = _player_roster_fit_prompt(
                            player.get("full_name", "Player"),
                            gm_data["pergame"],
                            adv,
                            asset_label,
                            timeline_label,
                            lens,
                        )
                        with st.spinner("Analyzing roster fit…"):
                            try:
                                text = ai_generate_text(model, prompt, max_output_tokens=4096, temperature=0.65)
                                st.session_state["player_roster_fit_output"] = text or "No response."
                                st.session_state["player_roster_fit_player_name"] = player.get("full_name", "Selected Player")
                                st.session_state["player_roster_fit_summary"] = f"{lens['contender_label']} • {lens['rebuild_label']}"
                                st.session_state["player_report_mode"] = "roster-fit"
                                st.rerun()
                            except Exception as e:
                                st.warning(_friendly_ai_error_message(e))
                                st.caption(f"Details: {type(e).__name__}")


def render_gm_workspace(model) -> None:
    st.title("🏗 GM / Team Building")
    render_stat_text(
        "A front-office workspace for league-wide trade ideas, asset value, timeline decisions, and roster construction.",
        small=True,
    )
    st.markdown("### 🔁 Trade Package Simulator")
    _render_trade_package_builder(st.session_state.get("gm_player"), model, "gm_workspace")

    gm_player = st.session_state.get("gm_player")
    signature = _gm_player_signature(gm_player)
    if st.session_state.get("_gm_player_signature") != signature:
        _clear_gm_outputs()
        st.session_state["_gm_player_signature"] = signature

    if gm_player:
        st.divider()
        st.markdown("### 🎯 Focal Player GM View")
        render_stat_text(
            f"Using {gm_player.get('full_name', 'the selected player')} as the focal player for trade value, co-star fit, and roster-building analysis.",
            small=True,
        )
        gm_tab(gm_player, model, show_title=False)
    else:
        st.info("Use the GM sidebar to select a focal player if you want player-specific trade value, co-star, and roster-fit analysis.")
