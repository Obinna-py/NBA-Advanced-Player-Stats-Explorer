# ideas.py
from __future__ import annotations
import numpy as np
import streamlit as st
import json
import re
import csv
import io

# ---------- simple presets exposed to UI ----------
def presets():
    choices = ["Overview", "Scoring & Efficiency", "Playmaking & TOs", "Rebounding & Defense", "Peak & Trends"]
    topic_map = {
        "Overview": "balanced overview; mix of efficiency, usage, passing, rebounding, trends",
        "Scoring & Efficiency": "PPG, TS%, eFG%, PPS, shot selection, 3PAr, FTr, per-36 scoring",
        "Playmaking & TOs": "AST%, AST/TO, assist trends, turnover rate, usage vs passing load",
        "Rebounding & Defense": "TRB%, ORB%, DRB%, STL/36, BLK/36, defensive impact proxies",
        "Peak & Trends": "best season, worst season, year-over-year changes, prime window indicators",
    }
    return choices, topic_map

# ---------- internal seed question generators ----------
def _seed_eval_questions(player_name: str, ctx: dict) -> list[str]:
    season = ctx.get("season", "latest")
    ts = ctx.get("ts", np.nan); ppg = ctx.get("ppg", np.nan); usg = ctx.get("usg", np.nan)
    base = [
        f"What was {player_name}'s best season by TS%?",
        f"In {season}, is {player_name} more scorer or playmaker (PPG vs APG)?",
        f"Is {player_name} an efficient scorer (TS% vs league ~57%)?",
        f"Has {player_name}'s usage outpaced efficiency (USG vs TS%)?",
        f"Is {player_name} a good passer (AST% and AST/TO trend)?",
        f"Did {player_name} peak already? Which season looks like the peak?",
        f"How does {player_name}'s eFG% trend over the last 3 seasons?",
        f"Is {player_name} a volume 3PT shooter (3PAr) or selective?",
        f"Does {player_name} draw fouls (FTr) at a high rate?",
        f"Is {player_name} turnover-prone (TOV and AST/TO trend)?",
        f"What is {player_name}'s best scoring rate (PTS/36) season?",
        f"Did rebounding improve or decline (ORB%/DRB%/TRB% trend)?",
        f"How do FG% / 3P% / FT% evolve season to season?",
        f"Did {player_name} change role after a team switch (USG/AST% jump)?",
        f"Is {player_name}'s efficiency stable at higher minutes?",
        f"Which season combines peak scoring and playmaking (PTS/36 + AST%)?",
        f"What was the worst season and what slipped most?",
        f"What’s {player_name}'s prime window based on trends?",
    ]
    if not np.isnan(ts) and ts >= 60:
        base.insert(0, f"Is {player_name} sustaining elite TS% (≥{ts:.1f}%)?")
    if not np.isnan(ppg) and ppg >= 25:
        base.insert(1, f"Is {player_name}'s volume ({ppg:.1f} PPG) backed by TS%?")
    if not np.isnan(usg) and usg >= 28:
        base.insert(2, f"Does high usage (~{usg:.1f}%) hurt or help efficiency?")
    # dedupe
    out, seen = [], set()
    for q in base:
        k = q.lower()
        if k in seen: 
            continue
        seen.add(k); out.append(q)
    return out[:20]

def _seed_compare_questions(p1: str, p2: str, c1: dict, c2: dict) -> list[str]:
    import numpy as np

    def _as_ctx_dict(x):
        # Make sure we always have a dict with .get(...)
        if isinstance(x, dict):
            return x
        try:
            return dict(x)
        except Exception:
            # string / other → empty dict (no personalization)
            return {}

    c1 = _as_ctx_dict(c1)
    c2 = _as_ctx_dict(c2)

    q = [
        f"Who had the better peak season by TS%: {p1} or {p2}?",
        f"Who is the better passer (AST% and AST/TO)?",
        f"Who scores more efficiently (TS%, eFG%)?",
        f"Whose usage is higher and does it match efficiency?",
        f"Who is the better rebounder for position (TRB%)?",
        f"Which season was each player’s statistical peak?",
        f"Who improved more year over year (TS%/PPS trend)?",
        f"Who is more turnover-prone (TOV, AST/TO)?",
        f"Who gets to the line more (FTr)?",
        f"Who takes more threes (3PAr) and is it efficient?",
        f"Who contributes more per-36 on scoring and playmaking?",
        f"Whose efficiency holds at higher usage?",
        f"Who is closer to their prime based on recent trends?",
    ]

    # tiny personalizations (only if keys exist & are numeric)
    for label, ctx in [(p1, c1), (p2, c2)]:
        ts  = ctx.get("ts",  np.nan)
        usg = ctx.get("usg", np.nan)
        ppg = ctx.get("ppg", np.nan)
        if not np.isnan(ts) and ts >= 60:
            q.insert(0, f"Is {label} sustaining elite TS% (≥{ts:.1f}%)?")
        if not np.isnan(usg) and usg >= 28:
            q.insert(1, f"Does {label}'s usage (~{usg:.1f}%) outpace efficiency?")
        if not np.isnan(ppg) and ppg >= 25:
            q.insert(2, f"Is {label}'s {ppg:.1f} PPG more efficient than the other?")

    # dedupe & cap
    out, seen = [], set()
    for s in q:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k); out.append(s)
    return out[:20]

# ---------- optional model helpers ----------
def _ai_question_ideas(player_name: str, ctx: dict, model=None, topic_hint: str="") -> list[str]:
    seeds = _seed_eval_questions(player_name, ctx)
    if model is None:
        return seeds
    prompt = (
        "Generate 20 short, evaluative questions about ONE NBA player that can be answered "
        "strictly from these metrics: PPG, RPG, APG, FG%, 3P%, FT%, TS%, eFG%, PPS, 3PAr, FTr, "
        "USG% (true), AST%, TRB%/ORB%/DRB%, per-36 stats, TOV, AST/TO, and trends. "
        "No tactics/lineups/on-off. ≤ 14 words each. Numbered list.\n\n"
        f"Player: {player_name}\nContext: {ctx}\nFocus hint: {topic_hint or 'none'}"
    )
    try:
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 256, "temperature": 0.4})
        text = getattr(resp, "text", "") or ""
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        items = []
        for l in lines:
            items.append(l.split(" ", 1)[1] if l[:2].isdigit() and " " in l else l.lstrip("0123456789.) ").strip())
        filtered = [i for i in items if i and len(i) <= 80 and all(bad not in i.lower() for bad in ["coverage","scheme","lineup","on/off","play type"])]
        return filtered[:20] if filtered else seeds
    except Exception:
        return seeds

def _ai_compare_question_ideas(p1: str, p2: str, c1: dict, c2: dict, model=None, topic_hint: str="") -> list[str]:
    seeds = _seed_compare_questions(p1, p2, c1, c2)
    if model is None:
        return seeds
    prompt = (
        "Generate 18 short comparison questions about TWO NBA players answerable from per-season stats: "
        "PPG, RPG, APG, FG%, 3P%, FT%, TS%, eFG%, PPS, 3PAr, FTr, USG% (true), AST%, TRB%/ORB%/DRB%, "
        "TOV, AST/TO, per-36, simple trends. No tactics/lineups/on-off. ≤ 14 words. Numbered list.\n\n"
        f"Player A: {p1}  Context: {c1}\n"
        f"Player B: {p2}  Context: {c2}\n"
        f"Focus hint: {topic_hint or 'none'}"
    )
    try:
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": 256, "temperature": 0.4})
        text = getattr(resp, "text", "") or ""
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        items = []
        for l in lines:
            items.append(l.split(" ", 1)[1] if l[:2].isdigit() and " " in l else l.lstrip("0123456789.) ").strip())
        filtered = [i for i in items if i and len(i) <= 80 and all(bad not in i.lower() for bad in ["coverage","scheme","lineup","on/off","play type"])]
        return filtered[:20] if filtered else seeds
    except Exception:
        return seeds

# ---------- CACHED WRAPPERS (NOTICE THE _model ARG) ----------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_ai_question_ideas(player_name: str, ctx: dict, topic_hint: str, use_model: bool, _model=None) -> list[str]:
    """Cache-safe: _model is ignored by Streamlit's hasher (leading underscore)."""
    model_to_use = _model if use_model and _model is not None else None
    return _ai_question_ideas(player_name, ctx, model=model_to_use, topic_hint=topic_hint)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_ai_compare_question_ideas(p1: str, p2: str, c1: dict, c2: dict, topic_hint: str, use_model: bool, _model=None) -> list[str]:
    """Cache-safe: _model is ignored by Streamlit's hasher (leading underscore)."""
    model_to_use = _model if use_model and _model is not None else None
    return _ai_compare_question_ideas(p1, p2, c1, c2, model=model_to_use, topic_hint=topic_hint)


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m:
        return None

    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _fallback_career_phases_from_table(phase_table: str) -> dict:
    seasons = []
    try:
        reader = csv.DictReader(io.StringIO(phase_table))
        for row in reader:
            season = str(row.get("Season", "")).strip()
            if season:
                seasons.append(season)
    except Exception:
        seasons = []

    seasons = [s for s in seasons if s]
    if not seasons:
        return {
            "peak_season": "",
            "early": [],
            "prime": [],
            "late": [],
            "reasoning_short": {
                "early": "Not enough clean season data was available.",
                "prime": "Not enough clean season data was available.",
                "late": "Not enough clean season data was available.",
            },
            "confidence": 0.25,
        }

    n = len(seasons)
    if n == 1:
        early, prime, late = [], [seasons[0]], []
    elif n == 2:
        early, prime, late = [seasons[0]], [seasons[1]], []
    elif n == 3:
        early, prime, late = [seasons[0]], seasons[1:], []
    else:
        early_end = max(1, round(n * 0.3))
        late_start = max(early_end + 1, n - max(1, round(n * 0.25)))
        early = seasons[:early_end]
        prime = seasons[early_end:late_start]
        late = seasons[late_start:]
        if not prime:
            prime = [seasons[-1]]

    peak = prime[-1] if prime else seasons[-1]
    return {
        "peak_season": peak,
        "early": early,
        "prime": prime,
        "late": late,
        "reasoning_short": {
            "early": "This fallback groups the earliest seasons as developmental years when the model does not return valid JSON.",
            "prime": "This fallback treats the strongest recent stretch as the current prime window, especially for short careers.",
            "late": "This fallback only labels late-career seasons when there is a long enough timeline to justify that bucket.",
        },
        "confidence": 0.45 if n >= 3 else 0.35,
    }


def _ai_detect_career_phases(player_name: str, phase_table: str, model) -> dict:
    """
    phase_table: a string version of a compact table (csv or markdown).
    model: your Gemini model instance.
    Returns dict with early/prime/late and explanations.
    """

    prompt = f"""
You are an NBA analytics assistant.

Task:
Given ONLY the provided season-by-season stats, label the player's career into:
- early career
- prime
- late career

Rules:
- Use only the numbers in the table. Do not use outside knowledge.
- Prime can be long (e.g., sustained plateau) or short (1-3 seasons). Choose what the stats justify.
- Early should generally be before prime, late generally after prime.
- If the stats show a second peak, you may still output a single PRIME block; include that note in reasoning.
- Output MUST be valid JSON ONLY. No markdown, no extra text.

Output JSON schema:
{{
  "peak_season": "YYYY-YY",
  "early": ["YYYY-YY", ...],
  "prime": ["YYYY-YY", ...],
  "late": ["YYYY-YY", ...],
  "reasoning_short": {{
     "early": "1-2 sentences",
     "prime": "1-2 sentences",
     "late": "1-2 sentences"
  }},
  "confidence": 0.0
}}

Season-by-season table:
{phase_table}
"""

    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1200,
            "response_mime_type": "application/json",
        }
    )

    text = resp.text if hasattr(resp, "text") else str(resp)
    data = _extract_json_object(text)
    if data is not None:
        return data
    return _fallback_career_phases_from_table(phase_table)


@st.cache_data(show_spinner=False, ttl=3600)
def ai_detect_career_phases(player_name: str, phase_table: str, use_model: bool, _model=None) -> dict:
    """Cache-safe wrapper: _model is ignored by Streamlit's hasher."""
    model_to_use = _model if use_model and _model is not None else None
    if model_to_use is None:
        raise ValueError("AI model is not available.")
    return _ai_detect_career_phases(player_name, phase_table, model_to_use)
