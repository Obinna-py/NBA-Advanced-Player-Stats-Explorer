# ideas.py
from __future__ import annotations
import numpy as np
import streamlit as st

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
