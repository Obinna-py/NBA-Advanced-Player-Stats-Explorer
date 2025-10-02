# nba_app/utils.py
import streamlit as st
import streamlit.components.v1 as components

def abbrev(text: str, max_len: int = 40) -> str:
    return text if len(text) <= max_len else text[:max_len-1] + "…"

def public_cols(df):
    hide_ids = {"PLAYER_ID","TEAM_ID","LEAGUE_ID"}
    return [c for c in df.columns if not c.startswith("TEAM_") and c not in hide_ids]

def make_anchor(id_):
    st.markdown(f"<div id='{id_}'></div>", unsafe_allow_html=True)

def smooth_scroll_to(id_):
    components.html(f"""
        <script>
          const el = window.parent.document.querySelector('#{id_}');
          if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
        </script>
    """, height=0)
