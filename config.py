# nba_app/config.py
import os
import streamlit as st
import google.generativeai as genai

def ensure_page_config():
    st.set_page_config(page_title="NBA Advanced Player Stats Explorer", layout="wide")

# Paste your key here if you don't want to use env vars or secrets:
# LOCAL fallback (only used if secrets/env vars are missing)
LOCAL_GEMINI_API_KEY = ""  # e.g., "AIza..."
LOCAL_BALLDONTLIE_API_KEY = ""

def _load_key(secret_names: list[str], env_names: list[str], local_fallback: str = ""):
    key = None

    if hasattr(st, "secrets"):
        for name in secret_names:
            key = st.secrets.get(name)
            if key:
                return key

    for name in env_names:
        key = os.getenv(name)
        if key:
            return key

    return local_fallback or None

_API_KEY = _load_key(
    ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    LOCAL_GEMINI_API_KEY,
)
BALLDONTLIE_API_KEY = _load_key(
    ["BALLDONTLIE_API_KEY"],
    ["BALLDONTLIE_API_KEY"],
    LOCAL_BALLDONTLIE_API_KEY,
)

if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        AI_ENABLED = True
    except Exception as e:
        # If something goes wrong configuring the SDK, disable AI gracefully
        st.warning(f"Gemini setup error: {e}")
        model = None
        AI_ENABLED = False
else:
    model = None
    AI_ENABLED = False
