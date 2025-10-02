# nba_app/config.py
import os
import streamlit as st
import google.generativeai as genai

def ensure_page_config():
    st.set_page_config(page_title="NBA Advanced Player Stats Explorer", layout="wide")

# Paste your key here if you don't want to use env vars or secrets:
# LOCAL fallback (only used if secrets/env vars are missing)
LOCAL_GEMINI_API_KEY = ""  # e.g., "AIza..."

def _load_api_key():
    # 1) Streamlit secrets (recommended for Streamlit Cloud / local .streamlit/secrets.toml)
    key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None

    # 2) Environment variables
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        key = os.getenv("GOOGLE_API_KEY")

    # 3) Local fallback (last resort; only used if you paste it above)
    if not key and LOCAL_GEMINI_API_KEY:
        key = LOCAL_GEMINI_API_KEY

    return key

_API_KEY = _load_api_key()

if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        AI_ENABLED = True
    except Exception as e:
        # If something goes wrong configuring the SDK, disable AI gracefully
        st.warning(f"Gemini setup error: {e}")
        model = None
        AI_ENABLED = False
else:
    model = None
    AI_ENABLED = False
