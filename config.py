import os
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def ensure_page_config():
    st.set_page_config(page_title="NBA Advanced Player Stats Explorer", layout="wide")

# Paste your key here if you don't want to use env vars or secrets:
# LOCAL fallback (only used if secrets/env vars are missing)
LOCAL_OPENAI_API_KEY = ""
LOCAL_BALLDONTLIE_API_KEY = ""
AI_MODEL_NAME = "gpt-5.4-mini"

def _load_key(secret_names: list[str], env_names: list[str], local_fallback: str = ""):
    key = None

    try:
        secrets_obj = st.secrets
    except Exception:
        secrets_obj = None

    if secrets_obj is not None:
        for name in secret_names:
            try:
                key = secrets_obj.get(name)
            except Exception:
                key = None
            if key:
                return key

    for name in env_names:
        key = os.getenv(name)
        if key:
            return key

    return local_fallback or None

_API_KEY = _load_key(
    ["OPENAI_API_KEY"],
    ["OPENAI_API_KEY"],
    LOCAL_OPENAI_API_KEY,
)
BALLDONTLIE_API_KEY = _load_key(
    ["BALLDONTLIE_API_KEY"],
    ["BALLDONTLIE_API_KEY"],
    LOCAL_BALLDONTLIE_API_KEY,
)
AI_SETUP_ERROR = None

if _API_KEY:
    try:
        if OpenAI is None:
            raise ImportError("openai is not installed in this environment.")
        model = OpenAI(api_key=_API_KEY)
        AI_ENABLED = True
    except Exception as e:
        # If something goes wrong configuring the SDK, disable AI gracefully
        AI_SETUP_ERROR = str(e)
        model = None
        AI_ENABLED = False
else:
    model = None
    AI_ENABLED = False


def ai_generate_text(
    client,
    prompt: str | None = None,
    *,
    messages: list[dict] | None = None,
    max_output_tokens: int = 1024,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> str:
    if client is None:
        raise ValueError("AI client is not available.")

    if messages is None:
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided.")
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": AI_MODEL_NAME,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
    }

    # Temperature is supported on standard text generation models.
    payload["temperature"] = temperature

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**payload)
    message = response.choices[0].message if response and response.choices else None
    return (message.content or "").strip() if message else ""
