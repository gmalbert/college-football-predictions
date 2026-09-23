"""utils/config.py — Secrets and configuration helpers."""
from __future__ import annotations
import os
from pathlib import Path


# streamlit is imported lazily inside get_secret() rather than at module scope.
# Importing it here pulled a ~57 MB dependency, and a streamlit import, into
# every consumer — including the FastAPI build and the command-line pipeline
# jobs, none of which touch st.secrets. Only get_secret() needs it, and only
# when actually running under Streamlit.


# Command-line jobs use this module directly, so load the project's ignored local
# secret file as well as accepting environment variables supplied by GitHub Actions.
_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _DOTENV_PATH.exists():
    for _line in _DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        if not _line or _line.lstrip().startswith("#") or "=" not in _line:
            continue
        _key, _value = _line.split("=", 1)
        _key = _key.strip()
        if _key:
            os.environ.setdefault(_key, _value.strip().strip("\"'").strip())


def _streamlit_secrets():
    """Streamlit's secrets store, or ``None`` when Streamlit is not available.

    Returns ``None`` both when streamlit is not installed and when it is
    installed but there is no running Streamlit runtime to read from.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        return st.secrets
    except Exception:  # noqa: BLE001 - no Streamlit runtime; fall back to env
        return None


def get_secret(section: str, key: str) -> str:
    """
    Fetch a secret from Streamlit secrets (Cloud/local) or environment variables.

    Environment variable lookup tries both the standard name and a 'CBBD_'
    variant to handle the common CFBD/CBBD spelling mix-up.
    """
    secrets = _streamlit_secrets()
    if secrets is not None:
        try:
            return secrets[section][key]
        except (KeyError, FileNotFoundError):
            pass

    env_key = f"{section.upper()}_{key.upper()}"
    value = os.environ.get(env_key)

    # Handle common CFBD → CBBD typo in .env files
    if value is None:
        alt_key = env_key.replace("CFBD_", "CBBD_")
        value = os.environ.get(alt_key)

    if value is None:
        raise ValueError(
            f"Secret '{section}.{key}' not found in Streamlit secrets "
            f"or environment variable '{env_key}'."
        )
    return value

