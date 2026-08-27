"""utils/config.py — Secrets and configuration helpers."""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st


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


def get_secret(section: str, key: str) -> str:
    """
    Fetch a secret from Streamlit secrets (Cloud/local) or environment variables.

    Environment variable lookup tries both the standard name and a 'CBBD_'
    variant to handle the common CFBD/CBBD spelling mix-up.
    """
    try:
        return st.secrets[section][key]
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

