"""utils/config.py — Secrets and configuration helpers."""
from __future__ import annotations
import os
import streamlit as st


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

