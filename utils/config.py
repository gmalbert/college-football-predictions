"""utils/config.py — Secrets and configuration helpers."""
from __future__ import annotations
import os
import streamlit as st


def get_secret(section: str, key: str) -> str:
    """
    Fetch a secret from Streamlit secrets (Cloud) or environment variables
    (local fallback).
    """
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        env_key = f"{section.upper()}_{key.upper()}"
        value = os.environ.get(env_key)
        if value is None:
            raise ValueError(
                f"Secret '{section}.{key}' not found in Streamlit secrets "
                f"or environment variable '{env_key}'."
            )
        return value
