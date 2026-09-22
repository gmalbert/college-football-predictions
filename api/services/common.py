"""Helpers shared across page services."""
from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Mirrors ``utils.ui_components._BROWSER_TIMEZONE_ABBREVIATIONS``.
BROWSER_TIMEZONE_ABBREVIATIONS = {
    "America/New_York": "ET",
    "America/Chicago": "CT",
    "America/Denver": "MT",
    "America/Los_Angeles": "PT",
    "America/Anchorage": "AKT",
    "Pacific/Honolulu": "HT",
}


def resolve_timezone(timezone_name: str | None) -> tuple[ZoneInfo, str]:
    """Return ``(tzinfo, short_label)`` for a browser-supplied IANA name.

    Streamlit reads the browser timezone from ``st.context.timezone`` and falls
    back to UTC for headless runs.  The React client sends the same IANA name so
    kickoff columns render identically.
    """
    if not timezone_name:
        return ZoneInfo("UTC"), "UTC"
    try:
        return ZoneInfo(timezone_name), BROWSER_TIMEZONE_ABBREVIATIONS.get(
            timezone_name, timezone_name.rsplit("/", 1)[-1].replace("_", " ")
        )
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC"), "UTC"
