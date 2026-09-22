"""JSON conversion helpers shared by every API service.

The Streamlit pages format almost every value into a display string before it
reaches the browser (``f"{wp:.0%}"``, ``f"{ms:+.1f}"`` …).  To guarantee that
the React build renders byte-identical text we format on the server the exact
same way and ship display-ready strings alongside the raw numbers that the
charts need.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "jsonable",
    "records",
    "dash",
    "fmt_signed1",
    "fmt_plain1",
    "fmt_pct",
    "fmt_pct1",
    "fmt_int",
    "fmt_american",
    "fmt_money",
]


def jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-serialisable ones."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def records(frame: pd.DataFrame | None) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe row dicts."""
    if frame is None or len(frame) == 0:
        return []
    return [
        {str(column): jsonable(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def dash(value: Any) -> str:
    """The Streamlit pages' em-dash placeholder for missing values."""
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return str(value)


def fmt_signed1(value: Any) -> str:
    """``+3.5`` / ``-7.0`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.1f}" if math.isfinite(number) else "—"


def fmt_plain1(value: Any) -> str:
    """``3.5`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1f}" if math.isfinite(number) else "—"


def fmt_pct(value: Any) -> str:
    """``62%`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.0%}" if math.isfinite(number) else "—"


def fmt_pct1(value: Any) -> str:
    """``62.4%`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1%}" if math.isfinite(number) else "—"


def fmt_int(value: Any) -> str:
    """``1,234`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{int(number):,}" if math.isfinite(number) else "—"


def fmt_american(value: Any) -> str:
    """``+150`` / ``-110`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{int(number):+d}" if math.isfinite(number) else "—"


def fmt_money(value: Any) -> str:
    """``$1,000`` / ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${number:,.0f}" if math.isfinite(number) else "—"
