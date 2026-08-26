"""College-football season calendar helpers."""
from __future__ import annotations

from datetime import date, datetime


def current_cfb_season(at: date | datetime | None = None) -> int:
    """Return the active season label, including Jan/Feb postseason games."""
    value = at or datetime.now()
    return value.year - 1 if value.month <= 2 else value.year


def rolling_season_window(
    *, completed_seasons: int = 5, at: date | datetime | None = None
) -> list[int]:
    if completed_seasons < 1:
        raise ValueError("completed_seasons must be positive")
    active = current_cfb_season(at)
    return list(range(active - completed_seasons, active + 1))
