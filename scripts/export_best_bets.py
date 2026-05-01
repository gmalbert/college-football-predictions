"""
scripts/export_best_bets.py — College Football (college-football-predictions)
Reads data_files/processed/games.parquet + lines.parquet, runs recommendation logic,
and writes data_files/best_bets_today.json.
LOOKAHEAD_DAYS = 6 because pipeline runs weekly on Tuesdays.
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPORT = "NCAAF"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = ROOT / "data_files" / "best_bets_today.json"
LOOKAHEAD_DAYS = 6  # Weekly pipeline — cover the upcoming weekend


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier_from_confidence(conf_label: str) -> str:
    label_map = {
        "Strong":   "Elite",
        "Moderate": "Strong",
        "Lean":     "Good",
        "Elite":    "Elite",
        "High":     "Elite",
        "Medium":   "Strong",
    }
    return label_map.get(str(conf_label), "Good")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = date.today()
    end_date = today + timedelta(days=LOOKAHEAD_DAYS)

    # NCAAF season: August–January
    month = today.month
    if not (8 <= month or month <= 1):
        _write([], "NCAAF off-season")
        return

    try:
        import pandas as pd
    except ImportError:
        _write([], "pandas not available")
        return

    games_path = ROOT / "data_files" / "processed" / "games.parquet"
    lines_path = ROOT / "data_files" / "processed" / "lines.parquet"
    picks_path = ROOT / "data_files" / "picks_today.json"

    # Prefer pre-computed picks_today.json if available
    if picks_path.exists():
        try:
            with open(picks_path) as f:
                raw = json.load(f)
            bets_raw = raw if isinstance(raw, list) else raw.get("bets", [])
            bets = []
            for b in bets_raw:
                game_date = str(b.get("game_date", ""))
                try:
                    gd = date.fromisoformat(game_date)
                except ValueError:
                    continue
                if not (today <= gd <= end_date):
                    continue
                tier_raw = str(b.get("tier", b.get("confidence", "Good")))
                bets.append({
                    "game_date": game_date,
                    "game_time": b.get("game_time"),
                    "game": b.get("game", ""),
                    "home_team": b.get("home_team", ""),
                    "away_team": b.get("away_team", ""),
                    "bet_type": b.get("bet_type", "Spread"),
                    "pick": b.get("pick", ""),
                    "confidence": _safe_float(b.get("confidence", b.get("win_prob"))),
                    "edge": _safe_float(b.get("edge")),
                    "tier": _tier_from_confidence(tier_raw),
                    "odds": b.get("odds"),
                    "line": _safe_float(b.get("line")),
                    "league": "NCAAF",
                })
            _write(bets, "" if bets else f"No NCAAF picks in next {LOOKAHEAD_DAYS} days")
            return
        except Exception:
            pass

    # Fall back to raw parquets
    if not games_path.exists() or not lines_path.exists():
        _write([], "No NCAAF parquet data found — run weekly pipeline first")
        return

    try:
        games = pd.read_parquet(games_path)
        lines = pd.read_parquet(lines_path)
    except Exception as e:
        _write([], f"Failed to read parquet: {e}")
        return

    # Filter to upcoming window
    date_col = next((c for c in ["start_date", "game_date", "date"] if c in games.columns), None)
    if not date_col:
        _write([], "No date column in games.parquet")
        return

    games[date_col] = pd.to_datetime(games[date_col], errors="coerce").dt.date
    upcoming = games[(games[date_col] >= today) & (games[date_col] <= end_date)]

    if upcoming.empty:
        _write([], f"No NCAAF games in next {LOOKAHEAD_DAYS} days")
        return

    # Try to join lines
    merge_cols = [c for c in ["game_id", "id"] if c in upcoming.columns and c in lines.columns]
    if merge_cols:
        try:
            upcoming = upcoming.merge(lines, on=merge_cols[:1], how="left", suffixes=("", "_line"))
        except Exception:
            pass

    bets = []
    for _, row in upcoming.iterrows():
        home = str(row.get("home_team", row.get("home", "")))
        away = str(row.get("away_team", row.get("away", "")))
        game = f"{away} @ {home}"
        game_date = str(row[date_col])

        conf = _safe_float(row.get("win_prob", row.get("home_win_prob")))
        edge = _safe_float(row.get("edge"))
        spread = _safe_float(row.get("spread", row.get("line")))

        if conf is None or conf < 0.52:
            continue
        if edge is not None and edge < 0:
            continue

        bet: dict = {
            "game_date": game_date,
            "game_time": str(row.get("start_time", "")) or None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": "Spread",
            "pick": home if (conf or 0) >= 0.5 else away,
            "confidence": round(conf, 4) if conf else None,
            "edge": round(edge, 4) if edge else None,
            "tier": "Strong" if (conf or 0) >= 0.65 else "Good",
            "odds": -110,  # standard spread odds
            "line": spread,
            "league": "NCAAF",
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying NCAAF picks in next {LOOKAHEAD_DAYS} days")


if __name__ == "__main__":
    main()
