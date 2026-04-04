"""utils/fetch_historical.py

Pull 5 years of CFBD historical data (2021-2025) with minimal API calls.
One call per endpoint per year; caches raw JSON so re-runs hit disk only.
Then processes raw JSON into Parquet tables for fast downstream access.

Usage:
    python -m utils.fetch_historical           # skip already-cached data
    python -m utils.fetch_historical --force   # re-pull everything
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from utils.cfbd_client import (
    get_games,
    get_game_team_stats,
    get_lines,
    get_advanced_stats,
    get_sp_ratings,
    get_elo_ratings,
    get_team_recruiting,
    get_teams,
    get_talent,
)
from utils.storage import RAW_DIR, PROCESSED_DIR, save_parquet
from utils.logger import get_logger

# 5 most-recent complete seasons (as of April 2026)
HISTORICAL_YEARS = list(range(2021, 2026))

# Polite delay between API calls to stay well within rate limits
API_DELAY = 0.75

logger = get_logger(__name__)


# ─────────────────────────────── helpers ──────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert cfbd SDK objects → plain JSON-serializable dicts."""
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if hasattr(obj, "to_dict"):
        return _to_serializable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _pull(name: str, fn, *args, force: bool = False, **kwargs) -> list:
    """Fetch data via *fn*, cache as JSON, return deserialized list."""
    path = RAW_DIR / f"{name}.json"
    if not force and path.exists():
        logger.info(f"  cached    — {name}")
        with open(path) as fh:
            return json.load(fh)
    logger.info(f"  pulling   — {name} …")
    data = fn(*args, **kwargs)
    serialized = _to_serializable(data)
    with open(path, "w") as fh:
        json.dump(serialized, fh, default=str)
    time.sleep(API_DELAY)
    return serialized


def _load_raw(name: str) -> list:
    path = RAW_DIR / f"{name}.json"
    if not path.exists():
        return []
    with open(path) as fh:
        return json.load(fh)


def _pull_team_game_stats(year: int, force: bool = False) -> list:
    """Fetch team game stats week-by-week (CFBD v5 requires week/team/conference)."""
    path = RAW_DIR / f"team_game_stats_{year}.json"
    if not force and path.exists():
        logger.info(f"  cached    — team_game_stats_{year}")
        with open(path) as fh:
            return json.load(fh)
    logger.info(f"  pulling   — team_game_stats_{year} (weeks 1-16) …")
    combined: list = []
    for week in range(1, 17):
        rows = get_game_team_stats(year, week=week)
        combined.extend(rows)
        time.sleep(API_DELAY)
    # Postseason (bowls / playoffs)
    ps = get_game_team_stats(year, week=1, season_type="postseason")
    combined.extend(ps)
    time.sleep(API_DELAY)
    serialized = _to_serializable(combined)
    with open(path, "w") as fh:
        json.dump(serialized, fh, default=str)
    return serialized


# ─────────────────────────────── ingestion ────────────────────────────────────

def run(force: bool = False) -> None:
    """Pull all data types for HISTORICAL_YEARS and build processed tables."""
    logger.info("=== CFBD historical pull — 5 seasons (2021‑2025) ===")
    logger.info(f"    API delay: {API_DELAY}s between calls")

    # Teams — single call for all FBS teams
    _pull("teams", get_teams, force=force)

    for year in HISTORICAL_YEARS:
        logger.info(f"--- {year} ---")
        # 2 season-type calls per year (regular + bowls/playoffs)
        _pull(f"games_{year}_regular",    get_games, year, "regular",    force=force)
        _pull(f"games_{year}_postseason", get_games, year, "postseason",  force=force)
        # team game stats: iterate week-by-week (API requires week/team/conference)
        _pull_team_game_stats(year, force=force)
        _pull(f"lines_{year}",            get_lines, year,                force=force)
        _pull(f"advanced_stats_{year}",   get_advanced_stats, year,       force=force)
        _pull(f"sp_ratings_{year}",       get_sp_ratings, year,           force=force)
        _pull(f"elo_ratings_{year}",      get_elo_ratings, year,          force=force)
        _pull(f"recruiting_{year}",       get_team_recruiting, year,      force=force)
        _pull(f"talent_{year}",           get_talent, year,               force=force)

    logger.info("=== Building processed Parquet tables ===")
    build_processed_tables(force=force)
    logger.info("=== Done ===")


# ─────────────────────────────── processing ───────────────────────────────────

def build_processed_tables(force: bool = False) -> None:
    """Convert raw JSON caches → analysis-ready Parquet files."""
    _build_games(force)
    _build_lines(force)
    _build_ratings(force)
    _build_advanced_stats(force)
    _build_team_game_stats(force)
    _build_recruiting(force)
    _build_elo(force)


def _build_games(force: bool) -> None:
    dst = PROCESSED_DIR / "games.parquet"
    if not force and dst.exists():
        logger.info("  processed/games.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for stype in ("regular", "postseason"):
            for g in _load_raw(f"games_{year}_{stype}"):
                rows.append({
                    "game_id":         g.get("id"),
                    "season":          g.get("season"),
                    "week":            g.get("week"),
                    "season_type":     stype,
                    "home_team":       g.get("homeTeam"),
                    "away_team":       g.get("awayTeam"),
                    "home_score":      g.get("homePoints"),
                    "away_score":      g.get("awayPoints"),
                    "neutral_site":    g.get("neutralSite", False),
                    "conference_game": g.get("conferenceGame", False),
                    "home_conference": g.get("homeConference"),
                    "away_conference": g.get("awayConference"),
                    "start_date":      g.get("startDate"),
                    "venue":           g.get("venue"),
                })
    if not rows:
        logger.warning("  No game data found — skipping games.parquet")
        return
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_win"]     = (df["home_score"] > df["away_score"]).astype(int)
    df["home_margin"]  = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["total_points"] = df["home_score"].astype(float) + df["away_score"].astype(float)
    save_parquet(df, "games")
    logger.info(f"  games.parquet: {len(df):,} rows")


def _build_lines(force: bool) -> None:
    dst = PROCESSED_DIR / "lines.parquet"
    if not force and dst.exists():
        logger.info("  processed/lines.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for g in _load_raw(f"lines_{year}"):
            gid = g.get("id")
            for ln in g.get("lines") or []:
                rows.append({
                    "game_id":        gid,
                    "provider":       ln.get("provider"),
                    "spread":         ln.get("spread"),
                    # cfbd v5 uses camelCase in the provider line sub-object
                    "over_under":     ln.get("overUnder"),
                    "home_moneyline": ln.get("homeMoneyline"),
                    "away_moneyline": ln.get("awayMoneyline"),
                })
    if not rows:
        logger.warning("  No lines data — skipping lines.parquet")
        return
    df = pd.DataFrame(rows)
    df["spread"]     = pd.to_numeric(df["spread"],     errors="coerce")
    df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")
    # Average across providers for a single market line per game
    market = (
        df.groupby("game_id")[["spread", "over_under", "home_moneyline", "away_moneyline"]]
        .mean()
        .reset_index()
        .rename(columns={"spread": "market_spread", "over_under": "market_total"})
    )
    save_parquet(market, "lines")
    logger.info(f"  lines.parquet: {len(market):,} rows")


def _build_ratings(force: bool) -> None:
    dst = PROCESSED_DIR / "ratings.parquet"
    if not force and dst.exists():
        logger.info("  processed/ratings.parquet — already exists")
        return
    sp_rows: list[dict] = []
    talent_by: dict[tuple, float] = {}

    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"sp_ratings_{year}"):
            off = r.get("offense") or {}
            dfe = r.get("defense") or {}
            # SP+ uses 'year' not 'season'
            sp_rows.append({
                "season":     r.get("year"),
                "team":       r.get("team"),
                "sp_overall": r.get("rating"),
                "sp_offense": off.get("rating") if isinstance(off, dict) else None,
                "sp_defense": dfe.get("rating") if isinstance(dfe, dict) else None,
            })
        for t in _load_raw(f"talent_{year}"):
            # talent uses 'year', 'team', 'talent'
            talent_by[(t.get("year"), t.get("team"))] = t.get("talent")

    if not sp_rows:
        logger.warning("  No SP+ data — skipping ratings.parquet")
        return
    df = pd.DataFrame(sp_rows)
    df["talent"] = df.apply(
        lambda r: talent_by.get((r["season"], r["team"])), axis=1
    )
    df["talent"] = pd.to_numeric(df["talent"], errors="coerce")
    save_parquet(df, "ratings")
    logger.info(f"  ratings.parquet: {len(df):,} rows")


def _build_advanced_stats(force: bool) -> None:
    dst = PROCESSED_DIR / "advanced_stats.parquet"
    if not force and dst.exists():
        logger.info("  processed/advanced_stats.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"advanced_stats_{year}"):
            off = r.get("offense") or {}
            dfe = r.get("defense") or {}
            off_havoc = off.get("havoc") or {}
            def_havoc = dfe.get("havoc") or {}
            rows.append({
                "season":            r.get("season"),
                "team":              r.get("team"),
                "conference":        r.get("conference"),
                # Top-level efficiency (cfbd v5 stores camelCase in raw JSON)
                "off_epa":           off.get("ppa"),
                "off_success_rate":  off.get("successRate"),
                "off_explosiveness": off.get("explosiveness"),
                "off_power_success": off.get("powerSuccess"),
                "off_stuff_rate":    off.get("stuffRate"),
                "off_line_yards":    off.get("lineYards"),
                "off_scoring_ops":   off.get("scoringOpportunities"),
                "off_pts_per_opp":   off.get("pointsPerOpportunity"),
                "def_epa":           dfe.get("ppa"),
                "def_success_rate":  dfe.get("successRate"),
                "def_explosiveness": dfe.get("explosiveness"),
                "def_power_success": dfe.get("powerSuccess"),
                "def_stuff_rate":    dfe.get("stuffRate"),
                "def_line_yards":    dfe.get("lineYards"),
                # Rushing / passing EPA breakdown
                "off_rushing_epa":  (off.get("rushingPlays") or {}).get("ppa"),
                "off_passing_epa":  (off.get("passingPlays") or {}).get("ppa"),
                "off_rushing_sr":   (off.get("rushingPlays") or {}).get("successRate"),
                "off_passing_sr":   (off.get("passingPlays") or {}).get("successRate"),
                "def_rushing_epa":  (dfe.get("rushingPlays") or {}).get("ppa"),
                "def_passing_epa":  (dfe.get("passingPlays") or {}).get("ppa"),
                # Havoc breakdown
                "def_havoc":        def_havoc.get("total")     if isinstance(def_havoc, dict) else None,
                "def_havoc_front":  def_havoc.get("frontSeven") if isinstance(def_havoc, dict) else None,
                "def_havoc_db":     def_havoc.get("db")        if isinstance(def_havoc, dict) else None,
                "off_havoc":        off_havoc.get("total")     if isinstance(off_havoc, dict) else None,
            })
    if not rows:
        logger.warning("  No advanced stats — skipping advanced_stats.parquet")
        return
    df = pd.DataFrame(rows)
    for col in df.columns.difference(["team", "conference"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "advanced_stats")
    logger.info(f"  advanced_stats.parquet: {len(df):,} rows")


def _build_team_game_stats(force: bool) -> None:
    dst = PROCESSED_DIR / "team_game_stats.parquet"
    if not force and dst.exists():
        logger.info("  processed/team_game_stats.parquet — already exists")
        return
    STAT_MAP = {
        "rushingAttempts":    "rushing_attempts",
        "rushingYards":       "rushing_yards",
        "rushingTDs":         "rushing_tds",
        "passingCompletions": "pass_completions",
        "passingAttempts":    "pass_attempts",
        "netPassingYards":    "pass_yards",
        "passingTDs":         "pass_tds",
        "interceptions":      "interceptions",
        "totalYards":         "total_yards",
        "turnovers":          "turnovers",
        "fumblesLost":        "fumbles_lost",
        "penalties":          "penalties",
        "penaltyYards":       "penalty_yards",
        "kickReturns":        "kick_returns",
        "kickReturnYards":    "kick_return_yards",
        "puntReturns":        "punt_returns",
        "puntReturnYards":    "punt_return_yards",
        "thirdDownEff":       "third_down_eff",
        "fourthDownEff":      "fourth_down_eff",
        "redZoneAttempts":    "red_zone_attempts",
        "redZoneConversions": "red_zone_conversions",
        "possessionTime":     "possession_minutes",
        "sacks":              "sacks",
    }
    rows = []
    for year in HISTORICAL_YEARS:
        for game in _load_raw(f"team_game_stats_{year}"):
            game_id = game.get("id") or game.get("game_id")
            for td in game.get("teams") or []:
                team      = td.get("school") or td.get("team")
                home_away = td.get("home_away") or td.get("homeAway")
                row: dict = {
                    "game_id":   game_id,
                    "season":    year,
                    "team":      team,
                    "home_away": home_away,
                }
                for s in td.get("stats") or []:
                    cat  = s.get("category") or s.get("name") or ""
                    dest = STAT_MAP.get(cat)
                    if dest is None:
                        continue
                    val = s.get("stat")
                    if dest in ("third_down_eff", "fourth_down_eff"):
                        try:
                            made, att = str(val).split("-")
                            row[f"{dest}_made"] = int(made)
                            row[f"{dest}_att"]  = int(att)
                            row[dest] = int(made) / int(att) if int(att) else float("nan")
                        except (ValueError, AttributeError):
                            row[dest] = float("nan")
                    elif dest == "possession_minutes":
                        try:
                            mm, ss = str(val).split(":")
                            row[dest] = int(mm) + int(ss) / 60.0
                        except (ValueError, IndexError):
                            row[dest] = float("nan")
                    else:
                        try:
                            row[dest] = float(val) if val is not None else float("nan")
                        except (ValueError, TypeError):
                            row[dest] = float("nan")
                rows.append(row)
    if not rows:
        logger.warning("  No team game stats — skipping team_game_stats.parquet")
        return
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("game_id", "season", "team", "home_away")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "team_game_stats")
    logger.info(f"  team_game_stats.parquet: {len(df):,} rows")


def _build_recruiting(force: bool) -> None:
    dst = PROCESSED_DIR / "recruiting.parquet"
    if not force and dst.exists():
        logger.info("  processed/recruiting.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"recruiting_{year}"):
            # cfbd v5 uses 'year' field in recruiting
            rows.append({
                "season": r.get("year") or r.get("season"),
                "team":   r.get("team"),
                "rank":   r.get("rank"),
                "points": r.get("points"),
            })
    if not rows:
        logger.warning("  No recruiting data — skipping recruiting.parquet")
        return
    df = pd.DataFrame(rows)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    save_parquet(df, "recruiting")
    logger.info(f"  recruiting.parquet: {len(df):,} rows")


def _build_elo(force: bool) -> None:
    dst = PROCESSED_DIR / "elo_ratings.parquet"
    if not force and dst.exists():
        logger.info("  processed/elo_ratings.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"elo_ratings_{year}"):
            rows.append({
                # cfbd v5 uses 'year' field (no week — this is end-of-season Elo)
                "season": r.get("year") or r.get("season"),
                "team":   r.get("team"),
                "elo":    r.get("elo"),
            })
    if not rows:
        logger.warning("  No Elo data — skipping elo_ratings.parquet")
        return
    df = pd.DataFrame(rows)
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    save_parquet(df, "elo_ratings")
    logger.info(f"  elo_ratings.parquet: {len(df):,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull 5 years (2021-2025) of CFBD data with minimal API calls."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-pull and reprocess even if cached data exists."
    )
    args = parser.parse_args()
    run(force=args.force)
