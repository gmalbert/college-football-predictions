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
    get_venues,
    get_fpi_ratings,
    get_srs_ratings,
    get_pregame_win_prob,
    get_ppa_teams,
    get_wepa_team_season,
    get_returning_production,
    get_transfer_portal,
    get_game_media,
    get_coaches,
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

    # Venues — single call (static)
    _pull("venues", get_venues, force=force)

    # Coaches — single call per year
    for year in HISTORICAL_YEARS:
        _pull(f"coaches_{year}", get_coaches, year=year, force=force)

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
        # NEW: P0 — FPI, SRS, Returning production (weather via Open-Meteo below)
        _pull(f"fpi_ratings_{year}",      get_fpi_ratings, year,          force=force)
        _pull(f"srs_ratings_{year}",      get_srs_ratings, year,          force=force)
        _pull(f"returning_production_{year}", get_returning_production, year, force=force)
        # NEW: P1 — PPA by down, WEPA, pre-game WP
        _pull(f"ppa_teams_{year}",        get_ppa_teams, year,            force=force)
        _pull(f"wepa_{year}",             get_wepa_team_season, year,     force=force)
        _pull(f"pregame_wp_{year}",       get_pregame_win_prob, year,     force=force)
        # NEW: P2 — Transfer portal, game media
        _pull(f"transfer_portal_{year}",  get_transfer_portal, year,      force=force)
        _pull(f"game_media_{year}",       get_game_media, year,           force=force)

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
    # NEW tables
    _build_weather(force)
    _build_venues(force)
    _build_fpi(force)
    _build_srs(force)
    _build_returning_production(force)
    _build_ppa_teams(force)
    _build_wepa(force)
    _build_pregame_wp(force)
    _build_transfer_portal(force)
    _build_game_media(force)
    _build_coaches(force)


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


def _build_weather(force: bool) -> None:
    """Build weather.parquet from Open-Meteo Historical API (free, no key needed).

    For each outdoor game, fetches hourly weather at the venue location on the
    game date, then picks the slot closest to kickoff UTC time.
    Dome games are recorded with game_indoors=True and wind_speed=0.
    Responses are in-process cached by (date, lat, lon) to minimise API calls.
    """
    import urllib.parse
    import requests as _req

    dst = PROCESSED_DIR / "weather.parquet"
    if not force and dst.exists():
        logger.info("  processed/weather.parquet — already exists")
        return

    games_path  = PROCESSED_DIR / "games.parquet"
    venues_path = PROCESSED_DIR / "venues.parquet"
    if not games_path.exists() or not venues_path.exists():
        logger.warning("  games.parquet or venues.parquet not found — skipping weather")
        return

    games_df  = pd.read_parquet(games_path)
    venues_df = pd.read_parquet(venues_path)

    vdf = (
        venues_df[["name", "latitude", "longitude", "dome"]]
        .drop_duplicates(subset=["name"], keep="first")
    )
    merged = games_df.merge(vdf, left_on="venue", right_on="name", how="left")
    merged["_dt"] = pd.to_datetime(merged["start_date"], utc=True, errors="coerce")

    # In-process cache: (date_str, lat_rounded, lon_rounded) -> hourly dict
    _cache: dict = {}

    def _fetch(lat: float, lon: float, date_str: str) -> dict:
        key = (date_str, round(lat, 2), round(lon, 2))
        if key in _cache:
            return _cache[key]
        params = {
            "latitude":         round(lat, 4),
            "longitude":        round(lon, 4),
            "start_date":       date_str,
            "end_date":         date_str,
            "hourly":           "temperature_2m,wind_speed_10m,precipitation,weathercode,relativehumidity_2m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":         "UTC",
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
        try:
            resp = _req.get(url, timeout=20)
            resp.raise_for_status()
            _cache[key] = resp.json().get("hourly", {})
        except Exception as exc:
            logger.debug(f"  Open-Meteo: {date_str} lat={lat:.2f} lon={lon:.2f} — {exc}")
            _cache[key] = {}
        time.sleep(0.05)
        return _cache[key]

    rows = []
    total = len(merged)
    for i, row in merged.iterrows():
        game_id = row.get("game_id")
        season  = row.get("season")
        week    = row.get("week")
        venue   = row.get("venue")
        is_dome = bool(row.get("dome") or False)

        if is_dome:
            rows.append({
                "game_id": game_id, "season": season, "week": week,
                "venue": venue, "game_indoors": True,
                "temperature": None, "wind_speed": 0.0,
                "precipitation": 0.0, "humidity": None, "weather_condition": None,
            })
            continue

        lat = row.get("latitude")
        lon = row.get("longitude")
        dt  = row.get("_dt")

        if pd.isna(lat) or pd.isna(lon) or pd.isna(dt):
            rows.append({
                "game_id": game_id, "season": season, "week": week,
                "venue": venue, "game_indoors": False,
                "temperature": None, "wind_speed": None,
                "precipitation": None, "humidity": None, "weather_condition": None,
            })
            continue

        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        hourly   = _fetch(float(lat), float(lon), date_str)

        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        winds  = hourly.get("wind_speed_10m", [])
        precip = hourly.get("precipitation", [])
        codes  = hourly.get("weathercode", [])
        humid  = hourly.get("relativehumidity_2m", [])

        # Closest hourly slot to kickoff UTC
        kickoff_str = pd.Timestamp(dt).strftime("%Y-%m-%dT%H:00")
        idx = times.index(kickoff_str) if kickoff_str in times else 0

        rows.append({
            "game_id":           game_id,
            "season":            season,
            "week":              week,
            "venue":             venue,
            "game_indoors":      False,
            "temperature":       temps[idx]  if idx < len(temps)  else None,
            "wind_speed":        winds[idx]  if idx < len(winds)  else None,
            "precipitation":     precip[idx] if idx < len(precip) else None,
            "humidity":          humid[idx]  if idx < len(humid)  else None,
            "weather_condition": codes[idx]  if idx < len(codes)  else None,
        })
        if (i + 1) % 200 == 0:
            logger.info(f"    weather: {i + 1}/{total} games processed ({len(_cache)} API calls made)")

    if not rows:
        logger.warning("  No weather rows built — skipping weather.parquet")
        return

    df = pd.DataFrame(rows)
    for col in ["temperature", "wind_speed", "precipitation", "humidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "weather")
    logger.info(f"  weather.parquet: {len(df):,} rows ({len(_cache)} Open-Meteo API calls)")


def _build_venues(force: bool) -> None:
    dst = PROCESSED_DIR / "venues.parquet"
    if not force and dst.exists():
        logger.info("  processed/venues.parquet — already exists")
        return
    rows = []
    for v in _load_raw("venues"):
        rows.append({
            "venue_id":  v.get("id"),
            "name":      v.get("name"),
            "city":      v.get("city"),
            "state":     v.get("state"),
            "capacity":  v.get("capacity"),
            "grass":     v.get("grass"),
            "dome":      v.get("dome"),
            "elevation": v.get("elevation"),
            "latitude":  v.get("location", {}).get("x") if isinstance(v.get("location"), dict) else v.get("latitude"),
            "longitude": v.get("location", {}).get("y") if isinstance(v.get("location"), dict) else v.get("longitude"),
            "timezone":  v.get("timezone"),
        })
    if not rows:
        logger.warning("  No venue data — skipping venues.parquet")
        return
    df = pd.DataFrame(rows)
    for col in ["capacity", "elevation", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "venues")
    logger.info(f"  venues.parquet: {len(df):,} rows")


def _build_fpi(force: bool) -> None:
    dst = PROCESSED_DIR / "fpi_ratings.parquet"
    if not force and dst.exists():
        logger.info("  processed/fpi_ratings.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"fpi_ratings_{year}"):
            eff = r.get("efficiencies") or {}
            rows.append({
                "season":      year,
                "team":        r.get("team"),
                "conference":  r.get("conference"),
                "fpi":         r.get("fpi"),
                "fpi_offense": eff.get("offense") if isinstance(eff, dict) else None,
                "fpi_defense": eff.get("defense") if isinstance(eff, dict) else None,
                "fpi_special_teams": eff.get("specialTeams") if isinstance(eff, dict) else None,
                "fpi_overall": eff.get("overall") if isinstance(eff, dict) else None,
            })
    if not rows:
        logger.warning("  No FPI data — skipping fpi_ratings.parquet")
        return
    df = pd.DataFrame(rows)
    for col in ["fpi", "fpi_offense", "fpi_defense", "fpi_special_teams", "fpi_overall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "fpi_ratings")
    logger.info(f"  fpi_ratings.parquet: {len(df):,} rows")


def _build_srs(force: bool) -> None:
    dst = PROCESSED_DIR / "srs_ratings.parquet"
    if not force and dst.exists():
        logger.info("  processed/srs_ratings.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"srs_ratings_{year}"):
            rows.append({
                "season":     year,
                "team":       r.get("team"),
                "conference": r.get("conference"),
                "division":   r.get("division"),
                "srs_rating": r.get("rating"),
                "srs_ranking": r.get("ranking"),
            })
    if not rows:
        logger.warning("  No SRS data — skipping srs_ratings.parquet")
        return
    df = pd.DataFrame(rows)
    df["srs_rating"] = pd.to_numeric(df["srs_rating"], errors="coerce")
    df["srs_ranking"] = pd.to_numeric(df["srs_ranking"], errors="coerce")
    save_parquet(df, "srs_ratings")
    logger.info(f"  srs_ratings.parquet: {len(df):,} rows")


def _build_returning_production(force: bool) -> None:
    dst = PROCESSED_DIR / "returning_production.parquet"
    if not force and dst.exists():
        logger.info("  processed/returning_production.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"returning_production_{year}"):
            rows.append({
                "season":               r.get("season") or year,
                "team":                 r.get("team"),
                "conference":           r.get("conference"),
                "total_ppa":            r.get("totalPPA") or r.get("total_ppa"),
                "total_passing_ppa":    r.get("totalPassingPPA") or r.get("total_passing_ppa"),
                "total_receiving_ppa":  r.get("totalReceivingPPA") or r.get("total_receiving_ppa"),
                "total_rushing_ppa":    r.get("totalRushingPPA") or r.get("total_rushing_ppa"),
                "percent_ppa":          r.get("percentPPA") or r.get("percent_ppa"),
                "percent_passing_ppa":  r.get("percentPassingPPA") or r.get("percent_passing_ppa"),
                "percent_receiving_ppa": r.get("percentReceivingPPA") or r.get("percent_receiving_ppa"),
                "percent_rushing_ppa":  r.get("percentRushingPPA") or r.get("percent_rushing_ppa"),
                "usage":                r.get("usage"),
                "passing_usage":        r.get("passingUsage") or r.get("passing_usage"),
                "receiving_usage":      r.get("receivingUsage") or r.get("receiving_usage"),
                "rushing_usage":        r.get("rushingUsage") or r.get("rushing_usage"),
            })
    if not rows:
        logger.warning("  No returning production data — skipping returning_production.parquet")
        return
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("season", "team", "conference")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "returning_production")
    logger.info(f"  returning_production.parquet: {len(df):,} rows")


def _build_ppa_teams(force: bool) -> None:
    dst = PROCESSED_DIR / "ppa_teams.parquet"
    if not force and dst.exists():
        logger.info("  processed/ppa_teams.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"ppa_teams_{year}"):
            off = r.get("offense") or {}
            dfe = r.get("defense") or {}
            rows.append({
                "season":            r.get("season") or year,
                "team":              r.get("team"),
                "conference":        r.get("conference"),
                "off_overall":       off.get("overall") if isinstance(off, dict) else r.get("off_overall"),
                "off_passing":       off.get("passing") if isinstance(off, dict) else r.get("off_passing"),
                "off_rushing":       off.get("rushing") if isinstance(off, dict) else r.get("off_rushing"),
                "off_first_down":    off.get("firstDown") if isinstance(off, dict) else r.get("off_first_down"),
                "off_second_down":   off.get("secondDown") if isinstance(off, dict) else r.get("off_second_down"),
                "off_third_down":    off.get("thirdDown") if isinstance(off, dict) else r.get("off_third_down"),
                "def_overall":       dfe.get("overall") if isinstance(dfe, dict) else r.get("def_overall"),
                "def_passing":       dfe.get("passing") if isinstance(dfe, dict) else r.get("def_passing"),
                "def_rushing":       dfe.get("rushing") if isinstance(dfe, dict) else r.get("def_rushing"),
                "def_first_down":    dfe.get("firstDown") if isinstance(dfe, dict) else r.get("def_first_down"),
                "def_second_down":   dfe.get("secondDown") if isinstance(dfe, dict) else r.get("def_second_down"),
                "def_third_down":    dfe.get("thirdDown") if isinstance(dfe, dict) else r.get("def_third_down"),
            })
    if not rows:
        logger.warning("  No PPA team data — skipping ppa_teams.parquet")
        return
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("season", "team", "conference")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "ppa_teams")
    logger.info(f"  ppa_teams.parquet: {len(df):,} rows")


def _build_wepa(force: bool) -> None:
    dst = PROCESSED_DIR / "wepa.parquet"
    if not force and dst.exists():
        logger.info("  processed/wepa.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"wepa_{year}"):
            off = r.get("offense") or {}
            dfe = r.get("defense") or {}
            rows.append({
                "season":        r.get("season") or year,
                "team":          r.get("team"),
                "conference":    r.get("conference"),
                "wepa_off_ppa":  off.get("ppa") if isinstance(off, dict) else None,
                "wepa_def_ppa":  dfe.get("ppa") if isinstance(dfe, dict) else None,
                "wepa_off_success": off.get("successRate") if isinstance(off, dict) else None,
                "wepa_def_success": dfe.get("successRate") if isinstance(dfe, dict) else None,
                "wepa_off_explosiveness": off.get("explosiveness") if isinstance(off, dict) else None,
                "wepa_def_explosiveness": dfe.get("explosiveness") if isinstance(dfe, dict) else None,
            })
    if not rows:
        logger.warning("  No WEPA data — skipping wepa.parquet")
        return
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("season", "team", "conference")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "wepa")
    logger.info(f"  wepa.parquet: {len(df):,} rows")


def _build_pregame_wp(force: bool) -> None:
    dst = PROCESSED_DIR / "pregame_wp.parquet"
    if not force and dst.exists():
        logger.info("  processed/pregame_wp.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"pregame_wp_{year}"):
            rows.append({
                "game_id":        r.get("gameId") or r.get("game_id"),
                "season":         r.get("season") or year,
                "week":           r.get("week"),
                "home_team":      r.get("homeTeam") or r.get("home_team"),
                "away_team":      r.get("awayTeam") or r.get("away_team"),
                "spread":         r.get("spread"),
                "home_win_prob":  r.get("homeWinProb") or r.get("home_win_prob"),
                "away_win_prob":  r.get("awayWinProb") or r.get("away_win_prob"),
            })
    if not rows:
        logger.warning("  No pregame WP data — skipping pregame_wp.parquet")
        return
    df = pd.DataFrame(rows)
    for col in ["spread", "home_win_prob", "away_win_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    save_parquet(df, "pregame_wp")
    logger.info(f"  pregame_wp.parquet: {len(df):,} rows")


def _build_transfer_portal(force: bool) -> None:
    dst = PROCESSED_DIR / "transfer_portal.parquet"
    if not force and dst.exists():
        logger.info("  processed/transfer_portal.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"transfer_portal_{year}"):
            rows.append({
                "season":       r.get("season") or year,
                "first_name":   r.get("firstName") or r.get("first_name"),
                "last_name":    r.get("lastName") or r.get("last_name"),
                "position":     r.get("position"),
                "origin":       r.get("origin"),
                "destination":  r.get("destination"),
                "transfer_date": r.get("transferDate") or r.get("transfer_date"),
                "rating":       r.get("rating"),
                "stars":        r.get("stars"),
                "eligibility":  r.get("eligibility"),
            })
    if not rows:
        logger.warning("  No transfer portal data — skipping transfer_portal.parquet")
        return
    df = pd.DataFrame(rows)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["stars"]  = pd.to_numeric(df["stars"],  errors="coerce")
    save_parquet(df, "transfer_portal")
    logger.info(f"  transfer_portal.parquet: {len(df):,} rows")


def _build_game_media(force: bool) -> None:
    dst = PROCESSED_DIR / "game_media.parquet"
    if not force and dst.exists():
        logger.info("  processed/game_media.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for r in _load_raw(f"game_media_{year}"):
            rows.append({
                "game_id":   r.get("id") or r.get("game_id"),
                "season":    r.get("season") or year,
                "week":      r.get("week"),
                "home_team": r.get("homeTeam") or r.get("home_team"),
                "away_team": r.get("awayTeam") or r.get("away_team"),
                "tv":        r.get("outlet") or r.get("tv"),
                "media_type": r.get("mediaType") or r.get("media_type"),
            })
    if not rows:
        logger.warning("  No game media data — skipping game_media.parquet")
        return
    df = pd.DataFrame(rows)
    # Aggregate to one row per game (keep first TV network listed)
    df = df.drop_duplicates(subset=["game_id"], keep="first")
    save_parquet(df, "game_media")
    logger.info(f"  game_media.parquet: {len(df):,} rows")


def _build_coaches(force: bool) -> None:
    dst = PROCESSED_DIR / "coaches.parquet"
    if not force and dst.exists():
        logger.info("  processed/coaches.parquet — already exists")
        return
    rows = []
    for year in HISTORICAL_YEARS:
        for c in _load_raw(f"coaches_{year}"):
            seasons = c.get("seasons") or []
            # Find the entry for this year
            yr_data = next((s for s in seasons if s.get("year") == year), None)
            rows.append({
                "season":          year,
                "first_name":      c.get("firstName") or c.get("first_name"),
                "last_name":       c.get("lastName") or c.get("last_name"),
                "team":            yr_data.get("school") if yr_data else c.get("team"),
                "wins":            yr_data.get("wins") if yr_data else None,
                "losses":          yr_data.get("losses") if yr_data else None,
                "tenure_years":    len([s for s in seasons if (s.get("year") or 0) <= year]),
            })
    if not rows:
        logger.warning("  No coach data — skipping coaches.parquet")
        return
    df = pd.DataFrame(rows)
    # Keep one head coach per team/season (there may be interim coaches)
    df = df.dropna(subset=["team"])
    df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce").fillna(1)
    # Keep the row with max tenure (most likely the actual head coach, not interim)
    df = df.sort_values("tenure_years", ascending=False).drop_duplicates(
        subset=["season", "team"], keep="first"
    )
    save_parquet(df, "coaches")
    logger.info(f"  coaches.parquet: {len(df):,} rows")


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
