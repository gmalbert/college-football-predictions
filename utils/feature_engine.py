"""utils/feature_engine.py

Build the game-level feature matrix by joining all processed Parquet tables.
Saves to data_files/features/feature_matrix.parquet.

Call build_feature_matrix() after fetch_historical.run() completes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.storage import FEATURES_DIR, load_parquet, save_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────── feature column lists ─────────────────────────

WIN_FEATURES = [
    # Ratings & power
    "elo_diff",
    "sp_plus_diff",
    "sp_offense_diff",
    "sp_defense_diff",
    # Season-level efficiency diffs
    "off_epa_diff",
    "def_epa_diff",
    "off_explosiveness_diff",
    "def_havoc_diff",
    "off_success_diff",
    # Rushing / passing diffs
    "off_rushing_epa_diff",
    "off_passing_epa_diff",
    # Recruiting & talent
    "recruiting_diff",
    "talent_diff",
    "recruiting_rank_diff",
    # Schedule & context
    "home_flag",
    "conference_game",
    "rest_advantage",
    # Rolling 5-game stats (populated after team_game_stats.parquet is built)
    "turnover_margin_l5",
    "rushing_yards_diff_l5",
    "pass_yards_diff_l5",
    "penalty_yards_diff_l5",
    # NEW: Power ratings
    "fpi_diff",
    "srs_diff",
    # NEW: Returning production
    "returning_ppa_diff",
    # NEW: PPA by down
    "ppa_off_diff",
    "ppa_def_diff",
    "ppa_third_down_off_diff",
    "ppa_third_down_def_diff",
    # NEW: WEPA (opponent-adjusted EPA)
    "wepa_off_diff",
    "wepa_def_diff",
    # NEW: Pre-game WP consensus
    "cfbd_pregame_wp_diff",
    # NEW: Coaching tenure
    "coach_tenure_diff",
    # P3: Drive-level efficiency
    "scoring_drive_pct_diff",
    "three_and_out_pct_diff",
    # P3: Play-by-play tendency
    "off_pass_rate_diff",
    # P3: Player usage
    "top_rb_usage_diff",
]

SPREAD_FEATURES = WIN_FEATURES + ["market_spread"]

TOTAL_FEATURES = [
    "home_off_epa",
    "away_off_epa",
    "home_def_epa",
    "away_def_epa",
    "home_off_explosiveness",
    "away_off_explosiveness",
    "home_off_rushing_epa",
    "away_off_rushing_epa",
    "home_off_passing_epa",
    "away_off_passing_epa",
    "home_flag",
    "rest_days_home",
    "rest_days_away",
    "market_total",
    # NEW: Weather features (strong total predictors)
    "is_dome",
    "temperature",
    "wind_speed",
    "adverse_weather",
    "high_wind",
    # NEW: Venue altitude
    "high_altitude",
    # NEW: Prime-time indicator
    "is_primetime",
]

ALL_FEATURE_COLS = list(
    dict.fromkeys(WIN_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES)
)

# ───────────────────────────────── helpers ─────────────────────────────────

def _rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest for home and away teams from start_date."""
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.sort_values(["season", "week"])
    for side in ("home", "away"):
        prev = df.groupby(f"{side}_team")["_dt"].shift(1)
        df[f"rest_days_{side}"] = (df["_dt"] - prev).dt.days
    df["rest_advantage"] = df["rest_days_home"] - df["rest_days_away"]
    return df.drop(columns=["_dt"])


def _rolling_team_stats(
    games: pd.DataFrame,
    tgs: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute per-team rolling-window stat averages using pre-game history only.
    Returns (team, season, week, {stat}_l{window}) with no data leakage —
    the current game is excluded via .shift(1).
    """
    gw = games[["game_id", "week"]].drop_duplicates()
    tgs = tgs.merge(gw, on="game_id", how="left").dropna(subset=["week"])
    stat_cols = [c for c in tgs.columns
                 if c not in ("game_id", "season", "week", "team", "home_away")]
    parts: list[pd.DataFrame] = []
    for (team, season), grp in tgs.groupby(["team", "season"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        rolled = (
            grp[stat_cols]
            .apply(pd.to_numeric, errors="coerce")
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )
        rolled.columns = [f"{c}_l{window}" for c in stat_cols]
        rolled["team"]   = team
        rolled["season"] = int(season)
        rolled["week"]   = grp["week"].values
        parts.append(rolled)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ─────────────────────────────── main builder ────────────────────────────────

def _add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data and create weather-related features."""
    wdf = weather_df.copy()
    wdf["is_dome"] = wdf["game_indoors"].fillna(False).astype(int)
    wdf["adverse_weather"] = (
        (wdf["wind_speed"].fillna(0) > 15) |
        (wdf["precipitation"].fillna(0) > 0) |
        (wdf["temperature"].fillna(60) < 35)
    ).astype(int)
    wdf["high_wind"] = (wdf["wind_speed"].fillna(0) > 20).astype(int)
    merge_cols = ["game_id", "is_dome", "temperature", "wind_speed",
                  "humidity", "precipitation", "adverse_weather", "high_wind"]
    merge_cols = [c for c in merge_cols if c in wdf.columns]
    return df.merge(wdf[merge_cols], on="game_id", how="left")


def _add_venue_features(df: pd.DataFrame, venues_df: pd.DataFrame,
                        games_df: pd.DataFrame) -> pd.DataFrame:
    """Add venue-based features (elevation, dome, capacity) via game venue name."""
    if venues_df.empty:
        return df
    vdf = venues_df.copy()
    vdf["elevation"] = pd.to_numeric(vdf["elevation"], errors="coerce").fillna(0)
    vdf["is_venue_dome"] = vdf["dome"].fillna(False).astype(int)
    vdf["is_grass"] = vdf["grass"].fillna(False).astype(int)
    vdf["high_altitude"] = (vdf["elevation"] > 5000).astype(int)
    # Join via venue name from games
    if "venue" not in df.columns:
        return df
    vdf_unique = vdf.drop_duplicates(subset=["name"], keep="first")
    venue_map = vdf_unique.set_index("name")[["elevation", "is_venue_dome", "high_altitude", "is_grass"]]
    df = df.copy()
    df["elevation"]    = df["venue"].map(venue_map["elevation"])
    df["high_altitude"] = df["venue"].map(venue_map["high_altitude"]).fillna(0).astype(int)
    # Only override is_dome if weather didn't already set it
    if "is_dome" not in df.columns:
        df["is_dome"] = df["venue"].map(venue_map["is_venue_dome"]).fillna(0).astype(int)
    return df


def _add_fpi_features(df: pd.DataFrame, fpi_df: pd.DataFrame) -> pd.DataFrame:
    """Add FPI-based power rating features."""
    fpi = fpi_df[["season", "team", "fpi", "fpi_offense", "fpi_defense",
                   "fpi_special_teams"]].copy()
    home = fpi.rename(columns={c: f"home_{c}" for c in fpi.columns
                                if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = fpi.rename(columns={c: f"away_{c}" for c in fpi.columns
                                if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    df["fpi_diff"] = df["home_fpi"].fillna(0) - df["away_fpi"].fillna(0)
    return df


def _add_srs_features(df: pd.DataFrame, srs_df: pd.DataFrame) -> pd.DataFrame:
    """Add SRS rating diff as a feature."""
    srs = srs_df[["season", "team", "srs_rating"]].copy()
    home_srs = srs.rename(columns={"team": "home_team", "srs_rating": "home_srs"})
    df = df.merge(home_srs, on=["season", "home_team"], how="left")
    away_srs = srs.rename(columns={"team": "away_team", "srs_rating": "away_srs"})
    df = df.merge(away_srs, on=["season", "away_team"], how="left")
    df["srs_diff"] = df["home_srs"].fillna(0) - df["away_srs"].fillna(0)
    return df


def _add_pregame_wp_features(df: pd.DataFrame, wp_df: pd.DataFrame) -> pd.DataFrame:
    """Add CFBD pre-game win probability as a consensus model feature."""
    wp = wp_df[["game_id", "home_win_prob"]].dropna(subset=["game_id"]).copy()
    df = df.merge(wp, on="game_id", how="left")
    df["cfbd_pregame_wp_diff"] = df["home_win_prob"].fillna(0.5) - 0.5
    df = df.rename(columns={"home_win_prob": "cfbd_home_win_prob"})
    return df


def _add_ppa_features(df: pd.DataFrame, ppa_df: pd.DataFrame) -> pd.DataFrame:
    """Add PPA-based features broken down by down."""
    cols = ["season", "team", "off_overall", "off_passing", "off_rushing",
            "off_first_down", "off_second_down", "off_third_down",
            "def_overall", "def_passing", "def_rushing",
            "def_first_down", "def_second_down", "def_third_down"]
    avail = [c for c in cols if c in ppa_df.columns]
    ppa = ppa_df[avail].copy()

    home = ppa.rename(columns={c: f"home_ppa_{c}" for c in avail
                                if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = ppa.rename(columns={c: f"away_ppa_{c}" for c in avail
                                if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_ppa_off_overall" in df.columns and "away_ppa_off_overall" in df.columns:
        df["ppa_off_diff"] = df["home_ppa_off_overall"].fillna(0) - df["away_ppa_off_overall"].fillna(0)
    if "home_ppa_def_overall" in df.columns and "away_ppa_def_overall" in df.columns:
        df["ppa_def_diff"] = df["home_ppa_def_overall"].fillna(0) - df["away_ppa_def_overall"].fillna(0)
    if "home_ppa_off_third_down" in df.columns and "away_ppa_off_third_down" in df.columns:
        df["ppa_third_down_off_diff"] = (
            df["home_ppa_off_third_down"].fillna(0) - df["away_ppa_off_third_down"].fillna(0)
        )
    if "home_ppa_def_third_down" in df.columns and "away_ppa_def_third_down" in df.columns:
        df["ppa_third_down_def_diff"] = (
            df["home_ppa_def_third_down"].fillna(0) - df["away_ppa_def_third_down"].fillna(0)
        )
    return df


def _add_wepa_features(df: pd.DataFrame, wepa_df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent-adjusted WEPA features."""
    cols = ["season", "team", "wepa_off_ppa", "wepa_def_ppa",
            "wepa_off_success", "wepa_def_success"]
    avail = [c for c in cols if c in wepa_df.columns]
    wepa = wepa_df[avail].copy()

    home = wepa.rename(columns={c: f"home_{c}" for c in avail
                                 if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = wepa.rename(columns={c: f"away_{c}" for c in avail
                                 if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_wepa_off_ppa" in df.columns and "away_wepa_off_ppa" in df.columns:
        df["wepa_off_diff"] = df["home_wepa_off_ppa"].fillna(0) - df["away_wepa_off_ppa"].fillna(0)
    if "home_wepa_def_ppa" in df.columns and "away_wepa_def_ppa" in df.columns:
        df["wepa_def_diff"] = df["home_wepa_def_ppa"].fillna(0) - df["away_wepa_def_ppa"].fillna(0)
    return df


def _add_returning_production(df: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """Add returning production features."""
    cols = ["season", "team", "percent_ppa", "percent_passing_ppa",
            "percent_receiving_ppa", "percent_rushing_ppa"]
    avail = [c for c in cols if c in ret_df.columns]
    ret = ret_df[avail].copy()

    home = ret.rename(columns={"team": "home_team",
                                "percent_ppa": "home_ret_ppa_pct",
                                "percent_passing_ppa": "home_ret_pass_pct",
                                "percent_receiving_ppa": "home_ret_recv_pct",
                                "percent_rushing_ppa": "home_ret_rush_pct"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = ret.rename(columns={"team": "away_team",
                                "percent_ppa": "away_ret_ppa_pct",
                                "percent_passing_ppa": "away_ret_pass_pct",
                                "percent_receiving_ppa": "away_ret_recv_pct",
                                "percent_rushing_ppa": "away_ret_rush_pct"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_ret_ppa_pct" in df.columns and "away_ret_ppa_pct" in df.columns:
        df["returning_ppa_diff"] = (
            df["home_ret_ppa_pct"].fillna(0.5) - df["away_ret_ppa_pct"].fillna(0.5)
        )
    return df


def _add_coach_features(df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """Flag first-year coaches and add tenure features."""
    coaches = coaches_df[["season", "team", "tenure_years"]].copy()
    coaches["first_year_coach"] = (coaches["tenure_years"] == 1).astype(int)

    home_c = coaches.rename(columns={"team": "home_team",
                                      "first_year_coach": "home_first_yr_coach",
                                      "tenure_years": "home_coach_tenure"})
    df = df.merge(home_c, on=["season", "home_team"], how="left")

    away_c = coaches.rename(columns={"team": "away_team",
                                      "first_year_coach": "away_first_yr_coach",
                                      "tenure_years": "away_coach_tenure"})
    df = df.merge(away_c, on=["season", "away_team"], how="left")

    df["coach_tenure_diff"] = (
        df["home_coach_tenure"].fillna(3) - df["away_coach_tenure"].fillna(3)
    )
    return df


def _add_transfer_portal_features(df: pd.DataFrame, portal_df: pd.DataFrame) -> pd.DataFrame:
    """Add net transfer portal impact per team."""
    if portal_df.empty:
        return df
    portal = portal_df.copy()
    portal["rating"] = pd.to_numeric(portal.get("rating"), errors="coerce").fillna(0)

    gains = (portal.groupby(["season", "destination"])["rating"]
             .agg(portal_gains_sum="sum", portal_gains_count="count")
             .reset_index()
             .rename(columns={"destination": "team"}))

    losses = (portal.groupby(["season", "origin"])["rating"]
              .agg(portal_losses_sum="sum", portal_losses_count="count")
              .reset_index()
              .rename(columns={"origin": "team"}))

    merged = gains.merge(losses, on=["season", "team"], how="outer").fillna(0)
    merged["portal_net_rating"] = merged["portal_gains_sum"] - merged["portal_losses_sum"]

    home_p = merged.rename(columns={"team": "home_team",
                                     "portal_net_rating": "home_portal_net"})
    df = df.merge(home_p[["season", "home_team", "home_portal_net"]],
                  on=["season", "home_team"], how="left")

    away_p = merged.rename(columns={"team": "away_team",
                                     "portal_net_rating": "away_portal_net"})
    df = df.merge(away_p[["season", "away_team", "away_portal_net"]],
                  on=["season", "away_team"], how="left")

    df["portal_net_diff"] = df["home_portal_net"].fillna(0) - df["away_portal_net"].fillna(0)
    return df


def _add_media_features(df: pd.DataFrame, media_df: pd.DataFrame) -> pd.DataFrame:
    """Add TV/prime-time features."""
    if media_df.empty:
        return df
    mdf = media_df.copy()
    prime_networks = {"ESPN", "ABC", "FOX", "CBS", "NBC", "ESPN2"}
    mdf["is_primetime"] = mdf["tv"].apply(
        lambda x: int(any(net in str(x) for net in prime_networks)) if x else 0
    )
    return df.merge(mdf[["game_id", "is_primetime"]], on="game_id", how="left")


def _add_pbp_features(df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
    """Add play-by-play aggregated features (pass tendency, explosive rate)."""
    if plays_df.empty:
        return df
    cols = ["season", "team", "off_pass_rate", "off_explosive_rate_pbp", "off_rz_rate"]
    avail = [c for c in cols if c in plays_df.columns]
    plays = plays_df[avail].copy()

    home = plays.rename(columns={c: f"home_{c}" for c in avail if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = plays.rename(columns={c: f"away_{c}" for c in avail if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_off_pass_rate" in df.columns and "away_off_pass_rate" in df.columns:
        df["off_pass_rate_diff"] = (
            df["home_off_pass_rate"].fillna(0.5) - df["away_off_pass_rate"].fillna(0.5)
        )
    return df


def _add_drive_features(df: pd.DataFrame, drives_df: pd.DataFrame) -> pd.DataFrame:
    """Add drive-level efficiency features."""
    if drives_df.empty:
        return df
    cols = ["season", "team", "scoring_drive_pct", "three_and_out_pct",
            "off_avg_drive_yards", "off_turnover_drive_pct"]
    avail = [c for c in cols if c in drives_df.columns]
    drives = drives_df[avail].copy()

    home = drives.rename(columns={c: f"home_{c}" for c in avail if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = drives.rename(columns={c: f"away_{c}" for c in avail if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_scoring_drive_pct" in df.columns and "away_scoring_drive_pct" in df.columns:
        df["scoring_drive_pct_diff"] = (
            df["home_scoring_drive_pct"].fillna(0) - df["away_scoring_drive_pct"].fillna(0)
        )
    if "home_three_and_out_pct" in df.columns and "away_three_and_out_pct" in df.columns:
        # Away minus home: lower three-and-out = better offense, so positive = home advantage
        df["three_and_out_pct_diff"] = (
            df["away_three_and_out_pct"].fillna(0) - df["home_three_and_out_pct"].fillna(0)
        )
    return df


def _add_player_usage_features(df: pd.DataFrame, usage_df: pd.DataFrame) -> pd.DataFrame:
    """Add player usage concentration features."""
    if usage_df.empty:
        return df
    cols = ["season", "team", "top_rb_usage", "top_wr_usage", "top_skill_usage"]
    avail = [c for c in cols if c in usage_df.columns]
    usage = usage_df[avail].copy()

    home = usage.rename(columns={c: f"home_{c}" for c in avail if c not in ("season", "team")})
    home = home.rename(columns={"team": "home_team"})
    df = df.merge(home, on=["season", "home_team"], how="left")

    away = usage.rename(columns={c: f"away_{c}" for c in avail if c not in ("season", "team")})
    away = away.rename(columns={"team": "away_team"})
    df = df.merge(away, on=["season", "away_team"], how="left")

    if "home_top_rb_usage" in df.columns and "away_top_rb_usage" in df.columns:
        df["top_rb_usage_diff"] = (
            df["home_top_rb_usage"].fillna(0) - df["away_top_rb_usage"].fillna(0)
        )
    return df


def build_feature_matrix(force: bool = False) -> pd.DataFrame:
    """
    Join processed tables into a game-level feature matrix.
    Returns the DataFrame and writes it to features/feature_matrix.parquet.
    """
    dst = FEATURES_DIR / "feature_matrix.parquet"
    if not force and dst.exists():
        logger.info("feature_matrix.parquet found — loading cached version")
        return pd.read_parquet(dst)

    # ── load tables ─────────────────────────────────────────────────────────
    try:
        games      = load_parquet("games")
        lines      = load_parquet("lines")
        ratings    = load_parquet("ratings")
        adv        = load_parquet("advanced_stats")
        elo_raw    = load_parquet("elo_ratings")
        recruiting = load_parquet("recruiting")
    except FileNotFoundError as exc:
        logger.error(
            f"Missing processed data ({exc}). "
            "Run `python -m utils.fetch_historical` first."
        )
        return pd.DataFrame()

    # Optional: per-game team stats for rolling features
    try:
        tgs = load_parquet("team_game_stats")
    except FileNotFoundError:
        tgs = None

    # NEW: Optional new data tables
    def _try_load(name: str) -> pd.DataFrame:
        try:
            return load_parquet(name)
        except FileNotFoundError:
            return pd.DataFrame()

    weather_df   = _try_load("weather")
    venues_df    = _try_load("venues")
    fpi_df       = _try_load("fpi_ratings")
    srs_df       = _try_load("srs_ratings")
    pregame_wp_df = _try_load("pregame_wp")
    ppa_df       = _try_load("ppa_teams")
    wepa_df      = _try_load("wepa")
    ret_df       = _try_load("returning_production")
    coaches_df   = _try_load("coaches")
    portal_df    = _try_load("transfer_portal")
    media_df     = _try_load("game_media")
    plays_df     = _try_load("plays_agg")
    drives_df    = _try_load("drives_agg")
    usage_df     = _try_load("player_usage_agg")

    # ── Elo: season-level ratings (CFBD v5 provides end-of-season only) ──────
    elo = elo_raw[["season", "team", "elo"]].copy()

    # ── Recruiting: 3-year rolling talent average ────────────────────────────
    recruiting = recruiting.sort_values(["team", "season"])
    recruiting["points_3yr"] = (
        recruiting.groupby("team")["points"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    rec_cols = [c for c in ["season", "team", "points_3yr", "rank"] if c in recruiting.columns]
    rec = recruiting[rec_cols]

    # ── Start building from games ────────────────────────────────────────────
    df = games.copy()

    # Lines
    df = df.merge(lines, on="game_id", how="left")

    # SP+ / talent (home)
    sp_cols = [c for c in ["season", "team", "sp_overall", "sp_offense", "sp_defense", "talent"]
               if c in ratings.columns]
    rat = ratings[sp_cols]
    df = df.merge(
        rat.rename(columns={"team": "home_team", "sp_overall": "home_sp_plus",
                            "sp_offense": "home_sp_offense", "sp_defense": "home_sp_defense",
                            "talent": "home_talent"}),
        on=["season", "home_team"], how="left",
    )
    # SP+ / talent (away)
    df = df.merge(
        rat.rename(columns={"team": "away_team", "sp_overall": "away_sp_plus",
                            "sp_offense": "away_sp_offense", "sp_defense": "away_sp_defense",
                            "talent": "away_talent"}),
        on=["season", "away_team"], how="left",
    )

    # Advanced stats (home + away) — expand to include explosiveness, havoc, rushing/passing EPA
    adv_base = ["season", "team", "off_epa", "def_epa", "off_success_rate", "def_success_rate"]
    adv_ext  = ["off_explosiveness", "def_explosiveness", "def_havoc", "off_havoc",
                "off_rushing_epa", "off_passing_epa", "def_rushing_epa", "def_passing_epa"]
    adv_cols = [c for c in adv_base + adv_ext if c in adv.columns]
    adv_rename_home = {
        "team":              "home_team",
        "off_epa":           "home_off_epa",
        "def_epa":           "home_def_epa",
        "off_success_rate":  "home_off_success",
        "def_success_rate":  "home_def_success",
        "off_explosiveness": "home_off_explosiveness",
        "def_explosiveness": "home_def_explosiveness",
        "def_havoc":         "home_def_havoc",
        "off_havoc":         "home_off_havoc",
        "off_rushing_epa":   "home_off_rushing_epa",
        "off_passing_epa":   "home_off_passing_epa",
        "def_rushing_epa":   "home_def_rushing_epa",
        "def_passing_epa":   "home_def_passing_epa",
    }
    adv_rename_away = {
        "team":              "away_team",
        "off_epa":           "away_off_epa",
        "def_epa":           "away_def_epa",
        "off_success_rate":  "away_off_success",
        "def_success_rate":  "away_def_success",
        "off_explosiveness": "away_off_explosiveness",
        "def_explosiveness": "away_def_explosiveness",
        "def_havoc":         "away_def_havoc",
        "off_havoc":         "away_off_havoc",
        "off_rushing_epa":   "away_off_rushing_epa",
        "off_passing_epa":   "away_off_passing_epa",
        "def_rushing_epa":   "away_def_rushing_epa",
        "def_passing_epa":   "away_def_passing_epa",
    }
    df = df.merge(
        adv[adv_cols].rename(columns=adv_rename_home),
        on=["season", "home_team"], how="left",
    )
    # Advanced stats (away)
    df = df.merge(
        adv[adv_cols].rename(columns=adv_rename_away),
        on=["season", "away_team"], how="left",
    )

    # Elo (home)
    df = df.merge(
        elo.rename(columns={"team": "home_team", "elo": "home_elo"}),
        on=["season", "home_team"], how="left",
    )
    # Elo (away)
    df = df.merge(
        elo.rename(columns={"team": "away_team", "elo": "away_elo"}),
        on=["season", "away_team"], how="left",
    )

    # Recruiting (home)
    df = df.merge(
        rec.rename(columns={"team": "home_team", "points_3yr": "home_recruiting",
                            "rank": "home_recruiting_rank"}),
        on=["season", "home_team"], how="left",
    )
    # Recruiting (away)
    df = df.merge(
        rec.rename(columns={"team": "away_team", "points_3yr": "away_recruiting",
                            "rank": "away_recruiting_rank"}),
        on=["season", "away_team"], how="left",
    )

    # ── derived features ─────────────────────────────────────────────────────
    df["elo_diff"]               = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)
    df["sp_plus_diff"]           = df["home_sp_plus"]          - df["away_sp_plus"]
    df["sp_offense_diff"]        = df["home_sp_offense"]       - df["away_sp_offense"]
    # sp_defense: lower = better → away−home positive = home defense advantage
    df["sp_defense_diff"]        = df["away_sp_defense"]       - df["home_sp_defense"]
    df["off_epa_diff"]           = df["home_off_epa"]          - df["away_off_epa"]
    # def_epa: lower (more negative) = better; positive diff = home D advantage
    df["def_epa_diff"]           = df["away_def_epa"]          - df["home_def_epa"]
    df["off_explosiveness_diff"] = df["home_off_explosiveness"]- df["away_off_explosiveness"]
    df["def_havoc_diff"]         = df["home_def_havoc"]        - df["away_def_havoc"]
    df["off_success_diff"]       = df["home_off_success"]      - df["away_off_success"]
    df["off_rushing_epa_diff"]   = df["home_off_rushing_epa"]  - df["away_off_rushing_epa"]
    df["off_passing_epa_diff"]   = df["home_off_passing_epa"]  - df["away_off_passing_epa"]
    df["recruiting_diff"]        = df["home_recruiting"]       - df["away_recruiting"]
    df["talent_diff"]            = df["home_talent"]           - df["away_talent"]
    # recruiting rank: lower number = better → away_rank−home_rank positive = home ranked higher
    if "home_recruiting_rank" in df.columns and "away_recruiting_rank" in df.columns:
        df["recruiting_rank_diff"] = df["away_recruiting_rank"] - df["home_recruiting_rank"]

    # home_flag: 1 = home, 0.5 = neutral site
    df["home_flag"]       = np.where(df["neutral_site"].fillna(False), 0.5, 1.0)
    df["conference_game"] = df["conference_game"].fillna(False).astype(float)

    # ── Rest days ─────────────────────────────────────────────────────────────
    if "start_date" in df.columns:
        df = _rest_days(df)

    # ── Rolling team game stats (optional) ────────────────────────────────────
    if tgs is not None and not tgs.empty:
        rolling = _rolling_team_stats(df, tgs, window=5)
        if not rolling.empty:
            want = ["turnovers", "rushing_yards", "pass_yards", "penalty_yards"]
            keep = ["team", "season", "week"] + [
                f"{s}_l5" for s in want if f"{s}_l5" in rolling.columns
            ]
            rolling = rolling[[c for c in keep if c in rolling.columns]]
            home_r = rolling.rename(columns={"team": "home_team", **{
                f"{s}_l5": f"home_{s}_l5" for s in want}})
            away_r = rolling.rename(columns={"team": "away_team", **{
                f"{s}_l5": f"away_{s}_l5" for s in want}})
            df = df.merge(home_r, on=["home_team", "season", "week"], how="left")
            df = df.merge(away_r, on=["away_team", "season", "week"], how="left")
            if "home_turnovers_l5"     in df.columns and "away_turnovers_l5"     in df.columns:
                df["turnover_margin_l5"]    = df["away_turnovers_l5"]    - df["home_turnovers_l5"]
            if "home_rushing_yards_l5" in df.columns and "away_rushing_yards_l5" in df.columns:
                df["rushing_yards_diff_l5"] = df["home_rushing_yards_l5"] - df["away_rushing_yards_l5"]
            if "home_pass_yards_l5"    in df.columns and "away_pass_yards_l5"    in df.columns:
                df["pass_yards_diff_l5"]    = df["home_pass_yards_l5"]    - df["away_pass_yards_l5"]
            if "home_penalty_yards_l5" in df.columns and "away_penalty_yards_l5" in df.columns:
                df["penalty_yards_diff_l5"] = df["away_penalty_yards_l5"] - df["home_penalty_yards_l5"]

    # ── NEW: Weather ──────────────────────────────────────────────────────────
    if not weather_df.empty:
        df = _add_weather_features(df, weather_df)
        logger.info("  merged weather features")

    # ── NEW: Venue altitude / dome (fallback if weather didn't provide dome) ──
    if not venues_df.empty:
        df = _add_venue_features(df, venues_df, games)
        logger.info("  merged venue features")

    # Ensure is_dome exists (default 0 if neither weather nor venues provided it)
    if "is_dome" not in df.columns:
        df["is_dome"] = 0

    # ── NEW: FPI ratings ──────────────────────────────────────────────────────
    if not fpi_df.empty:
        df = _add_fpi_features(df, fpi_df)
        logger.info("  merged FPI features")

    # ── NEW: SRS ratings ──────────────────────────────────────────────────────
    if not srs_df.empty:
        df = _add_srs_features(df, srs_df)
        logger.info("  merged SRS features")

    # ── NEW: Pre-game win probability ──────────────────────────────────────────
    if not pregame_wp_df.empty:
        df = _add_pregame_wp_features(df, pregame_wp_df)
        logger.info("  merged pre-game WP features")

    # ── NEW: PPA by down ──────────────────────────────────────────────────────
    if not ppa_df.empty:
        df = _add_ppa_features(df, ppa_df)
        logger.info("  merged PPA by-down features")

    # ── NEW: WEPA (opponent-adjusted EPA) ─────────────────────────────────────
    if not wepa_df.empty:
        df = _add_wepa_features(df, wepa_df)
        logger.info("  merged WEPA features")

    # ── NEW: Returning production ─────────────────────────────────────────────
    if not ret_df.empty:
        df = _add_returning_production(df, ret_df)
        logger.info("  merged returning production features")

    # ── NEW: Coach tenure ─────────────────────────────────────────────────────
    if not coaches_df.empty:
        df = _add_coach_features(df, coaches_df)
        logger.info("  merged coach tenure features")

    # ── NEW: Transfer portal ──────────────────────────────────────────────────
    if not portal_df.empty:
        df = _add_transfer_portal_features(df, portal_df)
        logger.info("  merged transfer portal features")

    # ── NEW: Game media / prime-time ──────────────────────────────────────────
    if not media_df.empty:
        df = _add_media_features(df, media_df)
        logger.info("  merged game media features")
    # ── P3: Play-by-play features ──────────────────────────────────────────────
    if not plays_df.empty:
        df = _add_pbp_features(df, plays_df)
        logger.info("  merged play-by-play features")

    # ── P3: Drive-level features ──────────────────────────────────────────────────
    if not drives_df.empty:
        df = _add_drive_features(df, drives_df)
        logger.info("  merged drive-level features")

    # ── P3: Player usage features ──────────────────────────────────────────────
    if not usage_df.empty:
        df = _add_player_usage_features(df, usage_df)
        logger.info("  merged player usage features")
    # ── save ─────────────────────────────────────────────────────────────────
    save_parquet(df, "feature_matrix", layer="features")
    logger.info(
        f"feature_matrix.parquet: {len(df):,} rows × {len(df.columns)} columns"
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()
    build_feature_matrix(force=args.force)
