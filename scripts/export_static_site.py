"""Materialize the dashboard's page payloads for static hosting.

The React app currently requests these payloads from FastAPI.  This exporter
runs in the data workflow after artifacts are refreshed and writes the same
page-builder outputs under ``frontend/public/data/snapshots``.  Cloudflare Pages
copies this directory to ``/data`` during its normal Vite build.

The files are an intermediate static-data contract: page-specific selections
are added to the catalog as the frontend moves its filtering into the browser.
"""
from __future__ import annotations

import hashlib
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.jsonutil import jsonable  # noqa: E402
from api.main import LOGO_PATH, meta  # noqa: E402
from api.services import (  # noqa: E402
    data_quality,
    historical,
    home,
    model_performance,
    preseason,
    team_explorer,
    total_signals,
    value_bets,
    weekly,
    win_probability,
)

OUTPUT_DIR = ROOT / "frontend" / "public" / "data" / "snapshots"


def _pages() -> dict[str, Callable[[], dict]]:
    """Return the canonical initial payload for each React page."""
    return {
        "meta": meta,
        "home": home.build_home,
        "weekly-predictions": lambda: weekly.build_weekly(timezone_name="UTC"),
        "value-bets": lambda: value_bets.build_value_bets(timezone_name="UTC"),
        "team-explorer": team_explorer.build_team_explorer,
        "historical-analysis": historical.build_historical,
        "model-performance": model_performance.build_model_performance,
        "preseason-outlook": preseason.build_preseason,
        "data-quality": data_quality.build_data_quality,
        "total-market-signals": total_signals.build_total_signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("full", "market"),
        default="full",
        help="Full rebuild for the weekly pipeline, or refresh only market-sensitive pages.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    prior_manifest = {}
    if args.scope == "market":
        if not manifest_path.exists():
            raise RuntimeError(
                "No full static catalog exists yet. Run the weekly pipeline before "
                "the market-only static refresh."
            )
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        for folder in ("weekly", "teams", "preseason", "total-signals", "historical", "win-probability"):
            generated = OUTPUT_DIR / folder
            if generated.is_dir():
                for stale in generated.glob("*.json"):
                    stale.unlink()

    entries: dict[str, dict] = dict(prior_manifest.get("pages", {}))
    page_builders = _pages()
    if args.scope == "market":
        market_pages = {
            "meta", "home", "weekly-predictions", "value-bets",
            "data-quality", "total-market-signals",
        }
        page_builders = {
            name: build for name, build in page_builders.items() if name in market_pages
        }

    for name, build in page_builders.items():
        payload = build()
        if name == "meta" and LOGO_PATH.exists():
            # The API normally serves a resized logo from /api/logo. A static
            # Pages build needs a real file URL instead of that backend route.
            static_logo = OUTPUT_DIR.parent / "tailgate-logo.png"
            static_logo.write_bytes(LOGO_PATH.read_bytes())
            payload.setdefault("sidebar", {})["logo"] = "/data/tailgate-logo.png"
        elif name == "home" and LOGO_PATH.exists():
            # The home payload has its own logo field, separate from sidebar
            # metadata, and must also use the copied static asset on Pages.
            payload["logo"] = "/data/tailgate-logo.png"
        payload = jsonable(payload)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        path = OUTPUT_DIR / f"{name}.json"
        path.write_bytes(encoded + b"\n")
        entries[name] = {
            "url": f"/data/snapshots/{name}.json",
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        print(f"{name}: {len(encoded):,} bytes -> {path.relative_to(ROOT)}")

    if args.scope == "full":
        weekly_catalog = _export_weekly_catalog()
        team_catalog = _export_team_catalog()
        preseason_catalog = _export_preseason_catalog()
        total_signals_catalog = _export_total_signals_catalog()
        historical_catalog = _export_historical_catalog()
        value_bets_catalog = _export_value_bets_catalog()
        win_probability_catalog = _export_win_probability_catalog()
    else:
        weekly_catalog = _export_weekly_catalog()
        value_bets_catalog = _export_value_bets_catalog()
        total_signals_catalog = _export_total_signals_catalog()
        team_catalog = prior_manifest["team_catalog"]
        preseason_catalog = prior_manifest["preseason_catalog"]
        historical_catalog = prior_manifest["historical_catalog"]
        win_probability_catalog = prior_manifest["win_probability_catalog"]

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": _git_revision(),
        "pages": entries,
        "weekly_catalog": weekly_catalog,
        "team_catalog": team_catalog,
        "preseason_catalog": preseason_catalog,
        "total_signals_catalog": total_signals_catalog,
        "historical_catalog": historical_catalog,
        "value_bets_catalog": value_bets_catalog,
        "win_probability_catalog": win_probability_catalog,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {len(entries)} page snapshots")
    return 0


def _export_weekly_catalog() -> dict:
    """Write one unfiltered response per season/week for browser-side filtering."""
    frame = weekly.load_feature_matrix()
    if frame.empty:
        raise RuntimeError("Cannot create weekly catalog: feature matrix is empty.")

    options = weekly._options(frame)
    weeks_by_season: dict[str, list[int]] = {}
    output = OUTPUT_DIR / "weekly"
    output.mkdir(parents=True, exist_ok=True)

    for season in options["seasons"]:
        # The API exposes the latest season's week selector for every season,
        # including combinations that have no games. Preserve those empty
        # selections too so the static UI presents the same choices.
        weeks = list(options["weeks"])
        weeks_by_season[str(int(season))] = weeks
        for week_number in weeks:
            payload = jsonable(
                weekly.build_weekly(
                    season=int(season),
                    week=week_number,
                    conference="All",
                    min_edge=0,
                    sort_by="Edge (High→Low)",
                    timezone_name="UTC",
                    static_catalog=True,
                )
            )
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            path = output / f"{int(season)}-{week_number}.json"
            path.write_bytes(encoded + b"\n")
            print(f"weekly {season}/{week_number}: {len(encoded):,} bytes")

    return {
        "seasons": options["seasons"],
        "default_season": options["season"],
        "default_week": options["default_week"],
        "weeks_by_season": weeks_by_season,
        "snapshot_pattern": "/data/snapshots/weekly/{season}-{week}.json",
        "snapshot_count": sum(len(weeks) for weeks in weeks_by_season.values()),
    }


def _write_snapshot(relative_path: str, payload: dict) -> str:
    """Write one deterministic, compact JSON snapshot and return its URL."""
    path = OUTPUT_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    print(f"{relative_path}: {len(encoded):,} bytes")
    return f"/data/snapshots/{relative_path}"


def _export_team_catalog() -> dict:
    """Precompute every team/season choice exposed by Team Explorer."""
    initial = team_explorer.build_team_explorer()
    teams = initial.get("teams", [])
    seasons = initial.get("seasons", [])
    snapshots: dict[str, dict[str, str]] = {}
    for team in teams:
        team_id = hashlib.sha1(team.encode("utf-8")).hexdigest()[:12]
        snapshots[team] = {}
        for season in seasons:
            relative = f"teams/{team_id}-{int(season)}.json"
            payload = team_explorer.build_team_explorer(team=team, season=int(season))
            snapshots[team][str(int(season))] = _write_snapshot(relative, payload)
    return {
        "teams": teams,
        "seasons": seasons,
        "default_team": initial.get("selection", {}).get("team"),
        "default_season": initial.get("selection", {}).get("season"),
        "snapshots": snapshots,
    }


def _export_preseason_catalog() -> dict:
    """Precompute each season and conference selection on Preseason Outlook."""
    initial = preseason.build_preseason()
    seasons = initial.get("sidebar", {}).get("seasons", [])
    snapshots: dict[str, dict[str, str]] = {}
    for season in seasons:
        baseline = preseason.build_preseason(season=int(season), conference="All")
        efficiency = next(
            (tab for tab in baseline.get("tabs", []) if tab.get("id") == "efficiency"),
            {},
        )
        conferences = efficiency.get("conferences", ["All"])
        snapshots[str(int(season))] = {}
        for conference in conferences:
            relative = f"preseason/{int(season)}-{hashlib.sha1(conference.encode('utf-8')).hexdigest()[:10]}.json"
            payload = preseason.build_preseason(
                season=int(season), conference=str(conference)
            )
            snapshots[str(int(season))][str(conference)] = _write_snapshot(relative, payload)
    return {
        "seasons": seasons,
        "default_season": initial.get("sidebar", {}).get("season"),
        "snapshots": snapshots,
    }


def _export_total_signals_catalog() -> dict:
    """Precompute each held-out season selectable on Total Market Signals."""
    initial = total_signals.build_total_signals()
    seasons = initial.get("decisions", {}).get("seasons", [])
    snapshots = {}
    for season in seasons:
        relative = f"total-signals/{int(season)}.json"
        payload = total_signals.build_total_signals(season=int(season))
        snapshots[str(int(season))] = _write_snapshot(relative, payload)
    return {
        "seasons": seasons,
        "default_season": initial.get("decisions", {}).get("season"),
        "snapshots": snapshots,
    }


def _export_historical_catalog() -> dict:
    """Precompute all season-range charts and the source rows for H2H lookup."""
    initial = historical.build_historical()
    seasons = initial.get("seasons", [])
    teams = initial.get("teams", [])
    selection = initial.get("selection", {})
    default_team_a = selection.get("team_a")
    default_team_b = selection.get("team_b")
    snapshots: dict[str, dict[str, str]] = {}

    for season_from in seasons:
        snapshots[str(int(season_from))] = {}
        for season_to in seasons:
            relative = f"historical/{int(season_from)}-{int(season_to)}.json"
            payload = historical.build_historical(
                season_from=int(season_from),
                season_to=int(season_to),
                team_a=default_team_a,
                team_b=default_team_b,
            )
            snapshots[str(int(season_from))][str(int(season_to))] = _write_snapshot(
                relative, payload
            )

    source = historical.load_data()
    columns = [
        column
        for column in (
            "season", "week", "home_team", "away_team", "home_margin",
            "home_score", "away_score", "neutral_site", "market_spread",
        )
        if column in source.columns
    ]
    games_url = _write_snapshot(
        "historical/games.json",
        {"rows": source[columns].to_dict(orient="records")},
    )
    return {
        "seasons": seasons,
        "teams": teams,
        "default_season_from": selection.get("season_from"),
        "default_season_to": selection.get("season_to"),
        "default_team_a": default_team_a,
        "default_team_b": default_team_b,
        "snapshots": snapshots,
        "games_url": games_url,
    }


def _export_value_bets_catalog() -> dict:
    """Precompute each season/week's complete candidate set for browser filters."""
    initial = value_bets.build_value_bets(
        bet_type="All",
        min_edge=0.5,
        min_conf="All",
        sort_by="Edge (High→Low)",
        timezone_name="UTC",
        static_catalog=True,
    )
    controls = initial.get("controls", {})
    seasons = controls.get("seasons", [])
    snapshots: dict[str, dict[str, str]] = {}
    weeks_by_season: dict[str, list[int]] = {}
    default_week_by_season: dict[str, int] = {}

    for season in seasons:
        season_payload = value_bets.build_value_bets(
            season=int(season),
            bet_type="All",
            min_edge=0.5,
            min_conf="All",
            sort_by="Edge (High→Low)",
            timezone_name="UTC",
            static_catalog=True,
        )
        weeks = season_payload.get("controls", {}).get("weeks", [])
        weeks_by_season[str(int(season))] = [int(week) for week in weeks]
        default_week_by_season[str(int(season))] = int(
            season_payload.get("controls", {}).get("default_week", weeks[0] if weeks else 1)
        )
        snapshots[str(int(season))] = {}
        for week in weeks:
            payload = (
                season_payload
                if int(week) == season_payload.get("controls", {}).get("week")
                else value_bets.build_value_bets(
                    season=int(season),
                    week=int(week),
                    bet_type="All",
                    min_edge=0.5,
                    min_conf="All",
                    sort_by="Edge (High→Low)",
                    timezone_name="UTC",
                    static_catalog=True,
                )
            )
            relative = f"value-bets/{int(season)}-{int(week)}.json"
            snapshots[str(int(season))][str(int(week))] = _write_snapshot(
                relative, payload
            )

    return {
        "seasons": seasons,
        "default_season": controls.get("season"),
        "default_week": controls.get("week"),
        "weeks_by_season": weeks_by_season,
        "default_week_by_season": default_week_by_season,
        "snapshots": snapshots,
    }



def _export_local_win_probability_snapshots(
    season_options: list[int],
    games_frame,
) -> dict[tuple[int, int], str]:
    """Export cross-fitted in-house charts where cached PBP is available."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    from api.jsonutil import records
    from models.epa_model import compute_epa, train_ep_model
    from models.wpa_model import predict_wp, train_wp_model

    snapshots: dict[tuple[int, int], str] = {}
    # Keep workflow cost bounded while retaining the current and previous seasons.
    for season in season_options[:2]:
        raw_path = ROOT / "data_files" / "raw" / f"plays_{int(season)}.json"
        if not raw_path.exists() or raw_path.stat().st_size <= 2:
            continue

        season_games = games_frame[
            (pd.to_numeric(games_frame["season"], errors="coerce") == int(season))
            & games_frame["home_score"].notna()
            & games_frame["away_score"].notna()
        ].copy()
        finals = {}
        for row in season_games.itertuples(index=False):
            game_id = int(row.game_id)
            home_score = float(row.home_score)
            away_score = float(row.away_score)
            if home_score == away_score:
                continue
            finals[game_id] = {
                "home": str(row.home_team),
                "away": str(row.away_team),
                "winner": str(row.home_team if home_score > away_score else row.away_team),
                "home_score": int(home_score),
                "away_score": int(away_score),
            }
        if len(finals) < 5:
            continue

        raw_plays = json.loads(raw_path.read_text(encoding="utf-8"))
        def _number(value, default=0):
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        rows = []
        for item in raw_plays:
            try:
                game_id = int(item.get("gameId"))
            except (TypeError, ValueError):
                continue
            if game_id not in finals:
                continue
            clock = item.get("clock") or {}
            offense = str(item.get("offense") or "")
            home = str(item.get("home") or "")
            away = str(item.get("away") or "")
            if not home or not away or offense not in (home, away):
                continue

            scoring = item.get("scoring", False)
            if isinstance(scoring, str):
                scoring = scoring.strip().lower() in {"true", "1", "yes"}

            offense_score = _number(item.get("offenseScore"))
            defense_score = _number(item.get("defenseScore"))
            if offense == home:
                home_score, away_score = offense_score, defense_score
            else:
                home_score, away_score = defense_score, offense_score
            rows.append(
                {
                    "game_id": game_id,
                    "play_number": item.get("playNumber"),
                    "period": item.get("period"),
                    "down": item.get("down"),
                    "distance": item.get("distance"),
                    "yards_to_goal": item.get("yardsToGoal"),
                    "clock_minutes": clock.get("minutes"),
                    "clock_seconds": clock.get("seconds"),
                    "score_diff": offense_score - defense_score,
                    "scoring_play": bool(scoring),
                    "play_type": item.get("playType") or "",
                    "home": home,
                    "away": away,
                    "pos_team": offense,
                    "winner": finals[game_id]["winner"],
                    "pos_team_timeouts_rem_before": item.get("offenseTimeouts"),
                    "def_pos_team_timeouts_rem_before": item.get("defenseTimeouts"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "play_text": item.get("playText") or "",
                }
            )

        plays = pd.DataFrame(rows)
        if plays.empty:
            continue
        plays["game_id"] = pd.to_numeric(plays["game_id"], errors="coerce")
        plays["play_number"] = pd.to_numeric(plays["play_number"], errors="coerce")
        plays["period"] = pd.to_numeric(plays["period"], errors="coerce")
        plays = plays.dropna(
            subset=["game_id", "play_number", "period", "down", "pos_team", "home", "winner"]
        ).sort_values(["game_id", "play_number"])
        game_count = int(plays["game_id"].nunique())
        if game_count < 5:
            continue

        predictions = []
        splitter = GroupKFold(n_splits=min(5, game_count))
        for train_indices, test_indices in splitter.split(plays, groups=plays["game_id"]):
            train_plays = plays.iloc[train_indices].copy()
            test_plays = plays.iloc[test_indices].copy()

            ep_model = train_ep_model(train_plays, persist=False)
            if not ep_model:
                continue
            train_epa = compute_epa(ep_model, train_plays)
            wp_model = train_wp_model(train_epa, persist=False)
            if not wp_model:
                continue

            test_epa = compute_epa(ep_model, test_plays)
            possession_wp = predict_wp(wp_model, test_epa)
            scored = test_epa.copy()
            scored["home_wp"] = np.where(
                scored["pos_team"].to_numpy() == scored["home"].to_numpy(),
                possession_wp,
                1 - possession_wp,
            )
            predictions.append(scored)

        if not predictions:
            continue
        predicted = pd.concat(predictions, ignore_index=True)
        for game_id, game in predicted.groupby("game_id", sort=True):
            game_id = int(game_id)
            final = finals.get(game_id)
            if not final:
                continue
            chart = game.sort_values("play_number").copy()
            chart = chart.rename(columns={"play_number": "play"})
            chart = chart[chart["home_wp"].notna()]
            if chart.empty:
                continue

            home_name = final["home"]
            away_name = final["away"]
            final_home_wp = float(np.clip(chart["home_wp"].iloc[-1], 0, 1))
            score = (
                f"{home_name} {final['home_score']} – "
                f"{final['away_score']} {away_name}"
            )
            payload = {
                "page": "win_probability",
                "title": "📈 In-Game Win Probability",
                "caption": (
                    f"Research-only in-house model. This {season} game was held out "
                    "of training; this curve is not the CFBD win-probability model."
                ),
                "metrics": [
                    {"label": f"{home_name} Win Prob", "value": f"{final_home_wp:.1%}", "delta": None, "help": None},
                    {"label": f"{away_name} Win Prob", "value": f"{1 - final_home_wp:.1%}", "delta": None, "help": None},
                    {"label": "Final Score", "value": score, "delta": None, "help": None},
                ],
                "figure": win_probability._wp_figure(chart, home_name, away_name),
                "raw_expander": {
                    "label": "Raw play data",
                    "expanded": False,
                    "columns": ["play", "home_wp", "home_score", "away_score", "play_text"],
                    "rows": records(
                        chart[["play", "home_wp", "home_score", "away_score", "play_text"]]
                    ),
                    "height": 300,
                },
            }
            snapshots[(int(season), game_id)] = _write_snapshot(
                f"win-probability/{int(season)}/{game_id}.json",
                payload,
            )
        print(f"win probability {season}: {sum(1 for key in snapshots if key[0] == int(season))} game charts")
    return snapshots


def _export_win_probability_catalog() -> dict:
    """Export schedules plus game-held-out charts from cached local play data."""
    import pandas as pd

    games_path = ROOT / "data_files" / "processed" / "games.parquet"
    if not games_path.exists():
        raise RuntimeError(f"Missing local game schedule: {games_path.relative_to(ROOT)}")
    frame = pd.read_parquet(games_path)
    required = {"game_id", "season", "week", "season_type", "home_team", "away_team"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Game schedule is missing columns: {', '.join(sorted(missing))}")

    chart_snapshots = _export_local_win_probability_snapshots(
        win_probability.SEASON_OPTIONS,
        frame,
    )
    unavailable_url = _write_snapshot(
        "win-probability/unavailable.json",
        {
            "page": "win_probability",
            "title": "📈 In-Game Win Probability",
            "caption": (
                "Research-only in-house win-probability charts are generated from "
                "cached play-by-play where enough data is available."
            ),
            "warnings": [
                "No precomputed play-by-play chart is available for this game."
            ],
            "stopped": True,
        },
    )
    no_games_url = _write_snapshot(
        "win-probability/no-games.json",
        {
            "page": "win_probability",
            "title": "📈 In-Game Win Probability",
            "caption": (
                "Research-only in-house win-probability charts are generated from "
                "cached play-by-play where enough data is available."
            ),
            "info": "No games found for the selected week.",
            "stopped": True,
        },
    )

    schedule: dict[str, dict[str, dict[str, dict]]] = {}
    games_seen_by_season = {str(int(year)): 0 for year in win_probability.SEASON_OPTIONS}
    for season in win_probability.SEASON_OPTIONS:
        season_key = str(int(season))
        schedule[season_key] = {}
        for season_type in win_probability.SEASON_TYPES:
            kind = season_type.lower()
            schedule[season_key][season_type] = {}
            season_games = frame[
                (pd.to_numeric(frame["season"], errors="coerce") == int(season))
                & (frame["season_type"].astype(str).str.lower() == kind)
            ]
            for week in win_probability._week_options(kind):
                week_games = season_games[
                    pd.to_numeric(season_games["week"], errors="coerce") == int(week)
                ]
                game_options: dict[str, int] = {}
                for row in week_games.itertuples(index=False):
                    if pd.isna(row.game_id):
                        continue
                    try:
                        game_id = int(row.game_id)
                    except (TypeError, ValueError):
                        continue
                    game_options[f"{row.away_team} @ {row.home_team}"] = game_id

                snapshots = {
                    str(game_id): chart_snapshots.get((int(season), game_id), unavailable_url)
                    for game_id in game_options.values()
                }
                if game_options:
                    games_seen_by_season[season_key] += len(game_options)
                schedule[season_key][season_type][str(int(week))] = {
                    "games": sorted(game_options),
                    "game_options": game_options,
                    "snapshots": snapshots,
                    **({"empty_url": no_games_url} if not game_options else {}),
                }

    missing_seasons = [
        season for season, count in games_seen_by_season.items() if count == 0
    ]
    if missing_seasons:
        raise RuntimeError(
            "Local game schedule has no games for Win Probability seasons "
            f"{', '.join(missing_seasons)}."
        )

    return {
        "seasons": win_probability.SEASON_OPTIONS,
        "default_season": win_probability.SEASON_OPTIONS[0],
        "default_season_type": win_probability.SEASON_TYPES[0],
        "default_week": 1,
        "season_types": win_probability.SEASON_TYPES,
        "schedule": schedule,
        "chart_seasons": sorted({season for season, _ in chart_snapshots}),
        "chart_count": len(chart_snapshots),
    }


def _git_revision() -> str | None:
    """Return the workflow checkout revision without requiring GitPython."""
    import os

    return os.environ.get("GITHUB_SHA")


if __name__ == "__main__":
    raise SystemExit(main())

