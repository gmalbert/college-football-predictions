"""Repository-level data, model, freshness, and leakage audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from utils.contracts import (
    Severity,
    validate_feature_matrix,
    validate_games,
    validate_team_game_stats,
)
from utils.storage import DATA_DIR
from utils.seasons import current_cfb_season
from utils.release import load_current_release


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    message: str
    value: object = None


LEAKAGE_RISK_COLUMNS = {
    "sp_plus_diff": "season-final SP+ joined to earlier games",
    "off_epa_diff": "season aggregate EPA joined to earlier games",
    "fpi_diff": "season snapshot without available_at",
    "srs_diff": "season-final SRS joined to earlier games",
    "ppa_off_diff": "season PPA aggregate without an as-of week",
    "wepa_off_diff": "season WEPA aggregate without an as-of week",
}


def _contract_checks(report) -> list[AuditCheck]:
    if report.ok and not report.issues:
        return [AuditCheck(f"{report.contract}_contract", "pass", "Contract passed", report.row_count)]
    checks = []
    for issue in report.issues:
        checks.append(
            AuditCheck(
                f"{report.contract}:{issue.code}",
                "fail" if issue.severity == Severity.ERROR else "warn",
                issue.message,
                issue.rows,
            )
        )
    return checks


FRESHNESS_CHECK_NAMES = ("best_bets_freshness", "shadow_total_freshness")


def _age_hours(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    generated = pd.to_datetime(
        payload.get("meta", {}).get("generated_at"), errors="coerce", utc=True
    )
    if pd.isna(generated):
        return float("inf")
    return (pd.Timestamp.now(tz="UTC") - generated).total_seconds() / 3600


def release_metadata_check() -> AuditCheck:
    release = load_current_release()
    return AuditCheck(
        "release_metadata",
        "pass" if release.get("artifact_fingerprint") else "warn",
        "Release metadata is present" if release else "No atomic release metadata has been published",
        release.get("release_id"),
    )


def empty_raw_cache_check(data_path: Path) -> AuditCheck:
    raw_path = data_path / "raw"
    raw_files = list(raw_path.glob("*.json")) if raw_path.exists() else []
    empty_raw = [path.name for path in raw_files if path.stat().st_size <= 2]
    return AuditCheck(
        "empty_raw_cache",
        "warn" if empty_raw else "pass",
        f"{len(empty_raw)} empty JSON caches; new ingestion code ignores and does not create empty caches",
        len(empty_raw),
    )


def season_schedule_check(latest_season: int) -> AuditCheck:
    """Compare the newest scheduled season against the current year.

    ``latest_season`` is a property of the committed games artifact; only the
    comparison against today's year changes, so a stored report can supply the
    former and this recomputes the latter.
    """
    current_season = current_cfb_season(datetime.now(timezone.utc))
    return AuditCheck(
        "current_season_schedule",
        "pass" if latest_season >= current_season else "warn",
        f"Latest scheduled season is {latest_season}; current year is {current_season}",
        latest_season,
    )


def freshness_checks(data_path: Path) -> list[AuditCheck]:
    """The time-dependent checks.

    Split out because they are the only part of the audit whose answer changes
    between pipeline runs. A persisted audit report can be served as-is for the
    structural checks, with these recomputed at view time so the page does not
    report a staleness age that was true when the workflow last ran.
    """
    best_bets_age = _age_hours(data_path / "best_bets_today.json")
    shadow_age = _age_hours(data_path / "shadow_total_signals.json")
    return [
        AuditCheck(
            "best_bets_freshness",
            "pass" if best_bets_age <= 30 else "warn",
            f"Best-bets export age is {best_bets_age:.1f} hours",
            best_bets_age,
        ),
        AuditCheck(
            "shadow_total_freshness",
            "pass" if shadow_age <= 30 else "warn",
            f"Shadow-total export age is {shadow_age:.1f} hours",
            shadow_age,
        ),
    ]


def merge_live_checks(report: dict, data_path: Path) -> dict:
    """Return ``report`` with its live-state checks recomputed for right now.

    Most checks are properties of the committed Parquet artifacts and do not
    change between pipeline runs, so a published report can be served as-is.
    The ones listed in ``LIVE_STATE_CHECKS`` read something outside those
    artifacts — wall-clock time, the release manifest, the raw cache directory —
    and would otherwise be reported as they were when the workflow last ran.
    """
    live = {check.name: asdict(check) for check in freshness_checks(data_path)}
    live["release_metadata"] = asdict(release_metadata_check())
    live["empty_raw_cache"] = asdict(empty_raw_cache_check(data_path))

    checks: list[dict] = []
    for check in report.get("checks", []):
        name = check.get("name")
        if name == "current_season_schedule":
            # The stored value is the newest scheduled season; only the
            # comparison against the current year needs redoing.
            try:
                live[name] = asdict(season_schedule_check(int(check.get("value"))))
            except (TypeError, ValueError):
                pass
        checks.append(live.get(name, check))

    strict_blockers = {"best_bets_freshness", "shadow_total_freshness", "release_metadata"}
    return {
        **report,
        "checks": checks,
        "summary": {
            "pass": sum(c["status"] == "pass" for c in checks),
            "warn": sum(c["status"] == "warn" for c in checks),
            "fail": sum(c["status"] == "fail" for c in checks),
        },
        "live_checks_recomputed_at": datetime.now(timezone.utc).isoformat(),
        "strict_failures": sum(
            c["status"] == "fail"
            or (c["status"] == "warn" and c["name"] in strict_blockers)
            for c in checks
        ),
    }


def run_repository_audit(
    root: str | Path | None = None,
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict:
    """Audit the published artifacts.

    ``frames`` lets a caller that already holds these artifacts hand them over
    instead of having them read again. The API keeps every one of them in its
    own mtime-keyed cache, and re-reading them here materialised a second full
    copy of the feature matrix — measured at +119 MB of resident memory in
    scripts/measure_memory.py.

    Keys match the artifact names used by ``utils.storage.load_parquet``:
    ``games``, ``feature_matrix``, ``team_game_stats``, ``line_snapshots``,
    ``feature_observations`` and ``model_backtest``.
    """
    root_path = Path(root) if root is not None else DATA_DIR.parent
    data_path = root_path / "data_files"
    checks: list[AuditCheck] = []
    supplied = frames or {}

    def _frame(name: str, path: Path) -> pd.DataFrame:
        cached = supplied.get(name)
        if cached is not None:
            return cached
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    games_path = data_path / "processed" / "games.parquet"
    feature_path = data_path / "features" / "feature_matrix.parquet"
    team_stats_path = data_path / "processed" / "team_game_stats.parquet"
    games = _frame("games", games_path)
    features = _frame("feature_matrix", feature_path)
    team_stats = _frame("team_game_stats", team_stats_path)

    if games.empty:
        checks.append(AuditCheck("games_available", "fail", "games.parquet is missing or empty"))
    else:
        checks.extend(_contract_checks(validate_games(games)))
    if features.empty:
        checks.append(AuditCheck("features_available", "fail", "feature_matrix.parquet is missing or empty"))
    else:
        checks.extend(_contract_checks(validate_feature_matrix(features)))
    if team_stats.empty:
        checks.append(AuditCheck("team_stats_available", "warn", "team_game_stats.parquet is unavailable"))
    else:
        checks.extend(_contract_checks(validate_team_game_stats(team_stats)))

    if not games.empty and not features.empty:
        expected = int(games["game_id"].nunique())
        actual = int(features["game_id"].nunique())
        checks.append(
            AuditCheck(
                "feature_game_coverage",
                "pass" if actual == expected else "warn",
                f"Feature artifact covers {actual:,} of {expected:,} unique games",
                actual / expected if expected else 0,
            )
        )
        latest_season = int(pd.to_numeric(games["season"], errors="coerce").max())
        checks.append(season_schedule_check(latest_season))

    risky_present = [column for column in LEAKAGE_RISK_COLUMNS if column in features.columns]
    checks.append(
        AuditCheck(
            "unsafe_exploration_columns",
            "warn" if risky_present else "pass",
            "Season aggregates may remain for UI exploration but are excluded from production feature lists: "
            + ", ".join(risky_present) if risky_present else "No known unsafe season aggregates present",
            len(risky_present),
        )
    )

    checks.append(release_metadata_check())

    checks.append(empty_raw_cache_check(data_path))

    snapshots_path = data_path / "processed" / "line_snapshots.parquet"
    snapshots = _frame("line_snapshots", snapshots_path)
    checks.append(
        AuditCheck(
            "market_snapshot_history",
            "pass" if not snapshots.empty else "warn",
            f"Timestamped market history contains {len(snapshots):,} quotes; "
            "movement and CLV require forward collection",
            len(snapshots),
        )
    )

    observations_path = data_path / "processed" / "feature_observations.parquet"
    observations = _frame("feature_observations", observations_path)
    checks.append(
        AuditCheck(
            "context_feature_observations",
            "pass" if not observations.empty else "warn",
            f"Timestamped contextual observations contain {len(observations):,} rows"
            if not observations.empty else "No timestamped contextual source has been ingested",
            len(observations),
        )
    )

    metrics_path = data_path / "models" / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    scope = metrics.get("evaluation_scope")
    checks.append(
        AuditCheck(
            "model_evaluation_scope",
            "pass" if scope == "walk_forward_season_oos" else "fail",
            f"Evaluation scope is {scope or 'undeclared'}",
            scope,
        )
    )
    decision = metrics.get("release_decision", {})
    checks.append(
        AuditCheck(
            "model_release_decision",
            "pass" if decision.get("decision") == "promote" else "warn",
            "Release decision is " + str(decision.get("decision", "undeclared"))
            + "; failed gates: " + ", ".join(decision.get("failed_gates", [])),
            decision.get("decision"),
        )
    )

    backtest_path = data_path / "features" / "model_backtest.parquet"
    backtest = _frame("model_backtest", backtest_path)
    duplicate_backtests = (
        int(backtest["game_id"].duplicated().sum())
        if not backtest.empty and "game_id" in backtest.columns else 0
    )
    oos_columns = {
        "win_prob_oos", "predicted_spread_oos", "predicted_total_oos",
        "total_over_prob_oos",
    }
    backtest_ok = (
        not backtest.empty
        and not duplicate_backtests
        and oos_columns.issubset(backtest.columns)
        and backtest[list(oos_columns)].notna().any().all()
    )
    checks.append(
        AuditCheck(
            "oos_prediction_artifact",
            "pass" if backtest_ok else "fail",
            f"OOS artifact has {len(backtest):,} rows and {duplicate_backtests} duplicate games",
            len(backtest),
        )
    )

    manifests = list((data_path / "models").glob("*_manifest.json"))
    checks.append(
        AuditCheck(
            "model_manifests",
            "pass" if len(manifests) >= 4 else "fail",
            f"Found {len(manifests)} model artifact manifests",
            len(manifests),
        )
    )

    checks.extend(freshness_checks(data_path))

    summary = {
        "pass": sum(check.status == "pass" for check in checks),
        "warn": sum(check.status == "warn" for check in checks),
        "fail": sum(check.status == "fail" for check in checks),
    }
    strict_blockers = {
        "best_bets_freshness",
        "shadow_total_freshness",
        "release_metadata",
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "checks": [asdict(check) for check in checks],
        "release_status": "shadow" if metrics.get("total_cover_model", {}).get("strategy_release", {}).get("status") == "shadow" else "hold",
        "strict_failures": sum(
            check.status == "fail" or (check.status == "warn" and check.name in strict_blockers)
            for check in checks
        ),
    }
