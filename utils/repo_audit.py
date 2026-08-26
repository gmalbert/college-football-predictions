"""Repository-level data, model, freshness, and leakage audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

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


def run_repository_audit(root: str | Path | None = None) -> dict:
    root_path = Path(root) if root is not None else DATA_DIR.parent
    data_path = root_path / "data_files"
    checks: list[AuditCheck] = []

    games_path = data_path / "processed" / "games.parquet"
    feature_path = data_path / "features" / "feature_matrix.parquet"
    team_stats_path = data_path / "processed" / "team_game_stats.parquet"
    games = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()
    features = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    team_stats = pd.read_parquet(team_stats_path) if team_stats_path.exists() else pd.DataFrame()

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
        current_season = current_cfb_season(datetime.now(timezone.utc))
        latest_season = int(pd.to_numeric(games["season"], errors="coerce").max())
        checks.append(
            AuditCheck(
                "current_season_schedule",
                "pass" if latest_season >= current_season else "warn",
                f"Latest scheduled season is {latest_season}; current year is {current_season}",
                latest_season,
            )
        )

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

    release = load_current_release()
    checks.append(
        AuditCheck(
            "release_metadata",
            "pass" if release.get("artifact_fingerprint") else "warn",
            "Release metadata is present" if release else "No atomic release metadata has been published",
            release.get("release_id"),
        )
    )

    raw_path = data_path / "raw"
    raw_files = list(raw_path.glob("*.json")) if raw_path.exists() else []
    empty_raw = [path.name for path in raw_files if path.stat().st_size <= 2]
    checks.append(
        AuditCheck(
            "empty_raw_cache",
            "warn" if empty_raw else "pass",
            f"{len(empty_raw)} empty JSON caches; new ingestion code ignores and does not create empty caches",
            len(empty_raw),
        )
    )

    snapshots_path = data_path / "processed" / "line_snapshots.parquet"
    snapshots = pd.read_parquet(snapshots_path) if snapshots_path.exists() else pd.DataFrame()
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
    observations = pd.read_parquet(observations_path) if observations_path.exists() else pd.DataFrame()
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
    backtest = pd.read_parquet(backtest_path) if backtest_path.exists() else pd.DataFrame()
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

    export_path = data_path / "best_bets_today.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else {}
    generated = pd.to_datetime(export.get("meta", {}).get("generated_at"), errors="coerce", utc=True)
    age_hours = (
        (pd.Timestamp.now(tz="UTC") - generated).total_seconds() / 3600
        if pd.notna(generated) else float("inf")
    )
    checks.append(
        AuditCheck(
            "best_bets_freshness",
            "pass" if age_hours <= 30 else "warn",
            f"Best-bets export age is {age_hours:.1f} hours",
            age_hours,
        )
    )

    shadow_path = data_path / "shadow_total_signals.json"
    shadow = json.loads(shadow_path.read_text(encoding="utf-8")) if shadow_path.exists() else {}
    shadow_generated = pd.to_datetime(
        shadow.get("meta", {}).get("generated_at"), errors="coerce", utc=True
    )
    shadow_age = (
        (pd.Timestamp.now(tz="UTC") - shadow_generated).total_seconds() / 3600
        if pd.notna(shadow_generated) else float("inf")
    )
    checks.append(
        AuditCheck(
            "shadow_total_freshness",
            "pass" if shadow_age <= 30 else "warn",
            f"Shadow-total export age is {shadow_age:.1f} hours",
            shadow_age,
        )
    )

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
