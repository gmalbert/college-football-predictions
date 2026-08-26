"""Data contracts and guarded joins for the prediction pipeline.

The project intentionally keeps these checks dependency-light.  They provide the
most important guarantees of a dataframe schema library without requiring a new
runtime package: declared grain, required columns, value ranges, key uniqueness,
and point-in-time availability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import pandas as pd


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Severity = Severity.ERROR
    rows: int | None = None


@dataclass
class ValidationReport:
    contract: str
    row_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        rows: int | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(code, message, severity, rows))

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in self.issues
            if issue.severity == Severity.ERROR
        )
        raise ValueError(f"{self.contract} contract failed: {details}")

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "row_count": self.row_count,
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class FrameContract:
    name: str
    grain: str
    required: tuple[str, ...]
    unique_key: tuple[str, ...] = ()
    non_null: tuple[str, ...] = ()
    ranges: Mapping[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(self.name, len(frame))
        missing = [column for column in self.required if column not in frame.columns]
        if missing:
            report.add("missing_columns", f"missing required columns: {missing}")
            return report

        if self.unique_key:
            duplicates = int(frame.duplicated(list(self.unique_key), keep=False).sum())
            if duplicates:
                report.add(
                    "duplicate_grain",
                    f"{duplicates} rows violate {self.grain} grain on {self.unique_key}",
                    rows=duplicates,
                )

        for column in self.non_null:
            if column not in frame.columns:
                continue
            nulls = int(frame[column].isna().sum())
            if nulls:
                report.add(
                    "null_required_value",
                    f"{column} contains {nulls} null values",
                    rows=nulls,
                )

        for column, (lower, upper) in self.ranges.items():
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = pd.Series(False, index=frame.index)
            if lower is not None:
                invalid |= values < lower
            if upper is not None:
                invalid |= values > upper
            count = int(invalid.fillna(False).sum())
            if count:
                report.add(
                    "out_of_range",
                    f"{column} has {count} values outside [{lower}, {upper}]",
                    rows=count,
                )
        return report


GAME_CONTRACT = FrameContract(
    name="games",
    grain="one row per game",
    required=(
        "game_id", "season", "week", "home_team", "away_team", "start_date",
    ),
    unique_key=("game_id",),
    non_null=("game_id", "season", "week", "home_team", "away_team"),
    ranges={"season": (1869, 2100), "week": (0, 30)},
)

TEAM_GAME_CONTRACT = FrameContract(
    name="team_game_stats",
    grain="one row per team per game",
    required=("game_id", "season", "team"),
    unique_key=("game_id", "team"),
    non_null=("game_id", "season", "team"),
    ranges={"season": (1869, 2100)},
)

LINE_SNAPSHOT_CONTRACT = FrameContract(
    name="line_snapshots",
    grain="one quote per sportsbook, market, side, and capture time",
    required=(
        "game_id", "sportsbook", "market", "side", "captured_at", "odds",
    ),
    unique_key=("game_id", "sportsbook", "market", "side", "captured_at"),
    # Some feeds expose a line without its associated price. Keep that absence
    # explicit; price-aware EV code must reject rather than invent such odds.
    non_null=("game_id", "sportsbook", "market", "side", "captured_at"),
)

FEATURE_OBSERVATION_CONTRACT = FrameContract(
    name="feature_observations",
    grain="one entity feature observation per source version and availability time",
    required=("entity_id", "entity_type", "feature_name", "value", "available_at", "source_version"),
    unique_key=("entity_id", "entity_type", "feature_name", "available_at", "source_version"),
    non_null=("entity_id", "entity_type", "feature_name", "available_at", "source_version"),
)

FEATURE_CONTRACT = FrameContract(
    name="feature_matrix",
    grain="one prediction row per game and prediction timestamp",
    required=("game_id", "season", "home_team", "away_team"),
    unique_key=("game_id",),
    non_null=("game_id", "season", "home_team", "away_team"),
    ranges={"season": (1869, 2100)},
)


def ensure_utc(values: pd.Series) -> pd.Series:
    """Parse timestamps as timezone-aware UTC values."""
    return pd.to_datetime(values, errors="coerce", utc=True)


def validate_games(frame: pd.DataFrame) -> ValidationReport:
    report = GAME_CONTRACT.validate(frame)
    if {"home_team", "away_team"}.issubset(frame.columns):
        same = int((frame["home_team"] == frame["away_team"]).fillna(False).sum())
        if same:
            report.add(
                "same_team_matchup",
                f"{same} games have the same home and away team",
                rows=same,
            )
    if "start_date" in frame.columns:
        bad_dates = int(ensure_utc(frame["start_date"]).isna().sum())
        if bad_dates:
            report.add(
                "invalid_start_date",
                f"{bad_dates} games have an invalid start_date",
                severity=Severity.WARNING,
                rows=bad_dates,
            )
    return report


def validate_team_game_stats(frame: pd.DataFrame) -> ValidationReport:
    return TEAM_GAME_CONTRACT.validate(frame)


def validate_line_snapshots(frame: pd.DataFrame) -> ValidationReport:
    report = LINE_SNAPSHOT_CONTRACT.validate(frame)
    if "captured_at" in frame.columns:
        bad_dates = int(ensure_utc(frame["captured_at"]).isna().sum())
        if bad_dates:
            report.add(
                "invalid_capture_time",
                f"{bad_dates} quotes have an invalid captured_at value",
                rows=bad_dates,
            )
    return report


def validate_feature_observations(frame: pd.DataFrame) -> ValidationReport:
    report = FEATURE_OBSERVATION_CONTRACT.validate(frame)
    if "available_at" in frame.columns:
        invalid = int(ensure_utc(frame["available_at"]).isna().sum())
        if invalid:
            report.add("invalid_available_at", f"{invalid} observations have invalid availability", rows=invalid)
    return report


def validate_feature_matrix(frame: pd.DataFrame) -> ValidationReport:
    report = FEATURE_CONTRACT.validate(frame)
    if {"feature_as_of", "start_date"}.issubset(frame.columns):
        future = ensure_utc(frame["feature_as_of"]) > ensure_utc(frame["start_date"])
        count = int(future.fillna(False).sum())
        if count:
            report.add(
                "features_after_kickoff",
                f"{count} feature rows were materialized after kickoff",
                rows=count,
            )
    return report


def assert_point_in_time(
    frame: pd.DataFrame,
    *,
    available_at: str = "available_at",
    prediction_time: str = "prediction_time",
) -> None:
    """Raise when any observation became available after prediction time."""
    missing = [c for c in (available_at, prediction_time) if c not in frame.columns]
    if missing:
        raise KeyError(f"point-in-time audit requires columns: {missing}")
    available = ensure_utc(frame[available_at])
    predicted = ensure_utc(frame[prediction_time])
    invalid = available.isna() | predicted.isna() | (available > predicted)
    if invalid.any():
        raise ValueError(
            f"point-in-time violation in {int(invalid.sum())} rows: "
            f"{available_at} must be <= {prediction_time}"
        )


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | Sequence[str],
    how: str = "left",
    validate: str = "many_to_one",
    allow_row_growth: bool = False,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """Merge with an explicit cardinality contract and row-growth guard."""
    before = len(left)
    merged = left.merge(
        right,
        on=on,
        how=how,
        validate=validate,
        suffixes=suffixes,
    )
    if not allow_row_growth and how in {"left", "inner"} and len(merged) > before:
        raise ValueError(
            f"guarded merge grew from {before:,} to {len(merged):,} rows on {on}; "
            "check right-side grain"
        )
    return merged


def missingness_summary(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a compact, sortable missing-data profile."""
    selected = list(columns) if columns is not None else list(frame.columns)
    selected = [column for column in selected if column in frame.columns]
    if not selected:
        return pd.DataFrame(columns=["column", "null_count", "null_pct"])
    counts = frame[selected].isna().sum()
    return (
        pd.DataFrame(
            {
                "column": counts.index,
                "null_count": counts.values.astype(int),
                "null_pct": counts.values / max(len(frame), 1),
            }
        )
        .sort_values(["null_pct", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
