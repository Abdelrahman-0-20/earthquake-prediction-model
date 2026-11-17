"""Cleaning and validation gates.

Hard checks (fail the build, exit 3):
  * unique event ``id`` after deduplication
  * latitude / longitude inside valid ranges
  * ``time`` and ``updated`` parse as timezone-aware UTC
  * ``mag`` and ``depth`` numeric and finite

Soft checks (measured and reported, never silently fixed):
  * missing-value rate per column — ``nst``, ``gap``, ``dmin``, ``rms`` are
    frequently absent in the catalog; missingness is measured, not filled
  * negative depths — physically near-surface noise, retained but flagged
  * non-earthquake event types (quarry blasts etc.) — dropped from the
    processed dataset, with counts recorded in the report

Output: ``data/processed/earthquakes.parquet`` + a data-quality report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..errors import ValidationError
from .ingestion import atomic_write_parquet, require_raw_data

log = logging.getLogger("seismo.preprocessing")


@dataclass
class ValidationReport:
    generated_at: str = ""
    source_files: int = 0
    total_raw_rows: int = 0
    duplicates_removed: int = 0
    non_earthquake_removed: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    date_range: dict[str, str] = field(default_factory=dict)
    magnitude_range: dict[str, float] = field(default_factory=dict)
    depth_range: dict[str, float] = field(default_factory=dict)
    negative_depth_rows: int = 0
    # Soft check: events the API admitted under the configured magnitude filter
    # whose *current* magnitude sits below it — catalog revisions lower a
    # magnitude after index time. Measured, never silently dropped.
    below_threshold_rows: int = 0
    below_threshold_min_mag: float | None = None
    hard_violations: list[str] = field(default_factory=list)
    missingness: dict[str, dict[str, float]] = field(default_factory=dict)
    processed_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(cfg: Config) -> tuple[pd.DataFrame, list[Path]]:
    """Concatenate every raw month file into one frame."""
    files = require_raw_data(cfg)
    frames = [pd.read_parquet(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    log.info("loaded %s rows from %d raw files", f"{len(frame):,}", len(files))
    return frame, files


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate event ids, keeping the latest catalog revision.

    USGS revises events (new ``updated`` timestamp) between pulls, so the same
    id can arrive twice with different values — the most recent wins.
    """
    before = len(frame)
    if before == 0:
        return frame, 0
    if "updated" in frame.columns:
        frame = frame.sort_values("updated", kind="stable")
    cleaned = frame.drop_duplicates(subset="id", keep="last")
    return cleaned, before - len(cleaned)


def filter_event_types(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Keep only configured event types; return (frame, dropped, type counts)."""
    if frame.empty or "type" not in frame.columns:
        return frame, 0, {}
    counts = frame["type"].fillna("(missing)").value_counts().to_dict()
    keep = set(cfg.validation.get("keep_event_types", ["earthquake"]))
    mask = frame["type"].isin(keep)
    dropped = int((~mask).sum())
    return frame[mask].copy(), dropped, {str(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Hard validation
# ---------------------------------------------------------------------------

def check_hard(frame: pd.DataFrame, cfg: Config) -> list[str]:
    """Return a list of hard violations (empty list == pass)."""
    violations: list[str] = []
    if frame.empty:
        return violations

    lat_lo, lat_hi = cfg.validation["latitude_range"]
    lon_lo, lon_hi = cfg.validation["longitude_range"]

    # Unique ids
    if frame["id"].isna().any():
        violations.append(f"{int(frame['id'].isna().sum())} rows have a missing event id")
    dupes = int(frame["id"].duplicated().sum())
    if dupes:
        violations.append(f"{dupes} duplicate event ids remain after deduplication")

    # Coordinates
    lat = pd.to_numeric(frame["latitude"], errors="coerce")
    bad_lat = lat.isna() | (lat < lat_lo) | (lat > lat_hi)
    if bad_lat.any():
        violations.append(
            f"{int(bad_lat.sum())} rows have latitude outside [{lat_lo}, {lat_hi}] "
            f"or non-numeric"
        )
    lon = pd.to_numeric(frame["longitude"], errors="coerce")
    bad_lon = lon.isna() | (lon < lon_lo) | (lon > lon_hi)
    if bad_lon.any():
        violations.append(
            f"{int(bad_lon.sum())} rows have longitude outside [{lon_lo}, {lon_hi}] "
            f"or non-numeric"
        )

    # Timestamps
    for column in ("time", "updated"):
        if column not in frame.columns:
            violations.append(f"required column {column!r} is missing")
            continue
        parsed = pd.to_datetime(frame[column], format="ISO8601", utc=True, errors="coerce")
        bad = parsed.isna()
        if bad.any():
            violations.append(f"{int(bad.sum())} rows have an unparseable {column!r}")

    # Magnitude and depth
    for column in ("mag", "depth"):
        if column not in frame.columns:
            violations.append(f"required column {column!r} is missing")
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = numeric.apply(
            lambda v: pd.notna(v) and float("-inf") < v < float("inf")
        )
        bad = numeric.isna() | ~finite
        if bad.any():
            violations.append(f"{int(bad.sum())} rows have non-numeric or non-finite {column!r}")

    return violations


# ---------------------------------------------------------------------------
# Soft checks / reporting
# ---------------------------------------------------------------------------

def missingness(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-column missing counts and percentages (nulls + empty strings)."""
    total = len(frame)
    report: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        series = frame[column]
        missing = series.isna()
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            missing = missing | series.astype(str).str.strip().eq("") | series.astype(str).eq("nan")
        count = int(missing.sum())
        report[column] = {
            "missing": count,
            "pct": round(100.0 * count / total, 3) if total else 0.0,
        }
    return report


def _range(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(series.min()), "max": float(series.max())}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_processed(cfg: Config, *, strict: bool = True) -> ValidationReport:
    """Clean raw data, run gates, and write the processed dataset + report.

    The report is always written (even on failure) so the user can inspect what
    went wrong; ``strict=True`` raises ``ValidationError`` on hard violations.
    """
    raw, files = load_raw(cfg)
    report = ValidationReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_files=len(files),
        total_raw_rows=len(raw),
    )

    # Soft: event-type accounting, then filter to analysis types.
    typed, dropped, type_counts = filter_event_types(raw, cfg)
    report.non_earthquake_removed = dropped
    report.event_type_counts = type_counts

    # Dedup (keeps latest revision).
    deduped, dupes = deduplicate(typed)
    report.duplicates_removed = dupes

    # Hard gates.
    report.hard_violations = check_hard(deduped, cfg)

    # Type coercion for the processed dataset.
    processed = deduped.copy()
    if not processed.empty:
        processed["time"] = pd.to_datetime(processed["time"], format="ISO8601", utc=True)
        processed["updated"] = pd.to_datetime(processed["updated"], format="ISO8601", utc=True)
        for column in ("latitude", "longitude", "depth", "mag"):
            processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed = processed.sort_values("time").reset_index(drop=True)
        processed["year"] = processed["time"].dt.year
        processed["month"] = processed["time"].dt.month

        report.date_range = {
            "start": processed["time"].min().isoformat(),
            "end": processed["time"].max().isoformat(),
        }
        report.magnitude_range = _range(processed["mag"])
        report.depth_range = _range(processed["depth"])
        report.negative_depth_rows = int((processed["depth"] < 0).sum())

        # Soft: magnitude filter vs. current catalog value (revisions lower mag).
        threshold = cfg.min_magnitude
        below = processed[processed["mag"] < threshold]
        report.below_threshold_rows = int(len(below))
        report.below_threshold_min_mag = float(below["mag"].min()) if len(below) else None

    report.missingness = missingness(processed)
    report.processed_rows = len(processed)

    # Persist report first so failures remain diagnosable.
    _write_report(cfg, report)

    if report.hard_violations and strict:
        detail = "\n".join(f"    - {v}" for v in report.hard_violations)
        raise ValidationError(
            f"validation failed with {len(report.hard_violations)} hard violation(s):\n{detail}",
            hint=f"Inspect {cfg.quality_report_dir}/data_quality.json, then re-run "
                 "`seismo data fetch --force` for the affected months.",
        )

    out_path = cfg.processed_dir / "earthquakes.parquet"
    atomic_write_parquet(processed, out_path)
    log.info("wrote %s rows to %s", f"{len(processed):,}", out_path)
    return report


def _write_report(cfg: Config, report: ValidationReport) -> Path:
    directory = cfg.quality_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data_quality.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_processed(cfg: Config) -> pd.DataFrame:
    """Load the processed dataset, with a helpful error when it is absent."""
    from ..errors import MissingArtifact

    path = cfg.processed_dir / "earthquakes.parquet"
    if not path.is_file():
        raise MissingArtifact(
            f"Processed dataset not found at {path}",
            hint="Run `seismo data fetch` then `seismo data validate`.",
        )
    return pd.read_parquet(path)
