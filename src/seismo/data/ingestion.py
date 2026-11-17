"""USGS ComCat ingestion.

Design contract (see docs/ARCHITECTURE.md §3.2):

* Monthly chunks — the API caps a single response at ``max_rows_per_request``
  (20,000) records, so busy months paginate automatically with limit/offset.
* Idempotent and resumable — one Parquet file per month in ``data/raw/`` plus a
  ``manifest.json`` recording request params, row count, and sha256. Re-running
  skips completed months; ``--force`` refetches.
* Atomic — a month is written to a temp file, verified, then renamed into place,
  so a crash never leaves a half-written file that looks complete.
* Honest failure — a month that still fails after retries is marked ``failed``
  in the manifest, the run continues, and the process exits non-zero listing
  every failed month.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import Config
from ..errors import ApiError, MissingArtifact

log = logging.getLogger("seismo.ingestion")

MANIFEST_NAME = "manifest.json"

#: Canonical USGS CSV columns — used to build an empty frame for months with no
#: events (the API answers 204 No Content) so concatenation stays well-formed.
CSV_COLUMNS = [
    "time", "latitude", "longitude", "depth", "mag", "magType", "nst", "gap",
    "dmin", "rms", "net", "id", "updated", "place", "type", "horizontalError",
    "depthError", "magError", "magNst", "status", "locationSource", "magSource",
]

# Statuses that make a month "complete" and therefore skippable on re-run.
_OK_STATUSES = {"ok", "empty"}


# ---------------------------------------------------------------------------
# Time chunking
# ---------------------------------------------------------------------------

def month_windows(start: date, end_inclusive: date) -> list[tuple[date, date]]:
    """Half-open monthly windows ``[w_start, w_end)`` covering [start, end].

    The final window ends the day after ``end_inclusive`` so the configured end
    date is included. Windows never overlap and never leave a gap.
    """
    if end_inclusive < start:
        raise ValueError(f"end date {end_inclusive} is before start date {start}")

    exclusive_end = end_inclusive + timedelta(days=1)
    windows: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)

    while cursor < exclusive_end:
        # First day of the following month.
        following = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        window_end = min(following, exclusive_end)
        windows.append((cursor, window_end))
        cursor = following
    return windows


def month_label(window_start: date) -> str:
    return window_start.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class _ResponseLike:
    """Structural type for a requests response (lets tests inject fakes)."""

    status_code: int
    text: str
    headers: dict[str, str]


def _retry_after_seconds(response: Any) -> float | None:
    raw = getattr(response, "headers", {}).get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def request_csv(
    session: Any,
    params: dict[str, Any],
    *,
    url: str,
    timeout: float,
    max_retries: int,
    backoff_base: float,
) -> str:
    """GET ``url`` with ``params``, retrying 429/5xx/connection failures.

    Returns the response body (empty string for 204 No Content — no events).
    Client errors (4xx other than 429) are raised immediately: retrying a
    rejected request cannot succeed.
    """
    last_reason = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_reason = f"connection error: {exc}"
        else:
            status = response.status_code
            if status == 204:
                return ""
            if status == 200:
                return response.text
            if status in (429, 500, 502, 503, 504):
                last_reason = f"HTTP {status}"
            else:
                body = str(getattr(response, "text", ""))[:300]
                raise ApiError(
                    f"USGS API rejected the request (HTTP {status}): {body}",
                    hint="Check the query parameters in config/config.yaml "
                         "(dates must be ISO 'YYYY-MM-DD').",
                )

        if attempt < max_retries:
            delay = _retry_after_seconds(response) if "response" in dir() else None
            if delay is None:
                delay = backoff_base * (2 ** (attempt - 1))
            log.warning(
                "request failed (%s) — retry %d/%d in %.1fs",
                last_reason, attempt, max_retries, delay,
            )
            time.sleep(delay)

    raise ApiError(
        f"USGS API request failed after {max_retries} attempts: {last_reason}",
        hint="The catalog endpoint may be down or rate-limiting; "
             "re-run `seismo data fetch` — completed months are skipped.",
    )


def parse_csv(text: str) -> pd.DataFrame:
    """Parse a CSV response body into a DataFrame (empty → canonical columns)."""
    if not text or not text.strip():
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_window(
    session: Any,
    window_start: date,
    window_end: date,
    *,
    base_params: dict[str, Any],
    cfg: Config,
) -> pd.DataFrame:
    """Fetch one time window, paginating when the API caps the response."""
    coll = cfg.collection
    page_size = int(coll["max_rows_per_request"])
    frames: list[pd.DataFrame] = []
    # USGS FDSN `offset` is 1-based: the first record is offset=1, not 0.
    offset = 1

    while True:
        params = {
            **base_params,
            "starttime": window_start.isoformat(),
            "endtime": window_end.isoformat(),
            "format": "csv",
            "limit": page_size,
            "offset": offset,
        }
        text = request_csv(
            session,
            params,
            url=str(coll["api_query_url"]),
            timeout=float(coll["timeout_seconds"]),
            max_retries=int(coll["max_retries"]),
            backoff_base=float(coll["backoff_base_seconds"]),
        )
        frame = parse_csv(text)
        frames.append(frame)

        if len(frame) < page_size:
            break
        offset += len(frame)
        if offset > 1_000_000:
            raise ApiError(
                f"Pagination guard tripped for {month_label(window_start)} "
                f"at {offset} rows",
                hint="Raise max_rows_per_request or narrow the date range.",
            )

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class MonthRecord:
    month: str
    status: str = "pending"          # pending | ok | empty | failed
    rows: int = 0
    sha256: str = ""
    fetched_at: str = ""
    attempts: int = 0
    error: str | None = None
    request: dict[str, Any] = field(default_factory=dict)


def load_manifest(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / MANIFEST_NAME
    if not path.is_file():
        return {"version": 1, "created_at": _now_iso(), "months": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissingArtifact(
            f"Manifest {path} is corrupt: {exc}",
            hint="Delete data/raw/manifest.json and re-run `seismo data fetch` "
                 "— already-downloaded months will be refetched.",
        ) from exc


def save_manifest(raw_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist the manifest so a crash never corrupts resume state."""
    path = raw_dir / MANIFEST_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _now_iso()
    _atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True))


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> str:
    """Write Parquet atomically; returns the sha256 of the written bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False, engine="pyarrow")
        digest = _sha256_file(tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def month_path(raw_dir: Path, label: str) -> Path:
    return raw_dir / f"{label}.parquet"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class FetchSummary:
    total: int = 0
    fetched: int = 0
    empty: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    failed_months: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_fetch(
    cfg: Config,
    *,
    force: bool = False,
    limit: int | None = None,
    start: date | None = None,
    end: date | None = None,
    session: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> FetchSummary:
    """Fetch the configured catalog into ``data/raw/``.

    Idempotent: months already recorded as complete are skipped unless
    ``force``. The manifest is saved after every month so an interrupted run
    resumes without refetching.
    """
    start = start or cfg.start_date
    end = end or cfg.end_date

    windows = month_windows(start, end)
    if limit is not None:
        windows = windows[:limit]

    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)
    months: dict[str, Any] = manifest.setdefault("months", {})

    base_params: dict[str, Any] = {"minmagnitude": cfg.min_magnitude}
    owns_session = session is None
    if owns_session:
        session = requests.Session()
        session.headers.update({"User-Agent": "SeismoAI/0.1 (portfolio research)"})

    summary = FetchSummary(total=len(windows))

    try:
        for index, (w_start, w_end) in enumerate(windows, start=1):
            label = month_label(w_start)
            path = month_path(raw_dir, label)
            record = months.get(label)

            if not force and record and record.get("status") in _OK_STATUSES and path.is_file():
                summary.skipped += 1
                if on_progress:
                    on_progress(f"[{index}/{len(windows)}] {label} — up to date, skipping")
                continue

            request_params = {**base_params, "starttime": w_start.isoformat(),
                              "endtime": w_end.isoformat()}
            attempts = int(record.get("attempts", 0)) + 1 if record else 1
            if on_progress:
                on_progress(f"[{index}/{len(windows)}] {label} — fetching…")

            try:
                frame = fetch_window(session, w_start, w_end, base_params=base_params, cfg=cfg)
            except ApiError as exc:
                months[label] = asdict(MonthRecord(
                    month=label, status="failed", attempts=attempts,
                    error=str(exc), request=request_params,
                ))
                save_manifest(raw_dir, manifest)
                summary.failed += 1
                summary.failed_months.append(label)
                if on_progress:
                    on_progress(f"[{index}/{len(windows)}] {label} — FAILED: {exc}")
                continue

            digest = atomic_write_parquet(frame, path)
            status = "empty" if frame.empty else "ok"
            months[label] = asdict(MonthRecord(
                month=label, status=status, rows=int(len(frame)), sha256=digest,
                fetched_at=_now_iso(), attempts=attempts, request=request_params,
            ))
            save_manifest(raw_dir, manifest)

            summary.rows += int(len(frame))
            if status == "empty":
                summary.empty += 1
            else:
                summary.fetched += 1
            if on_progress:
                on_progress(f"[{index}/{len(windows)}] {label} — {status} ({len(frame):,} rows)")
    finally:
        if owns_session:
            session.close()

    manifest["params"] = {"minmagnitude": cfg.min_magnitude,
                          "start": start.isoformat(), "end": end.isoformat()}
    save_manifest(raw_dir, manifest)
    return summary


def require_raw_data(cfg: Config) -> list[Path]:
    """Return sorted raw month files, raising a helpful error when absent."""
    files = sorted(cfg.raw_dir.glob("*.parquet"))
    if not files:
        raise MissingArtifact(
            f"No raw data found in {cfg.raw_dir}",
            hint="Run `seismo data fetch` to download the USGS catalog first.",
        )
    return files
