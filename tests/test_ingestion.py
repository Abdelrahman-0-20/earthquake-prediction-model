"""Ingestion tests: chunking, pagination, retries, manifest idempotency."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from conftest import FakeResponse, FakeSession, csv_body

from seismo.data.ingestion import (
    load_manifest,
    month_windows,
    parse_csv,
    request_csv,
    run_fetch,
)
from seismo.errors import ApiError

# ---------------------------------------------------------------------------
# Time chunking
# ---------------------------------------------------------------------------

class TestMonthWindows:
    def test_covers_range_without_gaps_or_overlaps(self):
        windows = month_windows(date(2000, 1, 15), date(2000, 4, 10))
        # Starts are aligned to month beginnings (chunking is by month).
        assert windows[0] == (date(2000, 1, 1), date(2000, 2, 1))
        # Contiguous: each window's end == next window's start.
        for (_, end), (start, _) in zip(windows, windows[1:], strict=False):
            assert end == start
        # Final window includes the inclusive end date.
        assert windows[-1][1] == date(2000, 4, 11)

    def test_leap_february(self):
        windows = month_windows(date(2024, 2, 1), date(2024, 2, 29))
        assert windows == [(date(2024, 2, 1), date(2024, 3, 1))]

    def test_year_boundary(self):
        windows = month_windows(date(2023, 12, 1), date(2024, 1, 31))
        assert windows == [
            (date(2023, 12, 1), date(2024, 1, 1)),
            (date(2024, 1, 1), date(2024, 2, 1)),
        ]

    def test_single_day(self):
        assert month_windows(date(2024, 6, 15), date(2024, 6, 15)) == [
            (date(2024, 6, 1), date(2024, 6, 16))
        ]

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="before start"):
            month_windows(date(2024, 2, 1), date(2024, 1, 1))


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class TestParseCsv:
    def test_empty_body_returns_canonical_columns(self):
        frame = parse_csv("")
        assert list(frame.columns) == [
            "time", "latitude", "longitude", "depth", "mag", "magType", "nst",
            "gap", "dmin", "rms", "net", "id", "updated", "place", "type",
            "horizontalError", "depthError", "magError", "magNst", "status",
            "locationSource", "magSource",
        ]
        assert frame.empty

    def test_parses_quoted_place_field(self):
        body = csv_body([
            {"time": "2024-01-02T00:00:00.000Z", "latitude": 35.0, "longitude": 31.0,
             "depth": 10.0, "mag": 4.1, "id": "ev1", "place": "3 km N of Cairo, Egypt",
             "type": "earthquake", "updated": "2024-01-03T00:00:00.000Z"},
        ])
        frame = parse_csv(body)
        assert len(frame) == 1
        assert frame.loc[0, "place"] == "3 km N of Cairo, Egypt"


# ---------------------------------------------------------------------------
# HTTP retries
# ---------------------------------------------------------------------------

class TestRequestCsv:
    def _params(self):
        return {"minmagnitude": 2.5, "format": "csv"}

    def test_returns_body_on_200(self):
        session = FakeSession([FakeResponse(200, "a,b\n1,2")])
        assert request_csv(session, self._params(), url="u", timeout=1,
                           max_retries=3, backoff_base=0) == "a,b\n1,2"

    def test_204_means_no_events(self):
        session = FakeSession([FakeResponse(204, "")])
        assert request_csv(session, self._params(), url="u", timeout=1,
                           max_retries=3, backoff_base=0) == ""

    def test_retries_429_then_succeeds(self):
        session = FakeSession([FakeResponse(429, "slow down"),
                               FakeResponse(200, "ok")])
        body = request_csv(session, self._params(), url="u", timeout=1,
                           max_retries=3, backoff_base=0)
        assert body == "ok"
        assert len(session.calls) == 2

    def test_retries_503_then_succeeds(self):
        session = FakeSession([FakeResponse(503, ""), FakeResponse(200, "ok")])
        assert request_csv(session, self._params(), url="u", timeout=1,
                           max_retries=3, backoff_base=0) == "ok"

    def test_gives_up_after_max_retries(self):
        session = FakeSession([FakeResponse(500, "") for _ in range(3)])
        with pytest.raises(ApiError, match="after 3 attempts"):
            request_csv(session, self._params(), url="u", timeout=1,
                        max_retries=3, backoff_base=0)

    def test_client_error_fails_fast_without_retry(self):
        session = FakeSession([FakeResponse(400, "Bad Request: bad time")])
        with pytest.raises(ApiError, match="rejected the request"):
            request_csv(session, self._params(), url="u", timeout=1,
                        max_retries=5, backoff_base=0)
        # No pointless retries on a request the API will never accept.
        assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Fetch orchestration (idempotency, pagination, failure handling)
# ---------------------------------------------------------------------------

class TestRunFetch:
    def _rows(self, n: int, prefix: str) -> list[dict]:
        return [
            {"time": "2024-01-02T00:00:00.000Z", "latitude": 10.0, "longitude": 20.0,
             "depth": 5.0, "mag": 3.0, "id": f"{prefix}{i}",
             "updated": "2024-01-03T00:00:00.000Z", "type": "earthquake",
             "place": "X"}
            for i in range(n)
        ]

    def test_fetch_writes_month_files_and_manifest(self, cfg):
        session = FakeSession([FakeResponse(200, csv_body(self._rows(3, "a")))])
        summary = run_fetch(cfg, limit=1, session=session)

        assert summary.fetched == 1
        assert summary.failed == 0
        assert (cfg.raw_dir / "2024-01.parquet").is_file()

        manifest = load_manifest(cfg.raw_dir)
        record = manifest["months"]["2024-01"]
        assert record["status"] == "ok"
        assert record["rows"] == 3
        assert len(record["sha256"]) == 64

    def test_rerun_skips_completed_months(self, cfg):
        session = FakeSession([FakeResponse(200, csv_body(self._rows(3, "a")))])
        run_fetch(cfg, limit=1, session=session)
        assert len(session.calls) == 1

        # Second run: nothing to fetch.
        second = FakeSession([])
        summary = run_fetch(cfg, limit=1, session=second)
        assert summary.skipped == 1
        assert summary.fetched == 0
        assert second.calls == []

    def test_force_refetches(self, cfg):
        run_fetch(cfg, limit=1,
                  session=FakeSession([FakeResponse(200, csv_body(self._rows(2, "a")))]))
        forced = FakeSession([FakeResponse(200, csv_body(self._rows(5, "b")))])
        summary = run_fetch(cfg, limit=1, force=True, session=forced)
        assert summary.fetched == 1

        manifest = load_manifest(cfg.raw_dir)
        assert manifest["months"]["2024-01"]["rows"] == 5

    def test_failed_month_is_recorded_and_run_continues(self, cfg):
        # Month 1 fails permanently, month 2 and 3 succeed.
        session = FakeSession([
            FakeResponse(500, ""),
            FakeResponse(500, ""),
            FakeResponse(200, csv_body(self._rows(4, "b"))),   # 2024-02
            FakeResponse(200, csv_body(self._rows(6, "c"))),   # 2024-03
        ])
        summary = run_fetch(cfg, session=session)

        assert summary.failed == 1
        assert summary.failed_months == ["2024-01"]
        assert summary.fetched == 2

        manifest = load_manifest(cfg.raw_dir)
        assert manifest["months"]["2024-01"]["status"] == "failed"
        assert not (cfg.raw_dir / "2024-01.parquet").exists()
        assert (cfg.raw_dir / "2024-02.parquet").is_file()

    def test_limit_restricts_months(self, cfg):
        session = FakeSession([FakeResponse(200, csv_body(self._rows(1, "a")))])
        summary = run_fetch(cfg, limit=1, session=session)
        assert summary.total == 1
        assert len(session.calls) == 1

    def test_empty_month_recorded_as_empty(self, cfg):
        session = FakeSession([FakeResponse(204, "")])
        summary = run_fetch(cfg, limit=1, session=session)
        assert summary.empty == 1
        manifest = load_manifest(cfg.raw_dir)
        assert manifest["months"]["2024-01"]["status"] == "empty"

    def test_pagination_when_response_hits_cap(self, cfg, monkeypatch):
        """A month at the 20k cap must be fetched in multiple pages."""

        cfg.collection["max_rows_per_request"] = 2  # shrink the cap for the test

        page1 = csv_body(self._rows(2, "p1"))
        page2 = csv_body(self._rows(2, "p2"))
        page3 = csv_body(self._rows(1, "p3"))
        session = FakeSession([
            FakeResponse(200, page1),
            FakeResponse(200, page2),
            FakeResponse(200, page3),
        ])
        summary = run_fetch(cfg, limit=1, session=session)

        assert summary.fetched == 1
        frame = pd.read_parquet(cfg.raw_dir / "2024-01.parquet")
        assert len(frame) == 5
        # USGS offsets are 1-based: page N starts at 1 + N*page_size.
        offsets = [call["params"]["offset"] for call in session.calls]
        assert offsets == [1, 3, 5]

    def test_manifest_survives_as_json(self, cfg):
        run_fetch(cfg, limit=1, session=FakeSession([FakeResponse(204, "")]))
        raw = json.loads((cfg.raw_dir / "manifest.json").read_text())
        assert raw["version"] == 1
        assert "2024-01" in raw["months"]
