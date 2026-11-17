"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from seismo.config import load_config
from seismo.data.ingestion import CSV_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """A minimal project tree with a config file, isolated from real data."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        "collection:\n"
        '  start_date: "2024-01-01"\n'
        '  end_date: "2024-03-31"\n'
        "  min_magnitude: 2.5\n"
        "  max_retries: 2\n"
        "  backoff_base_seconds: 0.0\n"
        "  timeout_seconds: 5\n",
        encoding="utf-8",
    )
    for sub in ("data/raw", "data/processed", "data/features",
                "reports/data_quality", "models"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def cfg(project_root: Path):
    return load_config(project_root)


def make_events(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-shaped frame; rows only need the keys they actually vary."""
    base = {column: None for column in CSV_COLUMNS}
    records = []
    for row in rows:
        record = dict(base)
        record.update(row)
        records.append(record)
    return pd.DataFrame(records, columns=CSV_COLUMNS)


@pytest.fixture()
def sample_events() -> pd.DataFrame:
    """A tiny catalog exercising dedup, filtering, and range checks."""
    return make_events(
        [
            # Good event.
            dict(id="ev1", time="2024-01-02T00:00:00.000Z",
                 updated="2024-01-03T00:00:00.000Z",
                 latitude=35.0, longitude=31.0, depth=10.0, mag=4.1,
                 type="earthquake", place="Cairo, Egypt", magType="ml"),
            # Duplicate id, newer revision with a corrected magnitude.
            dict(id="ev1", time="2024-01-02T00:00:00.000Z",
                 updated="2024-01-05T00:00:00.000Z",
                 latitude=35.0, longitude=31.0, depth=10.0, mag=4.4,
                 type="earthquake", place="Cairo, Egypt", magType="ml"),
            # Non-earthquake event — should be dropped from processed output.
            dict(id="ev2", time="2024-01-04T00:00:00.000Z",
                 updated="2024-01-04T00:00:00.000Z",
                 latitude=36.0, longitude=32.0, depth=5.0, mag=3.0,
                 type="quarry blast", place="Cairo, Egypt", magType="ml"),
            # A second good event with a negative (near-surface) depth.
            dict(id="ev3", time="2024-01-06T00:00:00.000Z",
                 updated="2024-01-06T00:00:00.000Z",
                 latitude=-10.5, longitude=170.2, depth=-1.2, mag=5.3,
                 type="earthquake", place="Tonga region", magType="mww"),
            # Missing catalog-quality fields (nst/gap/dmin/rms) — soft case.
            dict(id="ev4", time="2024-01-08T00:00:00.000Z",
                 updated="2024-01-08T00:00:00.000Z",
                 latitude=0.0, longitude=0.0, depth=30.0, mag=2.9,
                 type="earthquake", place="Mid-Atlantic", magType="mb"),
        ]
    )


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int = 200, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class FakeSession:
    """Scripted session: pops responses in order and records every request."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.headers: dict = {}

    def get(self, url: str, params: dict | None = None, timeout: float | None = None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self._responses:
            raise AssertionError(f"FakeSession exhausted; unexpected request to {url}")
        return self._responses.pop(0)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def csv_body(rows: list[dict]) -> str:
    """Serialize rows as USGS-shaped CSV text."""
    import io

    frame = pd.DataFrame(rows)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture()
def fake_session_cls():
    return FakeSession


@pytest.fixture()
def window():
    return (date(2024, 1, 1), date(2024, 2, 1))
