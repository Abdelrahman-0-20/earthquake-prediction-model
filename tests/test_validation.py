"""Validation gates: dedup, event filtering, hard checks, missingness."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from seismo.data import preprocessing as pp
from seismo.errors import MissingArtifact, ValidationError


def _write_raw(cfg, frame: pd.DataFrame, label: str = "2024-01") -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.raw_dir / f"{label}.parquet", index=False)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_keeps_latest_revision(self, sample_events):
        deduped, removed = pp.deduplicate(sample_events)
        assert removed == 1
        # ev1's newer revision (mag 4.4) must win over the older 4.1.
        ev1 = deduped[deduped["id"] == "ev1"]
        assert len(ev1) == 1
        assert float(ev1.iloc[0]["mag"]) == 4.4

    def test_no_duplicates_is_noop(self, sample_events):
        unique = sample_events.drop_duplicates(subset="id")
        deduped, removed = pp.deduplicate(unique)
        assert removed == 0
        assert len(deduped) == len(unique)

    def test_empty_frame(self):
        frame = pd.DataFrame(columns=["id", "updated"])
        deduped, removed = pp.deduplicate(frame)
        assert removed == 0
        assert deduped.empty


# ---------------------------------------------------------------------------
# Event-type filtering
# ---------------------------------------------------------------------------

class TestFilterEventTypes:
    def test_drops_non_earthquake_and_reports_counts(self, cfg, sample_events):
        filtered, dropped, counts = pp.filter_event_types(sample_events, cfg)
        assert dropped == 1
        assert "quarry blast" not in set(filtered["type"])
        assert counts["earthquake"] == 4
        assert counts["quarry blast"] == 1

    def test_all_kept_when_type_allowed(self, cfg, sample_events):
        filtered, dropped, _ = pp.filter_event_types(sample_events, cfg)
        # Only the quarry blast is dropped; the four earthquakes remain.
        assert dropped == 1
        assert len(filtered) == 4


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------

class TestCheckHard:
    def test_clean_data_passes(self, cfg, sample_events):
        cleaned, _ = pp.deduplicate(sample_events)
        assert pp.check_hard(cleaned, cfg) == []

    def test_latitude_out_of_range_fails(self, cfg, sample_events):
        bad = sample_events.copy()
        bad.loc[0, "latitude"] = 999.0
        assert any("latitude" in v for v in pp.check_hard(bad, cfg))

    def test_longitude_out_of_range_fails(self, cfg, sample_events):
        bad = sample_events.copy()
        bad.loc[0, "longitude"] = -500.0
        assert any("longitude" in v for v in pp.check_hard(bad, cfg))

    def test_unparseable_time_fails(self, cfg, sample_events):
        bad = sample_events.copy()
        bad.loc[0, "time"] = "not-a-timestamp"
        assert any("time" in v for v in pp.check_hard(bad, cfg))

    def test_non_numeric_mag_fails(self, cfg, sample_events):
        bad = sample_events.copy()
        # pandas 3 refuses str→float64 assignment; widen the dtype explicitly.
        bad["mag"] = bad["mag"].astype(object)
        bad.loc[0, "mag"] = "huge"
        assert any("mag" in v for v in pp.check_hard(bad, cfg))

    def test_duplicate_ids_fail(self, cfg, sample_events):
        # Deduplication not applied → duplicate ids remain.
        assert any("duplicate" in v for v in pp.check_hard(sample_events, cfg))

    def test_missing_id_fails(self, cfg, sample_events):
        bad = sample_events.copy()
        bad.loc[0, "id"] = None
        assert any("missing event id" in v for v in pp.check_hard(bad, cfg))

    def test_empty_frame_passes(self, cfg):
        assert pp.check_hard(pd.DataFrame(columns=["id"]), cfg) == []


# ---------------------------------------------------------------------------
# Missingness (soft — measured, never filled)
# ---------------------------------------------------------------------------

class TestMissingness:
    def test_counts_nulls_and_empty_strings(self, cfg, sample_events):
        report = pp.missingness(sample_events)
        # nst/gap/dmin/rms were never provided → 100% missing.
        for column in ("nst", "gap", "dmin", "rms"):
            assert report[column]["pct"] == 100.0, column
        # id is always present.
        assert report["id"]["missing"] == 0

    def test_percentages_sum_sensibly(self, cfg, sample_events):
        report = pp.missingness(sample_events)
        total = len(sample_events)
        for column, stats in report.items():
            assert 0 <= stats["missing"] <= total, column
            assert 0.0 <= stats["pct"] <= 100.0, column


# ---------------------------------------------------------------------------
# End-to-end processed build
# ---------------------------------------------------------------------------

class TestBuildProcessed:
    def test_end_to_end(self, cfg, sample_events):
        _write_raw(cfg, sample_events)
        report = pp.build_processed(cfg, strict=True)

        # 5 raw → -1 non-earthquake → -1 duplicate = 3 processed.
        assert report.total_raw_rows == 5
        assert report.non_earthquake_removed == 1
        assert report.duplicates_removed == 1
        assert report.processed_rows == 3
        assert report.hard_violations == []

        out = cfg.processed_dir / "earthquakes.parquet"
        assert out.is_file()
        frame = pd.read_parquet(out)

        # Typed, sorted, with derived columns.
        assert str(frame["time"].dtype).startswith("datetime64")
        assert frame["time"].is_monotonic_increasing
        assert {"year", "month"} <= set(frame.columns)
        # Negative depth retained but counted (soft check).
        assert report.negative_depth_rows == 1

        # Report written and parseable.
        quality = json.loads(
            (cfg.quality_report_dir / "data_quality.json").read_text()
        )
        assert quality["processed_rows"] == 3
        assert "nst" in quality["missingness"]

    def test_strict_mode_raises_on_bad_coordinates(self, cfg, sample_events):
        bad = sample_events.copy()
        # Row 0 is the superseded ev1 duplicate (discarded by dedup) — corrupt a
        # row that actually survives into the processed dataset.
        bad.loc[4, "latitude"] = 123.456
        _write_raw(cfg, bad)

        with pytest.raises(ValidationError, match="latitude"):
            pp.build_processed(cfg, strict=True)

    def test_strict_false_still_writes_report(self, cfg, sample_events):
        bad = sample_events.copy()
        bad.loc[4, "latitude"] = 123.456
        _write_raw(cfg, bad)

        report = pp.build_processed(cfg, strict=False)
        assert report.hard_violations
        assert (cfg.quality_report_dir / "data_quality.json").is_file()

    def test_no_raw_data_gives_actionable_error(self, cfg):
        with pytest.raises(MissingArtifact) as exc:
            pp.build_processed(cfg)
        # The actionable fix ships in the hint (rendered as "Fix: …").
        assert "seismo data fetch" in (exc.value.hint or "")

    def test_multiple_months_concatenate(self, cfg, sample_events):
        _write_raw(cfg, sample_events, label="2024-01")
        _write_raw(cfg, sample_events, label="2024-02")
        report = pp.build_processed(cfg, strict=True)
        # Duplicates across months are also removed by id.
        assert report.source_files == 2
        assert report.processed_rows == 3

    def test_below_threshold_is_measured_not_dropped(self, cfg, sample_events):
        """Catalog revisions can push a magnitude below the filter — keep and count."""
        low = sample_events.iloc[[0]].copy()
        low["id"] = "low1"
        low["mag"] = 1.2          # below the configured 2.5 threshold
        _write_raw(cfg, pd.concat([sample_events, low], ignore_index=True))

        report = pp.build_processed(cfg, strict=True)

        # Hard gate does NOT fire: this is a soft, reported condition.
        assert report.hard_violations == []
        assert report.below_threshold_rows == 1
        assert report.below_threshold_min_mag == 1.2
        # The row itself is retained in the processed dataset.
        frame = pd.read_parquet(cfg.processed_dir / "earthquakes.parquet")
        assert "low1" in set(frame["id"])
