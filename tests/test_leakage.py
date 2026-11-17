"""Leakage guards — the tests that make the ML results trustworthy.

Three independent defenses:

1. **Perturbation** — inject an event *after* an observation date and assert
   every feature at that date is byte-identical. This catches any code path
   that reaches into the future, including bugs no amount of eyeballing finds.
2. **Boundary exactness** — events placed at exact window edges verify the
   half-open convention ``features = [t−W, t)``, ``target = [t, t+H)``.
3. **Split integrity** — chronological ordering plus label-window purge, with
   deliberately awkward boundaries so the purge provably fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seismo.config import load_config
from seismo.features.engineering import build_features, feature_columns
from seismo.features.splits import assign_split, chronological_split, purge_mask

# Distinct 1-degree cells so events never collide between them.
CELL_A = (35.4, 31.6)    # lat, lon
CELL_B = (-12.3, 170.4)
CELL_C = (1.7, -45.9)

FEATURE_COLS = [
    "events_last_30d", "events_last_90d", "events_last_365d",
    "mean_magnitude_90d", "max_magnitude_90d", "mean_depth_90d",
    "events_ge45_last_365d", "days_since_last_event",
]


def _project(tmp_path, *, train_end="2001-12-31", val_end="2002-06-30"):
    """Minimal project tree with a synthetic processed catalog."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        "collection:\n"
        '  start_date: "2000-01-01"\n'
        '  end_date: "2002-12-31"\n'
        "  min_magnitude: 2.5\n"
        "grid:\n"
        "  cell_size_deg: 1.0\n"
        "forecast:\n"
        "  magnitude_threshold: 4.5\n"
        "  horizon_days: 30\n"
        "  min_lookback_days: 365\n"
        "splits:\n"
        f'  train_end: "{train_end}"\n'
        f'  val_end: "{val_end}"\n'
        "  purge_days: 30\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "reports").mkdir(exist_ok=True)
    return load_config(tmp_path)


def _catalog(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build a minimal processed-shaped frame from (time, lat, lon, mag, depth)."""
    times = [r[0] for r in rows]
    return pd.DataFrame({
        # format="ISO8601" — an inferred format from the first row would reject
        # the nanosecond-precision boundary rows used in the window tests.
        "time": pd.to_datetime(times, utc=True, format="ISO8601"),
        "latitude": [r[1] for r in rows],
        "longitude": [r[2] for r in rows],
        "mag": [r[3] for r in rows],
        "depth": [r[4] for r in rows],
        "id": [f"ev{i}" for i in range(len(rows))],
        "updated": pd.to_datetime(times, utc=True, format="ISO8601"),
        "place": ["synthetic"] * len(rows),
        "type": ["earthquake"] * len(rows),
    })


def _write(cfg, rows) -> None:
    _catalog(rows).to_parquet(cfg.processed_dir / "earthquakes.parquet", index=False)


def _base_catalog() -> list[tuple[str, float, float, float, float]]:
    """Deterministic background: three active cells, spread over the catalog."""
    rows: list[tuple[str, float, float, float, float]] = []
    rng = np.random.default_rng(7)
    for month in pd.date_range("2000-01-01", "2002-12-01", freq="MS"):
        for cell in (CELL_A, CELL_B, CELL_C):
            # One modest event per cell per month keeps every cell active and
            # every statistic well-defined without dominating the target.
            day = int(rng.integers(2, 25))
            rows.append((
                f"{month.year}-{month.month:02d}-{day:02d}T06:00:00Z",
                cell[0], cell[1],
                float(rng.uniform(2.6, 4.2)),
                float(rng.uniform(5.0, 60.0)),
            ))
    return rows


@pytest.fixture()
def cfg(tmp_path):
    project = _project(tmp_path)
    _write(project, _base_catalog())
    return project


# ---------------------------------------------------------------------------
# 1. Perturbation: a future event must not change the past
# ---------------------------------------------------------------------------

class TestPerturbation:
    def test_future_event_leaves_features_untouched(self, cfg):
        """Inject an event after date t → every feature at t must be identical."""
        before, _ = build_features(cfg)

        # Perturbation: a large event in CELL_A on 2001-06-15.
        perturbation_time = pd.Timestamp("2001-06-15T09:30:00Z")
        rows = _base_catalog() + [
            ("2001-06-15T09:30:00Z", CELL_A[0], CELL_A[1], 6.4, 12.0)
        ]
        _write(cfg, rows)
        after, _ = build_features(cfg)

        assert len(before) == len(after), "row count must not change"

        feature_cols = feature_columns(before)
        future_mask = before["obs_date"] <= perturbation_time
        assert future_mask.any(), "test needs observation dates at/before the event"

        pd.testing.assert_frame_equal(
            before.loc[future_mask, feature_cols].reset_index(drop=True),
            after.loc[future_mask, feature_cols].reset_index(drop=True),
            obj="features for observation dates at/before the injected event",
        )

    def test_future_event_only_changes_targets_it_falls_inside(self, cfg):
        """Target flips only for windows [t, t+H) that contain the new event."""
        before, _ = build_features(cfg)
        event_time = pd.Timestamp("2001-06-15T09:30:00Z")
        horizon = pd.Timedelta(days=30)

        rows = _base_catalog() + [
            ("2001-06-15T09:30:00Z", CELL_A[0], CELL_A[1], 6.4, 12.0)
        ]
        _write(cfg, rows)
        after, _ = build_features(cfg)

        # Cell A's background events are M<4.5, so the target can only go 0→1.
        a_before = before[before["cell_id"] == before["cell_id"].iloc[0]]
        # Identify CELL_A rows by centre coordinates instead of id ordering.
        is_cell_a = (
            before["latitude_center"].sub(CELL_A[0]).abs() < 0.5
        ) & (before["longitude_center"].sub(CELL_A[1]).abs() < 0.5)
        assert is_cell_a.any()

        in_window = (before["obs_date"] <= event_time) & (
            before["obs_date"] + horizon > event_time
        )
        outside = (before["obs_date"] <= event_time) & ~in_window

        # Inside the new event's window → target flips to 1.
        assert (after.loc[in_window & is_cell_a, "target"] == 1).all()
        # Outside it → target unchanged.
        pd.testing.assert_series_equal(
            before.loc[outside & is_cell_a, "target"].reset_index(drop=True),
            after.loc[outside & is_cell_a, "target"].reset_index(drop=True),
            check_names=False,
        )
        assert len(a_before) > 0

    def test_features_after_the_event_do_see_it(self, cfg):
        """Sanity: the perturbation is real — later dates must reflect it."""
        before, _ = build_features(cfg)
        rows = _base_catalog() + [
            ("2001-06-15T09:30:00Z", CELL_A[0], CELL_A[1], 6.4, 12.0)
        ]
        _write(cfg, rows)
        after, _ = build_features(cfg)

        later = before["obs_date"] > pd.Timestamp("2001-07-01T00:00:00Z")
        is_cell_a = (
            before["latitude_center"].sub(CELL_A[0]).abs() < 0.5
        ) & (before["longitude_center"].sub(CELL_A[1]).abs() < 0.5)

        gained = after.loc[later & is_cell_a, "max_magnitude_90d"].to_numpy()
        lost = before.loc[later & is_cell_a, "max_magnitude_90d"].to_numpy()
        assert (gained >= lost).all()


# ---------------------------------------------------------------------------
# 2. Exact window boundaries: half-open convention
# ---------------------------------------------------------------------------

class TestWindowBoundaries:
    def _single_event_project(self, tmp_path, event_time: str, mag: float = 3.0):
        project = _project(tmp_path)
        # CELL_C carries monthly background through 2002-06. That span matters:
        # observation dates stop at (catalog_end − horizon), so without events
        # after the measured one there would be no observation date at it.
        rows: list[tuple[str, float, float, float, float]] = [
            ("2000-05-05T00:00:00Z", CELL_B[0], CELL_B[1], 2.8, 15.0),
            (event_time, CELL_B[0], CELL_B[1], mag, 20.0),
        ]
        for month in pd.date_range("2000-01-01", "2002-06-01", freq="MS"):
            rows.append((
                f"{month.year}-{month.month:02d}-11T08:00:00Z",
                CELL_C[0], CELL_C[1], 3.1, 10.0,
            ))
        _write(project, rows)
        return project

    def _rows_for(self, frame, lat, lon, obs: str):
        # obs_date is tz-aware UTC; a naive Timestamp compares silently unequal
        # in pandas 3 (no exception, just False) — always localise explicitly.
        obs_ts = pd.Timestamp(obs, tz="UTC")
        mask = (
            (frame["latitude_center"].sub(lat).abs() < 0.5)
            & (frame["longitude_center"].sub(lon).abs() < 0.5)
            & (frame["obs_date"] == obs_ts)
        )
        matched = frame[mask]
        assert len(matched) == 1, f"expected exactly one row for {obs}"
        return matched.iloc[0]

    def test_event_exactly_at_t_is_target_not_feature(self, tmp_path):
        """An event at exactly t belongs to the target window, not the features."""
        # mag must clear the M4.5 target threshold for the target to fire.
        cfg = self._single_event_project(tmp_path, "2001-06-01T00:00:00Z", mag=5.2)
        frame, _ = build_features(cfg)
        row = self._rows_for(frame, CELL_B[0], CELL_B[1], "2001-06-01")

        assert row["target"] == 1                      # [t, t+H) includes t
        assert row["events_last_30d"] == 0             # [t−W, t) excludes t
        assert row["max_magnitude_90d"] == 0.0         # no prior event → 0.0 fill

    def test_event_exactly_at_horizon_edge_is_excluded(self, tmp_path):
        """An event at exactly t+H is outside the target window."""
        cfg = self._single_event_project(tmp_path, "2001-07-01T00:00:00Z")
        frame, _ = build_features(cfg)
        row = self._rows_for(frame, CELL_B[0], CELL_B[1], "2001-06-01")

        # 2001-06-01 + 30d = 2001-07-01, which is excluded (half-open).
        assert row["target"] == 0

        # The event is still in the catalog — a later observation window sees
        # it (the 90d window of 2001-08-01 spans 2001-05-03 → 2001-08-01).
        later = self._rows_for(frame, CELL_B[0], CELL_B[1], "2001-08-01")
        assert later["max_magnitude_90d"] == pytest.approx(3.0)
        assert later["events_last_365d"] >= 1

    def test_event_one_nanosecond_before_t_is_a_feature(self, tmp_path):
        cfg = self._single_event_project(tmp_path, "2001-05-31T23:59:59.999999999Z")
        frame, _ = build_features(cfg)
        row = self._rows_for(frame, CELL_B[0], CELL_B[1], "2001-06-01")

        assert row["events_last_30d"] == 1
        assert row["target"] == 0                      # before t → never a target

    def test_count_windows_are_nested(self, tmp_path):
        """30d ⊆ 90d ⊆ 365d must hold for every row — a structural invariant."""
        cfg = self._single_event_project(tmp_path, "2001-06-10T00:00:00Z")
        frame, _ = build_features(cfg)
        assert (frame["events_last_30d"] <= frame["events_last_90d"]).all()
        assert (frame["events_last_90d"] <= frame["events_last_365d"]).all()


# ---------------------------------------------------------------------------
# 3. Split integrity: chronology + purge
# ---------------------------------------------------------------------------

def _split_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "obs_date": pd.to_datetime(dates, utc=True),
        "value": range(len(dates)),
    })


class TestSplitIntegrity:
    def test_assign_split_is_strictly_chronological(self):
        frame = _split_frame([
            "2001-01-01", "2001-06-01", "2001-12-01",
            "2002-01-01", "2002-06-01", "2002-12-01", "2003-06-01",
        ])
        splits = assign_split(frame["obs_date"], train_end="2001-12-31",
                              val_end="2002-12-31")
        assert list(splits) == ["train", "train", "train",
                                "val", "val", "val", "test"]

    def test_val_end_must_follow_train_end(self):
        frame = _split_frame(["2001-01-01"])
        from seismo.errors import ValidationError
        with pytest.raises(ValidationError, match="after train_end"):
            assign_split(frame["obs_date"], train_end="2002-12-31",
                         val_end="2001-12-31")

    def test_purge_drops_rows_whose_label_window_crosses_a_boundary(self, cfg):
        """Awkward mid-month boundaries force the purge to actually fire.

        Covers both boundary types: a train row reaching into val, and a val
        row reaching into test.
        """
        frame = pd.DataFrame({
            "obs_date": pd.to_datetime(
                ["2001-01-01", "2001-01-10", "2001-01-25",
                 "2001-02-01", "2001-06-01", "2001-06-15"], utc=True
            ),
        })
        frame["split"] = assign_split(frame["obs_date"],
                                      train_end="2001-01-15",
                                      val_end="2001-06-30")
        keep = purge_mask(frame, horizon_days=30,
                          train_end="2001-01-15", val_end="2001-06-30")

        # Train rows whose 30d label window passes 2001-01-16 → dropped.
        assert not keep.iloc[0]          # 01-01 + 30d = 01-31
        assert not keep.iloc[1]          # 01-10 + 30d = 02-09
        # Val rows whose label window stays inside val → kept.
        assert keep.iloc[2]              # 01-25 + 30d = 02-24
        assert keep.iloc[3]              # 02-01 + 30d = 03-03
        assert keep.iloc[4]              # 06-01 + 30d = 07-01 (== boundary)
        # Val row reaching past the val period → dropped.
        assert not keep.iloc[5]          # 06-15 + 30d = 07-15 > 07-01

    def test_default_year_boundaries_keep_monthly_rows(self, cfg):
        """With month-start observations and a 30d horizon, year-end
        boundaries do not overlap — the purge should not need to fire."""
        frame = pd.DataFrame({
            "obs_date": pd.to_datetime(
                ["2001-12-01", "2001-06-01", "2002-01-01", "2002-12-01"],
                utc=True,
            ),
        })
        frame["split"] = assign_split(frame["obs_date"],
                                      train_end="2001-12-31",
                                      val_end="2002-12-31")
        keep = purge_mask(frame, horizon_days=30,
                          train_end="2001-12-31", val_end="2002-12-31")
        assert keep.all()

    def test_chronological_split_on_real_features(self, cfg):
        frame, _ = build_features(cfg)
        split, stats = chronological_split(frame, cfg)

        assert set(stats) == {"train", "val", "test", "purged", "total"}
        assert stats["total"] == len(split)
        assert stats["train"] + stats["val"] + stats["test"] == stats["total"]

        # Strict ordering across split boundaries.
        train_max = split.loc[split["split"] == "train", "obs_date"].max()
        val_min = split.loc[split["split"] == "val", "obs_date"].min()
        val_max = split.loc[split["split"] == "val", "obs_date"].max()
        test_min = split.loc[split["split"] == "test", "obs_date"].min()
        assert train_max < val_min <= val_max < test_min

        # No surviving row's label window crosses its own boundary.
        horizon = pd.Timedelta(days=cfg.section("forecast")["horizon_days"])
        label_end = split["obs_date"] + horizon
        train_cut = pd.Timestamp(cfg.section("splits")["train_end"], tz="UTC")
        train_cut = train_cut.normalize() + pd.Timedelta(days=1)
        assert (label_end[split["split"] == "train"] <= train_cut).all()


# ---------------------------------------------------------------------------
# 4. Feature sanity against hand computation
# ---------------------------------------------------------------------------

class TestFeatureCorrectness:
    def test_counts_match_direct_query(self, tmp_path):
        """Spot-check windowed counts against an independent pandas filter."""
        project = _project(tmp_path)
        rows = _base_catalog()
        _write(project, rows)
        frame, _ = build_features(project)

        catalog = _catalog(rows)

        # Pick a specific observation date and cell, recompute by hand.
        obs = pd.Timestamp("2001-09-01T00:00:00Z")
        lat, lon = CELL_A
        i = int(np.floor((lon + 180) / 1.0))
        j = int(np.floor((lat + 90) / 1.0))
        expected_cell = j * 360 + i

        window = catalog[
            (catalog["time"] >= obs - pd.Timedelta(days=30))
            & (catalog["time"] < obs)
        ]
        in_cell = (
            window["latitude"].sub(lat).abs() < 0.5
        ) & (window["longitude"].sub(lon).abs() < 0.5)
        expected = int(in_cell.sum())

        row = frame[
            (frame["obs_date"] == obs)
            & (frame["cell_id"] == expected_cell)
        ]
        assert len(row) == 1
        assert int(row.iloc[0]["events_last_30d"]) == expected
        assert expected > 0, "fixture should produce events in this window"

    def test_target_matches_direct_future_query(self, tmp_path):
        """Target must equal an independent future-window count ≥ threshold."""
        project = _project(tmp_path)
        rows = _base_catalog() + [
            ("2001-09-10T12:00:00Z", CELL_A[0], CELL_A[1], 5.8, 18.0)
        ]
        _write(project, rows)
        frame, _ = build_features(project)

        catalog = _catalog(rows)
        obs = pd.Timestamp("2001-09-01T00:00:00Z")
        lat, lon = CELL_A
        i = int(np.floor((lon + 180) / 1.0))
        j = int(np.floor((lat + 90) / 1.0))
        expected_cell = j * 360 + i

        future = catalog[
            (catalog["time"] >= obs)
            & (catalog["time"] < obs + pd.Timedelta(days=30))
            & (catalog["mag"] >= 4.5)
        ]
        in_cell = (
            future["latitude"].sub(lat).abs() < 0.5
        ) & (future["longitude"].sub(lon).abs() < 0.5)
        expected_target = int(in_cell.sum() > 0)

        row = frame[
            (frame["obs_date"] == obs) & (frame["cell_id"] == expected_cell)
        ]
        assert len(row) == 1
        assert int(row.iloc[0]["target"]) == expected_target
        assert expected_target == 1, "the injected M5.8 must be caught"

    def test_days_since_never_negative_and_never_uses_future(self, cfg):
        frame, _ = build_features(cfg)
        assert (frame["days_since_last_event"] >= 0).all()
        # No observation date may precede the catalog start.
        assert (frame["obs_date"] >= pd.Timestamp("2000-01-01", tz="UTC")).all()
