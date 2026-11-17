"""Feature engineering for Module 02: cell × month prediction points.

Problem statement
-----------------
> Given historical seismic activity in a geographic grid cell, estimate the
> probability that at least one earthquake of magnitude ≥ 4.5 occurs in that
> cell during the following 30 days.

Temporal contract (the whole point of this module)
--------------------------------------------------
For an observation date ``t``:

* **Features** use events with ``event_time ∈ [t − W, t)`` — strictly *before*
  ``t``, so they are information available at prediction time.
* **Target** uses events with ``event_time ∈ [t, t + H)`` — strictly *after*
  ``t``.

The two windows are disjoint and cover ``(t−W, t+H)`` with no overlap, so a
feature can never see an event that contributes to its own target. This is
enforced by ``tests/test_leakage.py``, which includes a perturbation test: an
event injected after ``t`` must leave every feature at ``t`` unchanged.

Implementation notes
--------------------
Events are sorted by time once; each observation date then slices that sorted
array with ``np.searchsorted``, so windowing is exact (no month-boundary
approximation) and costs O(log n) per date. ``days_since_last_event`` is
maintained incrementally with a single forward pointer, giving O(total events)
overall instead of O(dates × events).

Missing-window convention: cells with no events in a statistic's window get
``0.0`` for that statistic. Magnitude never equals 0 in this catalog
(min 0.7), and the count features disambiguate — so ``0.0`` reads as
"nothing happened" and contributes no signal, rather than NaN which would
break linear models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.preprocessing import load_processed
from ..errors import ValidationError
from ..geospatial.analysis import grid_from_config

log = logging.getLogger("seismo.features")

DAY_NS = 86_400 * 1_000_000_000  # nanoseconds in a day

#: Trailing windows (days) → output column names.
FEATURE_WINDOWS: dict[int, str] = {
    30: "events_last_30d",
    90: "events_last_90d",
    365: "events_last_365d",
}


@dataclass
class FeatureReport:
    """Build summary — also the basis of the data dictionary checks."""

    rows: int = 0
    active_cells: int = 0
    observation_dates: int = 0
    first_obs_date: str = ""
    last_obs_date: str = ""
    positive_rows: int = 0
    positive_rate: float = 0.0
    magnitude_threshold: float = 0.0
    horizon_days: int = 0
    lookback_days: int = 0
    cell_size_deg: float = 1.0
    dropped_short_lookback: int = 0
    dropped_short_horizon: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def observation_dates(
    catalog_start: pd.Timestamp,
    catalog_end: pd.Timestamp,
    *,
    lookback_days: int,
    horizon_days: int,
) -> tuple[pd.DatetimeIndex, int, int]:
    """Month-start observation dates with a full lookback *and* horizon.

    Returns ``(dates, n_too_early, n_too_late)`` — the two counts are reported
    so a shrinking catalog visibly shrinks the dataset instead of silently
    producing short windows.
    """
    if catalog_end <= catalog_start:
        raise ValidationError(
            f"Catalog end ({catalog_end}) is not after start ({catalog_start})",
            hint="Run `seismo data fetch` to populate data/processed/.",
        )

    earliest = catalog_start + timedelta(days=lookback_days)
    latest = catalog_end - timedelta(days=horizon_days)

    all_months = pd.date_range(
        catalog_start.replace(day=1), catalog_end.replace(day=1), freq="MS", tz="UTC"
    )
    valid = all_months[(all_months >= earliest) & (all_months <= latest)]
    if valid.empty:
        raise ValidationError(
            f"Catalog span {(catalog_end - catalog_start).days} days is shorter "
            f"than lookback ({lookback_days}) + horizon ({horizon_days}) days",
            hint="Extend the collection window in config/config.yaml or reduce "
                 "forecast.lookback_days.",
        )
    too_early = int((all_months < earliest).sum())
    too_late = int((all_months > latest).sum())
    return valid, too_early, too_late


def build_features(
    cfg: Config,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Build the (cell, month) dataset with leak-free features and target."""
    forecast = cfg.section("forecast")
    horizon_days = int(forecast.get("horizon_days", 30))
    lookback_days = int(forecast.get("min_lookback_days", 365))
    mag_threshold = float(forecast.get("magnitude_threshold", 4.5))

    grid = grid_from_config(cfg)
    events = load_processed(cfg)
    if events.empty:
        raise ValidationError(
            "Processed dataset is empty",
            hint="Run `seismo data fetch` then `seismo data validate`.",
        )

    # --- cells ----------------------------------------------------------
    events = events.copy()
    events["cell_id"] = grid.assign(events["latitude"].to_numpy(),
                                    events["longitude"].to_numpy())
    active_cells = np.unique(events["cell_id"].to_numpy())
    n_active = int(active_cells.size)

    # --- observation dates ---------------------------------------------
    catalog_start = pd.Timestamp(cfg.start_date, tz="UTC")
    catalog_end = pd.Timestamp(events["time"].max())
    obs_dates, too_early, too_late = observation_dates(
        catalog_start, catalog_end,
        lookback_days=lookback_days, horizon_days=horizon_days,
    )
    n_obs = len(obs_dates)

    if on_progress:
        on_progress(
            f"{n_obs} observation dates × {n_active:,} active cells "
            f"→ {n_obs * n_active:,} rows"
        )

    # --- sort once; windowing is exact against this ordering -------------
    # pandas may store datetime64 at 's', 'ms' or 'us' resolution depending on
    # the source, while Timestamp.value / our day constants are nanoseconds.
    # Normalise everything to nanoseconds here so the arithmetic below can
    # never mix units — a silent unit mismatch turns a 30-day window into
    # "all of history".
    events = events.sort_values("time", kind="stable").reset_index(drop=True)
    t_ns = events["time"].dt.as_unit("ns").astype("int64").to_numpy()
    cell = events["cell_id"].to_numpy()
    mag = events["mag"].to_numpy(dtype=np.float64)
    depth = events["depth"].to_numpy(dtype=np.float64)
    n_events = t_ns.size

    obs_ns = obs_dates.as_unit("ns").astype("int64").to_numpy()
    horizon_ns = horizon_days * DAY_NS
    window_ns = {w: w * DAY_NS for w in FEATURE_WINDOWS}
    catalog_start_ns = int(catalog_start.value)

    # --- preallocate output blocks --------------------------------------
    n_rows = n_obs * n_active
    counts = {w: np.zeros(n_rows, dtype=np.int32) for w in FEATURE_WINDOWS}
    mean_mag_90 = np.zeros(n_rows, dtype=np.float64)
    max_mag_90 = np.zeros(n_rows, dtype=np.float64)
    mean_depth_90 = np.zeros(n_rows, dtype=np.float64)
    ge45_365 = np.zeros(n_rows, dtype=np.int32)
    days_since = np.zeros(n_rows, dtype=np.float64)
    target = np.zeros(n_rows, dtype=np.int8)

    # Incremental "last event before t" state (nanoseconds since epoch).
    INT64_MIN = np.iinfo(np.int64).min
    last_ns = np.full(grid.n_cells, INT64_MIN, dtype=np.int64)
    pointer = 0

    # --- main loop: one pass per observation date ------------------------
    for obs_idx, t in enumerate(obs_ns):
        base = obs_idx * n_active

        # Advance the pointer: every event strictly before t updates its cell.
        while pointer < n_events and t_ns[pointer] < t:
            last_ns[cell[pointer]] = t_ns[pointer]
            pointer += 1

        # Features: events in [t − W, t)
        i_end = int(np.searchsorted(t_ns, t, side="left"))
        for window in FEATURE_WINDOWS:
            i_start = int(np.searchsorted(t_ns, t - window_ns[window], side="left"))
            if i_end <= i_start:
                continue  # no events in this window → zeros (correct default)
            sl = slice(i_start, i_end)
            pos = np.searchsorted(active_cells, cell[sl])
            counts[window][base:base + n_active] = np.bincount(
                pos, minlength=n_active
            ).astype(np.int32)

            if window == 90:
                seg_mag = mag[sl]
                seg_depth = depth[sl]
                sum_mag = np.bincount(pos, weights=seg_mag, minlength=n_active)
                sum_depth = np.bincount(pos, weights=seg_depth, minlength=n_active)
                n90 = counts[90][base:base + n_active].astype(np.float64)
                nonzero = n90 > 0
                mean_mag_90[base:base + n_active] = np.where(
                    nonzero, sum_mag / np.where(nonzero, n90, 1), 0.0
                )
                mean_depth_90[base:base + n_active] = np.where(
                    nonzero, sum_depth / np.where(nonzero, n90, 1), 0.0
                )
                # Max needs an actual reduction per cell — pandas groupby
                # does this in C and stays readable.
                sub = events.iloc[sl]
                mx = sub.groupby("cell_id")["mag"].max()
                idx = np.searchsorted(active_cells, mx.index.to_numpy())
                max_mag_90[base + idx] = mx.to_numpy(dtype=np.float64)

        # Trailing count of *target-class* events (same class we predict).
        i_ge_start = int(np.searchsorted(t_ns, t - 365 * DAY_NS, side="left"))
        if i_end > i_ge_start:
            sl = slice(i_ge_start, i_end)
            seg_cell = cell[sl]
            seg_mag = mag[sl]
            hits = seg_cell[seg_mag >= mag_threshold]
            if hits.size:
                pos = np.searchsorted(active_cells, hits)
                ge45_365[base:base + n_active] = np.bincount(
                    pos, minlength=n_active
                ).astype(np.int32)

        # days_since_last_event (never observed → days since catalog start)
        prior = last_ns[active_cells]
        has_prior = prior != INT64_MIN
        days_since[base:base + n_active] = np.where(
            has_prior,
            (t - prior) / DAY_NS,
            (t - catalog_start_ns) / DAY_NS,
        )

        # Target: events in [t, t + H) with mag ≥ threshold
        i_t1 = int(np.searchsorted(t_ns, t + horizon_ns, side="left"))
        if i_t1 > i_end:
            sl = slice(i_end, i_t1)
            seg_cell = cell[sl]
            seg_mag = mag[sl]
            hits = seg_cell[seg_mag >= mag_threshold]
            if hits.size:
                pos = np.searchsorted(active_cells, hits)
                target[base:base + n_active] = (
                    np.bincount(pos, minlength=n_active) > 0
                ).astype(np.int8)

        if on_progress and (obs_idx + 1) % 50 == 0:
            on_progress(f"  {obs_idx + 1}/{n_obs} observation dates…")

    # --- assemble --------------------------------------------------------
    lat_c, lon_c = grid.centers(np.tile(active_cells, n_obs))
    frame = pd.DataFrame({
        "obs_date": np.repeat(obs_dates, n_active),
        "cell_id": np.tile(active_cells, n_obs),
        "latitude_center": lat_c,
        "longitude_center": lon_c,
        "events_last_30d": counts[30],
        "events_last_90d": counts[90],
        "events_last_365d": counts[365],
        "mean_magnitude_90d": mean_mag_90,
        "max_magnitude_90d": max_mag_90,
        "mean_depth_90d": mean_depth_90,
        "events_ge45_last_365d": ge45_365,
        "days_since_last_event": days_since,
        "target": target,
    })

    positives = int(frame["target"].sum())
    report = FeatureReport(
        rows=len(frame),
        active_cells=n_active,
        observation_dates=n_obs,
        first_obs_date=str(obs_dates[0].date()),
        last_obs_date=str(obs_dates[-1].date()),
        positive_rows=positives,
        positive_rate=round(positives / len(frame), 6) if len(frame) else 0.0,
        magnitude_threshold=mag_threshold,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
        cell_size_deg=grid.cell_size_deg,
        dropped_short_lookback=too_early,
        dropped_short_horizon=too_late,
        notes=[
            "Features use events in [t-W, t); target uses events in [t, t+H). "
            "The windows are disjoint — see tests/test_leakage.py.",
            "Statistic windows with no events are 0.0 (magnitude never equals 0 "
            "in this catalog); the count features disambiguate.",
            "days_since_last_event falls back to 'days since catalog start' "
            "when a cell has no prior event.",
        ],
    )
    return frame, report


#: Columns that are never model input: identifiers, the label itself, and the
#: split assignment. Including `split` would hand the model the temporal
#: partition as a feature — a leak that silently inflates every metric.
NON_FEATURE_COLUMNS = {"obs_date", "cell_id", "target", "split"}

#: Cell coordinates. Included by default (they carry strong, legitimate
#: location signal), settable for an ablation using temporal features only.
COORDINATE_COLUMNS = ("latitude_center", "longitude_center")


def feature_columns(
    frame: pd.DataFrame, *, include_coordinates: bool = True
) -> list[str]:
    """Ordered model-input columns.

    Excludes identifiers, the target, and the split label. Coordinates are
    included by default because seismicity is strongly location-dependent;
    pass ``include_coordinates=False`` for a temporal-only ablation.
    """
    exclude = set(NON_FEATURE_COLUMNS)
    if not include_coordinates:
        exclude |= set(COORDINATE_COLUMNS)
    return [c for c in frame.columns if c not in exclude]
