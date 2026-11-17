"""Geospatial grid system and historical activity aggregation.

One grid definition serves the whole project: Module 02 forecasting cells and
Module 03 hotspot analysis share the *same* cells, so a map cell and a
forecast row always refer to the same piece of the Earth.

Conventions
-----------
* Cells are ``cell_size_deg`` square, indexed from the antimeridian/south pole:
  ``i = floor((lon + 180) / size)`` (west→east), ``j = floor((lat + 90) / size)``
  (south→north).
* ``cell_id = j * n_i + i`` — a dense integer so grouping is cheap and the
  mapping back to a centre is exact.
* Coordinates on the far edge (lon = 180, lat = 90) clamp into the last cell
  instead of falling outside the grid.

Scientific note: cell activity is a *historical activity indicator*, not a
disaster-risk measure — risk additionally needs exposure and vulnerability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ValidationError

#: Named regions used for regional deep-dives and the Forecasting page.
#: Bounding boxes deliberately avoid the antimeridian (no ±180 wrap-around),
#: so a region is always a simple ``lon_min <= lon <= lon_max`` test.
NAMED_REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "Japan": {"lat": (30.0, 46.0), "lon": (129.0, 147.0)},
    "Indonesia": {"lat": (-11.5, 6.5), "lon": (93.0, 142.0)},
    "Philippines": {"lat": (4.0, 20.0), "lon": (116.0, 128.0)},
    "Papua New Guinea & Solomons": {"lat": (-12.0, 0.0), "lon": (140.0, 160.0)},
    "New Zealand": {"lat": (-48.0, -33.0), "lon": (165.0, 179.0)},
    "Chile & Peru": {"lat": (-45.0, 0.0), "lon": (-76.0, -66.0)},
    "Central America & Mexico": {"lat": (5.0, 20.0), "lon": (-105.0, -83.0)},
    "California & US West": {"lat": (31.0, 42.5), "lon": (-125.5, -113.0)},
    "Alaska": {"lat": (51.0, 71.0), "lon": (-170.0, -128.0)},
    "Türkiye & Iran": {"lat": (25.0, 42.0), "lon": (34.0, 63.0)},
    "Greece & Italy": {"lat": (34.0, 47.0), "lon": (5.0, 27.0)},
    "Hawaii": {"lat": (17.0, 23.0), "lon": (-162.0, -154.0)},
    "Mid-Atlantic Ridge": {"lat": (-60.0, 60.0), "lon": (-30.0, -10.0)},
}


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid over the whole Earth."""

    cell_size_deg: float = 1.0

    def __post_init__(self) -> None:
        size = float(self.cell_size_deg)
        if size <= 0:
            raise ValidationError(
                f"cell_size_deg must be positive, got {size}",
                hint="Set grid.cell_size_deg in config/config.yaml (e.g. 1.0).",
            )
        if not (360.0 % size == 0 and 180.0 % size == 0):
            raise ValidationError(
                f"cell_size_deg={size} must divide both 360 and 180 evenly",
                hint="Use 0.5, 1, 2, 3, 4, 5, 6, 10, 15, 20, 30, 45 or 90.",
            )

    @property
    def n_i(self) -> int:
        """Number of cells west→east (longitude)."""
        return int(round(360.0 / self.cell_size_deg))

    @property
    def n_j(self) -> int:
        """Number of cells south→north (latitude)."""
        return int(round(180.0 / self.cell_size_deg))

    @property
    def n_cells(self) -> int:
        return self.n_i * self.n_j

    # --- assignment ----------------------------------------------------
    def assign(self, latitude: Any, longitude: Any) -> np.ndarray:
        """Return dense ``cell_id`` for coordinate(s); accepts arrays or scalars."""
        lat = np.asarray(latitude, dtype=np.float64)
        lon = np.asarray(longitude, dtype=np.float64)

        if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
            raise ValidationError(
                "Non-finite latitude/longitude cannot be assigned to a grid cell",
                hint="Run `seismo data validate` — coordinate checks should "
                     "have caught this.",
            )

        i = np.floor((lon + 180.0) / self.cell_size_deg).astype(np.int64)
        j = np.floor((lat + 90.0) / self.cell_size_deg).astype(np.int64)
        # Edge coordinates (lon = 180, lat = 90) clamp into the last cell.
        np.clip(i, 0, self.n_i - 1, out=i)
        np.clip(j, 0, self.n_j - 1, out=j)
        return j * self.n_i + i

    def centers(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(latitude_center, longitude_center)`` for cell id(s)."""
        ids = np.asarray(cell_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_cells):
            raise ValidationError(
                f"cell_id out of range for a {self.cell_size_deg}° grid "
                f"(valid: 0..{self.n_cells - 1})",
                hint="Rebuild features after changing grid.cell_size_deg — "
                     "stale cell ids are incompatible.",
            )
        i = ids % self.n_i
        j = ids // self.n_i
        lon_c = -180.0 + (i + 0.5) * self.cell_size_deg
        lat_c = -90.0 + (j + 0.5) * self.cell_size_deg
        return lat_c, lon_c

    def label(self, cell_id: int) -> str:
        """Human-readable ``i,j`` label for a cell id."""
        return f"{int(cell_id) % self.n_i},{int(cell_id) // self.n_i}"


def grid_from_config(cfg: Any) -> Grid:
    """Build the project grid from ``config.yaml`` → ``grid.cell_size_deg``."""
    return Grid(float(cfg.section("grid").get("cell_size_deg", 1.0)))


def assign_cells(frame: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """Return a copy of ``frame`` with a ``cell_id`` column added."""
    out = frame.copy()
    out["cell_id"] = grid.assign(out["latitude"].to_numpy(), out["longitude"].to_numpy())
    return out


def filter_region(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter events to a named region bounding box."""
    if region not in NAMED_REGIONS:
        known = ", ".join(sorted(NAMED_REGIONS))
        raise ValidationError(
            f"Unknown region {region!r}",
            hint=f"Choose one of: {known}",
        )
    box = NAMED_REGIONS[region]
    lat_lo, lat_hi = box["lat"]
    lon_lo, lon_hi = box["lon"]
    mask = (
        frame["latitude"].between(lat_lo, lat_hi)
        & frame["longitude"].between(lon_lo, lon_hi)
    )
    return frame[mask]


#: Label used for cells outside every named region bounding box.
UNCATEGORIZED = "Other / uncategorized"


def assign_regions(latitude: Any, longitude: Any) -> np.ndarray:
    """Label each coordinate with its named region, else ``UNCATEGORIZED``.

    Used by per-region evaluation: performance must be reported regionally,
    because a pooled metric can hide systematic failure in a single region
    (the positive rate already spans 0.62%–11.98% across named regions).

    Regions are checked in definition order and a cell is labelled by the
    first box containing it. The boxes are disjoint by construction except
    where documented, so ordering is not load-bearing — but it is deterministic.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    labels = np.full(lat.shape, UNCATEGORIZED, dtype=object)

    for name, box in NAMED_REGIONS.items():
        lat_lo, lat_hi = box["lat"]
        lon_lo, lon_hi = box["lon"]
        inside = (
            (labels == UNCATEGORIZED)
            & (lat >= lat_lo) & (lat <= lat_hi)
            & (lon >= lon_lo) & (lon <= lon_hi)
        )
        labels[inside] = name
    return labels


# ---------------------------------------------------------------------------
# Historical activity aggregation (Module 03)
# ---------------------------------------------------------------------------

def cell_activity(
    events: pd.DataFrame,
    grid: Grid,
    *,
    catalog_start: pd.Timestamp | None = None,
    catalog_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-cell historical activity indicators.

    Returns one row per cell that recorded at least one event, with counts,
    magnitude/depth statistics, recency, and a per-year activity rate.

    Parameters
    ----------
    events
        Processed catalog with ``cell_id``, ``time``, ``mag``, ``depth``.
    catalog_start, catalog_end
        Bounds used to annualise the rate. Defaults to the observed range.

    Notes
    -----
    ``events_per_year`` is a *historical activity rate*. It is not a hazard or
    risk estimate: detection coverage varies by region and era.
    """
    required = {"cell_id", "time", "mag", "depth"}
    missing = required - set(events.columns)
    if missing:
        raise ValidationError(
            f"cell_activity is missing required columns: {sorted(missing)}",
            hint="Run `seismo data validate` to rebuild the processed dataset.",
        )
    if events.empty:
        return pd.DataFrame(columns=[
            "cell_id", "n_events", "mean_mag", "max_mag", "mean_depth",
            "last_event_time", "days_since_last_event", "events_per_year",
        ])

    grouped = events.groupby("cell_id", sort=True)
    out = pd.DataFrame({
        "n_events": grouped.size(),
        "mean_mag": grouped["mag"].mean(),
        "max_mag": grouped["mag"].max(),
        "mean_depth": grouped["depth"].mean(),
        "last_event_time": grouped["time"].max(),
    }).reset_index()

    start = catalog_start if catalog_start is not None else events["time"].min()
    end = catalog_end if catalog_end is not None else events["time"].max()
    span_years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)

    now = events["time"].max()
    out["days_since_last_event"] = (now - out["last_event_time"]).dt.total_seconds() / 86400
    out["events_per_year"] = out["n_events"] / span_years

    # Centre coordinates so the table is directly plottable.
    lat_c, lon_c = grid_centers_for(out["cell_id"].to_numpy(), grid)
    out["latitude_center"] = lat_c
    out["longitude_center"] = lon_c
    return out.sort_values("n_events", ascending=False).reset_index(drop=True)


def grid_centers_for(cell_ids: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    return grid.centers(cell_ids)


def hotspot_cells(activity: pd.DataFrame, *, top_n: int = 50) -> pd.DataFrame:
    """Highest historical-activity cells, ranked by event count.

    Ranking is deliberately simple and explainable: raw counts. A statistical
    hotspot scan (e.g. Getis-Ord Gi*) is a documented extension, not a v1 goal.
    """
    cols = ["cell_id", "n_events", "mean_mag", "max_mag",
            "latitude_center", "longitude_center"]
    available = [c for c in cols if c in activity.columns]
    ranked = activity.sort_values("n_events", ascending=False)
    return ranked.head(top_n)[available].reset_index(drop=True)


def spatial_density(
    events: pd.DataFrame,
    grid: Grid,
    *,
    lat_bins: int = 180,
    lon_bins: int = 360,
) -> np.ndarray:
    """2D event-density histogram over lat/lon for map layers.

    Returns an array shaped ``(lat_bins, lon_bins)`` of event counts —
    pre-aggregated so the map never has to render 707k raw points.
    """
    hist, _, _ = np.histogram2d(
        events["latitude"].to_numpy(),
        events["longitude"].to_numpy(),
        bins=[lat_bins, lon_bins],
        range=[[-90, 90], [-180, 180]],
    )
    return hist
