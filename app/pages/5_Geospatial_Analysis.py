"""Geospatial Analysis — where recorded activity concentrates.

The map and every table below are built from the pre-aggregated cell-activity
artifact (one row per grid cell), never from raw event points: 11.5k cells
render comfortably, 707k points do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

# AppTest and `streamlit run app/pages/...` execute this file directly, so the
# repository roots have to be importable before any project module is imported.
ROOT = Path(__file__).resolve().parents[2]
for _c in (ROOT, ROOT / "src", ROOT / "app"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))


def _exc_note(exc: BaseException) -> str:
    """One user-facing line: the message plus the fix hint when there is one."""
    hint = getattr(exc, "hint", None)
    return f"{exc} — {hint}" if hint else str(exc)


def main() -> None:
    """Render the Geospatial Analysis page."""
    import data_access
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from components import charts, map_layers, theme
    from components.kpi_cards import fmt_int, render_kpis

    from seismo.geospatial.analysis import NAMED_REGIONS

    st.title("Geospatial Analysis")
    st.markdown(
        "Historical activity per grid cell — counts, rates, and the strongest event "
        "each cell has recorded. This is an **activity indicator**, not hazard or risk."
    )

    try:
        activity = data_access.load_cell_activity()
    except Exception as exc:
        st.error(f"Cell activity is not available — {_exc_note(exc)}")
        st.stop()

    if activity.empty:
        st.info("No cell activity has been built yet — run `seismo geospatial build`.")
        st.stop()

    # --- headline numbers ---------------------------------------------------
    busiest = activity.loc[activity["n_events"].idxmax()]
    render_kpis(
        [
            ("Cells with activity", fmt_int(len(activity)), "cells with at least one event"),
            ("Recorded events", fmt_int(activity["n_events"].sum()), "catalog total, M2.5+"),
            (
                "Busiest cell events",
                fmt_int(busiest["n_events"]),
                f"{busiest['longitude_center']:.1f}°, {busiest['latitude_center']:.1f}°",
            ),
            (
                "Median events per cell",
                fmt_int(activity["n_events"].median()),
                "the distribution is long-tailed",
            ),
        ],
        columns=4,
    )

    # --- map ----------------------------------------------------------------
    cell_size = float(data_access.get_config().section("grid").get("cell_size_deg", 1.0))
    st.subheader("Historical activity map")
    view = st.radio("View", ["Activity level", "Hotspots"], horizontal=True)
    deck = (
        map_layers.hotspot_layer(activity, top_n=50)
        if view == "Hotspots"
        else map_layers.activity_scatter(activity)
    )
    st.pydeck_chart(deck)
    st.caption(
        f"Circle area scales with recorded events. Each cell is a {cell_size:g}° × "
        f"{cell_size:g}° square (grid.cell_size_deg in config.yaml); the hotspot view "
        "marks the 50 most active cells over a muted base layer."
    )

    # --- top cells ----------------------------------------------------------
    st.subheader("Most active cells")
    top15 = activity.nlargest(15, "n_events").copy()
    top15["cell_label"] = [
        f"{longitude:.1f}°, {latitude:.1f}°"
        for longitude, latitude in zip(
            top15["longitude_center"], top15["latitude_center"], strict=True
        )
    ]
    st.plotly_chart(
        charts.horizontal_bars(
            top15,
            y="cell_label",
            x="n_events",
            title="Top 15 cells by recorded events",
            x_title="Recorded events (count)",
        ),
        width="stretch",
    )
    st.caption("Labels are cell-centre coordinates (longitude, latitude).")

    # --- region comparison ---------------------------------------------------
    st.subheader("Region comparison")
    region_rows: list[dict] = []
    named_cells = pd.Series(False, index=activity.index, dtype="bool")
    for name in sorted(NAMED_REGIONS):
        box = NAMED_REGIONS[name]
        lat_lo, lat_hi = box["lat"]
        lon_lo, lon_hi = box["lon"]
        mask = (
            activity["latitude_center"].between(lat_lo, lat_hi)
            & activity["longitude_center"].between(lon_lo, lon_hi)
        )
        named_cells = named_cells | mask
        subset = activity[mask]
        region_rows.append(
            {
                "region": name,
                "cells": int(len(subset)),
                "events": int(subset["n_events"].sum()),
                "events_per_year": float(subset["events_per_year"].sum()),
                "max_mag": float(subset["max_mag"].max()) if len(subset) else float("nan"),
            }
        )

    outside = activity[~named_cells]
    region_rows.append(
        {
            "region": "Other / uncategorized",
            "cells": int(len(outside)),
            "events": int(outside["n_events"].sum()),
            "events_per_year": float(outside["events_per_year"].sum()),
            "max_mag": float(outside["max_mag"].max()) if len(outside) else float("nan"),
        }
    )

    regions = (
        pd.DataFrame(region_rows).sort_values("events", ascending=False).reset_index(drop=True)
    )
    table = regions.rename(
        columns={
            "region": "Region",
            "cells": "Cells",
            "events": "Recorded events",
            "events_per_year": "Events per year",
            "max_mag": "Max magnitude (M)",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    catalog_events = float(activity["n_events"].sum()) or 1.0
    outside_share = float(outside["n_events"].sum()) / catalog_events
    st.caption(
        "Bounding boxes come from `seismo.geospatial.analysis.NAMED_REGIONS` — the same "
        "boxes the Forecasting page filters on — and a cell outside every box is listed "
        f"as uncategorized: {outside_share:.1%} of recorded events sit outside the "
        f"named regions ({int(len(outside)):,} cells)."
    )
    st.plotly_chart(
        charts.horizontal_bars(
            regions,
            y="region",
            x="events",
            title="Recorded events by region",
            x_title="Recorded events (count)",
        ),
        width="stretch",
    )

    # --- rate vs. strongest event --------------------------------------------
    st.subheader("Activity rate vs. strongest recorded event")
    usable = activity[
        (activity["events_per_year"] > 0)
        & activity["max_mag"].notna()
        & activity["mean_depth"].notna()
    ]
    if usable.empty:
        st.info("No cells have both a positive activity rate and a recorded magnitude.")
    else:
        scatter = px.scatter(
            usable,
            x="events_per_year",
            y="max_mag",
            color="mean_depth",
            color_continuous_scale=theme.DENSITY_CMAP,
            opacity=0.7,
            labels={
                "events_per_year": "Events per year (count/year)",
                "max_mag": "Maximum recorded magnitude (M)",
                "mean_depth": "Mean depth (km)",
            },
        )
        layout = theme.base_layout(
            title="Historical activity rate vs. strongest event per cell",
            x_title="Events per year (count/year, log scale)",
            y_title="Maximum recorded magnitude (M)",
            height=560,
        )
        layout["xaxis"] = dict(layout["xaxis"], type="log")
        scatter.update_layout(**layout)
        scatter.update_coloraxes(colorbar_title_text="Mean depth (km)")
        st.plotly_chart(scatter, width="stretch")

        log_rate = np.log10(usable["events_per_year"].to_numpy(dtype="float64"))
        magnitude = usable["max_mag"].to_numpy(dtype="float64")
        correlation = float(np.corrcoef(log_rate, magnitude)[0, 1])
        st.markdown(
            f"**Correlation:** Pearson r between log10(events per year) and maximum "
            f"recorded magnitude is **{correlation:+.3f}** across {len(usable):,} cells. "
            "Faster-recorded cells also tend to contain a stronger maximum event — "
            "partly a sampling effect, since a longer and denser record has more "
            "chances to contain a large event. This is a property of the catalog, "
            "not a physical law."
        )

    # --- standing scientific caveat ------------------------------------------
    st.markdown(
        theme.limitations_html(
            "Cell activity is a <b>historical activity indicator</b>, not hazard or "
            "risk: risk additionally needs exposure and vulnerability. Raw counts "
            "partly reflect monitoring density — regions with denser station networks "
            "record more small events. Absence of records is not evidence of absence "
            "of seismicity."
        ),
        unsafe_allow_html=True,
    )


main()
