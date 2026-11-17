"""Global Seismic Map — where the selected activity was recorded.

Two views over the same filtered selection: an aggregated density surface
(always safe at catalog size) and raw event points (auto-downsampled with the
numbers stated on screen, never silently).
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

import data_access  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import map_layers, theme  # noqa: E402
from components.filters import apply_filters, render_filters  # noqa: E402

MODES = ("Aggregated density", "Event points")


def _strongest_table(filtered: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """The strongest ``limit`` events in the selection, formatted for reading."""
    table = filtered.nlargest(limit, "mag")[
        ["time", "place", "mag", "depth", "latitude", "longitude"]
    ].copy()
    table["time"] = table["time"].dt.strftime("%Y-%m-%d %H:%M")
    table["latitude"] = table["latitude"].round(3)
    table["longitude"] = table["longitude"].round(3)
    table.columns = [
        "Time (UTC)", "Place", "Magnitude", "Depth (km)", "Latitude", "Longitude",
    ]
    return table


def main() -> None:
    st.title("Global Seismic Map")
    st.caption(
        "Filtered catalog plotted by epicentre — the map shows where events were "
        "recorded, not where future events will occur."
    )

    cfg = data_access.get_config()
    state = render_filters(
        regions=[],
        default_start=pd.Timestamp(cfg.start_date, tz="UTC"),
        default_end=pd.Timestamp(cfg.end_date, tz="UTC"),
        show_depth=True,
        show_regions=False,
        key_prefix="map",
    )

    catalog = data_access.load_processed()
    filtered = apply_filters(catalog, state)

    mode = st.radio(
        "View",
        list(MODES),
        index=0,
        horizontal=True,
        key="map_mode",
        help="Density aggregates the selection; points draw individual epicentres.",
    )

    if filtered.empty:
        st.info("No events match the current filters.")
        return

    selected = len(filtered)
    ceiling = map_layers.MAX_RAW_POINTS

    if mode == "Event points":
        if selected > ceiling:
            st.warning(
                f"Showing {ceiling:,} of {selected:,} selected events — individual "
                f"points are downsampled to {ceiling:,} so the map stays interactive. "
                "Narrow the date, magnitude or depth filters to see every event."
            )
        st.pydeck_chart(map_layers.event_scatter(filtered))
    else:
        deck = map_layers.density_heatmap(filtered)
        st.pydeck_chart(deck)
        drawn = min(selected, ceiling)
        if selected > ceiling:
            st.caption(
                f"Heat surface aggregated over a {drawn:,}-event sample of the "
                f"{selected:,} selected events — relative concentration, not an "
                "exact count per cell."
            )
        else:
            st.caption(
                f"Heat surface over all {selected:,} selected events, weighted by "
                "magnitude."
            )

    st.markdown(f"**{selected:,} events** match the current filters "
                f"(M{state.min_mag:.1f}–M{state.max_mag:.1f}, "
                f"{state.start.date()} → {state.end.date()}).")
    table = _strongest_table(filtered)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(f"{len(table):,} strongest events in the current selection.")

    st.markdown(
        theme.limitations_html(
            "Density on this map is a record of where monitoring networks detected "
            "and located events. Bright areas can reflect dense station coverage as "
            "much as higher seismic activity, and historical concentration is not a "
            "prediction of future epicentres or of disaster risk — risk additionally "
            "requires exposure and vulnerability."
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
