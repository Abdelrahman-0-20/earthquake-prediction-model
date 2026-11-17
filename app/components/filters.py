"""Shared sidebar filters.

Every page reads the same filter state, so moving between pages keeps the
user's date range and thresholds — the indicators on the Overview page and
the rows on the Forecasting page always describe the same selection.

State lives in ``st.session_state`` under one namespaced key so pages never
collide, and the filter block renders identically everywhere (clean
navigation, consistent labels, no surprises).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from components.theme import NAVY

DEFAULT_MIN_MAG = 2.5
DEFAULT_MAX_MAG = 9.1


@dataclass(frozen=True)
class FilterState:
    """A resolved, immutable view of the current sidebar selection."""

    start: pd.Timestamp
    end: pd.Timestamp
    min_mag: float
    max_mag: float
    regions: tuple[str, ...]
    depth_max: float | None

    @property
    def date_label(self) -> str:
        return f"{self.start.date()} → {self.end.date()}"


def render_filters(
    regions: list[str],
    *,
    default_start: pd.Timestamp,
    default_end: pd.Timestamp,
    show_depth: bool = True,
    show_regions: bool = True,
    key_prefix: str = "global",
) -> FilterState:
    """Render the sidebar filter block and return the resolved selection."""
    st.sidebar.markdown(
        f"<div style='color:{NAVY};font-weight:600;font-size:13px;"
        f"letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;'>"
        f"Filters</div>",
        unsafe_allow_html=True,
    )

    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start.date(), default_end.date()),
        key=f"{key_prefix}_dates",
        help="Inclusive range applied to every indicator on this page.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:  # user is still picking the second date
        start = end = date_range[0] if date_range else default_start.date()

    min_mag, max_mag = st.sidebar.slider(
        "Magnitude range",
        min_value=0.0,
        max_value=DEFAULT_MAX_MAG,
        value=(DEFAULT_MIN_MAG, DEFAULT_MAX_MAG),
        step=0.1,
        key=f"{key_prefix}_mag",
        help="Catalog completeness varies by region — small events are "
             "under-detected away from station networks.",
    )

    depth_max: float | None = None
    if show_depth:
        depth_max = st.sidebar.slider(
            "Max depth (km)",
            min_value=0.0,
            max_value=700.0,
            value=700.0,
            step=10.0,
            key=f"{key_prefix}_depth",
            help="Negative (near-surface) depths are included.",
        )

    selected: tuple[str, ...] = ()
    if show_regions:
        selected = tuple(
            st.sidebar.multiselect(
                "Named regions",
                options=regions,
                default=[],
                key=f"{key_prefix}_regions",
                help="Leave empty for global coverage.",
            )
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**{date_range[0] if isinstance(date_range, tuple) else start}"
        f" → {end}** · M{min_mag:.1f}–{max_mag:.1f}"
        + (f" · ≤{depth_max:.0f} km" if depth_max is not None else "")
    )

    import pandas as _pd

    return FilterState(
        start=_pd.Timestamp(start, tz="UTC"),
        end=_pd.Timestamp(end, tz="UTC") + _pd.Timedelta(days=1) - _pd.Timedelta(nanoseconds=1),
        min_mag=float(min_mag),
        max_mag=float(max_mag),
        regions=selected,
        depth_max=float(depth_max) if depth_max is not None else None,
    )


def apply_filters(frame: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Apply a FilterState to a catalog frame with time/mag/depth columns."""
    mask = (frame["time"] >= state.start) & (frame["time"] <= state.end)
    mask &= frame["mag"].between(state.min_mag, state.max_mag)
    if state.depth_max is not None and "depth" in frame.columns:
        mask &= frame["depth"] <= state.depth_max
    if state.regions and "region" in frame.columns:
        mask &= frame["region"].isin(state.regions)
    return frame[mask]
