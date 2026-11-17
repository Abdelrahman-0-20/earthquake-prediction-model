"""Overview — how much activity was recorded, how it is distributed, and how
complete the catalog is.

Every number on this page is computed from the filtered selection; nothing is
read from disk except through ``data_access``.
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
from components import charts, theme  # noqa: E402
from components.filters import apply_filters, render_filters  # noqa: E402
from components.kpi_cards import fmt_int, render_kpis  # noqa: E402

#: Columns whose missingness is shown in the data-quality panel.
QUALITY_COLUMNS = ("dmin", "nst", "gap", "rms")


def _region_labels(places: pd.Series) -> pd.Series:
    """Collapse the free-text ``place`` string to a region label.

    USGS places look like "14 km SW of Dragash, Kosovo" — the text after the
    last comma is the region. Rows without a comma ("Kuril Islands region")
    keep their own text, and blank entries fall back to a placeholder so no
    event disappears from the counts.
    """
    text = places.fillna("").astype(str).str.strip()
    labels = text.str.split(",").str[-1].str.strip()
    return labels.mask(labels.eq(""), "Unspecified region")


def _quality_panel() -> None:
    """Figures from the catalog quality report — counts, not claims."""
    try:
        quality = data_access.load_data_quality()
        render_kpis(
            [
                ("Non-earthquake rows removed", fmt_int(quality["non_earthquake_removed"]),
                 "quarantined during validation"),
                ("Duplicate rows removed", fmt_int(quality["duplicates_removed"]),
                 "within the raw ingest"),
                ("Negative depths recorded", fmt_int(quality["negative_depth_rows"]),
                 "near-surface solutions, kept"),
                ("Rows processed", fmt_int(quality["processed_rows"]),
                 f"of {fmt_int(quality['total_raw_rows'])} raw rows"),
            ],
            columns=4,
        )

        missingness = quality.get("missingness", {})
        rows = []
        for col in QUALITY_COLUMNS:
            entry = missingness.get(col)
            if isinstance(entry, dict) and "missing" in entry and "pct" in entry:
                rows.append(
                    {"Column": col, "Missing": entry["missing"], "Missing (%)": entry["pct"]}
                )
        if rows:
            st.markdown("**Missing station metadata**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                "dmin (station distance), nst (station count), gap (azimuth gap) and "
                "rms are reported by the contributing network, so their absence is a "
                "property of the feed — it does not remove the event from the catalog."
            )
    except Exception as exc:  # noqa: BLE001 - a missing report is guidance, not a crash
        st.info(f"Data-quality report unavailable — {exc}")


def main() -> None:
    st.title("Overview")
    st.caption(
        "Recorded earthquake activity, global catalog — counts and distributions "
        "describe what was captured by monitoring networks, not future risk."
    )

    cfg = data_access.get_config()
    state = render_filters(
        regions=[],
        default_start=pd.Timestamp(cfg.start_date, tz="UTC"),
        default_end=pd.Timestamp(cfg.end_date, tz="UTC"),
        show_depth=True,
        show_regions=False,
        key_prefix="overview",
    )

    catalog = data_access.load_processed()
    filtered = apply_filters(catalog, state)

    if filtered.empty:
        st.info("No events match the current filters.")
        st.stop()

    # --- headline indicators ------------------------------------------------
    threshold = float(cfg.section("forecast").get("magnitude_threshold", 4.5))
    strongest = filtered.loc[filtered["mag"].idxmax()]
    strongest_place = strongest["place"]
    if not isinstance(strongest_place, str) or not strongest_place.strip():
        strongest_place = "location not stated"
    strongest_place = strongest_place[:48] + ("..." if len(strongest_place) > 48 else "")

    span_start = filtered["time"].min()
    span_end = filtered["time"].max()
    span_days = int((span_end - span_start) / pd.Timedelta(days=1))

    render_kpis(
        [
            ("Recorded events", fmt_int(len(filtered)),
             f"M{state.min_mag:.1f}–M{state.max_mag:.1f} selected"),
            (f"Events ≥ M{threshold:g}", fmt_int((filtered["mag"] >= threshold).sum()),
             "threshold from config"),
            ("Mean magnitude", f"M{filtered['mag'].mean():.2f}", "of the selected events"),
            ("Strongest event", f"M{strongest['mag']:.1f}", strongest_place),
            ("Date span", f"{span_start:%Y-%m-%d} → {span_end:%Y-%m-%d}",
             f"{span_days:,} days, UTC"),
        ],
        columns=5,
    )

    # --- distributions ------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.magnitude_histogram(filtered["mag"], title="Magnitude distribution"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            charts.depth_histogram(filtered["depth"], title="Depth distribution"),
            width="stretch",
        )

    # --- activity over time -------------------------------------------------
    monthly = (
        filtered.groupby(["year", "month"], observed=True)
        .size()
        .reset_index(name="events")
    )
    monthly["month_start"] = pd.to_datetime(
        {"year": monthly["year"], "month": monthly["month"], "day": 1}
    )
    st.plotly_chart(
        charts.time_series(
            monthly,
            x="month_start",
            y="events",
            title="Recorded events per month",
            y_title="Events per month",
            x_title="Month",
            rolling=12,
        ),
        width="stretch",
    )
    st.caption(
        f"{len(monthly):,} months of records with a 12-month rolling mean; "
        f"peak month {int(monthly.loc[monthly['events'].idxmax(), 'events']):,} events."
    )

    # --- where, and what was recorded most recently -------------------------
    region_counts = (
        _region_labels(filtered["place"])
        .value_counts()
        .head(10)
        .rename_axis("region")
        .reset_index(name="events")
    )
    col_bars, col_recent = st.columns(2)
    with col_bars:
        st.plotly_chart(
            charts.horizontal_bars(
                region_counts,
                y="region",
                x="events",
                title="Top 10 regions by recorded events",
                x_title="Events",
            ),
            width="stretch",
        )
    with col_recent:
        st.markdown("#### Recent records")
        recent = filtered.nlargest(15, "time")[["time", "place", "mag", "depth"]].copy()
        recent["time"] = recent["time"].dt.strftime("%Y-%m-%d %H:%M")
        recent.columns = ["Time (UTC)", "Place", "Magnitude", "Depth (km)"]
        st.dataframe(recent, width="stretch", hide_index=True)
        st.caption("15 most recent events in the current selection.")

    # --- data quality -------------------------------------------------------
    with st.expander("Data quality"):
        _quality_panel()

    st.markdown(
        theme.limitations_html(
            "Recorded counts partly reflect monitoring density: regions with dense "
            "station networks catalogue far more small events, so a higher count "
            "indicates better detection as much as higher seismic activity. "
            "Absence of records is not absence of seismicity, and nothing here "
            "predicts when or where the next event will occur."
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
