"""Historical Analysis — how recorded activity changed over the catalog period.

Everything here is a historical indicator computed from the filtered
selection: annual totals, monthly seasonality, magnitude distributions by era
and depth by region. None of it predicts when or where the next event occurs.
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
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from components import charts, theme  # noqa: E402
from components.filters import apply_filters, render_filters  # noqa: E402
from components.kpi_cards import fmt_int, render_kpis  # noqa: E402

#: Magnitude bins (0.5-wide) covering the catalog range.
MAG_BINS = [round(0.5 * i, 2) for i in range(1, 20)]

#: Era boundaries — three comparable windows over the 2000–2026 record.
ERA_BINS = [2000, 2008, 2016, 2027]
ERA_LABELS = ["2000–2007", "2008–2015", "2016–2026"]

TOP_REGIONS = 5
TOP_MONTHS = 10
VIEW_RAW = "Raw counts"
VIEW_CUMULATIVE = "Cumulative"


def _region_labels(places: pd.Series) -> pd.Series:
    """Free-text ``place`` collapsed to a region label (text after the comma)."""
    text = places.fillna("").astype(str).str.strip()
    labels = text.str.split(",").str[-1].str.strip()
    return labels.mask(labels.eq(""), "Unspecified region")


def _monthly_counts(filtered: pd.DataFrame) -> pd.DataFrame:
    """Events per calendar month, with a timestamp column for plotting."""
    monthly = (
        filtered.groupby(["year", "month"], observed=True)
        .size()
        .reset_index(name="events")
    )
    monthly["month_start"] = pd.to_datetime(
        {"year": monthly["year"], "month": monthly["month"], "day": 1}
    )
    return monthly


def _depth_by_region(filtered: pd.DataFrame):
    """Box plot of depth for the most active regions in the selection."""
    frame = filtered.assign(region=_region_labels(filtered["place"]))
    top = frame["region"].value_counts().head(TOP_REGIONS).index
    subset = frame[frame["region"].isin(top)]
    fig = px.box(
        subset,
        x="region",
        y="depth",
        color="region",
        color_discrete_sequence=theme.SERIES_COLORS,
    )
    fig.update_layout(
        **theme.base_layout(
            title=f"Depth distribution of the {TOP_REGIONS} most active regions",
            x_title="Region",
            y_title="Depth (km)",
            height=460,
        )
    )
    fig.update_traces(boxpoints=False, hovertemplate="%{y:.1f} km<extra></extra>")
    return fig, len(subset)


def main() -> None:
    st.title("Historical Analysis")
    st.caption(
        "Recorded activity over time — annual totals, monthly counts, magnitude "
        "distributions by era and depth by region, all computed from the current "
        "filter selection."
    )

    cfg = data_access.get_config()
    state = render_filters(
        regions=[],
        default_start=pd.Timestamp(cfg.start_date, tz="UTC"),
        default_end=pd.Timestamp(cfg.end_date, tz="UTC"),
        show_depth=False,
        show_regions=False,
        key_prefix="hist",
    )

    catalog = data_access.load_processed()
    filtered = apply_filters(catalog, state)

    if filtered.empty:
        st.info("No events match the current filters.")
        st.stop()

    # --- headline indicators ------------------------------------------------
    annual = filtered.groupby("year", observed=True).size().reset_index(name="events")
    years = int(annual["year"].nunique())
    strongest = filtered.loc[filtered["mag"].idxmax()]
    strongest_place = strongest["place"]
    if not isinstance(strongest_place, str) or not strongest_place.strip():
        strongest_place = "location not stated"

    render_kpis(
        [
            ("Years covered", f"{years:,}",
             f"{int(annual['year'].min())} → {int(annual['year'].max())}"),
            ("Events per year", fmt_int(len(filtered) / max(years, 1)),
             "mean of the selection"),
            ("Mean magnitude", f"M{filtered['mag'].mean():.2f}", "of the selected events"),
            ("Strongest event", f"M{strongest['mag']:.1f}", strongest_place[:48]),
        ],
        columns=4,
    )

    # --- annual totals ------------------------------------------------------
    st.markdown("#### Recorded events per year")
    annual_table = annual.rename(columns={"year": "Year", "events": "Events"})
    annual_table["Share of selection (%)"] = (
        100 * annual["events"] / annual["events"].sum()
    ).round(2)
    st.dataframe(annual_table, hide_index=True, width="stretch")

    view = st.radio(
        "Yearly view",
        [VIEW_RAW, VIEW_CUMULATIVE],
        index=0,
        horizontal=True,
        key="hist_view",
        help="Cumulative totals make long-run growth easier to read; raw counts "
             "keep year-to-year variation visible.",
    )
    if view == VIEW_CUMULATIVE:
        yearly = annual.assign(cumulative=annual["events"].cumsum())
        fig = charts.grouped_bar(
            yearly,
            x="year",
            y="cumulative",
            title="Cumulative recorded events",
            y_title="Cumulative events",
        )
    else:
        fig = charts.grouped_bar(
            annual,
            x="year",
            y="events",
            title="Recorded events per year",
            y_title="Events per year",
        )
    st.plotly_chart(fig.update_xaxes(title_text="Year"), width="stretch")

    # --- monthly activity ---------------------------------------------------
    monthly = _monthly_counts(filtered)
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
        f"{len(monthly):,} months in the selection; the 12-month rolling mean "
        "smooths short-term reporting gaps."
    )

    # --- magnitude distribution by era --------------------------------------
    st.markdown("#### Magnitude distribution by era")
    binned = filtered.assign(
        mag_bin=pd.cut(filtered["mag"], bins=MAG_BINS, right=False),
        era=pd.cut(filtered["year"], bins=ERA_BINS, labels=ERA_LABELS, right=False),
    ).dropna(subset=["mag_bin", "era"])
    distribution = (
        binned.groupby(["mag_bin", "era"], observed=True)
        .size()
        .reset_index(name="count")
    )
    distribution["mag_bin"] = distribution["mag_bin"].astype(str)

    fig = charts.grouped_bar(
        distribution,
        x="mag_bin",
        y="count",
        group="era",
        title="Recorded events by magnitude and era",
        y_title="Events",
        height=460,
    )
    st.plotly_chart(fig.update_xaxes(title_text="Magnitude"), width="stretch")

    st.markdown(
        """
**Scientific caveat.** The apparent growth in recorded events over time
partly reflects improving detection coverage — denser station networks,
faster review and wider magnitude reporting — rather than an increase in
seismicity. Counts are a historical indicator of *recorded* activity, not
evidence that the ground is shaking more often, and not a prediction of
future events.
        """
    )

    # --- depth --------------------------------------------------------------
    st.markdown("#### Depth analysis")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.depth_histogram(filtered["depth"], title="Depth distribution"),
            width="stretch",
        )
    with right:
        box_fig, box_rows = _depth_by_region(filtered)
        st.plotly_chart(box_fig, width="stretch")
        st.caption(
            f"{box_rows:,} events across the {TOP_REGIONS} most active regions in "
            "the current selection; negative depths are near-surface solutions."
        )

    st.markdown(
        theme.limitations_html(
            "Deeper and more frequent records do not mean higher risk: catalog "
            "completeness varies by region and era, and disaster risk additionally "
            "requires exposure and vulnerability. Nothing on this page predicts the "
            "time, location or magnitude of individual earthquakes."
        ),
        unsafe_allow_html=True,
    )

    # --- most active months -------------------------------------------------
    with st.expander(f"Most active months (top {TOP_MONTHS})"):
        top_months = monthly.nlargest(TOP_MONTHS, "events").copy()
        top_months["Month"] = top_months["month_start"].dt.strftime("%Y-%m")
        st.dataframe(
            top_months[["Month", "events"]].rename(columns={"events": "Events"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Months ranked by recorded event count within the current selection. "
            "A busy month can reflect a single large sequence plus its aftershocks "
            "and improved reporting; the ranking is a historical indicator of "
            "recorded activity, not a seasonal forecast."
        )


if __name__ == "__main__":
    main()
