"""Forecasting — the production model's probability estimates.

Everything shown here is computed from shipped artifacts: the promoted model
in the registry, the feature dataset, and the validation calibration report.
The page reports a probability estimate for a *grid cell* over the next
horizon, always next to the observed base rate and the calibration error, so
uncertainty is visible instead of smoothed away.
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
    """Render the Forecasting page."""
    import data_access
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from components import charts, theme
    from components.kpi_cards import fmt_int, fmt_pct, render_kpis

    from seismo.geospatial.analysis import NAMED_REGIONS

    st.title("Forecasting")
    st.markdown(
        "What the **production** model — the version a named reviewer promoted — "
        "estimates for the selected region, shown beside the observed base rate and "
        "the calibration error. Every figure is computed from the shipped artifacts."
    )

    # --- 1. the human promotion gate --------------------------------------
    registry = data_access.load_registry()
    production = registry.get("production") or {}
    if not production:
        st.info("No model has been promoted yet — `seismo model promote` is the human gate.")
        st.stop()

    prod_name = "xgboost" if "xgboost" in production else sorted(production)[0]
    history = registry.get("history") or []
    last_promotion = history[-1] if history else {}
    reviewer = last_promotion.get("reviewed_by")

    # --- 2. the promoted artifact -----------------------------------------
    try:
        model, metadata = data_access.load_production_model()
    except Exception as exc:
        st.error(f"The production model could not be loaded — {_exc_note(exc)}")
        st.stop()

    feature_cols = list(metadata.get("feature_cols") or [])
    version = str(metadata.get("version") or "unknown")

    cfg = data_access.get_config()
    forecast_cfg = cfg.section("forecast")
    horizon_days = int(forecast_cfg.get("horizon_days", 30))
    magnitude_threshold = float(forecast_cfg.get("magnitude_threshold", 4.5))

    render_kpis(
        [
            ("Production model", prod_name, f"version {version}"),
            (
                "Reviewed by",
                str(reviewer) if reviewer else "not recorded",
                "promotion requires a named reviewer",
            ),
            (
                "Forecast horizon",
                f"{horizon_days} days",
                f"target: at least one M{magnitude_threshold:.1f} event in the cell",
            ),
            (
                "Promoted",
                str(last_promotion.get("promoted_at") or "—")[:16],
                "last registry entry (UTC)",
            ),
        ],
        columns=4,
    )

    # --- 3. region selection ----------------------------------------------
    features = data_access.load_features()
    region = st.selectbox(
        "Region",
        ["Global", *sorted(NAMED_REGIONS)],
        help="Bounding-box filter on cell centres (grid.cell_size_deg in config.yaml).",
    )

    if region == "Global":
        region_rows = features
    else:
        box = NAMED_REGIONS[region]
        lat_lo, lat_hi = box["lat"]
        lon_lo, lon_hi = box["lon"]
        inside = (
            features["latitude_center"].between(lat_lo, lat_hi)
            & features["longitude_center"].between(lon_lo, lon_hi)
        )
        region_rows = features[inside]

    if region_rows.empty:
        st.info(f"No feature rows fall inside the {region} bounding box.")
        st.stop()

    render_kpis(
        [
            ("Selection", region, f"{region_rows['cell_id'].nunique():,} active cells"),
            ("Feature rows", fmt_int(len(region_rows)), "cell-months in scope"),
            (
                "Observed positive rate",
                fmt_pct(float(region_rows["target"].mean())),
                f"labels: at least one M{magnitude_threshold:.1f} event within "
                f"{horizon_days} days",
            ),
            (
                "Observation months",
                fmt_int(region_rows["obs_date"].nunique()),
                "monthly rows, 2001 → 2026",
            ),
        ],
        columns=4,
    )

    # --- 4. observed target rate and activity distribution -----------------
    st.subheader(f"Observed history — {region}")

    monthly = (
        region_rows.groupby("obs_date", sort=True)
        .agg(positive_share=("target", "mean"), cell_months=("target", "size"))
        .reset_index()
    )
    rate_figure = charts.time_series(
        monthly,
        x="obs_date",
        y="positive_share",
        title=f"Positive cell-months by observation month — {region}",
        y_title="Share of cell-months (fraction)",
        x_title="Observation month",
        rolling=12,
        height=440,
    )
    rate_figure.update_yaxes(tickformat=".1%")
    st.plotly_chart(rate_figure, width="stretch")
    st.caption(
        f"Monthly share of cell-months whose label is positive (at least one "
        f"M{magnitude_threshold:.1f} event within {horizon_days} days), with a "
        f"12-month trailing mean. Observed history — not a model output."
    )

    def event_count_distribution(counts: pd.Series) -> pd.DataFrame:
        """Bin trailing-year event counts on a fixed, heavy-tail-friendly scale."""
        values = np.maximum(counts.to_numpy(dtype="float64"), 0.0)
        edges = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype="float64")
        labels = [
            "0–1",
            "2",
            "3–5",
            "6–10",
            "11–20",
            "21–50",
            "51–100",
            "101–200",
            "201–500",
            "501–1000",
        ]
        tallied, _ = np.histogram(values[values <= edges[-1]], bins=edges)
        tallied = tallied.astype("int64")
        over = int((values > edges[-1]).sum())
        if over:
            tallied = np.append(tallied, over)
            labels = [*labels, ">1000"]
        return pd.DataFrame({"events_last_365d": labels, "cell_months": tallied})

    distribution = event_count_distribution(region_rows["events_last_365d"])
    dist_figure = px.bar(distribution, x="events_last_365d", y="cell_months")
    dist_figure.update_traces(
        marker_color=theme.ACCENT,
        marker_line_color=theme.NAVY,
        marker_line_width=0.4,
        hovertemplate="%{x} events<br>%{y:,} cell-months<extra></extra>",
    )
    dist_figure.update_layout(
        theme.base_layout(
            title=f"Trailing-year event counts — {region}",
            x_title="Events recorded in the previous 365 days (count, binned)",
            y_title="Cell-months (count)",
            height=420,
        )
    )
    st.plotly_chart(dist_figure, width="stretch")
    st.caption(
        "Bin widths grow with the count because the distribution is heavy-tailed; "
        "'>1000' groups the tail."
    )

    # --- 5. live cell scoring ---------------------------------------------
    st.subheader("Live cell scoring")
    st.caption(
        "The production model scores the most recent feature row of one of the ten "
        "most active cells — a probability estimate for a cell, never a prediction "
        "of an individual earthquake."
    )

    activity = data_access.load_cell_activity()
    top10 = activity.nlargest(10, "n_events").reset_index(drop=True)
    cell_labels = [
        f"cell {int(row.cell_id)} · {row.longitude_center:.1f}°, "
        f"{row.latitude_center:.1f}° · {int(row.n_events):,} events"
        for row in top10.itertuples()
    ]

    with st.form("live_cell_scoring"):
        chosen = st.selectbox("Cell", cell_labels)
        st.form_submit_button("Score cell")

    selected_cell = top10.iloc[cell_labels.index(chosen)]
    cell_rows = features[features["cell_id"] == int(selected_cell["cell_id"])].sort_values(
        "obs_date"
    )

    if cell_rows.empty:
        st.info("This cell has no feature rows yet — there is nothing to score.")
    else:
        latest = cell_rows.iloc[[-1]]
        observed = pd.Timestamp(latest["obs_date"].iloc[0])
        audit = pd.DataFrame(
            {
                "feature": feature_cols,
                "value": [latest.iloc[0][column] for column in feature_cols],
            }
        )
        st.caption(
            f"Cell {int(selected_cell['cell_id'])} · "
            f"{selected_cell['longitude_center']:.1f}°, "
            f"{selected_cell['latitude_center']:.1f}° · "
            f"feature row observed {observed:%Y-%m-%d}"
        )
        st.dataframe(audit, width="stretch", hide_index=True)

        try:
            probability = float(
                model.predict_proba(latest[feature_cols].astype("float64"))[0][1]
            )
        except Exception as exc:
            st.error(f"The model could not score this cell — {_exc_note(exc)}")
        else:
            st.markdown(
                theme.kpi_html(
                    "Estimated probability",
                    f"{probability:.1%}",
                    f"at least one M{magnitude_threshold:.1f} event in this cell within "
                    f"the next {horizon_days} days",
                    accent=True,
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Probability estimate from historical observations — it does not "
                "predict when, where, or how large an individual earthquake will be."
            )

    # --- 6. calibration ----------------------------------------------------
    st.subheader("Calibration on the validation split")
    try:
        calibration = data_access.load_calibration(prod_name, "val")
        validation = data_access.load_model_comparison("val")
        promoted_row = validation[validation["model"] == prod_name]
        ece = float(promoted_row["ece"].iloc[0]) if not promoted_row.empty else None
        brier = float(promoted_row["brier"].iloc[0]) if not promoted_row.empty else None
    except Exception as exc:
        st.info(f"Calibration data is not available — {_exc_note(exc)}")
    else:
        st.plotly_chart(
            charts.reliability_diagram(
                calibration, model=prod_name, ece=ece, brier=brier
            ),
            width="stretch",
        )
        st.caption(
            "Each point is a probability bin: how often the label came true against how "
            "often the model said it would. The dashed grey line is perfect calibration."
        )

    # --- 7. standing scientific caveat -------------------------------------
    st.markdown(
        theme.limitations_html(
            "Probabilities are estimated from historical observations — this system "
            "does <b>not</b> predict the time, location, or magnitude of individual "
            "earthquakes. Historical activity is <b>not</b> disaster risk: risk "
            "additionally requires exposure and vulnerability. Catalog completeness "
            "varies by region and era, so absence of records is not evidence of "
            "absence of seismicity."
        ),
        unsafe_allow_html=True,
    )


main()
