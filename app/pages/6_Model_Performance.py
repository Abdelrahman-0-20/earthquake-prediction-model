"""Model Performance — does the model beat counting history, and is it calibrated?

Two yardsticks only: ranking quality on the rare positive class (PR-AUC) and
probabilistic skill against the climatological baseline (Brier Skill Score,
ECE). Every table and figure is read from the evaluation reports through
``data_access``; the held-out test split is displayed, never tuned on.
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

#: Display headers for every metric frame this page renders.
_TABLE_RENAME = {
    "model": "Model",
    "version": "Version",
    "split": "Split",
    "n_rows": "Rows",
    "positive_rate": "Positive rate",
    "threshold": "Threshold",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "brier": "Brier",
    "brier_skill_score": "Brier skill score",
    "ece": "ECE",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "region": "Region",
    "reliable": "Reliable",
    "feature": "Feature",
    "mean_abs_shap": "Mean |SHAP|",
    "share_of_total": "Share of total",
}

#: Continuous metrics rounded for display (never for computation).
_ROUNDED = (
    "positive_rate",
    "threshold",
    "pr_auc",
    "roc_auc",
    "brier",
    "brier_skill_score",
    "ece",
    "f1",
    "precision",
    "recall",
    "mean_abs_shap",
    "share_of_total",
)


def _exc_note(exc: BaseException) -> str:
    """One user-facing line: the message plus the fix hint when there is one."""
    hint = getattr(exc, "hint", None)
    return f"{exc} — {hint}" if hint else str(exc)


def main() -> None:
    """Render the Model Performance page."""
    import data_access
    import streamlit as st
    from components import charts, theme
    from components.kpi_cards import fmt_int, fmt_pct, render_kpis

    st.title("Model Performance")
    st.markdown(
        "Every model is judged against two yardsticks: how well it ranks the rare "
        "positive class, and whether it beats simply counting history. Nothing on "
        "this page was tuned on the test split."
    )

    # --- 1. evaluation reports ---------------------------------------------
    try:
        comparison = data_access.load_model_comparison("val")
    except Exception as exc:
        st.error(f"Validation metrics are not available — {_exc_note(exc)}")
        st.stop()

    try:
        test = data_access.load_model_comparison("test")
    except Exception as exc:
        test = None
        st.info(f"The held-out test split has not been scored yet — {_exc_note(exc)}")

    # Class balance is measured from the shipped feature dataset, not quoted.
    try:
        features = data_access.load_features()
        prevalence = float(features["target"].mean())
    except Exception as exc:
        features = None
        prevalence = None
        st.info(f"Class balance could not be measured — {_exc_note(exc)}")

    cfg = data_access.get_config()
    splits_cfg = cfg.section("splits")
    models_cfg = cfg.section("models") or {}
    horizon_days = int(cfg.section("forecast").get("horizon_days", 30))

    best = comparison.iloc[0]
    render_kpis(
        [
            ("Models evaluated", fmt_int(len(comparison)), "validation split"),
            (
                "Best validation BSS",
                f"{float(best['brier_skill_score']):+.4f}",
                str(best["model"]),
            ),
            (
                "Positive rate",
                fmt_pct(prevalence) if prevalence is not None else "—",
                "positive cell-months, all splits",
            ),
            (
                "Test split",
                "scored once" if test is not None else "not scored",
                "held out for final reporting only",
            ),
        ],
        columns=4,
    )

    def reading_notes() -> str:
        """The three claims a reader must take away, with the measured imbalance."""
        balance = (
            f"at {prevalence:.2%} positive-class imbalance"
            if prevalence is not None
            else "under the observed positive-class imbalance"
        )
        return "\n\n".join(
            [
                f"**PR-AUC leads {balance}.** With only a few percent of rows positive, "
                "accuracy and ROC-AUC flatter a model that ranks well while mispricing "
                "probabilities; PR-AUC scores the positive class deployment cares about.",
                "**Brier Skill Score > 0 beats counting history.** A BSS of exactly 0 "
                "ties the climatological baseline and a negative score means counting "
                "history was better — that result is reported, not hidden.",
                "**The test split was scored exactly once** and must never be used for "
                "tuning: thresholds, hyperparameters, and model selection were frozen "
                "on validation before the test split was touched.",
            ]
        )

    def format_table(frame):
        """Round the continuous metrics and label every column with its unit."""
        display = frame.copy()
        for column in _ROUNDED:
            if column in display.columns:
                display[column] = display[column].astype("float64").round(4)
        if "n_rows" in display.columns:
            display["n_rows"] = display["n_rows"].map(lambda value: f"{int(value):,}")
        columns = [column for column in _TABLE_RENAME if column in display.columns]
        return display[columns].rename(columns=_TABLE_RENAME)

    # --- 2. validation and test, side by side -------------------------------
    def render_split(frame, label: str) -> None:
        st.markdown(f"#### {label}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.metric_bars(
                    frame,
                    column="pr_auc",
                    title=f"PR-AUC — {label}",
                    x_title="PR-AUC (area under the precision–recall curve)",
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                charts.metric_bars(
                    frame,
                    column="ece",
                    title=f"Expected calibration error — {label}",
                    x_title="ECE (lower is better; 0 is perfect)",
                ),
                width="stretch",
            )
        st.dataframe(format_table(frame), width="stretch", hide_index=True)
        st.markdown(reading_notes())

    tab_val, tab_test = st.tabs(["Validation", "Test (held out)"])
    with tab_val:
        render_split(comparison, "Validation")
    with tab_test:
        if test is None:
            st.info(
                "The held-out test split has not been scored — run "
                "`seismo model evaluate --split test` once the model is frozen."
            )
        else:
            render_split(test, "Test (held out)")

    # --- 3. the honest yardstick -------------------------------------------
    st.subheader("Brier Skill Score — the honest yardstick")
    st.plotly_chart(charts.skill_score_chart(comparison), width="stretch")
    st.caption(
        "Validation split. Positive beats the climatological baseline, negative loses "
        "to it, and the dashed zero line is counting history exactly."
    )

    # --- 4. per-model calibration and regional performance ------------------
    st.subheader("Calibration and regional performance")
    available = list(comparison["model"].drop_duplicates())
    selected = st.selectbox(
        "Model",
        available,
        help="Listed in order of validation Brier Skill Score (best first).",
    )

    try:
        calibration = data_access.load_calibration(selected, "val")
    except Exception as exc:
        calibration = None
        st.info(f"No calibration data for {selected} — {_exc_note(exc)}")
    if calibration is not None:
        selected_row = comparison[comparison["model"] == selected]
        ece = float(selected_row["ece"].iloc[0]) if not selected_row.empty else None
        brier = float(selected_row["brier"].iloc[0]) if not selected_row.empty else None
        st.plotly_chart(
            charts.reliability_diagram(calibration, model=selected, ece=ece, brier=brier),
            width="stretch",
        )
        st.caption(
            "Each point is a probability bin: observed frequency against mean predicted "
            "probability, on the validation split."
        )

    try:
        region_metrics = data_access.load_region_metrics(selected, "val")
    except Exception as exc:
        region_metrics = None
        st.info(f"No regional metrics for {selected} — {_exc_note(exc)}")
    if region_metrics is not None:
        st.markdown(f"**Regional performance — {selected} (validation)**")
        st.dataframe(
            format_table(region_metrics), width="stretch", hide_index=True
        )
        noisy = int((~region_metrics["reliable"].astype(bool)).sum())
        st.caption(
            "Rows marked Reliable = False have fewer than 500 scored rows and are too "
            f"noisy to claim from — {noisy} of {len(region_metrics)} rows in this "
            "selection. A pooled metric can hide a region where the model fails, which "
            "is why this table exists."
        )

    # --- 5. feature attribution ---------------------------------------------
    st.subheader(f"Feature attribution — {selected}")
    try:
        shap_frame = data_access.load_shap_importance(selected, "val")
    except Exception as exc:
        shap_frame = None
        st.info(f"No SHAP importance for {selected} — {_exc_note(exc)}")

    if shap_frame is not None and not shap_frame.empty:
        top12 = shap_frame.nlargest(12, "mean_abs_shap")
        st.plotly_chart(
            charts.horizontal_bars(
                top12,
                y="feature",
                x="mean_abs_shap",
                title=f"Top 12 features by mean |SHAP| — {selected} (validation)",
                x_title="Mean |SHAP| (model output units)",
            ),
            width="stretch",
        )
        st.markdown(
            "SHAP explains **the model**, not causality: a large value means the model "
            "leans on that feature, not that the feature causes earthquakes. The "
            "trailing-window features (`events_last_30d`, `events_last_90d`, "
            "`events_last_365d`) are correlated by construction — each window contains "
            "the shorter one — so their shares must be read together."
        )

    # --- 6. methodology ------------------------------------------------------
    with st.expander("Methodology"):
        if features is None:
            st.info("Split sizes could not be read — the feature dataset is unavailable.")
        else:
            counts = features["split"].value_counts()
            bounds = features.groupby("split", observed=True)["obs_date"].agg(["min", "max"])
            split_rows = []
            for split_name in ("train", "val", "test"):
                if split_name not in counts.index:
                    continue
                start = bounds.loc[split_name, "min"]
                end = bounds.loc[split_name, "max"]
                split_rows.append(
                    f"| {split_name} | {int(counts[split_name]):,} | "
                    f"{start:%Y-%m-%d} → {end:%Y-%m-%d} |"
                )
            purge_days = int(splits_cfg.get("purge_days", 30))
            train_end = splits_cfg.get("train_end", "2018-12-31")
            val_end = splits_cfg.get("val_end", "2022-12-31")
            use_class_weights = models_cfg.get("use_class_weights", False)

            st.markdown(
                "\n\n".join(
                    [
                        "**Splits are chronological and purged, never random.** "
                        f"Training ends at {train_end} and validation at {val_end}; the "
                        f"test split covers 2023–2026 and is scored exactly once.",
                        "| Split | Rows | Observation months |\n|---|---|---|\n"
                        + "\n".join(split_rows),
                        f"**Purge.** A {purge_days}-day window is removed at each "
                        f"boundary, so the {horizon_days}-day label of one split cannot "
                        "reach into the next — no row is trained on a label it could "
                        "not have known.",
                        "**Thresholds.** Decision thresholds are tuned on validation "
                        "only, then frozen; the threshold column above is that frozen "
                        "operating point, not a test-set choice.",
                        f"**Class weighting is disabled** "
                        f"(`use_class_weights: {use_class_weights}` in config.yaml) "
                        "because a weighted fit destroyed calibration: it predicted a "
                        "mean probability of 0.28 against a 0.032 base rate, i.e. a "
                        "Brier Skill Score of about −4 versus counting history. "
                        "Balancing forces the effective prior toward 50/50, so the "
                        "probabilities stop meaning anything.",
                    ]
                )
            )

    # --- standing scientific caveat ------------------------------------------
    st.markdown(
        theme.limitations_html(
            "Metrics describe how well a model estimates probabilities on held-out "
            "historical rows — they do <b>not</b> make it possible to predict the time, "
            "location, or magnitude of an individual earthquake. Skill is measured "
            "against a climatological baseline on this catalog only: catalog "
            "completeness varies by region and era, so a model that beats counting "
            "history here has not been shown to beat it anywhere else."
        ),
        unsafe_allow_html=True,
    )


main()
