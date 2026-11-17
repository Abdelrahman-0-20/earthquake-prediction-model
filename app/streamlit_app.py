"""SeismoAI — application shell.

Run with::

    streamlit run app/streamlit_app.py

The shell owns three things and nothing else: page configuration, the
sidebar (navigation context, artifact status, standing limitations), and a
graceful path when an artifact is missing — a build instruction, never a
stack trace. Page logic lives in ``app/pages/`` and never reads the
filesystem directly; it goes through ``data_access``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the pipeline package importable when the app is launched from any cwd,
# and make sibling app modules (components, data_access) importable too.
ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "app"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import streamlit as st  # noqa: E402
from components.theme import ACCENT, CAPTION_SIZE, LIGHT, MUTED, NAVY  # noqa: E402

st.set_page_config(
    page_title="SeismoAI — Seismic Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- theme -----------------------------------------------------------------
st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{ font-family: Inter, 'Source Sans 3', sans-serif; }}
      h1 {{ color: {NAVY}; font-weight: 650; letter-spacing: -0.01em; }}
      h2, h3 {{ color: {NAVY}; font-weight: 600; }}
      section[data-testid="stSidebar"] {{
          background: {LIGHT};
          border-right: 1px solid #DCE4EC;
      }}
      div[data-testid="stMetric"] {{
          background: #FFFFFF; border: 1px solid {LIGHT};
          border-radius: 6px; padding: 12px 14px;
      }}
      div[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 600; }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 11px;
          text-transform: uppercase; letter-spacing: .05em; }}
      .stButton > button {{
          border: 1px solid {LIGHT}; background: #FFFFFF;
          color: {NAVY}; border-radius: 5px;
      }}
      .stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
      div[data-testid="stExpander"] {{ border: 1px solid {LIGHT};
          border-radius: 6px; }}
      footer {{ color: {MUTED}; font-size: {CAPTION_SIZE}px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- helpers ---------------------------------------------------------------

def _sidebar_header() -> None:
    st.sidebar.markdown(
        f"""
        <div style="padding:4px 0 2px 2px;">
          <div style="color:{NAVY};font-size:21px;font-weight:700;
                      letter-spacing:-0.01em;">SeismoAI</div>
          <div style="color:{MUTED};font-size:11.5px;margin-top:-2px;">
            Seismic intelligence &amp; geospatial analysis
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


def _artifact_panel() -> None:
    """Show what is available and, if something is missing, how to build it."""
    import data_access

    status = data_access.artifact_status()
    missing = status[~status["available"]]

    with st.sidebar.expander("Data status", expanded=not missing.empty):
        if missing.empty:
            st.caption("✓ All artifacts available")
        else:
            for _, row in missing.iterrows():
                st.error(f"**{row['artifact']}**\n\n`{row['build_command']}`")


def _limitations_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="color:{NAVY};font-size:11.5px;line-height:1.55;
                    background:#FFFFFF;border-left:3px solid {ACCENT};
                    border-radius:4px;padding:10px 12px;">
          <strong style="font-size:10.5px;letter-spacing:.05em;
                         text-transform:uppercase;color:{ACCENT};">
            Standing limitations</strong><br>
          Probabilities are estimated from historical observations — this system
          does <b>not</b> predict the time, location, or magnitude of individual
          earthquakes. Historical activity is <b>not</b> disaster risk: risk
          additionally requires exposure and vulnerability. Catalog completeness
          varies by region and era, so absence of records is not evidence of
          absence of seismicity.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Data: USGS ComCat · Models: scikit-learn / XGBoost · "
        "Metrics verified on a held-out 2023–2026 test split."
    )


# --- shell -----------------------------------------------------------------

def _render_missing(requirements) -> None:
    st.error("Some artifacts needed by this page have not been built yet.")
    for req in requirements:
        st.markdown(f"**{req.label}**")
        st.code(req.command, language="bash")
    st.stop()


def main() -> None:
    _sidebar_header()
    _artifact_panel()
    _limitations_footer()

    try:
        import data_access

        missing = data_access.require_available()
        if missing:
            st.warning(
                "Running with a partial dataset — some pages will show build "
                "instructions instead of content."
            )

        if st.session_state.get("_booted") is None:
            st.session_state["_booted"] = True

        # Home content — the six analytical pages live in app/pages/.
        st.title("Seismic Intelligence Platform")
        st.markdown(
            """
            A data-driven analysis of **707,438 earthquakes** recorded by the
            USGS between 2000 and 2026 — engineered end to end from raw API
            ingestion through validated datasets, geospatial analysis, and
            probabilistic forecasting models evaluated on a held-out test set.

            Use the pages in the sidebar:

            | Page | What it answers |
            |---|---|
            | **Overview** | How much activity, where, and how is it distributed? |
            | **Seismic Map** | What does the global footprint look like, filtered? |
            | **Historical Analysis** | How has recorded activity changed over 26 years? |
            | **Forecasting** | What probability does the model assign, and how confident? |
            | **Geospatial Analysis** | Where is historical activity concentrated? |
            | **Model Performance** | Does the model beat counting history — and is it calibrated? |
            """
        )

        st.markdown("---")
        _home_stats()

    except Exception as exc:  # noqa: BLE001 - render, never crash
        from seismo.errors import MissingArtifact, SeismoError

        if isinstance(exc, MissingArtifact):
            st.error(exc.render())
            st.stop()
        if isinstance(exc, SeismoError):
            st.error(exc.render())
            st.stop()
        raise


def _home_stats() -> None:
    """Row of headline numbers — all read through data_access."""
    import data_access
    from components.kpi_cards import render_kpis

    try:
        catalog = data_access.load_processed()
        comparison = data_access.load_model_comparison("test")
        best = comparison.iloc[0]

        render_kpis([
            ("Recorded events", f"{len(catalog):,}", "2000-01 → 2026-11"),
            ("Magnitude range",
             f"{catalog['mag'].min():.1f} – {catalog['mag'].max():.1f}", ""),
            ("Strongest event",
             f"M{catalog['mag'].max():.1f}",
             str(catalog.loc[catalog['mag'].idxmax(), 'place'])[:48]),
            ("Best model", str(best["model"]),
             f"test BSS {best['brier_skill_score']:+.4f}"),
            ("Models beating baseline",
             f"{int((comparison['brier_skill_score'] > 0).sum())} / {len(comparison)}",
             "Brier Skill Score > 0"),
        ])
    except Exception as exc:  # noqa: BLE001
        st.info(f"Headline statistics unavailable — {exc}")


if __name__ == "__main__":
    main()
