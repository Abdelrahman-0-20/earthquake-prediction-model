"""Flat KPI cards — thin borders, no shadows, no gradients."""

from __future__ import annotations

import streamlit as st

from components.theme import kpi_html


def render_kpis(cards: list[tuple[str, str, str]], *, columns: int = 5) -> None:
    """Render a row of KPI cards.

    ``cards`` is a list of ``(label, value, note)`` triples. Values are
    pre-formatted strings so pages control their own units and precision —
    the card component never guesses.
    """
    if not cards:
        return
    for start in range(0, len(cards), columns):
        row = cards[start:start + columns]
        cols = st.columns(len(row))
        for col, (label, value, note) in zip(cols, row, strict=False):
            with col:
                st.markdown(kpi_html(label, value, note), unsafe_allow_html=True)


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_mag(value: float) -> str:
    return f"M{value:.1f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"
