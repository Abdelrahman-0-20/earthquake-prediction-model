"""Chart builders — every figure goes through the shared theme.

Pages call these instead of constructing Plotly figures directly, so axis
labels, units, colours, fonts and gridlines stay identical across all six
pages. A chart with an inconsistent label style is a defect in a scientific
product, not a cosmetic detail.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import (
    ACCENT,
    ALERT,
    LIGHT,
    MUTED,
    NAVY,
    POSITIVE,
    SERIES_COLORS,
    base_layout,
)


def _finalize(fig: go.Figure, layout: dict[str, Any]) -> go.Figure:
    fig.update_layout(**layout)
    return fig


# --- distributions -------------------------------------------------------

def magnitude_histogram(
    magnitudes: pd.Series, *, bin_size: float = 0.25, title: str | None = None
) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=magnitudes,
        xbins={"size": bin_size},
        marker_color=ACCENT,
        marker_line_color=NAVY,
        marker_line_width=0.4,
        name="events",
    ))
    fig.update_traces(hovertemplate="M%{x:.1f}<br>%{y:,} events<extra></extra>")
    return _finalize(fig, base_layout(
        title=title or "Magnitude distribution",
        x_title="Magnitude", y_title="Events", height=420,
    ))


def depth_histogram(depths: pd.Series, *, title: str | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=depths, xbins={"size": 10},
        marker_color=NAVY, marker_line_color=LIGHT, marker_line_width=0.4,
        name="events",
    ))
    fig.update_traces(hovertemplate="Depth %{x:.0f} km<br>%{y:,} events<extra></extra>")
    return _finalize(fig, base_layout(
        title=title or "Depth distribution",
        x_title="Depth (km)", y_title="Events", height=420,
    ))


# --- time series ---------------------------------------------------------

def time_series(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    y_title: str,
    x_title: str = "",
    rolling: int | None = None,
    color: str = ACCENT,
    height: int = 420,
    log_y: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame[x], y=frame[y], mode="lines",
        line={"color": color, "width": 1.6},
        name="monthly",
        hovertemplate="%{x|%b %Y}<br>%{y:,}<extra></extra>",
    ))
    if rolling and len(frame) >= rolling:
        smoothed = frame[y].rolling(rolling, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=frame[x], y=smoothed, mode="lines",
            line={"color": NAVY, "width": 2.6},
            name=f"{rolling}-period mean",
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        ))
    return _finalize(fig, base_layout(
        title=title, x_title=x_title, y_title=y_title, height=height, y_log=log_y,
    ))


# --- categorical ---------------------------------------------------------

def horizontal_bars(
    frame: pd.DataFrame,
    *,
    y: str,
    x: str,
    title: str,
    x_title: str,
    color: str = ACCENT,
    height: int | None = None,
    log_x: bool = False,
    text: str | None = None,
) -> go.Figure:
    ordered = frame.sort_values(x)
    fig = go.Figure(go.Bar(
        x=ordered[x], y=ordered[y], orientation="h",
        marker_color=color, marker_line_color=NAVY, marker_line_width=0.4,
        text=ordered[text] if text else None,
        textposition="outside",
        hovertemplate=f"%{{y}}<br>{x_title}: %{{x:,.3f}}<extra></extra>",
    ))
    return _finalize(fig, base_layout(
        title=title, x_title=x_title, y_title="", height=height or (380 + 16 * len(ordered)),
        y_log=log_x,
    ))


def grouped_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str,
    y_title: str,
    x_title: str = "",
    height: int = 420,
    log_y: bool = False,
) -> go.Figure:
    if group:
        fig = px.bar(frame, x=x, y=y, color=group, barmode="group",
                     color_discrete_sequence=SERIES_COLORS)
    else:
        fig = px.bar(frame, x=x, y=y, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_color=NAVY, marker_line_width=0.4)
    return _finalize(fig, base_layout(
        title=title, x_title=x_title, y_title=y_title, height=height, y_log=log_y,
    ))


# --- model-quality charts ------------------------------------------------

def reliability_diagram(
    calibration: pd.DataFrame,
    *,
    model: str,
    ece: float | None = None,
    brier: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"color": "#999999", "dash": "dash", "width": 1.4},
        name="perfect", hoverinfo="skip",
    ))
    subtitle = ""
    if ece is not None:
        subtitle += f"ECE {ece:.4f}"
    if brier is not None:
        subtitle += f" · Brier {brier:.4f}"
    fig.add_trace(go.Scatter(
        x=calibration["mean_predicted"], y=calibration["observed_frequency"],
        mode="lines+markers",
        line={"color": ACCENT, "width": 2.4}, marker={"size": 7},
        name=model,
        customdata=calibration[["count", "abs_gap"]],
        hovertemplate=(
            "Predicted %{x:.4f}<br>Observed %{y:.4f}<br>"
            "rows %{customdata[0]:,}<br>gap %{customdata[1]:.4f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=calibration["mean_predicted"], y=calibration["mean_predicted"],
        mode="lines", line={"color": LIGHT, "width": 0},
        showlegend=False, hoverinfo="skip",
    ))
    return _finalize(fig, base_layout(
        title=f"Reliability — {model}" + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
        x_title="Predicted probability", y_title="Observed frequency",
        height=560,
    ))


def skill_score_chart(comparison: pd.DataFrame) -> go.Figure:
    """Brier Skill Score with a hard zero line — the honest yardstick."""
    ordered = comparison.dropna(subset=["brier_skill_score"]).sort_values(
        "brier_skill_score"
    )
    colors = [POSITIVE if v > 0 else (NAVY if v == 0 else ALERT)
              for v in ordered["brier_skill_score"]]
    fig = go.Figure(go.Bar(
        x=ordered["brier_skill_score"], y=ordered["model"], orientation="h",
        marker_color=colors, marker_line_color=NAVY, marker_line_width=0.4,
        customdata=ordered[["brier"]],
        hovertemplate="%{y}<br>BSS %{x:.4f}<br>Brier %{customdata[0]:.5f}"
                      "<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=ALERT, line_dash="dash", line_width=1.4)
    return _finalize(fig, base_layout(
        title="Brier Skill Score vs climatological baseline",
        x_title="BSS  (>0 beats counting history, <0 is worse)", y_title="",
        height=420,
    ))


def metric_bars(comparison: pd.DataFrame, *, column: str, title: str,
                x_title: str, color: str = ACCENT) -> go.Figure:
    ordered = comparison.dropna(subset=[column]).sort_values(column)
    fig = go.Figure(go.Bar(
        x=ordered[column], y=ordered["model"], orientation="h",
        marker_color=color, marker_line_color=NAVY, marker_line_width=0.4,
        hovertemplate="%{y}<br>" + x_title + " %{x:.4f}<extra></extra>",
    ))
    return _finalize(fig, base_layout(
        title=title, x_title=x_title, y_title="", height=420,
    ))


def pr_curve(precision: list[float], recall: list[float], *, pr_auc: float,
             base_rate: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        line={"color": ACCENT, "width": 2.4}, name="model",
        hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=base_rate, line_dash="dash", line_color=MUTED,
                  annotation_text=f"base rate {base_rate:.3f}",
                  annotation_font_color=MUTED)
    return _finalize(fig, base_layout(
        title=f"Precision–Recall (PR-AUC {pr_auc:.4f})",
        x_title="Recall", y_title="Precision", height=480,
    ))


__all__ = [
    "grouped_bar", "horizontal_bars", "magnitude_histogram", "depth_histogram",
    "metric_bars", "pr_curve", "reliability_diagram", "skill_score_chart",
    "time_series",
]
