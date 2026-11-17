"""Design tokens and shared chart styling.

One source of truth for the visual system so all six pages look like one
product: deep navy text, blue accent for data, light-gray gridlines, flat
cards with thin borders. No gradients, no decorative animation, no emojis —
this is a scientific instrument.
"""

from __future__ import annotations

# --- palette -------------------------------------------------------------
NAVY = "#14263D"        # primary text / titles
WHITE = "#FFFFFF"
LIGHT = "#E9EEF3"       # surfaces, gridlines
ACCENT = "#2583A6"      # data, links, primary actions
ALERT = "#C0392B"       # negative skill, warnings
MUTED = "#6B7A8C"       # secondary text
POSITIVE = "#1E8E5A"    # positive skill

# Sequential/diverging palettes that read correctly in print and for
# colour-vision deficiency (viridis/cividis family, not jet).
DENSITY_CMAP = "viridis"
HEAT_CMAP = "YlOrRd"

SERIES_COLORS = [ACCENT, NAVY, POSITIVE, ALERT, "#8E6FB8", "#D98C2B", "#4A6FA5"]

# --- typography ----------------------------------------------------------
FONT_FAMILY = "Inter, 'Source Sans 3', -apple-system, 'Segoe UI', sans-serif"
TITLE_SIZE = 20
SUBTITLE_SIZE = 13
CAPTION_SIZE = 11


def base_layout(
    title: str | None = None,
    *,
    height: int | None = None,
    y_title: str | None = None,
    x_title: str | None = None,
    y_log: bool = False,
) -> dict:
    """Plotly layout template applied to every chart for consistency."""
    layout: dict = {
        "template": "plotly_white",
        "font": {"family": FONT_FAMILY, "color": NAVY, "size": 13},
        "title": {
            "text": title,
            "font": {"size": 16, "color": NAVY, "family": FONT_FAMILY},
            "x": 0.01,
            "xanchor": "left",
        },
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "margin": {"l": 60, "r": 24, "t": 60 if title else 30, "b": 50},
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12}},
            "gridcolor": LIGHT,
            "linecolor": LIGHT,
            "zerolinecolor": LIGHT,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12}},
            "gridcolor": LIGHT,
            "linecolor": LIGHT,
            "zerolinecolor": LIGHT,
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12},
            "orientation": "h",
            "y": 1.12,
            "x": 0,
            "xanchor": "left",
        },
        "hoverlabel": {
            "bgcolor": WHITE,
            "font": {"family": FONT_FAMILY, "color": NAVY, "size": 12},
            "bordercolor": LIGHT,
        },
    }
    if height:
        layout["height"] = height
    if y_log:
        layout["yaxis"]["type"] = "log"
    return layout


def kpi_html(label: str, value: str, note: str = "", *, accent: bool = False) -> str:
    """Flat KPI card: thin border, no shadow, no gradient."""
    border = f"2px solid {ACCENT}" if accent else f"1px solid {LIGHT}"
    note_html = (
        f'<div style="font-size:11px;color:{MUTED};margin-top:4px;">{note}</div>'
        if note else ""
    )
    return f"""
    <div style="
        background:{WHITE};
        border:{border};
        border-radius:6px;
        padding:14px 16px;
        font-family:{FONT_FAMILY};
    ">
      <div style="font-size:11px;letter-spacing:.06em;text-transform:uppercase;
                  color:{MUTED};">{label}</div>
      <div style="font-size:26px;font-weight:600;color:{NAVY};line-height:1.25;
                  margin-top:2px;">{value}</div>
      {note_html}
    </div>"""


def limitations_html(text: str) -> str:
    """Standing scientific-caveat panel used on forecasting/geospatial pages."""
    return f"""
    <div style="
        background:{LIGHT};
        border-left:3px solid {ACCENT};
        border-radius:4px;
        padding:12px 16px;
        font-family:{FONT_FAMILY};
        font-size:12.5px;
        color:{NAVY};
        line-height:1.55;
    ">
      <strong style="font-size:11px;letter-spacing:.06em;text-transform:uppercase;
                    color:{ACCENT};">Scientific limitation</strong><br>{text}
    </div>"""
