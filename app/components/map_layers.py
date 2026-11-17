"""Map layers — pre-aggregated, never 707k raw points.

Performance contract: the global map renders **aggregated** data (hexbin-style
grid counts or binned density), because streaming half a million points into
deck.gl makes the browser choke and the page useless. Raw points are only ever
drawn for a narrowed selection (a zoomed region with magnitude filters
applied), where the row count is genuinely small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk

from components.theme import NAVY

#: Conservative ceiling: beyond this we aggregate instead of drawing points.
MAX_RAW_POINTS = 4000

MAPBOX_STYLE_LIGHT = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"


def base_view(latitude: float = 20.0, longitude: float = 0.0, zoom: float = 1.4) -> pdk.ViewState:
    return pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0)


def activity_scatter(activity: pd.DataFrame, *, radius_scale: float = 0.9) -> pdk.Deck:
    """Per-cell activity sized by event count (Module 03 layer).

    Data is already aggregated to one row per cell by ``seismo geospatial
    build`` — at most ~11.5k rows, so it renders comfortably.
    """
    frame = activity.copy()
    frame["radius"] = (
        np.sqrt(frame["n_events"].astype(float)) * radius_scale
    ).clip(lower=150, upper=12_000)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[longitude_center, latitude_center]",
        get_radius="radius",
        radius_min_pixels=1.5,
        radius_max_pixels=45,
        get_fill_color=[37, 131, 166, 150],
        get_line_color=[20, 38, 61, 220],
        line_width_min_pixels=0.6,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "text": "{longitude_center}°, {latitude_center}°\n"
                "{n_events:,} events · max M{max_mag:.1f}",
        "style": {"color": "#14263D", "backgroundColor": "#FFFFFF"},
    }
    return pdk.Deck(
        layers=[layer],
        initial_view_state=base_view(),
        tooltip=tooltip,
        map_style=None,
    )


def event_scatter(events: pd.DataFrame, *, max_points: int = MAX_RAW_POINTS) -> pdk.Deck:
    """Raw event points — caller must have filtered to a small selection.

    If the selection still exceeds ``max_points`` we downsample rather than
    render a page that cannot be interacted with; the caller is told via the
    subtitle so the number on screen is never a silent lie.
    """
    frame = events
    if len(frame) > max_points:
        frame = frame.sample(n=max_points, random_state=0)

    frame = frame.copy()
    frame["color_r"] = np.where(frame["mag"] >= 5.0, 192, 37)
    frame["color_g"] = np.where(frame["mag"] >= 5.0, 57, 131)
    frame["color_b"] = np.where(frame["mag"] >= 5.0, 38, 166)
    frame["point_radius"] = (2.0 + (frame["mag"] - frame["mag"].min()) * 3.5).clip(2, 22)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[longitude, latitude]",
        get_radius="point_radius",
        radius_min_pixels=1.2,
        radius_max_pixels=26,
        get_fill_color="[color_r, color_g, color_b, 165]",
        get_line_color=[20, 38, 61, 200],
        line_width_min_pixels=0.4,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "text": "{place}\nM{mag} · {depth} km · {time}",
        "style": {"color": "#14263D", "backgroundColor": "#FFFFFF"},
    }
    return pdk.Deck(
        layers=[layer],
        initial_view_state=base_view(),
        tooltip=tooltip,
        map_style=None,
    )


def density_heatmap(events: pd.DataFrame, *, max_points: int = MAX_RAW_POINTS) -> pdk.Deck:
    """Heat layer over a filtered event selection.

    Note: despite being a *density* layer, the underlying rows are still sent
    to the browser, so selections larger than ``max_points`` are sampled —
    callers must disclose the sampled size (see ``MAX_RAW_POINTS``).
    """
    frame = events
    if len(frame) > max_points:
        frame = frame.sample(n=max_points, random_state=0)

    layer = pdk.Layer(
        "HeatmapLayer",
        data=frame,
        get_position="[longitude, latitude]",
        get_weight="mag",
        radius_pixels=45,
        intensity=0.7,
        threshold=0.03,
        color_range=[
            [255, 255, 217], [237, 248, 177], [199, 233, 180],
            [127, 205, 187], [65, 182, 196], [29, 145, 192],
            [34, 94, 168], [37, 47, 125], [8, 29, 88],
        ],
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=base_view(),
        map_style=None,
        tooltip={"text": "Filtered selection — aggregated density"},
    )


def hotspot_layer(activity: pd.DataFrame, *, top_n: int = 50) -> pdk.Deck:
    """Top-N cells by historical activity, emphasised over the base layer."""
    top = activity.nlargest(top_n, "n_events").copy()
    top["radius"] = (np.sqrt(top["n_events"].astype(float)) * 2.4).clip(800, 22_000)

    base = pdk.Layer(
        "ScatterplotLayer",
        data=activity,
        get_position="[longitude_center, latitude_center]",
        get_radius=600,
        radius_min_pixels=0.8,
        get_fill_color=[107, 122, 140, 70],
        pickable=False,
    )
    hot = pdk.Layer(
        "ScatterplotLayer",
        data=top,
        get_position="[longitude_center, latitude_center]",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=40,
        get_fill_color=[192, 57, 38, 165],
        get_line_color=[20, 38, 61, 220],
        line_width_min_pixels=0.8,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "text": "Top-{top_n} hotspot\n{n_events:,} events · M{max_mag:.1f} max",
        "style": {"color": "#14263D", "backgroundColor": "#FFFFFF"},
    }
    return pdk.Deck(
        layers=[base, hot],
        initial_view_state=base_view(),
        tooltip=tooltip,
        map_style=None,
    )


def region_box_layer(regions: dict[str, dict[str, tuple[float, float]]],
                     selected: list[str] | None = None) -> pdk.Deck | None:
    """Bounding boxes for named regions — drawn only when regions are chosen."""
    if not selected:
        return None
    import pydeck as _pdk

    polys = []
    for name in selected:
        if name not in regions:
            continue
        box = regions[name]
        (lat_lo, lat_hi), (lon_lo, lon_hi) = box["lat"], box["lon"]
        polys.append({
            "name": name,
            "polygon": [
                [lon_lo, lat_lo], [lon_hi, lat_lo],
                [lon_hi, lat_hi], [lon_lo, lat_hi], [lon_lo, lat_lo],
            ],
        })
    if not polys:
        return None

    layer = _pdk.Layer(
        "PolygonLayer",
        data=polys,
        get_polygon="polygon",
        get_fill_color=[37, 131, 166, 28],
        get_line_color=[20, 38, 61, 200],
        line_width_min_pixels=1.4,
        pickable=True,
        auto_highlight=True,
    )
    return _pdk.Deck(
        layers=[layer],
        initial_view_state=base_view(),
        tooltip={"text": "{name}", "style": {"color": NAVY, "backgroundColor": "#FFFFFF"}},
        map_style=None,
    )


ACCENT_RGB = [37, 131, 166]
ALERT_RGB = [192, 57, 38]

__all__ = [
    "activity_scatter", "base_view", "density_heatmap", "event_scatter",
    "hotspot_layer", "region_box_layer", "MAX_RAW_POINTS",
    "ACCENT_RGB", "ALERT_RGB", "MAPBOX_STYLE_LIGHT",
]
