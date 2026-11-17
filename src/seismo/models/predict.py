"""Inference helpers: turn a persisted model into per-row probabilities.

Kept deliberately thin — all metric logic lives in ``evaluate``, all fitting in
``train``. This module's job is to guarantee that a model is scored on exactly
the feature columns it was trained on, in the same order, which is the most
common silent failure when a model is reloaded elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..errors import ValidationError
from ..geospatial.analysis import assign_regions
from .train import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, load_feature_frame, split_frame


def predict_positive(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return P(target=1) for every row, honouring the model's own feature order.

    Order mismatch is a real hazard: a tree model silently scores wrong values
    if columns are reordered after training. We therefore pass the *stored*
    column list and let ``_design`` assert each one exists.
    """
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"Frame is missing features the model requires: {missing}",
            hint="Rebuild features with `seismo features build` — the model and "
                 "dataset disagree about the schema.",
        )
    probabilities = model.predict_proba(frame)[:, 1]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValidationError(
            f"Model emitted {int((~np.isfinite(probabilities)).sum())} non-finite probabilities",
            hint="Inspect the training features for degenerate columns.",
        )
    return np.clip(probabilities, 0.0, 1.0)


def scored_frame(
    cfg: Config,
    model: Any,
    feature_cols: list[str],
    *,
    split: str,
    frame: pd.DataFrame | None = None,
    with_regions: bool = True,
) -> pd.DataFrame:
    """One row per prediction: identifiers, truth, probability, region, period.

    Returns the columns every downstream report needs — metrics, calibration,
    and slices all read from this single object rather than re-deriving it.
    """
    if split not in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        raise ValidationError(
            f"Unknown split {split!r}",
            hint="Use --split train, --split val (default) or --split test.",
        )
    if frame is None:
        frame = load_feature_frame(cfg)
    frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValidationError(
            f"Split {split!r} contains no rows",
            hint="Check splits.train_end / splits.val_end in config.yaml.",
        )

    out = pd.DataFrame({
        "obs_date": frame["obs_date"].to_numpy(),
        "cell_id": frame["cell_id"].to_numpy(),
        "latitude_center": frame["latitude_center"].to_numpy(),
        "longitude_center": frame["longitude_center"].to_numpy(),
        "target": frame["target"].to_numpy(dtype=np.int64),
        "prob": predict_positive(model, frame, feature_cols),
    })

    if with_regions:
        out["region"] = assign_regions(out["latitude_center"], out["longitude_center"])
    out["year"] = pd.DatetimeIndex(out["obs_date"]).year
    out["period"] = pd.cut(
        out["year"], bins=[2000, 2012, 2018, 2022, 2100],
        labels=["2001–2012", "2013–2018", "2019–2022", "2023–2026"],
    ).astype(str)
    return out.reset_index(drop=True)


def reference_probabilities(
    cfg: Config,
    frame: pd.DataFrame,
    split: str,
    feature_cols: list[str],
) -> np.ndarray:
    """Climatological-rate forecasts for the same rows — the Brier benchmark.

    Fitted on **train only**, then applied to the requested split, so the
    reference never learns from the rows it is scoring.
    """
    from .baselines import ClimatologicalRate

    splits = split_frame(frame)
    reference = ClimatologicalRate(
        prior_strength=float(
            (cfg.section("models") or {}).get("climatology_prior_strength", 50.0)
        )
    )
    reference.fit(splits[SPLIT_TRAIN])
    target_rows = frame[frame["split"] == split]
    return reference.predict_proba(target_rows)[:, 1]
