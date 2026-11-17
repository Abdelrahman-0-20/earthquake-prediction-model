"""Model subpackage.

``MODEL_NAMES`` lives here rather than in ``train`` so the CLI can print help
text without importing pandas/joblib on the hot path — `seismo --help` must
stay fast, and every millisecond of import cost is paid on every invocation.
"""

from __future__ import annotations

#: Baselines: simple, explainable, and the bar every ML model must beat.
BASELINES: tuple[str, ...] = (
    "baseline_global_rate",
    "baseline_climatology",
    "baseline_poisson",
)

#: Trained classifiers.
ML_MODELS: tuple[str, ...] = (
    "logreg",
    "random_forest",
    "xgboost",
)

#: Canonical model names accepted by the CLI (training order).
MODEL_NAMES: tuple[str, ...] = BASELINES + ML_MODELS

__all__ = ["BASELINES", "ML_MODELS", "MODEL_NAMES"]
