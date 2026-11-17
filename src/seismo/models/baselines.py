"""Reference models: the bar every ML model must clear.

Three baselines, deliberately simple and explainable:

``climatology``
    Per-cell historical frequency with empirical-Bayes shrinkage toward the
    global rate. This is *the* baseline: if a trained model cannot beat it on
    Brier Skill Score, the model has added no value over counting.

``poisson``
    A log-linear (Poisson) rate model over the trailing features, converted to
    a probability via the Poisson "at least one" identity.

``random``
    Constant global rate — a floor check. If anything beats this only
    marginally, the features carry little signal.

Interface contract
------------------
Every model implements::

    fit(frame, feature_cols)         # feature_cols optional for cell-based models
    predict_proba(frame) -> (n, 2)    # columns [P(0), P(1)]

so training and evaluation code can treat baselines and ML models uniformly.
Models read only the columns they need: ``climatology`` needs ``cell_id``,
everything else uses the configured feature list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import ValidationError

EPS = 1e-9


def _clip(probabilities: np.ndarray) -> np.ndarray:
    """Keep probabilities strictly inside (0, 1) so logs/ratios stay finite."""
    return np.clip(probabilities, EPS, 1.0 - EPS)


# ---------------------------------------------------------------------------
# Climatological rate — the baseline of record
# ---------------------------------------------------------------------------

class ClimatologicalRate:
    """Empirical per-cell frequency, shrunk toward the global rate.

    Posterior mean for a cell observed ``n`` times with ``k`` positives::

        rate = (k + prior_strength * global_rate) / (n + prior_strength)

    Small-sample cells therefore move toward the global rate instead of
    reporting an overconfident 0/1, which is exactly the failure mode that
    inflates Brier scores on rare events. Cells never seen in training fall
    back to the global rate — no cell may be scored on knowledge it lacks.
    """

    name = "baseline_climatology"
    kind = "baseline"
    needs_cell_id = True

    def __init__(self, prior_strength: float = 50.0) -> None:
        self.prior_strength = float(prior_strength)
        self.cell_rate_: dict[int, float] = {}
        self.global_rate_: float = 0.0
        self.n_train_rows_: int = 0

    def fit(
        self, frame: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> ClimatologicalRate:
        # `feature_cols` is accepted and ignored: climatology conditions on
        # cell_id alone. The parameter exists so every model satisfies the
        # same `fit(frame, feature_cols)` contract and training never branches.
        if "cell_id" not in frame.columns or "target" not in frame.columns:
            raise ValidationError(
                "ClimatologicalRate.fit requires 'cell_id' and 'target' columns",
                hint="Train on the features dataset built by `seismo features build`.",
            )
        self.n_train_rows_ = int(len(frame))
        self.global_rate_ = float(frame["target"].mean())
        if not 0 < self.global_rate_ < 1:
            raise ValidationError(
                f"Global positive rate {self.global_rate_} is degenerate",
                hint="Training data must contain both classes.",
            )

        stats = frame.groupby("cell_id")["target"].agg(["sum", "count"])
        k = self.prior_strength * self.global_rate_
        self.cell_rate_ = {
            int(cell): float((row["sum"] + k) / (row["count"] + self.prior_strength))
            for cell, row in stats.iterrows()
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.cell_rate_:
            raise ValidationError(
                "ClimatologicalRate is not fitted",
                hint="Call fit() before predict_proba().",
            )
        rates = frame["cell_id"].map(self.cell_rate_)
        rates = rates.fillna(self.global_rate_).to_numpy(dtype=np.float64)
        rates = _clip(rates)
        return np.column_stack([1.0 - rates, rates])


# ---------------------------------------------------------------------------
# Poisson rate — statistical baseline
# ---------------------------------------------------------------------------

class PoissonRate:
    """Log-linear Poisson model of the expected qualifying-event count.

    A PoissonRegressor (log link, L2) is fitted on the trailing features, and
    the probability of *at least one* event follows the Poisson identity::

        P(N >= 1) = 1 - exp(-lambda)

    Approximation note: the target is observed as a 0/1 indicator rather than
    a true count, so cells with 2+ qualifying events are censored to 1. For
    rare events (2.92% base rate) ``1 - e^-lambda`` and ``lambda`` differ by
    well under a percent, and the identity is retained because it states the
    count assumption explicitly rather than hiding it.
    """

    name = "baseline_poisson"
    kind = "baseline"
    needs_cell_id = False

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.model_: Any = None
        self.feature_cols_: list[str] = []

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> PoissonRate:
        from sklearn.linear_model import PoissonRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        _require_target(frame)
        self.feature_cols_ = list(feature_cols)
        X = _design(frame, self.feature_cols_)
        y = frame["target"].to_numpy(dtype=np.float64)
        # Standardisation is required here, not optional: the log link computes
        # exp(X @ coef), and raw features span 0–12,458 (counts), 0–9,770
        # (days) and ±180 (coordinates). Unscaled inputs overflow the IRLS
        # matmul and emit NaN gradients before convergence.
        self.model_ = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter, tol=1e-8),
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        _require_fitted(self.model_, self.name)
        X = _design(frame, self.feature_cols_)
        lam = np.clip(self.model_.predict(X), 0.0, 50.0)
        p1 = _clip(1.0 - np.exp(-lam))
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# Constant floor
# ---------------------------------------------------------------------------

class GlobalRate:
    """Constant prediction equal to the training base rate."""

    name = "baseline_global_rate"
    kind = "baseline"
    needs_cell_id = False

    def __init__(self) -> None:
        self.rate_: float = 0.0

    def fit(self, frame: pd.DataFrame, feature_cols: list[str] | None = None) -> GlobalRate:
        _require_target(frame)
        self.rate_ = float(frame["target"].mean())
        if not 0 < self.rate_ < 1:
            raise ValidationError(
                f"Global positive rate {self.rate_} is degenerate",
                hint="Training data must contain both classes.",
            )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not 0 < self.rate_ < 1:
            raise ValidationError(
                f"{self.name} is not fitted (rate={self.rate_})",
                hint="Call fit() before predict_proba().",
            )
        rate = np.full(len(frame), self.rate_, dtype=np.float64)
        return np.column_stack([1.0 - rate, rate])


# ---------------------------------------------------------------------------
# Generic sklearn wrapper
# ---------------------------------------------------------------------------

class SklearnProbabilistic:
    """Uniform wrapper around any classifier exposing ``predict_proba``.

    Keeps the baseline interface (`fit(frame, cols)` / `predict_proba(frame)`)
    so training code never branches on model type.
    """

    needs_cell_id = False
    kind = "ml"

    def __init__(self, estimator: Any, name: str) -> None:
        self.model_ = estimator
        self.name = name
        self.feature_cols_: list[str] = []

    def fit(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        **fit_kwargs: Any,
    ) -> SklearnProbabilistic:
        """Fit the wrapped estimator.

        ``**fit_kwargs`` forwards estimator-specific arguments (XGBoost's
        ``eval_set``/``verbose`` for early stopping). Callers must only pass
        kwargs the underlying estimator accepts — sklearn Pipelines reject
        ``eval_set``, so training code guards it per model.
        """
        _require_target(frame)
        self.feature_cols_ = list(feature_cols)
        X = _design(frame, self.feature_cols_)
        y = frame["target"].to_numpy(dtype=np.int64)
        if np.unique(y).size < 2:
            raise ValidationError(
                "Training data contains a single class",
                hint="Check the split: both target=0 and target=1 rows are required.",
            )
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols_:
            raise ValidationError(
                f"{self.name} is not fitted",
                hint="Call fit() before predict_proba().",
            )
        X = _design(frame, self.feature_cols_)
        return self.model_.predict_proba(X)

    @property
    def estimator(self) -> Any:
        """Underlying estimator (for coefficient inspection / tree analysis)."""
        return getattr(self.model_, "steps", [(None, self.model_)])[-1][1] \
            if hasattr(self.model_, "steps") else self.model_


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _require_target(frame: pd.DataFrame) -> None:
    if "target" not in frame.columns:
        raise ValidationError(
            "Feature frame is missing the 'target' column",
            hint="Build features with `seismo features build`.",
        )


def _require_fitted(model: Any, name: str) -> None:
    if model is None:
        raise ValidationError(
            f"{name} is not fitted",
            hint="Call fit() before predict_proba().",
        )


def _design(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"Feature frame is missing columns: {missing}",
            hint="Rebuild features with `seismo features build` — the model was "
                 "trained on a different feature set.",
        )
    values = frame[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValidationError(
            f"{bad} non-finite values in the feature matrix",
            hint="Features must be finite; check the zero-fill convention in "
                 "docs/DATA_DICTIONARY.md.",
        )
    return values


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Convenience: column P(target=1) from ``predict_proba``."""
    return model.predict_proba(frame)[:, 1]
