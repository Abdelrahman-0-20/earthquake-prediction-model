"""Evaluation metrics for probabilistic forecasting.

Why these metrics and not accuracy
----------------------------------
The positive class occurs in only 2.92% of rows, so a model that always
predicts 0 scores 97.08% accuracy and is worthless. Discrimination is reported
with **PR-AUC** (primary under imbalance) and ROC-AUC; probabilistic quality
with **Brier Score** and — crucially — **Brier Skill Score against a reference
forecast**, because a Brier score of 0.028 means nothing in isolation. A model
must beat the naive climatological rate to be worth deploying.

Calibration is first-class: a forecast that says "20%" must verify ~20% of the
time. `expected_calibration_error` and `calibration_table` make that testable.

Every function here is pure — same inputs, same outputs, no file or model I/O —
so the science is unit-testable in isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import ValidationError

DEFAULT_CALIBRATION_BINS = 10
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _as_arrays(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape:
        raise ValidationError(
            f"y_true shape {y.shape} != y_prob shape {p.shape}",
            hint="Both arrays must align row-for-row before scoring.",
        )
    if y.size == 0:
        raise ValidationError(
            "Cannot evaluate an empty prediction set",
            hint="Check that the requested split actually contains rows.",
        )
    if not np.isfinite(p).all():
        raise ValidationError(
            f"{int((~np.isfinite(p)).sum())} predictions are NaN or infinite",
            hint="A model emitted non-finite probabilities — inspect training "
                 "features for constant/degenerate columns.",
        )
    if p.min() < 0 or p.max() > 1:
        raise ValidationError(
            f"Probabilities outside [0, 1]: min={p.min():.4f} max={p.max():.4f}",
            hint="Predictions must be calibrated probabilities, not raw scores; "
                 "use predict_proba, not decision_function.",
        )
    return y.astype(np.int64), p


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def roc_auc(y_true: Any, y_prob: Any) -> float:
    """Area under the ROC curve."""
    from sklearn.metrics import roc_auc_score

    y, p = _as_arrays(y_true, y_prob)
    if np.unique(y).size < 2:
        raise ValidationError(
            "ROC-AUC is undefined when the set contains only one class",
            hint="Choose a split/window that has both positive and negative rows.",
        )
    return float(roc_auc_score(y, p))


def pr_auc(y_true: Any, y_prob: Any) -> float:
    """Area under the precision-recall curve (average precision).

    Primary discrimination metric under class imbalance.
    """
    from sklearn.metrics import average_precision_score

    y, p = _as_arrays(y_true, y_prob)
    if np.unique(y).size < 2:
        raise ValidationError(
            "PR-AUC is undefined when the set contains only one class",
            hint="Choose a split/window that has both positive and negative rows.",
        )
    return float(average_precision_score(y, p))


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Mean squared error of the probability — lower is better.

    Decomposes naturally into calibration + refinement, which is why it is the
    headline probabilistic metric for this project.
    """
    y, p = _as_arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y_true: Any, y_prob: Any, y_ref: Any, p_ref: Any) -> float:
    """1 − BS(model) / BS(reference).

    0 = no better than the reference, < 0 = *worse* than the reference, 1 =
    perfect. Reporting this (not just Brier) is what makes an honest
    comparison against the climatological baseline possible.
    """
    model = brier_score(y_true, y_prob)
    reference = brier_score(y_ref, p_ref)
    if reference <= 0:
        raise ValidationError(
            "Reference Brier score is zero — skill score is undefined",
            hint="A perfect reference implies the target is trivial; check the "
                 "reference forecast being used.",
        )
    return float(1.0 - model / reference)


def calibration_table(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Reliability data: predicted probability vs. observed frequency per bin.

    ``strategy='uniform'`` uses equal-width bins (classic reliability diagram);
    ``'quantile'`` gives equal-count bins so sparse high-probability regions
    still have enough rows to measure.
    """
    y, p = _as_arrays(y_true, y_prob)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    elif strategy == "quantile":
        # Rank-based binning; ties land in the same bin by construction.
        ranks = pd.Series(p).rank(method="first").to_numpy()
        bin_idx = np.clip(
            ((ranks - 1) / len(p) * n_bins).astype(int), 0, n_bins - 1
        )
    else:
        raise ValidationError(
            f"Unknown calibration strategy {strategy!r}",
            hint="Use strategy='uniform' or strategy='quantile'.",
        )

    frame = pd.DataFrame({"bin": bin_idx, "p": p, "y": y})
    grouped = frame.groupby("bin", sort=True).agg(
        count=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_frequency=("y", "mean"),
        bin_lower=("p", "min"),
        bin_upper=("p", "max"),
    )
    grouped["abs_gap"] = (grouped["mean_predicted"] - grouped["observed_frequency"]).abs()
    grouped["weight"] = grouped["count"] / len(frame)
    return grouped.reset_index()[
        ["bin", "count", "weight", "bin_lower", "bin_upper",
         "mean_predicted", "observed_frequency", "abs_gap"]
    ]


def expected_calibration_error(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    strategy: str = "uniform",
) -> float:
    """Weighted mean absolute gap between predicted and observed frequency."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return float((table["abs_gap"] * table["weight"]).sum())


def find_optimal_threshold(
    y_true: Any,
    y_prob: Any,
    *,
    objective: str = "f1",
) -> tuple[float, float]:
    """Tune the decision threshold on a validation set.

    Returns ``(threshold, score)``. Never call this on the test set — tuning
    there would leak the holdout into model selection.
    """
    from sklearn.metrics import precision_recall_curve

    y, p = _as_arrays(y_true, y_prob)
    if objective == "f1":
        precision, recall, thresholds = precision_recall_curve(y, p)
        # precision_recall_curve returns thresholds aligned to the first
        # len(thresholds) entries of precision/recall. `divide(..., where=)`
        # is required rather than np.where: numpy evaluates every branch, so a
        # plain 2pr/(p+r) would still raise on 0/0 before np.where could pick.
        precision, recall = precision[:-1], recall[:-1]
        denominator = precision + recall
        f1 = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        best = int(np.argmax(f1))
        return float(thresholds[best]), float(f1[best])

    if objective in ("youden", "j"):
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y, p)
        best = int(np.argmax(tpr - fpr))
        return float(thresholds[best]), float(tpr[best] - fpr[best])

    raise ValidationError(
        f"Unknown threshold objective {objective!r}",
        hint="Use objective='f1' (precision/recall balance) or "
             "objective='youden' (Youden's J).",
    )


def classification_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Threshold-dependent view: precision, recall, F1, confusion counts."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    y, p = _as_arrays(y_true, y_prob)
    y_pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def evaluate_predictions(
    y_true: Any,
    y_prob: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    reference_prob: Any | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """Full metric bundle for one model on one split."""
    y, p = _as_arrays(y_true, y_prob)

    bundle: dict[str, Any] = {
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
        "mean_predicted": float(p.mean()),
    }
    bundle.update(classification_metrics(y, p, threshold=threshold))

    if reference_prob is not None:
        ref = np.asarray(reference_prob, dtype=np.float64)
        bundle["reference_brier"] = brier_score(y, ref)
        bundle["brier_skill_score"] = brier_skill_score(y, p, y, ref)
        bundle["reference_mean_predicted"] = float(ref.mean())

    return bundle


def slice_metrics(
    frame: pd.DataFrame,
    *,
    prob_col: str = "prob",
    target_col: str = "target",
    by: str = "region",
    min_rows: int = 500,
    reference: pd.Series | None = None,
) -> pd.DataFrame:
    """Metric breakdown grouped by a categorical column.

    Regions (or periods) with fewer than ``min_rows`` are reported but flagged
    as unreliable rather than silently dropped — small-sample metrics that look
    precise are a common way to over-claim.
    """
    if by not in frame.columns:
        raise ValidationError(
            f"Cannot slice by {by!r} — column not in frame",
            hint=f"Available columns: {', '.join(map(str, frame.columns[:20]))}",
        )

    rows = []
    for key, group in frame.groupby(by, sort=False, dropna=False):
        y = group[target_col].to_numpy()
        p = group[prob_col].to_numpy(dtype=np.float64)
        if len(y) < 2 or np.unique(y).size < 2:
            rows.append({
                by: key, "n_rows": len(y), "positive_rate": float(y.mean()) if len(y) else 0.0,
                "pr_auc": None, "roc_auc": None, "brier": brier_score(y, p) if len(y) else None,
                "reliable": False, "note": "single class or <2 rows",
            })
            continue

        ref = reference.loc[group.index] if reference is not None else None
        metrics = evaluate_predictions(y, p, reference_prob=ref)
        rows.append({
            by: key,
            "n_rows": metrics["n_rows"],
            "positive_rate": metrics["positive_rate"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
            "brier_skill_score": metrics.get("brier_skill_score"),
            "ece": metrics["ece"],
            "reliable": len(y) >= min_rows,
        })

    result = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    return result.reset_index(drop=True)
