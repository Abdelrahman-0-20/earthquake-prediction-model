"""Evaluation orchestration: metrics → calibration → slices → SHAP → reports.

What gets written
-----------------
``reports/model_metrics/<name>/<version>/``
    ``<split>.json``        full metric bundle for the evaluated split
    ``<split>_calibration.json``  reliability data
    ``<split>_by_region.json``    per-region metrics (pooled metrics can hide
                                  regional failure)
    ``<split>_by_period.json``    per-period metrics (drift over time)
    ``figures/*.png``       reliability, PR curve, ROC curve, SHAP summary

``models/<name>/<version>/metrics.json``
    validation metrics only. Deliberately: **promotion requires a prior
    evaluation**, and the test split never writes into the model directory —
    it is touched exactly once and lives in ``reports/`` alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..errors import ValidationError
from . import BASELINES, ML_MODELS
from .evaluate import (
    DEFAULT_THRESHOLD,
    calibration_table,
    evaluate_predictions,
    find_optimal_threshold,
    slice_metrics,
)
from .predict import reference_probabilities, scored_frame
from .train import (
    SPLIT_VAL,
    load_feature_frame,
    load_model,
)

log = logging.getLogger("seismo.reporting")

NAVY = "#14263D"
ACCENT = "#2583A6"
LIGHT = "#E9EEF3"
ALERT = "#C0392B"

#: Sampled because SHAP cost grows with both row count *and* tree size; 2,000
#: rows is ample for stable global importances while keeping evaluation in
#: seconds rather than minutes on a memory-constrained machine.
SHAP_SAMPLE_ROWS = 2000


@dataclass
class EvaluationResult:
    model: str
    version: str
    split: str
    threshold: float
    threshold_objective_f1: float
    metrics: dict[str, Any]
    calibration: list[dict[str, Any]] = field(default_factory=list)
    by_region: list[dict[str, Any]] = field(default_factory=list)
    by_period: list[dict[str, Any]] = field(default_factory=list)
    shap_top_features: list[dict[str, Any]] = field(default_factory=list)
    report_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    cfg: Config,
    name: str,
    *,
    version: str | None = None,
    split: str = SPLIT_VAL,
    frame: pd.DataFrame | None = None,
    threshold: float | None = None,
    write: bool = True,
    shap_enabled: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> EvaluationResult:
    """Score one model on one split and (optionally) write the full report."""
    model, metadata = load_model(cfg, name, version)
    version = metadata["version"]
    feature_cols = list(metadata["feature_cols"])

    if frame is None:
        frame = load_feature_frame(cfg)

    def note(message: str) -> None:
        if on_progress:
            on_progress(message)

    note(f"{name}/{version}: scoring {split}…")
    scored = scored_frame(cfg, model, feature_cols, split=split, frame=frame)

    # Reference forecasts — climatology fitted on train, applied to this split.
    reference = reference_probabilities(cfg, frame, split, feature_cols)

    # Threshold is tuned on validation only; test reuses the chosen threshold
    # so the holdout never participates in model selection.
    if threshold is None:
        if split == SPLIT_VAL:
            threshold, best_f1 = find_optimal_threshold(scored["target"], scored["prob"])
            note(f"{name}: optimal validation threshold {threshold:.4f} (F1 {best_f1:.4f})")
        else:
            threshold, best_f1 = _stored_threshold(cfg, name, version)
            note(f"{name}: reusing stored threshold {threshold:.4f} on {split}")

    metrics = evaluate_predictions(
        scored["target"], scored["prob"],
        threshold=threshold, reference_prob=reference,
    )
    metrics.update({"model": name, "version": version, "split": split})

    calibration = calibration_table(scored["target"], scored["prob"], n_bins=15)
    by_region = slice_metrics(scored, by="region", reference=_ref_series(scored, reference))
    by_period = slice_metrics(scored, by="period", reference=_ref_series(scored, reference))

    result = EvaluationResult(
        model=name,
        version=version,
        split=split,
        threshold=round(float(threshold), 6),
        threshold_objective_f1=metrics["f1"],
        metrics=metrics,
        calibration=calibration.to_dict(orient="records"),
        by_region=by_region.to_dict(orient="records"),
        by_period=by_period.to_dict(orient="records"),
    )

    if shap_enabled and name in ML_MODELS:
        note(f"{name}: computing SHAP on {min(SHAP_SAMPLE_ROWS, len(scored)):,} rows…")
        result.shap_top_features = _shap_importance(model, frame, feature_cols, split)
        metrics["shap_top_features"] = result.shap_top_features

    if write:
        result.report_dir = str(write_report(cfg, result, scored))
        if split == SPLIT_VAL:
            _write_model_metrics(cfg, name, version, metrics)
            _store_threshold(cfg, name, version, threshold)
        note(f"{name}/{version}: report → {result.report_dir}")

    return result


def evaluate_all(
    cfg: Config,
    *,
    names: tuple[str, ...] | list[str] | None = None,
    split: str = SPLIT_VAL,
    **kwargs: Any,
) -> list[EvaluationResult]:
    """Evaluate every requested model against one loaded frame.

    When specific models are requested their errors **propagate** — asking for
    one model and receiving a table of dashes is a silent failure. Only the
    sweep-all case isolates failures, so one broken model cannot hide the rest.
    """
    frame = load_feature_frame(cfg)
    requested = tuple(names) if names is not None else (*BASELINES, *ML_MODELS)
    propagate = names is not None

    results = []
    failures: list[str] = []
    for name in requested:
        try:
            results.append(evaluate_model(cfg, name, split=split, frame=frame, **kwargs))
        except Exception as exc:  # noqa: BLE001
            if propagate:
                raise
            log.warning("evaluation skipped for %s: %s", name, exc)
            failures.append(name)
            results.append(EvaluationResult(
                model=name, version="n/a", split=split, threshold=DEFAULT_THRESHOLD,
                threshold_objective_f1=0.0,
                metrics={"error": str(exc)},
            ))
    if not propagate and failures and len(failures) == len(requested):
        raise ValidationError(
            f"No model could be evaluated: {', '.join(failures)}",
            hint="Train them first with `seismo model train --all`.",
        )
    return results


def comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side metrics — the table the whole project exists to produce."""
    rows = []
    for r in results:
        m = dict(r.metrics)
        rows.append({
            "model": r.model,
            "split": r.split,
            "threshold": r.threshold,
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "brier": m.get("brier"),
            "brier_skill_score": m.get("brier_skill_score"),
            "ece": m.get("ece"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "n_rows": m.get("n_rows"),
            "error": m.get("error"),
        })
    table = pd.DataFrame(rows)
    return table.sort_values("brier_skill_score", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _shap_importance(
    model: Any,
    frame: pd.DataFrame,
    feature_cols: list[str],
    split: str,
    *,
    n_rows: int = SHAP_SAMPLE_ROWS,
) -> list[dict[str, Any]]:
    """Global SHAP importance on a deterministic sample of one split.

    Sampled because SHAP cost grows quickly with row count; 5,000 rows is
    plenty for stable global importances and keeps evaluation minutes, not
    hours. Sampling is seeded so results are reproducible.
    """
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        log.warning("shap unavailable: %s", exc)
        return []

    subset = frame[frame["split"] == split]
    if subset.empty:
        return []
    if len(subset) > n_rows:
        subset = subset.sample(n=n_rows, random_state=0)

    estimator = model.estimator
    X = subset[feature_cols].to_numpy(dtype=np.float64)

    try:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(X)
        # Shape varies by shap version and model type: a list of per-class
        # arrays, or a single (n, k) / (n, k, classes) / (classes, n, k) array.
        # Normalise to (n, k) for the positive class before reducing.
        if isinstance(raw, list):
            raw = raw[-1]
        values = np.asarray(raw)
        if values.ndim == 3:
            class_axis = next(
                (i for i, s in enumerate(values.shape) if s == 2), None
            )
            if class_axis is None:
                raise ValueError(f"cannot locate class axis in {values.shape}")
            values = np.take(values, 1, axis=class_axis)
        if values.ndim == 1:
            importance = np.abs(values)
        elif values.ndim == 2:
            importance = np.abs(values).mean(axis=0)
        else:
            raise ValueError(f"unexpected SHAP shape {values.shape}")
    except Exception as exc:  # noqa: BLE001 - fallback keeps evaluation alive
        log.warning("TreeExplainer failed (%s); using feature importances", exc)
        return _impurity_importance(model, feature_cols)

    order = np.argsort(importance)[::-1]
    total = float(importance.sum()) or 1.0
    return [
        {"feature": feature_cols[int(i)],
         "mean_abs_shap": float(importance[int(i)]),
         "share_of_total": float(importance[int(i)] / total)}
        for i in order
    ]


def _impurity_importance(model: Any, feature_cols: list[str]) -> list[dict[str, Any]]:
    estimator = model.estimator
    if not hasattr(estimator, "feature_importances_"):
        return []
    raw = np.asarray(estimator.feature_importances_, dtype=np.float64)
    total = float(raw.sum()) or 1.0
    order = np.argsort(raw)[::-1]
    return [
        {"feature": feature_cols[i], "mean_abs_shap": float(raw[i]),
         "share_of_total": float(raw[i] / total), "source": "impurity"}
        for i in order
    ]


# ---------------------------------------------------------------------------
# Persistence of thresholds
# ---------------------------------------------------------------------------

def _threshold_path(cfg: Config, name: str, version: str) -> Path:
    return cfg.models_dir / name / version / "threshold.json"


def _store_threshold(cfg: Config, name: str, version: str, threshold: float) -> None:
    _threshold_path(cfg, name, version).write_text(
        json.dumps({"threshold": float(threshold), "objective": "f1",
                    "tuned_on": "validation"}, indent=2),
        encoding="utf-8",
    )


def _stored_threshold(cfg: Config, name: str, version: str) -> tuple[float, float]:
    path = _threshold_path(cfg, name, version)
    if not path.is_file():
        raise ValidationError(
            f"No tuned threshold stored for {name}/{version}",
            hint=f"Run `seismo model evaluate --model {name} --version {version} "
                 f"--split val` first — thresholds must be tuned on validation "
                 f"before the test split is scored.",
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["threshold"]), 0.0


def _write_model_metrics(cfg: Config, name: str, version: str, metrics: dict[str, Any]) -> None:
    path = cfg.models_dir / name / version / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def _ref_series(scored: pd.DataFrame, reference: np.ndarray) -> pd.Series:
    return pd.Series(reference, index=scored.index)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def write_report(cfg: Config, result: EvaluationResult, scored: pd.DataFrame) -> Path:
    """Write JSON artifacts and figures for one model/split evaluation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = cfg.reports_dir / "model_metrics" / result.model / result.version
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    (directory / f"{result.split}.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / f"{result.split}_calibration.json").write_text(
        json.dumps(result.calibration, indent=2), encoding="utf-8"
    )
    (directory / f"{result.split}_by_region.json").write_text(
        json.dumps(result.by_region, indent=2), encoding="utf-8"
    )
    (directory / f"{result.split}_by_period.json").write_text(
        json.dumps(result.by_period, indent=2), encoding="utf-8"
    )

    y = scored["target"].to_numpy()
    p = scored["prob"].to_numpy()
    m = result.metrics

    # --- reliability diagram -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1.2, label="perfect")
    cal = pd.DataFrame(result.calibration)
    ax.plot(cal["mean_predicted"], cal["observed_frequency"], "-o",
            color=ACCENT, lw=2, ms=5, label=result.model)
    ax.fill_between(cal["mean_predicted"], cal["mean_predicted"],
                    cal["observed_frequency"], color=ACCENT, alpha=0.15)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability — {result.model} ({result.split})\n"
                 f"ECE {m.get('ece', float('nan')):.4f} · Brier {m.get('brier', float('nan')):.4f}")
    ax.legend(frameon=False)
    ax.grid(color=LIGHT)
    fig.tight_layout()
    fig.savefig(figures / f"{result.split}_reliability.png", dpi=150)
    plt.close(fig)

    # --- PR curve --------------------------------------------------------
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color=ACCENT, lw=2)
    ax.axhline(y.mean(), ls="--", color="#999999", lw=1.2,
               label=f"base rate {y.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — {result.model} ({result.split})\n"
                 f"PR-AUC {m.get('pr_auc', float('nan')):.4f}")
    ax.legend(frameon=False)
    ax.grid(color=LIGHT)
    fig.tight_layout()
    fig.savefig(figures / f"{result.split}_pr_curve.png", dpi=150)
    plt.close(fig)

    # --- predicted-probability distribution ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, max(0.05, float(np.quantile(p, 0.999))), 60)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, label="target = 0", color=NAVY)
    ax.hist(p[y == 1], bins=bins, alpha=0.6, label="target = 1", color=ALERT)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Rows")
    ax.set_title(f"Score distribution — {result.model} ({result.split})")
    ax.legend(frameon=False)
    ax.grid(color=LIGHT)
    fig.tight_layout()
    fig.savefig(figures / f"{result.split}_score_distribution.png", dpi=150)
    plt.close(fig)

    # --- SHAP summary -----------------------------------------------------
    if result.shap_top_features:
        top = pd.DataFrame(result.shap_top_features).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(top["feature"], top["mean_abs_shap"], color=ACCENT,
                edgecolor=NAVY, linewidth=0.4)
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"Global feature importance (SHAP) — {result.model}")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=LIGHT)
        fig.tight_layout()
        fig.savefig(figures / f"{result.split}_shap_importance.png", dpi=150)
        plt.close(fig)

    return directory
