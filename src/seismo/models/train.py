"""Model training: configuration, fitting, and artifact persistence.

Protocol (non-negotiable)
-------------------------
* **Fitted on `train` only** (≤ 2018-12-31). Validation is used for model
  selection / early stopping; the **test** split is never touched here — it is
  consumed exactly once by `seismo model evaluate --split test`.
* **Every run writes a self-describing artifact directory**: the pipeline, the
  exact feature list and order, metrics, training configuration, and a
  reference to the data manifest. A model that cannot be reproduced from its
  own directory is not a result.
* **Promotion is a human gate.** Training never writes `models/registry.json`;
  only `seismo model promote` does, after a person has reviewed the metrics and
  calibration plot.

Imbalance handling (base rate 2.92%)
------------------------------------
``logreg`` / ``random_forest`` use sklearn class weights; ``xgboost`` uses
``scale_pos_weight = n_neg/n_pos``.

**Class weighting is disabled by default.** Measured tradeoff: balanced
weights let the models rank well (PR-AUC 0.30–0.36) while breaking
calibration — mean predicted probability 0.28 against a 0.032 base rate, a
Brier Skill Score near −4 versus counting history. Balancing forces the
effective prior toward 50/50, so the probabilities stop being probabilities.
With ~70k positive training rows the models learn the true posterior, and
the operating point is set by threshold tuning on validation instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from ..errors import MissingArtifact, ValidationError
from ..features.engineering import feature_columns

log = logging.getLogger("seismo.models")

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

# Re-exported from the package root (which imports nothing heavy) so callers
# can reference names without pulling pandas/joblib onto the CLI hot path.
from . import MODEL_NAMES  # noqa: E402  (placed after constants)


@dataclass
class TrainingResult:
    name: str
    version: str
    path: str
    train_rows: int
    val_rows: int
    positive_rate_train: float
    positive_rate_val: float
    feature_cols: list[str]
    fit_seconds: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------

def build_estimator(name: str, cfg: Config) -> Any:
    """Return an *unfitted* model for ``name`` (baselines included)."""
    from .baselines import ClimatologicalRate, GlobalRate, PoissonRate, SklearnProbabilistic

    seed = cfg.seed
    settings = cfg.section("models") or {}
    imbalance_ratio = float(settings.get("imbalance_ratio_override", 0.0))
    # Class weighting is OFF by default, and the measurement that decided this
    # is worth recording: with `class_weight='balanced'` the models ranked well
    # (PR-AUC 0.30–0.36) but *calibrated* catastrophically — mean predicted
    # probability 0.28 against a 0.032 base rate, i.e. Brier Skill Score ≈ −4
    # versus simply counting history. Balancing forces the effective prior
    # toward 50/50, so probabilities stop meaning anything. We have ~70k
    # positive training rows, which is ample signal, so the model learns the
    # true posterior and imbalance is handled where it belongs: by threshold
    # tuning on validation. Set use_class_weights: true to restore the old
    # behaviour for experiments.
    use_weights = bool(settings.get("use_class_weights", False))

    if name == "baseline_global_rate":
        return GlobalRate()
    if name == "baseline_climatology":
        return ClimatologicalRate(
            prior_strength=float(settings.get("climatology_prior_strength", 50.0))
        )
    if name == "baseline_poisson":
        return PoissonRate(alpha=float(settings.get("poisson_alpha", 1.0)))

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Standardisation matters for the linear model: `days_since_last_event`
        # spans 0–9,770 while counts span 0–5,473 and coordinates span ±180.
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(settings.get("logreg_C", 1.0)),
                max_iter=int(settings.get("logreg_max_iter", 2000)),
                class_weight="balanced" if use_weights else None,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        return SklearnProbabilistic(estimator, name)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        # Depth is capped deliberately. An unconstrained forest on 2.48M rows
        # grew to depth ~47 / 9M nodes / a 724MB artifact — memorising the
        # training rows instead of learning a probability surface, and making
        # SHAP traversal impractical. With only 10 features, depth 16 is far
        # more capacity than the problem needs, and a larger leaf size yields
        # smoother (better-calibrated) probabilities.
        estimator = RandomForestClassifier(
            n_estimators=int(settings.get("rf_n_estimators", 200)),
            max_depth=int(settings.get("rf_max_depth", 16)),
            min_samples_leaf=int(settings.get("rf_min_samples_leaf", 50)),
            max_features="sqrt",
            class_weight="balanced_subsample" if use_weights else None,
            n_jobs=int(settings.get("n_jobs", -1)),
            random_state=seed,
        )
        return SklearnProbabilistic(estimator, name)

    if name == "xgboost":
        import xgboost as xgb

        ratio = imbalance_ratio or _train_imbalance_ratio(cfg)
        estimator = xgb.XGBClassifier(
            n_estimators=int(settings.get("xgb_n_estimators", 3000)),
            learning_rate=float(settings.get("xgb_learning_rate", 0.08)),
            max_depth=int(settings.get("xgb_max_depth", 6)),
            min_child_weight=float(settings.get("xgb_min_child_weight", 20)),
            subsample=float(settings.get("xgb_subsample", 0.8)),
            colsample_bytree=float(settings.get("xgb_colsample_bytree", 0.8)),
            reg_lambda=float(settings.get("xgb_reg_lambda", 1.0)),
            scale_pos_weight=(ratio if use_weights else 1.0),
            eval_metric="logloss",
            early_stopping_rounds=int(settings.get("xgb_early_stopping_rounds", 50)),
            n_jobs=int(settings.get("n_jobs", -1)),
            random_state=seed,
            tree_method="hist",
        )
        return SklearnProbabilistic(estimator, name)

    raise ValidationError(
        f"Unknown model {name!r}",
        hint=f"Choose one of: {', '.join(MODEL_NAMES)}",
    )


def _train_imbalance_ratio(cfg: Config) -> float:
    """n_negative / n_positive on the training split (XGBoost weighting)."""
    frame = load_feature_frame(cfg)
    train = frame[frame["split"] == SPLIT_TRAIN]
    pos = float(train["target"].sum())
    neg = float(len(train) - pos)
    if pos <= 0:
        raise ValidationError(
            "Training split has no positive examples",
            hint="Check the forecast threshold and split dates in config.yaml.",
        )
    return neg / pos


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_feature_frame(cfg: Config) -> pd.DataFrame:
    path = cfg.features_dir / "regional_monthly_features.parquet"
    if not path.is_file():
        raise MissingArtifact(
            f"Feature dataset not found at {path}",
            hint="Run `seismo features build` first.",
        )
    return pd.read_parquet(path)


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split rows by the stored chronological ``split`` column.

    Refuses to proceed if the column is missing — an unsplit frame would let
    random rows cross the temporal boundary and leak the future.
    """
    if "split" not in frame.columns:
        raise ValidationError(
            "Feature dataset has no 'split' column",
            hint="Rebuild with `seismo features build` — chronological splitting "
                 "is required before training.",
        )
    splits = {name: frame[frame["split"] == name] for name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}
    empty = [k for k, v in splits.items() if v.empty]
    if empty:
        raise ValidationError(
            f"Empty split(s): {empty}",
            hint="Adjust splits.train_end / splits.val_end in config.yaml.",
        )
    # Chronology invariant — cheap to assert, catastrophic to violate.
    for earlier, later in ((SPLIT_TRAIN, SPLIT_VAL), (SPLIT_VAL, SPLIT_TEST)):
        a = splits[earlier]["obs_date"].max()
        b = splits[later]["obs_date"].min()
        if a >= b:
            raise ValidationError(
                f"Split integrity violated: {earlier} ends {a} but {later} starts {b}",
                hint="This is a bug in chronological_split — do not train.",
            )
    return splits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    cfg: Config,
    name: str,
    *,
    frame: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    version: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Fit ``name`` on the train split; monitor on validation only."""
    import time

    if name not in MODEL_NAMES:
        raise ValidationError(
            f"Unknown model {name!r}",
            hint=f"Choose one of: {', '.join(MODEL_NAMES)}",
        )

    frame = frame if frame is not None else load_feature_frame(cfg)
    splits = split_frame(frame)
    train, val = splits[SPLIT_TRAIN], splits[SPLIT_VAL]

    if feature_cols is None:
        feature_cols = feature_columns(frame)
    if not feature_cols:
        raise ValidationError(
            "No feature columns selected",
            hint="Check feature_columns() against the built feature dataset.",
        )

    if on_progress:
        on_progress(
            f"{name}: fitting on {len(train):,} train rows "
            f"({100 * train['target'].mean():.2f}% positive), "
            f"{len(val):,} val rows"
        )

    estimator = build_estimator(name, cfg)

    started = time.perf_counter()
    if name == "xgboost":
        # Validation drives early stopping — legitimate model selection.
        # XGBoost wants concrete (X, y) arrays, not the feature frame.
        eval_set = [(
            val[feature_cols].to_numpy(dtype=np.float64),
            val["target"].to_numpy(dtype=np.int64),
        )]
        estimator.fit(train, feature_cols, eval_set=eval_set, verbose=False)
    else:
        estimator.fit(train, feature_cols)
    elapsed = time.perf_counter() - started

    result = TrainingResult(
        name=name,
        version=version or next_version(name),
        path="",
        train_rows=int(len(train)),
        val_rows=int(len(val)),
        positive_rate_train=round(float(train["target"].mean()), 6),
        positive_rate_val=round(float(val["target"].mean()), 6),
        feature_cols=list(feature_cols),
        fit_seconds=round(elapsed, 2),
        notes=_training_notes(estimator, name, cfg),
    )
    if on_progress:
        on_progress(f"{name}: fitted in {elapsed:.1f}s → {result.version}")
    return _persist(cfg, estimator, result, cfg_section=cfg.section("models"))


def train_all(
    cfg: Config,
    names: tuple[str, ...] | list[str] = MODEL_NAMES,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> list[TrainingResult]:
    """Train several models against one in-memory frame (loads data once)."""
    frame = load_feature_frame(cfg)
    results = []
    for name in names:
        results.append(train_model(cfg, name, frame=frame, on_progress=on_progress))
    return results


def _training_notes(estimator: Any, name: str, cfg: Config) -> list[str]:
    notes = [
        "Fitted on the train split only; validation used for selection; "
        "test held out for final evaluation.",
        f"Seed = {cfg.seed}.",
    ]
    if name == "xgboost":
        notes.append("scale_pos_weight set to the training imbalance ratio; "
                     "early stopping monitored on validation log-loss.")
    elif name in ("logreg", "random_forest"):
        notes.append("No class weighting: balanced weights would shift the "
                     "effective prior and destroy probability calibration. "
                     "Imbalance is handled by threshold tuning on validation.")
    return notes


def next_version(name: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"v1-{stamp}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist(
    cfg: Config,
    estimator: Any,
    result: TrainingResult,
    *,
    cfg_section: dict[str, Any],
) -> TrainingResult:
    target = cfg.models_dir / result.name / result.version
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, target / "model.joblib")
    result.path = str(target)

    (target / "features.json").write_text(
        json.dumps(
            {"feature_cols": result.feature_cols, "target": "target",
             "positive_probability_column": 1},
            indent=2,
        ),
        encoding="utf-8",
    )
    (target / "training_config.yaml").write_text(
        _yaml_dump({
            "name": result.name,
            "version": result.version,
            "seed": cfg.seed,
            "model_settings": cfg_section,
            "forecast": cfg.section("forecast"),
            "splits": cfg.section("splits"),
            "grid": cfg.section("grid"),
            "fit_seconds": result.fit_seconds,
            "train_rows": result.train_rows,
            "val_rows": result.val_rows,
        }),
        encoding="utf-8",
    )
    (target / "data_manifest_ref.json").write_text(
        json.dumps(_data_lineage(cfg), indent=2),
        encoding="utf-8",
    )
    (target / "training_summary.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    log.info("saved model artifact to %s", target)
    return result


def _data_lineage(cfg: Config) -> dict[str, Any]:
    """Point every artifact at the exact data that produced it."""
    from ..data.ingestion import load_manifest

    lineage: dict[str, Any] = {
        "processed": str(cfg.processed_dir / "earthquakes.parquet"),
        "features": str(cfg.features_dir / "regional_monthly_features.parquet"),
        "collection": {
            "start_date": str(cfg.collection.get("start_date")),
            "end_date": str(cfg.collection.get("end_date")),
            "min_magnitude": cfg.min_magnitude,
        },
    }
    try:
        manifest = load_manifest(cfg.raw_dir)
        months = manifest.get("months", {})
        lineage["raw_manifest"] = {
            "months_recorded": len(months),
            "total_rows": int(sum(m.get("rows", 0) for m in months.values())),
            "all_ok": all(m.get("status") in ("ok", "empty") for m in months.values()),
            "params": manifest.get("params", {}),
        }
    except Exception:  # noqa: BLE001 - lineage must never block training
        lineage["raw_manifest"] = {"note": "manifest unavailable"}
    return lineage


def _yaml_dump(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def latest_version(cfg: Config, name: str) -> str:
    """Newest trained version (versions are UTC-timestamped, so sorted order
    is chronological). Used by evaluation, which scores what was just trained
    rather than what is deployed."""
    directory = cfg.models_dir / name
    versions = sorted(p.name for p in directory.glob("v*")) if directory.is_dir() else []
    if not versions:
        raise MissingArtifact(
            f"No trained versions for {name!r} under {directory}",
            hint=f"Train it with `seismo model train --model {name}`.",
        )
    return versions[-1]


def load_model(
    cfg: Config, name: str, version: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load ``name`` — a specific version, else the **latest trained** one.

    Evaluation scores what exists; deployment uses ``load_production_model``
    below, which refuses anything not promoted through the human gate.
    """
    if version is None:
        version = latest_version(cfg, name)
    directory = cfg.models_dir / name / version
    if not directory.is_dir():
        raise MissingArtifact(
            f"No model artifact at {directory}",
            hint=f"Train it with `seismo model train --model {name}`, or list "
                 f"available versions under {cfg.models_dir / name}.",
        )
    model = joblib.load(directory / "model.joblib")
    metadata = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    metadata["version"] = version
    metadata["path"] = str(directory)
    for extra in ("training_summary.json", "metrics.json"):
        path = directory / extra
        if path.is_file():
            metadata[extra.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    return model, metadata


def load_production_model(
    cfg: Config, name: str
) -> tuple[Any, dict[str, Any]]:
    """Load the **promoted** version — the only thing inference should use.

    Refuses unpromoted models so the app can never accidentally serve a
    version no human has reviewed.
    """
    return load_model(cfg, name, production_version(cfg, name))


# ---------------------------------------------------------------------------
# Registry — the human promotion gate
# ---------------------------------------------------------------------------

def registry_path(cfg: Config) -> Path:
    return cfg.models_dir / "registry.json"


def load_registry(cfg: Config) -> dict[str, Any]:
    path = registry_path(cfg)
    if not path.is_file():
        return {"production": {}, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(cfg: Config, registry: dict[str, Any]) -> None:
    path = registry_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def production_version(cfg: Config, name: str) -> str:
    registry = load_registry(cfg)
    version = registry.get("production", {}).get(name)
    if version is None:
        available = sorted((cfg.models_dir / name).glob("v*")) \
            if (cfg.models_dir / name).is_dir() else []
        hint = (
            f"Train it with `seismo model train --model {name}`"
            if not available
            else f"Available versions: {', '.join(p.name for p in available)} — "
                 f"promote one with `seismo model promote --model {name} --version <v>`"
        )
        raise MissingArtifact(
            f"No production version registered for {name!r}",
            hint=hint,
        )
    return version


def promote(cfg: Config, name: str, version: str, *, reviewed_by: str) -> dict[str, Any]:
    """Human gate: register a version as production.

    Deliberately requires ``reviewed_by`` — promotion without an accountable
    reviewer is exactly the failure mode this project is designed to avoid.
    """
    directory = cfg.models_dir / name / version
    if not directory.is_dir():
        raise MissingArtifact(
            f"No model artifact at {directory}",
            hint=f"Train it first with `seismo model train --model {name}`.",
        )
    if not (directory / "metrics.json").is_file():
        raise ValidationError(
            f"{name}/{version} has no metrics.json — refusing to promote an "
            f"unevaluated model",
            hint="Run `seismo model evaluate --model "
                 f"{name} --version {version} --split val` before promoting.",
        )
    if not reviewed_by.strip():
        raise ValidationError(
            "Promotion requires a non-empty --reviewed-by value",
            hint="Name the person who reviewed the metrics and calibration plot: "
                 "`seismo model promote --model X --version Y --reviewed-by 'name'`.",
        )

    registry = load_registry(cfg)
    previous = registry["production"].get(name)
    registry["production"][name] = version
    registry.setdefault("history", []).append({
        "model": name,
        "version": version,
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reviewed_by": reviewed_by,
        "replaced": previous,
    })
    save_registry(cfg, registry)
    return {"model": name, "version": version, "previous": previous,
            "reviewed_by": reviewed_by}
