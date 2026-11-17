"""The single module allowed to touch the filesystem.

Everything the app reads flows through this file, so there is exactly one
place to reason about caching, missing artifacts, and error handling. Pages
import *functions* from here and never call ``pd.read_parquet`` directly —
that rule is what keeps the six pages consistent and makes the app's data
dependencies enumerable.

Caching strategy
----------------
* ``st.cache_data`` for DataFrames — hashed by arguments, shared across
  reruns and sessions. The processed catalog (707k rows) and feature dataset
  (3.57M rows) are expensive to load, so they are loaded once.
* ``st.cache_resource`` for model objects — loaded once per process; joblib
  deserialisation of a 172MB forest or a booster is far too slow per rerun.

Missing artifacts never crash the app: ``Requirement`` carries a message and
the exact CLI command that produces it, which the shell renders as guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from seismo.config import Config, load_config
from seismo.errors import MissingArtifact


@dataclass(frozen=True)
class Requirement:
    """A missing artifact, described the way the CLI would describe it."""

    label: str
    command: str

    @property
    def message(self) -> str:
        return f"{self.label} is not available.\n\nRun `{self.command}` to build it."


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Project config (paths, thresholds, grid)."""
    return load_config()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@st.cache_data
def load_processed() -> pd.DataFrame:
    """Cleaned catalog — 707,438 events, 2000-01 → 2026-11."""
    path = get_config().processed_dir / "earthquakes.parquet"
    if not path.is_file():
        raise MissingArtifact(
            f"Processed catalog not found at {path}",
            hint="Run `seismo data fetch` then `seismo data validate`.",
        )
    return pd.read_parquet(path)


@st.cache_data
def load_features() -> pd.DataFrame:
    """Module 02 dataset — 3,565,930 (cell, month) rows with split labels."""
    path = get_config().features_dir / "regional_monthly_features.parquet"
    if not path.is_file():
        raise MissingArtifact(
            f"Feature dataset not found at {path}",
            hint="Run `seismo features build`.",
        )
    return pd.read_parquet(path)


@st.cache_data
def load_cell_activity() -> pd.DataFrame:
    """Per-cell historical activity (Module 03 map layer)."""
    path = get_config().features_dir / "cell_activity.parquet"
    if not path.is_file():
        raise MissingArtifact(
            f"Cell activity not found at {path}",
            hint="Run `seismo geospatial build`.",
        )
    return pd.read_parquet(path)


@st.cache_data
def load_data_quality() -> dict[str, Any]:
    """Catalog quality report (missingness, event types, soft checks)."""
    path = get_config().quality_report_dir / "data_quality.json"
    if not path.is_file():
        raise MissingArtifact(
            f"Data quality report not found at {path}",
            hint="Run `seismo data validate`.",
        )
    return json_load(path)


# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------

@st.cache_resource
def load_production_model() -> tuple[Any, dict[str, Any]]:
    """The promoted model — the only thing the app is allowed to serve.

    Uses ``load_production_model`` (registry-gated) rather than ``load_model``
    (latest), so an unpromoted experiment can never reach a user.
    """
    from seismo.models.train import load_production_model as _load

    cfg = get_config()
    return _load(cfg, "xgboost")


@st.cache_data
def load_registry() -> dict[str, Any]:
    from seismo.models.train import load_registry as _load

    return _load(get_config())


@st.cache_data
def load_model_comparison(split: str = "val") -> pd.DataFrame:
    """Side-by-side metrics for every evaluated model on ``split``."""
    import glob
    import json

    root = get_config().reports_dir / "model_metrics"
    rows = []
    for path in sorted(glob.glob(str(root / "*" / "*" / f"{split}.json"))):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({
            "model": m.get("model"),
            "version": Path(path).parent.name,
            "split": m.get("split"),
            "n_rows": m.get("n_rows"),
            "positive_rate": m.get("positive_rate"),
            "threshold": m.get("threshold"),
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "brier": m.get("brier"),
            "brier_skill_score": m.get("brier_skill_score"),
            "ece": m.get("ece"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
        })
    if not rows:
        raise MissingArtifact(
            f"No {split!r} evaluation reports under {root}",
            hint="Run `seismo model train --all` then `seismo model evaluate`.",
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("brier_skill_score", ascending=False).reset_index(drop=True)


@st.cache_data
def load_calibration(model: str, split: str = "val") -> pd.DataFrame:
    root = get_config().reports_dir / "model_metrics" / model
    paths = sorted(root.glob(f"*/{split}_calibration.json"))
    if not paths:
        raise MissingArtifact(
            f"No calibration data for {model!r} ({split})",
            hint=f"Run `seismo model evaluate --model {model} --split {split}`.",
        )
    return pd.DataFrame(json_load(paths[-1]))


@st.cache_data
def load_region_metrics(model: str, split: str = "val") -> pd.DataFrame:
    root = get_config().reports_dir / "model_metrics" / model
    paths = sorted(root.glob(f"*/{split}_by_region.json"))
    if not paths:
        raise MissingArtifact(
            f"No regional metrics for {model!r} ({split})",
            hint=f"Run `seismo model evaluate --model {model} --split {split}`.",
        )
    return pd.DataFrame(json_load(paths[-1]))


@st.cache_data
def load_shap_importance(model: str, split: str = "val") -> pd.DataFrame:
    root = get_config().reports_dir / "model_metrics" / model
    for path in sorted(root.glob(f"*/{split}.json")):
        payload = json_load(path)
        importance = payload.get("shap_top_features") or []
        if importance:
            return pd.DataFrame(importance)
    raise MissingArtifact(
        f"No SHAP importance recorded for {model!r}",
        hint=f"Run `seismo model evaluate --model {model} --split {split}` "
             f"(baselines do not produce SHAP values).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_load(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def artifact_status() -> pd.DataFrame:
    """Inventory of every artifact the app can display — drives the shell's
    'what is missing' guidance instead of a stack trace."""
    cfg = get_config()
    rows = [
        ("Processed catalog", cfg.processed_dir / "earthquakes.parquet",
         "seismo data fetch && seismo data validate"),
        ("Feature dataset", cfg.features_dir / "regional_monthly_features.parquet",
         "seismo features build"),
        ("Cell activity", cfg.features_dir / "cell_activity.parquet",
         "seismo geospatial build"),
        ("Data quality report", cfg.quality_report_dir / "data_quality.json",
         "seismo data validate"),
        ("Model registry", cfg.models_dir / "registry.json",
         "seismo model train --all"),
        ("Model reports", cfg.reports_dir / "model_metrics",
         "seismo model evaluate --split val"),
    ]
    return pd.DataFrame([
        {"artifact": label, "available": Path(path).exists(), "build_command": cmd}
        for label, path, cmd in rows
    ])


def require_available() -> list[Requirement]:
    """Return requirements for anything missing (empty list = app fully ready)."""
    missing: list[Requirement] = []
    for _, row in artifact_status().iterrows():
        if not row["available"]:
            missing.append(Requirement(label=row["artifact"], command=row["build_command"]))
    return missing


__all__ = [
    "Requirement",
    "artifact_status",
    "get_config",
    "load_calibration",
    "load_cell_activity",
    "load_data_quality",
    "load_features",
    "load_model_comparison",
    "load_processed",
    "load_production_model",
    "load_registry",
    "load_region_metrics",
    "load_shap_importance",
    "require_available",
]
