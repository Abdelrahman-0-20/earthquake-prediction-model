"""Configuration loading: YAML config file, resolved against the project root.

Precedence: CLI flags > environment > config/config.yaml > built-in defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "SEISMO_CONFIG"

# Built-in defaults — used only when config/config.yaml is absent or partial.
DEFAULTS: dict[str, Any] = {
    "project": {"name": "SeismoAI", "version": "0.1.0"},
    "collection": {
        "start_date": "2000-01-01",
        "end_date": "2026-11-15",
        "min_magnitude": 2.5,
        "chunk": "monthly",
        "format": "csv",
        "api_query_url": "https://earthquake.usgs.gov/fdsnws/event/1/query",
        "max_rows_per_request": 20000,
        "timeout_seconds": 60,
        "max_retries": 5,
        "backoff_base_seconds": 2.0,
    },
    "paths": {
        "raw": "data/raw",
        "processed": "data/processed",
        "features": "data/features",
        "samples": "data/samples",
        "reports": "reports",
        "models": "models",
    },
    "validation": {
        "latitude_range": [-90.0, 90.0],
        "longitude_range": [-180.0, 180.0],
        "keep_event_types": ["earthquake"],
        "report_dir": "reports/data_quality",
    },
    "grid": {"cell_size_deg": 1.0},
    "forecast": {"magnitude_threshold": 4.5, "horizon_days": 30, "min_lookback_days": 365},
    "splits": {"train_end": "2018-12-31", "val_end": "2022-12-31", "purge_days": 30},
    "seed": 42,
}


class ConfigError(RuntimeError):
    """Raised when configuration cannot be located or parsed."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: cwd) looking for config/config.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise ConfigError(
        "config/config.yaml not found in this directory or any parent.\n"
        "  Fix: run `seismo` from the SeismoAI repository root, or pass "
        "`--root /path/to/SeismoAI`."
    )


@dataclass
class Config:
    """Project configuration with resolved paths."""

    root: Path
    data: dict[str, Any] = field(default_factory=dict)

    # --- path accessors -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.root / self.data["paths"]["raw"]

    @property
    def processed_dir(self) -> Path:
        return self.root / self.data["paths"]["processed"]

    @property
    def features_dir(self) -> Path:
        return self.root / self.data["paths"]["features"]

    @property
    def samples_dir(self) -> Path:
        return self.root / self.data["paths"]["samples"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.data["paths"]["reports"]

    @property
    def models_dir(self) -> Path:
        return self.root / self.data["paths"]["models"]

    @property
    def quality_report_dir(self) -> Path:
        return self.root / self.data["validation"]["report_dir"]

    # --- typed section accessors ---------------------------------------
    @property
    def collection(self) -> dict[str, Any]:
        return self.data["collection"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.data["validation"]

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    # --- date helpers ---------------------------------------------------
    @property
    def start_date(self) -> date:
        return date.fromisoformat(str(self.collection["start_date"]))

    @property
    def end_date(self) -> date:
        return date.fromisoformat(str(self.collection["end_date"]))

    @property
    def min_magnitude(self) -> float:
        return float(self.collection["min_magnitude"])

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})


def load_config(root: str | Path | None = None) -> Config:
    """Load config/config.yaml merged over built-in defaults.

    `root` explicitly selects the project root (used by tests and `--root`).
    Otherwise the root is discovered by walking up from the current directory,
    or taken from $SEISMO_CONFIG when set.
    """
    if root is not None:
        project_root = Path(root).resolve()
        config_file = project_root / "config" / "config.yaml"
    else:
        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            config_file = Path(env_path).resolve()
            project_root = config_file.parent.parent
        else:
            project_root = find_project_root()
            config_file = project_root / "config" / "config.yaml"

    user_data: dict[str, Any] = {}
    if config_file.is_file():
        try:
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Could not parse {config_file}:\n  {exc}\n"
                "  Fix: correct the YAML syntax, or delete the file to use defaults."
            ) from exc
        if not isinstance(loaded, dict):
            raise ConfigError(
                f"{config_file} must contain a YAML mapping at the top level.\n"
                "  Fix: check for stray list items or scalar values at the root."
            )
        user_data = loaded

    return Config(root=project_root, data=_deep_merge(DEFAULTS, user_data))


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, requiring timezone awareness."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ConfigError(f"Timestamp must be timezone-aware (…Z or +00:00): {value!r}")
    return dt
