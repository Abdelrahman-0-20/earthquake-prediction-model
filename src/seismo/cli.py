"""seismo — command-line interface for the SeismoAI pipeline.

Command surface (verb-noun, consistent flags):

    seismo data fetch      download the USGS catalog into data/raw/
    seismo data validate   clean raw data, run gates, build processed dataset
    seismo data status     show the fetch manifest summary

Global flags mean the same thing everywhere:
    --root <path>   project root (default: discovered from cwd)
    --json          machine-readable output, no progress chatter
    -v/--verbose    debug logging
    -h/--help       teach this command
    --version       print version

Exit codes are an API: 0 ok · 2 usage/config · 3 validation · 4 API · 5 missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import date
from typing import Any

from . import __version__
from .config import ConfigError as _ConfigError
from .config import load_config
from .errors import (
    EXIT_API,
    EXIT_MISSING,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VALIDATION,
    SeismoError,
)

# Cheap: seismo.models/__init__ defines only string tuples (no pandas/joblib),
# so help text can name the models without paying import cost on every run.
from .models import MODEL_NAMES

# Re-exported so tests can assert the documented contract.
__all__ = [
    "main",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXIT_VALIDATION",
    "EXIT_API",
    "EXIT_MISSING",
]

log = logging.getLogger("seismo")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class Output:
    """Dual-audience output: rich for humans, plain JSON when --json."""

    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json

    def line(self, text: str = "") -> None:
        if not self.as_json:
            print(text)

    def result(
        self,
        payload: dict[str, Any],
        human: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        if self.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        elif human:
            human(payload)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date — use YYYY-MM-DD (e.g. 2024-01-01)"
        ) from exc


def cmd_data_fetch(args: argparse.Namespace) -> int:
    from .data.ingestion import run_fetch

    cfg = load_config(args.root)
    out = Output(args.json)

    if args.min_mag is not None:
        cfg.collection["min_magnitude"] = float(args.min_mag)
    if args.start is not None:
        cfg.collection["start_date"] = args.start.isoformat()
    if args.end is not None:
        cfg.collection["end_date"] = args.end.isoformat()

    def progress(message: str) -> None:
        out.line(message)

    summary = run_fetch(
        cfg,
        force=args.force,
        limit=args.limit,
        on_progress=progress,
    )
    payload = summary.as_dict()
    payload["raw_dir"] = str(cfg.raw_dir)

    def human(s: dict[str, Any]) -> None:
        print()
        print(f"Fetched {s['fetched']} month(s), {s['empty']} empty, "
              f"{s['skipped']} up to date, {s['failed']} failed.")
        print(f"Total rows this run: {s['rows']:,}   →  {cfg.raw_dir}")
        if s["failed_months"]:
            print(f"Failed months: {', '.join(s['failed_months'])}")

    out.result(payload, human)
    return EXIT_API if summary.failed else EXIT_OK


def cmd_data_validate(args: argparse.Namespace) -> int:
    from .data.preprocessing import build_processed

    cfg = load_config(args.root)
    out = Output(args.json)

    report = build_processed(cfg, strict=not args.lenient)
    payload = report.as_dict()
    payload["processed_path"] = str(cfg.processed_dir / "earthquakes.parquet")

    def human(r: dict[str, Any]) -> None:
        print(f"Processed {r['processed_rows']:,} rows "
              f"({r['total_raw_rows']:,} raw, "
              f"-{r['duplicates_removed']} duplicates, "
              f"-{r['non_earthquake_removed']} non-earthquake).")
        if r["date_range"]:
            start, end = r["date_range"]["start"][:10], r["date_range"]["end"][:10]
            print(f"Date range: {start} → {end}")
            print(f"Magnitude:  {r['magnitude_range']['min']:.1f} → "
                  f"{r['magnitude_range']['max']:.1f}")
        print(f"Report:     {cfg.quality_report_dir / 'data_quality.json'}")
        print(f"Dataset:    {cfg.processed_dir / 'earthquakes.parquet'}")

    out.result(payload, human)
    return EXIT_OK


def cmd_data_status(args: argparse.Namespace) -> int:
    from .data.ingestion import load_manifest

    cfg = load_config(args.root)
    out = Output(args.json)

    manifest = load_manifest(cfg.raw_dir)
    months: dict[str, Any] = manifest.get("months", {})

    counts: dict[str, int] = {}
    rows = 0
    for record in months.values():
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        rows += int(record.get("rows", 0))

    payload = {
        "raw_dir": str(cfg.raw_dir),
        "months_recorded": len(months),
        "status_counts": counts,
        "total_rows": rows,
        "params": manifest.get("params", {}),
        "updated_at": manifest.get("updated_at", ""),
        "failed_months": sorted(
            label for label, rec in months.items() if rec.get("status") == "failed"
        ),
    }

    def human(p: dict[str, Any]) -> None:
        if not p["months_recorded"]:
            print(f"No fetch manifest yet in {cfg.raw_dir} — run `seismo data fetch`.")
            return
        print(f"Raw dir:  {p['raw_dir']}")
        print(f"Months:   {p['months_recorded']} recorded, "
              f"{p['total_rows']:,} total rows")
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(p["status_counts"].items()))
        print(f"Status:   {breakdown}")
        if p["failed_months"]:
            print(f"Failed:   {', '.join(p['failed_months'])} "
                  f"(re-run `seismo data fetch` to retry)")

    out.result(payload, human)
    return EXIT_OK


def cmd_features_build(args: argparse.Namespace) -> int:
    """Build the (cell, month) ML dataset with leak-free features + target."""
    from .data.ingestion import atomic_write_parquet
    from .features.engineering import build_features, feature_columns
    from .features.splits import chronological_split

    cfg = load_config(args.root)
    out = Output(args.json)

    frame, report = build_features(cfg, on_progress=out.line)
    frame, split_stats = chronological_split(frame, cfg)

    path = cfg.features_dir / "regional_monthly_features.parquet"
    atomic_write_parquet(frame, path)

    payload = {
        **report.as_dict(),
        "split_counts": split_stats,
        "feature_columns": feature_columns(frame),
        "output_path": str(path),
    }

    def human(p: dict[str, Any]) -> None:
        print()
        print(f"Rows:      {p['rows']:,}  ({p['active_cells']:,} cells × "
              f"{p['observation_dates']} dates)")
        print(f"Window:    {p['first_obs_date']} → {p['last_obs_date']}")
        print(f"Positives: {p['positive_rows']:,} ({100 * p['positive_rate']:.2f}%) "
              f"— threshold M≥{p['magnitude_threshold']}, {p['horizon_days']}d horizon")
        s = p["split_counts"]
        print(f"Splits:    train {s['train']:,} · val {s['val']:,} · "
              f"test {s['test']:,} · purged {s['purged']:,}")
        print(f"Dataset:   {path}")

    out.result(payload, human)
    return EXIT_OK


def cmd_geospatial_build(args: argparse.Namespace) -> int:
    """Aggregate per-cell historical activity (Module 03 map layer)."""
    import pandas as pd

    from .data.ingestion import atomic_write_parquet
    from .data.preprocessing import load_processed
    from .geospatial.analysis import assign_cells, cell_activity, grid_from_config

    cfg = load_config(args.root)
    out = Output(args.json)

    grid = grid_from_config(cfg)
    events = load_processed(cfg)
    events = assign_cells(events, grid)

    activity = cell_activity(
        events, grid,
        catalog_start=pd.Timestamp(cfg.start_date, tz="UTC"),
        catalog_end=pd.Timestamp(events["time"].max()),
    )
    path = cfg.features_dir / "cell_activity.parquet"
    atomic_write_parquet(activity, path)

    payload = {
        "cells_with_activity": int(len(activity)),
        "cell_size_deg": grid.cell_size_deg,
        "top_cell_events": int(activity["n_events"].max()) if len(activity) else 0,
        "output_path": str(path),
    }

    def human(p: dict[str, Any]) -> None:
        print(f"Cells:     {p['cells_with_activity']:,} with ≥1 event "
              f"({p['cell_size_deg']}° grid)")
        print(f"Busiest:   {p['top_cell_events']:,} events in one cell")
        print(f"Dataset:   {path}")

    out.result(payload, human)
    return EXIT_OK


def cmd_data_export(args: argparse.Namespace) -> int:
    """Export a dataset to CSV (interchange) alongside the Parquet originals."""
    import pandas as pd

    from .errors import MissingArtifact

    cfg = load_config(args.root)
    out = Output(args.json)

    candidates = {
        "processed": cfg.processed_dir / "earthquakes.parquet",
        "features": cfg.features_dir / "regional_monthly_features.parquet",
        "activity": cfg.features_dir / "cell_activity.parquet",
    }
    if args.dataset == "raw":
        # Raw lives as one file per month; export concatenates them.
        from .data.preprocessing import load_raw

        frame, files = load_raw(cfg)
        source_label = f"{len(files)} monthly files"
    else:
        source = candidates[args.dataset]
        if not source.is_file():
            raise MissingArtifact(
                f"{args.dataset} dataset not found at {source}",
                hint="Run `seismo data fetch`, `seismo data validate` or "
                     "`seismo features build` first.",
            )
        frame = pd.read_parquet(source)
        source_label = str(source)

    export_dir = cfg.root / "data" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{args.dataset}.csv"

    frame.to_csv(target, index=False)

    payload = {
        "dataset": args.dataset,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "source": source_label,
        "output": str(target),
        "bytes": target.stat().st_size,
    }

    def human(p: dict[str, Any]) -> None:
        print(f"Exported {p['rows']:,} rows × {len(p['columns'])} columns")
        print(f"  from: {p['source']}")
        print(f"  to:   {p['output']}  ({p['bytes'] / 1e6:.1f} MB)")

    out.result(payload, human)
    return EXIT_OK


def cmd_model_train(args: argparse.Namespace) -> int:
    """Fit model(s) on the train split and write versioned artifacts."""
    from .models.train import train_all

    cfg = load_config(args.root)
    out = Output(args.json)

    names = tuple(MODEL_NAMES) if (args.all or not args.model) else (args.model,)
    out.line(f"Training {len(names)} model(s) on the chronological train split…\n")

    results = train_all(cfg, names, on_progress=out.line)
    payload = {"trained": [r.as_dict() for r in results]}

    def human(p: dict[str, Any]) -> None:
        print()
        print(f"{'model':26s} {'version':24s} {'train rows':>11s} {'fit s':>7s}")
        print("-" * 74)
        for r in p["trained"]:
            print(f"{r['name']:26s} {r['version']:24s} "
                  f"{r['train_rows']:>11,} {r['fit_seconds']:>7.1f}")
        print("\nTest split NOT evaluated — run `seismo model evaluate --split test` "
              "once, after review.")

    out.result(payload, human)
    return EXIT_OK


def cmd_model_evaluate(args: argparse.Namespace) -> int:
    """Score model(s): metrics, calibration, slices, SHAP, figures."""
    import pandas as pd

    from .models.reporting import comparison_table, evaluate_all

    cfg = load_config(args.root)
    out = Output(args.json)

    names = (args.model,) if args.model else None
    results = evaluate_all(
        cfg, names=names, split=args.split, on_progress=out.line
    )
    table = comparison_table(results)
    payload = {
        "split": args.split,
        "comparison": table.to_dict(orient="records"),
        "evaluations": [r.as_dict() for r in results],
    }

    def human(p: dict[str, Any]) -> None:
        print()
        t = pd.DataFrame(p["comparison"])
        cols = ["model", "pr_auc", "roc_auc", "brier", "brier_skill_score",
                "ece", "f1", "threshold"]
        available = [c for c in cols if c in t.columns]
        view = t[available].copy()
        for c in view.columns:
            if c not in ("model", "threshold", "split"):
                view[c] = view[c].map(
                    lambda v: f"{v:.4f}" if isinstance(v, float) and v == v else "—"
                )
        print(f"=== {p['split'].upper()} split — ranked by Brier Skill Score ===")
        print(view.to_string(index=False))
        print("\nBrier Skill Score: >0 beats the climatological baseline, "
              "<0 is worse than it.")
        if p["split"] == "test":
            print("NOTE: the test split has now been touched — further tuning "
                  "on it invalidates the holdout.")

    out.result(payload, human)
    return EXIT_OK


def cmd_model_promote(args: argparse.Namespace) -> int:
    """Human gate: register an evaluated version as production."""
    from .models.train import promote

    cfg = load_config(args.root)
    out = Output(args.json)
    result = promote(cfg, args.model, args.version, reviewed_by=args.reviewed_by)

    out.result(result, lambda r: print(
        f"Promoted {r['model']} → {r['version']} "
        f"(replaced {r['previous'] or 'none'}; reviewed by {r['reviewed_by']})"
    ))
    return EXIT_OK


def cmd_model_list(args: argparse.Namespace) -> int:
    """Show trained versions and the production registry."""
    from .models.train import load_registry

    cfg = load_config(args.root)
    out = Output(args.json)
    registry = load_registry(cfg)

    models = {}
    for name in MODEL_NAMES:
        directory = cfg.models_dir / name
        versions = sorted(p.name for p in directory.glob("v*")) if directory.is_dir() else []
        models[name] = {
            "versions": versions,
            "production": registry.get("production", {}).get(name),
            "has_metrics": any(
                (directory / v / "metrics.json").is_file() for v in versions
            ),
        }

    payload = {"registry": registry, "models": models,
               "models_dir": str(cfg.models_dir)}

    def human(p: dict[str, Any]) -> None:
        prod = p["registry"].get("production", {})
        print(f"{'model':26s} {'production':24s} {'versions':>9s}  evaluated")
        print("-" * 74)
        for name, info in p["models"].items():
            marker = "✓" if info["has_metrics"] else "·"
            print(f"{name:26s} {str(info['production'] or '—'):24s} "
                  f"{len(info['versions']):>9d}  {marker}")
        if prod:
            print("\nProduction registry:")
            for name, version in prod.items():
                print(f"  {name} → {version}")
        else:
            print("\nNo models promoted yet — `seismo model promote` is the "
                  "human gate.")

    out.result(payload, human)
    return EXIT_OK


def cmd_app_run(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard (foreground; Ctrl-C to stop)."""
    import shutil
    import subprocess
    from pathlib import Path

    from .errors import MissingArtifact

    cfg = load_config(args.root)
    app = Path(cfg.root) / "app" / "streamlit_app.py"
    if not app.is_file():
        raise MissingArtifact(
            f"Application not found at {app}",
            hint="This repository ships the app under app/streamlit_app.py — "
                 "check the checkout.",
        )

    streamlit = shutil.which("streamlit")
    if streamlit is None:
        # Fall back to the interpreter's own scripts directory so the command
        # works even when the venv is not on PATH.
        candidate = Path(sys.executable).parent / "streamlit"
        streamlit = str(candidate) if candidate.exists() else None
    if streamlit is None:
        raise MissingArtifact(
            "streamlit is not installed",
            hint="Install the app stack: `pip install -e \".[app]\"` "
                 "or `pip install streamlit plotly pydeck`.",
        )

    command = [
        streamlit, "run", str(app),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false",
    ]
    print(f"Starting SeismoAI → http://localhost:{args.port}")
    print("Stop with Ctrl-C.\n")
    try:
        return int(subprocess.call(command, cwd=str(cfg.root)))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seismo",
        description="SeismoAI — seismic intelligence and geospatial risk analysis.",
        epilog=(
            "Examples:\n"
            "  seismo data fetch                     # download the catalog (resumable)\n"
            "  seismo data fetch --limit 3           # first 3 months only (smoke test)\n"
            "  seismo data validate                  # clean, validate, build processed data\n"
            "  seismo data status                    # show what has been fetched\n"
            "  seismo features build                 # build the ML dataset (cell × month)\n"
            "  seismo geospatial build               # build per-cell activity for maps\n"
            "\nExit codes: 0 ok · 2 usage/config · 3 validation · 4 API · 5 missing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"seismo {__version__}")
    parser.add_argument("--root", type=str, default=None,
                        help="project root containing config/config.yaml")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable JSON output (implies quiet progress)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- data ------------------------------------------------------------
    data = subparsers.add_parser(
        "data", help="data collection and validation",
        description="USGS catalog ingestion and quality gates.",
    )
    data_sub = data.add_subparsers(dest="subcommand", metavar="<subcommand>")

    fetch = data_sub.add_parser(
        "fetch", help="download the USGS catalog into data/raw/",
        description="Fetch monthly chunks; re-running skips completed months.",
    )
    fetch.add_argument("--start", type=_parse_date, default=None,
                       help="start date YYYY-MM-DD (default: config)")
    fetch.add_argument("--end", type=_parse_date, default=None,
                       help="end date YYYY-MM-DD, inclusive (default: config)")
    fetch.add_argument("--min-mag", type=float, default=None,
                       help="minimum magnitude (default: config)")
    fetch.add_argument("--limit", type=int, default=None,
                       help="fetch only the first N months (smoke testing)")
    fetch.add_argument("--force", action="store_true",
                       help="refetch months already marked complete")
    fetch.set_defaults(func=cmd_data_fetch)

    validate = data_sub.add_parser(
        "validate", help="clean raw data, run gates, build processed dataset",
        description="Dedup, filter, hard/missingness gates → data/processed/earthquakes.parquet",
    )
    validate.add_argument("--lenient", action="store_true",
                          help="report hard violations but still write output (exit 0)")
    validate.set_defaults(func=cmd_data_validate)

    status = data_sub.add_parser(
        "status", help="show the fetch manifest summary",
        description="Report which months are fetched, empty, or failed.",
    )
    status.set_defaults(func=cmd_data_status)

    export = data_sub.add_parser(
        "export", help="export a dataset to CSV",
        description="Write a CSV copy alongside the Parquet originals "
                    "(data/export/).",
    )
    export.add_argument("--dataset", choices=["raw", "processed", "features", "activity"],
                        default="processed",
                        help="which dataset to export (default: processed)")
    export.set_defaults(func=cmd_data_export)

    # --- features -------------------------------------------------------
    features = subparsers.add_parser(
        "features", help="ML feature dataset (cell × month)",
        description="Leak-free rolling features and forecasting targets.",
    )
    features_sub = features.add_subparsers(dest="subcommand", metavar="<subcommand>")

    features_build = features_sub.add_parser(
        "build", help="build regional_monthly_features.parquet",
        description="One row per (grid cell, month): trailing features + 30-day target.",
    )
    features_build.set_defaults(func=cmd_features_build)

    # --- geospatial -----------------------------------------------------
    geospatial = subparsers.add_parser(
        "geospatial", help="grid activity and hotspot layers (Module 03)",
        description="Per-cell historical activity indicators for maps.",
    )
    geo_sub = geospatial.add_subparsers(dest="subcommand", metavar="<subcommand>")

    geo_build = geo_sub.add_parser(
        "build", help="build cell_activity.parquet",
        description="Aggregate per-cell counts, magnitudes, recency, and rate.",
    )
    geo_build.set_defaults(func=cmd_geospatial_build)

    # --- model ----------------------------------------------------------
    model = subparsers.add_parser(
        "model", help="train, evaluate, and promote forecasting models",
        description="Fits on the train split only; test is scored exactly once.",
    )
    model_sub = model.add_subparsers(dest="subcommand", metavar="<subcommand>")

    train = model_sub.add_parser(
        "train", help="fit model(s) and write versioned artifacts",
        description="Fits on train, monitors on validation, writes models/<name>/<version>/.",
    )
    train.add_argument("--model", default=None,
                       help=f"one of: {', '.join(MODEL_NAMES)} (default: all)")
    train.add_argument("--all", action="store_true", help="train every model")
    train.set_defaults(func=cmd_model_train)

    evaluate = model_sub.add_parser(
        "evaluate", help="score model(s) and write metrics + figures",
        description="Reports PR-AUC, Brier Skill Score vs climatology, calibration, "
                    "SHAP, and per-region/per-period slices.",
    )
    evaluate.add_argument("--model", default=None,
                          help=f"one of: {', '.join(MODEL_NAMES)} (default: all)")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="val",
                          help="which split to score (default: val)")
    evaluate.add_argument("--version", default=None,
                          help="artifact version (default: latest trained)")
    evaluate.set_defaults(func=cmd_model_evaluate)

    promote = model_sub.add_parser(
        "promote", help="register a version as production (human gate)",
        description="Requires prior evaluation and a named reviewer.",
    )
    promote.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    promote.add_argument("--version", required=True)
    promote.add_argument("--reviewed-by", required=True,
                         help="who reviewed the metrics and calibration plot")
    promote.set_defaults(func=cmd_model_promote)

    listing = model_sub.add_parser(
        "list", help="show trained versions and the production registry",
        description="Lists artifacts under models/ and what registry.json promotes.",
    )
    listing.set_defaults(func=cmd_model_list)

    # --- app ------------------------------------------------------------
    app = subparsers.add_parser(
        "app", help="run the Streamlit dashboard",
        description="Launch the six-page analytical application.",
    )
    app_sub = app.add_subparsers(dest="subcommand", metavar="<subcommand>")

    app_run = app_sub.add_parser(
        "run", help="start the dashboard",
        description="streamlit run app/streamlit_app.py (foreground).",
    )
    app_run.add_argument("--port", type=int, default=8501,
                         help="port to serve on (default: 8501)")
    app_run.add_argument("--headless", action="store_true",
                         help="run without opening a browser")
    app_run.set_defaults(func=cmd_app_run)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    func: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if func is None:
        # `seismo data` with no subcommand → teach, don't error.
        target = parser._subparsers._group_actions[0].choices[args.command]  # type: ignore[union-attr]
        target.print_help()
        return EXIT_USAGE if args.command == "data" else EXIT_OK

    try:
        return int(func(args))
    except SeismoError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code
    except _ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\nInterrupted — re-run `seismo data fetch` to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
