#!/usr/bin/env python3
"""Build notebooks: convert jupytext percent-format .py sources to .ipynb and
execute them so outputs are visible when the notebooks are viewed on GitHub.

Convention (see notebooks/README.md):
    notebooks/03_eda.py      ← source of truth (lintable, diffable)
    notebooks/03_eda.ipynb   ← generated + executed (do not edit by hand)

Usage:
    tools/build_notebooks.py              # convert + execute every notebook
    tools/build_notebooks.py 03           # only notebooks matching "03"
    tools/build_notebooks.py --convert    # convert only (skip execution)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
VENV_BIN = ROOT / ".venv" / "bin"


def _run(cmd: list[str], label: str) -> None:
    print(f"  {label}…", flush=True)
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(
            f"\n✗ {label} failed after {elapsed:.1f}s (exit {result.returncode})\n"
            f"  Fix: run the notebook cell manually — the traceback above names "
            f"the failing line."
        )
    print(f"    done in {elapsed:.1f}s")


def _execute(path: Path) -> None:
    """Execute a notebook in place with the kernel's cwd pinned to the repo root.

    nbconvert's *CLI* executes with cwd = the notebook's own directory, which
    breaks any notebook using repo-root-relative data paths such as
    ``data/processed/earthquakes.parquet``. Driving the preprocessor through
    its Python API lets us set ``resources['metadata']['path']`` explicitly, so
    notebooks can document plain repo-root paths and still run anywhere.
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    print(f"  execute {path.name}…", flush=True)
    start = time.perf_counter()
    notebook = nbformat.read(str(path), as_version=4)
    executor = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    try:
        executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable fix
        elapsed = time.perf_counter() - start
        cell = getattr(exc, "traceback", None) or str(exc)
        sys.stderr.write(f"{cell}\n")
        sys.exit(
            f"\n✗ execute {path.name} failed after {elapsed:.1f}s\n"
            f"  Fix: the traceback above names the failing cell — run that "
            f"cell manually in Jupyter to debug."
        )
    nbformat.write(notebook, str(path))
    print(f"    done in {time.perf_counter() - start:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filter", nargs="?", default="",
                        help="only notebooks whose filename contains this text")
    parser.add_argument("--convert", action="store_true",
                        help="convert to .ipynb but do not execute")
    args = parser.parse_args()

    sources = sorted(NOTEBOOKS.glob("*.py"))
    if args.filter:
        sources = [s for s in sources if args.filter in s.name]
    if not sources:
        print(f"No notebook sources matched {args.filter!r} in {NOTEBOOKS}",
              file=sys.stderr)
        return 1

    jupytext = VENV_BIN / "jupytext"

    for source in sources:
        target = source.with_suffix(".ipynb")
        print(f"▸ {source.name}")
        _run([str(jupytext), "--to", "ipynb", str(source)], f"convert {source.name}")

        if not args.convert:
            _execute(target)

    print(f"\n✓ Built {len(sources)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
