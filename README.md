# SeismoAI — Earthquake Forecasting & Geospatial Risk Analysis

A data-driven platform for analyzing historical seismic activity, estimating
regional earthquake probabilities, and visualizing seismic patterns.

> **Scientific stance.** SeismoAI estimates *probabilities* from historical
> observations. It does **not** predict the exact time, location, or magnitude
> of individual earthquakes — that kind of prediction is not scientifically
> reliable. Outputs are seismic *activity indicators*, not disaster risk.

## Status

| Week | Scope | State |
|---|---|---|
| 1 | Data engineering, validation, EDA | ✅ complete |
| 2 | Geospatial grid, feature engineering, leakage guards | ✅ complete |
| 3 | Machine learning (baselines + 3 models, calibration, SHAP) | ✅ complete |
| 4 | Streamlit application + deployment | 🔄 in progress |

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**,
**[docs/PLAN.md](PLAN.md)** and **[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)**.

## Results so far

| Artifact | Value |
|---|---|
| Raw events | 715,183 across 323 months (2000-01 → 2026-11), 0 fetch failures |
| Processed events | 707,438 (7,745 non-earthquake removed, 0 duplicates) |
| Grid | 1° cells → 11,503 cells with activity |
| Feature dataset | 3,565,930 rows (cells × 310 month-start observation dates) |
| Target | ≥1 event M≥4.5 in next 30 days → **104,140 positives (2.92%, 1:33)** |
| Regional positive rate | **0.62% → 11.98%** across named regions (19× spread) |
| Splits | train 2,484,648 · val 552,144 · test 529,138 (chronological, purged) |
| Tests | 78 passing incl. a data-leakage suite |
| Notebooks | 5 executed, 0 errors, 19 figures |

### EDA headline results

| Finding | Value |
|---|---|
| **Gutenberg-Richter b-value** | **b = 1.101 ± 0.002** (95% CI [1.097, 1.106]) for M ≥ 4.4, N = 234,532 |
| Magnitude of completeness | Mc = 4.4 (MaxC of the globally-complete mode) |
| Mc sensitivity | 1.02 – 1.15 across Mc 4.4–6.0 and LSQ cross-check → consistent with the global ≈1.0 |
| Full-range b (M ≥ 2.5) | 0.319 ± 0.000 — **biased by incompleteness, not usable** |
| Depth | median 20.4 km · 80.1% ≤ 70 km · 3.81% > 300 km (subduction) |
| Most active regions | Alaska 122,898 (17.4%) · Indonesia 45,822 · Puerto Rico 29,362 |
| Busiest month | 2018-07 = 7,636 events (Kīlauea swarm) |

### Model results (Week 3)

Six models trained on 2.48M rows (chronological split), evaluated on the
**test holdout scored exactly once** (2023 → 2026), thresholds frozen from
validation. Reference = the climatological per-cell rate (BSS = 0 by definition).

| model | PR-AUC | ROC-AUC | Brier | **BSS** | ECE |
|---|---|---|---|---|---|
| baseline_global_rate | 0.031 | 0.500 | 0.0297 | −0.2261 | 0.0024 |
| baseline_poisson | 0.249 | 0.869 | 0.0292 | −0.2036 | 0.0030 |
| logreg | 0.282 | 0.870 | 0.0254 | −0.0491 | 0.0037 |
| baseline_climatology | 0.335 | 0.891 | 0.0242 | 0.0000 | 0.0040 |
| random_forest | 0.348 | 0.895 | 0.0239 | **+0.0158** | **0.0006** |
| **xgboost** | **0.352** | **0.899** | **0.0238** | **+0.0189** | 0.0010 |

**Reading the result honestly.** Two models beat simply counting history, and
the margin is *small* (+1.9% Brier skill) — which is the truthful shape of
earthquake forecasting: recent activity and location carry real but limited
signal. Test improved slightly over validation (xgboost +0.0146 → +0.0189),
so the chronological protocol did not overfit.

Two engineering findings worth recording:

* **Class weighting destroyed calibration.** With `class_weight='balanced'`
  the models ranked well but predicted 0.28 against a 0.032 base rate —
  Brier Skill Score ≈ **−4**. Balancing forces the effective prior toward
  50/50, so probabilities stop being probabilities. Disabled by default
  (`models.use_class_weights`), with the measurement documented in code.
* **An unconstrained RandomForest grew to depth 47 / 9M nodes / 724 MB**,
  making SHAP traversal impractical. Capped at `max_depth=16`,
  `min_samples_leaf=50` → 172 MB, depth 16, and evaluation dropped from
  30+ min to ~5 min.

## Quick start (golden path)

```bash
make setup        # venv + pinned dependencies
make data         # fetch the USGS catalog (resumable)
make validate     # clean, validate, build processed dataset
make features     # build the ML dataset (cell × month)
make train        # train all six models
make evaluate     # metrics, calibration, SHAP, per-region slices
make notebooks    # convert + execute all notebooks
make test         # run the test suite
make app          # run the Streamlit dashboard
```

CLI directly:

```bash
seismo data fetch                    # download into data/raw/ (resumable, idempotent)
seismo data validate                 # build data/processed/earthquakes.parquet + quality report
seismo data status                   # what has been fetched, what failed
seismo data export --dataset processed   # CSV copy for interchange (data/export/)
seismo features build                # build the ML dataset (cell × month)
seismo geospatial build              # per-cell activity for maps
seismo model train --all             # fit six models → models/<name>/<version>/
seismo model evaluate --split val    # metrics + calibration + SHAP + figures
seismo model evaluate --split test   # the holdout — exactly once, after review
seismo model list                    # trained versions + production registry
seismo model promote --model xgboost --version <v> --reviewed-by "name"
seismo app run                       # launch the dashboard (http://localhost:8501)
```

`--json` gives machine-readable output; exit codes are a contract:
`0` ok · `2` usage/config · `3` validation · `4` API · `5` missing artifact.

## Running the application

```bash
make app                       # or: seismo app run
# open http://localhost:8501
```

Six pages: **Overview · Seismic Map · Historical Analysis · Forecasting ·
Geospatial Analysis · Model Performance**. All artifact dependencies are
checked on startup; if something is missing the sidebar shows the exact
command that builds it instead of raising a traceback.

**Streamlit Community Cloud:** point it at this repository with
`app/streamlit_app.py` as the entry point and `requirements.txt` as the
dependency file. Data and model artifacts are gitignored, so a deployment
needs them provisioned separately (or a small sample dataset for demo mode) —
see `docs/METHODOLOGY.md` §10 for what is and is not committed.

## Repository layout

```
app/         Streamlit application (6 pages) — Week 4
src/seismo/  pipeline package: data, features, models, geospatial
tests/       78 tests incl. leakage, split-integrity, CLI contract
notebooks/   01–05 (source .py → executed .ipynb; see notebooks/README.md)
config/      config.yaml — single source of truth
data/        raw / processed / features / export (gitignored) + samples/ (committed)
reports/     figures, model metrics, data-quality reports
docs/        architecture, plan, data dictionary
tools/       notebook build tooling
```

## Data source

USGS ComCat earthquake catalog —
[catalog documentation](https://earthquake.usgs.gov/data/comcat/) ·
[API](https://earthquake.usgs.gov/fdsnws/event/1/).
Raw data and model artifacts are not committed; reproducible fetch scripts and a
small sample dataset are.

## Known catalog limitations (carried into every analysis)

* **Magnitude of completeness varies by region and era.** The global catalog is
  incomplete below ≈M4.4 — small quakes are only detected near dense station
  networks. The frequency-magnitude histogram is bimodal for exactly this
  reason (peak at the M2.5 collection floor, second peak at M4.4, counts
  doubling at M4.0). Gutenberg-Richter b must therefore be estimated above the
  completeness magnitude: **b = 1.101 ± 0.002 for M ≥ 4.4**, never the
  full-range value of 0.32.
* **Absence of records ≠ absence of seismicity.** Detection coverage differs
  spatially and temporally.
* **Historical activity ≠ disaster risk.** Risk additionally needs exposure and
  vulnerability data (deferred; see `docs/PLAN.md` §7).
