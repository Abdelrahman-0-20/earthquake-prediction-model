# SeismoAI — System Architecture

**Status:** Draft for review — architecture and plan only. No code, data, or repository scaffolding has been created yet.
**Source brief:** `plan1` (project proposal).
**Companion document:** [PLAN.md](PLAN.md) — roadmap, work breakdown, risks, Phase 1 checklist.

**Confirmed scope decisions (2026-11-16):**

| Decision | Choice |
|---|---|
| Geographic coverage | Global USGS catalog (regional slices as filtered views) |
| Development environment | Local VS Code + Python venv, Git |
| Forecasting objective | 30-day probability of ≥1 event above a magnitude threshold per grid cell |

---

## 1. Scope and scientific stance

SeismoAI is a portfolio-grade data science platform with three analytical systems:

| Module | Name | Question it answers |
|---|---|---|
| 01 | Historical Seismic Intelligence | What happened? — patterns in time, magnitude, depth, and geography |
| 02 | Earthquake Probability Modeling | What is the estimated probability of ≥1 event ≥ M4.5 in a grid cell in the next 30 days? |
| 03 | Geospatial Risk Analysis | Where is historical activity concentrated? (hotspots, spatial density, regional comparison) |

### Non-negotiable scientific rules

1. **Probabilities, not predictions.** We estimate probabilities from historical observations. We never claim a forecast of the exact time, location, or magnitude of an individual earthquake.
2. **Activity indicator ≠ disaster risk.** Module 02/03 output is a *seismic activity indicator*. Real risk requires hazard × exposure × vulnerability; exposure/vulnerability data (WorldPop, OpenStreetMap, USGS hazard) is an explicitly deferred extension.
3. **No data leakage.** Every feature computed for a prediction point at time `t` uses only events with `event_time < t`.
4. **Temporal evaluation with purge.** Train/validation/test respect chronology, with a 30-day embargo around split boundaries so overlapping target windows cannot leak.
5. **Catalog completeness is uneven.** Detection and reporting vary by region and era. "No recorded events" ≠ "low seismic activity"; missingness is measured and reported, and a magnitude-of-completeness caveat ships with the app.
6. **Reproducibility.** Raw data is immutable; every derived artifact is regenerable from code + config + seeds; every model is reproducible from a documented data manifest.

---

## 2. System overview

```text
                      ┌────────────────────────┐
                      │     USGS ComCat API    │  external, rate-limited
                      │  fdsnws/event/1/query  │
                      └───────────┬────────────┘
                                  │  chunked CSV/GeoJSON pulls (monthly)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L1  DATA LAYER                                                           │
│  ingestion (resumable, idempotent) ──► data/raw/        immutable        │
│  cleaning + validation gates        ──► data/processed/ earthquakes.pq   │
│  feature engineering (grid, targets)──► data/features/  monthly features │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │  validated dataset contracts (schemas + manifests)
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L2  ANALYSIS LAYER                                                       │
│  Module 01  EDA + statistics (Gutenberg-Richter, rates, trends, depth)   │
│  Module 03  geospatial aggregation (grid activity, hotspots, density)    │
└───────────────────────────┬──────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L3  MODEL LAYER                                                          │
│  baselines (empirical rate, Poisson)  →  ML (LogReg / RF / XGBoost)      │
│  temporal evaluation → calibration → explainability (SHAP)               │
│  persistence: models/<name>/<version>/  +  reports/model_metrics/        │
└───────────────────────────┬──────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L4  PRESENTATION LAYER — Streamlit                                       │
│  01 Overview │ 02 Seismic Map │ 03 Historical │ 04 Forecasting │         │
│  05 Geospatial │ 06 Model Performance                                    │
└──────────────────────────────────────────────────────────────────────────┘

Cross-cutting: config/ · tests/ · docs/ · CLI (`seismo`) · CI · logging · design tokens
```

### Layer contracts (the interfaces between stages)

| Artifact | Produced by | Consumed by | Contract |
|---|---|---|---|
| `data/raw/YYYY-MM.parquet` + manifest | ingestion | cleaning | immutable, one file per month, manifest holds request params, row count, sha256 |
| `data/processed/earthquakes.parquet` | cleaning/validation | EDA, features, maps | validated schema (below), deduplicated event ids |
| `data/features/regional_monthly_features.parquet` | feature engineering | model training, Forecasting page | one row per (cell, observation month), target + features, data dictionary |
| `models/<name>/<version>/` | training | app, evaluation | `model.joblib`, `features.json`, `metrics.json`, `training_config.yaml`, `data_manifest_ref` |
| `reports/model_metrics/` | evaluation | Model Performance page | JSON metrics + PNG/Plotly figures |
| `models/registry.json` | promotion gate (human-approved) | app | pointer to the current production version per model |

Rule: **the app never reads raw files ad hoc.** All page data flows through one access layer (`app/data_access.py`) that loads only named artifacts above.

---

## 3. Data architecture

### 3.1 Source

USGS ComCat — `https://earthquake.usgs.gov/fdsnws/event/1/query`
Official catalog of events from contributing seismic networks; supports filtering by time, magnitude, depth, and geographic bounds; returns CSV or GeoJSON.

**Collection parameters (frozen for reproducibility):**

| Parameter | Value |
|---|---|
| Start date | 2000-01-01 |
| End date | 2026-11-15 (frozen; "now" is not reproducible) |
| Minimum magnitude | 2.5 |
| Geographic coverage | Global |
| Output format | CSV per chunk (GeoJSON optional per chunk) |
| Storage | Parquet (raw + processed + features) |

### 3.2 Ingestion contract

- **Chunking:** monthly windows (`starttime`/`endtime`). The API caps results at **20,000 records per request**; if a month hits the cap, paginate with `limit`/`offset` ordered by time and merge chunks before writing.
- **Idempotency:** one output file per month, `data/raw/YYYY-MM.parquet`. Fetch writes to a temp file, verifies, then atomically renames. Re-running `seismo data fetch` skips months already recorded in the manifest; `--force` refetches.
- **Manifest:** `data/raw/manifest.json` records, per month: request parameters, fetch timestamp, row count, sha256, API warnings. This is the data lineage record referenced by every model artifact.
- **Retries:** exponential backoff on 429/5xx; a month that still fails is marked `failed` in the manifest and the run continues — the run ends nonzero and lists the failed months.

**Ingestion failure modes:**

| Failure | Behaviour |
|---|---|
| Network/API error | retry with backoff → mark month failed → continue → nonzero exit with summary |
| Month exceeds 20k rows | automatic pagination, merged chunk written once |
| API schema drift (new/renamed columns) | validation gate blocks processed build; schema diff printed; pipeline halts (never silently drops columns) |
| Partial/corrupt file | checksum + row-count verification at write time; corrupt months are refetched |
| Event revised between pulls (duplicate `id`, newer `updated`) | dedup keeps the record with the latest `updated` timestamp |

### 3.3 Cleaning and validation gates

Run by `seismo data validate`; nonzero exit on hard failures.

- **Hard checks (fail):** unique `id`; `latitude ∈ [-90, 90]`; `longitude ∈ [-180, 180]`; `time` parses as UTC; `mag` and `depth` numeric and finite.
- **Soft checks (report, do not silently fix):** missing-value rate per column (`nst`, `gap`, `dmin`, `rms` are frequently absent — **measure, never blindly fill**); negative depths (physically near-surface noise) — retained but flagged; magnitude/depth outliers listed.
- **Catalog-quality report:** `reports/data_quality/missingness.json` + a completeness discussion (magnitude of completeness varies by region and era).

### 3.4 Processed dataset schema

`data/processed/earthquakes.parquet` — validated (Pandera schema once Week 1 lands):

| Field | Type | Unit | Notes |
|---|---|---|---|
| `id` | string | — | USGS event id, primary key |
| `time` | datetime64[ns, UTC] | — | event timestamp |
| `updated` | datetime64[ns, UTC] | — | catalog revision time (dedup) |
| `latitude` | float | degrees | |
| `longitude` | float | degrees | |
| `depth` | float | km | negative values flagged, retained |
| `mag` | float | magnitude | |
| `mag_type` | string | — | ml, mw, mb… |
| `place` | string | — | location description |
| `nst`, `gap`, `dmin`, `rms` | float | — | catalog quality; nullable, missingness reported |
| `type` | string | — | event type; keep `earthquake` rows only in analysis |
| `year`, `month` | int | — | derived, for fast grouping |

### 3.5 Feature dataset (Module 02 input)

**Unit of analysis:** one row per (grid cell, monthly observation date).

- **Grid:** global 1°×1° cells (configurable in `config/config.yaml`). The full grid is 64,800 cells, but we materialize rows only for **active cells** (≥1 qualifying event in the catalog history) — estimated 3,000–8,000 cells.
- **Observation dates:** monthly from 2001-01 (first date with a full 365-day lookback) to 2026-10.
- **Row budget:** roughly 1–3M rows, ~100–300 MB parquet — comfortable locally. Optional size lever: emit rows only for cells with activity in the trailing 365 days.

**Features (all computed strictly from events before the observation date):**

| Feature | Meaning |
|---|---|
| `events_last_30d` / `events_last_90d` / `events_last_365d` | event counts in trailing windows |
| `mean_magnitude_90d` | average magnitude (trailing 90 days) |
| `max_magnitude_90d` | maximum magnitude (trailing 90 days) |
| `mean_depth_90d` | average depth (trailing 90 days) |
| `days_since_last_event` | recency (log-transformed / capped) |
| `latitude_center`, `longitude_center` | grid cell centre |
| *(optional)* `magnitude_weighted_count_90d`, `rms_mean_90d`, `gap_mean_90d`, month sin/cos | magnitude weighting, catalog quality, seasonality |

**Target (computed from future observations only):**

```text
target = 1  if ≥1 event with mag ≥ 4.5 occurs in the cell in (t, t + 30 days]
target = 0  otherwise
```

**Purge rule:** any row whose 30-day target window crosses a train/val/test boundary is dropped (embargo). A dedicated leakage test suite asserts both the feature-window and target-window rules.

---

## 4. Analytical architecture

### Module 01 — Historical Seismic Intelligence

- Magnitude distribution + Gutenberg-Richter fit (`log10 N = a − bM`) with b-value estimate per region.
- Event frequency over time (monthly/yearly counts, rolling averages, trend decomposition).
- Depth distribution and depth-vs-magnitude relationships.
- Geographic clustering and most-active regions.
- Statistical baseline outputs reused by Module 02 (climatological rates per cell).

### Module 03 — Geospatial Risk Analysis

- Grid-based activity counts (the same grid as Module 02 — one grid definition, one source of truth).
- Hotspot identification: cells/regions with elevated historical rates; spatial density surfaces (hexbin/KDE) for the map layer.
- Regional comparison: normalized rates (events per cell-year) across named regions.
- Communication guardrail: displayed as **activity indicators**, with the limitations panel visible on the page.

---

## 5. Machine learning architecture

### 5.1 Problem framing

> Given historical seismic activity in a geographic grid cell, estimate the probability that at least one earthquake of magnitude ≥ 4.5 occurs in that cell during the following 30 days.

This is probabilistic binary classification over (cell, month) prediction points — a forecasting experiment with honest uncertainty, not a claim of reliable earthquake prediction.

### 5.2 Temporal splits (chronological, purged)

| Split | Period (default, configurable) |
|---|---|
| Train | 2000-01 → 2018-12 |
| Validation | 2019-01 → 2022-12 |
| Test | 2023-01 → 2026-11 (held out until final evaluation) |

- 30-day embargo around boundaries; rows with overlapping target windows dropped.
- Random row splitting is forbidden — it leaks future information in a temporal catalog.
- Test set is touched once, at the end of Week 3, for the final report.

### 5.3 Models

| Model | Role |
|---|---|
| Empirical climatological rate | cell-level historical frequency — the baseline everything must beat |
| Poisson count model | statistical baseline (rate estimation per cell/region) |
| Logistic Regression | linear baseline, standardized features |
| Random Forest | nonlinear interactions |
| XGBoost | advanced tabular model |

Imbalance handling: `class_weight` / `scale_pos_weight`, threshold tuning on validation (optimize F1 or a stated recall target), and PR-AUC as the headline discriminative metric. Hyperparameter search is modest (randomized search on train/val only) to keep the experiment reproducible.

### 5.4 Evaluation and calibration

| Metric | Why |
|---|---|
| PR-AUC, ROC-AUC | discriminative performance under imbalance |
| Precision / Recall / F1 at tuned threshold | operational behaviour |
| **Brier score + Brier Skill Score vs climatology** | probabilistic quality — a forecast that says 20% must verify ~20% of the time |
| Calibration curve (reliability diagram), ECE | calibration is a first-class requirement, not an afterthought |
| Confusion matrix, FP/FN analysis | error inspection |
| Per-region and per-time-period slices | a global average can hide regional failure |

### 5.5 Explainability

- SHAP (TreeExplainer for XGBoost/Random Forest): global feature importance + per-prediction explanations for a selected (cell, date).
- Documented caveat: trailing-window features are correlated; SHAP attributions are explanatory, not causal.

### 5.6 Model persistence and promotion

- Each training run writes `models/<name>/<version>/` containing the fitted pipeline, `features.json` (exact feature list/order), `metrics.json`, `training_config.yaml`, and `data_manifest_ref` (data lineage).
- `models/registry.json` points to the approved production version consumed by the app.
- **Promotion gate (human-in-the-loop):** a model version becomes production only after a human reviews validation metrics, calibration plot, and regional slices. App code never trains — it only loads registry entries.
- Deferred (documented, not built in v1): MLflow tracking, automated retraining triggers, drift monitoring on newly observed months.

---

## 6. Application architecture (Streamlit)

### 6.1 Pages

| # | Page | Purpose | Primary data |
|---|---|---|---|
| 01 | Overview | KPIs, global activity map, frequency trend, magnitude/depth distributions, top regions, recent events | processed catalog |
| 02 | Global Seismic Map | interactive map; date/magnitude/depth/geographic filters; event detail | processed catalog |
| 03 | Historical Analysis | time-series trends, distributions, regional comparison, clusters | processed catalog |
| 04 | Forecasting | region/cell selection, probability estimates, model comparison, uncertainty | features + model registry |
| 05 | Geospatial Analysis | grid activity, hotspots, spatial density, regional indicators | features + grid aggregates |
| 06 | Model Performance | model comparison, confusion matrices, PR curves, calibration, feature importance, SHAP | `reports/model_metrics/` |

All page indicators respond to the shared filter state (date range, magnitude, depth, region).

### 6.2 Components and data access

- `app/components/filters.py` — shared sidebar filter state (`st.session_state`).
- `app/components/kpi_cards.py` — flat KPI cards (thin border, minimal shadow).
- `app/components/charts.py` — Plotly theme wrapper: consistent axis titles, units, and fonts on every chart.
- `app/components/map_layers.py` — PyDeck scatter/heatmap layers with pre-aggregated data.
- `app/data_access.py` — the only module that touches the filesystem; loads named artifacts (Section 2), cached.

### 6.3 Design system

| Token | Value |
|---|---|
| Deep navy (primary) | `#14263D` |
| White (surface) | `#FFFFFF` |
| Light gray (background) | `#E9EEF3` |
| Blue accent | `#2583A6` |
| Typography | Inter or Source Sans 3; clear hierarchy; consistent sizes |

Principles: flat KPI cards, thin borders, minimal shadows, consistent chart labels, clean navigation. **No gradients, decorative animation, or emojis** — this is a scientific instrument. Plotly for charts; PyDeck (or Folium) for geography; Streamlit native layout/theme instead of a custom frontend.

### 6.4 Performance

- `st.cache_data` (with TTL) for dataset loads; `st.cache_resource` for model objects.
- Maps render pre-aggregated layers (hexbin/grid aggregates), not 500k raw points; raw points only on zoomed event-detail views with magnitude filters.
- Heavy artifacts (SHAP, large figures) load lazily on their page only.
- Where this matters: Streamlit re-executes the script on every interaction — expensive computation must live behind caches and named aggregate artifacts.

### 6.5 Runtime choice: web, not desktop

| Concern | Streamlit web (chosen) | Electron/Tauri desktop (rejected for v1) |
|---|---|---|
| Distribution | shareable URL (Streamlit Community Cloud) | per-OS installers, signing + notarization gauntlet |
| Update burden | redeploy | staged auto-update fleet — the hardest code you own |
| Footprint | browser only | 80–150 MB Electron / webview variance with Tauri |
| Fit for audience | portfolio reviewers, remote viewers | offline/field use — not a v1 requirement |

A desktop wrapper (Tauri around a local Streamlit server) remains a documented future option; it would add IPC, signing, and updater work with no v1 benefit.

---

## 7. Cross-cutting concerns

### 7.1 Configuration

Single source of truth: `config/config.yaml` — collection window, magnitude threshold, grid resolution, feature windows, split dates, seeds, thresholds. CLI flags override env vars override config file (documented precedence).

### 7.2 Reproducibility

- Pinned dependencies (`requirements.txt` with exact versions) + documented Python version.
- Fixed random seeds recorded in `training_config.yaml`.
- Data manifest (row counts, hashes, request params) referenced by every model artifact.
- Raw data and model artifacts are **not** committed to Git; the repo ships reproducible fetch scripts + a small sample dataset for tests and app development.

### 7.3 Testing (built in from Week 1)

| Test type | What it protects |
|---|---|
| Ingestion tests (mocked API) | chunking, pagination, resume, manifest integrity |
| Schema tests | processed dataset contract |
| **Leakage tests** | features at `t` use only events `< t`; target uses only `(t, t+30d]` |
| Split-integrity tests | chronological order, no overlap, purge respected |
| Feature/target spot checks | hand-verified rows against raw events |
| App smoke tests | each page renders against the sample dataset |

### 7.4 Observability

Structured per-stage logs (`reports/logs/`) with run ids — every derived artifact records which run produced it, mirroring trace-style lineage.

### 7.5 Ethics, safety, and honesty in the product

- No personal data involved; external API content is treated as untrusted and schema-validated before ingestion.
- Every Forecasting/Geospatial view carries a visible limitations panel: probabilities are historical estimates, catalog completeness varies, and this is not a safety or early-warning system.
- The app never emits deterministic claims ("an earthquake will hit X").

---

## 8. Explicit non-goals (v1)

- Exact time/location/magnitude prediction of individual earthquakes.
- Disaster risk estimation (needs exposure + vulnerability data) — extension path documented.
- Real-time alerting or notification systems.
- Kubernetes, distributed infrastructure, or cloud data platforms.
- Desktop packaging (see 6.5), MLflow, Pandera in the first iteration (added only if Week 1–3 finish ahead of schedule).
