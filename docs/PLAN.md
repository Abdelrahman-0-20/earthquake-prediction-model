# SeismoAI — Implementation Plan

**Status:** Plan for approval — nothing has been built yet. On approval, execution starts at Phase 1 below.
**Companion document:** [ARCHITECTURE.md](ARCHITECTURE.md) — system design, data contracts, ML and app architecture.
**Source brief:** `plan1`.

**Confirmed scope decisions (2026-11-16):**

| Decision | Choice | Consequence |
|---|---|---|
| Geographic coverage | Global USGS catalog | Full 2000–2026 catalog; regional deep-dives as filtered views |
| Development environment | Local VS Code + venv | Reproducible CLI pipeline + notebooks + app in one workspace |
| Forecasting objective | 30-day threshold probability | Cell-level probabilistic classification; calibration is a headline metric |

---

## 1. Repository layout

```text
SeismoAI/
│
├── app/
│   ├── streamlit_app.py           # shell, navigation, theme
│   ├── data_access.py             # ONLY module touching the filesystem
│   ├── pages/
│   │   ├── overview.py
│   │   ├── seismic_map.py
│   │   ├── historical_analysis.py
│   │   ├── forecasting.py
│   │   ├── geospatial_analysis.py
│   │   └── model_performance.py
│   └── components/
│       ├── kpi_cards.py
│       ├── charts.py
│       ├── filters.py
│       └── map_layers.py
│
├── src/
│   ├── data/
│   │   ├── ingestion.py           # USGS API client, chunking, manifest
│   │   └── preprocessing.py       # cleaning + validation gates
│   ├── features/
│   │   └── engineering.py         # grid, rolling features, targets
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── geospatial/
│       └── analysis.py            # grid aggregation, hotspots, density
│
├── cli/
│   └── seismo.py                  # `seismo` command entry point
│
├── data/                          # gitignored except samples/
│   ├── raw/                       # immutable USGS pulls + manifest.json
│   ├── processed/                 # earthquakes.parquet
│   ├── features/                  # regional_monthly_features.parquet
│   └── samples/                   # tiny committed sample for tests/app dev
│
├── models/                        # gitignored (registry.json may be committed)
├── reports/
│   ├── figures/
│   ├── model_metrics/
│   ├── data_quality/
│   └── logs/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_geospatial_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_model_experiments.ipynb
│
├── tests/
├── config/
│   └── config.yaml                # single source of truth
├── docs/                          # architecture, methodology, decisions
├── requirements.txt               # pinned versions
├── Makefile                       # golden-path commands
├── README.md
└── .gitignore
```

Git policy: raw data, processed data, features, and model binaries are **not** committed (too large / regenerable). The repo carries reproducible scripts, the small sample dataset, configs, tests, and reports/figures that are small enough to share.

---

## 2. Developer experience: the golden path

The pipeline is exposed as one CLI (`seismo`) so the entire project runs from documented commands — no tribal knowledge, no notebook-only steps.

```text
seismo data fetch    --start 2000-01-01 --end 2026-11-15 --min-mag 2.5 --chunk monthly
seismo data validate                       # hard gates + missingness report
seismo features build --grid 1.0 --window 30
seismo model train   --model xgboost       # or --all
seismo model evaluate --split test         # writes reports/model_metrics/
seismo app run                             # streamlit run app/streamlit_app.py
```

**CLI contract (tooling discipline):**

| Behaviour | Rule |
|---|---|
| `--help` | every command and subcommand has examples, not just flags |
| Errors | name what failed, why, and the exact fix; stack traces only with `--verbose` |
| Scriptability | `--json` output, `--quiet`, `--dry-run` for anything that writes |
| Exit codes | `0` success · `2` usage error · `3` validation failed · `4` API/network failure · `5` missing artifact |
| Speed | lazy imports; `seismo --help` budget < 1.2 s (checked in CI) |

**Makefile golden path** (clone → running app):

```text
make setup       # venv + pinned dependencies
make data        # fetch + validate
make features    # build feature dataset
make train       # train all models
make evaluate    # evaluation + reports
make app         # run Streamlit
make test        # pytest + lint
```

Guardrail: the safe path is the easy path — destructive commands (`--force` refetch, model overwrite) require explicit flags and support `--dry-run`.

---

## 3. Four-week roadmap

### Week 1 — Data engineering & EDA

**Goal:** a reproducible, validated earthquake dataset and first scientific analysis.

Tasks:
1. Repo init: structure above, `.gitignore`, pinned `requirements.txt`, venv, `config/config.yaml`.
2. `seismo` CLI skeleton with `--help`, `--json`, exit codes.
3. USGS ingestion: monthly chunking, 20k pagination, retries, atomic writes, resumable manifest.
4. Cleaning + validation gates (hard/soft checks, missingness report).
5. Processed dataset build (`earthquakes.parquet`).
6. Notebooks 01–03: collection, cleaning, EDA (magnitude distribution, frequency over time, depth, geography, Gutenberg-Richter b-value).
7. Tests: ingestion (mocked API), schema, CLI; CI workflow (lint + tests).

**Deliverable:** clean reproducible dataset + EDA notebook + figures in `reports/figures/`.
**Acceptance criteria:**
- `seismo data fetch` is resumable and idempotent (re-run skips completed months; `--force` refetches).
- `seismo data validate` exits 0 on good data and 3 on corrupted input, producing a missingness report.
- Row counts, date range, and magnitude range documented against the manifest.
- EDA answers: magnitude distribution, temporal trend, depth profile, most active regions.

### Week 2 — Geospatial analysis & feature engineering

**Goal:** an ML-ready (cell, month) dataset with provably leak-free features.

Tasks:
1. Grid system (1° global, config-driven) + named region definitions.
2. Monthly per-cell aggregation; historical activity indicators (Module 03 basis).
3. Rolling features (`events_last_30/90/365d`, magnitude/depth statistics, recency).
4. Target construction: ≥1 event M4.5 in the following 30 days.
5. **Leakage test suite** + split-integrity tests (chronology + 30-day purge).
6. Class-imbalance and catalog-completeness analysis; row/size budget documented.
7. Notebook 04–05; feature data dictionary.

**Deliverable:** `data/features/regional_monthly_features.parquet` + data dictionary + leakage tests.
**Acceptance criteria:**
- Leakage tests pass: features at `t` use only events `< t`; targets use only `(t, t+30d]`.
- Target verified by hand on sampled rows against raw events.
- Positive-class rate quantified globally and per region (imbalance characterized, not assumed).

### Week 3 — Machine learning

**Goal:** evaluated, calibrated, explainable models — honestly compared.

Tasks:
1. Baselines: empirical climatological rate + Poisson.
2. Logistic Regression, Random Forest, XGBoost with chronological train/val protocol.
3. Threshold tuning on validation; imbalance handling documented per model.
4. Calibration: reliability diagrams, Brier + Brier Skill Score vs baseline, ECE.
5. Final test-set evaluation (touched once), per-region and per-period slices, FP/FN analysis.
6. SHAP explanations (global + per-prediction).
7. Model persistence + registry; **promotion gate** = human review of metrics/calibration before registry update.
8. Notebook 06 + `reports/model_metrics/`.

**Deliverable:** evaluated models, metrics reports, saved prediction pipeline.
**Acceptance criteria:**
- All four model types reported side by side; Brier Skill Score vs climatology reported honestly (a model that cannot beat baseline is reported, not hidden).
- Calibration plot included; "a 20% estimate verifies ~20% of the time" is testable in the report.
- Full retrain from config reproduces metrics (seeds + manifest).

### Week 4 — Streamlit app & deployment

**Goal:** a professional six-page analytical dashboard, publicly deployed.

Tasks:
1. App shell: navigation, theme tokens (`#14263D` / `#E9EEF3` / `#2583A6`), Inter/Source Sans 3.
2. Shared components (filters, KPI cards, chart theme, map layers) + `data_access.py`.
3. Six pages (Overview, Seismic Map, Historical, Forecasting, Geospatial, Model Performance).
4. Model integration via registry; SHAP lazy-loaded on Model Performance.
5. Performance pass: caching, map pre-aggregation, first-render timing.
6. Limitations panels on Forecasting/Geospatial; app smoke tests on sample data.
7. Documentation: README (setup + golden path), METHODOLOGY.md, limitations; deploy to Streamlit Community Cloud.

**Deliverable:** public SeismoAI application + polished GitHub repository.
**Acceptance criteria:**
- All six pages reflect shared filters; each shows consistent labels/units.
- First render acceptable locally (< ~5 s with cached data); maps aggregated, not raw 500k points.
- Repository passes the success-criteria table (Section 6) end to end.

---

## 4. Phase 1 kickoff checklist (first execution step after approval)

1. Create `SeismoAI` repository + directory structure (Section 1).
2. `.gitignore` for `data/raw`, `data/processed`, `data/features`, `models/`, venv, notebooks checkpoints.
3. Python venv + pinned dependencies (`pandas numpy requests pyarrow` first; geospatial/ML/streamlit pins next).
4. `config/config.yaml` with frozen collection window and thresholds.
5. `seismo` CLI skeleton (help, version, exit-code conventions) + `Makefile`.
6. `src/data/ingestion.py`: chunked fetch, pagination, retries, atomic write, manifest.
7. First real data pull (monthly chunks) + validation report.
8. Initial tests + CI workflow.

Deliverable of Phase 1: ingestion runs end to end on real USGS data, resumable and validated.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| API 20k/request cap or rate limiting on big months | Medium | Medium | monthly chunks + automatic pagination + backoff + resume manifest |
| API schema drift breaks pipeline | Low | High | validation gate halts with schema diff; never silently drop columns |
| Data leakage from rolling/target windows | Medium | **Critical** | dedicated leakage + split-integrity tests from Week 2; purge rules |
| Class imbalance inflates accuracy illusion | High | Medium | PR-AUC + Brier headline metrics; threshold tuning on validation |
| Catalog incompleteness misread as low activity | High | Medium | missingness reporting + completeness caveat in-app |
| Random splits leak future (methodological error) | Medium | Critical | chronological splits enforced by config and tested |
| Streamlit slow with global catalog | Medium | Medium | cached loads, pre-aggregated map layers, filter-driven queries |
| Overclaiming ("prediction") damages credibility | Medium | High | terminology locked to "probability estimate / activity indicator"; visible limitations panels |
| Scope creep (MLflow, exposure data, desktop) | Medium | Medium | non-goals list in ARCHITECTURE §8; extensions documented, not built |
| Large artifacts bloating Git | Medium | Low | gitignore policy + sample dataset + reproducible fetch scripts |

---

## 6. Success criteria → evidence

| Area | Expected evidence | Produced in |
|---|---|---|
| Data engineering | API ingestion + reproducible cleaning (`seismo data fetch/validate`) | Week 1 |
| EDA | notebooks 01–03 + `reports/figures/` | Week 1 |
| Statistics | Gutenberg-Richter/rates analysis + probability baseline | Weeks 1–3 |
| ML | four comparable models (baseline + 3 ML) | Week 3 |
| Geospatial | interactive global map + grid/hotspot analysis | Weeks 2, 4 |
| Evaluation | temporal testing, purge, calibration, Brier Skill Score | Week 3 |
| Explainability | SHAP + feature importance on Model Performance page | Weeks 3–4 |
| Deployment | functional Streamlit app (6 pages) | Week 4 |
| Documentation | README + METHODOLOGY + architecture/limitations docs | Week 4 |

---

## 7. Extension path (documented, explicitly deferred)

1. **Exposure-aware risk layer:** WorldPop population + OSM infrastructure + USGS hazard → real risk framing (required before the word "risk" is used in output).
2. **Experiment tracking:** MLflow (runs, comparison, registry) once the model count grows.
3. **Data validation at scale:** Pandera schemas enforced in CI on the full dataset.
4. **Drift monitoring:** score/calibration drift on newly observed months + retraining policy.
5. **Desktop/field distribution:** Tauri wrapper around a local Streamlit server (only if offline use becomes a requirement).

---

## 8. Next steps

1. Review and approve (or amend) this plan and the architecture document.
2. Execute **Phase 1** (Section 4): repository, environment, ingestion pipeline, first validated data pull.
3. Proceed week by week; each week ends with its deliverable + acceptance criteria checked.
