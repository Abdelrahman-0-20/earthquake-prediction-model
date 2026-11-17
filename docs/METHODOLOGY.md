# SeismoAI — Methodology

How every number in this project was produced, and what the numbers do and
do not mean. This document is the reference for anyone reviewing, extending,
or reproducing the work.

---

## 1. Scientific stance

**SeismoAI estimates probabilities from historical observations. It does not
predict the time, location, or magnitude of individual earthquakes** — that
kind of prediction is not scientifically reliable, and no output of this
system should be read as one.

Three consequences follow, and they are enforced in code, not just stated:

1. Every probabilistic output ships with calibration evidence (does a stated
   "20%" verify ~20%?) rather than a point claim.
2. Every map and ranking is labelled a **historical activity indicator**.
   Disaster risk additionally requires *exposure* and *vulnerability*, which
   this project does not model.
3. Absence of recorded events is never treated as evidence of absence of
   seismicity — catalog completeness varies by region and era.

---

## 2. Data source and collection

| Parameter | Value |
|---|---|
| Source | USGS ComCat — `https://earthquake.usgs.gov/fdsnws/event/1/query` |
| Window | 2000-01-01 → 2026-11-15 (frozen; "now" is not reproducible) |
| Minimum magnitude | 2.5 |
| Coverage | Global |
| Chunking | Monthly, with automatic pagination above 20,000 rows/request |
| Storage | Parquet (canonical) + CSV export for interchange |

**Result:** 323/323 months fetched, 715,183 events, zero failures.

Ingestion is idempotent and resumable: one file per month plus a
`manifest.json` recording request parameters, row count, and sha256 per
month. Re-running skips completed months; `--force` refetches. Writes are
atomic (temp file → verify → rename), so an interrupted run never leaves a
file that looks complete but is not.

Retries use exponential backoff with `Retry-After` support on 429/5xx; 4xx
responses fail immediately because retrying a rejected request cannot succeed.

---

## 3. Cleaning and validation

Two tiers, deliberately separated.

**Hard gates** (fail the build, exit code 3):
unique event ids; coordinates within valid ranges; timestamps parse as UTC;
magnitude and depth numeric and finite.

**Soft checks** (measured and reported, never silently fixed):

| Finding | Count | Handling |
|---|---|---|
| Non-earthquake events (explosions, mining blasts, ice quakes, …) | 7,745 across 19 types | removed from the analysis dataset, counts recorded |
| Duplicate event ids | 0 | keep latest catalog revision |
| Negative depths (near-surface events) | 4,359 | retained, flagged |
| Events below the M2.5 collection filter | 5 | retained, explained (magnitude revised downward *after* index time — the catalog is a living document) |
| Missing catalog-quality fields | `dmin` 51%, `magError` 50%, `nst` 33%, `gap` 16.5%, `rms` 15.2% | **measured, never imputed** |

Missingness is not filled because these fields are network-dependent:
imputing `nst` or `gap` would fabricate instrument information that was never
recorded. The processed dataset guarantees: 707,438 unique earthquakes,
2000-01-01 → 2026-11-15, magnitudes 0.7–9.1.

---

## 4. Magnitude of completeness — why b = 0.32 is not the answer

The frequency–magnitude histogram is **bimodal**: a peak at the M2.5
collection floor (46,678 events in [2.5, 2.6)), a valley at M3.75 (10,879),
and a second peak at M4.4 (45,939). Counts *double* exactly at M4.0.

The cause is spatially varying detection: small earthquakes are only recorded
near dense station networks, while M4.4+ events are detected globally. The
catalog is therefore a mixture of a network-complete population and a
globally-complete one, and it does not follow a single Gutenberg–Richter law
over the full M2.5+ range.

| Estimate | b-value | Verdict |
|---|---|---|
| Full range, M ≥ 2.5 | 0.319 ± 0.000 | **biased by incompleteness — not usable** |
| Above completeness, M ≥ 4.4 | **1.101 ± 0.002** (95% CI [1.097, 1.106]) | the value to quote |

Estimator: Aki (1965) maximum likelihood for binned data,
`b = log10(e) / (mean(m) − Mc + Δm/2)`, Δm = 0.1, N = 234,532. Magnitude of
completeness by maximum curvature. Sensitivity across Mc = 4.4–6.0 and a
least-squares cross-check gives the range **1.02–1.15**, consistent with the
globally typical value near 1.0.

Cross-checked against the live USGS API (a 2020-01 M4.4–4.5 query returned
exactly the rows stored locally) to confirm the step is a property of the
source catalog, not an ingestion bug.

---

## 5. Feature engineering — the temporal contract

One row per **(1° grid cell, month-start observation date)**:
**3,565,930 rows = 11,503 active cells × 310 dates** (2001-01 → 2026-10).

For observation date `t`:

| | Window | Rationale |
|---|---|---|
| **Features** | `[t − W, t)` | strictly before `t` — available at prediction time |
| **Target** | `[t, t + 30d)` | strictly after `t` |

The windows are disjoint with no overlap, so a feature can never see an event
that contributes to its own label. Features: trailing event counts (30/90/365
days), mean/max magnitude and mean depth over 90 days, trailing count of
target-class (M≥4.5) events, days since last event, and cell coordinates.

**Conventions:**
- No NaNs. Statistic windows with no events are filled `0.0` — unambiguous
  for magnitude (catalog minimum 0.7) and neutral for linear models; the count
  features disambiguate.
- `days_since_last_event` falls back to *days since catalog start* when a cell
  has no prior event.
- Observation dates are month starts; the target window is 30 days, so the
  dataset stops at 2026-10 (2026-11 would need data past the catalog end).

**Target prevalence:** 104,140 positives = **2.92%** (1 : 33.2). Rate by
split: train 2.83%, val 3.18%, test 3.07%. Positive rate by named region
spans **0.62% → 11.98%** — a 19× spread that makes per-region evaluation
mandatory rather than optional.

### Leakage defences

Three independent tests in `tests/test_leakage.py`:

1. **Perturbation** — inject an event after `t`; every feature at `t` must be
   byte-identical. Catches any code path that reaches into the future.
2. **Boundary exactness** — events placed at exact window edges verify the
   half-open convention (`event at t` → target, not feature; `event at t+30d`
   → excluded from the target window).
3. **Split integrity** — chronological ordering plus label-window purge,
   tested with deliberately awkward mid-month boundaries so the purge provably
   fires (with default year-end boundaries and month-start observations it
   never needs to: Dec 1 + 30d = Dec 31 < Jan 1, hence **0 rows purged**).

A separate test asserts the feature list can never contain `split`, `target`,
`cell_id`, or `obs_date`.

---

## 6. Modelling protocol

### Splits (chronological, never random)

| Split | Period | Rows |
|---|---|---|
| train | ≤ 2018-12-31 | 2,484,648 |
| val | 2019-01-01 → 2022-12-31 | 552,144 |
| test | 2023-01-01 → 2026-10-01 | 529,138 |

Random row splitting is forbidden: in a temporal catalog it places future rows
in training. Training touches **train only**; validation drives early stopping
and threshold selection; **the test split is scored exactly once**, at the
end, using the threshold frozen from validation.

### Models

| Model | Role |
|---|---|
| `baseline_global_rate` | constant training base rate — the floor |
| `baseline_climatology` | per-cell historical frequency with empirical-Bayes shrinkage toward the global rate — **the baseline of record** |
| `baseline_poisson` | log-linear Poisson rate → `P(≥1) = 1 − e^(−λ)` |
| `logreg` | standardized logistic regression |
| `random_forest` | `max_depth=16`, `min_samples_leaf=50`, 200 trees |
| `xgboost` | depth 6, early stopping on validation log-loss |

### Two measurements that changed the design

**Class weighting destroyed calibration.** With `class_weight='balanced'`
(and `scale_pos_weight` for XGBoost) the models ranked well — PR-AUC
0.30–0.36 — but predicted a mean probability of **0.28 against a 0.032 base
rate**, giving Brier Skill Score ≈ **−4** versus counting history. Balancing
forces the effective prior toward 50/50, so probabilities stop being
probabilities. Since calibration is a headline requirement of this project,
weighting is **disabled by default** (`models.use_class_weights`), imbalance
being handled where it belongs — threshold tuning on validation. The
measurement is recorded in `src/seismo/models/train.py` and `config.yaml`.

**An unconstrained RandomForest grew to depth 47 / 9M nodes / 724 MB**,
making SHAP traversal impractical (35+ minutes on an 8 GB machine). Capped at
`max_depth=16`, `min_samples_leaf=50` → depth 16, 172 MB, and evaluation
dropped to ~5 minutes.

---

## 7. Evaluation

**Metrics.** At 2.92% prevalence, accuracy is meaningless (predicting "never"
scores 97.08%), so discrimination leads with **PR-AUC**, supported by
ROC-AUC. Probabilistic quality uses **Brier Score** and — decisively —
**Brier Skill Score against the climatological reference**:

```
BSS = 1 − BS(model) / BS(reference)
```

`> 0` beats counting history, `= 0` adds nothing, `< 0` is harmful.
Calibration is measured with **ECE** and a reliability diagram.

**Thresholds** are tuned for F1 on validation only, then frozen. Test reuses
the stored threshold — never re-derives it.

### Test-split results (holdout, scored once)

| model | PR-AUC | ROC-AUC | Brier | **BSS** | ECE |
|---|---|---|---|---|---|
| baseline_global_rate | 0.031 | 0.500 | 0.0297 | −0.2261 | 0.0024 |
| baseline_poisson | 0.249 | 0.869 | 0.0292 | −0.2036 | 0.0030 |
| logreg | 0.282 | 0.870 | 0.0254 | −0.0491 | 0.0037 |
| baseline_climatology | 0.335 | 0.891 | 0.0242 | 0.0000 | 0.0040 |
| random_forest | 0.348 | 0.895 | 0.0239 | **+0.0158** | **0.0006** |
| **xgboost** | **0.352** | **0.899** | **0.0238** | **+0.0189** | 0.0010 |

**Interpretation — stated honestly.** Two of six models beat simply counting
history, and the margin is **small (+1.9% Brier skill)**. That is the
truthful shape of this problem: recent activity and location carry real but
limited signal for a 30-day, cell-level, M≥4.5 event. Test improved slightly
over validation (xgboost +0.0146 → +0.0189), so the chronological protocol
did not overfit. Logistic Regression and the Poisson baseline failed to beat
the climatological reference, and that is reported rather than buried.

Two regions show marginally negative skill for the production model —
Central America & Mexico (−0.003, n = 8,832) and Indonesia (−0.00003,
n = 27,648) — reinforcing that the pooled metric cannot stand in for
regional evaluation.

---

## 8. Explainability

SHAP (TreeExplainer) on a deterministic 2,000-row sample per split for global
feature importance, plus per-prediction explanations in the Forecasting page.

**Caveat, stated in-product:** SHAP explains *the model*, not causality.
Trailing-window features are correlated by construction (30d ⊆ 90d ⊆ 365d),
so attributions among them are not independent causal contributions.

---

## 9. Model promotion — a human gate

Training never writes `models/registry.json`. `seismo model promote` requires:

1. an evaluated model (`metrics.json` must exist — unevaluated models are
   refused), and
2. a non-empty `--reviewed-by` name, recorded in the promotion history.

The application loads **only** `load_production_model()`, which reads the
registry — an unpromoted experiment can never reach a user.

**Current production:** `xgboost` → `v1-20261117T132818Z`, reviewed by abdel.

---

## 10. Reproducibility

- Pinned dependencies (`requirements.txt`), documented Python version (3.12).
- Fixed seed (`config.yaml` → `seed: 42`) recorded in every training config.
- Every model artifact directory is self-describing: `model.joblib`,
  `features.json` (exact feature list and order), `metrics.json`,
  `training_config.yaml` (including data lineage), `data_manifest_ref.json`
  (raw collection parameters, month count, row total), `threshold.json`.
- Raw and processed data are gitignored; the repository ships reproducible
  fetch scripts and a committed sample dataset.
- `make notebooks` regenerates all executed notebooks from `.py` sources.

---

## 11. Known limitations

1. **Detection completeness** varies spatially and temporally; negative labels
   inherit that.
2. **Activity ≠ risk.** Exposure (WorldPop) and vulnerability (OSM) data are
   documented extensions, not modelled here.
3. **Global metrics hide regional behaviour** — 19× spread in base rate; the
   app surfaces per-region metrics alongside pooled ones.
4. **Skill is modest.** +1.9% Brier skill over counting history is real but
   small; the system is a demonstration of rigorous method, not an
   operational warning service.
5. **No real-time ingestion.** The catalog window is frozen for
   reproducibility; refreshing requires re-running the pipeline.
