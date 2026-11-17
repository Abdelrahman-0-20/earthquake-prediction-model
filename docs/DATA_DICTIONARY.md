# SeismoAI Data Dictionary

Artifacts produced by the pipeline, their schemas, and the conventions that
make them interpretable. Every derived column is defined here — no orphan
fields.

---

## 1. `data/raw/YYYY-MM.parquet`

Immutable USGS ComCat pulls, one file per calendar month, plus `manifest.json`.

| Column | Type | Notes |
|---|---|---|
| `time` | string (ISO-8601 UTC) | as returned by the API — deliberately unmodified |
| `updated` | string (ISO-8601 UTC) | catalog revision time |
| `latitude`, `longitude` | float | degrees |
| `depth` | float | km |
| `mag` | float | magnitude |
| `magType` | string | `ml`, `mw`, `mb`, … |
| `nst`, `gap`, `dmin`, `rms` | float | network-quality fields, frequently null |
| `id` | string | USGS event id (primary key) |
| `place` | string | e.g. `"3 km N of Cairo, Egypt"` |
| `net`, `type`, `status` | string | `type` distinguishes earthquakes from blasts |
| `horizontalError`, `depthError`, `magError`, `magNst` | float | uncertainties, frequently null |
| `locationSource`, `magSource` | string | contributing network |

**Manifest fields per month:** `status` (`ok`/`empty`/`failed`), `rows`,
`sha256`, `fetched_at`, `attempts`, `request` (exact query parameters).

---

## 2. `data/processed/earthquakes.parquet`

Cleaned catalog. **707,438 rows** (2000-01-01 → 2026-11-15).

Guarantees:
- unique `id` (latest catalog revision wins)
- coordinates within valid ranges; `time`/`updated` parse as UTC
- `type == "earthquake"` only (7,745 non-earthquake events removed)
- `year`, `month` derived for fast grouping

Adds: `year` (int), `month` (int).

Soft conditions that are **reported, not fixed** — see
`reports/data_quality/data_quality.json`:
- 4,359 negative depths (near-surface events; retained)
- 5 events below the M2.5 collection filter (magnitude revised downward after
  index time — the catalog is a living document)
- Missingness: `dmin` 51%, `horizontalError` 51%, `magError` 50%, `nst` 33%,
  `magNst` 30%, `depthError` 30%, `gap` 16.5%, `rms` 15.2%

---

## 3. `data/features/cell_activity.parquet`

Per-cell historical activity (Module 03 map layer). **11,503 rows** — one per
1° cell that recorded ≥1 event.

| Column | Type | Meaning |
|---|---|---|
| `cell_id` | int64 | `j * n_i + i` on the configured grid |
| `n_events` | int | total catalog events in the cell |
| `mean_mag`, `max_mag` | float | magnitude statistics |
| `mean_depth` | float | km |
| `last_event_time` | datetime64[us, UTC] | most recent event |
| `days_since_last_event` | float | recency at catalog end |
| `events_per_year` | float | `n_events / catalog_span_years` |
| `latitude_center`, `longitude_center` | float | cell centre, directly plottable |

> `events_per_year` is a **historical activity rate**, not hazard or risk.
> Detection coverage varies by region and era.

---

## 4. `data/features/regional_monthly_features.parquet`

The Module 02 ML dataset. **3,565,930 rows = 11,503 cells × 310 observation
dates** (2001-01-01 → 2026-10-01, month starts).

### Temporal contract

For observation date `t`:

| | Window | Rationale |
|---|---|---|
| **Features** | `[t − W, t)` | strictly before `t` — available at prediction time |
| **Target** | `[t, t + 30d)` | strictly after `t` |

The windows are disjoint with no overlap. Enforced by
`tests/test_leakage.py`, including a perturbation test that injects a future
event and asserts every earlier feature is byte-identical.

### Columns

| Column | Type | Range / notes |
|---|---|---|
| `obs_date` | datetime64[us, UTC] | month-start observation date |
| `cell_id` | int64 | grid cell |
| `latitude_center`, `longitude_center` | float | cell centre |
| `events_last_30d` | int32 | events in `[t−30d, t)` — 92.8% zeros |
| `events_last_90d` | int32 | events in `[t−90d, t)` — 85.7% zeros |
| `events_last_365d` | int32 | events in `[t−365d, t)` — 70.5% zeros |
| `mean_magnitude_90d` | float | mean magnitude in `[t−90d, t)` |
| `max_magnitude_90d` | float | max magnitude in `[t−90d, t)`, peaks at 9.1 |
| `mean_depth_90d` | float | km; can be negative (near-surface events) |
| `events_ge45_last_365d` | int32 | trailing count of *target-class* (M≥4.5) events |
| `days_since_last_event` | float | days to most recent prior event |
| `target` | int8 | **1** if ≥1 event with `mag ≥ 4.5` in `[t, t+30d)` |

### Conventions (important when modelling)

- **No NaNs anywhere.** Statistic windows with no events are filled with
  `0.0`. This is unambiguous for magnitude (catalog minimum is 0.7) and
  contributes no signal for linear models; the count features disambiguate
  "no events" from "events with a low value".
- **`days_since_last_event`** falls back to *days since catalog start* when a
  cell has no prior event (max observed ≈ 9,770 days), so it never goes
  negative and never needs imputation.
- **Class balance:** 104,140 positives = **2.92%** overall
  (train 2.83% · val 3.18% · test 3.07%). Use PR-AUC, not accuracy.
- **Zero rows purged** with default config: Dec 1 + 30d = Dec 31 < Jan 1, so
  no label window crosses a year boundary. The purge logic is tested with
  deliberately awkward mid-month boundaries in `tests/test_leakage.py`.

### Split assignment (`splits` in `config.yaml`)

| Split | Period | Rows |
|---|---|---|
| train | ≤ 2018-12-31 | 2,484,648 |
| val | 2019-01-01 → 2022-12-31 | 552,144 |
| test | ≥ 2023-01-01 | 529,138 |

Strictly chronological — random row splits are forbidden because they leak
future information in a temporal catalog.

---

## 5. Model artifacts (`models/<name>/<version>/`) — Week 3

Each training run writes: fitted pipeline (`model.joblib`), `features.json`
(exact feature list and order), `metrics.json`, `training_config.yaml`,
`data_manifest_ref` (data lineage). The app consumes only `models/registry.json`.
