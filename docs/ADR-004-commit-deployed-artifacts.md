# ADR-004: Commit the deployed data and the production model artifact

## Status
Accepted — supersedes the "artifacts are not committed" policy stated in
`README.md` and `docs/METHODOLOGY.md` §10 (both updated in the same change).

## Context

The pipeline regenerates every artifact from scripts in ~15 minutes, so the
original policy — commit code and a small sample, gitignore the bulk — was
correct for a local-only project and is standard practice.

Deployment to **Streamlit Community Cloud** breaks that assumption. A cloud
instance clones the repository and runs it; it cannot run `make data && make
train`, and it has no access to this machine's `data/` or `models/`
directories. Under the old policy the deployed app renders six pages of
"artifact missing — run `seismo data fetch`" panels, which is a failure mode
rather than a product.

Constraints measured before deciding:

| Artifact | File | In RAM | Under GitHub's 100 MB/file hard limit |
|---|---|---|---|
| processed catalog | 38.2 MB | 177 MB | yes |
| feature dataset | 42.3 MB | 292 MB | yes |
| production model (xgboost) | 3.0 MB | small | yes |
| random forest | 137.4 MB | large | yes, but unused at runtime |
| CSV export | 141 MB | — | **no — GitHub rejects it** |

Worst case with both datasets resident: ~470 MB data + ~150–200 MB Streamlit
≈ 650–670 MB, inside the ~1 GB free-tier envelope but with limited headroom.

## Decision

1. **Commit** `data/processed/earthquakes.parquet`,
   `data/features/regional_monthly_features.parquet`,
   `data/features/cell_activity.parquet`, and the production model directory
   `models/xgboost/`.
2. **Keep gitignored:** `data/raw/` (55 MB, re-fetchable and the least useful
   to a reviewer), `data/export/` (contains a 141 MB file GitHub would reject),
   `models/` for every non-production version (the random forest is 137 MB and
   the app never loads it — the registry points at xgboost), and `.venv/`.
3. **Leave the repo's reproducibility story intact:** the committed artifacts
   are *outputs of* the pipeline, not a substitute for it. `make data` /
   `make train` still reproduce them byte-for-byte from the documented
   manifest, and `data/samples/` remains for environments where even this is
   too large.

## Consequences

**Easier**
- A fresh clone and a cloud deployment both render the real application with
  the real 707,438-event catalog and the real reported metrics — no demo-mode
  fork of the data, no divergence between "what the README claims" and "what
  the app shows".
- Reviewers see the artifact whose lineage `data_manifest_ref.json` points at,
  so the reproducibility chain is inspectable end to end.

**Harder**
- Repository size grows to ~92 MB (clone time, fork cost).
- The policy is no longer "no data in git" — it is "the *deployed subset* in
  git, everything else regenerated". Future contributors must know which
  artifacts are in that subset; this ADR is the list.
- Community Cloud memory headroom is thin if a future page loads both large
  datasets plus a heavier model. If the instance restarts under memory
  pressure, the mitigation is column projection in `data_access` (parquet
  reads only the columns a page needs), **not** removing data from the repo.

**Rejected alternatives**
- *Demo/sample-only mode:* keeps the repo tiny but shows a reviewer a
  different dataset than the one every documented number was computed on —
  the exact inconsistency this project's methodology section exists to avoid.
- *GitHub Release assets + download on first run:* no repo growth, but adds
  download/verify logic, a cold-start dependency on GitHub availability, and a
  new failure mode (release missing or corrupt) for no gain while the files
  fit comfortably under the hard limit.
- *Commit everything including the random forest:* +137 MB for a model the
  production registry does not reference.
