# %% [markdown]
# # 02 — Data cleaning: every decision, stated and defended
#
# **Purpose.** The processed catalog (`data/processed/earthquakes.parquet`) is the input
# to every later stage of this project, so each transformation applied to it has to be
# explicit, justified, and reproducible. This notebook restates those decisions against
# the data itself rather than against a spec.
#
# **Project rule (repeated here because it is load-bearing):**
# *missingness is measured, never blindly filled.* See section 2.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

# The build tool executes the kernel with cwd = notebooks/; move to the repository root
# so every relative path below follows the documented repo-root convention.
if not Path("data").exists():
    os.chdir(Path.cwd().parent)
assert Path("data/processed/earthquakes.parquet").exists(), "run from the repository root"

NAVY = "#14263D"
ACCENT = "#2583A6"
LIGHT = "#E9EEF3"

plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.grid": True,
        "grid.color": LIGHT,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> Path:
    """Write reports/figures/<name>, embed it in the notebook, free the memory."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    display(Image(filename=str(path)))
    return path


df = pd.read_parquet("data/processed/earthquakes.parquet")
print(f"processed shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %% [markdown]
# ## 1. Event-type filtering: only earthquakes are kept
#
# The raw store mixes seismic **earthquakes** with every other event type ComCat
# indexes: explosions, mining blasts, quarry blasts, ice quakes, landslides, nuclear
# tests, sonic booms. Counting `type` across all 323 raw files (715,183 rows) shows
# exactly what was dropped.

# %%
raw_files = sorted(Path("data/raw").glob("*.parquet"))
raw_types = pd.concat(
    (pd.read_parquet(p, columns=["type"]) for p in raw_files),
    ignore_index=True,
)["type"]
print(f"raw rows read: {len(raw_types):,} from {len(raw_files)} files")

type_counts = raw_types.value_counts()
non_earthquake = type_counts.drop(labels=["earthquake"])
print(f"\nnon-earthquake events removed: {non_earthquake.sum():,}")
print(f"distinct non-earthquake types: {len(non_earthquake)}")
print(non_earthquake.to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 6))
plot_counts = non_earthquake.sort_values()
ax.barh(plot_counts.index, plot_counts.values, color=ACCENT, edgecolor=NAVY, linewidth=0.4)
ax.set_xscale("log")
ax.set_title("7,745 non-earthquake events removed: explosions and mining blasts dominate")
ax.set_xlabel("Event count (log scale)")
ax.set_ylabel("Event type")
save(fig, "eda_event_types_removed.png")

kept = df["type"].value_counts()
print("processed type breakdown:")
print(kept.to_string())
assert set(kept.index) == {"earthquake"}, "processed data must contain only earthquakes"

# %% [markdown]
# **Decision.** Keep `type == "earthquake"` only (707,438 rows), drop the other 19
# types (7,745 rows, 1.08% of raw). A quarry blast and a magnitude-4 earthquake can look
# identical to a magnitude scale, but they are not the same population: mixing them
# would inflate regional counts (e.g. `mining explosion`, 2,809 events) and distort the
# frequency-magnitude statistics that notebook 03 depends on.

# %% [markdown]
# ## 2. Missingness: measured, never blindly filled
#
# Quality fields are the most heavily missing columns in the catalog. The chart below is
# computed from the processed data — it is a measurement, not an assumption.

# %%
missing = df.isna().sum().to_frame("missing_rows")
missing["pct"] = 100 * missing["missing_rows"] / len(df)
missing = missing.sort_values("pct", ascending=False)
print(missing.round(3).to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(missing.index[::-1], missing["pct"].values[::-1], color=ACCENT,
        edgecolor=NAVY, linewidth=0.4)
ax.set_title("Quality fields are the only missing columns: dmin 51.3%, nst 32.9%")
ax.set_xlabel("Missing values (% of 707,438 rows)")
ax.set_ylabel("Column")
save(fig, "eda_missingness.png")

# %% [markdown]
# ### The rule: missingness is measured, never blindly filled
#
# **No column in this project is mean-, median- or zero-imputed.** The reason is that
# the missing values are *informative about the observation process*, not about the
# earthquake:
#
# * `nst` (number of reporting stations, 32.9% missing), `gap` (azimuthal gap, 16.5%),
#   `dmin` (nearest-station distance, 51.3%) and `rms` (15.2%) are **network-dependent**.
#   They exist only when enough stations reported to the contributing network. An empty
#   `dmin` does not mean "stations were far away"; it usually means the solution was
#   produced without one being recorded.
# * Filling these with a constant would fabricate instrument information: every imputed
#   row would claim, falsely, that a specific network geometry was observed. Downstream
#   models would then learn from invented metadata instead of from data.
# * The same applies to `horizontalError` (50.6% missing), `magError` (50.4%) and
#   `depthError` (29.5%): they are reported for some solution types and not others, so
#   their absence is a property of the *solution*, not of the event.
#
# The defensible alternatives are (a) leave as `NaN` and let each analysis decide what to
# do with them, or (b) restrict to the subset where the field exists — both keep the
# provenance honest. This project does (a), and reports coverage whenever a quality
# field is used.
#
# Core event fields (`time`, `latitude`, `longitude`, `depth`, `mag`, `id`) have **0**
# missing rows, so no imputation is needed for the scientific variables themselves.

# %%
core = ["time", "latitude", "longitude", "depth", "mag", "id", "updated", "year", "month"]
core_missing = df[core].isna().sum()
print("missing rows in core scientific columns:")
print(core_missing.to_string())
print(f"\ntotal core missing: {int(core_missing.sum())}")

# %% [markdown]
# ## 3. Negative depths: 4,359 events kept on purpose
#
# `depth` is in kilometres below sea level, so a small negative value means the
# hypocentre is *above* sea level — a near-surface event. These are real, reviewed
# locations for shallow volcanic, mining and quarry-related seismicity recorded by
# networks that report elevations relative to sea level rather than to a local datum.

# %%
neg = df.loc[df["depth"] < 0, "depth"]
print(f"negative depths: {len(neg):,} rows ({100 * len(neg) / len(df):.2f}%)")
print(f"range: {neg.min():.2f} km … {neg.max():.2f} km, median {neg.median():.2f} km")
print(f"rows with depth exactly 0: {(df['depth'] == 0).sum():,}")
print("\nshallowest 5 events:")
cols = ["time", "depth", "mag", "net", "place"]
print(df.nsmallest(5, "depth")[cols].to_string(index=False))

# %% [markdown]
# **Decision: retain, do not delete.** Dropping them would (a) remove 4,359 genuine
# events — 0.62% of the catalog — biased towards volcanically and industrially active
# regions, and (b) silently redefine `depth` as "depth below sea level, except when we
# do not like the answer". Nothing downstream needs strictly positive depths: the depth
# analyses in notebook 03 bin them explicitly and label the negative bin.

# %% [markdown]
# ## 4. The five events below the M2.5 threshold: the catalog is a living document
#
# The query asked for `minmagnitude = 2.5`, yet the processed magnitude range starts at
# **0.7**. Five events sit below the threshold. They are not a filtering bug — they are
# evidence that ComCat records are *revised after they are indexed*.

# %%
below = df.loc[df["mag"] < 2.5, ["time", "updated", "mag", "magType", "net", "id", "place"]]
print(f"events with mag < 2.5: {len(below)}")
print(below.to_string(index=False))

# %%
revised = (below["updated"] - below["time"]).dt.total_seconds() / (365.25 * 24 * 3600)
print("years between event and its last catalog revision:")
print(revised.round(1).to_string(index=False))

# %% [markdown]
# **Explanation.** The web-service filter is evaluated against the magnitude stored in
# the event *index* at query time, while the `mag` field returned for an event is its
# current **preferred** magnitude. When a reviewer re-solves an event after it has been
# indexed and the revised magnitude comes back below 2.5, the event still passes the
# M2.5 filter but is delivered with `mag < 2.5`. All five are `reviewed` solutions whose
# `updated` timestamps postdate the event — e.g. `us20008ia2` (Central Alaska) was
# indexed above threshold and later revised down to M0.70.
#
# Consequences for this project:
#
# * the collection threshold is a **query-time** property, not a guarantee about every
#   row — so completeness statements must be made from the observed distribution
#   (notebook 03), not from the request parameters;
# * the catalog is a living document: any future re-run may return slightly different
#   rows for the same query, which is exactly why the manifest records per-month
#   sha256 checksums (notebook 01).

# %% [markdown]
# ## 5. Duplicate handling: unique ids, latest revision wins
#
# The ComCat event `id` is the natural primary key. The cleaning rule was:
# *if the same `id` appears more than once, keep the row with the latest `updated`
# timestamp* (the most recent catalog revision). The check below confirms the rule's
# effect on this run.

# %%
n_dup_ids = int(df["id"].duplicated().sum())
print(f"rows: {len(df):,}")
print(f"distinct ids: {df['id'].nunique():,}")
print(f"duplicate ids remaining: {n_dup_ids}")

# Apply the documented rule explicitly; with zero duplicates it must be a no-op.
deduped = (
    df.sort_values("updated").drop_duplicates(subset="id", keep="last").sort_index()
)
print(f"rows after applying keep-latest-updated rule: {len(deduped):,}")
assert len(deduped) == len(df), "keep-latest-updated rule must not change row count"

# Near-duplicates on time+location are a different (and much rarer) phenomenon.
key = (
    df["time"].astype("int64").astype(str)
    + "|"
    + df["latitude"].round(3).astype(str)
    + "|"
    + df["longitude"].round(3).astype(str)
)
print(f"rows sharing identical (time, lat, lon): {int(key.duplicated(keep=False).sum())}")
print("  (distinct networks can catalogue the same quake under different ids)")

# %% [markdown]
# ## 6. What the processed dataset guarantees
#
# After the cleaning stage, `data/processed/earthquakes.parquet` (707,438 rows × 24
# columns) satisfies all of the following — each claim is checked above or in
# `reports/data_quality/data_quality.json` (which records zero hard violations):
#
# 1. **Earthquakes only** — every row has `type == "earthquake"`; 7,745 non-earthquake
#    events were removed from the 715,183 raw rows.
# 2. **One row per event** — `id` is unique (0 duplicates); where an id did repeat, the
#    latest `updated` revision would be kept (rule verified as a no-op here).
# 3. **Bounded coverage** — event times from 2000-01-01 to 2026-11-15, magnitudes 0.7 to
#    9.1, depths −3.74 km to 735.8 km; the 5 sub-threshold magnitudes are documented
#    revisions, not filter failures.
# 4. **Honest missingness** — quality fields (`dmin` 51.3%, `nst` 32.9%, `gap` 16.5%,
#    `rms` 15.2%, …) are left as `NaN`; nothing is imputed.
# 5. **Negative depths retained** — 4,359 near-surface events kept and labelled, not
#    deleted.
# 6. **Derived keys present** — `year` and `month` are precomputed for temporal
#    analysis; `latitude`/`longitude`/`depth`/`mag`/`time` have zero missing values.
#
# What it does *not* guarantee: uniform detection. Completeness varies by region and by
# era, and that caveat governs every interpretation in notebook 03.
