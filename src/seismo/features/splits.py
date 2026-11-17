"""Chronological train/validation/test splitting with label-window purge.

Why not random splits?
----------------------
Random row splitting leaks future information in a temporal catalog: a row
observed in 2024 would sit in "train" while its neighbour from 2009 sits in
"test", and the model learns from the future. Splits here are strictly
ordered in time and enforced by ``tests/test_leakage.py``.

The purge rule
--------------
Each row's **label window** is ``[t, t + horizon)``. If that window crosses
the boundary of the row's split, the row's label was derived from events that
belong to the *next* period — so the row is dropped rather than allowed to
smuggle future information across the boundary.

Because observation dates are month starts and the horizon is 30 days, most
rows never cross a year-end boundary (Dec 1 + 30d = Dec 31 < Jan 1). The
purge is therefore a safety property that rarely fires with default
config — and it is tested with deliberately awkward boundaries to prove the
logic actually works.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ..errors import ValidationError

SPLIT_ORDER = ("train", "val", "test")


def assign_split(
    obs_date: pd.Series,
    *,
    train_end: date | str | pd.Timestamp,
    val_end: date | str | pd.Timestamp,
) -> pd.Series:
    """Label each observation date train / val / test by chronology."""
    train_end = pd.Timestamp(train_end, tz="UTC")
    val_end = pd.Timestamp(val_end, tz="UTC")
    if val_end <= train_end:
        raise ValidationError(
            f"val_end ({val_end.date()}) must be after train_end ({train_end.date()})",
            hint="Set splits.train_end and splits.val_end in config/config.yaml.",
        )
    return pd.Series(
        pd.cut(
            obs_date,
            bins=[pd.Timestamp.min.tz_localize("UTC"), train_end, val_end,
                  pd.Timestamp.max.tz_localize("UTC")],
            labels=["train", "val", "test"],
        ),
        index=obs_date.index,
    )


def purge_mask(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    train_end: date | str | pd.Timestamp,
    val_end: date | str | pd.Timestamp,
) -> pd.Series:
    """True where a row's label window stays inside its own split.

    The test split has no following boundary, so test rows are always kept.
    """
    train_end = pd.Timestamp(train_end, tz="UTC")
    val_end = pd.Timestamp(val_end, tz="UTC")

    # Each period's exclusive upper bound (the instant the next period starts).
    train_exclusive_end = train_end.normalize() + pd.Timedelta(days=1)
    val_exclusive_end = val_end.normalize() + pd.Timedelta(days=1)

    label_end = frame["obs_date"] + pd.Timedelta(days=horizon_days)
    split = frame["split"]

    keep = pd.Series(True, index=frame.index)
    train_rows = split == "train"
    val_rows = split == "val"
    keep[train_rows] = label_end[train_rows] <= train_exclusive_end
    keep[val_rows] = label_end[val_rows] <= val_exclusive_end
    return keep


def chronological_split(
    frame: pd.DataFrame,
    cfg: Any,
    *,
    horizon_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add a ``split`` column and drop label windows that cross boundaries.

    Returns ``(frame_with_split, stats)`` where ``stats`` records how many rows
    landed in each split and how many were purged.
    """
    if "obs_date" not in frame.columns:
        raise ValidationError(
            "chronological_split requires an 'obs_date' column",
            hint="Build features first with `seismo features build`.",
        )

    splits_cfg = cfg.section("splits")
    train_end = splits_cfg.get("train_end")
    val_end = splits_cfg.get("val_end")
    if not train_end or not val_end:
        raise ValidationError(
            "splits.train_end and splits.val_end must be set in config/config.yaml",
            hint="Add e.g. train_end: '2018-12-31' and val_end: '2022-12-31'.",
        )
    if horizon_days is None:
        horizon_days = int(cfg.section("forecast").get("horizon_days", 30))

    out = frame.copy()
    out["split"] = assign_split(out["obs_date"], train_end=train_end, val_end=val_end)

    keep = purge_mask(out, horizon_days=horizon_days,
                      train_end=train_end, val_end=val_end)
    purged = int((~keep).sum())
    out = out[keep].reset_index(drop=True)

    stats = {
        "train": int((out["split"] == "train").sum()),
        "val": int((out["split"] == "val").sum()),
        "test": int((out["split"] == "test").sum()),
        "purged": purged,
        "total": int(len(out)),
    }

    # Chronology is a hard invariant, not a hope.
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        a = out.loc[out["split"] == earlier, "obs_date"]
        b = out.loc[out["split"] == later, "obs_date"]
        if not a.empty and not b.empty and a.max() >= b.min():
            raise ValidationError(
                f"Split integrity violated: {earlier} ends at {a.max()} but "
                f"{later} starts at {b.min()}",
                hint="This indicates a bug in assign_split — report it; "
                     "do not ship model results.",
            )
    return out, stats
