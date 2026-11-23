"""On-disk parquet cache for fetched data.

The cache exists so a reviewer can run the project repeatedly without hammering a
free public service, and so a flaky network does not lose an hour of downloads.

It is deliberately dumb: one parquet per dataset, keyed by date. ``update()`` asks
what date range is already present and fetches only what is missing, so re-running
``twbt fetch`` after a week pulls a week.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .sources import equities, taifex

# datasets that come from the equity cross-section rather than TAIFEX
EQUITY_DATASETS = ("equities",)

DEFAULT_DIR = Path(os.environ.get("TWBT_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache"))
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixture"


def _path(name: str, root: Path) -> Path:
    return root / f"{name}.parquet"


def load(name: str, root: Path | None = None, fixture: bool = False) -> pd.DataFrame:
    """Read one dataset. ``fixture=True`` reads the committed offline sample."""
    root = FIXTURE_DIR if fixture else (root or DEFAULT_DIR)
    p = _path(name, root)
    if not p.exists():
        where = "fixture" if fixture else "cache"
        raise FileNotFoundError(
            f"{name} not in {where} ({p}). "
            + ("The fixture should be committed — is the clone complete?"
               if fixture else "Run `twbt fetch` first, or pass --fixture.")
        )
    df = pd.read_parquet(p)
    return df.sort_values("date").reset_index(drop=True)


def covered(name: str, root: Path | None = None) -> tuple[date, date] | None:
    """The [first, last] date already cached, or None if the dataset is absent."""
    root = root or DEFAULT_DIR
    p = _path(name, root)
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p, columns=["date"])["date"]
    except Exception:                                  # noqa: BLE001
        return None
    if d.empty:
        return None
    return d.min().date(), d.max().date()


def update(name: str, start: date, end: date, root: Path | None = None,
           verbose: bool = True) -> pd.DataFrame:
    """Fetch whatever part of [start, end] is missing and merge it into the cache.

    Only the two ends are considered, not interior gaps: TAIFEX returns contiguous
    ranges, so a hole in the middle means a failed run rather than missing data,
    and silently patching it would hide that. Delete the parquet to force a refetch.
    """
    root = root or DEFAULT_DIR
    root.mkdir(parents=True, exist_ok=True)
    have = covered(name, root)
    todo: list[tuple[date, date]] = []
    if have is None:
        todo.append((start, end))
    else:
        lo, hi = have
        if start < lo:
            todo.append((start, lo - timedelta(days=1)))
        if end > hi:
            todo.append((hi + timedelta(days=1), end))
        if not todo and verbose:
            print(f"  {name}: cached {lo}..{hi}, nothing to fetch")

    frames = []
    if have is not None:
        frames.append(load(name, root))
    for a, b in todo:
        if verbose:
            print(f"  {name}: fetching {a}..{b}")
        if name in EQUITY_DATASETS:
            # write partial progress as it arrives: this fetch runs for minutes,
            # and losing all of it to one interruption is not acceptable
            def _save(partial, _frames=frames, _name=name, _root=root):
                try:
                    combined = (pd.concat(_frames + [partial], ignore_index=True)
                                .drop_duplicates(subset=["date", "ticker"])
                                .sort_values("date").reset_index(drop=True))
                    combined.to_parquet(_path(_name, _root), index=False)
                except Exception:                      # noqa: BLE001
                    pass                               # a failed checkpoint is not fatal

            got = equities.fetch_range(a, b, verbose=verbose, checkpoint=_save)
        else:
            got = taifex.fetch_range(name, a, b, verbose=verbose)
        if verbose:
            print(f"  {name}: +{len(got):,} rows")
        if len(got):
            frames.append(got)

    if not frames:
        return pd.DataFrame(columns=["date"])
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates()
           .sort_values("date")
           .reset_index(drop=True))
    out.to_parquet(_path(name, root), index=False)
    return out
