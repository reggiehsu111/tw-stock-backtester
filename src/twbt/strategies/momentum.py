"""A plain trailing-return strategy, included as a reference point.

This is *supposed* to be unremarkable. Its job is to sit next to the other strategy
in the tearsheet and show what a result with no particular edge looks like once
costs and the honest statistics are applied — so that when something does look
good, the reader has a calibration point rather than only a claim.

It also demonstrates the overlap problem concretely. With ``lookback=20`` and a
hold of 20 sessions, consecutive observations share 19/20 of their window, and the
tearsheet's OLS and disjoint t-statistics will visibly disagree. That disagreement
is the point of displaying both.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .. import panel as panel_mod
from ..backtest import Strategy


@dataclass(frozen=True)
class Momentum(Strategy):
    """Long after a strong trailing return, short after a weak one."""

    name: str = "momentum"
    # Price is observable at the close, so in principle this could trade the same
    # session. It does not: a close-to-close return cannot be acted on at that same
    # close. One session is the minimum honest lag for a close-based signal.
    publication_lag: int = 1
    published_columns: tuple[str, ...] = ()

    lookback: int = 20
    hold: int = 20
    entry_z: float = 1.0
    vol_scale: bool = False

    def signal(self, p: pd.DataFrame) -> pd.Series:
        if "close" not in p.columns:
            return pd.Series(0.0, index=p.index)

        trail = p["close"].pct_change(self.lookback) * 1e4
        z = panel_mod.zscore(trail, window=250)

        raw = pd.Series(0.0, index=p.index)
        raw[z >= self.entry_z] = 1.0
        raw[z <= -self.entry_z] = -1.0

        # hold the position for `hold` sessions after it fires, rather than
        # re-deciding daily; forward-fill is bounded so a single signal cannot
        # silently become a permanent position
        held = raw.replace(0.0, np.nan).ffill(limit=max(self.hold - 1, 0)).fillna(0.0)

        if self.vol_scale:
            # Inverse-vol sizing is offered but OFF by default. It is widely assumed
            # to help and in the research behind this project it hurt — Sharpe fell
            # from 0.57 to 0.36 on the signal it was tried on. Left switchable so the
            # tearsheet can show that rather than assert it.
            v = p.get("vol_20")
            if v is not None:
                scale = (v.median() / v).clip(upper=3.0)
                held = held * scale.fillna(1.0)

        return held

    @property
    def overlap(self) -> int:
        """Observations sharing data, for the overlap-aware regression."""
        return max(self.hold, 1)

    def grid(self) -> list[dict[str, Any]]:
        return [{"lookback": lb, "entry_z": z}
                for lb, z in product((10, 20, 60), (0.5, 1.0, 1.5))]

    @property
    def n_variants(self) -> int:
        return len(self.grid())

    def describe(self) -> str:
        return (f"Long when the trailing {self.lookback}-session return has z >= "
                f"{self.entry_z:+.2f}, short on the mirror, held {self.hold} sessions"
                + (", inverse-vol sized" if self.vol_scale else "")
                + f". Traded with a {self.publication_lag}-session lag.")
