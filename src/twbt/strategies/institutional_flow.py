"""Trade the next session against dealer positioning and with foreign flow.

The premise, which is ordinary and widely known: in Taiwan index futures, the
foreign institutional net position (外資及陸資) behaves like directional
positioning, while the dealer net position (自營商) is substantially hedging flow
from the warrant book and tends to lean against the market. So: long when foreign
open interest is building and dealers are short, flat otherwise.

Three things about this implementation are worth more than the premise.

PUBLICATION LAG. TAIFEX releases these figures at 15:00 TPE, after the day session
has closed. The earliest tradable session is the next one. The strategy declares
``published_columns`` and the engine enforces the lag; running it with lag 0 is
refused rather than merely discouraged. The difference is not cosmetic — with lag 0
this strategy looks dramatically better, which is exactly why it has to be
impossible to run that way by accident.

NORMALISATION. Raw lot counts are not comparable across years: open interest has
grown several-fold and a "large" position in 2020 is unremarkable now. The signal
uses trailing z-scores from ``twbt.panel``, so the threshold on any given day comes
only from prior days.

THRESHOLDS. The grid below is small on purpose. Six combinations is six tests, and
the noise bar for six tests is already around 2.6 — a result from this sweep needs
to clear that, not 1.96, and the tearsheet reports it that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .. import panel as panel_mod
from ..backtest import Strategy

PUBLISHED = ("foreign_oi_net_lots", "dealer_oi_net_lots",
             "foreign_net_lots", "dealer_net_lots")


@dataclass(frozen=True)
class InstitutionalFlow(Strategy):
    """Long when foreign positioning builds and dealers are leaning short."""

    name: str = "institutional_flow"
    publication_lag: int = 1
    published_columns: tuple[str, ...] = PUBLISHED

    entry_z: float = 1.0          # how extreme foreign positioning must be
    dealer_z: float = 0.0         # dealers must be at least this short (z units)
    window: int = 250
    allow_short: bool = True

    def _inputs(self, p: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
        """Trailing z-scores of the two positioning series, or None if absent.

        Prefers open-interest net (a stock: where positioning *is*) over traded net
        (a flow: what changed today), because the premise is about positioning.
        """
        fo = next((c for c in ("foreign_oi_net_lots", "foreign_net_lots")
                   if c in p.columns and p[c].notna().any()), None)
        de = next((c for c in ("dealer_oi_net_lots", "dealer_net_lots")
                   if c in p.columns and p[c].notna().any()), None)
        if fo is None or de is None:
            return None
        return (panel_mod.zscore(p[fo], window=self.window),
                panel_mod.zscore(p[de], window=self.window))

    def signal(self, p: pd.DataFrame) -> pd.Series:
        got = self._inputs(p)
        if got is None:
            # No institutional columns in this panel — return flat rather than
            # inventing a signal. The engine reports "never took a position",
            # which is the honest outcome, not an error to be swallowed.
            return pd.Series(0.0, index=p.index)

        fz, dz = got
        long_ = (fz >= self.entry_z) & (dz <= -self.dealer_z)
        out = pd.Series(0.0, index=p.index)
        out[long_] = 1.0
        if self.allow_short:
            short = (fz <= -self.entry_z) & (dz >= self.dealer_z)
            out[short] = -1.0
        return out

    def grid(self) -> list[dict[str, Any]]:
        return [{"entry_z": e, "dealer_z": d}
                for e, d in product((0.75, 1.0, 1.5), (0.0, 0.5))]

    @property
    def n_variants(self) -> int:
        """How many settings were searched — feeds the noise bar."""
        return len(self.grid())

    def describe(self) -> str:
        return (f"Long when foreign positioning z >= {self.entry_z:+.2f} and dealer "
                f"z <= {-self.dealer_z:+.2f}"
                + (", short on the mirror condition" if self.allow_short else "")
                + f". Trailing window {self.window}d, traded with a "
                f"{self.publication_lag}-session publication lag.")
