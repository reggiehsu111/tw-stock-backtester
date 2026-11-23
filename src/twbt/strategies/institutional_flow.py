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
    # How the raw lot counts are made comparable across years. A parameter
    # rather than a hardcoded z-score, because the choice changes results more
    # than most people expect: on a skewed series a rank and a z-score can
    # correlate at 0.13. A signal that survives only one of them is telling you
    # something about itself.
    normalization: str = "zscore"

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
        norm = (lambda c: panel_mod.normalize(p[c], self.normalization,
                                              window=self.window))
        return norm(fo), norm(de)

    def signal(self, p: pd.DataFrame) -> pd.Series:
        got = self._inputs(p)
        if got is None:
            # No institutional columns in this panel — return flat rather than
            # inventing a signal. The engine reports "never took a position",
            # which is the honest outcome, not an error to be swallowed.
            return pd.Series(0.0, index=p.index)

        fz, dz = got
        entry, dealer = self.thresholds()
        if self.normalization in ("rank", "minmax"):
            # bounded in [0, 1], so "extreme" is distance from the middle
            lo_e, lo_d = 1.0 - entry, 1.0 - dealer
            long_ = (fz >= entry) & (dz <= lo_d)
            short = (fz <= lo_e) & (dz >= dealer)
        else:
            long_ = (fz >= entry) & (dz <= -dealer)
            short = (fz <= -entry) & (dz >= dealer)

        out = pd.Series(0.0, index=p.index)
        out[long_] = 1.0
        if self.allow_short:
            out[short] = -1.0
        return out

    def grid(self) -> list[dict[str, Any]]:
        return [{"entry_z": e, "dealer_z": d}
                for e, d in product((0.75, 1.0, 1.5), (0.0, 0.5))]

    def thresholds(self) -> tuple[float, float]:
        """Entry levels in the units the chosen normalisation produces.

        A rank lives in [0, 1] and a z-score does not, so a threshold of 1.0
        means "one standard deviation" under one and "impossible" under the
        other. Translating here keeps the same dial meaningful across methods.
        """
        if self.normalization in ("rank", "minmax"):
            hi = 0.5 + min(0.49, self.entry_z / 6.0)
            return hi, 0.5 + min(0.49, self.dealer_z / 6.0)
        return self.entry_z, self.dealer_z

    @property
    def n_variants(self) -> int:
        """How many settings were searched — feeds the noise bar."""
        return len(self.grid())

    def describe(self) -> str:
        entry, dealer = self.thresholds()
        return (f"Long when foreign positioning ({self.normalization}) "
                f">= {entry:+.2f} and dealer <= {-dealer:+.2f}"
                + (", short on the mirror condition" if self.allow_short else "")
                + f". Trailing window {self.window}d, traded with a "
                f"{self.publication_lag}-session publication lag.")
