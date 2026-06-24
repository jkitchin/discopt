"""The fixed performance panel (perf plan Stage 0).

A small, version-controlled set of **vendored** MINLPLib instances spanning the
measured cost centers (see ``docs/dev/performance-plan.md`` §1), with their
BARON-confirmed optima (``minlplib.solu``). Small enough to run in a few minutes;
representative enough to catch a regression in any cost center.

All current panel instances minimize, so the soundness check is one-directional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Vendored instance directory (self-contained — no external benchmark checkout).
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "python",
    "tests",
    "data",
    "minlplib",
)


@dataclass(frozen=True)
class PanelInstance:
    name: str
    time_limit: float
    oracle: float  # known global optimum (minlplib.solu)
    sense: str  # "min" | "max"
    cost_center: str  # which plan cost center it exercises

    @property
    def path(self) -> str:
        return os.path.join(DATA_DIR, f"{self.name}.nl")


# Budgets chosen so the certifying instances close (deterministic node_count) and
# the flagships expose the compile / throughput signal, with the whole panel
# running in ~4–5 min under the compile-logging instrumentation.
PANEL: list[PanelInstance] = [
    # Fast-certifying — deterministic node_count, the hard search-efficiency gate.
    PanelInstance("nvs03", 10, 16.0, "min", "CC3 small certify"),
    PanelInstance("nvs06", 10, 1.7703125, "min", "CC3 small certify"),
    PanelInstance("ex1226", 10, -17.0, "min", "CC3 small certify"),
    PanelInstance("nvs12", 20, -481.2, "min", "CC3 + #293 simplex-deadline"),
    PanelInstance("nvs17", 30, -1100.4, "min", "CC3 integer-product family"),
    PanelInstance("nvs22", 20, 6.05822, "min", "#277 cert soundness"),
    PanelInstance("st_ph10", 20, -10.5, "min", "#306 cert soundness"),
    # Flagships — compile-count / python-orchestration / throughput signal.
    PanelInstance("gear4", 60, 1.643428474, "min", "CC1/CC2/CC3 (#309, Stage-1 win)"),
    PanelInstance("ex1252", 30, 128893.741, "min", "CC5 expensive relaxation compiles"),
    PanelInstance("carton7", 25, 191.7295481, "min", "#288/#289 large bounded soundness"),
]
