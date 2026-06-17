"""QCQP relaxation-strength scoreboard (Wave-2 W1).

Measures the levers that relaxation-strengthening cuts (PSD / eigenvalue, SOC)
are meant to move, on the nonconvex QCQP instances registered in
``benchmarks.problems.qcqp_problems``:

* **root_gap** — ``(known_opt - root_bound) / max(|known_opt|, 1)``, the dual-bound
  gap *at the root node* (solve capped to one node). This is the direct measure
  of relaxation strength: tighter cuts shrink it.
* **node_count** / **wall_time** — search effort to prove global optimality.
* **incorrect** — the solved objective disagrees with the known optimum (the
  correctness gate; must stay 0 for every configuration).

The harness takes a dict of named solver configurations, so a cut family can be
A/B'd against the baseline simply::

    run_scoreboard(level="smoke", configs={"baseline": {}, "psd": {"psd_cuts": True}})

It is deliberately solver-config-agnostic: W2 just adds a config entry.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from benchmarks.problems.base import get_problems

if TYPE_CHECKING:
    from collections.abc import Callable

# Known-optima comparison tolerance (matches the benchmark correctness gate).
_ABS_TOL = 1e-4
_REL_TOL = 1e-4


@dataclass
class ScoreRow:
    instance: str
    config: str
    status: str
    objective: float | None
    known_optimum: float
    root_bound: float | None
    root_gap: float | None
    node_count: int
    wall_time: float
    incorrect: bool


@dataclass
class Scoreboard:
    rows: list[ScoreRow] = field(default_factory=list)

    @property
    def incorrect_count(self) -> int:
        return sum(1 for r in self.rows if r.incorrect)

    def by_config(self, config: str) -> list[ScoreRow]:
        return [r for r in self.rows if r.config == config]

    def format_table(self) -> str:
        header = (
            f"{'instance':<20} {'config':<10} {'status':<10} "
            f"{'obj':>12} {'known':>12} {'root_gap':>10} {'nodes':>7} {'time_s':>8} {'bad':>4}"
        )
        lines = [header, "-" * len(header)]
        for r in self.rows:
            obj = f"{r.objective:.4f}" if r.objective is not None else "—"
            rg = f"{r.root_gap:.4f}" if r.root_gap is not None else "—"
            lines.append(
                f"{r.instance:<20} {r.config:<10} {r.status:<10} "
                f"{obj:>12} {r.known_optimum:>12.4f} {rg:>10} "
                f"{r.node_count:>7} {r.wall_time:>8.2f} {'X' if r.incorrect else '':>4}"
            )
        return "\n".join(lines)


def _is_incorrect(status: str, objective: float | None, known: float) -> bool:
    """A configuration is incorrect if it claims optimality but the objective
    disagrees with the known optimum (or is missing)."""
    if status != "optimal":
        return False
    if objective is None:
        return True
    return abs(objective - known) > _ABS_TOL + _REL_TOL * abs(known)


def _finite(x) -> float | None:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def score_instance(build_fn: Callable, known_optimum: float, config: dict) -> ScoreRow:
    """Solve one instance under one configuration and return its scoreboard row.

    Solves twice from fresh models: once to optimality (objective / nodes / time)
    and once capped to a single node (the root dual bound), so the relaxation gap
    can be read off directly.
    """
    # Full solve.
    model = build_fn()
    t0 = time.perf_counter()
    full = model.solve(**config)
    wall = time.perf_counter() - t0
    objective = _finite(full.objective)
    status = str(full.status)
    node_count = int(getattr(full, "node_count", 0) or 0)

    # Root-only solve for the root dual bound.
    root_model = build_fn()
    root_kwargs = dict(config)
    root_kwargs["max_nodes"] = 1
    root = root_model.solve(**root_kwargs)
    root_bound = _finite(getattr(root, "bound", None))
    root_gap = None
    if root_bound is not None:
        root_gap = (known_optimum - root_bound) / max(abs(known_optimum), 1.0)

    return ScoreRow(
        instance=getattr(model, "name", "?"),
        config="",  # filled by caller
        status=status,
        objective=objective,
        known_optimum=known_optimum,
        root_bound=root_bound,
        root_gap=root_gap,
        node_count=node_count,
        wall_time=wall,
        incorrect=_is_incorrect(status, objective, known_optimum),
    )


def run_scoreboard(
    level: str = "smoke",
    configs: dict[str, dict] | None = None,
) -> Scoreboard:
    """Run the QCQP scoreboard over all registered ``qcqp`` instances.

    Parameters
    ----------
    level : str
        ``"smoke"`` (fast subset) or ``"full"`` (all instances).
    configs : dict[str, dict], optional
        Named solver configurations, ``label -> solve_kwargs``. Defaults to a
        single ``"baseline"`` config. A cut family is benchmarked by adding e.g.
        ``{"psd": {"psd_cuts": True}}``.
    """
    if configs is None:
        configs = {"baseline": {}}
    board = Scoreboard()
    for problem in get_problems("qcqp", level=level):
        for label, kwargs in configs.items():
            row = score_instance(problem.build_fn, problem.known_optimum, kwargs)
            row.instance = problem.name
            row.config = label
            board.rows.append(row)
    return board
