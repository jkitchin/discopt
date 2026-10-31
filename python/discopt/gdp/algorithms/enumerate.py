"""Configuration enumeration (#1775 G4.2, spec §5.3).

:class:`Enumerate` solves every configuration of the GDP -- one selection per
active disjunction, walking into the disjunctions nested in the selected
disjuncts -- as a global subproblem (§5.2), and keeps the best verified point.

* **Configurations.** An ``exactly(1)`` disjunction contributes each of its
  disjuncts; an OR (``AT_LEAST_ONE``) disjunction each nonempty subset of them.
  An implication (``if_then``) is left alone: its rows stay conditional in the
  subproblem, which is exact. Any other cardinality is refused.
* **Logic propagation (G5.3).** Configurations the GDP's logic refutes -- its
  ``logical()`` constraints or a linear row over indicators alone
  (:mod:`discopt.gdp.propagate`) -- are never generated: each is infeasible, so
  dropping it changes no bound. Pyomo enumerates ``itertools.product`` and lets
  infeasible logic fail inside the NLP. Measured on the panel: syngas 8388608
  configurations -> 1440, mod_hens 4.3e9 -> 4096, gdp_col 32768 -> 35.
* **Refusal, never truncation.** The surviving count is computed before anything
  is solved; above ``max_configurations`` (default 10⁴) the run raises.
* **Bound (§5.4).** The LB is the minimum, over every configuration, of its
  certified bound (``+inf`` for one certified infeasible). It is published only
  once every configuration has been solved *and* has a valid bound; until then
  there is none. ``optimal`` needs every configuration certified (the base
  class refuses it while any is unresolved).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterator, Optional

from discopt.gdp.algorithms.base import GDPAlgorithm
from discopt.gdp.algorithms.init import GDPSpace
from discopt.gdp.tree import DisjunctionNode, DisjunctNode
from discopt.modeling.core import Model, SelectorCardinality

#: The default ceiling on the configuration count (spec §5.3).
MAX_CONFIGURATIONS = 10_000


def _choices(node: DisjunctionNode) -> list[list[DisjunctNode]]:
    """The disjuncts ``node`` may select together, as lists."""
    card = node.semantics.cardinality
    if card is SelectorCardinality.EXACTLY_ONE:
        return [[d] for d in node.disjuncts]
    if card is SelectorCardinality.AT_LEAST_ONE:
        return [
            list(sub)
            for k in range(1, len(node.disjuncts) + 1)
            for sub in itertools.combinations(node.disjuncts, k)
        ]
    raise NotImplementedError(
        f"enumerate: disjunction {node.name!r} has cardinality {card.value}, which a "
        "configuration (one selection per disjunction) cannot represent"
    )


def _count(node: DisjunctionNode) -> int:
    """How many configurations ``node`` contributes when active."""
    if node.kind == "implication":
        return 1

    def under(d: DisjunctNode) -> int:
        return math.prod(_count(c) for c in d.children)

    card = node.semantics.cardinality
    if card is SelectorCardinality.EXACTLY_ONE:
        return sum(under(d) for d in node.disjuncts)
    if card is SelectorCardinality.AT_LEAST_ONE:
        return math.prod(1 + under(d) for d in node.disjuncts) - 1
    _choices(node)  # raises for the cardinality
    return 0


def _configurations(node: DisjunctionNode) -> Iterator[list[DisjunctNode]]:
    if node.kind == "implication":
        yield []
        return
    for chosen in _choices(node):
        nested = [_configurations(c) for d in chosen for c in d.children]
        for parts in itertools.product(*[list(g) for g in nested]):
            yield chosen + [d for part in parts for d in part]


def count_configurations(space: GDPSpace) -> int:
    """The number of configurations :class:`Enumerate` would solve."""
    return math.prod(_count(root) for root in space.tree.disjunctions)


def configurations(space: GDPSpace) -> Iterator[list[DisjunctNode]]:
    """Every configuration of ``space``, as a selection for ``solve_subproblem``."""
    roots = [list(_configurations(r)) for r in space.tree.disjunctions]
    for parts in itertools.product(*roots):
        yield [d for part in parts for d in part]


class Enumerate(GDPAlgorithm):
    """Configuration enumeration; see the module docstring."""

    route = "gdpopt-enumerate: configuration enumeration (global subproblems)"
    # The LB is published only once every configuration has a bound (iterate).
    bound_needs_resolution = False

    def __init__(
        self,
        model: Model,
        *,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        max_configurations: int = MAX_CONFIGURATIONS,
        propagate: bool = True,
        iteration_limit: Optional[int] = None,
        **kw: Any,
    ) -> None:
        if isinstance(max_configurations, bool) or not (
            isinstance(max_configurations, int) and max_configurations >= 1
        ):
            raise ValueError(
                f"max_configurations must be an integer >= 1, got {max_configurations!r}"
            )
        self.space = GDPSpace.from_model(model)
        super().__init__(
            self.space.work,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            iteration_limit=iteration_limit,
            **kw,
        )
        from discopt.gdp.propagate import LogicChecker, count_pruned, pruned_configurations

        self.n_configurations_raw = count_configurations(self.space)
        self.propagate = bool(propagate)
        if self.propagate:
            self.checker: Optional[LogicChecker] = LogicChecker(self.space)
            kept = count_pruned(self.space, max_configurations, self.checker)
        else:
            self.checker = None
            raw = self.n_configurations_raw
            kept = raw if raw <= max_configurations else None
        if kept is None:
            raise ValueError(
                f"enumerate: the GDP has more than max_configurations={max_configurations} "
                f"configurations ({self.n_configurations_raw} before logic propagation); "
                "raise the limit or use gdp_method='ric', 'gloa' or 'loa' (enumerate "
                "never truncates)"
            )
        self.n_configurations = kept
        self._todo: Iterator[list[DisjunctNode]] = (
            pruned_configurations(self.space, self.checker)
            if self.checker is not None
            else configurations(self.space)
        )
        #: Each solved configuration's bound (minimization convention): ``+inf``
        #: when certified infeasible, ``None`` when there is no valid bound.
        self.config_bounds: list[Optional[float]] = []
        #: Configurations closed by the incumbent cutoff (#1917).
        self.n_cutoff_dominated = 0
        #: Configurations re-solved without the cutoff (no proof under it).
        self.n_cutoff_retried = 0

    def iterate(self) -> bool:
        from discopt.gdp.algorithms.subproblem import solve_subproblem

        selection = next(self._todo, None)
        if selection is None:
            self._finish()
            return False
        index = len(self.config_bounds)
        c = self._cutoff()
        res = solve_subproblem(
            self.space.work,
            selection,
            self.deadline,
            mode="global",
            gap_tolerance=self.gap_tolerance,
            cutoff=None if c is None else self.obj_sign * c,
        )
        self.n_subproblems += 1
        if res.x is not None:
            self.offer_incumbent(res.x)
        if c is not None and res.status not in ("infeasible", "optimal"):
            # Neither proof came: a configuration whose optimum ties the incumbent
            # leaves the restriction infeasible by only ``delta``, which a solve may
            # not certify (measured: a point violating the cutoff row by 6.5e-6 and
            # no bound). Solve the configuration as without the flag instead.
            self.n_cutoff_retried += 1
            c = None
            res = solve_subproblem(
                self.space.work,
                selection,
                self.deadline,
                mode="global",
                gap_tolerance=self.gap_tolerance,
            )
            self.n_subproblems += 1
            if res.x is not None:
                self.offer_incumbent(res.x)
        bound = None if res.bound is None else self.obj_sign * float(res.bound)
        if c is not None:
            # The result is about the configuration restricted to f <= c. Its
            # minimum m is either <= c (the restriction removes no minimizer, so a
            # bound b on the restricted problem has b <= m) or > c (the restricted
            # problem is empty, and whatever bound it reported says nothing). In
            # both cases min(b, c) <= m, and a certified-empty restriction proves
            # m > c, so c is its bound.
            if res.status == "infeasible":
                self.n_cutoff_dominated += 1
                self.config_bounds.append(c)
                self.mark_solved(index)
                return True
            if bound is not None:
                bound = min(bound, c)
        if res.status == "infeasible":
            self.config_bounds.append(math.inf)
            self.mark_solved(index)
        elif res.status == "optimal":
            self.config_bounds.append(bound)
            self.mark_solved(index)
        else:
            self.config_bounds.append(bound)
            self.mark_unresolved(index)
        return True

    def _cutoff(self) -> Optional[float]:
        """The incumbent cutoff for the next configuration (minimization convention).

        ``DISCOPT_GDP_ENUMERATE_CUTOFF`` (default OFF, #1917): ``UB - delta`` with
        ``delta`` half the gap the convergence test allows at ``UB``, so a
        configuration bounded by it keeps the final gap within tolerance. ``None``
        with the flag off or no incumbent yet.
        """
        from discopt.solver_tuning import current as _tuning

        if not _tuning().gdp_enumerate_cutoff or not math.isfinite(self.ub):
            return None
        delta = 0.5 * self.gap_tolerance * abs(self.ub) + 0.5 * self.abs_gap_tolerance
        return self.ub - delta

    def _finish(self) -> None:
        if len(self.config_bounds) != self.n_configurations:
            raise RuntimeError(
                f"enumerate: solved {len(self.config_bounds)} configurations of "
                f"{self.n_configurations}"
            )
        if any(b is None for b in self.config_bounds):
            return  # a configuration with no valid bound: no LB at all
        if all(b == math.inf for b in self.config_bounds):
            if not self.unresolved:
                self.prove_infeasible()
            return
        self.offer_bound(min(b for b in self.config_bounds if b is not None))

    def extra_stats(self) -> dict[str, Any]:
        return {
            "n_configurations": int(self.n_configurations),
            "n_configurations_raw": int(self.n_configurations_raw),
            "n_configurations_solved": len(self.config_bounds),
            "n_cutoff_dominated": int(self.n_cutoff_dominated),
            "n_cutoff_retried": int(self.n_cutoff_retried),
        }


def solve_enumerate(model: Model, **options: Any):
    """Run :class:`Enumerate` on ``model`` and return its ``SolveResult``."""
    return Enumerate(model, **options).run()


__all__ = [
    "MAX_CONFIGURATIONS",
    "Enumerate",
    "configurations",
    "count_configurations",
    "solve_enumerate",
]
