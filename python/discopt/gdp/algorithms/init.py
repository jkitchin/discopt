"""Initialization strategies of the native GDP algorithms (#1775 G3.3).

Before its main loop an algorithm such as LOA solves the subproblems of a few
configurations, to seed the master with cuts and to find a first incumbent.
Pyomo GDPopt's ``init_algorithm`` names the strategies; this module implements
them over :class:`GDPSpace` (the model with every disjunct's indicator a master
column) and the caller's ``solve(selection)`` callback:

* ``"no_init"`` -- nothing.
* ``"fix_disjuncts"`` -- the one configuration whose indicators the user fixed
  at their active value (``lb == ub``); it must decide every active disjunction.
* ``"custom_disjuncts"`` -- the configurations the caller lists, in order.
* ``"max_binary"`` -- one master solve maximizing the number of active
  disjuncts, then its configuration.
* ``"set_covering"`` -- repeated master solves, each maximizing the number of
  not-yet-covered disjuncts; a disjunct is covered once it was active in a
  configuration whose subproblem returned a point, and disjuncts holding only
  linear rows start covered (the master already sees them exactly). Stops when
  every disjunct is covered, the master finds nothing uncovered or nothing at
  all, ``iterlim`` is reached, or the deadline expires.

The strategies only *choose* configurations. A configuration whose subproblem
fails is excluded from the strategy's own later master solves, never from the
caller's master: whether it may be excluded there is the algorithm's proof to
make (C-35). Every master solve gets the deadline remainder.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.algorithms.master import (
    DEFAULT_MASTER_BACKEND,
    MasterMILP,
    _add_no_good_cut,
    build_master,
)
from discopt.gdp.tree import DisjunctionNode, DisjunctNode, GDPTree, gdp_tree
from discopt.modeling.core import Model, _DisjunctiveConstraint

INIT_STRATEGIES = ("no_init", "fix_disjuncts", "custom_disjuncts", "max_binary", "set_covering")

Selection = list[DisjunctNode]


def _materialize_indicators(model: Model) -> None:
    """Read every top-level ``either_or``'s indicators, so each is a column."""
    for con in list(model._constraints):
        if isinstance(con, _DisjunctiveConstraint) and con.parent_indicator is None:
            con.indicators  # noqa: B018 - allocation on first read is the point


@dataclass
class GDPSpace:
    """A working copy of a GDP with every disjunct's indicator a column.

    ``work`` is a clone of the caller's model (the original is never touched)
    with every ``either_or`` indicator allocated; ``lowered`` is its big-M
    lowering, whose first columns are ``work``'s, so ``columns[d]`` indexes both.
    ``work`` is the model the subproblems are solved on.
    """

    work: Model
    lowered: Model
    tree: GDPTree
    columns: dict[DisjunctNode, int]
    nonlinear: set[DisjunctNode] = field(default_factory=set)
    #: The tree before the indicators were read, and its nodes' counterparts in
    #: ``tree`` -- a caller's own ``gdp_tree(model)`` resolves onto it.
    before: Optional[GDPTree] = None
    counterpart: dict[DisjunctNode, DisjunctNode] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: Model) -> "GDPSpace":
        from discopt._relax.gdp_reformulate import _extract_body_coeffs
        from discopt.transformations import get
        from discopt.validation.feasibility import _column_indices

        work = model.clone()
        before = gdp_tree(work)
        _materialize_indicators(work)
        lowered = get("gdp").apply(work, method="big-m")
        names = [v.name for v in work._variables]
        if [v.name for v in lowered._variables[: len(names)]] != names:
            raise RuntimeError("GDPSpace: the lowering reordered the model's columns")
        tree = gdp_tree(work)
        columns: dict[DisjunctNode, int] = {}
        nonlinear: set[DisjunctNode] = set()
        n = sum(int(v.size) for v in work._variables)
        for node in tree.walk():
            for d in node.disjuncts:
                if d.indicator is None:
                    raise NotImplementedError(
                        f"GDPSpace: a disjunct of {node.name or '<anon>'!r} has no indicator "
                        "column (a nested Model.disjunction); nest Disjunct blocks instead"
                    )
                cols = _column_indices(d.indicator.variable, work)
                if cols.size != 1:
                    raise RuntimeError(f"GDPSpace: indicator of {d!r} is not one column")
                columns[d] = int(cols[0])
                if any(_extract_body_coeffs(r.body, work, n) is None for r in d.rows):
                    nonlinear.add(d)
        old = [d for node in before.walk() for d in node.disjuncts]
        new = [d for node in tree.walk() for d in node.disjuncts]
        if [(d.name, [str(r) for r in d.rows]) for d in old] != [
            (d.name, [str(r) for r in d.rows]) for d in new
        ]:
            raise RuntimeError("GDPSpace: reading the indicators changed the GDP tree")
        return cls(work, lowered, tree, columns, nonlinear, before, dict(zip(old, new)))

    def master(self, backend: str = DEFAULT_MASTER_BACKEND) -> MasterMILP:
        return build_master(self.lowered, backend=backend)

    def selection_from_x(self, x: Any) -> Selection:
        """The disjuncts a master point selects, walking down from the roots."""
        x = np.asarray(x, dtype=np.float64)
        out: Selection = []

        def visit(node: DisjunctionNode) -> None:
            for d in node.disjuncts:
                if round(float(x[self.columns[d]])) == d.active_value:
                    out.append(d)
                    for child in d.children:
                        visit(child)

        for root in self.tree.disjunctions:
            visit(root)
        return out

    def check(self, selection: Iterable[DisjunctNode]) -> Selection:
        """``selection`` as a list of this space's nodes.

        A node of another tree of the same model -- the caller's own
        ``gdp_tree(model)``, or one rebuilt from ``work`` -- is resolved by its
        address with the structural check of :mod:`discopt.gdp.state` (#1794
        S2); a node of a different or changed model is refused.
        """
        from discopt.gdp.state import _resolve

        def one(d: DisjunctNode) -> DisjunctNode:
            if d in self.columns:
                return d
            try:
                return _resolve(self.tree, d)
            except ValueError:
                if self.before is None:
                    raise
                return self.counterpart[_resolve(self.before, d)]

        return [one(d) for d in selection]


@dataclass
class InitReport:
    """What a strategy tried: each configuration with the callback's result."""

    strategy: str
    tried: list[tuple[Selection, Any]] = field(default_factory=list)
    covered: set[DisjunctNode] = field(default_factory=set)
    uncovered: set[DisjunctNode] = field(default_factory=set)
    stop_reason: str = ""


def _activity_objective(space: GDPSpace, n: int, targets: Iterable[DisjunctNode]) -> np.ndarray:
    """Minimizing this maximizes how many of ``targets`` are active."""
    c = np.zeros(n)
    for d in targets:
        c[space.columns[d]] += -1.0 if d.active_value == 1 else 1.0
    return c


def _no_point(r: Any) -> str:
    return "master infeasible" if r.proven_infeasible else "master no solution"


def _solved(result: Any) -> bool:
    return getattr(result, "x", None) is not None


def initialize(
    strategy: str,
    space: GDPSpace,
    solve: Callable[[Selection], Any],
    deadline: Deadline,
    *,
    master: Optional[MasterMILP] = None,
    custom: Optional[Sequence[Iterable[DisjunctNode]]] = None,
    iterlim: int = 8,
    master_time_cap: Optional[float] = None,
) -> InitReport:
    """Run ``strategy`` (one of :data:`INIT_STRATEGIES`); see the module docstring.

    ``solve(selection)`` is the algorithm's: it solves the subproblem (and adds
    cuts, offers incumbents, records results). Its result counts as a point
    when it has a non-``None`` ``x``. ``master`` defaults to ``space.master()``
    and is never changed: the strategies' exclusions are per call.
    """
    if strategy not in INIT_STRATEGIES:
        raise ValueError(f"init strategy must be one of {INIT_STRATEGIES}, got {strategy!r}")
    if strategy != "custom_disjuncts" and custom is not None:
        raise ValueError("custom configurations are read only by 'custom_disjuncts'")
    if iterlim < 0:
        raise ValueError(f"iterlim must be >= 0, got {iterlim!r}")
    report = InitReport(strategy)
    if strategy == "no_init":
        report.stop_reason = "no_init"
        return report
    if strategy == "fix_disjuncts":
        from discopt.gdp.state import _fixed_selection

        fixed = _fixed_selection(space.work)
        if not fixed:
            raise ValueError(
                "fix_disjuncts: no indicator is fixed at its active value; fix one disjunct "
                "of every disjunction (lb == ub) or choose another init strategy"
            )
        sel = space.check(fixed)
        if not deadline.expired():
            report.tried.append((sel, solve(sel)))
        report.stop_reason = "done" if report.tried else "deadline"
        return report
    if strategy == "custom_disjuncts":
        if not custom:
            raise ValueError("custom_disjuncts needs at least one configuration in custom=")
        configs = [space.check(c) for c in custom]
        for sel in configs:
            if deadline.expired():
                report.stop_reason = "deadline"
                return report
            report.tried.append((sel, solve(sel)))
        report.stop_reason = "done"
        return report

    m = master if master is not None else space.master()
    if strategy == "max_binary":
        if deadline.expired():
            report.stop_reason = "deadline"
            return report
        c = _activity_objective(space, m.n_vars, space.columns)
        r = m.solve(deadline.sub_limit(master_time_cap), objective_override=c)
        if r.x is None:
            report.stop_reason = (
                "master infeasible" if r.proven_infeasible else "master no solution"
            )
            return report
        sel = space.selection_from_x(r.x)
        report.tried.append((sel, solve(sel)))
        report.stop_reason = "done"
        return report

    # set_covering
    covered = set(space.columns) - space.nonlinear
    # A failed selection is excluded on the indicator columns alone (always 0/1):
    # excluding the master's full integer configuration would let it propose the
    # same selection again with other auxiliary values (review of #1865).
    cols = sorted(set(space.columns.values()))
    failed: list[tuple[np.ndarray, float]] = []
    while True:
        uncovered = set(space.columns) - covered
        if not uncovered:
            report.stop_reason = "covered"
            break
        if len(report.tried) >= iterlim:
            report.stop_reason = "iterlim"
            break
        if deadline.expired():
            report.stop_reason = "deadline"
            break
        c = _activity_objective(space, m.n_vars, uncovered)
        r = m.solve(deadline.sub_limit(master_time_cap), objective_override=c, extra_cuts=failed)
        if r.x is None:
            report.stop_reason = (
                "master infeasible" if r.proven_infeasible else "master no solution"
            )
            break
        sel = space.selection_from_x(r.x)
        if not uncovered.intersection(sel):
            report.stop_reason = "nothing new to cover"
            break
        result = solve(sel)
        report.tried.append((sel, result))
        if _solved(result):
            covered.update(sel)
        else:
            assert r.x is not None
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            _add_no_good_cut(np.asarray(r.x), cols, rows, rhs, m.n_vars)
            failed.append((rows[0], rhs[0]))
    report.covered = covered & set(space.columns)
    report.uncovered = set(space.columns) - covered
    return report


__all__ = ["INIT_STRATEGIES", "GDPSpace", "InitReport", "Selection", "initialize"]
