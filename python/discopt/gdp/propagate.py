"""Logic propagation over a GDP's indicators (#1775 G5.3, spec §7).

A configuration (one selection per active disjunction, :mod:`.algorithms.enumerate`)
fixes every disjunct indicator. Two kinds of rows constrain those indicators alone,
and both can be decided on a *partial* assignment:

* **declared logic** -- the ``logical()`` constraints of the GDP tree, read with a
  three-valued (Kleene) evaluation: a leaf not yet assigned is unknown, and a
  constraint that is already ``False`` stays false for every completion;
* **indicator-only linear rows** -- a linear row of the model whose support is
  indicator columns only (Pyomo writes much of its logic this way): it is
  violated for every completion when its extreme over the unassigned 0/1
  indicators already breaks it.

Both checks are exact refutations -- they only ever reject an assignment none of
whose completions satisfies the row -- so pruning by them removes infeasible
configurations and nothing else. Pyomo's ``check_sat`` asks z3 the same question;
this needs no dependency. Leaves that are not indicators (a free ``BooleanVar``,
an implication's own selector) stay unknown, which only weakens the check.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator, Optional

import numpy as np

from discopt.gdp.tree import DisjunctionNode, DisjunctNode

#: A partial assignment: indicator column -> 0 or 1.
Assignment = dict[int, int]


def _kleene(expr: Any, value_of: Any) -> Optional[bool]:
    """``True``/``False`` if ``expr`` is decided under the assignment, else ``None``."""
    from discopt.modeling.core import (
        BooleanVar,
        LogicalAnd,
        LogicalAtLeast,
        LogicalAtMost,
        LogicalEquivalent,
        LogicalExactly,
        LogicalImplies,
        LogicalNot,
        LogicalOr,
    )

    if isinstance(expr, BooleanVar):
        leaf = value_of(expr)
        return None if leaf is None else bool(leaf)
    if isinstance(expr, LogicalNot):
        v = _kleene(expr.operand, value_of)
        return None if v is None else not v
    if isinstance(expr, LogicalAnd):
        a, b = _kleene(expr.left, value_of), _kleene(expr.right, value_of)
        if a is False or b is False:
            return False
        return True if (a and b) else None
    if isinstance(expr, LogicalOr):
        a, b = _kleene(expr.left, value_of), _kleene(expr.right, value_of)
        if a is True or b is True:
            return True
        return False if (a is False and b is False) else None
    if isinstance(expr, LogicalImplies):
        a, b = _kleene(expr.antecedent, value_of), _kleene(expr.consequent, value_of)
        if a is False or b is True:
            return True
        return False if (a is True and b is False) else None
    if isinstance(expr, LogicalEquivalent):
        a, b = _kleene(expr.left, value_of), _kleene(expr.right, value_of)
        return None if (a is None or b is None) else a == b
    if isinstance(expr, (LogicalAtLeast, LogicalAtMost, LogicalExactly)):
        vals = [_kleene(op, value_of) for op in expr.operands]
        t = sum(v is True for v in vals)
        u = sum(v is None for v in vals)
        k = int(expr.k)
        at_least: Optional[bool] = True if t >= k else (False if t + u < k else None)
        at_most: Optional[bool] = True if t + u <= k else (False if t > k else None)
        if isinstance(expr, LogicalAtLeast):
            return at_least
        if isinstance(expr, LogicalAtMost):
            return at_most
        if at_least is False or at_most is False:
            return False
        return True if (at_least and at_most) else None
    return None  # an unrecognised node decides nothing


class LogicChecker:
    """Exact refutation of partial indicator assignments; see the module docstring."""

    def __init__(self, space: Any) -> None:
        from discopt._relax.gdp_reformulate import _extract_body_coeffs
        from discopt.modeling.core import Constraint

        self.space = space
        work = space.work
        n = sum(int(v.size) for v in work._variables)
        self.indicator_cols = set(space.columns.values())
        # A BooleanVar leaf is read through its variable's (scalar) column.
        self._col_of_var: dict[int, int] = {}
        offset = 0
        for v in work._variables:
            if int(v.size) == 1:
                self._col_of_var[id(v)] = offset
            offset += int(v.size)
        self.logic = [lc.expression for lc in space.tree.logic]
        rows = []
        for c in work._constraints:
            if not isinstance(c, Constraint):
                continue
            coeffs = _extract_body_coeffs(c.body, work, n)
            if coeffs is None:
                continue
            a, const = coeffs
            a = np.asarray(a, dtype=np.float64)
            support = np.flatnonzero(np.abs(a) > 0.0)
            if support.size == 0 or not set(support.tolist()) <= self.indicator_cols:
                continue
            # ``body sense 0`` with body = a @ y + const.
            rows.append((support, a[support], c.sense, float(const)))
        self.rows = rows
        self.checks = 0
        self.refuted = 0

    def _value_of(self, assign: Assignment) -> Any:
        def value_of(bv: Any) -> Optional[bool]:
            col = self._col_of_var.get(id(bv.variable))
            if col is None or col not in assign:
                return None
            return bool(assign[col])

        return value_of

    def refutes(self, assign: Assignment, tol: float = 1e-9) -> bool:
        """Whether no completion of ``assign`` satisfies the logic and the rows."""
        self.checks += 1
        value_of = self._value_of(assign)
        for e in self.logic:
            if _kleene(e, value_of) is False:
                self.refuted += 1
                return True
        for support, coef, sense, const in self.rows:
            lo = hi = const
            for j, a in zip(support.tolist(), coef.tolist()):
                if j in assign:
                    lo += a * assign[j]
                    hi += a * assign[j]
                elif a > 0:
                    hi += a
                else:
                    lo += a
            if (
                (sense == "<=" and lo > tol)
                or (sense == ">=" and hi < -tol)
                or (sense == "==" and (lo > tol or hi < -tol))
            ):
                self.refuted += 1
                return True
        return False


def _subtree_assignments(
    node: DisjunctionNode, options: Any
) -> Iterator[tuple[list[DisjunctNode], Assignment]]:
    """For each configuration of ``node``: the selection and every indicator it fixes."""
    columns = options.columns

    def off(d: DisjunctNode, a: Assignment) -> None:
        a[columns[d]] = 1 - d.active_value
        for child in d.children:
            if child.kind != "implication":
                for dd in child.disjuncts:
                    off(dd, a)

    from discopt.gdp.algorithms.enumerate import _configurations

    for selection in _configurations(node):
        chosen = {id(d) for d in selection}
        a: Assignment = {}

        def visit(nd: DisjunctionNode) -> None:
            if nd.kind == "implication":
                return
            for d in nd.disjuncts:
                if id(d) in chosen:
                    a[columns[d]] = d.active_value
                    for child in d.children:
                        visit(child)
                else:
                    off(d, a)

        visit(node)
        yield selection, a


def pruned_configurations(
    space: Any, checker: Optional[LogicChecker] = None
) -> Iterator[list[DisjunctNode]]:
    """Every configuration of ``space`` that ``checker`` cannot refute, by a
    depth-first walk over the roots that abandons a refuted prefix."""
    checker = checker if checker is not None else LogicChecker(space)
    per_root = [list(_subtree_assignments(r, space)) for r in space.tree.disjunctions]

    def walk(i: int, chosen: list[DisjunctNode], assign: Assignment) -> Iterator[list]:
        if i == len(per_root):
            yield list(chosen)
            return
        for selection, a in per_root[i]:
            merged = {**assign, **a}
            if checker.refutes(merged):
                continue
            yield from walk(i + 1, chosen + selection, merged)

    yield from walk(0, [], {})


def count_pruned(space: Any, cap: int, checker: Optional[LogicChecker] = None) -> Optional[int]:
    """The number of unrefuted configurations, or ``None`` once it exceeds ``cap``."""
    n = 0
    for _ in itertools.islice(pruned_configurations(space, checker), cap + 1):
        n += 1
    return None if n > cap else n


__all__ = ["LogicChecker", "count_pruned", "pruned_configurations"]
