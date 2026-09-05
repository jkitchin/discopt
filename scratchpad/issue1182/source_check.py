"""Requirement 1 of #1182: residuals measured on the ORIGINAL predicates.

The simplex weights ``lambda_ij`` of Theorem 1 are *existential witnesses*, not
selectors. A fractional lambda is not a failed Boolean integrality, and it is not
a recoverable named Boolean assignment. So the only honest question to ask of a
returned point is the one this module asks: evaluate each original predicate
``p_jk(z)`` at the returned **source** point and report, per disjunction,

    min_j max_k p_jk(z)

which is <= 0 exactly when some disjunct holds -- the disjunction's own truth,
read off the declared operands and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def variables_in(expr) -> set:
    """Names of the variables an expression touches."""
    from discopt.modeling.core import Variable

    seen: set = set()
    stack = [expr]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, Variable):
            seen.add(node.name)
            continue
        for attr in ("operands", "args", "children"):
            kids = getattr(node, attr, None)
            if kids:
                stack.extend(kids)
                break
        else:
            for attr in ("left", "right", "operand", "base", "expr", "body"):
                kid = getattr(node, attr, None)
                if kid is not None and not isinstance(kid, (int, float, str)):
                    stack.append(kid)
    return seen


@dataclass
class DisjunctionResidual:
    name: str
    per_disjunct_max: list[float]
    violation: float          # min_j max_k p_jk ; <= 0 means the disjunction holds
    definition: str = "min_j max_k p_jk(z) on the declared disjunct rows"


@dataclass
class PredicateReport:
    """Source-predicate residuals. Carries no Boolean assignment, by design."""

    disjunctions: list[DisjunctionResidual] = field(default_factory=list)
    comparisons: int = 0

    @property
    def max_disjunction_violation(self) -> float:
        if not self.disjunctions:
            return float("nan")
        return max(d.violation for d in self.disjunctions)


def _flat_point(model, x_by_name) -> np.ndarray:
    parts = []
    for v in model._variables:
        if v.name not in x_by_name:
            raise KeyError(
                f"source variable {v.name!r} is absent from the returned point; "
                "a source residual cannot be faked from the lowered rows"
            )
        arr = np.asarray(x_by_name[v.name], dtype=np.float64).reshape(-1)
        if arr.size != v.size:
            raise ValueError(f"{v.name}: expected {v.size} values, got {arr.size}")
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0)


def predicate_report(source_model, x_by_name) -> PredicateReport:
    """Evaluate every disjunct row of ``source_model`` at the returned point."""
    from discopt._relax.dag_compiler import compile_expression
    from discopt.modeling.core import Constraint, _DisjunctiveConstraint

    flat = _flat_point(source_model, x_by_name)
    rep = PredicateReport()
    for idx, c in enumerate(source_model._constraints):
        if not isinstance(c, _DisjunctiveConstraint):
            continue
        name = c.name or f"disj{idx}"
        per_disjunct: list[float] = []
        for disjunct in c.disjuncts:
            worst = -np.inf
            for con in disjunct:
                if not isinstance(con, Constraint):
                    raise TypeError(
                        f"{name}: disjunct row is {type(con).__name__}; refusing to "
                        "report a residual over rows this probe cannot evaluate"
                    )
                bodies = []
                if con.sense == "<=":
                    bodies = [con.body]
                elif con.sense == ">=":
                    bodies = [-con.body]
                elif con.sense == "==":
                    bodies = [con.body, -con.body]
                else:
                    raise ValueError(f"{name}: unknown sense {con.sense!r}")
                for body in bodies:
                    val = float(np.max(np.asarray(compile_expression(body, source_model)(flat))))
                    worst = max(worst, val)
                    rep.comparisons += 1
            per_disjunct.append(worst)
        rep.disjunctions.append(
            DisjunctionResidual(name, per_disjunct, min(per_disjunct))
        )
    return rep
