"""#1154: does the ``SumOverExpression`` node actually reach the relaxation layer?

The capability panel found the ``auto`` and ``big-m`` routes bit-identical
between the ``Σ[...]`` arm and the folded-chain arm on all 108 cases. That is
only meaningful evidence about the downstream relaxer if the node *survives* the
GDP reformulation instead of being rebuilt into a BinaryOp chain on the way. This
probe counts the surviving nodes in the reformulated model.

Prints a node count and exits non-zero if it is zero (CLAUDE.md §6).
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.modeling.core import (
    BinaryOp,
    Constraint,
    FunctionCall,
    IndexExpression,
    SumOverExpression,
    UnaryOp,
)


def count_sumover(expr) -> int:
    stack, seen, n = [expr], set(), 0
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, SumOverExpression):
            n += 1
            stack.extend(e.terms)
        elif isinstance(e, BinaryOp):
            stack.extend((e.left, e.right))
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, IndexExpression):
            stack.append(e.base)
        elif isinstance(e, FunctionCall):
            stack.extend(e.args)
    return n


def build():
    m = dm.Model("survives")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]])
    m.minimize(-(x[0] + x[1] + x[2]))
    return m


total = 0
for method in ("big-m", "hull"):
    ref = reformulate_gdp(build(), method=method)
    n = sum(count_sumover(c.body) for c in ref._constraints if isinstance(c, Constraint))
    print(f"{method}: SumOverExpression nodes in the reformulated model = {n}")
    total += n

print(f"surviving_nodes={total}")
if total == 0:
    print("PROBE DID NOT FIRE: the node never reaches the relaxer", file=sys.stderr)
    sys.exit(1)
