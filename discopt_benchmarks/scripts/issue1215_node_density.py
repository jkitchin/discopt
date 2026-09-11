"""discopt's side of the node-density comparison with oximo (#1215, section 49).

`scripts/oximo_arm/node_density.rs` measures how many arena nodes oximo keeps
per constraint row. This is the same measurement for discopt, in both idioms, so
the two are like for like: it counts the Python `Expression` objects reachable
from the model AND the Rust arena nodes `model_to_repr` produces from them.

The point of the pair is that oximo is **not** vectorised -- it builds one
`Constraint` per row in a plain loop -- yet lands at the same us/row as
vectorised discopt. It gets there by making each element cheap (a flat enum
pushed into a `Vec`, with linear subexpressions fused into one node by the
operator overloads). discopt gets there by not having a per-element cost at all.

Run: `python discopt_benchmarks/scripts/issue1215_node_density.py`
"""

from __future__ import annotations

import gc
import sys

import discopt.modeling as dm
import numpy as np
from discopt._rust import model_to_repr
from discopt.modeling import Expression, Model

INF = 1e20
N = 1000

# Attributes an Expression may use to hold children. Traversing by name rather
# than by node type keeps the probe honest across variant changes: a new variant
# with children under one of these names is still counted.
_CHILD_ATTRS = ("left", "right", "operand", "arg", "args", "operands", "terms")


def py_dag_nodes(model):
    """Distinct `Expression` objects reachable from the model's rows/objective."""
    seen, stack = set(), []
    for c in model._constraints:
        body = getattr(c, "body", None)
        if body is not None:
            stack.append(body)
    if model._objective is not None:
        stack.append(getattr(model._objective, "expr", model._objective))
    while stack:
        node = stack.pop()
        if not isinstance(node, Expression) or id(node) in seen:
            continue
        seen.add(id(node))
        for attr in _CHILD_ATTRS:
            child = getattr(node, attr, None)
            if isinstance(child, Expression):
                stack.append(child)
            elif isinstance(child, (list, tuple)):
                stack.extend(child)
    return len(seen)


def elem_linear(n):
    """One `Constraint` per row -- the idiom oximo and Pyomo are limited to."""
    m = Model("linear")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=INF)
    z = m.continuous("z", shape=(n,), lb=0.0, ub=INF)
    for i in range(n):
        m.subject_to(x[i] + 2.0 * y[i] + 3.0 * z[i] <= 10.0 + i % 7, name=f"c[{i}]")
    m.minimize(dm.sum([x[i] for i in range(n)]))
    return m


def vec_linear(n):
    """One array-valued body for the whole family -- discopt's own idiom."""
    m = Model("linear")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=INF)
    z = m.continuous("z", shape=(n,), lb=0.0, ub=INF)
    m.subject_to(x + 2.0 * y + 3.0 * z <= 10.0 + np.arange(n) % 7, name="c")
    m.minimize(dm.sum(x))
    return m


def main() -> int:
    print(f"{'idiom':<14}{'rows':>6}{'py DAG objs':>13}{'/row':>8}{'arena nodes':>13}{'/row':>8}")
    checks = 0
    for label, build in (("per-element", elem_linear), ("vectorised", vec_linear)):
        m = build(N)
        py = py_dag_nodes(m)
        rep = model_to_repr(m, getattr(m, "_builder", None))
        nodes = rep.n_nodes
        # Row count from the repr, not from N: a build that dropped rows would
        # otherwise divide by a count it never produced.
        rows = rep.n_constraints
        expected = N if label == "per-element" else 1
        assert rows == expected, f"{label}: {rows} rows, expected {expected}"
        print(f"{label:<14}{rows:>6}{py:>13}{py / N:>8.2f}{nodes:>13}{nodes / N:>8.2f}")
        checks += 1
        del m, rep
        gc.collect()
    print(f"# executed: {checks} idiom measurements over {N} rows each", file=sys.stderr)
    return 0 if checks else 1


if __name__ == "__main__":
    sys.exit(main())
