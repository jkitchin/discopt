"""Issue #1544: a wide canonical sum/product must not reconstruct to a deep tree.

A canonical ``sum``/``prod`` node is n-ary; ``reconstruct`` used to left-fold it into
an ``n``-deep binary ``Expression``. Every consumer of that tree on the relaxation
path (``evaluate_interval`` in ``uniform_relax._Builder.bounds``, the curvature
classifier, the DAG compiler) recurses per level, so a *shallow* model whose
canonical form has one wide node raised ``RecursionError`` in the relaxation build:

* ``nvs09`` under ``x = y - c``: ``(prod x_i)**0.2`` becomes a product of ten shifted
  binomials, which distributes to a 1024-term sum under the power -- 490 nested
  interval-eval frames from a 24-deep objective;
* the class needs no shift: ``sin(sum(x)/100)`` with 600 ``x`` reached the same crash
  site, cost the solve its relaxation and returned an uncertified ``feasible``.

Wide nodes now fold as a balanced tree. Up to ``_RECONSTRUCT_FOLD_WIDTH`` the legacy
left fold is kept, so every node that already reconstructed is byte-identical.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.canonical_expr import (
    _RECONSTRUCT_FOLD_WIDTH,
    canonicalize,
    reconstruct,
)

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

from discopt._relax.dag_compiler import compile_expression
from discopt.modeling.core import BinaryOp, FunctionCall, UnaryOp


def _depth(expr) -> int:
    """Operator-tree depth, computed iteratively (the thing under test is depth)."""
    best = 0
    stack = [(expr, 1)]
    while stack:
        e, d = stack.pop()
        best = max(best, d)
        if isinstance(e, BinaryOp):
            stack.append((e.left, d + 1))
            stack.append((e.right, d + 1))
        elif isinstance(e, UnaryOp):
            stack.append((e.operand, d + 1))
        elif isinstance(e, FunctionCall):
            stack.extend((a, d + 1) for a in e.args)
    return best


def _wide_model(n: int, kind: str):
    """A shallow model (array variable + one reduction) whose canonical form has one
    ``n``-wide ``kind`` node."""
    m = dm.Model(f"wide_{kind}")
    x = m.continuous("x", shape=(n,), lb=0.5, ub=1.5)
    if kind == "sum":
        body = dm.sin(dm.sum(np.arange(1.0, n + 1.0) * x))
    else:
        # Multiply pairwise so the MODEL stays shallow; canonicalization flattens
        # the nested products into one n-ary ``prod`` node.
        level = [x[i] for i in range(n)]
        while len(level) > 1:
            nxt = [level[i] * level[i + 1] for i in range(0, len(level) - 1, 2)]
            level = nxt + ([level[-1]] if len(level) % 2 else [])
        body = level[0]
    m.minimize(body)
    return m, x


@pytest.mark.parametrize("kind", ["sum", "prod"])
def test_wide_node_reconstructs_shallow_and_equivalent(kind):
    n = 1024
    m, x = _wide_model(n, kind)
    dag = canonicalize(m)
    wide = [c for c in dag.nodes if c.kind == kind and len(c.children) >= n]
    assert len(wide) == 1, "fixture must canonicalize to one wide node"
    expr = reconstruct(dag.objective, m)
    assert _depth(expr) < 40, _depth(expr)

    # Value-equivalent to the model's own objective at random points.
    f_new = compile_expression(expr, m)
    f_old = compile_expression(m._objective.expression, m)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(5):
        pt = rng.uniform(0.5, 1.5, size=n)
        want = float(f_old(pt))
        assert float(f_new(pt)) == pytest.approx(want, rel=1e-9, abs=1e-12)
        checked += 1
    assert checked == 5


def test_narrow_node_keeps_legacy_left_fold():
    """At or below the threshold the operator tree is the legacy left fold."""
    n = _RECONSTRUCT_FOLD_WIDTH
    m, _ = _wide_model(n, "prod")
    expr = reconstruct(canonicalize(m).objective, m)
    assert _depth(expr) == n  # n factors -> n-1 multiplications -> depth n


def test_shallow_model_with_wide_sum_certifies():
    """No shift, no deep model expression: a 600-term sum under ``sin``.

    Before the fix the relaxer setup raised RecursionError and the solve ended
    ``feasible`` / uncertified.
    """
    m = dm.Model("wide_sin")
    x = m.continuous("x", shape=(600,), lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.sin(dm.sum(x) / 100.0) * y + y)
    m.subject_to(dm.sum(x) <= 300)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.gap_certified
    assert abs(r.objective) <= 1e-6  # y(1 + sin(s/100)) >= 0, attained at y = 0


# The issue's own instance (shifted nvs09) is covered end to end, certificate
# included, by ``test_1544_cancellation_lift.py::test_nvs09_translated_certifies``.
