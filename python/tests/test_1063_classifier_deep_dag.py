"""#1063: convexity classification must not hold a Python frame per DAG level.

``.nl`` writes an objective as one left-deep ``+`` chain, so DAG depth tracks
*model size*, not modeller nesting. ``squfl015-060``'s objective is 903 levels
deep, and the recursive classifier burned three frames per level
(``classify_expr_info`` -> ``_classify_impl`` -> ``_classify_binary``), so it
raised ``RecursionError`` at roughly node 330 of 4556 and killed the OA path.
"""

from __future__ import annotations

import ast
import contextlib
import pathlib
import sys

import discopt.modeling as dm
import pytest
from discopt._relax.convexity import rules
from discopt._relax.convexity.lattice import Curvature


@contextlib.contextmanager
def _recursion_limit(limit: int):
    """Pin the recursion limit for the duration of the block.

    The suite sets ``sys.setrecursionlimit(3000)`` in ``pyproject.toml`` while a
    user's process defaults to 1000, so a test that reads the ambient limit is
    three times further from failing than production. Pin it explicitly.
    """
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _deep_polynomial_chain(n_terms: int):
    """A left-deep ``+`` chain of squares, the shape ``from_nl`` produces."""
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(n_terms)]
    expr = xs[0] * xs[0]
    for x in xs[1:]:
        expr = expr + x * x
    return m, expr


def test_deep_additive_chain_classifies_without_a_python_frame_chain():
    n_terms = 900
    m, expr = _deep_polynomial_chain(n_terms)

    with _recursion_limit(1000):
        # Control: without this the test would pass on the recursive classifier
        # too, and prove nothing. Three frames per level against a 1000-frame
        # limit means 900 terms cannot fit.
        assert 3 * n_terms > sys.getrecursionlimit()
        curv = rules.classify_expr(expr, m)

    # A sum of squares is convex; asserting the verdict (not merely "it
    # returned") keeps the de-recursion honest about preserving the rules.
    assert curv == Curvature.CONVEX


def test_deep_chain_classification_matches_the_shallow_result():
    """The iterative driver must agree with the classifier on a shape that fits.

    A depth the recursive form handles comfortably, so any disagreement is the
    driver's visit order or its ``_UNDER_POLY_KEY`` threading, not the limit.
    """
    checked = 0
    for n_terms in (1, 2, 5, 40):
        m, expr = _deep_polynomial_chain(n_terms)
        assert rules.classify_expr(expr, m) == Curvature.CONVEX
        checked += 1

        m2, expr2 = _deep_polynomial_chain(n_terms)
        # Negating a sum of squares flips it to concave, which exercises the
        # UnaryOp arm on top of the same chain.
        assert rules.classify_expr(-expr2, m2) == Curvature.CONCAVE
        checked += 1
    assert checked == 8, checked


def _isinstance_types_in(func: ast.FunctionDef) -> set[str]:
    """Type names tested by ``isinstance(<anything>, X)`` inside ``func``."""
    names: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "isinstance"):
            continue
        target = node.args[1]
        for elt in target.elts if isinstance(target, ast.Tuple) else [target]:
            if isinstance(elt, ast.Name):
                names.add(elt.id)
    return names


def test_classify_children_covers_every_dispatched_node_type():
    """``_classify_children`` must know every composite type the rules dispatch on.

    A type handled by ``_classify_impl`` but missing from ``_classify_children``
    is not a wrong answer -- ``_classify_impl`` still reaches its children --
    but it silently reintroduces a frame chain for that shape, which is exactly
    the defect #1063 reports.
    """
    src = pathlib.Path(rules.__file__).read_text()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    dispatchers = [
        "_classify_impl",
        "_classify_binary",
        "_classify_function_call",
        "_classify_matmul",
    ]
    assert set(dispatchers) <= set(funcs), sorted(set(dispatchers) - set(funcs))

    dispatched: set[str] = set()
    for name in dispatchers:
        dispatched |= _isinstance_types_in(funcs[name])
    assert dispatched, "found no isinstance dispatch -- this test asserted nothing"

    covered = _isinstance_types_in(funcs["_classify_children"])
    # Leaves have no sub-expressions to walk into, by definition.
    leaves = {"Constant", "Parameter", "Variable"}
    missing = dispatched - covered - leaves
    assert not missing, f"_classify_children does not handle {sorted(missing)}"


@pytest.mark.parametrize("n_terms", [3, 200])
def test_maximal_polynomial_gate_still_fires_once_per_subtree(n_terms):
    """The #266 optimisation must survive the de-recursion.

    ``quadratic_curvature`` is O(N^3) over *all* model variables; it may run at
    a maximal polynomial subtree and must be skipped beneath one. Counting the
    calls is the only way to see the difference -- the curvature verdict is the
    same either way.
    """
    m, expr = _deep_polynomial_chain(n_terms)

    from discopt._relax.convexity import patterns

    calls = []
    real = patterns.quadratic_curvature

    def counting(e, model):
        calls.append(e)
        return real(e, model)

    patterns.quadratic_curvature = counting
    try:
        with _recursion_limit(1000):
            rules.classify_expr(expr, m)
    finally:
        patterns.quadratic_curvature = real

    # The chain classifies CONVEX from the local rules, so the fallback should
    # not be reached at all; what must never happen is one call per node.
    assert len(calls) <= 1, f"{len(calls)} eigendecompositions over {n_terms} terms"
