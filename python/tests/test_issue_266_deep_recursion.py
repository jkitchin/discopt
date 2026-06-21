"""Issue #266: deep expression graphs must not crash convexity classification
or degrade a pure-continuous solve to ``status="error"``.

The syntactic curvature walker recurses one Python frame per expression node, so
``from_nl``-rebuilt models with thousands of terms (e.g. pooling problems) could
exceed CPython's default 1000-frame recursion limit and raise ``RecursionError``.
That was caught upstream and demoted the model to convexity-unknown, which on a
pure-continuous model routed to a best-effort local NLP that could itself fail
and surface a bare ``status="error"`` with no diagnostic.

Two robustness fixes are exercised here:

* :func:`classify_model` runs the walk with size-scaled recursion headroom (on a
  large-stack worker thread) so deep graphs classify instead of raising, and
* a convexity-unknown pure-continuous solve that errors in the local NLP falls
  through to the sound spatial Branch-and-Bound rather than returning the error.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.convexity.rules import (  # noqa: E402
    _recursion_headroom_need,
    _run_with_deep_recursion,
    classify_model,
)

pytestmark = pytest.mark.unit


def _deep_sum_model(n: int) -> dm.Model:
    """A model whose single nonlinear constraint is a depth-``n`` left-deep sum.

    Python's ``sum`` over ``n`` Expression terms builds a binary-add tree of
    depth ~``n``; with ``n`` well above the default 1000-frame recursion limit,
    the curvature walk would ``RecursionError`` without the headroom fix.
    """
    m = dm.Model("deep")
    x = m.continuous("x", shape=(n,), lb=-1.0, ub=1.0)
    # Bilinear terms keep the model nonconvex (so classification must walk the
    # whole graph rather than short-circuiting on an affine form).
    body = x[0] * x[1]
    for i in range(1, n - 1):
        body = body + x[i] * x[i + 1]
    m.minimize(body)
    m.subject_to(dm.sum([x[i] for i in range(n)]) >= 0.0)
    return m


def test_recursion_headroom_gate_small_model_is_zero():
    """Small models keep the default limit (no thread, no headroom)."""
    m = _deep_sum_model(20)
    assert _recursion_headroom_need(m) == 0


def test_recursion_headroom_scales_for_large_model():
    m = _deep_sum_model(2000)
    need = _recursion_headroom_need(m)
    assert need > 1000


def test_run_with_deep_recursion_runs_inline_when_not_needed():
    sentinel = object()
    assert _run_with_deep_recursion(lambda: sentinel, depth_need=10) is sentinel


def test_run_with_deep_recursion_allows_deep_call():
    """A recursion deeper than the default limit completes under the runner."""
    import sys

    target = sys.getrecursionlimit() + 4000

    def recurse(d: int) -> int:
        if d <= 0:
            return 0
        return 1 + recurse(d - 1)

    depth = sys.getrecursionlimit() + 2000
    result = _run_with_deep_recursion(lambda: recurse(depth), depth_need=target)
    assert result == depth


def test_run_with_deep_recursion_propagates_exceptions():
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        _run_with_deep_recursion(boom, depth_need=10**6)


def test_classify_deep_graph_does_not_raise(monkeypatch):
    """A graph deeper than the recursion limit classifies, not RecursionError.

    pytest raises the interpreter limit well above the production default, so to
    keep the test small and fast we lower the limit to below the model's
    expression depth (simulating production) and drop the size gate so the
    headroom path engages on a modest model. Without the fix the worker walk
    would ``RecursionError`` at the lowered limit; with it the classify completes.
    """
    import sys

    from discopt._jax.convexity import rules

    monkeypatch.setattr(rules, "_DEEP_RECURSION_SIZE_GATE", 50)
    n = 600  # expression depth ~600
    m = _deep_sum_model(n)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(400)  # below the ~n-deep walk; the runner must raise it
    try:
        is_convex, mask = classify_model(m, use_certificate=False)
    finally:
        sys.setrecursionlimit(old)
    assert is_convex is False
    assert len(mask) == len(m._constraints)


def test_quadratic_fallback_runs_once_per_maximal_region(monkeypatch):
    """The whole-expression quadratic fallback must fire O(1) times on a deep
    additive polynomial, not once per prefix node (issue #266 super-linear cost).

    A depth-n additive quadratic is a single maximal polynomial subtree, so the
    eigendecomposition fallback should run a small constant number of times,
    independent of n — not ~n times.
    """
    import sys

    from discopt._jax.convexity import patterns

    calls = {"n": 0}
    real = patterns.quadratic_curvature

    def counting(expr, model):
        calls["n"] += 1
        return real(expr, model)

    monkeypatch.setattr(patterns, "quadratic_curvature", counting)

    # Use enough depth that a per-prefix fallback would be unmistakably >n, but
    # raise the recursion limit so the count — not the ambient interpreter limit
    # (CI's default 1000 is below this depth) — is what the test measures.
    n = 400
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        m = _deep_sum_model(n)
        classify_model(m, use_certificate=False)
    finally:
        sys.setrecursionlimit(old)
    # One maximal region for the objective body + one for the affine constraint
    # row; allow generous slack but assert it is NOT proportional to n.
    assert calls["n"] <= 8, f"quadratic_curvature called {calls['n']} times for n={n}"


def test_deep_pure_continuous_solve_does_not_return_error():
    """A convexity-unknown deep continuous model returns a sound status, not error.

    With a tight time limit the convexity walk may abandon to unknown; the solve
    must still route to the spatial B&B and return a valid status (optimal /
    feasible / time_limit / infeasible) rather than ``status="error"``.

    The recursion limit is raised for the duration: the relaxation/FBBT walkers
    on the (deliberately deep) synthetic expression are not headroom-protected
    like the convexity walk, and CI's default 1000-frame limit is below this
    depth. Real models reach this routing path via many shallow constraints, not
    one giant expression, so this only affects the synthetic stress model.
    """
    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        m = _deep_sum_model(600)
        r = m.solve(time_limit=20.0)
    finally:
        sys.setrecursionlimit(old)
    assert r.status != "error", f"got error result: {getattr(r, '_explanation', None)}"
    assert r.status in ("optimal", "feasible", "time_limit", "infeasible", "limit")
