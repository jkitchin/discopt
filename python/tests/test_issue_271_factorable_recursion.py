"""Issue #271: deep expression graphs must not crash factorable reformulation
(or the solve it feeds) with an uncaught ``RecursionError``.

``_find_clearable_denominator`` (and the other factorable-reform expression
walkers, plus the relaxation compiler's node walk downstream) recurse one Python
frame per expression node. ``from_nl`` rebuilds a sum/division chain of N terms
as a left-deep binary tree of depth ~N, so a single constraint body of tens of
thousands of nodes (e.g. MINLPLib's watercontamination0202r ~53k,
graphpart_clique-30 ~7.6k) exceeded CPython's default 1000-frame recursion limit
and raised an *uncaught* ``RecursionError`` straight out of ``solve()``.

The fix mirrors the convexity walk's issue-#266 fix: the public factorable-reform
entry points (and the whole solve) run the deep walk with a size-scaled recursion
limit on a large-stack worker thread, gated so shallow models are unaffected; the
``_model_contains_custom_call`` walker was made explicit-stack iterative.

These tests are written to be robust to the *ambient* recursion limit (CI runs at
the default 1000; pytest may raise it locally). Two strategies, both valid, and
any explicit limit change is restored in a ``finally``:

* the cheap scans size the model relative to ``sys.getrecursionlimit()`` so the
  inner walk is always deeper than the ambient limit;
* the tests that run the *rewrite* lower the limit under a fixed-size model
  instead. Sizing those against the ambient limit makes their cost a function of
  the environment — the rewrite grows ~n^1.55 — which timed out the coverage
  lane at >120s on 2026-09-20.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax import factorable_reform  # noqa: E402
from discopt._relax.factorable_reform import (  # noqa: E402
    _find_clearable_denominator,
    _max_expr_node_count,
    _recursion_headroom_need,
    factorable_reformulate,
    has_clearable_denominator,
    has_factorable_work,
)

pytestmark = pytest.mark.unit


def _deep_division_model(n: int) -> dm.Model:
    """A model whose single constraint body is a depth-``n`` left-deep sum of
    sign-definite divisions ``x[i] / x[i+1]``.

    The variables are bounded strictly away from zero so every denominator is
    sign-definite and clearable — exercising the ``_find_clearable_denominator``
    recursion the bug was reported against. The chain of ``n`` ``+`` terms builds
    a binary-add tree of depth ~``n``; with ``n`` above the recursion limit the
    per-node walk would ``RecursionError`` without the headroom fix.
    """
    m = dm.Model("deep_div")
    x = m.continuous("x", shape=(n,), lb=1.0, ub=2.0)
    body = x[0] / x[1]
    for i in range(1, n - 1):
        body = body + x[i] / x[i + 1]
    m.subject_to(body <= float(n))
    m.minimize(dm.sum([x[i] for i in range(n)]))
    return m


def _depth_for_limit() -> int:
    """A node depth comfortably above the ambient recursion limit, so the inner
    (unwrapped) walk would overflow it but the wrapped entry points must not."""
    return sys.getrecursionlimit() + 1500


def test_headroom_gate_small_model_is_zero():
    """A shallow model keeps the default limit (no thread, no headroom)."""
    m = _deep_division_model(20)
    assert _recursion_headroom_need(m) == 0


def test_headroom_scales_for_deep_expression():
    """A deep single body trips the gate even with few constraints/variables.

    This is the watercontamination0202r failure mode: ~hundreds of constraints
    but one body of tens of thousands of nodes. The node-count gate must catch it
    where a constraint/variable count would not.
    """
    m = _deep_division_model(2000)
    assert _max_expr_node_count(m) > 1000
    assert _recursion_headroom_need(m) > sys.getrecursionlimit()


def test_inner_walker_recurses_past_default_limit():
    """Sanity: the bare recursive walker DOES overflow a limit below the depth.

    Guards the regression: if a future refactor makes the inner walk shallow (or
    iterative) this asserts the test model is still deep enough to be meaningful
    against the lowered limit used below.
    """
    n = 1200
    m = _deep_division_model(n)
    body = m._constraints[0].body
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        with pytest.raises(RecursionError):
            _find_clearable_denominator(body, m)
    finally:
        sys.setrecursionlimit(old)


def test_has_factorable_work_no_recursion_error():
    """The public scan completes on a deep graph instead of ``RecursionError``."""
    n = _depth_for_limit()
    m = _deep_division_model(n)
    # A sign-definite division chain is clearable work, so this is True — and it
    # must be reached without overflowing the ambient recursion limit.
    assert has_factorable_work(m) is True


def test_has_clearable_denominator_no_recursion_error():
    n = _depth_for_limit()
    m = _deep_division_model(n)
    assert has_clearable_denominator(m) is True


def test_factorable_reformulate_no_recursion_error():
    """The full reformulation pass completes on a deep graph and stays sound.

    Sized against a *lowered* limit rather than the ambient one. The claim —
    "a body deeper than the live recursion limit reformulates instead of
    ``RecursionError``" — is identical either way, but ``_depth_for_limit()``
    makes the model size a function of the environment, and the reformulation
    walk grows ~n^1.55 (measured: 0.10s at n=1200, 0.31s at n=2500, 0.56s at
    n=3500, x3.3 under ``--cov``). That is what timed out the coverage lane at
    >120s on 2026-09-20 while the same test took 1.3s locally. Lowering the
    limit instead fixes the cost at ~0.2s on every machine.

    The ambient-limit path is NOT given up: ``test_has_factorable_work_…`` and
    ``test_has_clearable_denominator_…`` still run at ``_depth_for_limit()``,
    and cost 0.03-0.06s there because only the rewrite is expensive. Unlike
    ``test_factorable_walk_under_lowered_limit`` below, this test does not
    monkeypatch ``_DEEP_RECURSION_SIZE_GATE``, so the *production* gate is what
    has to fire.
    """
    n = 700
    lowered = 400
    m = _deep_division_model(n)

    # §6: prove the setup is meaningful before trusting the result. Without
    # these the test degrades to "a small model reformulates", which says
    # nothing about #271 and would still report a pass.
    assert _max_expr_node_count(m) > factorable_reform._DEEP_RECURSION_SIZE_GATE, (
        "model is below the production size gate — the deep path would not engage"
    )
    assert _recursion_headroom_need(m) > lowered, (
        "headroom need does not exceed the lowered limit — nothing to test"
    )
    assert n > lowered, "body is shallower than the limit; the walk would not overflow"

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(lowered)
    try:
        out = factorable_reformulate(m)
    finally:
        sys.setrecursionlimit(old)
    # The pass returns a model (rewritten or, defensively, the original) — never
    # a crash. Clearing the sign-definite denominators changes the model.
    assert isinstance(out, dm.Model)


def test_factorable_walk_under_lowered_limit(monkeypatch):
    """With the size gate dropped and the limit lowered below the model depth, the
    headroom path must engage and classify, not ``RecursionError``.

    pytest can raise the interpreter limit well above production's default, so we
    simulate production by lowering the limit and dropping the gate onto a modest
    model — exactly the strategy of ``test_issue_266``'s deep-graph test.
    """
    from discopt._relax import factorable_reform

    monkeypatch.setattr(factorable_reform, "_DEEP_RECURSION_SIZE_GATE", 50)
    n = 700
    m = _deep_division_model(n)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(400)  # below the ~n-deep walk; the runner must raise it
    try:
        assert has_factorable_work(m) is True
        out = factorable_reformulate(m)
    finally:
        sys.setrecursionlimit(old)
    assert isinstance(out, dm.Model)


@pytest.mark.slow
def test_deep_division_solve_does_not_crash(monkeypatch):
    """A deep-body model solves to a sound status, not a ``RecursionError``.

    End-to-end: ``solve()`` routes through factorable reformulation and the
    relaxation compiler, both of which walk the deep body. The whole solve runs
    under size-scaled recursion headroom (``_scoped_deep_recursion``), so it must
    return a valid status rather than crashing.

    To keep the test fast and deterministic we simulate production rather than
    build a giant model: drop the solve-path depth gate onto a modest body and
    lower the recursion limit below that body's depth, so the headroom worker
    thread is what carries the solve (mirroring ``test_issue_266``'s deep-graph
    test). Without the fix the solve would ``RecursionError`` at the lowered
    limit; with it the solve completes with a sound status.
    """
    from discopt import solver as solver_mod

    monkeypatch.setattr(solver_mod, "_DEEP_SOLVE_DEPTH_GATE", 50)
    n = 250  # body depth ~250, comfortably above the lowered limit below
    m = _deep_division_model(n)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(150)  # below the ~n-deep walk; the runner must raise it
    try:
        r = m.solve(time_limit=15.0)
    finally:
        sys.setrecursionlimit(old)
    assert r.status != "error", f"got error result: {getattr(r, '_explanation', None)}"
    assert r.status in ("optimal", "feasible", "time_limit", "infeasible", "limit")
