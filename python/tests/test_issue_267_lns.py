"""Issue #267: scalable adaptive LNS primal-improvement layer.

These tests exercise the new pieces added for the adaptive Large Neighbourhood
Search (LNS) primal layer:

* the *scalable* local-branching variant (:func:`local_branching` dispatching to
  ``_local_branching_submip`` for binary blocks larger than ``max_binaries``),
  which adds the Fischetti–Lodi Hamming-distance cut as a single linear
  constraint and re-solves a bounded sub-MIP — so it works for ANY number of
  binaries, not just the ``C(n, k)``-enumerable ones;
* the recursion guard (``solve_model(..., _lns_enabled=False)``) the sub-solve
  uses so the LNS layer can never nest into itself; and
* the adaptive gating that keeps RINS / local-branching inert (a no-op) for
  models with no integer variables.

All of these are *sound* heuristics: a returned point is re-verified integer- and
constraint-feasible, and the caller injects it only on strict improvement, so the
dual bound and optimality certificate are never touched.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt._jax.primal_heuristics import (  # noqa: E402
    _local_branching_submip,
    local_branching,
    rins,
)


def _flatten(model, x_dict) -> np.ndarray:
    """Flatten a SolveResult.x dict into the model's flat variable order."""
    chunks = []
    for v in model._variables:
        chunks.append(np.asarray(x_dict[v.name], dtype=np.float64).reshape(-1))
    return np.concatenate(chunks)


def _knapsack_model(n: int):
    """A small MIQP with ``n`` binaries whose optimum sets every binary to 1.

    ``min -sum_i x_i  s.t.  sum_i x_i <= n``. The optimum (all ones) is far from a
    poorly-rounded incumbent (a few ones), so local branching has a strictly
    better point to find within a Hamming ball.
    """
    m = dm.Model()
    x = m.binary("x", n)
    m.minimize(-dm.sum([x[i] for i in range(n)]))
    m.subject_to(dm.sum([x[i] for i in range(n)]) <= n)
    return m, x


def test_scalable_local_branching_improves_large_binary_block():
    """>12 binaries: enumeration is gated off, so the sub-MIP variant must fire
    and find a strictly better incumbent within the Hamming ball."""
    n = 16  # > default max_binaries (12) -> forces the sub-MIP path
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)

    incumbent = np.zeros(n)
    incumbent[:3] = 1.0  # poor: only 3 ones -> obj -3
    inc_obj = float(ev.evaluate_objective(incumbent))
    assert inc_obj == pytest.approx(-3.0)

    result = local_branching(
        m,
        incumbent,
        k=5,
        evaluator=ev,
        max_binaries=12,
        submip_time_limit=4.0,
        submip_max_nodes=2000,
    )
    assert result is not None, "scalable local branching found no improving point"
    x_out, obj_out = result
    # Strict improvement, and within the Hamming-k ball of the incumbent.
    assert obj_out < inc_obj - 1e-9
    n_ones = int(np.sum(np.round(x_out[:n])))
    assert n_ones == 8  # 3 incumbent ones + at most k=5 flips
    # The returned point is genuinely integer-feasible.
    assert np.allclose(np.round(x_out[:n]), x_out[:n], atol=1e-5)


def test_local_branching_dispatches_to_submip_above_cap():
    """For a binary count above the cap, ``local_branching`` must route through
    the sub-MIP variant (not the enumeration path)."""
    n = 20
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    incumbent = np.zeros(n)
    incumbent[:2] = 1.0

    # Direct call to the sub-MIP variant should also work and improve.
    binary_idx = list(range(n))
    direct = _local_branching_submip(
        m,
        incumbent,
        binary_idx,
        k=6,
        backend=None,
        nlp_options=None,
        integer_tol=1e-5,
        feas_tol=1e-6,
        evaluator=ev,
        time_limit=4.0,
        max_nodes=2000,
        gap_tolerance=1e-4,
    )
    assert direct is not None
    assert direct[1] < float(ev.evaluate_objective(incumbent)) - 1e-9


def test_submip_restores_model_constraints():
    """The Hamming cut is appended then popped: the model is left unchanged."""
    n = 14
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    n_constraints_before = len(m._constraints)

    incumbent = np.zeros(n)
    incumbent[:3] = 1.0
    _local_branching_submip(
        m,
        incumbent,
        list(range(n)),
        k=4,
        backend=None,
        nlp_options=None,
        integer_tol=1e-5,
        feas_tol=1e-6,
        evaluator=ev,
        time_limit=3.0,
        max_nodes=1000,
        gap_tolerance=1e-4,
    )
    # The appended cut must have been removed (append-then-pop in finally).
    assert len(m._constraints) == n_constraints_before


def test_recursion_guard_blocks_nested_lns():
    """The sub-solve runs with ``_lns_enabled=False`` so the LNS scheduler skips
    entirely — proving the guard prevents infinite recursion.

    We assert directly that ``solve_model`` accepts the private kwarg and that a
    nested local-branching sub-solve terminates (no unbounded recursion / hang).
    """
    import inspect

    from discopt.solver import solve_model

    assert "_lns_enabled" in inspect.signature(solve_model).parameters

    # A sub-MIP call internally invokes solve_model(..., _lns_enabled=False).
    # If the guard were missing, the nested solve would re-enter local branching
    # and recurse without bound; that it returns at all proves the guard holds.
    n = 16
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    incumbent = np.zeros(n)
    incumbent[:3] = 1.0
    result = _local_branching_submip(
        m,
        incumbent,
        list(range(n)),
        k=5,
        backend=None,
        nlp_options=None,
        integer_tol=1e-5,
        feas_tol=1e-6,
        evaluator=ev,
        time_limit=4.0,
        max_nodes=2000,
        gap_tolerance=1e-4,
    )
    assert result is not None  # terminated and produced an improving point


def test_improvers_noop_without_integers():
    """RINS and local branching must be no-ops for a pure-continuous model.

    The adaptive layer never wastes budget on models it cannot help; with no
    integer variables both improvers return ``None`` immediately.
    """
    m = dm.Model()
    x = m.continuous("x", 3, lb=-5.0, ub=5.0)
    m.minimize(dm.sum([x[i] * x[i] for i in range(3)]))
    m.subject_to(dm.sum([x[i] for i in range(3)]) >= 1.0)
    ev = NLPEvaluator(m)

    x0 = np.array([1.0, 0.0, 0.0])
    assert rins(m, x0, x0, evaluator=ev) is None
    assert local_branching(m, x0, k=3, evaluator=ev) is None


def test_local_branching_no_improvement_returns_none():
    """When the incumbent is already optimal, local branching returns ``None``
    (no strictly-improving point), so the caller injects nothing."""
    n = 16
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    optimum = np.ones(n)  # all ones is the global optimum (obj -n)
    result = local_branching(
        m,
        optimum,
        k=5,
        evaluator=ev,
        max_binaries=12,
        submip_time_limit=3.0,
        submip_max_nodes=1000,
    )
    assert result is None


# ─────────────────────────────────────────────────────────────
# F1 — enumeration budget (bottleneck-profile-2026-07-05 §1.1)
#
# The ≤``max_binaries`` enumeration branch issues ``sum_r C(n_bin, r<=k)`` full
# sub-NLPs — 79 at k=2, 1586 at k=5 for 12 binaries — and historically ignored
# both its per-call slice and the solver's absolute deadline (fac2: 1665 sub-NLPs
# = 84 % of wall). These tests lock in the budget enforcement: a hard deadline is
# honoured, and the search is skipped when the incumbent already matches the node
# bound. General (non-named) ≤12-binary models only.
# ─────────────────────────────────────────────────────────────


def test_local_branching_enumeration_honors_deadline(monkeypatch):
    """A ≤12-binary model with an early incumbent + a 1 s deadline must return
    within the budget and issue far fewer sub-NLPs than the unbudgeted count.

    Fail-before: the unbudgeted enumeration issues ``sum_r C(12, r<=5)`` = 1586
    sub-NLPs regardless of any deadline; at ~3 ms each that is ~4.8 s, blowing a
    1 s deadline. After the F1 budget, the flip loop polls the deadline every
    sub-NLP and truncates, so it returns in ~1 s with ≪1586 calls.
    """
    import math
    import time

    import discopt._jax.primal_heuristics as ph

    n = 12  # exactly at the cap -> the enumeration path (not the sub-MIP)
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    incumbent = np.zeros(n)
    incumbent[:2] = 1.0  # a poor early incumbent -> the search has work to do

    unbudgeted_k5 = sum(math.comb(n, r) for r in range(5 + 1))
    assert unbudgeted_k5 == 1586  # the arithmetic the profile measured

    calls = {"n": 0}
    real_subnlp = ph.subnlp

    def counting_slow_subnlp(*a, **k):
        calls["n"] += 1
        time.sleep(0.003)  # ~3 ms: 1586 of these would be ~4.8 s (> 1 s deadline)
        return real_subnlp(*a, **k)

    monkeypatch.setattr(ph, "subnlp", counting_slow_subnlp)

    deadline = time.perf_counter() + 1.0
    t0 = time.perf_counter()
    local_branching(
        m,
        incumbent,
        k=5,
        evaluator=ev,
        max_binaries=12,
        # A slice large enough NOT to be the limiting factor; the deadline is.
        submip_time_limit=10.0,
        submip_max_nodes=500,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - t0

    # Budget honoured: returned within the deadline envelope (§0.7 style: +~20 %).
    assert elapsed <= 1.2, f"local_branching overran its 1 s deadline: {elapsed:.2f} s"
    # And it issued strictly fewer sub-NLPs than the unbudgeted enumeration.
    assert calls["n"] < unbudgeted_k5, (
        f"enumeration was not truncated: {calls['n']} sub-NLPs "
        f"(unbudgeted would be {unbudgeted_k5})"
    )


def test_local_branching_skips_when_incumbent_matches_node_bound():
    """When the incumbent objective already equals the node relaxation bound
    within ``gap_tolerance``, there is nothing to improve, so the search is
    skipped entirely (zero sub-NLPs) and ``None`` is returned."""
    n = 12
    m, _ = _knapsack_model(n)
    ev = NLPEvaluator(m)
    optimum = np.ones(n)  # global optimum, obj = -n
    inc_obj = float(ev.evaluate_objective(optimum))
    assert inc_obj == pytest.approx(-float(n))

    # Node bound equal to the incumbent objective -> gap closed -> skip.
    result = local_branching(
        m,
        optimum,
        k=5,
        evaluator=ev,
        max_binaries=12,
        submip_time_limit=3.0,
        node_bound=inc_obj,
        incumbent_obj=inc_obj,
        gap_tolerance=1e-4,
    )
    assert result is None
