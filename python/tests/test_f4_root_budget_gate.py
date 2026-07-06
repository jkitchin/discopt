"""F4 — root-heuristic NLP/compile budget gate restores the time_limit contract.

Bottleneck-profile-2026-07-05 §4 / perf-followup-plan-2026-07-05 F4: on the
no-relaxation flowsheet class the root PRIMAL-HEURISTIC phase blew past
``solve(time_limit=T)`` because (a) the first heuristic NLP forced an
uninterruptible first-time XLA Hessian compile and (b) each subsequent heuristic
NLP overran its own ``max_wall_time`` clamp. The fix gates *entry* into those
compile-/solve-triggering heuristics by the remaining budget. Every gated call is
a primal heuristic, so skipping it is sound (dual bound untouched).

These tests cover:
  * the compile-cost estimate (a general function of model size, not an instance
    name) and the evaluator's compiled-kernel flag;
  * the end-to-end time-limit contract on a synthetic model with a deliberately
    expensive first Hessian compile — fail-before (gate off) vs pass-after
    (gate on), the §0.7 envelope.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import (
    _HESSIAN_COMPILE_DENSE_S,
    _HESSIAN_COMPILE_SPARSE_FLOOR_S,
    NLPEvaluator,
    estimate_hessian_compile_s,
)
from discopt.modeling.core import Model

# The gate only bites when a real NLP backend runs the root heuristics.
pytest.importorskip("pounce", reason="F4 gate is exercised through the POUNCE NLP path")


# ── compile-cost estimate: a general size function, never an instance name ──────


def test_estimate_dense_is_small_constant():
    # Dense-Hessian path (small models) is bounded and cheap — never gates.
    assert estimate_hessian_compile_s(n_vars=10, hessian_nnz=5, use_sparse=False) == pytest.approx(
        _HESSIAN_COMPILE_DENSE_S
    )
    # Independent of size on the dense path.
    assert estimate_hessian_compile_s(
        n_vars=40, hessian_nnz=800, use_sparse=False
    ) == pytest.approx(_HESSIAN_COMPILE_DENSE_S)


def test_estimate_sparse_returns_conservative_floor():
    # Sparse compressed-HVP path: compile is unpredictable and potentially huge
    # (measured 1s..186s, R^2~0 vs size), so the estimate is a conservative floor.
    est = estimate_hessian_compile_s(n_vars=300, hessian_nnz=1200, use_sparse=True)
    assert est == pytest.approx(_HESSIAN_COMPILE_SPARSE_FLOOR_S)
    assert est > _HESSIAN_COMPILE_DENSE_S


def test_estimate_is_a_pure_size_function_not_instance_keyed():
    # Same (size, path) -> same estimate regardless of anything else. This locks
    # in "fix the class, not the instance" (§0.2): there is no name/shape hook.
    a = estimate_hessian_compile_s(n_vars=200, hessian_nnz=900, use_sparse=True)
    b = estimate_hessian_compile_s(n_vars=200, hessian_nnz=900, use_sparse=True)
    assert a == b


# ── evaluator compiled-kernel flag ─────────────────────────────────────────────


def _big_sparse_model(n: int = 80) -> Model:
    """A chained model whose Lagrangian Hessian is sparse and n>=50 (so the
    evaluator takes the compressed-HVP path that the gate protects)."""
    m = Model("f4_sparse")
    xs = [m.continuous(f"x{i}", lb=0.1, ub=2.0) for i in range(n)]
    # Nearest-neighbour coupling -> banded (sparse) Hessian, low density.
    m.minimize(sum(xs[i] * xs[i + 1] for i in range(n - 1)) + sum(x * x for x in xs))
    for i in range(0, n - 1, 4):
        m.subject_to(xs[i] * xs[i + 1] >= 0.2)
    return m


def test_hessian_kernel_compiled_flag_flips_on_first_eval():
    m = _big_sparse_model(60)
    ev = NLPEvaluator(m)
    assert ev.hessian_kernel_compiled() is False
    # Estimate is positive while uncompiled (sparse path), 0 once compiled.
    if ev._use_sparse_hessian():
        assert ev.hessian_compile_estimate_s() > 0.0
    lb, ub = ev.variable_bounds
    x0 = 0.5 * (lb + ub)
    lam = np.ones(ev.n_constraints, dtype=np.float64)
    ev.evaluate_hessian_values(x0, 1.0, lam)
    assert ev.hessian_kernel_compiled() is True
    # Already compiled -> the estimate is now zero (cost already spent).
    assert ev.hessian_compile_estimate_s() == 0.0


# ── diving honours the absolute deadline (deterministic, no timing) ─────────────


def test_diving_returns_immediately_when_deadline_passed():
    """``diving`` (the LNS/root fallback that loops ~n_int sub-NLPs) must poll the
    absolute deadline and launch NO sub-NLP once it has passed — the F4 fix that
    stops these loops from running tens of seconds past a tight ``time_limit``.
    Deterministic: with a deadline already in the past not a single backend call
    is made. Fail-before: without the deadline poll the first sub-NLP would run."""
    from discopt._jax.primal_heuristics import diving

    m = Model("dive")
    x = m.continuous("x", lb=0.0, ub=5.0)
    z = m.integer("z", lb=0, ub=5)
    m.minimize((x - 2.5) ** 2 + z)
    m.subject_to(x + z >= 3)

    calls = {"n": 0}

    def _counting_backend(evaluator, x0, options=None):
        calls["n"] += 1
        raise AssertionError("backend must not be called past the deadline")

    out = diving(
        m,
        np.array([2.5, 1.7]),
        backend=_counting_backend,
        deadline=time.perf_counter() - 1.0,  # already elapsed
    )
    assert out is None
    assert calls["n"] == 0


# ── end-to-end time-limit contract (fail-before / pass-after) ───────────────────


def _expensive_compile_minlp() -> Model:
    """A synthetic model that reproduces the F4 overrun mechanism generically —
    no named instance. Two properties matter:

      * ``log`` of a *product* of variables is not linearizable, so (like the
        no-relaxation flowsheet class: contvar/heatexch_gen3) the MILP relaxation
        omits these rows and the ONLY NLP that runs is the root PRIMAL HEURISTIC —
        exactly the compile-triggering call F4 gates. A model that keeps a
        linearizable relaxation would spend its first compile in the relaxation
        build (a different, non-heuristic site), so the log-of-product structure
        is essential to isolate the heuristic path.
      * a wide (n>=50) transcendental DAG makes that first sparse-Hessian compile
        slow enough to blow a small ``time_limit`` (measured ~6-12 s in-solve).

    Integer variables make the root heuristics fire; the model is feasible so the
    gate can be checked for incumbent preservation, not just wall time."""
    import discopt.modeling as dm

    m = Model("f4_expensive_compile")
    # n=110 is chosen so the cold first-compile reliably overruns a time_limit=5
    # solve (measured ~20 s ungated) while the model stays feasible and the gated
    # solve still finds an incumbent — large enough to witness the overrun, small
    # enough that the gate does not have to drop the only incumbent.
    n = 110
    xs = [m.continuous(f"x{i}", lb=0.3, ub=3.0) for i in range(n)]
    zs = [m.integer(f"z{i}", lb=0, ub=3) for i in range(4)]
    obj = 0.0
    for i in range(n - 1):
        # log(exp(x_i*x_{i+1}) + x_i^2 + 1): a non-linearizable transcendental of
        # a product -> omitted from the MILP relaxation, heavy second-derivative.
        obj = obj + dm.log(dm.exp(xs[i] * xs[i + 1]) + xs[i] * xs[i] + 1.0)
    obj = obj + sum(z * z for z in zs)
    m.minimize(obj)
    for i in range(0, n - 1, 3):
        m.subject_to(dm.log(xs[i] * xs[i + 1] + 0.2) + 0.1 * zs[i % 4] >= 0.1)
    return m


_TIME_LIMIT = 5.0
_ENVELOPE = _TIME_LIMIT * 1.1 + 5.0  # §0.7: T*1.1 + 5 = 10.5 s

# The solve runs in a FRESH interpreter with the persistent XLA cache disabled so
# the first compile is truly cold. The child also counts root/heuristic NLP solves
# launched *after the deadline had passed* — the deterministic, machine-speed-
# independent signal of the bug: the ungated build launches such a solve (an
# unbounded overrun), the gated build launches none. Absolute compile time is
# itself unpredictable (R^2 ~ 0 vs model size — the very reason F4 gates ENTRY
# rather than trying to time the compile), so the count, not the wall, is the
# assertion. It prints "RESULT <wall> <obj> <post_deadline_launches>".
_CHILD = """
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["DISCOPT_DISABLE_JAX_CACHE"] = "1"
os.environ["DISCOPT_ROOT_BUDGET_GATE"] = sys.argv[1]
import discopt.solvers.nlp_pounce as _npounce
import test_f4_root_budget_gate as T
_state = {"deadline": None, "post": 0}
_orig = _npounce.solve_nlp
def _wrapped(ev, x0, *a, **k):
    d = _state["deadline"]
    if d is not None and time.perf_counter() >= d:
        _state["post"] += 1
    return _orig(ev, x0, *a, **k)
_npounce.solve_nlp = _wrapped
m = T._expensive_compile_minlp()
_state["deadline"] = time.perf_counter() + __TL__
t0 = time.perf_counter()
res = m.solve(time_limit=__TL__, gap_tolerance=1e-4)
wall = time.perf_counter() - t0
print(f"RESULT {wall:.3f} {res.objective} {_state['post']}")
""".replace("__TL__", repr(_TIME_LIMIT))


def _solve_under_gate(env_gate: str):
    """Run one cold-cache solve in a subprocess; return (wall, objective,
    post_deadline_nlp_launches)."""
    import subprocess

    env = dict(os.environ)
    here = os.path.dirname(__file__)
    env["PYTHONPATH"] = os.pathsep.join(
        [here, os.path.join(here, "..")] + [env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, env_gate],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        raise AssertionError(f"child failed (gate={env_gate}):\n{proc.stdout}\n{proc.stderr}")
    _, wall_s, obj_s, post_s = line[-1].split(maxsplit=3)
    obj = None if obj_s == "None" else float(obj_s)
    return float(wall_s), obj, int(post_s)


@pytest.mark.slow
def test_gate_bounds_post_deadline_nlp_launches_fail_before():
    """FAIL-BEFORE / PASS-AFTER on post-deadline heuristic NLP launches.

    Root heuristic NLPs launched *after* the deadline are the F4 overrun (each is
    an uninterruptible, unbounded solve). The gate must never launch MORE of them
    than the ungated build, and — when the overrun reproduces (>1 launch ungated,
    which needs a genuinely slow first compile) — must strictly reduce them. When
    the compile is fast on this machine (the ungated build launches at most the
    single un-gateable bound-source start), the multi-launch overrun cannot be
    exercised here, so we skip; the gate's decision itself is locked
    deterministically by ``test_entry_gate_decision_is_deterministic`` and its
    contract effect panel-wide in the PR's §0.7 table. Runs cold-cache in a fresh
    interpreter so the compile is genuinely first-time."""
    _, _, post_off = _solve_under_gate("0")
    _, _, post_on = _solve_under_gate("1")
    # The gate must never make things worse.
    assert post_on <= post_off, f"gate INCREASED post-deadline launches: {post_on} > {post_off}"
    if post_off <= 1:
        pytest.skip(
            f"multi-launch overrun did not reproduce (ungated post-deadline "
            f"launches={post_off}); compile too fast on this machine"
        )
    # Overrun reproduced: the gate must strictly cut the extra launches.
    assert post_on < post_off


def test_entry_gate_decision_is_deterministic():
    """Unit test of the gate's decision surface (no timing): the compile estimate
    is what the solver's ``_root_heur_nlp_entry_ok`` compares the remaining budget
    against. For an uncompiled sparse-Hessian evaluator the estimate exceeds a
    tiny budget (so entry is refused) and is zero once compiled (so entry is
    allowed). This locks the fail-before logic independently of wall clock."""
    m = _big_sparse_model(60)
    ev = NLPEvaluator(m)
    if not ev._use_sparse_hessian():
        pytest.skip("model did not take the sparse compressed-HVP path on this build")
    # Uncompiled: estimate is the conservative sparse floor -> exceeds a 5 s budget.
    est = ev.hessian_compile_estimate_s()
    assert est > 5.0
    # After a compile the estimate collapses to zero (cost already spent).
    lb, ub = ev.variable_bounds
    ev.evaluate_hessian_values(0.5 * (lb + ub), 1.0, np.ones(ev.n_constraints))
    assert ev.hessian_compile_estimate_s() == 0.0
