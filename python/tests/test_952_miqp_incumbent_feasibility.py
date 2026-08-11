"""Issue #952: the MIQP B&B path must verify the incumbent it returns.

Background
----------
``_solve_miqp_bb``'s only feasibility gate was
``_check_lp_solution_feasibility(A_eq_full, b_eq, x_full)`` — an equality residual
at ``tol=1e-4``, with no inequality-row check, no bound check, and no verification
of the point whose objective is returned as both ``objective`` and ``bound`` on an
``optimal`` status.

That gate was a **tautology** on the path that actually runs. The batched
structured-QP node solver (``_pounce_qp_relaxation_nodes``) solves only the
structural columns and then *reconstructs* the slacks as
``z = S⁺(b_eq - A_struct x_s)``, which makes ``A_eq_full [x_s, z] == b_eq`` hold to
machine precision for **any** ``x_s``: a violated inequality row comes back as a
negative slack, and the gate never looked at slack bounds. Measured on `main`
@ 8bfce1f1 over the 40-seed family below: 212 gate invocations, worst equality
residual **8.9e-16**, while every returned incumbent sat ~9e-9 outside a declared
inequality row.

The excursions were small — inside the repo's declared ``abs=1e-6`` — so this was
never a live false certificate. The defect is that *nothing bounded them*: their
size was set by whatever the QP IPM converged to, and the gate that admitted them
would equally have admitted ``1e-4``.

What is pinned here
-------------------
1. :func:`test_returned_incumbent_is_within_declared_tolerance` — the 40-seed panel,
   asserting the returned point is inside every declared row and bound at the
   declared ``abs=1e-6``, with an executed-comparison count so it cannot pass by
   measuring nothing (CLAUDE.md §6).
2. :func:`test_old_equality_only_gate_was_a_tautology` — the mechanism, pinned
   directly: a point 1e-3 outside an inequality row still satisfies the
   equality-plus-reconstructed-slack form exactly, so the retired gate rated it
   feasible and the arbiter rejects it.
3. :func:`test_exit_gate_refuses_an_off_row_incumbent` — the gate FIRES: an
   incumbent pushed off a tight row is refused loudly rather than returned as
   ``optimal``. This is the test that fails before the fix.
4. :func:`test_milp_exit_gate_refuses_an_off_row_incumbent` (with
   :func:`test_milp_baseline_solve_is_unaffected` as its control) — the same gate on
   ``_solve_milp_bb``, whose incumbent exit was structurally identical (round,
   unpack, return) with nothing verifying the returned point. Fixing only the MIQP
   path would have been a single-instance fix of a per-path defect (CLAUDE.md §2);
   the same shape on the *dual* side is tracked in #933.
5. :func:`test_bounds_check_vectorisation_matches_the_original_loop` — the
   arbiter's bounds check was vectorised for the per-node call site; this pins that
   it did not change semantics.
6. :func:`test_injection_funnel_declines_an_off_row_heuristic_incumbent` — the
   funnel, not just the exit. ``_pounce_snap_incumbent`` fixes the integers and
   takes the POUNCE IPM's continuous completion, which honours the slack bounds
   standing in for inequality rows only to ~1e-8 *relative*; on a big-M row that is
   a 1e-2 absolute excursion, delivered as an exactly-integral point that the MILP
   funnel used to wave through and the exit gate then refused the whole solve over.
   The MIQP funnel already checked this; the MILP one now does too.
"""

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S

# The repo's declared absolute feasibility tolerance (see CLAUDE.md "Key
# Constraints" and conftest); the gate is held to this, not to the retired 1e-4.
DECLARED_ABS_TOL = 1e-6


def _panel_model(seed: int) -> dm.Model:
    """The issue's family: a convex MIQP that dispatches to ``_solve_miqp_bb``.

    ``min Σ aⱼxⱼ² + Σ cᵢyᵢ`` s.t. ``Σxⱼ ≥ tgt``, ``xⱼ ≤ 4y_{j mod 2}``, ``y``
    binary, ``x ∈ [0,4]³`` — extended from ``test_946_gbd_degenerate_multiplier``'s
    4 seeds to 40. The ``Σxⱼ ≥ tgt`` row is active at every optimum, which is what
    makes the excursion observable.
    """
    rng = np.random.default_rng(seed)
    ny, nx = 2, 3
    m = dm.Model("panel952")
    y = m.binary("y", shape=(ny,))
    x = m.continuous("x", shape=(nx,), lb=0, ub=4)
    a = rng.uniform(0.5, 2.0, nx)
    cy = rng.uniform(1, 3, ny)
    tgt = rng.uniform(2, 5)
    m.minimize(
        sum(float(a[j]) * x[j] * x[j] for j in range(nx))
        + sum(float(cy[i]) * y[i] for i in range(ny))
    )
    m.subject_to(sum(x[j] for j in range(nx)) >= tgt)
    for j in range(nx):
        m.subject_to(x[j] <= 4 * y[j % ny])
    return m


def _flatten(model: dm.Model, x_dict: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(x_dict[v.name], dtype=np.float64).flatten() for v in model._variables]
    )


def _worst_violation(model: dm.Model, x_dict: dict) -> tuple[float, int]:
    """Max violation of the returned point over every declared row AND bound.

    Measured through ``NLPEvaluator.evaluate_constraints`` against the model as
    declared — deliberately not through the solver's own helpers, so this test
    cannot inherit the defect it is checking for. Returns ``(worst, n_checked)``;
    ``n_checked`` is the executed-comparison count (CLAUDE.md §6).
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    x = _flatten(model, x_dict)
    ev = NLPEvaluator(model)
    cl, cu = (np.asarray(b, dtype=np.float64) for b in _infer_constraint_bounds(ev))
    cons = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
    n = min(len(cons), len(cl))
    assert n > 0, "no constraint rows evaluated: this probe would measure nothing"
    lo, hi = ev.variable_bounds
    viols = np.concatenate([cons[:n] - cu[:n], cl[:n] - cons[:n], x - hi, lo - x])
    return float(np.max(viols)), int(2 * n + 2 * len(x))


@pytest.mark.correctness
@pytest.mark.slow
def test_returned_incumbent_is_within_declared_tolerance():
    """The 40-seed panel: no returned incumbent sits outside a declared row or
    bound by more than the declared absolute tolerance.

    This is the standing watch. It passed before the fix too — today's excursions
    are ~9e-9 — because the point was never to catch a live false certificate but
    to bound something that nothing bounded. ``test_exit_gate_refuses_an_off_row_
    incumbent`` is the one that fails without the fix.
    """
    comparisons = 0
    checks = 0
    worst = (-np.inf, -1)

    for seed in range(40):
        m = _panel_model(seed)
        r = m.solve(time_limit=60)
        assert r.status == "optimal", f"seed {seed}: status {r.status}, panel assumes optimal"
        assert r.x is not None
        viol, n_checked = _worst_violation(m, r.x)
        comparisons += 1
        checks += n_checked
        if viol > worst[0]:
            worst = (viol, seed)

    # §6: prove the probe fired rather than skipping every seed.
    assert comparisons == 40, f"only {comparisons}/40 seeds compared"
    assert checks == 40 * (2 * 4 + 2 * 5), f"unexpected comparison count {checks}"
    assert worst[0] <= DECLARED_ABS_TOL, (
        f"seed {worst[1]} returned a point {worst[0]:.6e} outside a declared row/bound, "
        f"beyond the declared abs tolerance {DECLARED_ABS_TOL:.0e}"
    )


def test_old_equality_only_gate_was_a_tautology():
    """Pin the mechanism, not just the symptom.

    Build the slack form of ``x ≤ 1`` and a point ``x = 1 + 1e-3`` that violates it.
    Reconstructing the slack the way ``_pounce_qp_relaxation_nodes`` does gives
    ``s = -1e-3``, so ``A_eq_full [x, s] == b_eq`` holds *exactly* — the retired
    equality-only gate (``tol=1e-4``) rated this feasible. The arbiter does not.
    """
    A_eq_full = np.array([[1.0, 1.0]])  # x + s == 1, s >= 0  <=>  x <= 1
    b_eq_full = np.array([1.0])
    x_s = np.array([1.0 + 1e-3])
    s = b_eq_full - A_eq_full[:, :1] @ x_s  # the pinv reconstruction, 1-D case
    x_full = np.concatenate([x_s, s])

    # What the retired gate computed, reproduced inline (it no longer exists).
    residual = float(np.max(np.abs(A_eq_full @ x_full - b_eq_full)))
    assert residual <= 1e-4, "the reconstructed slack should satisfy the equality exactly"
    assert residual < 1e-12, f"expected machine-zero residual, got {residual:.3e}"

    # The arbiter, over the decomposed inequality row, rejects the same point.
    A_ub, b_ub = np.array([[1.0]]), np.array([1.0])
    assert not S._matrix_solution_feasible(x_s, A_ub, b_ub, None, None, np.array([[0.0, 10.0]])), (
        "a point 1e-3 outside its inequality row must not pass the arbiter"
    )

    # And the slack it implies is outside the slack's own [0, inf) bound — the
    # thing the equality-only gate never looked at.
    assert s[0] < -1e-4


def _patch_offrow_tree(monkeypatch, cont_slice, shift):
    """Make the tree hand back an incumbent nudged off a tight row.

    Proxies ``PyTreeManager`` and perturbs only the vector returned by
    ``incumbent()``, so the search itself is untouched and the exit gate is the
    only thing under test. Returns a dict whose ``applied`` flag lets the caller
    prove the perturbation actually happened (CLAUDE.md §6) — without it a passing
    ``pytest.raises`` could be pinning some unrelated error.
    """
    real_tree_cls = S.PyTreeManager
    state = {"applied": False}

    class _OffRowTree:
        def __init__(self, *args, **kwargs):
            self._inner = real_tree_cls(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def incumbent(self):
            inc = self._inner.incumbent()
            if inc is None:
                return None
            sol, obj = inc
            sol = np.asarray(sol, dtype=np.float64).copy()
            block = sol[cont_slice]
            if np.all(np.isfinite(block)) and float(np.max(block)) > 1e-2:
                sol[cont_slice.start + int(np.argmax(block))] += shift
                state["applied"] = True
            return sol, obj

    monkeypatch.setattr(S, "PyTreeManager", _OffRowTree)
    return state


def test_exit_gate_refuses_an_off_row_incumbent(monkeypatch):
    """The MIQP gate fires: an incumbent off a declared row is refused, not returned.

    ``Σxⱼ ≥ tgt`` is active at the optimum, so pushing the largest ``x`` coordinate
    down by 1e-3 puts the incumbent 1e-3 outside that row while leaving it inside
    every variable bound and leaving the binaries integral. Before the fix this came
    back as ``optimal``; now it raises.
    """
    # Variable order is y (2 binaries) then x (3 continuous).
    state = _patch_offrow_tree(monkeypatch, slice(2, 5), -1e-3)

    m = _panel_model(0)
    with pytest.raises(RuntimeError, match="MIQP-BB returned an infeasible point labeled"):
        m.solve(time_limit=60)

    assert state["applied"], "the incumbent was never perturbed; the test proved nothing"


def test_milp_exit_gate_refuses_an_off_row_incumbent(monkeypatch):
    """The same gate on the MILP path, whose incumbent exit was structurally
    identical (round, unpack, return) with nothing verifying the returned point.

    ``x₀ + x₁ ≥ 3`` is active at the optimum, so nudging the larger ``x``
    coordinate down puts the point off that row while leaving the binary integral.
    """
    m = dm.Model("milp952")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.minimize(x[0] + 2 * x[1] + 3 * y)
    m.subject_to(x[0] + x[1] >= 3)
    m.subject_to(x[0] <= 4 * y)

    # Variable order is y (1 binary) then x (2 continuous).
    state = _patch_offrow_tree(monkeypatch, slice(1, 3), -1e-3)

    with pytest.raises(RuntimeError, match="MILP-BB returned an infeasible point labeled"):
        m.solve(time_limit=60)

    assert state["applied"], "the incumbent was never perturbed; the test proved nothing"


def test_milp_baseline_solve_is_unaffected():
    """Control for the test above: without the perturbation the same MILP solves
    normally, so the raise there is the gate firing and not the model being broken."""
    m = dm.Model("milp952")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.minimize(x[0] + 2 * x[1] + 3 * y)
    m.subject_to(x[0] + x[1] >= 3)
    m.subject_to(x[0] <= 4 * y)

    r = m.solve(time_limit=60)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(6.0, abs=1e-6)


def test_integer_snap_is_declined_when_it_would_leave_the_rows(monkeypatch):
    """The C-3 integer snap is adopted only if it keeps the point inside the rows.

    The exit gate found this: on `test_nn_equivalence::test_tree_ensemble_fixed_input`
    the incumbent satisfies its equalities to 4.4e-16 and the *snapped* point misses
    one by 1.55e-6, from per-coordinate snaps of at most 3.9e-7 over a 5-term row.
    The MILP call site's comment claimed the snap "cannot move a linear row by more
    than the integrality tol"; a row takes one snap per term, so it can.

    ``_round_incumbent_integers`` is documented to report ``feasible=False`` when
    rounding breaks feasibility, but only when handed a checker — and this call site
    passes none, so the flag was unconditionally True. Reproduced here in the small:
    five integers each 3e-7 under an integer, on a row with coefficient 2, moves the
    equality by 3e-6. The unrounded point satisfies *both* declared tolerances (rows
    exactly, integrality 3e-7 against 1e-5), so it is the one reported.
    """
    eps = 3e-7
    m = dm.Model("snap952")
    z = m.integer("z", shape=(5,), lb=0, ub=3)
    x = m.continuous("x", lb=-100, ub=100)
    m.minimize(x + sum(z[j] for j in range(5)))
    m.subject_to(2 * sum(z[j] for j in range(5)) + x == 10)

    z_near = np.full(5, 1.0 + eps)
    x_val = 10.0 - 2.0 * float(np.sum(z_near))  # equality holds exactly at z_near

    # Sanity, before relying on it: snapping really does break the row here, by more
    # than the declared abs tolerance. Without this the test could pass vacuously.
    assert abs(2 * np.sum(np.round(z_near)) + x_val - 10.0) > DECLARED_ABS_TOL

    real_tree_cls = S.PyTreeManager
    state = {"applied": False}

    class _NearIntegralTree:
        def __init__(self, *args, **kwargs):
            self._inner = real_tree_cls(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def incumbent(self):
            inc = self._inner.incumbent()
            if inc is None:
                return None
            _sol, obj = inc
            state["applied"] = True
            # Variable order is z (5 integers) then x.
            return np.concatenate([z_near, [x_val]]), obj

    monkeypatch.setattr(S, "PyTreeManager", _NearIntegralTree)

    r = m.solve(time_limit=60)  # must NOT raise: the unrounded point is feasible
    assert state["applied"], "the incumbent was never replaced; the test proved nothing"
    assert r.x is not None

    got = np.asarray(r.x["z"], dtype=np.float64).flatten()
    assert np.allclose(got, z_near, atol=0, rtol=0), (
        f"expected the unrounded incumbent {z_near} to be reported, got {got}"
    )


def test_bounds_check_vectorisation_matches_the_original_loop():
    """``_matrix_solution_feasible``'s bounds check was vectorised so the MIQP node
    gate can call it once per node. Semantics must be unchanged, including the
    ``zip``-style truncation when ``bounds`` is shorter than ``x``."""
    rng = np.random.default_rng(0)
    tol, rtol = 1e-6, 1e-9

    def original(x, bounds):
        for xi, (lo, hi) in zip(x, bounds):
            row_tol = tol + rtol * abs(xi)
            if xi < lo - row_tol or xi > hi + row_tol:
                return False
        return True

    checked = 0
    for _ in range(200):
        n = int(rng.integers(1, 6))
        x = rng.normal(0, 1e3, n)
        lo = x - rng.choice([0.0, 1e-9, 1e-7, 1e-5, 1.0], n)
        hi = x + rng.choice([0.0, 1e-9, 1e-7, 1e-5, 1.0], n)
        k = int(rng.integers(1, n + 1))  # bounds may be shorter than x
        bounds = list(zip(lo[:k].tolist(), hi[:k].tolist()))
        got = S._matrix_solution_feasible(x, None, None, None, None, bounds)
        assert got == original(x, bounds), f"divergence on x={x!r}, bounds={bounds!r}"
        checked += 1

    assert checked == 200, f"only {checked} comparisons executed"


@pytest.mark.parametrize("vub", [1e3, 1e6])
def test_injection_funnel_declines_an_off_row_heuristic_incumbent(vub, caplog):
    """An exactly-integral candidate outside a declared row must be declined where
    it is injected, not discovered at the exit gate.

    The reproducer is a two-row big-M master handed straight to the POUNCE-IPM
    matrix-MILP engine (``get_milp_solver(backend="pounce")`` ->
    ``_solve_milp_bb(prefer_pounce=True)``)::

        min  y + eta
        s.t. -M*y - eta <= 0          (a big-M row, slack at the optimum)
             -s*y - eta <= vub - s    (the binding row, rhs of order vub)
             y in {0,1},  eta >= -1e12

    The relaxation lands ``y`` a hair under 1, ``_pounce_snap_incumbent`` fixes
    ``y = 1`` and re-solves for ``eta``, and the IPM returns ``eta`` about
    ``1e-8*vub`` past the binding row — 1e-5 at ``vub = 1e3`` and 1e-2 at
    ``vub = 1e6``, four orders of magnitude outside the declared ``abs=1e-6``.
    Nothing downstream re-examines it (it is already integral), so it reached the
    exit gate as the answer and the whole solve died with ``MILP-BB returned an
    infeasible point labeled feasible/optimal``.

    Declining at the funnel is free — injection is a heuristic accelerator, the
    subtree stays open — so the solve completes and its answer is exact.

    This used to drive the same funnel through ``solve_benders`` on a linear
    model, i.e. through the classical-Benders master. #986 pinned that master to
    the exact-vertex simplex (an interior master point is not a valid Benders
    iterate and its objective is the reported dual bound), which removed the IPM
    path this test exists to exercise. The guard it pins is still needed for the
    ``_solve_milp_bb`` callers that do run on the IPM engine — the matrix-MILP
    seam used here, OA masters, GDP-LOA, ``milp_relaxation`` — so the test is
    re-pointed at one of those rather than deleted. The rows above are the shape
    the Benders master actually produced, and reproduce the funnel's violation
    magnitudes exactly (9.994994e-06 / 9.999995e-03).
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_backend import get_milp_solver

    milp = get_milp_solver(backend="pounce")
    big_m, slope = 3.0 * vub, 0.25
    A_ub = np.array([[-big_m, -1.0], [-slope, -1.0]])
    b_ub = np.array([0.0, vub - slope])

    with caplog.at_level(logging.DEBUG, logger="discopt.solver"):
        r = milp(
            np.array([1.0, 1.0]),
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0.0, 1.0), (-1e12, 1e20)],
            integrality=np.array([1, 0], dtype=np.int32),
            time_limit=60.0,
            gap_tolerance=1e-4,
        )

    declined = [
        rec.getMessage()
        for rec in caplog.records
        if "MILP-BB: rejected a snapped incumbent" in rec.getMessage()
    ]
    assert declined, (
        "the funnel never declined anything; this test would pass for the wrong "
        "reason (the off-row candidate is what it is meant to exercise)"
    )

    # And the solve still lands on the exact optimum (y=1, eta=-vub -> 1 - vub).
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(1.0 - vub, rel=1e-9)
