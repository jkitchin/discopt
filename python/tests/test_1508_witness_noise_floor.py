"""Regression test for #1508: the #853 Lagrangian witness search fired on
ulp-level multiplier noise when the variable box is huge.

``nlp_cvx_001_010`` (min x over 5 linear rows) reaches the convex certificate with
a presolve-derived box of +-1.3e12. At the exact optimum the reduced gradient
``grad + J^T lam`` is a cancellation; a relative error of 1e-15 in ``(x, lam)``
leaves a residue ~1e-16 that the Frank-Wolfe step of length ~1e12 turns into a
Lagrangian "gain" of 1e-5..1e-2 -- far above the old floor ``1e-12 (1 + |L0|)``.
The witness then demanded the rigorous tangent bound, which is hopelessly loose on
that box, and the certificate was withheld (20/20 seeds at 1e-15 and 1e-13).

The floor now also scales with the cancellation magnitude of the linearized gap,
``sum_j |d_j| (|grad_j| + (|J|^T |lam|)_j)``, i.e. the gain a 1e-12 relative error
in ``(x, lam)`` could produce. Real witnesses (#853, #1499) stay far above it --
those suites are the kill criterion for this change.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest

# The presolve-derived box the solver hands the certificate on this instance
# (captured from ``solve(solver="amp", nlp_solver="ipm")``), fixed here so the
# test does not depend on presolve or on the NLP backend's exact output.
_LB = np.array([-1.28587963e11, -1.28587963e12])
_UB = np.array([3.85763889e12, 1.92881944e12])


def _model():
    m = dm.Model("nlp_cvx_001_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x)
    m.subject_to(x + y <= 5)
    m.subject_to(2 * x - y <= 3)
    m.subject_to(3 * x + 9 * y >= -10)
    m.subject_to(10 * x - y >= -20)
    m.subject_to(-x + 2 * y <= 8)
    return m


def _exact_kkt(ev, cl, cu):
    """The exact optimum (rows 2, 3 active) and its multipliers from the KKT system."""
    x = np.array([-190.0 / 93.0, -40.0 / 93.0])
    g = np.asarray(ev.evaluate_gradient(x), dtype=np.float64)
    J = np.asarray(ev.evaluate_jacobian(x), dtype=np.float64).reshape(5, 2)
    active = [2, 3]
    lam = np.zeros(5)
    lam[active] = np.linalg.solve(J[active].T, -g)
    # dual feasibility: each multiplier points at a finite side of its row
    for i in active:
        assert (lam[i] > 0 and cu[i] < 1e19) or (lam[i] < 0 and cl[i] > -1e19)
    return x, lam


@pytest.mark.parametrize("rho", [1e-15, 1e-13])
def test_ulp_noise_does_not_manufacture_a_witness(rho):
    from discopt.solver import _convex_nlp_certificate, _make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    ev = _make_evaluator(_model())
    cl, cu = _infer_constraint_bounds(ev)
    x0, lam0 = _exact_kkt(ev, cl, cu)

    executed = 0
    withheld = []
    why: list = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = x0 * (1.0 + rho * rng.standard_normal(2))
        lam = lam0 * (1.0 + rho * rng.standard_normal(5))
        obj = float(ev.evaluate_objective(x))
        cert = _convex_nlp_certificate(ev, x, lam, _LB, _UB, cl, cu, obj, gap_tolerance=1e-6)
        executed += 1
        # Since #1596 a certified point carries an explicit dual bound; ``bound is
        # None`` is the withheld verdict.
        if cert is None or cert.bound is None or cert.better_x is not None:
            withheld.append(seed)
            # Enough to tell a state leak from a numerical one when this fails only
            # under xdist (CLAUDE.md sec. 7: an instrument must say why).
            why.append(
                (
                    seed,
                    type(ev).__name__,
                    None
                    if cert is None
                    else (
                        cert.bound if cert.bound is None else cert.bound - obj,
                        cert.better_obj,
                        cert.stationarity_rel,
                        cert.complementarity_rel,
                    ),
                )
            )
        else:
            assert cert.stationarity_rel < 1e-4 and cert.complementarity_rel < 1e-6
            assert cert.bound <= obj
            assert cert.bound == pytest.approx(obj, rel=1e-6, abs=1e-6)
    assert executed == 20  # the probe fired (CLAUDE.md sec. 6)
    assert withheld == [], (
        f"noise-level perturbation withheld/altered seeds {withheld}; "
        f"(seed, evaluator, (bound - obj, better_obj, stat, comp)): {why[:4]}; "
        f"env: {sorted((k, v) for k, v in os.environ.items() if k.startswith('DISCOPT_'))}"
    )


def _internals(ev, x, lam, cl, cu, f_x, cert):
    """What the bound computation saw, for a failure seen only under CI xdist."""
    import discopt.solver as S

    cons = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
    grad = np.asarray(ev.evaluate_gradient(x), dtype=np.float64)
    parts = S._rigorous_bound_parts(ev, x, lam, cons, grad, _LB, _UB, cl, cu, f_x, [])
    fns = {
        n: getattr(getattr(S, n), "__module__", "?")
        for n in (
            "_rigorous_bound_parts",
            "_outward_tangent_bound",
            "_tangent_box_bound",
            "_gap_values_converged",
            "_no_witness_certificate",
            "_certificate_box",
        )
    }
    return (
        f"cert={cert} f_x={f_x!r} parts={parts} cl={cl} cu={cu} cons={cons} grad={grad} "
        f"lam={lam} CONSTRAINT_INF={S._CONSTRAINT_INF} abs_gap={S._DEFAULT_ABS_GAP_TOL} "
        f"eps={np.finfo(np.float64).eps} fns={fns} np={np.__version__} "
        f"env={sorted((k, v) for k, v in os.environ.items() if k.startswith('DISCOPT_'))}"
    )


def test_exact_optimum_still_certifies_exactly():
    from discopt.solver import _convex_nlp_certificate, _make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    ev = _make_evaluator(_model())
    cl, cu = _infer_constraint_bounds(ev)
    x, lam = _exact_kkt(ev, cl, cu)
    cert = _convex_nlp_certificate(
        ev, x, lam, _LB, _UB, cl, cu, float(ev.evaluate_objective(x)), gap_tolerance=1e-6
    )
    # Since #1596 a point with no witness carries an explicit dual bound (``None``
    # now means "not certified"); at an exact KKT point it equals the objective to
    # rounding and never exceeds it, and no better point is reported.
    f_x = float(ev.evaluate_objective(x))
    assert cert is not None and cert.bound is not None and cert.better_x is None, _internals(
        ev, x, lam, cl, cu, f_x, cert
    )
    assert cert.bound <= f_x
    assert cert.bound == pytest.approx(f_x, rel=1e-9, abs=1e-9)
