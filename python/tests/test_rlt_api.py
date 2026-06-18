"""First-class ``rlt=`` solve option + reusable RLT cut-soundness audit (Phase 1).

Promotes the build-time level-1 RLT lever — historically reachable *only* via the
``DISCOPT_RLT=1`` environment variable — to a real solver option threaded through
``Model.solve(rlt=...)`` → ``solve_model`` → ``MccormickLPRelaxer``. The audit
harness (:mod:`_rlt_audit`) pins the correctness-critical invariant that every
RLT family obeys: a cut never removes a feasible point.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from _rlt_audit import (
    assert_quadratic_cuts_admit_feasible_points,
    audit_bound_factor_cuts,
    audit_quadratic_bound_factor_cuts,
    generate_quadratic_bound_factor_cuts,
    standard_lifted_info_quadratic,
)
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.milp_relaxation import build_milp_relaxation


def _qcqp_with_linear_constraint() -> dm.Model:
    # Bilinear objective + a linear constraint, so level-1 RLT is *applicable*
    # (it multiplies the linear constraint by variable bound factors).
    m = dm.Model("rlt_qcqp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x * y - 2 * x - y)
    m.subject_to(x + y <= 3.0)
    return m


def _weymouth_mini() -> dm.Model:
    # Nonconvex equality f**2 == pin - pout (Weymouth gas-flow shape) plus a
    # bilinear objective term, so quadratic constraint-factor RLT (Phase 2) is
    # applicable: the equality is multiplied by f's bound factors, expanding to
    # degree-3 monomials that are lifted on demand.
    m = dm.Model("weymouth_mini")
    f = m.continuous("f", lb=0.0, ub=3.0)
    pin = m.continuous("pin", lb=1.0, ub=10.0)
    pout = m.continuous("pout", lb=1.0, ub=10.0)
    m.maximize(f - 0.1 * pin * pout)
    m.subject_to(f * f == pin - pout)
    m.subject_to(pin + pout <= 12.0)
    return m


# ── The rlt= option threads to the build-time level-1 RLT lever ──────────────


def test_rlt_level1_option_engages_relaxer():
    m = _qcqp_with_linear_constraint()
    assert MccormickLPRelaxer(m)._rlt_applicable is False  # off by default
    assert MccormickLPRelaxer(m, rlt_level1=True)._rlt_applicable is True


def test_legacy_env_var_still_forces_level1_on():
    m = _qcqp_with_linear_constraint()
    os.environ["DISCOPT_RLT"] = "1"
    try:
        assert MccormickLPRelaxer(m)._rlt_applicable is True  # env override honored
    finally:
        del os.environ["DISCOPT_RLT"]
    assert MccormickLPRelaxer(m)._rlt_applicable is False  # and cleanly reset


def test_level1_not_applicable_without_linear_constraint():
    # No linear constraint to multiply → level-1 RLT cannot fire even when asked.
    m = dm.Model("box_qp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x * y - 2 * x - y)
    assert MccormickLPRelaxer(m, rlt_level1=True)._rlt_applicable is False


# ── End-to-end: the switch is correctness-neutral (sound at every setting) ────


@pytest.mark.parametrize("rlt", ["auto", True, False, "on", "off"])
def test_rlt_switch_is_correctness_neutral(rlt):
    # Whatever the RLT setting, the certified global optimum is identical: RLT
    # only tightens the relaxation, it can never change the solution.
    m = _qcqp_with_linear_constraint()
    res = m.solve(time_limit=30, gap_tolerance=1e-4, rlt=rlt)
    assert abs(float(res.objective) - (-4.0)) < 1e-4
    assert res.gap_certified is True


# ── Reusable soundness audit: no RLT cut removes a feasible point ─────────────


def test_audit_bound_factor_cuts_simplex():
    # x0 + x1 <= 1 on the unit box — the canonical RLT separation case.
    n_cuts = audit_bound_factor_cuts(
        constraints=[({0: 1.0, 1: 1.0}, 1.0)],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )
    assert n_cuts >= 1


def test_audit_bound_factor_cuts_three_var_two_constraints():
    n_cuts = audit_bound_factor_cuts(
        constraints=[({0: 1.0, 1: 1.0, 2: 1.0}, 2.0), ({0: 2.0, 2: -1.0}, 1.0)],
        bounds=[(0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)],
        seed=3,
    )
    assert n_cuts >= 1


# ── Phase 2: quadratic constraint-factor RLT — no row removes a feasible point ─


def test_audit_quadratic_cuts_unit_disk():
    # x0**2 + x1**2 - 1 <= 0 — a nonconvex *inequality* factor. Multiplying by
    # each variable's bound factor yields degree-3 rows over the lifted space.
    n_cuts = audit_quadratic_bound_factor_cuts(
        quad_forms=[({(0, 0): 1.0, (1, 1): 1.0}, {}, -1.0, "<=")],
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    )
    assert n_cuts >= 1


def test_audit_quadratic_cuts_mixed_bilinear_square_linear():
    # x0*x1 + 2*x2**2 - x0 - 3 <= 0 — exercises bilinear, square, and linear
    # parts of the quadratic factor simultaneously.
    n_cuts = audit_quadratic_bound_factor_cuts(
        quad_forms=[({(0, 1): 1.0, (2, 2): 2.0}, {0: -1.0}, -3.0, "<=")],
        bounds=[(-1.0, 2.0), (-1.0, 2.0), (-1.0, 1.0)],
        seed=5,
    )
    assert n_cuts >= 1


def test_audit_quadratic_equality_rows_hold_on_surface():
    # An *equality* factor x0*x1 - 1 == 0 emits two-sided equality rows; they
    # must hold exactly at any point on the surface (x0=t, x1=1/t). Off-surface
    # box sampling is skipped (a zero-measure set), so build the surface directly.
    eq_form = ({(0, 1): 1.0}, {}, -1.0, "==")
    bounds = [(0.25, 4.0), (0.25, 4.0)]
    info, n_total = standard_lifted_info_quadratic(len(bounds))
    cuts = generate_quadratic_bound_factor_cuts([eq_form], bounds, info, n_total)
    assert cuts and all(c.sense == "==" for c in cuts)
    rng = np.random.default_rng(0)
    on_surface = [np.array([t := rng.uniform(0.25, 4.0), 1.0 / t]) for _ in range(2000)]
    assert_quadratic_cuts_admit_feasible_points(cuts, info, n_total, on_surface, tol=1e-7)


# ── Phase 2 build path: engages, lifts on demand, stays correctness-neutral ────


def test_quadratic_rlt_build_path_emits_lifted_rows():
    # With quadratic RLT enabled the build path multiplies the f**2 == pin - pout
    # equality by f's bound factors, lifting degree-3 monomials on demand: strictly
    # more lifted columns and constraint rows than with the lever off.
    def _rows_cols(quad_flag: str) -> tuple[int, int]:
        os.environ["DISCOPT_RLT_QUAD"] = quad_flag
        try:
            r = MccormickLPRelaxer(_weymouth_mini(), rlt_level1=True)
            milp, _ = build_milp_relaxation(r._model, r._terms, r._disc, rlt_level1=True)
        finally:
            del os.environ["DISCOPT_RLT_QUAD"]
        n_rows = 0 if milp._A_ub is None else int(milp._A_ub.shape[0])
        return n_rows, int(milp._c.size)

    rows_off, cols_off = _rows_cols("0")
    rows_on, cols_on = _rows_cols("1")
    assert cols_on > cols_off  # degree-3 monomials lifted on demand
    assert rows_on > rows_off  # RLT product rows + their envelopes emitted


def test_quadratic_rlt_is_correctness_neutral():
    # The quadratic RLT lever only tightens the relaxation: the certified global
    # optimum of the nonconvex-equality model is identical with it on or off.
    base = _weymouth_mini().solve(time_limit=60, gap_tolerance=1e-4, rlt=False)
    tightened = _weymouth_mini().solve(time_limit=60, gap_tolerance=1e-4, rlt=True)
    assert tightened.gap_certified is True
    assert abs(float(tightened.objective) - float(base.objective)) < 1e-4
