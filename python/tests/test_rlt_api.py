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
import pytest
from _rlt_audit import audit_bound_factor_cuts
from discopt._jax.mccormick_lp import MccormickLPRelaxer


def _qcqp_with_linear_constraint() -> dm.Model:
    # Bilinear objective + a linear constraint, so level-1 RLT is *applicable*
    # (it multiplies the linear constraint by variable bound factors).
    m = dm.Model("rlt_qcqp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x * y - 2 * x - y)
    m.subject_to(x + y <= 3.0)
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
