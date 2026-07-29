"""Regression tests for the single incumbent feasibility verifier.

These lock the fix for the four defects measured on 2026-07-29 in the two
hand-rolled incumbent verifiers (``solver/native_kernel._native_kernel_verify_point``
and ``solvers/_convex_kernel._incumbent_is_feasible``), now both delegating to
``discopt.validation.feasibility.verify_point``:

1. a self-referential tolerance ``abs + rel*|residual|`` that collapses to a pure
   absolute 1e-6 on every row scale (wrongly REJECTS ``nvs22``'s certified optimum);
2. one row index per *constraint object* against one evaluator row per *flat element*
   (wrongly ACCEPTS a point violating a vector-constraint row by 5.0);
3. builder-resident linear rows never examined (wrongly ACCEPTS);
4. ``Constraint.rhs`` ignored, unknown constraint classes skipped or crashing.

The file is deliberately organised as "stricter" first and "scale-aware" second,
because the §0.4 claim being locked is that the scale term did not buy permissiveness
— every class of point the old form rejected for a *real* reason is still rejected,
and four classes it wrongly accepted now are not.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.modeling.core import Constraint, Model
from discopt.solver.native_kernel import _native_kernel_verify_point
from discopt.solvers._convex_kernel import _incumbent_is_feasible
from discopt.validation.feasibility import row_scales, verify_point


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _both(model, x):
    """(shared verifier, native-kernel entry point, convex-kernel entry point)."""
    x = np.asarray(x, dtype=float)
    shared = verify_point(model, x)
    nk_ok, _ = _native_kernel_verify_point(model, x)
    ck_ok = _incumbent_is_feasible(model, x)
    return shared, nk_ok, ck_ok


# --------------------------------------------------------------------------- #
# 1. STRICTER: points the old form wrongly ACCEPTED
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_vector_constraint_rows_beyond_the_first_are_checked():
    """Defect 2. The old loop advanced one index per constraint OBJECT, so a size-3
    vector constraint had rows 1 and 2 never examined. Measured: a point violating
    row 2 by 5.0 was returned as feasible by BOTH verifiers."""
    m = Model("vec")
    x = m.continuous("x", shape=(3,), lb=-10, ub=10)
    m.subject_to(x <= 1.0)
    m.minimize(x[0])

    shared, nk_ok, ck_ok = _both(m, [0.0, 0.0, 6.0])
    assert shared.n_rows_checked == 3, "probe did not fire on all three flat rows"
    assert not shared.ok and not nk_ok and not ck_ok
    assert shared.violations[0].violation == pytest.approx(5.0)

    # ... and the genuinely feasible point on the same model still passes.
    ok_shared, ok_nk, ok_ck = _both(m, [0.0, 0.0, 0.5])
    assert ok_shared.ok and ok_nk and ok_ck


@pytest.mark.smoke
def test_builder_resident_linear_rows_are_checked():
    """Defect 3. ``add_linear_constraints`` rows live only in the Rust builder, so a
    verifier iterating ``model._constraints`` sees an unconstrained model."""
    m = Model("builder")
    x = m.continuous("x", shape=(2,), lb=-100, ub=100)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, "<=", np.array([1.0]))
    m.minimize(x[0])
    assert not [c for c in m._constraints if isinstance(c, Constraint)], (
        "test premise broken: the fast path materialised a Python Constraint"
    )

    shared, nk_ok, ck_ok = _both(m, [10.0, 10.0])  # row activity 20 vs rhs 1
    assert shared.n_rows_checked >= 1, "probe did not fire: no builder row examined"
    assert not shared.ok and not nk_ok and not ck_ok

    ok_shared, ok_nk, ok_ck = _both(m, [0.25, 0.25])
    assert ok_shared.ok and ok_nk and ok_ck


@pytest.mark.smoke
def test_nonzero_rhs_is_subtracted_in_both_directions():
    """Defect 4a. The old loop compared the BODY against zero, ignoring ``rhs``: it
    rejected a feasible point on a ``body <= 5`` row and would accept a violating one
    on a ``body <= -5`` row."""
    m = Model("rhs_ok")
    w = m.continuous("w", lb=-100, ub=100)
    m._constraints.append(Constraint(w, "<=", 5.0, "wle5"))
    m.minimize(w)
    shared, nk_ok, ck_ok = _both(m, [3.0])  # 3 <= 5 : feasible
    assert shared.ok and nk_ok and ck_ok, "feasible point on a nonzero-rhs row rejected"

    m2 = Model("rhs_bad")
    w2 = m2.continuous("w2", lb=-100, ub=100)
    m2._constraints.append(Constraint(w2, "<=", -5.0, "wlem5"))
    m2.minimize(w2)
    bad, bad_nk, bad_ck = _both(m2, [-1.0])  # -1 <= -5 is FALSE
    assert not bad.ok and not bad_nk and not bad_ck


@pytest.mark.smoke
def test_unevaluable_constraint_class_is_refused_not_ignored():
    """Defect 4b. SOS / logical / indicator / disjunctive constraints are not in the
    evaluator's row set. Silently ignoring a whole constraint class while returning
    "feasible" is the failure mode this module exists to stop; the old convex-kernel
    loop additionally raised ``AttributeError`` mid-loop."""
    m = Model("sos")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-10, ub=10)
    m.sos1([a, b])
    m.minimize(a)

    shared, nk_ok, ck_ok = _both(m, [1.0, 1.0])
    assert shared.refusal is not None and "_SOSConstraint" in shared.refusal
    assert not shared.ok and not nk_ok and not ck_ok


@pytest.mark.smoke
def test_bounds_and_integrality_are_checked_on_the_convex_kernel_path():
    """``_incumbent_is_feasible`` checked neither before this change."""
    m = Model("int")
    y = m.integer("y", lb=0, ub=10)
    m.subject_to(y >= 0.0)
    m.minimize(y)

    frac = _incumbent_is_feasible(m, np.array([2.5]))
    assert not frac, "fractional value accepted for an INTEGER variable"
    oob = _incumbent_is_feasible(m, np.array([25.0]))
    assert not oob, "out-of-bounds value accepted"
    assert _incumbent_is_feasible(m, np.array([2.0]))


# --------------------------------------------------------------------------- #
# 2. THE NAMED DEFECT: scale-blindness
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_large_scale_equality_row_accepts_a_tiny_relative_residual():
    """The ``nvs22`` shape, synthesised. A defined-variable equality row of magnitude
    ~1e4 carrying a residual of 2.6e-4 is a relative residual of 2.6e-8 — a point the
    solver's own certificate is issued at — and the old ``abs + rel*|residual|`` form
    rejects it because that form is arithmetically a pure absolute 1e-6."""
    m = Model("bigrow")
    x = m.continuous("x", lb=1.0, ub=1e6)
    u = m.continuous("u", lb=1.0, ub=1e6)
    m.subject_to(u - 1.7329e4 * x == 0.0)
    m.minimize(x)

    x0 = 1.0
    resid = 2.641e-4
    shared, nk_ok, ck_ok = _both(m, [x0, 1.7329e4 * x0 + resid])
    assert shared.n_rows_checked == 1
    assert shared.ok and nk_ok and ck_ok, shared.describe()
    assert shared.worst_relative < 1e-7

    # The OLD form's verdict on the same point, computed inline so the regression is
    # locked against the arithmetic rather than against a remembered outcome.
    old_tol = 1e-6 + 1e-4 * abs(resid)
    assert resid > old_tol, "test premise broken: the old form would have accepted this"


@pytest.mark.smoke
def test_unit_scale_row_tolerance_is_unchanged():
    """The calibration claim: on a row of scale 1 the new tolerance is exactly the old
    absolute 1e-6, so the fix buys no permissiveness where the scale does not warrant
    it. A 5e-6 violation on a unit row must still be rejected."""
    m = Model("unit")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x == 0.0)
    m.minimize(x)

    assert verify_point(m, np.array([5e-7])).ok
    bad = verify_point(m, np.array([5e-6]))
    assert not bad.ok, "unit-scale row loosened"
    assert bad.violations[0].tol == pytest.approx(1e-6, rel=1e-9)
    assert bad.violations[0].scale == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. THE NAIVE WIDENINGS, each of which would accept a bad point
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_scale_is_row_local_not_model_global():
    """Naive widening A: scale the tolerance by the largest variable in the MODEL.
    A unit row violated by 1e-2 in a model that happens to contain a 1e9 variable
    would then be accepted with tolerance 1e3."""
    m = Model("global")
    small = m.continuous("small", lb=-10, ub=10)
    big = m.continuous("big", lb=0, ub=2e9)
    m.subject_to(small == 0.0, name="unit_row")
    m.subject_to(big - 1e9 <= 0.0, name="big_row")
    m.minimize(small)

    x = np.array([1e-2, 1e9])
    naive_global_tol = 1e-6 * max(1.0, float(np.max(np.abs(x))))
    assert 1e-2 < naive_global_tol, "test premise broken: naive form would reject anyway"

    shared, nk_ok, ck_ok = _both(m, x)
    assert not shared.ok and not nk_ok and not ck_ok
    offending = [v for v in shared.violations if v.label == "unit_row"]
    assert offending and offending[0].scale == pytest.approx(1.0), shared.describe()


@pytest.mark.smoke
def test_scale_is_the_infinity_norm_not_the_sum_of_terms():
    """Naive widening B: scale by the 1-norm of the row's linearised terms. A row of
    1000 cancelling terms of magnitude 1000 has a 1-norm of 1e6 — tolerance 1.0 —
    while its true scale is 1000 and a 0.5 violation must be rejected."""
    import discopt.modeling as dm

    n = 1000
    m = Model("norm")
    x = m.continuous("x", shape=(n,), lb=-1e4, ub=1e4)
    m.subject_to(dm.sum(x) == 0.0, name="cancelling_row")
    m.minimize(x[0])

    vals = np.empty(n)
    vals[0::2] = 1000.0
    vals[1::2] = -1000.0
    vals[0] += 0.5  # row activity 0.5, violation 0.5

    one_norm = float(np.sum(np.abs(vals)))
    inf_norm = float(np.max(np.abs(vals)))
    assert 0.5 < 1e-6 * one_norm, "test premise broken: 1-norm form would reject anyway"
    assert 0.5 > 1e-6 * inf_norm

    shared, nk_ok, ck_ok = _both(m, vals)
    assert not shared.ok and not nk_ok and not ck_ok, shared.describe()
    assert shared.violations[0].scale == pytest.approx(inf_norm, rel=1e-9)


@pytest.mark.smoke
def test_relative_coefficient_is_abs_tol_not_the_repo_rel_tol():
    """Naive widening C: reuse the repo's ``rel_tol`` (1e-4) as the coefficient on the
    row scale. The floor would then be 1.01e-4 and every unit-scale row would loosen
    by 100x, accepting a 5e-5 violation on a row of magnitude 1."""
    m = Model("coeff")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x == 0.0)
    m.minimize(x)

    v = 5e-5
    assert v < 1e-6 + 1e-4 * 1.0, "test premise broken: rel_tol form would reject anyway"
    assert not verify_point(m, np.array([v])).ok


@pytest.mark.smoke
def test_widening_the_absolute_floor_would_accept_a_unit_row_violation():
    """Naive widening D: raise the absolute tolerance to 1e-3 so ``nvs22`` passes. A
    unit-scale row violated by 5e-4 would then be accepted."""
    m = Model("absfloor")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x == 0.0)
    m.minimize(x)
    assert 5e-4 < 1e-3, "test premise"
    assert not verify_point(m, np.array([5e-4])).ok


# --------------------------------------------------------------------------- #
# 4. row_scales() unit properties and refusal contract
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_row_scales_floor_and_fallback():
    rhs = np.array([0.0, 7.0])
    x = np.array([0.5, 1000.0])
    # No Jacobian available -> |rhs| only, floored at 1: the STRICT direction.
    assert row_scales(None, rhs, x).tolist() == [1.0, 7.0]
    jac = np.array([[2.0, 0.0], [0.0, 3.0]])
    # row 0: max(1, 0, 2*max(1,0.5)=2) = 2 ; row 1: max(1, 7, 3*1000=3000) = 3000
    assert row_scales(jac, rhs, x).tolist() == [2.0, 3000.0]


@pytest.mark.unit
def test_row_scales_ignores_a_nonfinite_jacobian_row():
    """A scale we cannot trust must not widen a tolerance."""
    rhs = np.array([0.0])
    x = np.array([1.0])
    jac = np.array([[np.inf]])
    assert row_scales(jac, rhs, x).tolist() == [1.0]


@pytest.mark.unit
def test_refusal_is_not_a_pass_and_carries_a_reason():
    m = Model("nonfinite")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x == 0.0)
    m.minimize(x)
    v = verify_point(m, np.array([np.nan]))
    assert not v.ok and v.refusal is not None and not bool(v)
    assert "NOT VERIFIED" in v.describe()


@pytest.mark.unit
def test_point_length_mismatch_is_refused():
    m = Model("short")
    x = m.continuous("x", shape=(3,), lb=-10, ub=10)
    m.subject_to(x <= 1.0)
    m.minimize(x[0])
    v = verify_point(m, np.array([0.0, 0.0]))
    assert not v.ok and v.refusal is not None


# --------------------------------------------------------------------------- #
# 5. the real instance
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.correctness
def test_nvs22_certified_optimum_verifies():
    """The instance that surfaced the defect. Phase 5's differential panel scored
    ``cert-clean: FAIL(2)`` on it in BOTH arms; the failures were the verifier's.
    The certificate matches ``=opt= 6.05822``; the two offending rows carry relative
    residuals of 8.1e-9 and 1.5e-8."""
    from pathlib import Path

    from discopt.modeling.core import from_nl

    path = Path(__file__).parent / "data" / "minlplib" / "nvs22.nl"
    if not path.exists():  # pragma: no cover - corpus not present
        pytest.skip(f"corpus instance missing: {path}")

    model = from_nl(str(path))
    r = model.solve(time_limit=45)
    assert r.x is not None, "no incumbent — the regression this locks cannot be tested"
    assert r.objective == pytest.approx(6.05822, rel=1e-4)

    flat = np.concatenate([np.asarray(r.x[v.name], dtype=float).ravel() for v in model._variables])
    fresh = from_nl(str(path))
    verdict = verify_point(fresh, flat)
    assert verdict.n_rows_checked == 9, "probe did not fire on all rows"
    assert verdict.ok, verdict.describe()
    assert verdict.worst_relative < 1e-6

    ok_nk, obj_nk = _native_kernel_verify_point(from_nl(str(path)), flat)
    assert ok_nk and obj_nk == pytest.approx(6.05822, rel=1e-4)
    assert _incumbent_is_feasible(from_nl(str(path)), flat)
