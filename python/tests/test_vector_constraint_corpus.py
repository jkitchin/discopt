"""#908 — the incumbent verifiers must not vouch for points that violate a row.

Coverage note worth knowing before trusting any correctness benchmark here:
**every row in the in-repo ``.nl`` corpus is scalar**, so ``from_nl`` models cannot
exercise this class at all. A correctness sweep over the ``.nl`` corpus alone is
structurally blind to it. This corpus is therefore built through the modelling
API, which is where array-valued constraint bodies come from (``x <= 1`` on a
3-vector is ONE ``Constraint`` and THREE evaluator rows).

The pre-fix row loop is transcribed verbatim below as :func:`_pre908_verify` and
run against the same cases, so "it wrongly accepted these" is *reproduced* here
rather than argued from a commit message.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from discopt import Model
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.solver import _native_kernel_verify_point
from discopt.solvers._convex_kernel import _incumbent_is_feasible
from discopt.validation.feasibility import ABS_TOL, jacobian_row_scales, verify_point

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------
# The pre-fix implementation, transcribed verbatim.
# --------------------------------------------------------------------------
def _pre908_verify(model, x_flat) -> bool:
    """The row loop as it stood before #908: ONE index per ``Constraint`` object.

    Kept so the wrong-accept result is reproducible rather than asserted. Do not
    "fix" this — its wrongness is the point of the test.
    """
    from discopt.modeling.core import Constraint

    abs_tol, rel_tol = 1e-6, 1e-4
    ev = NLPEvaluator(model)
    if ev.n_constraints <= 0:
        return True
    cons = np.asarray(ev.evaluate_constraints(x_flat), dtype=np.float64)
    idx = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if idx >= cons.shape[0]:
            return False
        val = float(cons[idx])
        tol = abs_tol + rel_tol * abs(val)
        if c.sense == "<=":
            if val > tol:
                return False
        elif c.sense == ">=":
            if val < -tol:
                return False
        elif c.sense == "==":
            if abs(val) > tol:
                return False
        else:
            return False
        idx += 1
    return True


def _vector_le():
    m = Model("vec_le")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    m.subject_to(x <= 1.0)
    m.minimize(x[0])
    return m, np.array([0.5, 0.5, 6.0])


def _vector_eq():
    m = Model("vec_eq")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    m.subject_to(x == 0.0)
    m.minimize(x[0])
    return m, np.array([0.0, 3.0, 0.0])


def _vector_then_scalar():
    m = Model("vec_then_scalar")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0)
    m.subject_to(y <= 1.0)
    m.minimize(y)
    return m, np.array([0.0, 0.0, 0.0, 9.0])


def _two_vectors():
    m = Model("two_vec")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    y = m.continuous("y", shape=(2,), lb=0.0, ub=10.0)
    m.subject_to(x <= 1.0)
    m.subject_to(y <= 1.0)
    m.minimize(x[0])
    return m, np.array([0.0, 0.0, 0.0, 7.0])


BAD_CASES = [
    pytest.param(_vector_le, id="vector_le_row2_violated_by_5"),
    pytest.param(_vector_eq, id="vector_eq_row1_violated_by_3"),
    pytest.param(_vector_then_scalar, id="scalar_row_shifted_by_vector"),
    pytest.param(_two_vectors, id="two_vector_constraints"),
]


def _assert_point_is_in_bounds(model, pt):
    """The bad point must be in-bounds, so the ROW check is provably the only
    thing that can reject it. Without this the test could pass for the wrong
    reason (rejected on bounds, row misalignment untouched)."""
    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = pt[off : off + size]
        lb = np.asarray(v.lb, float).flatten()
        ub = np.asarray(v.ub, float).flatten()
        assert np.all(vals >= lb - 1e-9) and np.all(vals <= ub + 1e-9), (
            "point is out of bounds — the row check is not the discriminator here"
        )
        off += size


@pytest.mark.parametrize("build", BAD_CASES)
def test_infeasible_vector_point_is_rejected_by_all_entry_points(build):
    model, pt = build()
    _assert_point_is_in_bounds(model, pt)

    ok_native, obj = _native_kernel_verify_point(model, pt)
    assert ok_native is False, "native kernel vouched for a point violating a row"
    assert obj is None
    assert _incumbent_is_feasible(model, pt) is False
    assert verify_point(model, pt).ok is False


@pytest.mark.parametrize("build", BAD_CASES)
def test_pre908_row_loop_wrongly_accepted_these(build):
    """The discriminator: the pre-fix loop ACCEPTS every one of these.

    If this ever starts failing, the corpus has stopped exercising the
    misalignment and the tests above are no longer evidence of anything.
    """
    model, pt = build()
    assert _pre908_verify(model, pt) is True, (
        "the pre-fix loop rejected this point, so the case no longer discriminates"
    )


def test_scalar_control_is_rejected_by_old_and_new():
    """Control: with only scalar rows the two streams agree, so the OLD loop must
    reject too. This is what shows the corpus measures ALIGNMENT rather than the
    new verifier simply failing everything."""
    m = Model("scalar_ctl")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x <= 1.0)
    m.minimize(x)
    pt = np.array([7.0])
    assert _pre908_verify(m, pt) is False
    assert verify_point(m, pt).ok is False


def test_feasible_vector_point_is_still_accepted():
    """Control in the other direction: a genuinely feasible vector point must be
    ACCEPTED by all three entry points, so the fix cannot be 'reject everything'."""
    m = Model("vec_ok")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0)
    m.minimize(x[0])
    pt = np.array([1.0, 2.0, 3.0])
    ok, obj = _native_kernel_verify_point(m, pt)
    assert ok is True
    assert obj == pytest.approx(1.0)
    assert _incumbent_is_feasible(m, pt) is True
    assert verify_point(m, pt).ok is True


def test_maximize_objective_is_returned_in_model_units():
    m = Model("vec_max")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0)
    m.maximize(x[0] + x[1])
    ok, obj = _native_kernel_verify_point(m, np.array([2.0, 3.0]))
    assert ok is True
    assert obj == pytest.approx(5.0), "MAXIMIZE objective was not un-negated"


def test_builder_resident_linear_rows_are_examined():
    """#840 rows live in the builder, not ``model._constraints``. The evaluator
    includes them; a verifier walking ``_constraints`` never saw them at all."""
    m = Model("builder_rows")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    A = np.eye(3)
    m.add_linear_constraints(A, x, "<=", np.array([1.0, 1.0, 1.0]), name="cap")
    m.minimize(x[0])

    assert len(m._constraints) == 0, "precondition: these rows are builder-resident"
    assert len(m._builder_linear_constraints()) == 3

    pt = np.array([0.0, 0.0, 8.0])  # violates row 2 by 7
    _assert_point_is_in_bounds(m, pt)
    assert _pre908_verify(m, pt) is True, "precondition: the old loop could not see these"
    assert verify_point(m, pt).ok is False
    assert _native_kernel_verify_point(m, pt)[0] is False
    assert _incumbent_is_feasible(m, pt) is False


def test_row_map_covers_every_evaluator_row_exactly_once():
    """Structural invariant: the map partitions ``[0, n_constraints)``. This is
    what makes the misalignment class impossible rather than merely fixed."""
    m = Model("cover")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0)
    m.subject_to(y <= 1.0)
    m.add_linear_constraints(np.ones((1, 3)), x, "<=", np.array([9.0]), name="tot")
    m.minimize(y)

    ev = NLPEvaluator(m)
    rows = ev.constraint_row_map()
    covered = []
    for start, stop, _ in rows:
        covered.extend(range(start, stop))
    assert covered == list(range(ev.n_constraints)), "row map does not partition the rows"
    assert ev.n_constraints == 5  # 3 vector + 1 scalar + 1 builder


# --------------------------------------------------------------------------
# Tolerance: keyed on the ROW's scale, and not one notch looser.
# --------------------------------------------------------------------------
def test_large_scale_row_is_not_rejected_for_a_tiny_relative_residual():
    """The nvs22 direction: an absolute residual of 1.7e-5 against a row whose
    natural magnitude is ~2100 is a RELATIVE 8e-9 and must be accepted. The old
    flat 1e-6 rejected it."""
    m = Model("big_scale")
    x = m.continuous("x", lb=0.0, ub=1e4)
    m.subject_to(2121.64 * x == 2121.64)
    m.minimize(x)
    # Perturb so the row residual is ~1.7e-5 in absolute terms.
    pt = np.array([1.0 + 8.1e-9])
    ev = NLPEvaluator(m)
    resid = abs(float(np.asarray(ev.evaluate_constraints(pt))[0]))
    assert 1e-6 < resid < 1e-3, f"case does not exercise the regime (resid={resid:.3e})"
    assert verify_point(m, pt).ok is True, "scale-blind rejection of a valid incumbent"


def test_unit_scale_row_is_still_held_to_the_absolute_tolerance():
    """Anti-permissiveness: scale-keying must not loosen a unit-scale row."""
    m = Model("unit_scale")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x == 1.0)
    m.minimize(x)
    assert verify_point(m, np.array([1.0 + 1e-9])).ok is True
    assert verify_point(m, np.array([1.0 + 1e-3])).ok is False


# Each entry is a NAIVE WIDENING of
#     violation <= abs_tol * max(1, |rhs|, max_j |J_ij| * |x_j|)
# together with a (violation, scale) pair it accepts and the chosen form rejects.
# (#1151 replaced the `max(1,|x_j|)` factor this header used to name with the
# plain `|x_j|` term magnitude; the parametrized cases below take `scale` as a
# given and were unaffected, but the header was documenting a form the module no
# longer implements -- and, worse, one of the shapes it exists to reject.)
# The direction matters: every one of these is LOOSER, and three of them are
# looser in a way that grows with the violation itself.
NAIVE_WIDENINGS = [
    pytest.param(
        lambda viol, scale, val: viol <= 1e-4 * max(1.0, scale),
        1e-5,
        1.0,
        1e-5,
        id="rel_tol_coefficient_instead_of_abs_tol__100x_looser",
    ),
    pytest.param(
        lambda viol, scale, val: viol <= ABS_TOL + 1e-4 * abs(val),
        1e-2,
        1.0,
        1e3,
        id="self_referential_old_form__grows_with_the_residual",
    ),
    pytest.param(
        lambda viol, scale, val: viol <= ABS_TOL * max(1.0, abs(val)),
        1e-2,
        1.0,
        1e5,
        id="anchored_on_row_VALUE__tolerance_grows_with_violation",
    ),
    pytest.param(
        lambda viol, scale, val: viol <= ABS_TOL * max(1.0, 1000.0 * scale),
        1e-4,
        1.0,
        1e-4,
        id="sum_j_instead_of_max_j__scales_with_row_density",
    ),
]


@pytest.mark.parametrize("widened,viol,scale,val", NAIVE_WIDENINGS)
def test_naive_tolerance_widenings_accept_what_the_chosen_form_rejects(widened, viol, scale, val):
    chosen = viol <= ABS_TOL * max(1.0, scale)
    assert chosen is False, "case does not exercise the chosen form's boundary"
    assert widened(viol, scale, val) is True, (
        "this widening was supposed to be looser; the control no longer discriminates"
    )


def test_constraint_rhs_is_honoured():
    """``Constraint.rhs`` must re-centre the row, not be ignored.

    The evaluator compiles the **body only**. Bodies built through the operator
    API normalise to ``rhs == 0`` (``x <= 10`` becomes body ``x - 10``, rhs 0), so
    no production path in this tree is currently known to set a non-zero ``rhs``
    — this guards a *latent* inconsistency rather than reproducing an observed
    production failure. It is written on the ``>=`` side because that is the
    direction in which ignoring ``rhs`` WRONGLY ACCEPTS: comparing the raw body
    against 0 tests ``x >= 0``, which admits a point that violates ``x >= 2``.
    """
    from discopt.modeling.core import Constraint

    m = Model("rhs")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x >= 0.0)
    m.minimize(x)
    # body is exactly `x`, so the stored constraint reads `x >= 2`.
    m._constraints[0] = Constraint(body=x, sense=">=", rhs=2.0, name="floor")

    ev = NLPEvaluator(m)
    assert float(np.asarray(ev.evaluate_constraints(np.array([1.0])))[0]) == pytest.approx(1.0), (
        "precondition: the evaluator returns the RAW body, un-centred by rhs"
    )
    assert verify_point(m, np.array([3.0])).ok is True, "rhs=2 should admit x=3"
    assert verify_point(m, np.array([1.0])).ok is False, "rhs ignored: x=1 violates x>=2"


# --------------------------------------------------------------------------
# #1151 — the row scale must be a TERM magnitude, not a coefficient magnitude.
# --------------------------------------------------------------------------
def _quotient_aux_row_model():
    """The shape ``factorable_reform`` emits for a quotient in the objective.

    ``minimize x/y`` over a strictly-positive box is lifted to ``minimize w``
    with the defining equality ``w == x/y``, which ``_clear_divisions`` clears to
    the bilinear ``w*y - x == 0`` and then multiplies by ``1/dmin`` (``dmin`` the
    box minimum of the denominator) — here ``1/1e-3 = 1000`` — precisely so that a
    fixed ABSOLUTE residual test on the row bounds the error in ``w``.
    """
    m = Model("quotient_aux")
    x = m.continuous("x", lb=1e-3, ub=1e3)
    y = m.continuous("y", lb=1e-3, ub=1e3)
    w = m.continuous("w", lb=1e-6, ub=1e6)
    m.subject_to(1000.0 * (w * y) - 1000.0 * x == 0.0)
    m.minimize(w)
    return m


def test_row_scale_is_a_term_magnitude_not_a_coefficient_magnitude():
    """#1151: ``max_j |J_ij| * max(1,|x_j|)`` over-reads a scaled row's scale by
    ``1/|x_j|``, and that slack is exactly the amplification the ``1/dmin``
    scaling exists to remove.

    The point below is the one the solver actually returned on
    ``minimize x/y + y/x`` (global minimum 2 by AM-GM) before the fix. Its row is
    violated by 9.3e-4; the old form read the row's scale as 1000 — the
    coefficient on ``x`` — rather than 1.4, the magnitude ``1000*x`` attains at
    the point, and so licensed a tolerance of 1e-3 and accepted it. The residual
    maps to ``residual/(1000*y)`` of error in ``w``, which is how a reported
    objective landed BELOW the global minimum at ``status=optimal``.
    """
    m = _quotient_aux_row_model()
    pt = np.array([0.0014052502011193727, 0.0014073586395206353, 0.9978427215251631])

    ev = NLPEvaluator(m)
    resid = abs(float(np.asarray(ev.evaluate_constraints(pt))[0]))
    J = np.abs(np.asarray(ev.evaluate_jacobian(pt), dtype=np.float64)[0])
    old_scale = float((J * np.maximum(1.0, np.abs(pt))).max())
    new_scale = float((J * np.abs(pt)).max())

    # Preconditions: the case exercises the regime it claims to (§6 — an
    # assertion that cannot fire is not a test).
    assert resid == pytest.approx(9.276e-4, rel=1e-2), (
        f"case no longer reproduces the #1151 residual (got {resid:.3e})"
    )
    assert old_scale == pytest.approx(1000.0, rel=1e-2), "old form no longer reads 1000"
    assert new_scale == pytest.approx(1.405, rel=1e-2), "term magnitude is not ~1.4"
    assert resid <= ABS_TOL * old_scale, "the old form would NOT have accepted this point"
    assert resid > ABS_TOL * max(1.0, new_scale), "the term-magnitude form must reject it"

    assert verify_point(m, pt).ok is False, (
        "the incumbent verifier vouched for a row violated by 9.3e-4 — the #1151 false certificate"
    )
    assert _native_kernel_verify_point(m, pt)[0] is False


def test_term_magnitude_scale_still_admits_the_nvs22_direction():
    """The #1151 tightening must not undo #908's nvs22 widening.

    The nvs22 case is a row whose terms genuinely reach ~2121, so the term
    magnitude and the floored form agree and the 1.7e-5 absolute residual stays
    inside tolerance.
    """
    m = Model("big_scale_terms")
    x = m.continuous("x", lb=0.0, ub=1e4)
    m.subject_to(2121.64 * x == 2121.64)
    m.minimize(x)
    pt = np.array([1.0 + 8.1e-9])

    ev = NLPEvaluator(m)
    resid = abs(float(np.asarray(ev.evaluate_constraints(pt))[0]))
    J = np.abs(np.asarray(ev.evaluate_jacobian(pt), dtype=np.float64)[0])
    assert 1e-6 < resid < 1e-3, f"case does not exercise the regime (resid={resid:.3e})"
    assert float((J * np.abs(pt)).max()) == pytest.approx(2121.64, rel=1e-6)
    assert verify_point(m, pt).ok is True, "#1151 tightening re-broke the nvs22 direction"


def test_term_magnitude_scale_is_never_looser_than_the_floored_form():
    """Direction guard: ``|J_ij|*|x_j| <= |J_ij|*max(1,|x_j|)`` pointwise, so the
    new row scale — and hence the tolerance — can only ever shrink. No point this
    verifier accepts was rejected by the pre-#1151 form."""
    rng = np.random.default_rng(1151)
    compared = 0
    for _ in range(200):
        J = rng.normal(scale=10.0, size=6)
        x = rng.normal(scale=10.0, size=6)
        old = float((np.abs(J) * np.maximum(1.0, np.abs(x))).max())
        new = float((np.abs(J) * np.abs(x)).max())
        assert new <= old + 1e-12
        compared += 1
    assert compared == 200, "the direction guard compared nothing"


# --------------------------------------------------------------------------
# #1151 review finding 1+2: the row scale has ONE definition, and the two other
# consumers use it. Before the review this expression was written out by hand in
# three places; fixing one left the other two vouching for the point it rejects.
# --------------------------------------------------------------------------
def test_examiner_scaled_feasibility_rejects_what_the_incumbent_gate_rejects():
    """The user-facing feasibility tool must not certify a point the incumbent
    verifier refuses — on the very class #1151 is about.

    Measured before the fix, on this exact model and point::

        verify_point -> False: row 0 violated by 9.276e-04 (allowed 1.405e-06)
        examiner     -> [PASS] primal_con_feas (scaled) (tol=1.0e-06)
    """
    from types import SimpleNamespace

    from discopt.validation.examiner import examine

    m = _quotient_aux_row_model()
    pt = np.array([0.0014052502011193727, 0.0014073586395206353, 0.9978427215251631])

    assert verify_point(m, pt).ok is False, "precondition: the incumbent gate rejects this point"

    report = examine(
        SimpleNamespace(
            x={v.name: float(pt[i]) for i, v in enumerate(m._variables)},
            objective=None,
            bound=None,
            status="optimal",
        ),
        m,
        recover_duals=False,
    )
    scaled = [c for c in report.checks if c.name == "primal_con_feas (scaled)"]
    assert len(scaled) == 1, f"the scaled primal-feasibility check did not run: {report.checks}"
    assert scaled[0].passed is False, (
        "the examiner certified a point the incumbent verifier rejects — the "
        "#1151 row scale is back in examiner.py"
    )


def test_dual_recovery_active_set_uses_the_term_magnitude_scale():
    """``_dual_recovery``'s ``near`` test must not admit a row with 9.3e-4 of
    signed slack into the KKT active set.

    Surplus inactive rows contribute free multipliers to the least-squares solve,
    which can shrink the stationarity residual and let a non-KKT point pass — a
    validation weakened in the ACCEPTING direction.

    **Scope, measured rather than assumed.** The review that found this described
    it on a quotient aux's defining row. That row is an ``==``, and
    ``row_select`` takes ``is_eq`` unconditionally — ``near`` is never consulted
    for an equality — so the scale cannot change the active set there. The defect
    is real but bites on scaled **inequalities**, which ``_clear_divisions``
    also produces (it multiplies a ``<=`` through by a sign-definite denominator
    and flips the sense when that denominator is negative). This case is
    therefore built on the ``<=`` form, where the two scales genuinely disagree
    about membership; on the ``==`` form both select the row and there is nothing
    to test.
    """
    from discopt._dual_recovery import recover_multipliers, row_metadata

    # The `<=` form of the same scaled row, which is what `_clear_divisions`
    # emits for an inequality with a sign-definite denominator.
    m = Model("quotient_aux_le")
    m.continuous("x", lb=1e-3, ub=1e3)
    m.continuous("y", lb=1e-3, ub=1e3)
    m.continuous("w", lb=1e-6, ub=1e6)
    x, y, w = m._variables
    m.subject_to(1000.0 * (w * y) - 1000.0 * x <= 0.0)
    m.minimize(w)

    pt = np.array([0.0014052502011193727, 0.0014073586395206353, 0.9978427215251631])
    ev = NLPEvaluator(m)
    J = np.asarray(ev.evaluate_jacobian(pt), dtype=np.float64)
    body = np.asarray(ev.evaluate_constraints(pt), dtype=np.float64)
    sense_arr, rhs_arr, _labels = row_metadata(ev)
    assert sense_arr[0] == "<=", "precondition: this case needs an inequality row"
    signed = abs(float(body[0]))

    # Preconditions: this row IS the regime -- the floored form admits it, the
    # term magnitude does not. Asserted, so the case cannot quietly stop
    # exercising the boundary it was built for (§6).
    active_tol = 1e-6
    floored = float((np.abs(J[0]) * np.maximum(1.0, np.abs(pt))).max())
    term = float(jacobian_row_scales(J, pt)[0])
    assert signed <= active_tol * max(1.0, floored), (
        "precondition: the floored form admitted this row into the active set"
    )
    assert signed > active_tol * max(1.0, term), (
        "precondition: the term-magnitude scale must exclude this row"
    )

    # And the real code path agrees: the row is NOT in the recovered active set.
    grad = np.zeros(pt.size)
    grad[2] = 1.0  # objective is `w`
    lo = np.array([v.lb for v in m._variables], dtype=float).ravel()
    hi = np.array([v.ub for v in m._variables], dtype=float).ravel()
    dr = recover_multipliers(
        grad=grad,
        jac=J,
        body=body,
        sense_arr=sense_arr,
        rhs_arr=rhs_arr,
        x_flat=pt,
        lb=lo,
        ub=hi,
        is_continuous=np.ones(pt.size, dtype=bool),
        active_tol=active_tol,
    )
    assert 0 not in set(np.asarray(dr.row_select).tolist()), (
        "_dual_recovery admitted a row with 9.3e-4 of signed slack into the KKT "
        "active set — the #1151 row scale is back in _dual_recovery.py"
    )


def test_row_scale_has_a_single_definition_shared_by_all_three_consumers():
    """§2, structurally: the three call sites import one helper rather than each
    writing ``max_j |J_ij| * |x_j|`` out again. A hand-written copy is how #1151
    came to be fixed in one place and left standing in two others."""
    import inspect

    from discopt import _dual_recovery
    from discopt.validation import examiner, feasibility

    for mod in (examiner, _dual_recovery, feasibility):
        src = inspect.getsource(mod)
        assert "jacobian_row_scales" in src, (
            f"{mod.__name__} does not use the shared row-scale helper"
        )
        assert "np.maximum(1.0, np.abs(x_flat))" not in src, (
            f"{mod.__name__} still carries the pre-#1151 floored row scale by hand"
        )


def test_jacobian_row_scales_rejects_shape_mismatches():
    """The shared helper refuses inputs it cannot interpret rather than
    broadcasting them into a silently wrong scale (CLAUDE.md §7)."""
    with pytest.raises(ValueError, match="2-D Jacobian"):
        jacobian_row_scales(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="columns"):
        jacobian_row_scales(np.zeros((2, 3)), np.zeros(4))
    assert jacobian_row_scales(np.zeros((0, 3)), np.zeros(3)).shape == (0,)
    assert jacobian_row_scales(np.array([[2.0, -3.0]]), np.array([5.0, 1.0]))[0] == 10.0


def test_row_scale_is_finite_when_a_derivative_is_unbounded_at_zero():
    """#1157 second review: the non-finite guard must live WITH the definition.

    ``d/dx log(x)`` at ``x = 0`` is unbounded, so ``|J_ij| * |x_j|`` is
    ``inf * 0`` — a NaN, which numpy warns about and which compares False
    against every tolerance downstream. ``_row_scales`` had a guard for this;
    factoring the formula out into ``jacobian_row_scales`` left the two direct
    callers without one, so they emitted a RuntimeWarning and a spurious
    ``[FAIL] primal_con_feas (scaled)`` carrying ``scale=nan``.

    Direction: 0.0 means "no usable estimate", so the caller's floor applies and
    the row is held to the plain absolute tolerance — the STRICTEST answer. The
    pre-#1151 floored form gave that row ``inf``, hence an *infinite* tolerance,
    passing it unconditionally; this keeps the safe direction.
    """
    J = np.array([[np.inf, 1.0]])
    x = np.array([0.0, 2.0])

    assert not np.isfinite((np.abs(J) * np.maximum(1.0, np.abs(x))).max()), (
        "precondition: the pre-#1151 form yields a non-finite scale here"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scales = jacobian_row_scales(J, x)
    assert np.all(np.isfinite(scales)), f"non-finite row scale leaked: {scales}"
    assert scales[0] == 0.0, "a non-finite row must fall back to the caller's floor"

    # A finite row sharing the batch keeps its own scale; only the bad row is zeroed.
    mixed = jacobian_row_scales(np.array([[np.inf, 1.0], [2.0, 3.0]]), np.array([0.0, 4.0]))
    assert mixed.tolist() == [0.0, 12.0]


def test_row_scales_keeps_its_whole_batch_fallback():
    """``_row_scales`` must NOT inherit the public helper's per-row zeroing.

    One non-finite row there sends *every* suspect row to the Jacobian-free
    bound. Relaxing that to per-row would leave the co-occurring rows on their
    own larger scales — looser than before the helper was extracted, i.e. a
    relaxation in the accepting direction smuggled in by a refactor.
    """
    from discopt.validation.feasibility import _jacobian_row_scales_checked, _row_scales

    scales, all_finite = _jacobian_row_scales_checked(
        np.array([[np.inf, 1.0], [2.0, 3.0]]), np.array([0.0, 4.0])
    )
    assert all_finite is False, "the checked form must report the non-finite row"
    assert scales.tolist() == [0.0, 12.0]

    class _Ev:
        def evaluate_jacobian(self, _x):
            return np.array([[np.inf, 1.0], [2.0, 3.0]])

    assert _row_scales(_Ev(), np.array([0.0, 4.0]), np.array([0, 1])) is None, (
        "_row_scales must decline the whole batch, not zero one row and keep the rest"
    )


def test_examiner_does_not_warn_or_spuriously_fail_on_an_unbounded_derivative():
    """End to end: the tool this PR makes authoritative must stay quiet and
    correct at a feasible point with an unbounded derivative."""
    from types import SimpleNamespace

    import discopt.modeling as dm
    from discopt.validation.examiner import examine

    m = Model("logmodel")
    m.continuous("a", lb=0.0, ub=10.0)
    m.continuous("b", lb=0.0, ub=10.0)
    a, _b = m._variables
    m.subject_to(dm.log(a) + _b <= 100.0)
    m.minimize(_b)

    pt = SimpleNamespace(x={"a": 0.0, "b": 2.0}, objective=None, bound=None, status="optimal")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = examine(pt, m, recover_duals=False)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]

    assert not runtime, (
        f"examiner emitted numpy RuntimeWarnings: {[str(w.message) for w in runtime]}"
    )
    scaled = [c for c in report.checks if c.name == "primal_con_feas (scaled)"]
    assert len(scaled) == 1, "the scaled primal-feasibility check did not run"
    assert scaled[0].passed is True, "a NaN row scale made the scaled check fail a feasible point"
