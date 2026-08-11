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

import numpy as np
import pytest
from discopt import Model
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.solver import _native_kernel_verify_point
from discopt.solvers._convex_kernel import _incumbent_is_feasible
from discopt.validation.feasibility import ABS_TOL, verify_point

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
#     violation <= abs_tol * max(1, |rhs|, max_j |J_ij| * max(1,|x_j|))
# together with a (violation, scale) pair it accepts and the chosen form rejects.
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
