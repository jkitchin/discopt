"""#1831: robust counterparts must sign a constant multiplier element-wise.

``sign_tracking_substitute`` gave a constant vector multiplying an uncertain
parameter ONE sign, ``np.sign(np.sum(c))``. For a mixed-sign ``c`` every element
whose coefficient had the other sign was substituted at its *best* case, and the
solve certified a non-robust point as optimal: ``x + sum([3, -1] * p) <= 10``
over ``p in [0, 2]^2`` returned ``x = 6`` (robust truth ``x = 4``). The same
audit found that a negative or mixed-sign *divisor* (``p / d``) and a constant
*matrix* (``C @ p``) propagated no sign at all, and that the polyhedral
constant-coefficient path folded vectors with ``float(np.sum(.))`` and returned
``0.0`` for any node it did not recognise (``sum(c * p) <= b`` got a zero
coefficient and was never robustified).

Every robust optimum here is checked against an independent oracle: the
counterpart over a polytope is exactly the model with one copy of each uncertain
row per vertex, so each vertex model is solved and compared, and the returned
point is verified feasible at every vertex.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, PolyhedralUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke

PBAR = np.array([1.0, 1.0])
DELTA = np.array([1.0, 1.0])
VERTICES = [PBAR + DELTA * np.array(s) for s in itertools.product((-1.0, 1.0), repeat=2)]

C_MIX = np.array([3.0, -1.0])
M_MIX = np.array([[3.0, -1.0], [-2.0, 1.0]])

# Each case maps the parameter (a Parameter, or a numpy vertex for the oracle) and
# the scalar decision variable x to the left-hand side of ``lhs <= 10``; ``S`` is
# the sum (``dm.sum`` in a model, ``np.sum`` for the numeric vertex check).
ROW_CASES = {
    "sum_c_times_p": lambda p, x, S=dm.sum: x + S(C_MIX * p),
    "sum_p_times_c": lambda p, x, S=dm.sum: x + S(p * C_MIX),
    "c_matmul_p": lambda p, x, S=dm.sum: x + C_MIX @ p,
    "p_matmul_c": lambda p, x, S=dm.sum: x + p @ C_MIX,
    "div_mixed": lambda p, x, S=dm.sum: x + S(p / np.array([1.0 / 3.0, -1.0])),
    "div_negative": lambda p, x, S=dm.sum: x + S(p / np.array([-1.0, -2.0])),
    "matrix_negative": lambda p, x, S=dm.sum: x + S(np.array([[-1.0, -1.0]]) @ p),
    "neg_of_mixed": lambda p, x, S=dm.sum: x - S(C_MIX * p),
    # Same-sign and scalar shapes: their behaviour is locked, not changed.
    "same_sign_pos": lambda p, x, S=dm.sum: x + S(np.array([3.0, 1.0]) * p),
    "same_sign_neg": lambda p, x, S=dm.sum: x + S(np.array([-3.0, -1.0]) * p),
    "scalar_coeffs": lambda p, x, S=dm.sum: x + 3.0 * p[0] - 1.0 * p[1],
    # Occurrences of one element that cancel: worst-casing each occurrence on its
    # own (sign tracking) over-counts, so these need the aggregate coefficient.
    "sum_matrix_matmul_p": lambda p, x, S=dm.sum: x + S(M_MIX @ p),
    "sum_p_matmul_matrix": lambda p, x, S=dm.sum: x + S(p @ M_MIX),
    "same_term_cancels": lambda p, x, S=dm.sum: x + S(C_MIX * p) - S(C_MIX * p),
    "scalar_cancel": lambda p, x, S=dm.sum: x + 3.0 * p[0] - p[0] - 2.0 * p[1],
    "negation_cancels": lambda p, x, S=dm.sum: x + S(C_MIX * p) + S(-(C_MIX * p)) + p[0],
}


def _uncertainty(kind, p):
    if kind == "box":
        return BoxUncertaintySet(p, delta=DELTA)
    # The same box as a polytope {xi : |xi_j| <= delta_j}.
    return PolyhedralUncertaintySet(p, A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4))


def _vertex_oracle(row):
    """max x s.t. row(v, x) <= 10 at every vertex v: the exact robust optimum."""
    m = dm.Model("oracle")
    x = m.continuous("x", lb=0, ub=20)
    for v in VERTICES:
        m.subject_to(row(v, x) <= 10)
    m.maximize(x)
    r = m.solve()
    assert r.status == "optimal"
    return float(r.value(x))


def _robust(kind, row):
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(row(p, x) <= 10)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    return float(r.value(x))


def test_issue_repro_box_gives_robust_optimum():
    """The exact repro from #1831: x = 4, robust at every box vertex."""
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0, 1.0]))
    m.subject_to(x + dm.sum(np.array([3.0, -1.0]) * p) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=np.array([1.0, 1.0]))).formulate()
    r = m.solve()
    xv = float(r.value(x))
    # Pre-fix: optimal at x = 6, which violates the row at p = (2, 0) (lhs 12).
    assert xv == pytest.approx(4.0, abs=1e-6)
    checks = 0
    for v in VERTICES:
        assert xv + C_MIX @ v <= 10 + 1e-6
        checks += 1
    assert checks == 4


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize("case", sorted(ROW_CASES))
def test_robust_row_matches_vertex_oracle(kind, case):
    row = ROW_CASES[case]
    xv = _robust(kind, row)
    assert xv == pytest.approx(_vertex_oracle(row), abs=1e-5)
    checks = 0
    for v in VERTICES:
        lhs = float(row(v, xv, np.sum))
        assert lhs <= 10 + 1e-5, f"{case}: not robust at vertex {v} (lhs {lhs})"
        checks += 1
    assert checks == len(VERTICES)


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize(
    ("row", "rhs", "expected"),
    [
        # sum(M @ p) == p0 for this M, so the worst case is pbar0 + delta0 = 2.
        (lambda p, x: x + dm.sum(M_MIX @ p), 7.0, 5.0),
        # The uncertain terms cancel identically: the row is x <= 10.
        (lambda p, x: x + dm.sum(C_MIX * p) - dm.sum(C_MIX * p), 10.0, 10.0),
    ],
    ids=["sum_M_at_p", "c_p_minus_c_p"],
)
def test_cancelling_occurrences_are_exact(kind, row, rhs, expected):
    """#1831 review blocker: the mixed-sign split was sound but over-counted.

    Sign tracking worst-cases every occurrence of ``p_j`` separately, so a row
    whose occurrences cancel was charged ``sum |coeff|`` instead of
    ``|sum coeff|``: the first case came back INFEASIBLE (main: 5, truth 5), the
    second 2 (main: 6, truth 10). The box counterpart now uses the aggregate
    coefficient of a row that is affine in ``p``.
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(row(p, x) <= rhs)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize(("rhs", "feasible"), [(3.5, False), (6.5, True)])
def test_parameter_only_matmul_vector_row(kind, rhs, feasible):
    """``M @ p <= rhs``: row 0 reaches ``3*2 - 0 = 6``, row 1 ``-2*0 + 2 = 2``.

    So ``3.5`` is robustly infeasible and ``6.5`` feasible. origin/main returned
    OPTIMAL for ``3.5`` on both kinds (the polyhedral evaluator folded the vector
    coefficient with ``np.sum``).
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(M_MIX @ p <= rhs)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    r = m.solve()
    if feasible:
        assert r.status == "optimal"
        assert float(r.value(x)) == pytest.approx(20.0, abs=1e-6)
    else:
        assert r.status == "infeasible"


def test_polyhedral_constant_coefficient_must_be_scalar():
    """The polyhedral coefficient evaluator refuses a vector, never folds it.

    A row that reaches it unsplit (a shape the splitter cannot determine) would
    otherwise get ``sum(coeff)`` as its coefficient -- the #1831 fold.
    """
    from discopt.modeling.core import Constant, SumExpression
    from discopt.ro.formulations.polyhedral import _eval_constant_expr

    assert _eval_constant_expr(SumExpression(Constant(np.array([3.0, -1.0])))) == 2.0
    with pytest.raises(NotImplementedError):
        _eval_constant_expr(Constant(np.array([3.0, -1.0])))


CERTAIN_CASES = {
    "q_times_p": lambda p, q, x, S=dm.sum: x + S(q * p),
    "p_times_q": lambda p, q, x, S=dm.sum: x + S(p * q),
    "affine_of_q_times_p": lambda p, q, x, S=dm.sum: x + S((q * 2.0 - 1.0) * p),
}


def _param_names(e, out):
    from discopt.modeling.core import Parameter

    if isinstance(e, Parameter):
        out.append(e.name)
    for attr in ("left", "right", "operand", "base"):
        if hasattr(e, attr):
            _param_names(getattr(e, attr), out)
    for t in getattr(e, "terms", []) or getattr(e, "args", []):
        _param_names(t, out)
    return out


@pytest.mark.parametrize("q_value", [[3.0, -1.0], [1.5, 0.75]], ids=["mixed_q", "uniform_q"])
@pytest.mark.parametrize("case", sorted(CERTAIN_CASES))
def test_certain_parameter_multiplier_is_signed_and_committed(case, q_value):
    """A certain Parameter multiplying ``p`` is signed by its value and committed.

    ``q = [3, -1]`` is mixed-sign, so each element of ``p`` needs its own worst
    case (origin/main was non-robust on all three). The ``c+ + c-`` split bakes
    ``q``'s value into the counterpart, which records a *value* commitment. A
    uniform-sign bare ``q`` stays live in the row -- freezing it silently made
    a second ``RobustCounterpart`` over ``q`` a no-op and a later update of ``q``
    ignored (#1839 review) -- under a *sign* commitment. A compound multiplier
    ``f(q)`` stays live under a value commitment.
    """
    row = CERTAIN_CASES[case]
    q_value = np.array(q_value)
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    q = m.parameter("q", value=q_value.copy())
    m.subject_to(row(p, q, x) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=DELTA)).formulate()

    mixed = bool(np.any(q_value < 0))
    names = _param_names(m._constraints[0].body, [])
    assert "p" not in names
    assert ("q" in names) == (not mixed)
    kind = "sign" if (not mixed and case != "affine_of_q_times_p") else "value"
    assert m._ro_commitments["q"]["kind"] == kind
    np.testing.assert_array_equal(m._ro_commitments["q"]["value"], q_value)

    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))

    def oracle_row(v, xx):
        return row(v, q_value, xx, np.sum)

    assert xv == pytest.approx(_vertex_oracle(oracle_row), abs=1e-5)
    checks = 0
    for v in VERTICES:
        assert float(oracle_row(v, xv)) <= 10 + 1e-5
        checks += 1
    assert checks == len(VERTICES)


# --- #1839 review: a second RobustCounterpart over a certain Parameter --------
#
# ``x + q*p <= 10``, max x, with p in [0.5, 1.5] (pass 1) and q in [1, 3] (pass
# 2): q*p ranges over [0.5, 4.5], so the robust truth is x = 5.5. The first
# version of this PR froze q into a Constant in pass 1, so pass 2 found no q and
# returned x = 7 -- non-robust -- in every variant.

TWO_PASS_FORMS = {
    "q*p": lambda p, q: dm.sum(q * p),
    "p*q": lambda p, q: dm.sum(p * q),
}


def _two_pass_model(q_value, form):
    m = dm.Model("two_pass")
    x = m.continuous("x", lb=-20, ub=20)
    p = m.parameter("p", value=np.ones_like(np.asarray(q_value, dtype=float)))
    q = m.parameter("q", value=q_value)
    m.subject_to(x + TWO_PASS_FORMS[form](p, q) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    return m, x, q


@pytest.mark.parametrize("form", sorted(TWO_PASS_FORMS))
@pytest.mark.parametrize(
    ("second", "q_value"),
    [("box", 2.0), ("box", np.array([2.0])), ("ellipsoidal", np.array([2.0]))],
    ids=["box_scalar", "box_vec", "ellipsoidal_vec"],
)
def test_second_pass_over_uniform_sign_multiplier_is_robust(second, q_value, form):
    from discopt.ro import EllipsoidalUncertaintySet

    m, x, q = _two_pass_model(q_value, form)
    assert m._ro_commitments["q"]["kind"] == "sign"
    if second == "box":
        u2 = BoxUncertaintySet(q, delta=1.0)
    else:
        u2 = EllipsoidalUncertaintySet(q, rho=1.0)
    RobustCounterpart(m, u2).formulate()
    # Pass 2 eliminated q and checked its whole set against the sign: no
    # commitment on it remains.
    assert not hasattr(m, "_ro_commitments")
    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))
    assert xv == pytest.approx(5.5, abs=1e-6)
    checks = 0
    for qq in (1.0, 3.0):
        for pp in (0.5, 1.5):
            assert xv + qq * pp <= 10 + 1e-6
            checks += 1
    assert checks == 4


def test_second_pass_over_set_crossing_zero_refuses():
    """Pass 1 chose p's worst case for q >= 0; q in [-1, 5] breaks that."""
    m, _x, q = _two_pass_model(2.0, "q*p")
    with pytest.raises(ValueError, match="crosses zero"):
        RobustCounterpart(m, BoxUncertaintySet(q, delta=3.0)).formulate()


@pytest.mark.parametrize(
    ("q_value", "row"),
    [
        # Mixed sign: the c+ / c- split bakes q's value. Pass 2 used to return
        # x = 8 here (truth 6), on origin/main as well.
        (np.array([2.0, -2.0]), lambda p, q: dm.sum(q * p)),
        # Cancelling row: the aggregate coefficient J = q bakes q's value.
        (np.array([2.0]), lambda p, q: dm.sum(q * p) + dm.sum(3.0 * p) - dm.sum(3.0 * p)),
    ],
    ids=["mixed_sign_q", "aggregate_cancel"],
)
def test_second_pass_over_baked_value_refuses(q_value, row):
    m = dm.Model("two_pass")
    x = m.continuous("x", lb=-20, ub=20)
    p = m.parameter("p", value=np.ones_like(q_value))
    q = m.parameter("q", value=q_value)
    m.subject_to(x + row(p, q) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    assert m._ro_commitments["q"]["kind"] == "value"
    with pytest.raises(ValueError, match="baked the value of Parameter 'q'"):
        RobustCounterpart(m, BoxUncertaintySet(q, delta=1.0)).formulate()


# --- #1839 review: updating a certain Parameter after formulate() -------------


def _update_model(row, q0):
    m = dm.Model("update")
    x = m.continuous("x", lb=-50, ub=50)
    p = m.parameter("p", value=PBAR.copy())
    q = m.parameter("q", value=np.array(q0, dtype=float))
    m.subject_to(row(p, q, x) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=np.array([0.5, 0.5]))).formulate()
    return m, x, q


UPDATE_ROWS = {
    "q*p": lambda p, q, x, S=dm.sum: x + S(q * p),
    "q*(x+p)": lambda p, q, x, S=dm.sum: S(q * (x + p)),
    "cancel_row": lambda p, q, x, S=dm.sum: x + S(q * p) + S(3.0 * p) - S(3.0 * p),
}


@pytest.mark.parametrize("case", ["q*p", "q*(x+p)"])
def test_same_sign_update_is_honoured(case):
    """The live multiplier sees ``q = [3, 3]``; the first version of this PR
    solved with the frozen ``[2, 2]`` (x = 4 for ``q*p``, truth 1)."""
    row = UPDATE_ROWS[case]
    m, x, q = _update_model(row, [2.0, 2.0])
    q.value = np.array([3.0, 3.0])
    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))
    worst = max(
        float(row(PBAR + 0.5 * np.array(s), np.array([3.0, 3.0]), xv, np.sum))
        for s in itertools.product((-1.0, 1.0), repeat=2)
    )
    assert worst == pytest.approx(10.0, abs=1e-5)


@pytest.mark.parametrize(
    ("case", "q0", "q1", "match"),
    [
        ("q*p", [2.0, 2.0], [-2.0, -2.0], "breaks that sign"),
        ("q*(x+p)", [2.0, 2.0], [2.0, -1.0], "breaks that sign"),
        ("q*p", [2.0, -2.0], [3.0, -2.0], "baked its value"),
        ("cancel_row", [2.0, 2.0], [3.0, 3.0], "baked its value"),
    ],
)
def test_commitment_breaking_update_refuses_to_solve(case, q0, q1, match):
    m, _x, q = _update_model(UPDATE_ROWS[case], q0)
    q.value = np.array(q1)
    with pytest.raises(ValueError, match=match):
        m.solve()
    from discopt.solver import solve_model

    with pytest.raises(ValueError, match=match):
        solve_model(m)
    # Restoring the committed value makes the model solvable again.
    q.value = np.array(q0)
    assert m.solve().status == "optimal"


@pytest.mark.parametrize("carrier", ["clone", "deepcopy", "pickle"])
def test_commitments_travel_with_the_model(carrier):
    import copy
    import pickle

    m, _x, _q = _update_model(UPDATE_ROWS["q*p"], [2.0, -2.0])
    if carrier == "clone":
        m2 = m.clone()
    elif carrier == "deepcopy":
        m2 = copy.deepcopy(m)
    else:
        m2 = pickle.loads(pickle.dumps(m))
    assert m2._ro_commitments == m._ro_commitments
    (q2,) = [prm for prm in m2._parameters if prm.name == "q"]
    q2.value = np.array([3.0, -2.0])
    with pytest.raises(ValueError, match="baked its value"):
        m2.solve()


def test_serialize_refuses_a_committed_model():
    """The format has no key for the commitments; dropping them would let a
    reloaded model solve after a commitment-breaking update."""
    from discopt.serialize import SerializationError, dumps, loads

    m, _x, _q = _update_model(UPDATE_ROWS["q*p"], [2.0, -2.0])
    with pytest.raises(SerializationError, match=r"certain Parameter\(s\) \['q'\]"):
        dumps(m)
    # A counterpart that committed nothing serializes as before.
    m2 = dm.Model("plain")
    x = m2.continuous("x", lb=0, ub=20)
    p = m2.parameter("p", value=PBAR.copy())
    m2.subject_to(x + dm.sum(C_MIX * p) <= 10)
    m2.maximize(x)
    RobustCounterpart(m2, BoxUncertaintySet(p, delta=DELTA)).formulate()
    assert not hasattr(m2, "_ro_commitments")
    assert loads(dumps(m2)).solve().status == "optimal"


# --- #1839 review: non-affine rows that also hold a cancelling affine part ----
#
# The affine extractor must refuse each non-affine node; a mutant that let one
# through (a FunctionCall read as ``{}``, ``p*p`` read as ``p + p``, ``abs``
# passed through, ``1/(p+c)`` or ``**`` read as their left operand) builds the
# aggregate counterpart from a wrong coefficient and returns a non-robust point.
# Each row is checked on a dense grid over the box, exact for these scalar rows.

NONAFFINE_ROWS = {
    "exp_plus_cancel": (0.0, 1.0, lambda p, x, F: x + F.exp(p) + 3.0 * p - 3.0 * p),
    "max_plus_cancel": (
        0.0,
        1.0,
        lambda p, x, F: x + F.maximum(p, -p) - 2.0 * p + 3.0 * p - 3.0 * p,
    ),
    "min_plus_cancel": (
        0.0,
        1.0,
        lambda p, x, F: x - F.minimum(p, -p) - 2.0 * p + 3.0 * p - 3.0 * p,
    ),
    "dm_abs_plus_cancel": (0.0, 1.0, lambda p, x, F: x + F.abs(p) - 2.0 * p),
    "builtin_abs_plus_cancel": (0.0, 1.0, lambda p, x, F: x + abs(p) - 2.0 * p),
    "p_times_p_plus_cancel": (1.0, 1.0, lambda p, x, F: x + 3.0 * p - 3.0 * p + p * p),
    "reciprocal_plus_cancel": (1.0, 1.0, lambda p, x, F: x + 1.0 / (p + 3.0) + 3.0 * p - 3.0 * p),
    "pow1_plus_cancel": (1.0, 1.0, lambda p, x, F: x + (p - 1.0) ** 1 + 3.0 * p - 3.0 * p),
    "pow2_plus_cancel": (1.0, 2.0, lambda p, x, F: x + (p - 1.0) ** 2 + 3.0 * p - 3.0 * p),
}


@pytest.mark.parametrize("case", sorted(NONAFFINE_ROWS))
def test_nonaffine_row_with_cancelling_part_stays_robust(case):
    pbar, delta, row = NONAFFINE_ROWS[case]
    m = dm.Model("nonaffine")
    x = m.continuous("x", lb=-50, ub=50)
    p = m.parameter("p", value=pbar)
    m.subject_to(row(p, x, dm) <= 6.0)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))
    grid = np.linspace(pbar - delta, pbar + delta, 401)
    checks = 0
    for v in grid:
        assert float(row(v, xv, np)) <= 6.0 + 1e-6, (case, v, xv)
        checks += 1
    assert checks == grid.size


@pytest.mark.parametrize("affine", [True, False], ids=["affine_row", "sign_tracked_row"])
@pytest.mark.parametrize("divisor", [[2.0, 0.0], [-2.0, 0.0]], ids=["pos", "neg"])
def test_uniform_sign_divisor_with_zero_refuses(divisor, affine):
    """A zero entry in a constant divisor of ``p`` is refused on both routes.

    Before, only a *mixed*-sign divisor was checked; a one-signed divisor with a
    zero entry was folded to that sign and rebuilt. ``exp`` keeps the row off the
    affine route so the sign-tracking check is exercised too.
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    term = p / np.array(divisor)
    m.subject_to(x + dm.sum(term if affine else dm.exp(term)) <= 10)
    m.maximize(x)
    with pytest.raises(ValueError, match="zero entries"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=DELTA)).formulate()


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize("form", ["x + M@p", "M@p + x*ones"])
def test_vector_matrix_row_each_row_exact(kind, form):
    """``x + M @ p <= 10`` row-wise: each row worst-cased by its own signs.

    A ``@`` row had no statically known shape, so it was not split into scalar
    rows. The box counterpart is exact for it anyway (the ``c+ + c-`` split is
    row-wise), but the polyhedral one gave the whole vector row one set of
    duals and reported this feasible model infeasible. ``_shape_of`` now shapes
    matmul and reduction bodies, so both are split per element.
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    body = x + M_MIX @ p if form == "x + M@p" else M_MIX @ p + x * np.ones(2)
    m.subject_to(body <= 10)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    assert len(m._constraints) >= 2  # split per element
    r = m.solve()
    assert r.status == "optimal"
    # Row 0: max 3p0 - p1 = 6 -> x <= 4. Row 1: max -2p0 + p1 = 2 -> x <= 8.
    xv = float(r.value(x))
    assert xv == pytest.approx(4.0, abs=1e-5)
    checks = 0
    for v in VERTICES:
        assert xv + float(np.max(M_MIX @ v)) <= 10 + 1e-5
        checks += 1
    assert checks == len(VERTICES)


@pytest.mark.parametrize(
    "rhs,objective,expected",
    [
        # Pre-#1831 sign tracking certified y = (0, 2): lhs 2 > 1 at p = -1.
        (1.0, "y1 - y0", [1.0 / 3.0, 2.0]),
        (1.0, "y0 + y1", [1.0, 2.0]),
    ],
)
def test_mixed_constant_over_variable_coefficient_is_exact(rhs, objective, expected):
    """``sum([3, -1] * (p * y))``: the coefficient of ``p`` is ``3 y0 - y1``.

    Its sign depends on the decision, so the ``c+ + c-`` split would pick two
    different values of the one scalar ``p`` -- sound but conservative
    (objective 1 instead of 5/3 for the first case). The box counterpart routes
    the term to the |coeff| linearization, which is exact; checked against the
    vertex model (one row per endpoint of ``p``).
    """

    def build(model, y, pv=None):
        p = pv if pv is not None else model.parameter("p", value=np.array(0.0))
        model.subject_to(dm.sum(C_MIX * (p * y)) <= rhs)
        return p

    def obj(y):
        return y[1] - y[0] if objective == "y1 - y0" else y[0] + y[1]

    m = dm.Model("ro")
    y = m.continuous("y", lb=0, ub=2, shape=(2,))
    p = build(m, y)
    m.maximize(obj(y))
    RobustCounterpart(m, BoxUncertaintySet(p, delta=1.0)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    yv = np.asarray(r.value(y), dtype=float)

    o = dm.Model("oracle")
    yo = o.continuous("y", lb=0, ub=2, shape=(2,))
    for pv in (-1.0, 1.0):
        o.subject_to(dm.sum(C_MIX * (pv * yo)) <= rhs)
    o.maximize(obj(yo))
    ro = o.solve()
    assert ro.status == "optimal"
    assert r.objective == pytest.approx(ro.objective, abs=1e-5)
    assert yv == pytest.approx(expected, abs=1e-5)
    checks = 0
    for pv in (-1.0, 1.0):
        assert pv * float(C_MIX @ yv) <= rhs + 1e-6
        checks += 1
    assert checks == 2


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
def test_vector_row_mixed_sign_each_element(kind):
    """``x + c * p <= 10`` elementwise: row j needs p_j at its own worst case."""
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(x + np.array([3.0, -4.0]) * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    r = m.solve()
    # Row 0: 3*p0 <= 6 -> x <= 4. Row 1: -4*p1 <= 0 -> x <= 10.
    assert float(r.value(x)) == pytest.approx(4.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize(("rhs", "feasible"), [(5.5, False), (6.5, True)])
def test_parameter_only_row(kind, rhs, feasible):
    """A row with no variable reaches the constant-coefficient path.

    ``max over the box of 3*p0 - p1`` is 6, so ``<= 5.5`` is robustly infeasible
    and ``<= 6.5`` is feasible. Pre-fix both kinds returned ``optimal`` for 5.5:
    the box counterpart signed ``[3, -1]`` as ``+1`` (worst case 4) and the
    polyhedral evaluator returned a 0.0 coefficient for the ``SumExpression``.
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(dm.sum(C_MIX * p) <= rhs)
    m.maximize(x)
    RobustCounterpart(m, _uncertainty(kind, p)).formulate()
    r = m.solve()
    if feasible:
        assert r.status == "optimal"
        assert float(r.value(x)) == pytest.approx(20.0, abs=1e-6)
    else:
        assert r.status == "infeasible"


def test_box_objective_mixed_sign_cost():
    """min max_p  sum([3, -1] * p) + x  over the box: worst cost is 6 + x."""
    m = dm.Model("ro")
    x = m.continuous("x", lb=1, ub=2)
    p = m.parameter("p", value=PBAR.copy())
    m.minimize(dm.sum(C_MIX * p) + x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=DELTA)).formulate()
    r = m.solve()
    # Pre-fix: both p at their upper bound, objective 3*2 - 2 + 1 = 5.
    assert r.objective == pytest.approx(7.0, abs=1e-6)


def test_same_sign_formulation_unchanged():
    """A same-sign constant keeps the single ``c * p_wc`` node (no split)."""
    from discopt.modeling.core import BinaryOp, Constant, SumExpression

    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(x + dm.sum(np.array([3.0, 1.0]) * p) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=DELTA)).formulate()
    body = m._constraints[0].body  # (x + sum(c * p)) - 10
    s = body.left.right
    assert isinstance(s, SumExpression)
    prod = s.operand
    assert isinstance(prod, BinaryOp) and prod.op == "*"
    assert isinstance(prod.left, Constant) and isinstance(prod.right, Constant)
    np.testing.assert_array_equal(prod.left.value, [3.0, 1.0])
    np.testing.assert_array_equal(prod.right.value, PBAR + DELTA)


def test_eval_constant_expr_full_value_and_refusal():
    from discopt.modeling.core import Constant, IndexExpression, SumExpression
    from discopt.ro.formulations._common import eval_constant_expr

    c = Constant(np.array([3.0, -1.0]))
    np.testing.assert_array_equal(eval_constant_expr(c), [3.0, -1.0])
    assert float(eval_constant_expr(SumExpression(c * Constant(np.array([2.0, 2.0]))))) == 4.0
    assert float(eval_constant_expr(IndexExpression(c, 1))) == -1.0

    m = dm.Model("v")
    x = m.continuous("x")
    with pytest.raises(NotImplementedError, match="cannot evaluate"):
        eval_constant_expr(x)


def test_polyhedral_unbounded_support_refuses():
    """An unbounded support LP has no finite worst case: refuse, never guess.

    Pre-fix the constant-coefficient path fell back to ``sum|c| * max|b|`` = 4 on
    any non-optimal LP status, so a polytope unbounded above (``xi >= -1``) gave
    a finite "worst case" and the row was certified robustly feasible.
    """
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=PBAR.copy())
    m.subject_to(dm.sum(np.array([3.0, 1.0]) * p) <= 100)
    m.maximize(x)
    rc = RobustCounterpart(m, PolyhedralUncertaintySet(p, A=-np.eye(2), b=np.ones(2)))
    with pytest.raises(ValueError, match="support LP"):
        rc.formulate()
