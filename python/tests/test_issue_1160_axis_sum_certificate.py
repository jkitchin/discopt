"""Issue #1160: a constraint on ``dm.sum(X, axis=k)`` must not be solved as the
axis-*collapsed* model.

``dm.sum(A, axis=1)`` on a ``(2, 3)`` variable is **two** row sums; the relaxation
side folded it into one sum of six terms, so ``sum(A, axis=1) <= 2`` was solved as
``sum(A) <= 2`` — a strictly smaller feasible set — and the collapsed optimum was
returned with ``status="optimal"`` and a dual bound that excludes the true one.
Before the fix the reproducer below certified ``-2`` (bound ``-2``) on a model
whose true optimum is ``-4``, with discopt's own ``verify_point`` confirming the
``-4`` point feasible.

Every objective assertion here is paired with a **bound** assertion: for a
minimize, a valid dual bound must be ``<= true optimum``, and the objective alone
would pass on a build that merely got lucky on the primal. Each case also carries
an independently verified feasible witness at the true optimum, so "the solver and
the test agree" cannot be an agreement between two copies of the same mistake.

The `.nl` reader emits no ``SumExpression`` at all, so no `.nl` panel or gate
covers this class — these models are the coverage.
"""

from __future__ import annotations

import math

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._relax import problem_classifier as pc
from discopt._relax.scalarize import scalar_elements
from discopt.validation.feasibility import verify_point

# ``sum_result_shape`` / ``sum_is_full_reduction`` are imported inside the two
# tests that exercise them directly, not at module scope: a module-level import
# makes the whole file uncollectable on a tree without the fix, which downgrades
# the fail-before evidence for the solve tests from "certifies -2 against a true
# -4" to "ImportError".

TOL = 1e-4

#: ``_ENTERED`` counts calls into :func:`_assert_min_certificate`; ``_COMPARISONS``
#: counts the ones that reached the objective/bound comparison. A guard that skips
#: its own assertions reports a pass while measuring nothing (CLAUDE.md §6), so
#: teardown asserts the two agree — and that they are non-zero whenever any solve
#: test ran at all (all of them being deselected by a marker expression is not the
#: failure this guards against). It is a ``teardown_module`` hook and not a test on
#: purpose: as a test it would be position-dependent under ``pytest-randomly``.
_ENTERED = [0]
_COMPARISONS = [0]


def teardown_module(module):  # noqa: ANN001, ANN201 - pytest hook
    assert _COMPARISONS[0] == _ENTERED[0], (
        f"{_ENTERED[0] - _COMPARISONS[0]} certificate check(s) entered but never "
        "reached their objective/bound comparison"
    )
    if _ENTERED[0]:
        assert _COMPARISONS[0] > 0, "solve tests ran but measured nothing"


def _solve(model):
    return model.solve(time_limit=120.0, gap_tolerance=1e-6, max_nodes=200_000)


def _assert_min_certificate(model, true_opt: float, witness: np.ndarray, name: str) -> None:
    """The witness is feasible at ``true_opt``, and the solve certifies it.

    Asserts, for a *minimize* model: the witness is genuinely feasible with the
    claimed objective; the returned objective equals the true optimum; and the
    dual bound is ``<= true_opt`` — a bound above the optimum has fathomed it
    away, which is the #1160 failure even when the objective happens to be right.
    """
    _ENTERED[0] += 1
    scale = max(1.0, abs(true_opt))

    v = verify_point(model, witness, with_objective=True)
    assert v.ok, f"[{name}] the witness for the true optimum is not feasible: {v.reason}"
    assert v.objective == pytest.approx(true_opt, abs=TOL * scale), (
        f"[{name}] witness objective {v.objective} != claimed true optimum {true_opt}"
    )

    result = _solve(model)
    _COMPARISONS[0] += 1
    assert result.status in ("optimal", "feasible"), f"[{name}] status={result.status}"
    assert result.objective == pytest.approx(true_opt, abs=TOL * scale), (
        f"[{name}] objective {result.objective} != true optimum {true_opt} "
        "(the axis-collapsed model's answer?)"
    )
    if result.bound is not None:
        assert result.bound <= true_opt + TOL * scale, (
            f"[{name}] dual bound {result.bound} is ABOVE the true optimum {true_opt}: "
            "the certificate excludes the optimum"
        )


# --------------------------------------------------------------------------- #
# The reproducer from the issue, and its neighbours
# --------------------------------------------------------------------------- #


@pytest.mark.pr_correctness
def test_row_capped_axis_sum_is_not_solved_as_the_total_cap():
    """min -sum(A) s.t. sum(A, axis=1) <= 2 — two row caps, optimum -4.

    Collapsed to ``sum(A) <= 2`` this certifies -2 (issue #1160's reproducer).
    """
    m = dm.Model("sum_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(-dm.sum(A))
    _assert_min_certificate(m, -4.0, np.array([1.0, 1, 0, 1, 1, 0]), "sum_axis1")


@pytest.mark.correctness
def test_column_capped_axis_sum_reduces_the_other_axis():
    """axis=0 caps each of the 3 columns at 1 -> optimum -3, collapsed gives -1."""
    m = dm.Model("sum_axis0")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=0) <= 1)
    m.minimize(-dm.sum(A))
    _assert_min_certificate(m, -3.0, np.array([1.0, 1, 1, 0, 0, 0]), "sum_axis0")


@pytest.mark.correctness
def test_axis_sum_on_a_three_dimensional_variable():
    """A (2,2,2) variable reduced on axis 2: four caps of 1 -> optimum -4."""
    m = dm.Model("sum_axis_3d")
    A = m.continuous("A", shape=(2, 2, 2), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=2) <= 1)
    m.minimize(-dm.sum(A))
    _assert_min_certificate(m, -4.0, np.array([1.0, 0, 1, 0, 1, 0, 1, 0]), "sum_axis_3d")


@pytest.mark.correctness
def test_axis_sum_with_an_equality_sense():
    """== is the same class: each row sums to exactly 2, maximizing sum(A*A)."""
    m = dm.Model("sum_axis_eq")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) == 2)
    m.minimize(-dm.sum(A * A))
    _assert_min_certificate(m, -4.0, np.array([1.0, 1, 0, 1, 1, 0]), "sum_axis_eq")


@pytest.mark.pr_correctness
def test_integer_model_with_an_axis_sum():
    """MILP: binaries with a per-row cap of 2 -> optimum -4, collapsed gives -2."""
    m = dm.Model("sum_axis_milp")
    B = m.binary("B", shape=(2, 3))
    m.subject_to(dm.sum(B, axis=1) <= 2)
    m.minimize(-dm.sum(B))
    _assert_min_certificate(m, -4.0, np.array([1.0, 1, 0, 1, 1, 0]), "sum_axis_milp")


@pytest.mark.pr_correctness
def test_nonlinear_axis_sum_keeps_its_rows():
    """min -sum(A) s.t. sum(A*A, axis=1) <= 1/2 on A in [0,1]^(2x3).

    The linear class can be fixed by a term collector the nonlinear path does not
    share, so this case is not redundant. Per row, Cauchy–Schwarz gives
    ``sum(a) <= sqrt(3 * 1/2)`` with equality iff all three are equal at
    ``sqrt(1/6) <= 1``; two rows give ``6 * sqrt(1/6) = sqrt(6)``.
    """
    m = dm.Model("sum_axis_quadratic")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A * A, axis=1) <= 0.5)
    m.minimize(-dm.sum(A))
    _assert_min_certificate(
        m, -math.sqrt(6.0), np.full(6, math.sqrt(1.0 / 6.0)), "sum_axis_quadratic"
    )


@pytest.mark.correctness
def test_transcendental_axis_sum_keeps_its_rows():
    """sum(exp(A), axis=1) <= 1 + e per row; optimum -4*log((1+e)/2)."""
    a = math.log((1.0 + math.e) / 2.0)
    m = dm.Model("sum_axis_exp")
    A = m.continuous("A", shape=(2, 2), lb=0, ub=1)
    m.subject_to(dm.sum(dm.exp(A), axis=1) <= 1.0 + math.e)
    m.minimize(-dm.sum(A))
    _assert_min_certificate(m, -4.0 * a, np.full(4, a), "sum_axis_exp")


# --------------------------------------------------------------------------- #
# Controls: the fix must not refuse a *full* reduction
# --------------------------------------------------------------------------- #


@pytest.mark.pr_correctness
def test_full_reduction_still_takes_the_algebraic_fast_path():
    """A plain ``sum(x)`` body is still one algebraic row — no over-refusal.

    Without this control the guard could have been "abstain on every
    ``SumExpression``", which is sound and would silently retire the fast
    extractor for every model in the corpus.
    """
    m = dm.Model("plain_sum")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    m.subject_to(dm.sum(x) <= 2)
    m.minimize(-dm.sum(x))

    terms, const = pc._extract_linear_coefficients_sparse(m._constraints[0].body, m, 4)
    assert terms == {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    assert const == pytest.approx(-2.0)

    lp = pc.extract_lp_data_algebraic(m)
    assert np.asarray(pc.dense_A(lp.A_eq)).shape[0] == 1

    _assert_min_certificate(m, -2.0, np.array([1.0, 1.0, 0.0, 0.0]), "plain_sum")


@pytest.mark.correctness
def test_axis_sum_under_an_outer_full_reduction_still_collapses():
    """``sum(sum(A, axis=1))`` sums every element, so the collapse is exact there.

    The guard is on *scalar position*, not on the node type: an enclosing full
    reduction sums the partial sums' elements with the same uniform scale, which
    is summing the operand's elements.
    """
    m = dm.Model("nested_sum")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(dm.sum(A, axis=1)) <= 2)
    m.minimize(-dm.sum(A))

    terms, const = pc._extract_linear_coefficients_sparse(m._constraints[0].body, m, 6)
    assert terms == dict.fromkeys(range(6), 1.0)
    assert const == pytest.approx(-2.0)

    _assert_min_certificate(m, -2.0, np.array([1.0, 1, 0, 0, 0, 0]), "nested_sum")


@pytest.mark.correctness
def test_axis_zero_on_a_one_dimensional_operand_is_a_full_reduction():
    """``sum(x, axis=0)`` on a 1-D variable IS scalar; it must stay correct."""
    m = dm.Model("axis0_1d")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    m.subject_to(dm.sum(x, axis=0) <= 2)
    m.minimize(-dm.sum(x))
    _assert_min_certificate(m, -2.0, np.array([1.0, 1.0, 0.0, 0.0]), "axis0_1d")


# --------------------------------------------------------------------------- #
# The shared predicate and the guards that consume it
# --------------------------------------------------------------------------- #


def _model_with_shapes():
    m = dm.Model("shapes")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    return m, A, x


def test_sum_result_shape_reduces_only_the_named_axis():
    from discopt._relax.scalarize import sum_is_full_reduction, sum_result_shape

    _m, A, x = _model_with_shapes()
    cases = [
        (dm.sum(A), ()),
        (dm.sum(A, axis=0), (3,)),
        (dm.sum(A, axis=1), (2,)),
        (dm.sum(A, axis=-1), (2,)),
        (dm.sum(A * A, axis=1), (2,)),
        (dm.sum(x, axis=0), ()),
        (dm.sum(dm.sum(A, axis=1)), ()),
        (dm.sum(dm.sum(A, axis=1), axis=0), ()),
        (dm.sum(A, axis=5), None),  # numpy would raise: no guess
    ]
    for expr, expected in cases:
        assert sum_result_shape(expr) == expected, f"{expr!r} -> {sum_result_shape(expr)!r}"
        assert sum_is_full_reduction(expr) is (expected == ())
    assert len(cases) == 9


def test_the_derived_shape_matches_what_the_evaluator_actually_produces():
    """The predicate is checked against the model's own row fan-out, not asserted.

    ``sum_result_shape`` re-derives numpy's axis semantics, so it has to be
    measured against the thing that defines them here: the evaluator, which emits
    one row per flat element of a constraint body. A predicate that disagreed
    with the evaluator would put the guards on the wrong side of the fold.
    """
    from discopt._relax.scalarize import sum_result_shape
    from discopt._tape_nlp_evaluator import try_build

    checked = 0
    for shape, axis in [((2, 3), 1), ((2, 3), 0), ((2, 3), -1), ((2, 2, 2), 2), ((4,), 0)]:
        m = dm.Model("fanout")
        A = m.continuous("A", shape=shape, lb=0, ub=1)
        body_sum = dm.sum(A, axis=axis)
        m.subject_to(body_sum <= 1)
        m.minimize(-dm.sum(A))

        derived = sum_result_shape(body_sum)
        assert derived == np.sum(np.zeros(shape), axis=axis).shape

        ev = try_build(m)
        assert ev is not None, f"tape evaluator declined shape={shape} axis={axis}"
        rows = np.asarray(
            ev.evaluate_constraints(np.zeros(int(np.prod(shape)))), dtype=np.float64
        ).reshape(-1)
        assert rows.size == max(1, int(np.prod(derived))), (
            f"shape={shape} axis={axis}: derived {derived} but the evaluator "
            f"emitted {rows.size} row(s)"
        )
        checked += 1
    assert checked == 5


def test_scalar_elements_still_declines_an_axis_reduced_body():
    """`static_shape` got sharper; `scalar_elements` must not silently change.

    ``_elem`` has no ``SumExpression`` case, so an axis-reduced body is still
    "not statically scalarizable" — callers keep their previous path.
    """
    _m, A, _x = _model_with_shapes()
    assert scalar_elements(dm.sum(A, axis=1)) is None
    assert scalar_elements(dm.sum(A, axis=1) - 2) is None


def test_linear_extractor_refuses_an_axis_reduced_body():
    m, A, _x = _model_with_shapes()
    body = (dm.sum(A, axis=1) <= 2).body
    with pytest.raises(pc._NotLinearError):
        pc._extract_linear_coefficients_sparse(body, m, 6)


def test_quadratic_extractor_refuses_an_axis_reduced_body():
    m, A, _x = _model_with_shapes()
    body = (dm.sum(A * A, axis=1) <= 2).body
    with pytest.raises((pc._NotQuadraticError, pc._NotLinearError)):
        pc._extract_quadratic_terms(body, m, 6)


def test_lp_extraction_emits_one_row_per_surviving_element():
    """``extract_lp_data`` must fan the body out, not collapse it.

    The repr rung probes the Rust arena at unit vectors and reduces a constraint
    to ONE row; that rung is what produced the collapsed LP. It now declines (see
    the NaN test below) and the tape rung supplies two rows with two right-hand
    sides.
    """
    m = dm.Model("rows")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(-dm.sum(A))

    lp = pc.extract_lp_data(m)
    A_eq = np.asarray(pc.dense_A(lp.A_eq))
    assert A_eq.shape[0] == 2, f"expected one LP row per row of A, got {A_eq.shape}"
    assert np.allclose(np.asarray(lp.b_eq), [2.0, 2.0])
    # Row 0 must touch A[0, :] only, row 1 A[1, :] only — a collapsed row would
    # carry all six original columns.
    assert np.allclose(A_eq[0, :6], [1, 1, 1, 0, 0, 0])
    assert np.allclose(A_eq[1, :6], [0, 0, 0, 1, 1, 1])


def test_rust_repr_reports_an_axis_sum_as_not_scalar_representable():
    """The scalar Rust evaluator answers NaN, never the full sum.

    A number here is a wrong answer to a different model, and every repr-based
    extractor accepts it as a coefficient; NaN is the arena's existing
    "not scalar-representable" signal and makes those extractors decline.
    """
    from discopt._rust import model_to_repr

    m = dm.Model("repr_nan")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)  # constraint 0: array-valued
    m.subject_to(dm.sum(A) <= 5)  # constraint 1: full reduction, scalar
    m.minimize(-dm.sum(A))

    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    x = np.full(6, 0.5)
    assert math.isnan(repr_.evaluate_constraint(0, x))
    assert repr_.evaluate_constraint(1, x) == pytest.approx(3.0 - 5.0)
    assert repr_.evaluate_objective(x) == pytest.approx(-3.0)
