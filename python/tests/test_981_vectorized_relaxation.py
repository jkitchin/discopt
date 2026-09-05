"""A vectorized model must get the same relaxation as its scalar transcription (#981).

A ``Constraint`` body may be array-valued -- ``A @ x == b`` on a 3-vector is one
``Constraint`` object standing for three scalar rows. The relaxation engine
assumed one scalar row per ``Constraint``, so the interval walk hit an array
where it expected a float, raised ``TypeError: only 0-dimensional arrays can be
converted to Python scalars``, and the caller degraded to *no relaxation at
all*. The dual bound then fell back to the trivial objective floor -- exactly
``0.0`` for a sum of squares -- against which the relative gap is 100 %, so no
time limit could ever certify.

Measured on ``docs/notebooks/tutorial_dae.ipynb``'s parameter-estimation cell:
493 s to report ``feasible`` with ``bound=0.0``. The same cell now certifies
``optimal`` in ~10 s. These tests pin each link in that chain: the scalarizer is
value-preserving, a vectorized model's bound matches its scalar transcription,
and a scalar model is left byte-identical.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax.canonical_expr import canonicalize
from discopt._relax.dag_compiler import compile_expression
from discopt._relax.scalarize import scalar_elements, static_shape
from discopt._relax.term_classifier import classify_nonlinear_terms


def _dae_parameter_estimation(nfe: int = 5, ncp: int = 3, ub: float = 0.6) -> tuple[Model, object]:
    """``tutorial_dae``'s parameter-estimation model, shrunk but structurally identical."""
    from discopt.dae import ContinuousSet, DAEBuilder

    t_data = np.array([1.0, 2.0, 3.0])
    x_data = np.exp(-0.5 * t_data) + np.array([0.01, -0.02, 0.015])
    m = Model("pe")
    cs = ContinuousSet("t", bounds=(0, 3), nfe=nfe, ncp=ncp)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(0, 2))
    k = m.continuous("k", lb=0.4, ub=ub)
    dae.set_ode(lambda t, states, alg, ctrl: {"x": -k * states["x"]})
    dae.discretize()
    m.minimize(dae.least_squares("x", t_data, x_data))
    return m, k


# --------------------------------------------------------------------------- #
# The scalarizer itself: every element must evaluate to the array element
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_scalarization_is_value_preserving_on_every_supported_node():
    """``scalar_elements(e)[i]`` evaluates to ``np.ravel(e)[i]`` at every point.

    The rewrite is the load-bearing part of the fix: an element that does not
    reproduce its array position would put a *wrong* row into the relaxation,
    which is a soundness failure, not a bound-quality one. Checked numerically
    against the same DAG compiler the solver evaluates with.
    """
    rng = np.random.default_rng(0)
    m = Model("t")
    X = m.continuous("X", lb=-2, ub=2, shape=(2, 3))
    y = m.continuous("y", lb=-2, ub=2, shape=(3,))
    k = m.continuous("k", lb=-2, ub=2)
    A = rng.normal(size=(3, 2))

    cases = {
        "scalar_times_slice": (-k) * X[:, 1:],
        "matmul_2d": X @ A,
        "matmul_mat_vec": X @ y,
        "matmul_vec_mat": y @ A,
        "matmul_vec_vec": y @ np.arange(3.0),
        "broadcast": X * y,
        "elementwise_call": dm.exp(X) * k,
        "power": (X - 0.5) ** 2,
        "division": X / (k + 3.0),
        "sliced_vector": y[1:] * k,
        "reduction": dm.sum(X) * k,
        # The DAE collocation body's exact shape: matmul minus a broadcast
        # scalar-times-slice product. Its BinaryOp caches shape ``None`` (matmul
        # is outside the M8 shape guard), which is why the scalarizer must
        # recompute shapes bottom-up rather than trust the cached one.
        "collocation_row": (X @ A) - (np.ones((2, 2)) * ((-k) * X[:, 1:])),
    }

    n = sum(v.size for v in m._variables)
    element_checks = 0
    for name, expr in cases.items():
        rows = scalar_elements(expr)
        assert rows is not None, f"{name}: expected a static scalarization"
        whole = compile_expression(expr, m)
        parts = [compile_expression(r, m) for r in rows]
        for _ in range(3):
            x = rng.uniform(-2, 2, size=n)
            ref = np.ravel(np.asarray(whole(x)))
            got = np.array([float(np.asarray(p(x))) for p in parts])
            assert ref.shape == got.shape, f"{name}: {ref.shape} vs {got.shape}"
            assert np.allclose(ref, got, rtol=0, atol=1e-12), f"{name}: {ref} vs {got}"
            element_checks += ref.size

    # CLAUDE.md 6 -- a probe that silently compares nothing reads as a pass.
    assert element_checks == 132, f"expected 132 element comparisons, made {element_checks}"


@pytest.mark.smoke
def test_unscalarizable_expression_reports_unknown_not_empty():
    """An expression the rewrite cannot expand must return ``None``.

    ``None`` means "caller keeps its previous behaviour". Returning ``[]`` would
    silently drop every row of that constraint from the relaxation -- sound, but
    a weakened bound with no record, which is precisely how #981 stayed
    invisible for three and a half months.
    """
    m = Model("u")
    X = m.continuous("X", lb=0, ub=1, shape=(2, 3))

    # An axis-reduction is array-valued. Its shape used to be reported as
    # *unknown* here so the walk would not re-derive numpy's axis semantics; #1160
    # made those semantics load-bearing (a walker that treats the node as scalar
    # collapses rows and certifies the wrong model), so they are derived in one
    # place now -- `scalarize.sum_result_shape` -- and this is that answer.
    axis_reduction = dm.sum(X, axis=0)
    assert static_shape(axis_reduction) == (3,)

    # The contract this test exists for is unchanged: the rewrite still cannot
    # EXPAND an axis reduction (``_elem`` has no ``SumExpression`` case), so
    # callers keep their previous behaviour rather than losing rows.
    assert scalar_elements(axis_reduction) is None


# --------------------------------------------------------------------------- #
# Vectorized vs scalar transcription: same bound, same terms
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_vectorized_bilinear_matches_its_scalar_transcription():
    """Same model, two spellings -- the bound must not depend on the spelling.

    Before the fix the vectorized arm returned ``feasible`` with ``bound=0.0``
    while the scalar arm certified ``optimal`` at 4.5.
    """

    def build(vectorized: bool) -> Model:
        m = Model("v")
        x = m.continuous("x", lb=0.0, ub=4.0, shape=(3,))
        k = m.continuous("k", lb=0.5, ub=2.0)
        rhs = np.array([2.0, 3.0, 4.0])
        if vectorized:
            m.subject_to(k * x - rhs == 0.0)
        else:
            for i in range(3):
                m.subject_to(k * x[i] - float(rhs[i]) == 0.0)
        m.minimize(x[0] + x[1] + x[2])
        return m

    vec, scalar = build(True), build(False)

    # The classifier must see the same product structure through both spellings,
    # or the vectorized model gets no partition candidates and cannot branch.
    assert classify_nonlinear_terms(vec).bilinear == classify_nonlinear_terms(scalar).bilinear
    assert (
        classify_nonlinear_terms(vec).partition_candidates
        == classify_nonlinear_terms(scalar).partition_candidates
    )

    r_vec = vec.solve(time_limit=60.0)
    r_scalar = scalar.solve(time_limit=60.0)
    assert r_vec.status == r_scalar.status == "optimal"
    assert r_vec.objective == pytest.approx(r_scalar.objective, rel=1e-6)
    assert r_vec.bound == pytest.approx(r_scalar.bound, rel=1e-6)


@pytest.mark.smoke
def test_vectorized_constraint_expands_to_one_canonical_row_per_element():
    """The canonical DAG carries scalar rows, matching the NLP evaluator's
    convention (one row per flat element -- see ``validation/feasibility``)."""
    m = Model("rows")
    x = m.continuous("x", lb=0.0, ub=4.0, shape=(3,))
    k = m.continuous("k", lb=0.5, ub=2.0)
    m.subject_to(k * x - np.array([2.0, 3.0, 4.0]) == 0.0)  # 1 Constraint, 3 rows
    m.subject_to(x[0] + x[1] <= 5.0)  # already scalar
    m.minimize(x[0])

    dag = canonicalize(m)
    assert len(m._constraints) == 2
    assert len(dag.constraints) == 4
    assert dag.constraint_index == (0, 0, 0, 1)


@pytest.mark.smoke
def test_scalar_model_canonicalization_is_unchanged():
    """Bound-neutrality guard: a model with no array-valued body must expand to
    exactly its own constraints, with the SAME expression objects."""
    m = Model("scalar")
    x = m.continuous("x", lb=0.0, ub=4.0, shape=(3,))
    for i in range(3):
        m.subject_to(x[i] * x[(i + 1) % 3] <= 2.0)
    m.minimize(x[0] + x[1] + x[2])

    dag = canonicalize(m)
    assert len(dag.constraints) == len(m._constraints)
    assert dag.constraint_index == (0, 1, 2)
    for con, body in zip(m._constraints, dag.constraint_exprs):
        assert body is con.body  # identity, not a rebuilt copy


# --------------------------------------------------------------------------- #
# The reported instance
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_dae_collocation_certifies_a_real_bound():
    """The #981 instance: a DAE parameter-estimation model must certify.

    Before the fix this returned ``feasible`` with a dual bound of exactly 0.0 no
    matter the time limit. The bound assertion is the one that fails on the old
    code; the ``optimal`` status alone would not, since a small enough objective
    satisfies the ABSOLUTE gap tolerance against a zero bound.
    """
    m, k = _dae_parameter_estimation()
    result = m.solve(time_limit=120.0)

    assert result.status == "optimal"
    assert result.gap_certified
    assert result.bound > 0.0, "dual bound collapsed to the trivial sum-of-squares floor"
    assert result.bound <= result.objective + 1e-9, "dual bound above the incumbent"
    assert float(result.value(k)) == pytest.approx(0.5, abs=0.1)


@pytest.mark.slow
def test_dae_bound_survives_a_narrowed_parameter_box():
    """Box-narrowing discriminator from the issue: with ``k`` pinned to a
    width-1e-4 box the model is effectively convex in the states, so the bound
    must land on the optimum. It stayed at 0.0 on the old code -- the evidence
    that this was a defect rather than relaxation weakness."""
    m, _ = _dae_parameter_estimation(ub=0.4001)
    result = m.solve(time_limit=120.0)

    assert result.bound > 0.0
    assert result.bound == pytest.approx(result.objective, rel=1e-3)
