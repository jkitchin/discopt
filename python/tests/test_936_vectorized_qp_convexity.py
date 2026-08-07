"""#936: a convex QP written with the vectorized / indexed-summation API was
certified convex by no route, so it fell through to spatial McCormick B&B.

Three independent defects in the convexity-certification layer stacked up:

1. ``milp_relaxation._expr_to_polynomial`` had no ``SumOverExpression`` branch,
   so ``extract_quadratic`` abstained on every body written with
   ``dm.sum(f, over=...)`` / ``dm.sum(term for ...)`` — while the byte-identical
   body written with Python's builtin ``sum`` extracted fine.
2. ``DISCOPT_PSD_QFORM`` was default-OFF, leaving the interval Hessian +
   Gershgorin row-sum bound, which cannot certify a dense covariance Hessian.
3. ``convexity/interval_ad`` did ``float(np.asarray(expr.value))`` on
   ``Constant``/``Parameter`` leaves, raising ``TypeError`` for a vector-valued
   one. ``certify_convex`` catches only ``ValueError``, so the ``TypeError``
   escaped and was swallowed by the caller's broad ``except Exception`` —
   convexity-unknown by accident rather than by decision.

Net effect: **no model written in the vectorized API could be certified convex.**
Measured before the fix on the ``decision_focused_learning`` oracle QP below
(3 variables, 1 constraint, Hessian ``2·I``): ``feasible`` (not optimal),
14,115 nodes, 60.4 s. After: ``optimal``, 0 nodes, 0.6 s.

The fix takes the exact eigenvalue PSD verdict on the Hessian the *problem
classifier's* extractor already computes. ``classify_problem`` returns
``QP``/``MIQP`` only when every constraint is linear AND the objective is
exactly quadratic, so on that branch the Hessian is a constant matrix and the
eigenvalue test is a rigorous, box-independent global convexity proof — the same
argument ``_certify_quadratic_psd`` already makes, just fed from the extractor
that handles the vectorized API.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.convexity import classify_model  # noqa: E402
from discopt._jax.convexity.certificate import (  # noqa: E402
    certify_quadratic_objective_convex,
)
from discopt._jax.convexity.interval_ad import interval_hessian  # noqa: E402
from discopt._jax.nlp_evaluator import cached_evaluator  # noqa: E402
from discopt._jax.problem_classifier import (  # noqa: E402
    ProblemClass,
    classify_problem,
    dense_Q,
    extract_qp_data,
)
from discopt._jax.quadratic_form import extract_quadratic  # noqa: E402

# Tolerance for "the extracted form reproduces the body". The extraction is
# exact-or-abstain, so the only slack here is float64 summation order.
_EXACT_TOL = 1e-11


def _markowitz(n: int, seed: int = 42):
    """``min wᵀΣw`` s.t. ``Σw = 1``, ``μᵀw >= 0.08``, ``0 <= w <= 1``.

    ``Σ = L Lᵀ + 0.01 I`` is positive definite by construction, so the model is
    convex and any ``is_convex=False`` verdict is a false negative. The body is
    written with ``dm.sum(..., over=...)``, i.e. as ``SumOverExpression`` nodes.
    """
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, n))
    sigma = L @ L.T + 0.01 * np.eye(n)
    mu = rng.uniform(0.05, 0.15, size=n)
    m = dm.Model(f"markowitz{n}")
    w = m.continuous("w", shape=(n,), lb=0.0, ub=1.0)
    m.minimize(
        dm.sum(
            lambda i: dm.sum(lambda j: sigma[i, j] * w[i] * w[j], over=range(n)),
            over=range(n),
        )
    )
    m.subject_to(dm.sum(lambda i: w[i], over=range(n)) == 1.0)
    m.subject_to(dm.sum(lambda i: mu[i] * w[i], over=range(n)) >= 0.08)
    return m, sigma


def _allocation_qp(n: int = 3, seed: int = 0):
    """The ``decision_focused_learning`` oracle QP verbatim from the issue.

    Uses array variables, an array-valued ``Parameter`` and ``dm.sum`` over
    array expressions — the shape that made the interval walker raise
    ``TypeError``. Hessian is ``2·I``, so it is strictly convex.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model("allocation")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    c = m.parameter("cost", value=np.asarray(rng.normal(size=n), dtype=np.float64))
    m.minimize(dm.sum(c * x) + 1.0 * dm.sum(x * x))
    m.subject_to(dm.sum(x) == 1.0)
    return m


def _flat_n(model) -> int:
    return sum(v.size for v in model._variables)


def _max_reconstruction_error(model, quad, lin, const, *, half: bool, draws: int = 40) -> float:
    """``max |body(x) - (s·xᵀQx + cᵀx + d)|`` over random box points.

    ``half`` selects the ``QPData`` convention (``0.5·xᵀHx``) rather than the
    ``quadratic_form`` one (``xᵀQx``).
    """
    n = _flat_n(model)
    ev = cached_evaluator(model)
    scale = 0.5 if half else 1.0
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(draws):
        x = rng.uniform(0.0, 1.0, size=n)
        model_value = float(ev.evaluate_objective(x))
        form_value = float(scale * x @ quad @ x + lin @ x + const)
        worst = max(worst, abs(model_value - form_value))
    return worst


# ── Defect 1: the SumOverExpression branch ───────────────────────────────────


@pytest.mark.unit
def test_extract_quadratic_recognizes_indexed_summation():
    """Pre-fix this returned ``None``: ``_expr_to_polynomial`` walked
    ``SumExpression`` but not ``SumOverExpression``, so ``extract_quadratic``
    abstained on every ``dm.sum(..., over=...)`` body."""
    m, _ = _markowitz(6)
    assert type(m._objective.expression).__name__ == "SumOverExpression"
    result = extract_quadratic(m._objective.expression, _flat_n(m), m)
    assert result is not None, "extract_quadratic abstained on an indexed-summation body"
    Q, c, d = result
    assert Q.shape == (6, 6)


@pytest.mark.unit
@pytest.mark.parametrize("n", [6, 20])
def test_indexed_summation_extraction_is_exact(n):
    """The recognition must reproduce the body, not approximate it."""
    m, sigma = _markowitz(n)
    Q, c, d = extract_quadratic(m._objective.expression, _flat_n(m), m)
    assert _max_reconstruction_error(m, Q, c, d, half=False) <= _EXACT_TOL
    # The recovered Q *is* Σ (symmetrized), so its eigenvalues are Σ's.
    assert np.allclose(Q, sigma, atol=1e-12)


@pytest.mark.unit
def test_indexed_summation_matches_builtin_sum():
    """The same body written with Python's builtin ``sum`` extracted fine before
    the fix; the two spellings must now agree."""
    n = 5
    _, sigma = _markowitz(n)
    m = dm.Model("builtin")
    w = m.continuous("w", shape=(n,), lb=0.0, ub=1.0)
    m.minimize(sum(sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(n)))
    m2, _ = _markowitz(n)
    q_builtin = extract_quadratic(m._objective.expression, n, m)[0]
    q_indexed = extract_quadratic(m2._objective.expression, n, m2)[0]
    assert np.allclose(q_builtin, q_indexed, atol=1e-12)


# ── Defect 3: the escaping TypeError ─────────────────────────────────────────


@pytest.mark.unit
def test_interval_hessian_abstains_deliberately_on_array_leaves():
    """Pre-fix this raised ``TypeError: only 0-dimensional arrays can be
    converted to Python scalars``, which ``certify_convex``'s ``except
    ValueError`` did not catch. ``ValueError`` is the walker's established
    abstain signal — the same refusal array ``Variable``s already get."""
    m = _allocation_qp()
    with pytest.raises(ValueError, match="scalar"):
        interval_hessian(m._objective.expression, m)


@pytest.mark.unit
def test_certify_convex_abstains_instead_of_raising():
    """``certify_convex`` must return ``None``, not propagate."""
    from discopt._jax.convexity import certify_convex

    m = _allocation_qp()
    assert certify_convex(m._objective.expression, m) is None


@pytest.mark.unit
def test_scalar_leaves_still_evaluate():
    """The abstention must be confined to genuinely array-valued leaves; a
    scalar ``Parameter`` (including a 0-d / size-1 array) still walks."""
    m = dm.Model("scalar_param")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    p = m.parameter("p", value=np.float64(3.0))
    m.minimize(p * x * x)
    ad = interval_hessian(m._objective.expression, m)
    assert float(np.asarray(ad.hess.lo).ravel()[0]) == pytest.approx(6.0)


# ── The fix: exact QP/MIQP objective-Hessian convexity ───────────────────────


@pytest.mark.unit
def test_classifier_extractor_already_has_the_proof():
    """The premise of the fix: ``extract_qp_data`` handles the vectorized model
    that every convexity route failed on."""
    m = _allocation_qp()
    assert classify_problem(m) == ProblemClass.QP
    hessian = dense_Q(extract_qp_data(m).Q)
    assert np.allclose(hessian[:3, :3], 2.0 * np.eye(3), atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("factory", [lambda: _allocation_qp(), lambda: _markowitz(6)[0]])
def test_extracted_hessian_reproduces_the_body(factory):
    """The eigenvalue verdict is only rigorous if the extracted Hessian really is
    the body's Hessian — assert it numerically rather than assume it (the
    extractor has three routes: Rust repr, algebraic, autodiff)."""
    m = factory()
    n = _flat_n(m)
    data = extract_qp_data(m)
    hessian = dense_Q(data.Q)[:n, :n]
    lin = np.asarray(data.c)[:n]
    err = _max_reconstruction_error(m, hessian, lin, float(data.obj_const), half=True)
    assert err <= _EXACT_TOL


@pytest.mark.unit
@pytest.mark.parametrize("n", [5, 20])
def test_vectorized_convex_qp_is_certified_convex(n):
    """Pre-fix: ``classify_model`` returned ``is_convex=False`` on a positive
    definite covariance QP — a false negative that routed it to spatial B&B."""
    m, sigma = _markowitz(n)
    assert np.linalg.eigvalsh(sigma)[0] > 0.0, "the fixture must be genuinely convex"
    assert certify_quadratic_objective_convex(m) is True
    is_convex, mask = classify_model(m, use_certificate=True)
    assert is_convex is True
    assert all(mask)


@pytest.mark.unit
def test_allocation_qp_is_certified_convex():
    m = _allocation_qp()
    assert certify_quadratic_objective_convex(m) is True
    assert classify_model(m, use_certificate=True)[0] is True


@pytest.mark.unit
def test_route_is_opt_outable():
    """``DISCOPT_QP_EXACT_CONVEXITY=0`` restores the pre-fix abstention, so the
    legacy path stays reachable (CLAUDE.md §5)."""
    m = _allocation_qp()
    prior = os.environ.get("DISCOPT_QP_EXACT_CONVEXITY")
    os.environ["DISCOPT_QP_EXACT_CONVEXITY"] = "0"
    try:
        assert certify_quadratic_objective_convex(m) is False
    finally:
        if prior is None:
            os.environ.pop("DISCOPT_QP_EXACT_CONVEXITY", None)
        else:
            os.environ["DISCOPT_QP_EXACT_CONVEXITY"] = prior
    assert certify_quadratic_objective_convex(m) is True


# ── Soundness: the route must never certify a nonconvex model ───────────────


def _indefinite_qp():
    """``min x0² - x1²`` on ``[-3, 3]²`` s.t. ``x0 + x1 <= 4`` — true optimum -9."""
    m = dm.Model("indefinite")
    x = m.continuous("x", shape=(2,), lb=-3.0, ub=3.0)
    m.minimize(dm.sum(lambda i: (1.0 if i == 0 else -1.0) * x[i] * x[i], over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 4.0)
    return m


@pytest.mark.unit
def test_indefinite_qp_is_not_certified_convex():
    m = _indefinite_qp()
    assert certify_quadratic_objective_convex(m) is False
    assert classify_model(m, use_certificate=True)[0] is False


@pytest.mark.unit
def test_convex_maximize_is_not_certified_convex():
    """``max xᵀx`` is a *concave-minimize* problem: the route must refuse it.
    ``extract_qp_data`` returns minimize-form data (it negates for MAXIMIZE), so
    the PSD test answers the right question without a further sign flip."""
    m = dm.Model("convex_max")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.maximize(dm.sum(lambda i: x[i] * x[i], over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 6.0)
    assert certify_quadratic_objective_convex(m) is False


@pytest.mark.unit
def test_concave_maximize_is_certified_convex():
    """``max -(x-2)ᵀ(x-2)`` IS a convex problem and must be certified as one."""
    m = dm.Model("concave_max")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.maximize(dm.sum(lambda i: -1.0 * (x[i] - 2.0) * (x[i] - 2.0), over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 10.0)
    assert certify_quadratic_objective_convex(m) is True


def _transcendental_objective():
    m = dm.Model("transcendental")
    x = m.continuous("x", lb=0.1, ub=5.0)
    m.minimize(dm.log(x) * x)
    m.subject_to(x <= 4.0)
    return m


def _convex_but_not_quadratic():
    """``Σ exp(x_i)`` IS convex, but not *quadratic* — the route must still refuse
    it, because its whole rigor rests on the Hessian being constant."""
    m = dm.Model("exp_sum")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=5.0)
    m.minimize(dm.sum(lambda i: dm.exp(x[i]), over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 4.0)
    return m


def _quartic_objective():
    m = dm.Model("quartic")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=5.0)
    m.minimize(dm.sum(lambda i: x[i] ** 4, over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 4.0)
    return m


def _convex_objective_nonconvex_constraint():
    """The gate that matters most: a convex quadratic objective over a *nonconvex*
    feasible set. ``classify_problem`` must not call this a QP, or the model would
    route to a convex QP solver that assumes a polyhedron."""
    m = dm.Model("qcqp")
    x = m.continuous("x", shape=(2,), lb=-5.0, ub=5.0)
    m.minimize(dm.sum(lambda i: x[i] * x[i], over=range(2)))
    m.subject_to(x[0] * x[1] >= 1.0)
    return m


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        _transcendental_objective,
        _convex_but_not_quadratic,
        _quartic_objective,
        _convex_objective_nonconvex_constraint,
    ],
)
def test_non_qp_models_are_refused(factory):
    """The route's rigor rests entirely on ``classify_problem`` returning
    ``QP``/``MIQP`` — exactly-quadratic objective over a polyhedron. Anything
    else must be refused, convex or not."""
    m = factory()
    assert classify_problem(m) not in (ProblemClass.QP, ProblemClass.MIQP)
    assert certify_quadratic_objective_convex(m) is False


@pytest.mark.unit
def test_oversized_model_abstains():
    """The route declines above its size cap rather than spend seconds in
    ``eigvalsh`` inside the solver's dispatch budget. Abstaining is sound."""
    from discopt._jax.convexity import certificate as cert_mod

    m = _allocation_qp()
    prior = cert_mod._QP_EXACT_CONVEXITY_MAX_N
    cert_mod._QP_EXACT_CONVEXITY_MAX_N = 1
    try:
        assert certify_quadratic_objective_convex(m) is False
    finally:
        cert_mod._QP_EXACT_CONVEXITY_MAX_N = prior


@pytest.mark.unit
def test_expired_deadline_abstains():
    import time

    m = _allocation_qp()
    assert certify_quadratic_objective_convex(m, deadline=time.perf_counter() - 1.0) is False


# ── End-to-end: the models that blocked three notebooks ─────────────────────


@pytest.mark.smoke
def test_allocation_qp_solves_to_optimal():
    """The issue's repro. Pre-fix: ``feasible``, 14,115 nodes, 60.4 s (time
    limit). The certificate makes it a root solve."""
    m = _allocation_qp()
    res = m.solve(time_limit=60)
    assert res.status == "optimal", f"got {res.status} obj={res.objective!r}"
    assert res.node_count == 0, f"expected a root solve, explored {res.node_count} nodes"


@pytest.mark.smoke
def test_markowitz_qp_solves_to_optimal():
    """``qp_solver.ipynb``'s model. Pre-fix at n=20: ``feasible`` only, 557
    nodes, 120 s limit hit."""
    m, _ = _markowitz(20)
    res = m.solve(time_limit=60)
    assert res.status == "optimal", f"got {res.status} obj={res.objective!r}"
    assert res.node_count == 0, f"expected a root solve, explored {res.node_count} nodes"


@pytest.mark.smoke
def test_indefinite_qp_still_finds_the_global_optimum():
    """The soundness half: a genuinely nonconvex vectorized QP must keep routing
    to spatial B&B and return the true optimum, not a stationary point."""
    res = _indefinite_qp().solve(time_limit=60)
    assert res.status == "optimal", f"got {res.status}"
    assert float(res.objective) == pytest.approx(-9.0, abs=1e-4)


@pytest.mark.smoke
def test_convex_maximize_still_finds_the_global_optimum():
    m = dm.Model("convex_max")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.maximize(dm.sum(lambda i: x[i] * x[i], over=range(2)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(2)) <= 6.0)
    res = m.solve(time_limit=60)
    assert res.status == "optimal", f"got {res.status}"
    assert float(res.objective) == pytest.approx(18.0, abs=1e-4)
