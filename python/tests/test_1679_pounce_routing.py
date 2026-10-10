"""#1679: ``solver="pounce"`` routing, reporting and hand-off findings.

* the ``solve_report`` objective is in the model's sense (constant and MAXIMIZE
  flip included), so it agrees with ``res.objective`` on every route;
* ``Model.convexity()`` and the route's PSD proof are the same predicate, so they
  cannot disagree on a Hessian -- in particular neither certifies the float Gram
  matrix of a rank-deficient least-squares model, which is indefinite in exact
  arithmetic;
* the vector spelling ``dm.sum((K @ x - b)**2)`` of that model is recognised as a
  sum of affine squares and solved by qp-ipm on the exact lift, like the scalar one;
* the QP route hands POUNCE a sparse Hessian whichever extraction rung ran;
* ``warm_start=<previous result>`` seeds qp-ipm with the engine's full primal-dual
  iterate, not only ``x``;
* the ``qp_*`` refusal names the ``solve_qp`` argument when one exists.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model

pytestmark = [pytest.mark.requires_pounce]


def _pounce(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(solver="pounce", **kw)


def _report_objectives(rep):
    assert rep is not None
    return rep["solution"]["objective"], rep["statistics"]["final_objective"]


# --------------------------------------------------------------------------- item 1


@pytest.mark.parametrize("sense", ["min", "max"])
def test_qp_report_objective_includes_constant_and_sense(sense):
    m = Model("qp_const")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    q = dm.sum((x - np.array([1.0, -2.0, 0.5])) ** 2)
    if sense == "min":
        m.minimize(q + 1100.0)
    else:
        m.maximize(-q + 1100.0)
    res = _pounce(m)
    assert res.status == "optimal"
    assert res.algorithm_route.endswith("qp-ipm")
    sol, final = _report_objectives(res.solve_report)
    assert sol == pytest.approx(res.objective, rel=1e-9, abs=1e-9)
    assert final == pytest.approx(res.objective, rel=1e-9, abs=1e-9)
    assert res.objective == pytest.approx(1100.0, abs=1e-6)
    assert res.solve_report["problem"]["minimize"] is (sense == "min")


@pytest.mark.parametrize("sense", ["min", "max"])
def test_lp_report_objective_includes_constant_and_sense(sense):
    m = Model("lp_const")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 6)
    if sense == "min":
        m.minimize(-x - 2 * y + 50.0)
    else:
        m.maximize(x + 2 * y + 50.0)
    res = _pounce(m)
    assert res.status == "optimal"
    sol, final = _report_objectives(res.solve_report)
    assert sol == pytest.approx(res.objective, rel=1e-8, abs=1e-8)
    assert final == pytest.approx(res.objective, rel=1e-8, abs=1e-8)


def test_nlp_arm_report_objective_in_model_sense():
    m = Model("nlp_max")
    x = m.continuous("x", lb=0.1, ub=3)
    m.maximize(dm.log(x) - x + 7.0)
    res = _pounce(m)
    assert res.objective == pytest.approx(6.0, abs=1e-6)
    sol, final = _report_objectives(res.solve_report)
    assert sol == pytest.approx(res.objective, rel=1e-8)
    assert final == pytest.approx(res.objective, rel=1e-8)


def test_report_mapping_is_idempotent():
    from discopt.solvers._pounce_report import report_in_model_sense

    rep = {"solution": {"objective": 1.0}, "statistics": {"final_objective": 1.0}}
    report_in_model_sense(rep, 10.0, True)
    report_in_model_sense(rep, 10.0, True)
    assert rep["solution"]["objective"] == -11.0
    assert rep["statistics"]["final_objective"] == -11.0
    assert rep["solution"]["engine_objective"] == 1.0


# --------------------------------------------------------------------------- item 2


def _ls(form, rank_deficient=True):
    rng = np.random.default_rng(0)
    K = rng.normal(size=(40, 35))
    if rank_deficient:
        K[:, -1] = K[:, 0] + K[:, 1]
    b = rng.normal(size=40)
    m = Model("ls")
    x = m.continuous("x", shape=(35,), lb=-10, ub=10)
    if form == "gram":
        m.minimize(x @ (K.T @ K) @ x - 2 * ((K.T @ b) @ x) + float(b @ b))
    elif form == "vector":
        m.minimize(dm.sum((K @ x - b) ** 2))
    else:
        m.minimize(
            dm.sum([(dm.sum([K[i, j] * x[j] for j in range(35)]) - b[i]) ** 2 for i in range(40)])
        )
    return m, K, b


@pytest.mark.parametrize("rank_deficient", [True, False])
def test_convexity_and_route_psd_proof_agree_on_gram_form(rank_deficient):
    from discopt._relax.problem_classifier import dense_Q, extract_qp_data
    from discopt.solvers.convex_ipm_pounce import certify_psd

    m, _, _ = _ls("gram", rank_deficient)
    Q = dense_Q(extract_qp_data(m).Q)[:35, :35]
    route = certify_psd(Q)
    assert route is (not rank_deficient)
    assert m.convexity().is_convex is route


def test_rank_deficient_gram_is_genuinely_indefinite():
    """The refusal is a fact about the float matrix, not a weak test: exact
    rational elimination refutes PSD."""
    from discopt._relax.convexity.eigenvalue import exact_psd
    from discopt._relax.problem_classifier import dense_Q, extract_qp_data

    m, _, _ = _ls("gram")
    Q = dense_Q(extract_qp_data(m).Q)[:35, :35]
    assert exact_psd(0.5 * (Q + Q.T)) is False


def test_psd_certified_keeps_proofs_the_route_made():
    """Sharing the predicate never loses a proof: well-conditioned PSD, singular
    PSD with an exact zero (small and sparse), and a dense full-rank Gram."""
    from discopt._relax.convexity.eigenvalue import psd_certified

    rng = np.random.default_rng(3)
    A = rng.normal(size=(60, 45))
    lap = sp.diags([-np.ones(199), 2 * np.ones(200), -np.ones(199)], [-1, 0, 1]).tolil()
    lap[0, 0] = lap[-1, -1] = 1.0
    cases = [
        np.eye(4),
        np.array([[1.0, -1.0], [-1.0, 1.0]]),  # (x - y)**2
        A.T @ A,
        lap.tocsr(),
    ]
    for Q in cases:
        assert psd_certified(Q) is True
    assert psd_certified(np.diag([1.0, -1e-9])) is False


@pytest.mark.parametrize("form", ["scalar", "vector"])
def test_sum_of_squares_forms_route_to_qp_ipm(form):
    m, _, _ = _ls(form)
    res = _pounce(m)
    assert res.algorithm_route == "pounce:qp-ipm"
    assert res.status == "optimal"
    ref = _pounce(_ls("scalar")[0]).objective
    assert res.objective == pytest.approx(ref, rel=1e-7)


def test_vector_affine_rows_matches_dense_evaluation():
    from discopt._relax.convexity.patterns import weighted_affine_square_decomposition

    m, K, b = _ls("vector")
    dec = weighted_affine_square_decomposition(m._objective.expression, m)
    assert dec is not None
    w, A, bb, lin, c0 = dec
    assert np.allclose(w, 1.0) and w.size == 40
    np.testing.assert_allclose(A.toarray()[:, :35], K)
    np.testing.assert_allclose(bb, -b)


def _broadcast_cases():
    one31 = np.ones((3, 1))
    w = np.array([1.0, 2.0, 3.0])
    M = np.arange(6.0).reshape(2, 3)
    return [
        # (builder of the vector v, numpy value of sum(v**2) at x)
        (lambda x: x + one31, lambda x: float(np.sum((x + one31) ** 2))),
        (lambda x: x * one31, lambda x: float(np.sum((x * one31) ** 2))),
        (lambda x: x - np.ones((1, 3)), lambda x: float(np.sum((x - np.ones((1, 3))) ** 2))),
        (lambda x: w * x - 1.0, lambda x: float(np.sum((w * x - 1.0) ** 2))),
        (lambda x: M @ x + 2.0, lambda x: float(np.sum((M @ x + 2.0) ** 2))),
        (lambda x: x[1:] / 4.0 - x[:-1], lambda x: float(np.sum((x[1:] / 4.0 - x[:-1]) ** 2))),
    ]


@pytest.mark.parametrize("case", range(6))
def test_vector_square_decomposition_respects_broadcast_shapes(case):
    """``x + ones((3, 1))`` has 9 elements, not 3: the extractor must either
    reproduce the objective exactly or abstain -- never flatten a broadcast."""
    from discopt._relax.convexity.patterns import weighted_affine_square_decomposition

    build, value = _broadcast_cases()[case]
    m = Model("bc")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.minimize(dm.sum(build(x) ** 2))
    dec = weighted_affine_square_decomposition(m._objective.expression, m)
    if dec is None:
        assert case < 3  # the broadcasting traps may abstain; the rest must lift
        return
    wts, A, bb, lin, c0 = dec
    rng = np.random.default_rng(case)
    for _ in range(5):
        xv = rng.uniform(-5, 5, size=3)
        r = A.toarray()[:, :3] @ xv + bb
        got = float(wts @ r**2 + lin[:3] @ xv + c0)
        assert got == pytest.approx(value(xv), rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- item 3


def _mpc(N=60, UA=1.0):
    m = Model("mpc")
    x = m.continuous("x", shape=(N + 1,), lb=-50, ub=50)
    u = m.continuous("u", shape=(N,), lb=-2.0 * UA, ub=2.0 * UA)
    m.subject_to(x[0] == 10.0)
    m.subject_to(x[1:] == 0.95 * x[:-1] + (0.1 / UA) * u)
    m.minimize(dm.sum(x**2) + 0.01 * dm.sum(u**2) / UA**2)
    return m


def test_qp_route_hands_pounce_a_sparse_hessian(monkeypatch):
    import pounce.qp as pq

    seen = []
    orig = pq.solve_qp

    def spy(*a, **kw):
        seen.append(kw.get("P"))
        return orig(*a, **kw)

    monkeypatch.setattr(pq, "solve_qp", spy)
    res = _pounce(_mpc())
    assert res.status == "optimal"
    assert seen, "pounce.qp.solve_qp was never called"
    assert all(sp.issparse(P) for P in seen)


def test_tape_rung_honours_sparse_qp_matrices():
    from discopt._relax import problem_classifier as pc

    m = _mpc(N=20)
    dense = pc._extract_qp_data_autodiff(m)
    with pc.sparse_qp_matrices():
        sparse = pc._extract_qp_data_autodiff(m)
    assert not sp.issparse(dense.Q)
    assert sp.issparse(sparse.Q)
    np.testing.assert_array_equal(sparse.Q.toarray(), dense.Q)


# --------------------------------------------------------------------------- item 4


def _iters(res):
    return res.solve_report["statistics"]["iteration_count"]


def test_warm_start_result_seeds_qp_ipm_with_multipliers():
    m = _mpc()
    r1 = _pounce(m)
    assert r1.pounce_qp_iterate is not None
    assert set(r1.pounce_qp_iterate) >= {"x", "y"}
    x0 = {v: r1.value(v) for v in m._variables}
    primal_only = _pounce(m, initial_solution=x0)
    full = _pounce(m, warm_start=r1)
    assert full.status == primal_only.status == "optimal"
    assert full.objective == pytest.approx(r1.objective, rel=1e-8)
    assert _iters(full) < _iters(primal_only)


def test_solve_qp_validates_a_primal_dual_warm_start():
    from discopt.solvers.convex_ipm_pounce import solve_qp

    Q = np.diag([2.0, 2.0])
    c = np.array([-2.0, -4.0])
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    r = solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq)
    st = r.warm_start_state
    assert st["x"].shape == (2,) and st["y"].shape == (1,)
    again = solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, warm_start=st)
    np.testing.assert_allclose(again.x, r.x, atol=1e-7)
    with pytest.raises(ValueError, match="warm_start\\['y'\\]"):
        solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, warm_start={"x": st["x"], "y": np.zeros(3)})
    with pytest.raises(ValueError, match="keys"):
        solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, warm_start={"x": st["x"], "lam": 1})


# --------------------------------------------------------------------------- item 5


def test_qp_tau_refusal_names_the_solve_qp_argument():
    from discopt.solvers.convex_ipm_pounce import convex_engine_options

    with pytest.raises(ValueError) as exc:
        convex_engine_options({"qp_tau": 0.9})
    msg = str(exc.value)
    assert "'qp_tau' -> 'tau'" in msg
    assert "takes no such argument" not in msg
    with pytest.raises(ValueError, match="takes no such argument"):
        convex_engine_options({"qp_hsde": "yes"})
    assert convex_engine_options({"tau": 0.9}) == {"tau": 0.9}
