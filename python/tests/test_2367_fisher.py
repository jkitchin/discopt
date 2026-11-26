"""#2367: sparse, JAX-free, bound-aware Fisher information (``discopt.estimate.fisher``).

The oracle for the state-following derivative is an independent *sequential*
forward-sensitivity recursion of the same backward-Euler scheme, so the check is
not the implementation compared against itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.estimate import (
    Experiment,
    ExperimentModel,
    FisherInformation,
    estimate_parameters,
    fisher,
)

T_END, K1, K2, SIGMA = 4.0, 0.7, 0.3, 0.02


def _simulate(n: int, k1: float, k2: float):
    """Backward Euler of A' = -k1 A, B' = k1 A - k2 B, with forward sensitivities."""
    h = T_END / n
    A = np.empty(n + 1)
    B = np.empty(n + 1)
    A[0], B[0] = 1.0, 0.0
    sA = np.zeros((n + 1, 2))
    sB = np.zeros((n + 1, 2))
    for i in range(n):
        A[i + 1] = A[i] / (1 + h * k1)
        B[i + 1] = (B[i] + h * k1 * A[i + 1]) / (1 + h * k2)
        sA[i + 1, 0] = (sA[i, 0] - h * A[i + 1]) / (1 + h * k1)
        sA[i + 1, 1] = sA[i, 1] / (1 + h * k1)
        sB[i + 1, 0] = (sB[i, 0] + h * (A[i + 1] + k1 * sA[i + 1, 0])) / (1 + h * k2)
        sB[i + 1, 1] = (sB[i, 1] + h * (k1 * sA[i + 1, 1] - B[i + 1])) / (1 + h * k2)
    return A, B, sA, sB


def _be_model(n: int, k_ub: float = 5.0):
    """The backward-Euler model as discopt equations, with a deviance objective."""
    h = T_END / n
    _, b_true, _, _ = _simulate(n, K1, K2)
    idx = np.arange(1, n + 1, max(1, n // 20))
    data = b_true[idx] + SIGMA * np.random.default_rng(0).standard_normal(idx.size)
    m = dm.Model("be_fit")
    k1 = m.continuous("k1", lb=0.01, ub=k_ub)
    k2 = m.continuous("k2", lb=0.01, ub=5.0)
    A = m.continuous("A", shape=(n + 1,), lb=-10, ub=10)
    B = m.continuous("B", shape=(n + 1,), lb=-10, ub=10)
    m.subject_to(A[0] == 1.0)
    m.subject_to(B[0] == 0.0)
    m.subject_to(A[1:] - A[:-1] + h * k1 * A[1:] == 0)
    m.subject_to(B[1:] - B[:-1] - h * (k1 * A[1:] - k2 * B[1:]) == 0)
    resp = [B[int(i)] for i in idx]
    m.minimize(dm.sum([((float(d) - r) / SIGMA) ** 2 for d, r in zip(data, resp)]))
    return m, (k1, k2), resp, idx, data


@pytest.fixture(scope="module")
def be_fit():
    m, pars, resp, idx, data = _be_model(300)
    res = m.solve(solver="pounce")
    assert res.status in ("optimal", "local_optimal", "feasible")
    return m, pars, resp, idx, data, res


def test_gauss_newton_matches_sequential_sensitivity(be_fit):
    m, pars, resp, idx, _data, res = be_fit
    info = fisher(res, pars, resp, sigma=SIGMA)
    assert isinstance(info, FisherInformation)
    k1h, k2h = (float(res.x[p.name]) for p in pars)
    _, _, _, sB = _simulate(300, k1h, k2h)
    J_ref = sB[idx]
    np.testing.assert_allclose(info.jacobian, J_ref, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(info.fim, J_ref.T @ J_ref / SIGMA**2, rtol=1e-10)
    assert info.active_bounds == {}
    assert info.free.all() and info.rank == 2
    np.testing.assert_allclose(info.covariance, np.linalg.inv(info.fim), rtol=1e-10)
    # Parameter rows of dx/dθ are the identity.
    np.testing.assert_array_equal(info.sensitivities[[0, 1]], np.eye(2))


def test_observed_matches_reduced_objective_hessian(be_fit):
    """Observed information = ½ ∇²φ(θ), checked by differencing the exact reduced gradient."""
    m, pars, resp, idx, data, res = be_fit
    obs = fisher(res, pars, resp, sigma=SIGMA, hessian="observed")
    k = np.array([float(res.x[p.name]) for p in pars])

    def grad(kk):
        _, Bk, _, sBk = _simulate(300, *kk)
        r = (data - Bk[idx]) / SIGMA
        return -2.0 * (sBk[idx].T @ r) / SIGMA

    eps = 1e-6
    H = np.column_stack([(grad(k + eps * e) - grad(k - eps * e)) / (2 * eps) for e in np.eye(2)])
    np.testing.assert_allclose(obs.fim, 0.25 * (H + H.T), rtol=1e-6)
    # At the optimum the reduced gradient vanishes and nothing is on a bound.
    assert np.max(np.abs(obs.reduced_gradient)) < 1e-3 * np.max(np.abs(obs.fim))
    np.testing.assert_allclose(obs.reduced_gradient, grad(k), atol=1e-6 * np.abs(H).max())


def test_runs_without_jax():
    """The whole computation, solve included, must not import JAX."""
    body = textwrap.dedent(
        """
        import sys
        class _NoJax:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in ("jax", "jaxlib"):
                    raise ModuleNotFoundError(name, name=name)
        sys.meta_path.insert(0, _NoJax())
        import discopt._optional_jax as _oj
        _oj._available = False
        sys.path.insert(0, {tests!r})
        from test_2367_fisher import _be_model, SIGMA
        from discopt.estimate import fisher
        m, pars, resp, idx, data = _be_model(1200)
        res = m.solve(solver="pounce")
        gn = fisher(res, pars, resp, sigma=SIGMA)
        ob = fisher(res, pars, resp, sigma=SIGMA, hessian="observed")
        assert gn.fim.shape == (2, 2) and ob.fim.shape == (2, 2)
        loaded = [k for k, v in sys.modules.items()
                  if v is not None and k.partition(".")[0] == "jax"]
        assert not loaded, loaded
        print("OK", gn.state_jacobian_condition)
        """
    ).format(tests=os.path.dirname(__file__))
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_NLP_EVAL"}
    out = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, env=env, timeout=600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout


def test_parameter_on_bound_gets_no_wald_interval():
    """k1 capped below its fitted value: reported active, covariance on the free subspace."""
    m, pars, resp, idx, data = _be_model(200, k_ub=0.5)
    res = m.solve(solver="pounce")
    assert abs(float(res.x["k1"]) - 0.5) < 1e-6
    info = fisher(res, pars, resp, sigma=SIGMA)
    assert info.active_bounds == {"k1": "upper"}
    assert info.free.tolist() == [False, True]
    assert np.all(np.isnan(info.covariance[0])) and np.all(np.isnan(info.covariance[:, 0]))
    np.testing.assert_allclose(info.covariance[1, 1], 1.0 / info.fim[1, 1], rtol=1e-12)
    assert np.isnan(info.standard_errors["k1"])
    assert np.isfinite(info.standard_errors["k2"])
    # The reduced gradient stands in for the withheld bound multiplier: the
    # objective still decreases toward larger k1, so it is negative at an upper bound.
    assert info.reduced_gradient[0] < -1.0
    assert abs(info.reduced_gradient[1]) < 1e-3 * abs(info.reduced_gradient[0])


def test_fixed_parameter_reported_as_fixed(be_fit):
    m, pars, resp, idx, data, res = be_fit
    k1, k2 = pars
    saved = (k1.lb, k1.ub)
    try:
        k1.lb = k1.ub = np.asarray(float(res.x["k1"]))
        info = fisher(res, pars, resp, sigma=SIGMA)
    finally:
        k1.lb, k1.ub = saved
    assert info.active_bounds == {"k1": "fixed"}
    np.testing.assert_allclose(info.covariance[1, 1], 1.0 / info.fim[1, 1], rtol=1e-12)


def test_unidentifiable_parameter_warns():
    """y = (a + b) x: the free FIM is singular, which must be said, not pinv'd silently."""
    m = dm.Model("unident")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-10, ub=10)
    xs = np.array([1.0, 2.0, 3.0])
    resp = [(a + b) * float(x) for x in xs]
    m.minimize(dm.sum([(2.0 * float(x) - r) ** 2 for x, r in zip(xs, resp)]))
    with pytest.warns(RuntimeWarning, match="singular"):
        info = fisher({"a": 1.0, "b": 1.0}, [a, b], resp, sigma=0.1, model=m)
    assert info.rank == 1


def test_observed_equals_gauss_newton_for_zero_residual_curvature():
    """Responses linear in the states and θ: ∇²r = 0, so observed == expected exactly."""
    m = dm.Model("linear")
    k = m.continuous("k", lb=-10, ub=10)
    z = m.continuous("z", shape=(3,), lb=-100, ub=100)
    xs = [1.0, 2.0, 3.0]
    for i, x in enumerate(xs):
        m.subject_to(z[i] == k * x)
    m.minimize(dm.sum([((2.1 * x - z[i]) / 0.1) ** 2 for i, x in enumerate(xs)]))
    point = {"k": 1.7, "z": 1.7 * np.array(xs)}
    gn = fisher(point, k, z, sigma=0.1, model=m)
    ob = fisher(point, k, z, sigma=0.1, model=m, hessian="observed")
    np.testing.assert_allclose(gn.fim, [[sum(x * x for x in xs) / 0.01]], rtol=1e-12)
    np.testing.assert_allclose(ob.fim, gn.fim, rtol=1e-12)


def test_observed_nonlinear_state_analytic():
    """z_i == k² x_i: ½ d²/dk² Σ((d_i - k² x_i)/σ)² = Σ[(2kx)² - 2x(d - k²x)]/σ²."""
    xs = np.array([1.0, 2.0, 3.0])
    d = np.array([2.2, 3.9, 6.3])
    sig, kv = 0.1, 1.3
    m = dm.Model("nl_state")
    k = m.continuous("k", lb=0.1, ub=10)
    z = m.continuous("z", shape=(3,), lb=-100, ub=100)
    for i, x in enumerate(xs):
        m.subject_to(z[i] == k * k * float(x))
    m.minimize(dm.sum([((float(d[i]) - z[i]) / sig) ** 2 for i in range(3)]))
    x_flat = np.concatenate([[kv], kv * kv * xs])
    ob = fisher(x_flat, k, z, sigma=sig, model=m, hessian="observed")
    ref = np.sum((2 * kv * xs) ** 2 - 2 * xs * (d - kv * kv * xs)) / sig**2
    np.testing.assert_allclose(ob.fim, [[ref]], rtol=1e-12)
    gn = fisher(x_flat, k, z, sigma=sig, model=m)
    np.testing.assert_allclose(gn.fim, [[np.sum((2 * kv * xs) ** 2) / sig**2]], rtol=1e-12)


def test_observed_ignores_gauss_newton_evaluator():
    """A model solved with gauss_newton=True must still get the exact observed curvature."""
    xs = np.array([1.0, 2.0, 3.0])
    m = dm.Model("gn_flag")
    k = m.continuous("k", lb=0.1, ub=10)
    z = m.continuous("z", shape=(3,), lb=-100, ub=100)
    for i, x in enumerate(xs):
        m.subject_to(z[i] == k * k * float(x))
    m.minimize(dm.sum([((5.0 - z[i]) / 0.1) ** 2 for i in range(3)]))
    m._gauss_newton_hessian = True
    x_flat = np.concatenate([[1.3], 1.69 * xs])
    ob = fisher(x_flat, k, z, sigma=0.1, model=m, hessian="observed")
    ref = np.sum((2.6 * xs) ** 2 - 2 * xs * (5.0 - 1.69 * xs)) / 0.01
    np.testing.assert_allclose(ob.fim, [[ref]], rtol=1e-12)
    assert m._gauss_newton_hessian is True


def test_dae_builder_fit():
    """A DAEBuilder collocation model: dx(T)/dk = -T e^{-kT} for x' = -k x, x(0) = 1."""
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("dae_fisher")
    k = m.continuous("k", lb=0.8, ub=0.8)  # pinned for the simulation solve
    cs = ContinuousSet("t", bounds=(0, 2), nfe=20, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.set_ode(lambda t, s, a, c: {"x": -k * s["x"]})
    dae.discretize()
    xv = dae.get_state("x")
    m.minimize(0 * xv[0, 0])
    res = m.solve()
    assert res.status in ("optimal", "local_optimal", "feasible")
    t_pts, x_vals = dae.extract_solution(res, "x")
    assert t_pts[-1] == pytest.approx(2.0)
    k.lb, k.ub = np.asarray(0.01), np.asarray(5.0)  # free it: it is now the parameter
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        info = fisher(res, k, xv[-1, -1], sigma=0.01)
    assert info.active_bounds == {}
    np.testing.assert_allclose(info.jacobian[0, 0], -2.0 * np.exp(-1.6), rtol=1e-6)


def test_point_forms_agree(be_fit):
    m, pars, resp, idx, data, res = be_fit
    a = fisher(res, pars, resp, sigma=SIGMA)
    b = fisher(dict(res.x), pars, resp, sigma=SIGMA, model=m)
    x_flat = np.concatenate([np.ravel(res.x[v.name]) for v in m._variables])
    c = fisher(x_flat, {"k1": pars[0], "k2": pars[1]}, resp, sigma=SIGMA, model=m)
    np.testing.assert_array_equal(a.fim, b.fim)
    np.testing.assert_array_equal(a.fim, c.fim)
    with pytest.raises(ValueError, match="model= is required"):
        fisher(x_flat, pars, resp, sigma=SIGMA)
    with pytest.raises(ValueError, match="hessian must be"):
        fisher(res, pars, resp, hessian="exact")


class _CappedExperiment(Experiment):
    """y_i = k x_i with k capped below the value the data want."""

    def create_model(self, **kwargs):
        m = dm.Model("capped")
        k = m.continuous("k", lb=0.0, ub=1.5)
        c = m.continuous("c", lb=-10.0, ub=10.0)
        xs = [1.0, 2.0, 3.0]
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k, "c": c},
            design_inputs={},
            responses={f"y{i}": k * x + c for i, x in enumerate(xs)},
            measurement_error={f"y{i}": 0.1 for i in range(3)},
        )


def test_estimate_parameters_is_bound_aware():
    data = {"y0": 2.0, "y1": 4.0, "y2": 6.0}
    out = estimate_parameters(_CappedExperiment(), data)
    assert out.active_bounds == {"k": "upper"}
    assert np.isnan(out.standard_errors["k"])
    np.testing.assert_allclose(out.standard_errors["c"], np.sqrt(0.01 / 3), rtol=1e-6)
    assert out.information is not None and out.information.reduced_gradient[0] < 0
    obs = estimate_parameters(_CappedExperiment(), data, hessian="observed")
    np.testing.assert_allclose(obs.fim, out.fim, rtol=1e-8)  # affine residuals
