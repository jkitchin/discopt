"""Tests for the ``Surrogate`` protocol (``discopt.nn.surrogate``).

Verifies that (1) the built-in trainable surrogates conform, (2) conformance is
structural (a class missing the load-bearing ``__call__`` does not), and (3) the
protocol is the *real* contract: an arbitrary user-defined surrogate that
implements it trains end-to-end inside a collocation DAE with no framework
changes. Test 3 uses a symbolic-regression-style rational rate law — the exact
structure a tool like PySR/jaxsr would discover — with its constants trained in
the NLP, and checks it recovers the true constants.
"""

import numpy as np
import pytest
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.modeling.core import Variable
from discopt.nn import (
    Surrogate,
    TrainableDense,
    TrainableKernelExpansion,
    TrainableNetwork,
    train,
)

TF = 2.0
CA0, CB0 = 1.0, 0.0


def _r_true(c):
    return 1.5 * c**2 / (0.3 + c)


def _make_data(seed=0, n_obs=15):
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, TF, n_obs)
    sol = solve_ivp(
        lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
        (0, TF),
        [CA0, CB0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    return (
        t,
        sol.y[0] + 0.01 * rng.standard_normal(n_obs),
        sol.y[1] + 0.01 * rng.standard_normal(n_obs),
    )


# ── test 1: the built-in surrogates conform ─────────────────────────────────


def test_builtin_surrogates_conform():
    m = Model()
    net = TrainableNetwork(m, [1, 4, 1], name="n")
    ke = TrainableKernelExpansion(m, np.linspace(0.0, 1.0, 5), lengthscale=0.2, name="k")
    dense = TrainableDense(m, 2, 3, name="d")

    for s in (net, ke, dense):
        assert isinstance(s, Surrogate)
        # The conventional methods behave.
        assert s.n_parameters() == sum(p.size for p in s.parameters())
        assert s.l2_penalty() is not None
        vals = s.initial_values()
        assert set(vals) == set(s.parameters())


# ── test 2: conformance is structural (missing __call__ fails) ──────────────


def test_missing_call_does_not_conform():
    class NoCall:
        def parameters(self):
            return []

        def n_parameters(self):
            return 0

        def l2_penalty(self):
            return 0

        def initial_values(self):
            return {}

    assert not isinstance(NoCall(), Surrogate)


# ── test 3: a custom surrogate trains end-to-end ────────────────────────────


class RationalRate:
    """A user-defined surrogate: ``r(c) = a * c**2 / (b + c)`` (Michaelis-Menten).

    Implements the :class:`Surrogate` protocol with two trainable constants — the
    fixed-structure symbolic form a symbolic-regression search would return, with
    its constants fit in the NLP. No inheritance from anything in discopt.nn.
    """

    def __init__(self, model: Model, *, name: str):
        self.a = model.continuous(f"{name}_a", lb=0.0, ub=10.0)
        self.b = model.continuous(f"{name}_b", lb=1e-3, ub=10.0)

    def __call__(self, c):
        return self.a * c**2 / (self.b + c)

    def parameters(self) -> list[Variable]:
        return [self.a, self.b]

    def n_parameters(self) -> int:
        return 2

    def l2_penalty(self):
        return self.a**2 + self.b**2

    def initial_values(self) -> dict[Variable, np.ndarray]:
        return {self.a: np.array(1.0), self.b: np.array(1.0)}


@pytest.mark.smoke
def test_custom_surrogate_trains_end_to_end():
    t_data, yA, yB = _make_data()

    m = Model()
    rate = RationalRate(m, name="r")
    assert isinstance(rate, Surrogate)  # the custom class satisfies the contract

    cs = ContinuousSet("t", (0.0, TF), nfe=20, ncp=2, scheme="radau")
    dae = DAEBuilder(m, cs)
    dae.add_state("cA", bounds=(0.0, 1.5), initial=CA0)
    dae.add_state("cB", bounds=(0.0, 1.5), initial=CB0)
    dae.set_ode(lambda t, s, a, u: {"cA": -rate(s["cA"]), "cB": rate(s["cA"])})
    dvars = dae.discretize()

    m.minimize(
        dae.least_squares("cA", t_data, yA)
        + dae.least_squares("cB", t_data, yB)
        + 1e-6 * rate.l2_penalty()
    )

    tp = dae._element_points()
    warm = {
        dvars["cA"]: np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CA0], yA])),
        dvars["cB"]: np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CB0], yB])),
    } | rate.initial_values()

    res = train(m, initial_solution=warm, options={"max_iter": 3000, "tol": 1e-8})
    assert res.status.name == "OPTIMAL"

    # With the correct symbolic structure, the NLP recovers the true constants
    # (a=1.5, b=0.3) to within the noise-limited tolerance.
    from discopt.warm_start import unflatten_solution

    vals = unflatten_solution(m, res.x)
    a_hat, b_hat = float(vals[rate.a]), float(vals[rate.b])
    assert a_hat == pytest.approx(1.5, abs=0.15), f"a={a_hat:.3f}"
    assert b_hat == pytest.approx(0.3, abs=0.15), f"b={b_hat:.3f}"


# ── test 4: a custom surrogate composes with fit_trajectories ───────────────


@pytest.mark.smoke
def test_custom_surrogate_in_multitrajectory_fit():
    from discopt.dae import Trajectory, fit_trajectories

    trajs = []
    for k, ca0 in enumerate((1.0, 0.7)):
        rng = np.random.default_rng(k)
        from scipy.integrate import solve_ivp

        t = np.linspace(0.05, TF, 12)
        sol = solve_ivp(
            lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
            (0, TF),
            [ca0, 0.0],
            t_eval=t,
            rtol=1e-10,
            atol=1e-12,
        )
        trajs.append(
            Trajectory(
                t_data=t,
                y_data={"cA": sol.y[0] + 0.01 * rng.standard_normal(12)},
                initial={"cA": ca0},
            )
        )

    m = Model()
    rate = RationalRate(m, name="r")
    fit = fit_trajectories(
        m,
        trajectories=trajs,
        states=[("cA", dict(bounds=(0.0, 1.5)))],
        rhs=lambda t, s, a, u: {"cA": -rate(s["cA"])},
        t_span=(0.0, TF),
        nfe=10,
        ncp=2,
    )
    m.minimize(fit.least_squares() + 1e-6 * rate.l2_penalty())
    res = train(
        m,
        initial_solution=fit.warm_start() | rate.initial_values(),
        options={"max_iter": 3000, "tol": 1e-8},
    )
    assert res.status.name == "OPTIMAL"
