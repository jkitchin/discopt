"""Tests for the multi-trajectory fitting glue (``discopt.dae.fit``).

Covers the HM2 acceptance criteria from
``docs/dev/hybrid-ml-implementation-plan.md`` §6:

1. ``unflatten_solution`` round-trip (HM2.1);
2. two-trajectory shared-weight fit with the kernel surrogate (smoke);
3. ``align_grid`` places measurement times on element boundaries;
4. variable/constraint counts scale as n_traj * per-trajectory + shared weights.
"""

import numpy as np
import pytest
from discopt.dae import Trajectory, fit_trajectories
from discopt.modeling import Model
from discopt.nn import TrainableKernelExpansion, train
from discopt.warm_start import unflatten_solution, validate_initial_solution

TF = 2.0


def _r_true(c):
    return 1.5 * c**2 / (0.3 + c)


def _make_data(seed, ca0):
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, TF, 15)
    sol = solve_ivp(
        lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
        (0, TF),
        [ca0, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    yA = sol.y[0] + 0.01 * rng.standard_normal(15)
    yB = sol.y[1] + 0.01 * rng.standard_normal(15)
    return t, yA, yB


# ── test 1: unflatten round-trip ────────────────────────────────────────────


def test_unflatten_solution_round_trip():
    m = Model()
    a = m.continuous("a", lb=-5, ub=5)  # scalar
    b = m.continuous("b", shape=(3,), lb=-5, ub=5)  # 1-D
    c = m.continuous("c", shape=(2, 2), lb=-5, ub=5)  # 2-D
    d = {a: 1.5, b: np.array([1.0, 2.0, 3.0]), c: np.array([[0.1, 0.2], [0.3, 0.4]])}

    x = validate_initial_solution(m, d)
    back = unflatten_solution(m, x)

    assert float(back[a]) == pytest.approx(1.5)
    assert back[a].shape == ()
    np.testing.assert_allclose(back[b], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(back[c], [[0.1, 0.2], [0.3, 0.4]])
    assert back[c].shape == (2, 2)

    # Wrong length is refused.
    with pytest.raises(ValueError, match=r"expected"):
        unflatten_solution(m, np.zeros(3))


# ── test 2: two-trajectory shared-weight fit ────────────────────────────────


@pytest.mark.smoke
def test_two_trajectory_shared_weight_fit():
    trajs = []
    for k, ca0 in enumerate((1.0, 0.7)):
        t, yA, yB = _make_data(seed=k, ca0=ca0)
        trajs.append(
            Trajectory(t_data=t, y_data={"cA": yA, "cB": yB}, initial={"cA": ca0, "cB": 0.0})
        )

    m = Model()
    ke = TrainableKernelExpansion(m, np.linspace(0.0, 1.05, 10), lengthscale=0.15, name="k")

    def rhs(t, s, a, u):
        r = ke(s["cA"])
        return {"cA": -r, "cB": r}

    fit = fit_trajectories(
        m,
        trajectories=trajs,
        states=[("cA", dict(bounds=(0.0, 1.5))), ("cB", dict(bounds=(0.0, 1.5)))],
        rhs=rhs,
        t_span=(0.0, TF),
        nfe=8,
        ncp=2,
    )
    m.minimize(fit.least_squares() + 1e-4 * ke.l2_penalty())
    res = train(
        m,
        initial_solution=fit.warm_start() | ke.initial_values(),
        options={"max_iter": 3000, "tol": 1e-8},
    )
    assert res.status.name == "OPTIMAL"

    # Both trajectories fit below 2x the noise level.
    for k, tr in enumerate(trajs):
        t_traj, cA = fit.extract(res, k, "cA")
        pred = np.interp(tr.t_data, t_traj, cA)
        rmse = float(np.sqrt(np.mean((pred - tr.y_data["cA"]) ** 2)))
        assert rmse <= 0.02, f"trajectory {k} RMSE {rmse:.4f} exceeds 0.02"


# ── test 3: align_grid places measurement times on element boundaries ───────


def test_align_grid_snaps_measurement_times_to_boundaries():
    t, yA, yB = _make_data(seed=0, ca0=1.0)
    trajs = [Trajectory(t_data=t, y_data={"cA": yA}, initial={"cA": 1.0})]

    m = Model()
    ke = TrainableKernelExpansion(m, np.linspace(0.0, 1.05, 6), lengthscale=0.2, name="k")

    fit = fit_trajectories(
        m,
        trajectories=trajs,
        states=[("cA", dict(bounds=(0.0, 1.5)))],
        rhs=lambda t, s, a, u: {"cA": -ke(s["cA"])},
        t_span=(0.0, TF),
        nfe=15,
        ncp=2,
        align_grid=True,
    )
    boundaries = fit.builders[0]._element_boundaries()
    # Every interior measurement time close to a boundary should coincide with one.
    from discopt.dae.collocation import align_time_grid

    expected = align_time_grid((0.0, TF), 15, t)
    np.testing.assert_allclose(boundaries, expected)
    # At least one interior measurement time landed exactly on a boundary.
    interior = t[(t > 0.0) & (t < TF)]
    hits = sum(np.any(np.isclose(boundaries, ti)) for ti in interior)
    assert hits >= 1


# ── test 4: variable/constraint counts scale correctly ──────────────────────


def test_counts_scale_with_trajectories_plus_shared_weights():
    """n_traj identical blocks + shared weights: no accidental variable duplication."""

    def _build(n_traj):
        m = Model()
        ke = TrainableKernelExpansion(m, np.linspace(0.0, 1.05, 7), lengthscale=0.2, name="k")
        trajs = [
            Trajectory(
                t_data=_make_data(seed=k, ca0=1.0)[0],
                y_data={"cA": _make_data(seed=k, ca0=1.0)[1]},
                initial={"cA": 1.0},
            )
            for k in range(n_traj)
        ]
        fit_trajectories(
            m,
            trajectories=trajs,
            states=[("cA", dict(bounds=(0.0, 1.5)))],
            rhs=lambda t, s, a, u: {"cA": -ke(s["cA"])},
            t_span=(0.0, TF),
            nfe=6,
            ncp=2,
        )
        n_vars = sum(v.size for v in m._variables)
        n_cons = len(m._constraints)
        return n_vars, n_cons, ke.n_parameters()

    v1, c1, w1 = _build(1)
    v2, c2, w2 = _build(2)
    assert w1 == w2 == 7  # shared alpha, same in both
    per_traj_vars = v1 - w1
    per_traj_cons = c1
    assert v2 == w2 + 2 * per_traj_vars  # weights shared, state blocks doubled
    assert c2 == 2 * per_traj_cons  # constraint blocks doubled
