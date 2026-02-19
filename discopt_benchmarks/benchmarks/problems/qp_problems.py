"""QP benchmark problems using the discopt modeling API.

Each problem is registered via the ``register`` function from ``base.py``.
Smoke problems have analytically-verified known optima.
Full problems include randomly-generated convex QPs with known optima
constructed from primal-dual KKT pairs.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from benchmarks.problems.base import TestProblem, register

_APPLICABLE = ["ipm", "ripopt", "ipopt"]


# ────────────────────────────────────────────────────────────────
# Smoke problems (5)
# ────────────────────────────────────────────────────────────────


def _build_qp_unconstrained_2d() -> dm.Model:
    """min (x-1)^2 + (y-2)^2, x,y in [-10,10]. Opt: 0.0 at (1,2)."""
    m = dm.Model("qp_unconstrained_2d")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    return m


register(
    TestProblem(
        name="qp_unconstrained_2d",
        category="qp",
        level="smoke",
        build_fn=_build_qp_unconstrained_2d,
        known_optimum=0.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        tags=["unconstrained", "separable"],
    )
)


def _build_qp_box_constrained() -> dm.Model:
    """min (x-3)^2 + (y-1)^2, 0<=x<=2, 0<=y<=2. Opt: 1.0 at (2,1)."""
    m = dm.Model("qp_box_constrained")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize((x - 3) ** 2 + (y - 1) ** 2)
    return m


register(
    TestProblem(
        name="qp_box_constrained",
        category="qp",
        level="smoke",
        build_fn=_build_qp_box_constrained,
        known_optimum=1.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        tags=["box_constrained"],
    )
)


# --- Portfolio 3: compute known optimum at module level ---
_P3_Q = np.array(
    [
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.03],
        [0.02, 0.03, 0.16],
    ]
)
_P3_MU = np.array([0.10, 0.15, 0.20])
_P3_TARGET = 0.12

_P3_KKT = np.zeros((5, 5))
_P3_KKT[:3, :3] = 2 * _P3_Q
_P3_KKT[:3, 3] = -1.0
_P3_KKT[:3, 4] = -_P3_MU
_P3_KKT[3, :3] = 1.0
_P3_KKT[4, :3] = _P3_MU
_P3_SOL = np.linalg.solve(_P3_KKT, [0, 0, 0, 1, _P3_TARGET])
_P3_XSTAR = _P3_SOL[:3]
_P3_OPT = float(_P3_XSTAR @ _P3_Q @ _P3_XSTAR)


def _build_qp_portfolio_3() -> dm.Model:
    """Markowitz 3-asset portfolio.

    min x'Qx  s.t.  sum(x)=1,  mu'x >= 0.12,  x >= 0.
    """
    q = _P3_Q
    mu = _P3_MU
    m = dm.Model("qp_portfolio_3")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    obj = dm.sum(
        lambda i: dm.sum(lambda j: q[i, j] * x[i] * x[j], over=range(3)),
        over=range(3),
    )
    m.minimize(obj)
    m.subject_to(x[0] + x[1] + x[2] == 1.0, name="budget")
    m.subject_to(
        mu[0] * x[0] + mu[1] * x[1] + mu[2] * x[2] >= _P3_TARGET,
        name="return",
    )
    return m


register(
    TestProblem(
        name="qp_portfolio_3",
        category="qp",
        level="smoke",
        build_fn=_build_qp_portfolio_3,
        known_optimum=_P3_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=3,
        n_constraints=2,
        tags=["portfolio", "markowitz"],
    )
)


def _build_qp_constrained_eq() -> dm.Model:
    """min x^2 + y^2  s.t. x + y = 1. Opt: 0.5 at (0.5, 0.5)."""
    m = dm.Model("qp_constrained_eq")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y == 1.0, name="eq")
    return m


register(
    TestProblem(
        name="qp_constrained_eq",
        category="qp",
        level="smoke",
        build_fn=_build_qp_constrained_eq,
        known_optimum=0.5,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=1,
        tags=["equality_constrained"],
    )
)


def _build_qp_least_squares() -> dm.Model:
    """min ||Ax - b||^2, A 3x2, b 3x1.

    A = [[1,1],[1,-1],[0,1]], b = [2,0,1].
    x* = [1, 1] (exact solution), ||residual||^2 = 0.0.
    """
    a_mat = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]])
    b_vec = np.array([2.0, 0.0, 1.0])

    m = dm.Model("qp_least_squares")
    x = m.continuous("x", shape=(2,), lb=-10.0, ub=10.0)

    obj = dm.sum(
        lambda i: (a_mat[i, 0] * x[0] + a_mat[i, 1] * x[1] - b_vec[i]) ** 2,
        over=range(3),
    )
    m.minimize(obj)
    return m


register(
    TestProblem(
        name="qp_least_squares",
        category="qp",
        level="smoke",
        build_fn=_build_qp_least_squares,
        known_optimum=0.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        tags=["least_squares"],
    )
)


# ────────────────────────────────────────────────────────────────
# Helpers for constructing QPs with known optima
# ────────────────────────────────────────────────────────────────


def _make_random_qp_with_known_opt(
    n: int,
    m_ineq: int,
    m_eq: int,
    seed: int,
) -> tuple[dm.Model, float, int]:
    """Build a random convex QP with known optimal value.

    Strategy:
    1. Generate random PSD matrix q_mat = B'B + eps*I.
    2. Pick a random feasible x_star in [0.5, 9.5].
    3. Generate constraints (half active, half slack) and equalities.
    4. Choose dual variables >= 0 for active inequalities.
    5. Compute c from KKT stationarity.
    6. Known optimum = 0.5 x*'q_mat*x* + c'x*.

    Returns (model, known_optimum, n_constraints).
    """
    rng = np.random.RandomState(seed)

    # PSD Hessian
    b_mat = rng.randn(n, n)
    q_mat = b_mat.T @ b_mat + 0.1 * np.eye(n)

    # Random primal solution in [0, 10]
    x_star = rng.uniform(0.5, 9.5, size=n)

    # Inequality constraints: a_ineq @ x <= b_ineq
    a_ineq = rng.randn(m_ineq, n)
    b_ineq = a_ineq @ x_star
    n_active = m_ineq // 2
    b_ineq[n_active:] += rng.uniform(0.5, 2.0, size=m_ineq - n_active)

    # Equality constraints: c_eq @ x = d_eq
    c_eq = rng.randn(m_eq, n) if m_eq > 0 else np.zeros((0, n))
    d_eq = c_eq @ x_star if m_eq > 0 else np.zeros(0)

    # Dual variables (non-negative for active inequalities, zero for slack)
    lam = np.zeros(m_ineq)
    lam[:n_active] = rng.uniform(0.1, 2.0, size=n_active)
    nu = rng.randn(m_eq) if m_eq > 0 else np.zeros(0)

    # Stationarity: q_mat @ x* + c_vec + a_ineq.T @ lam + c_eq.T @ nu = 0
    c_vec = -q_mat @ x_star - a_ineq.T @ lam
    if m_eq > 0:
        c_vec -= c_eq.T @ nu

    known_opt = float(0.5 * x_star @ q_mat @ x_star + c_vec @ x_star)

    # Build model
    name = f"qp_random_n{n}_s{seed}"
    model = dm.Model(name)
    x = model.continuous("x", shape=(n,), lb=0.0, ub=10.0)

    # Objective: 0.5 x'Qx + c'x
    obj_quad = dm.sum(
        lambda i: dm.sum(
            lambda j: 0.5 * q_mat[i, j] * x[i] * x[j],
            over=range(n),
        ),
        over=range(n),
    )
    obj_lin = dm.sum(lambda i: c_vec[i] * x[i], over=range(n))
    model.minimize(obj_quad + obj_lin)

    for k in range(m_ineq):
        row_k = a_ineq[k]
        lhs = dm.sum(lambda j, r=row_k: r[j] * x[j], over=range(n))
        model.subject_to(lhs <= b_ineq[k], name=f"ineq_{k}")

    for k in range(m_eq):
        row_k = c_eq[k]
        lhs = dm.sum(lambda j, r=row_k: r[j] * x[j], over=range(n))
        model.subject_to(lhs == d_eq[k], name=f"eq_{k}")

    return model, known_opt, m_ineq + m_eq


def _make_portfolio_problem(
    n: int,
    seed: int,
    target_frac: float = 0.5,
) -> tuple[dm.Model, float, int]:
    """Build a Markowitz portfolio problem with known optimum.

    min x'Qx  s.t.  1'x = 1,  mu'x >= target,  x >= 0.

    Known optimum via KKT (assuming interior solution and active
    return constraint). Falls back to scipy if KKT solution is
    infeasible.
    """
    rng = np.random.RandomState(seed)

    # Random covariance via factor model: Q = FF' + diag
    n_factors = max(1, n // 3)
    f_mat = rng.randn(n, n_factors) * 0.1
    diag_vals = rng.uniform(0.01, 0.05, size=n)
    q_mat = f_mat @ f_mat.T + np.diag(diag_vals)

    mu = rng.uniform(0.05, 0.25, size=n)
    target = float(np.mean(mu) * target_frac + np.max(mu) * (1 - target_frac))

    # KKT system: 2Qx - lam1*1 - lam2*mu = 0, 1'x=1, mu'x=target
    dim = n + 2
    a_kkt = np.zeros((dim, dim))
    a_kkt[:n, :n] = 2 * q_mat
    a_kkt[:n, n] = -1.0
    a_kkt[:n, n + 1] = -mu
    a_kkt[n, :n] = 1.0
    a_kkt[n + 1, :n] = mu
    rhs = np.zeros(dim)
    rhs[n] = 1.0
    rhs[n + 1] = target

    sol = np.linalg.solve(a_kkt, rhs)
    x_star = sol[:n]

    if np.any(x_star < -1e-10):
        res = scipy_minimize(
            lambda xv: xv @ q_mat @ xv,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq", "fun": lambda xv: np.sum(xv) - 1},
                {
                    "type": "ineq",
                    "fun": lambda xv: mu @ xv - target,
                },
            ],
        )
        x_star = res.x

    known_opt = float(x_star @ q_mat @ x_star)

    name = f"qp_portfolio_n{n}_s{seed}"
    model = dm.Model(name)
    x = model.continuous("x", shape=(n,), lb=0.0, ub=1.0)

    obj = dm.sum(
        lambda i: dm.sum(lambda j: q_mat[i, j] * x[i] * x[j], over=range(n)),
        over=range(n),
    )
    model.minimize(obj)

    budget_lhs = dm.sum(lambda i: x[i], over=range(n))
    model.subject_to(budget_lhs == 1.0, name="budget")

    ret_lhs = dm.sum(lambda i: mu[i] * x[i], over=range(n))
    model.subject_to(ret_lhs >= target, name="return_target")

    return model, known_opt, 2


# ────────────────────────────────────────────────────────────────
# Full problems: random convex QPs (5)
# ────────────────────────────────────────────────────────────────

_RANDOM_QP_CONFIGS = [
    # (n, m_ineq, m_eq, seed)
    (5, 3, 1, 100),
    (10, 5, 2, 101),
    (20, 10, 3, 102),
    (50, 20, 5, 103),
    (100, 40, 10, 104),
]

for _n, _m_ineq, _m_eq, _seed in _RANDOM_QP_CONFIGS:

    def _factory(n=_n, mi=_m_ineq, me=_m_eq, s=_seed):
        def _build():
            model, _, _ = _make_random_qp_with_known_opt(n, mi, me, s)
            return model

        return _build

    _, _opt_tmp, _ncon = _make_random_qp_with_known_opt(_n, _m_ineq, _m_eq, _seed)
    register(
        TestProblem(
            name=f"qp_random_n{_n}_s{_seed}",
            category="qp",
            level="full",
            build_fn=_factory(),
            known_optimum=_opt_tmp,
            applicable_solvers=_APPLICABLE,
            n_vars=_n,
            n_constraints=_ncon,
            tags=["random", "convex"],
        )
    )


# ────────────────────────────────────────────────────────────────
# Full problems: portfolio at various sizes (4)
# ────────────────────────────────────────────────────────────────

_PORTFOLIO_CONFIGS = [
    # (n, seed)
    (5, 200),
    (10, 201),
    (20, 202),
    (50, 203),
]

for _n, _seed in _PORTFOLIO_CONFIGS:

    def _port_factory(n=_n, s=_seed):
        def _build():
            model, _, _ = _make_portfolio_problem(n, s)
            return model

        return _build

    _, _opt_tmp, _ncon = _make_portfolio_problem(_n, _seed)
    register(
        TestProblem(
            name=f"qp_portfolio_n{_n}_s{_seed}",
            category="qp",
            level="full",
            build_fn=_port_factory(),
            known_optimum=_opt_tmp,
            applicable_solvers=_APPLICABLE,
            n_vars=_n,
            n_constraints=_ncon,
            tags=["portfolio", "markowitz"],
        )
    )


# ────────────────────────────────────────────────────────────────
# Full: structured problems (6)
# ────────────────────────────────────────────────────────────────

# --- Ridge regression ---
_RIDGE_RNG = np.random.RandomState(300)
_RIDGE_N, _RIDGE_P = 20, 10
_RIDGE_A = _RIDGE_RNG.randn(_RIDGE_N, _RIDGE_P)
_RIDGE_XTRUE = _RIDGE_RNG.randn(_RIDGE_P)
_RIDGE_B = _RIDGE_A @ _RIDGE_XTRUE + 0.1 * _RIDGE_RNG.randn(_RIDGE_N)
_RIDGE_LAM = 0.1
_RIDGE_XSTAR = np.linalg.solve(
    _RIDGE_A.T @ _RIDGE_A + _RIDGE_LAM * np.eye(_RIDGE_P),
    _RIDGE_A.T @ _RIDGE_B,
)
_RIDGE_RES = _RIDGE_A @ _RIDGE_XSTAR - _RIDGE_B
_RIDGE_OPT = float(_RIDGE_RES @ _RIDGE_RES + _RIDGE_LAM * (_RIDGE_XSTAR @ _RIDGE_XSTAR))


def _build_qp_regularized_regression() -> dm.Model:
    """Ridge regression: min ||Ax-b||^2 + lam*||x||^2.

    A (20x10), b (20,), lambda=0.1.
    Closed form: x* = (A'A + lam*I)^{-1} A'b.
    """
    a_mat, b_vec, lam = _RIDGE_A, _RIDGE_B, _RIDGE_LAM
    n_obs, p = _RIDGE_N, _RIDGE_P

    m = dm.Model("qp_regularized_regression")
    x = m.continuous("x", shape=(p,), lb=-20.0, ub=20.0)

    data_term = dm.sum(
        lambda i: (dm.sum(lambda j: a_mat[i, j] * x[j], over=range(p)) - b_vec[i]) ** 2,
        over=range(n_obs),
    )
    reg_term = dm.sum(lambda j: lam * x[j] ** 2, over=range(p))
    m.minimize(data_term + reg_term)
    return m


register(
    TestProblem(
        name="qp_regularized_regression",
        category="qp",
        level="full",
        build_fn=_build_qp_regularized_regression,
        known_optimum=_RIDGE_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=10,
        n_constraints=0,
        tags=["regression", "ridge"],
    )
)


# --- Elastic net ---
_EN_RNG = np.random.RandomState(301)
_EN_NOBS, _EN_P = 15, 8
_EN_A = _EN_RNG.randn(_EN_NOBS, _EN_P)
_EN_XTRUE = np.zeros(_EN_P)
_EN_XTRUE[:3] = _EN_RNG.randn(3)
_EN_B = _EN_A @ _EN_XTRUE + 0.1 * _EN_RNG.randn(_EN_NOBS)
_EN_LAM1, _EN_LAM2 = 0.05, 0.05

_en_z0 = np.zeros(2 * _EN_P)
_en_z0[:_EN_P] = np.linalg.lstsq(_EN_A, _EN_B, rcond=None)[0]
_en_z0[_EN_P:] = np.abs(_en_z0[:_EN_P])

_EN_RES = scipy_minimize(
    lambda z: float(
        (_EN_A @ z[:_EN_P] - _EN_B) @ (_EN_A @ z[:_EN_P] - _EN_B)
        + _EN_LAM1 * np.sum(z[_EN_P:])
        + _EN_LAM2 * z[:_EN_P] @ z[:_EN_P]
    ),
    x0=_en_z0,
    method="SLSQP",
    bounds=[(-20, 20)] * _EN_P + [(0, 20)] * _EN_P,
    constraints=[{"type": "ineq", "fun": lambda z, j=j: z[_EN_P + j] - z[j]} for j in range(_EN_P)]
    + [{"type": "ineq", "fun": lambda z, j=j: z[_EN_P + j] + z[j]} for j in range(_EN_P)],
)
_EN_OPT = float(_EN_RES.fun)


def _build_qp_elastic_net() -> dm.Model:
    """Elastic net QP relaxation.

    min ||Ax-b||^2 + lam1*1'u + lam2*||x||^2
    s.t. u_j >= x_j, u_j >= -x_j  (L1 linearization).
    A (15x8), lam1=0.05, lam2=0.05.
    """
    a_mat, b_vec = _EN_A, _EN_B
    lam1, lam2 = _EN_LAM1, _EN_LAM2
    n_obs, p = _EN_NOBS, _EN_P

    m = dm.Model("qp_elastic_net")
    x = m.continuous("x", shape=(p,), lb=-20.0, ub=20.0)
    u = m.continuous("u", shape=(p,), lb=0.0, ub=20.0)

    data_term = dm.sum(
        lambda i: (dm.sum(lambda j: a_mat[i, j] * x[j], over=range(p)) - b_vec[i]) ** 2,
        over=range(n_obs),
    )
    l1_term = dm.sum(lambda j: lam1 * u[j], over=range(p))
    l2_term = dm.sum(lambda j: lam2 * x[j] ** 2, over=range(p))
    m.minimize(data_term + l1_term + l2_term)

    for j in range(p):
        m.subject_to(u[j] >= x[j], name=f"l1_pos_{j}")
        m.subject_to(u[j] >= -x[j], name=f"l1_neg_{j}")

    return m


register(
    TestProblem(
        name="qp_elastic_net",
        category="qp",
        level="full",
        build_fn=_build_qp_elastic_net,
        known_optimum=_EN_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=16,
        n_constraints=16,
        tags=["elastic_net", "regression", "l1_relaxation"],
    )
)


# --- Distance to polyhedron (projection) ---
_DIST_RNG = np.random.RandomState(302)
_DIST_N = 5
_DIST_P = _DIST_RNG.randn(_DIST_N) * 3
_DIST_M = 8
_DIST_A = _DIST_RNG.randn(_DIST_M, _DIST_N)
_DIST_B = np.abs(_DIST_A @ np.zeros(_DIST_N)) + _DIST_RNG.uniform(1, 3, size=_DIST_M)
_DIST_RES = scipy_minimize(
    lambda xv: float(np.sum((xv - _DIST_P) ** 2)),
    x0=np.zeros(_DIST_N),
    method="SLSQP",
    bounds=[(-10, 10)] * _DIST_N,
    constraints=[
        {
            "type": "ineq",
            "fun": lambda xv, k=k: _DIST_B[k] - _DIST_A[k] @ xv,
        }
        for k in range(_DIST_M)
    ],
)
_DIST_OPT = float(_DIST_RES.fun)


def _build_qp_distance_to_polyhedron() -> dm.Model:
    """min ||x - p||^2 s.t. Ax <= b (project onto polyhedron).

    n=5 variables, 8 inequality constraints.
    """
    n = _DIST_N
    p_vec = _DIST_P
    a_con, b_con = _DIST_A, _DIST_B
    m_con = _DIST_M

    model = dm.Model("qp_distance_to_polyhedron")
    x = model.continuous("x", shape=(n,), lb=-10.0, ub=10.0)

    obj = dm.sum(lambda i: (x[i] - p_vec[i]) ** 2, over=range(n))
    model.minimize(obj)

    for k in range(m_con):
        row_k = a_con[k]
        lhs = dm.sum(lambda j, r=row_k: r[j] * x[j], over=range(n))
        model.subject_to(lhs <= b_con[k], name=f"poly_{k}")

    return model


register(
    TestProblem(
        name="qp_distance_to_polyhedron",
        category="qp",
        level="full",
        build_fn=_build_qp_distance_to_polyhedron,
        known_optimum=_DIST_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=5,
        n_constraints=8,
        tags=["projection", "polyhedron"],
    )
)


# --- Weighted least squares ---
_WLS_RNG = np.random.RandomState(303)
_WLS_N, _WLS_P = 25, 8
_WLS_A = _WLS_RNG.randn(_WLS_N, _WLS_P)
_WLS_B = _WLS_RNG.randn(_WLS_N)
_WLS_W = _WLS_RNG.uniform(0.5, 2.0, size=_WLS_N)
_WLS_W_DIAG = np.diag(_WLS_W)
_WLS_XSTAR = np.linalg.solve(
    _WLS_A.T @ _WLS_W_DIAG @ _WLS_A,
    _WLS_A.T @ _WLS_W_DIAG @ _WLS_B,
)
_WLS_RES_VEC = _WLS_A @ _WLS_XSTAR - _WLS_B
_WLS_OPT = float(_WLS_RES_VEC @ _WLS_W_DIAG @ _WLS_RES_VEC)


def _build_qp_weighted_least_squares() -> dm.Model:
    """Weighted least squares: min sum_i w_i * (a_i'x - b_i)^2.

    A (25x8), diagonal weights w in [0.5, 2.0].
    Closed form via weighted normal equations.
    """
    a_mat, b_vec, w = _WLS_A, _WLS_B, _WLS_W
    n_obs, p = _WLS_N, _WLS_P

    m = dm.Model("qp_weighted_least_squares")
    x = m.continuous("x", shape=(p,), lb=-20.0, ub=20.0)

    obj = dm.sum(
        lambda i: w[i] * (dm.sum(lambda j: a_mat[i, j] * x[j], over=range(p)) - b_vec[i]) ** 2,
        over=range(n_obs),
    )
    m.minimize(obj)
    return m


register(
    TestProblem(
        name="qp_weighted_least_squares",
        category="qp",
        level="full",
        build_fn=_build_qp_weighted_least_squares,
        known_optimum=_WLS_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=8,
        n_constraints=0,
        tags=["least_squares", "weighted"],
    )
)


# --- Regularized regression with equality constraints ---
_REQ_RNG = np.random.RandomState(304)
_REQ_N, _REQ_P = 20, 10
_REQ_A = _REQ_RNG.randn(_REQ_N, _REQ_P)
_REQ_B = _REQ_RNG.randn(_REQ_N)
_REQ_LAM = 0.2
_REQ_NEQS = 3
_REQ_C = _REQ_RNG.randn(_REQ_NEQS, _REQ_P)
_REQ_D = _REQ_RNG.randn(_REQ_NEQS)

_req_lhs = np.block(
    [
        [_REQ_A.T @ _REQ_A + _REQ_LAM * np.eye(_REQ_P), _REQ_C.T],
        [_REQ_C, np.zeros((_REQ_NEQS, _REQ_NEQS))],
    ]
)
_req_rhs = np.concatenate([_REQ_A.T @ _REQ_B, _REQ_D])
_req_sol = np.linalg.solve(_req_lhs, _req_rhs)
_REQ_XSTAR = _req_sol[:_REQ_P]
_req_res = _REQ_A @ _REQ_XSTAR - _REQ_B
_REQ_OPT = float(_req_res @ _req_res + _REQ_LAM * (_REQ_XSTAR @ _REQ_XSTAR))


def _build_qp_constrained_regression() -> dm.Model:
    """Ridge regression with equality constraints.

    min ||Ax-b||^2 + lam*||x||^2  s.t.  Cx = d.
    A (20x10), 3 equality constraints, lam=0.2.
    """
    a_mat, b_vec, lam = _REQ_A, _REQ_B, _REQ_LAM
    c_mat, d_vec = _REQ_C, _REQ_D
    n_obs, p = _REQ_N, _REQ_P
    n_eq = _REQ_NEQS

    m = dm.Model("qp_constrained_regression")
    x = m.continuous("x", shape=(p,), lb=-20.0, ub=20.0)

    data_term = dm.sum(
        lambda i: (dm.sum(lambda j: a_mat[i, j] * x[j], over=range(p)) - b_vec[i]) ** 2,
        over=range(n_obs),
    )
    reg_term = dm.sum(lambda j: lam * x[j] ** 2, over=range(p))
    m.minimize(data_term + reg_term)

    for k in range(n_eq):
        row_k = c_mat[k]
        lhs = dm.sum(lambda j, r=row_k: r[j] * x[j], over=range(p))
        m.subject_to(lhs == d_vec[k], name=f"eq_{k}")

    return m


register(
    TestProblem(
        name="qp_constrained_regression",
        category="qp",
        level="full",
        build_fn=_build_qp_constrained_regression,
        known_optimum=_REQ_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=10,
        n_constraints=3,
        tags=["regression", "equality_constrained"],
    )
)


# --- Bounded least squares with inequality constraints ---
_BLS_RNG = np.random.RandomState(305)
_BLS_N, _BLS_P = 15, 6
_BLS_A = _BLS_RNG.randn(_BLS_N, _BLS_P)
_BLS_B = _BLS_RNG.randn(_BLS_N)
# Add inequality constraints: Cx <= d
_BLS_NC = 4
_BLS_C = _BLS_RNG.randn(_BLS_NC, _BLS_P)
_BLS_D = _BLS_RNG.uniform(1.0, 5.0, size=_BLS_NC)

_BLS_RES = scipy_minimize(
    lambda x: float((_BLS_A @ x - _BLS_B) @ (_BLS_A @ x - _BLS_B)),
    x0=np.zeros(_BLS_P),
    method="SLSQP",
    bounds=[(-10, 10)] * _BLS_P,
    constraints=[
        {"type": "ineq", "fun": lambda x, k=k: _BLS_D[k] - _BLS_C[k] @ x}
        for k in range(_BLS_NC)
    ],
)
_BLS_OPT = float(_BLS_RES.fun)


def _build_qp_bounded_ls() -> dm.Model:
    """Bounded least squares: min ||Ax-b||^2  s.t. Cx <= d.

    A (15x6), 4 inequality constraints.
    """
    a_mat, b_vec = _BLS_A, _BLS_B
    c_con, d_con = _BLS_C, _BLS_D
    n_obs, p = _BLS_N, _BLS_P

    m = dm.Model("qp_bounded_ls")
    x = m.continuous("x", shape=(p,), lb=-10.0, ub=10.0)

    obj = dm.sum(
        lambda i: (dm.sum(lambda j: a_mat[i, j] * x[j], over=range(p)) - b_vec[i]) ** 2,
        over=range(n_obs),
    )
    m.minimize(obj)

    for k in range(_BLS_NC):
        row_k = c_con[k]
        lhs = dm.sum(lambda j, r=row_k: r[j] * x[j], over=range(p))
        m.subject_to(lhs <= d_con[k], name=f"ineq_{k}")

    return m


register(
    TestProblem(
        name="qp_bounded_ls",
        category="qp",
        level="full",
        build_fn=_build_qp_bounded_ls,
        known_optimum=_BLS_OPT,
        applicable_solvers=_APPLICABLE,
        n_vars=_BLS_P,
        n_constraints=_BLS_NC,
        tags=["least_squares", "inequality_constrained"],
    )
)
