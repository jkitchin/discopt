"""Bilevel Phase 1: BilevelProblem + KKT reformulation of an LP lower level.

The end-to-end *certified* solve of a bilevel program needs the global MINLP
solver (Rust + pounce) and runs in CI. What is validated here without the solver
is the thing Phase 1 actually adds: that the KKT reformulation is a **correct
characterization of follower optimality**. For an LP lower level we compute the
follower optimum and its multipliers with an independent scipy oracle, then check
that the exact constraints `BilevelProblem.formulate` emitted — stationarity,
primal feasibility, dual feasibility, complementarity — are satisfied there, and
that stationarity genuinely binds (nonzero off the KKT multiplier). Plus the
structural contract and every loud-refusal gate.

Needs JAX (to compile/evaluate the emitted expressions) and scipy (the oracle);
no Rust extension.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
scipy_opt = pytest.importorskip("scipy.optimize")
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression_params  # noqa: E402
from discopt.bilevel import BilevelProblem  # noqa: E402
from discopt.modeling.core import Model  # noqa: E402

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _eval(body, model, values):
    """Evaluate an expression body at a {Variable: scalar} assignment."""
    flat = []
    for v in model._variables:
        flat.append(float(values.get(v, 0.0)))
    params = tuple(jnp.asarray(p.value) for p in model._parameters)
    return float(compile_expression_params(body, model)(jnp.asarray(flat), params))


def _follower_oracle(d, A, b, bounds):
    """Solve min d·y s.t. A y <= b, and recover KKT multipliers on the active set.

    Returns (y*, mu) with mu >= 0 aligned to the rows of (A, b). Instances are
    chosen so the binding rows are A-rows (not variable bounds), so the active-set
    multiplier solve is exact.
    """
    d = np.asarray(d, float)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    res = scipy_opt.linprog(c=d, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    assert res.success, res.message
    y = res.x
    slack = b - A @ y
    active = np.where(np.abs(slack) < 1e-7)[0]
    mu = np.zeros(len(b))
    if len(active):
        mu_a, *_ = np.linalg.lstsq(A[active].T, -d, rcond=None)
        mu[active] = mu_a
    return y, mu


# ---------------------------------------------------------------------------
# 1. Semantic correctness: KKT holds exactly at the follower optimum.
# ---------------------------------------------------------------------------


def _build_instance_min():
    """Follower: min_y y  s.t.  x + y >= 3,  y <= 2x,  y in [0,10]."""
    m = Model("min_follower")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)  # leader
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
        lower_sense="min",
    )
    bl.build_kkt_system()  # sound KKT math; the big-M encoding is tested separately
    # signed objective gradient d, and (A, b(x)) for the follower LP in y.
    #   g0: 3 - x - y <= 0  ->  -y <= x - 3
    #   g1: y - 2x   <= 0  ->   y <= 2x
    d = [1.0]
    A = [[-1.0], [1.0]]
    b_of_x = lambda xv: [xv - 3.0, 2.0 * xv]  # noqa: E731
    return m, x, y, bl, d, A, b_of_x


def _build_instance_max():
    """Follower: max_y y  s.t.  y <= 2x,  y <= 5,  y in [0,10]."""
    m = Model("max_follower")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-y - x)  # leader (arbitrary)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[y <= 2 * x, y <= 5],
        lower_sense="max",
    )
    bl.build_kkt_system()  # sound KKT math; the big-M encoding is tested separately
    # signed objective = -y (max y == min -y) -> d = [-1]
    d = [-1.0]
    A = [[1.0], [1.0]]
    b_of_x = lambda xv: [2.0 * xv, 5.0]  # noqa: E731
    return m, x, y, bl, d, A, b_of_x


@pytest.mark.parametrize(
    "build,xs",
    [
        (_build_instance_min, [1.2, 1.8, 2.5]),
        (_build_instance_max, [0.8, 1.5, 2.2]),
    ],
)
def test_kkt_satisfied_at_follower_optimum(build, xs):
    m, x, y, bl, d, A, b_of_x = build()
    for xv in xs:
        y_star, mu = _follower_oracle(d, A, b_of_x(xv), bounds=[(0.0, 10.0)])
        assign = {x: xv, y: float(y_star[0])}
        # oracle mu aligns with the user constraints (first); finite follower-bound
        # multipliers are 0 at these interior optima and default to 0 in _eval.
        for k, muv in enumerate(mu):
            assign[bl.kkt.multipliers[k]] = float(muv)

        # stationarity ∂L/∂y == 0
        for c in bl.kkt.stationarity:
            assert abs(_eval(c.body, m, assign)) < 1e-7, f"stationarity != 0 at x={xv}"
        # dual feasibility μ >= 0
        assert np.all(mu >= -1e-9)
        # primal feasibility g_i <= 0  and  complementarity μ_i·(-g_i) == 0
        for k, p in enumerate(bl.kkt.comp_pairs):
            g_side = _eval(p.g, m, assign)  # -g_i, must be >= 0 (primal feas)
            mu_side = _eval(p.f, m, assign)  # μ_i, must be >= 0
            assert g_side >= -1e-7, f"primal feas violated (row {k}) at x={xv}"
            assert mu_side >= -1e-9
            assert abs(mu_side * g_side) < 1e-7, f"complementarity != 0 (row {k}) at x={xv}"


def test_stationarity_actually_binds():
    """A perturbed multiplier must break stationarity (guards a 'always 0' bug)."""
    m, x, y, bl, d, A, b_of_x = _build_instance_min()
    xv = 1.5
    y_star, mu = _follower_oracle(d, A, b_of_x(xv), bounds=[(0.0, 10.0)])
    assign = {x: xv, y: float(y_star[0])}
    for k, muv in enumerate(mu):
        assign[bl.kkt.multipliers[k]] = float(muv)
    # correct: ~0
    assert abs(_eval(bl.kkt.stationarity[0].body, m, assign)) < 1e-7
    # perturb μ0 -> stationarity residual must move by exactly the coefficient (-1)
    assign[bl.kkt.multipliers[0]] += 1.0
    assert abs(_eval(bl.kkt.stationarity[0].body, m, assign) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# 2. Structural contract.
# ---------------------------------------------------------------------------


def test_structure_and_leader_objective_preserved():
    m = Model("s")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    leader_obj = m._objective.expression
    leader_sense = m._objective.sense
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
    )
    bl.build_kkt_system()
    # one multiplier per lower constraint (2 user + 2 finite follower bounds y in
    # [0,10]); one stationarity row per lower var; all four are inequalities.
    assert len(bl.lower_constraints_full) == 4
    assert len(bl.kkt.multipliers) == 4
    assert len(bl.kkt.stationarity) == 1
    assert len(bl.kkt.comp_pairs) == 4
    # leader objective untouched
    assert m._objective.expression is leader_obj
    assert m._objective.sense == leader_sense


def test_equality_lower_constraint_uses_free_multiplier_no_complementarity():
    m = Model("eq")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y")  # unbounded follower -> no synthesized bound constraints
    m.minimize(x + y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y == 2],  # equality -> free ν, no complementarity
    )
    bl.formulate(method="kkt", mpec_method="gdp")
    assert len(bl.kkt.multipliers) == 1
    assert len(bl.kkt.comp_pairs) == 0
    nu = bl.kkt.multipliers[0]
    assert float(nu.lb) < 0  # free (default negative lower bound), not >= 0


def test_double_formulate_refused():
    m = Model("d")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m, upper_vars=[x], lower_vars=[y], lower_objective=y, lower_constraints=[x + y >= 3]
    )
    # strong_duality has no big-M multiplier gate, so it formulates in place; the
    # double-call guard must then refuse a second formulate.
    bl.formulate(method="strong_duality")
    with pytest.raises(RuntimeError, match="already been called"):
        bl.formulate(method="strong_duality")


# ---------------------------------------------------------------------------
# 3. Loud-refusal gates.
# ---------------------------------------------------------------------------


def test_integer_lower_variable_refused():
    m = Model("int")
    x = m.continuous("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x - y)
    with pytest.raises(NotImplementedError, match="continuous"):
        BilevelProblem(
            m, upper_vars=[x], lower_vars=[y], lower_objective=y, lower_constraints=[x + y >= 3]
        )


def test_nonconvex_lower_objective_refused():
    # A concave objective for a *minimizing* follower is a nonconvex lower level:
    # KKT is not sufficient -> must refuse (convex-QP y*y is accepted, see Phase 2).
    m = Model("nl")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=-(y * y),  # concave -> nonconvex for min
        lower_constraints=[x + y >= 3],
    )
    with pytest.raises(NotImplementedError, match="nonconvex"):
        bl.formulate(method="kkt")


def test_bilinear_xy_is_allowed_affine_in_y():
    """x·y is a coefficient on y (affine in y), so the LP gate must accept it."""
    m = Model("bilin")
    x = m.continuous("x", lb=1, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=x * y,  # affine in y (x is the coefficient)
        lower_constraints=[x + y >= 3],
    )
    bl.build_kkt_system()  # convexity gate must accept affine-in-y and build the KKT
    assert bl.kkt is not None and len(bl.kkt.stationarity) == 1


def test_pessimistic_refused():
    m = Model("p")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m, upper_vars=[x], lower_vars=[y], lower_objective=y, lower_constraints=[x + y >= 3]
    )
    with pytest.raises(NotImplementedError, match="pessimistic"):
        bl.formulate(method="pessimistic")


def test_overlapping_upper_lower_refused():
    m = Model("ov")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    with pytest.raises(ValueError, match="both upper_vars and lower_vars"):
        BilevelProblem(
            m, upper_vars=[x], lower_vars=[x], lower_objective=x, lower_constraints=[x >= 1]
        )
