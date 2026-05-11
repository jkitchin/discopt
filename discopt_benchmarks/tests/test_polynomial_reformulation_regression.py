"""M4 + M5 regression: polynomial-to-quadratic reformulation + derived
McCormick aux bounds (issue #51).

The pass (``discopt._rust.PyModelRepr.reformulate_polynomial``) rewrites
every monomial of total degree > 2 in any constraint body or the
objective into a sequence of bilinear auxiliary equalities, and assigns
each new aux variable an interval bound derived from forward-interval
propagation of the defining product (McCormick range). This is the M4
core plus the M5 reduction-constraint derivation in a single shot.

The acceptance contract:

1. **Feasibility preservation (M4):** for every sampled point of the
   original variable box, the rewritten constraint body equals the
   original body (within 1e-9) once aux variables are filled by their
   defining products. No feasible point of the original is excluded.
2. **Aux-bound soundness (M5):** for every sampled feasible point and
   every aux variable, the value `a · b` lies inside the declared
   `[lb, ub]` for that aux. Bounds are sound enclosures.
3. **Idempotence:** running the pass twice produces no further
   rewrites — the second invocation reports
   `constraints_rewritten == 0` and introduces no new aux variables.
4. **Aux variable sharing:** identical bilinear products across
   monomials share a single aux variable (canonical pair cache).

Breaking any of these is a correctness regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._rust import model_to_repr

N_REGRESSION_SAMPLES = 10_000


def _fill_aux(new_repr, xi_yi: np.ndarray) -> np.ndarray:
    """Reconstruct full variable vector by evaluating each aux equality
    in order. Each aux equality has the form `aux_k - product == 0`, so
    setting aux_k = 0 gives body = -product, and we recover aux_k =
    -body."""
    n = new_repr.n_vars
    x = np.zeros(n)
    x[: len(xi_yi)] = xi_yi
    n_orig = len(xi_yi)
    n_aux = n - n_orig
    # Aux equality constraints follow the original constraints in order.
    n_orig_constraints = new_repr.n_constraints - n_aux
    for k_aux in range(n_aux):
        x_t = x.copy()
        x_t[n_orig + k_aux] = 0.0
        body = new_repr.evaluate_constraint(n_orig_constraints + k_aux, x_t)
        x[n_orig + k_aux] = -body
    return x


# ---------------------------------------------------------------------------
# 1. Feasibility preservation (M4)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.regression
def test_quartic_polynomial_reformulation_preserves_body():
    """x^3 * y + 0.5 * y^3 ≤ 1 — degree 4 monomial gets rewritten;
    rewritten body must equal original at every sample."""
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x * x * x * y + 0.5 * y * y * y <= 1.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.reformulate_polynomial()
    assert stats["constraints_rewritten"] >= 1
    assert stats["aux_variables_introduced"] >= 1

    rng = np.random.default_rng(0)
    n_samples = N_REGRESSION_SAMPLES
    xs = rng.uniform(-2.0, 2.0, n_samples)
    ys = rng.uniform(-2.0, 2.0, n_samples)
    diffs = np.empty(n_samples)
    for i in range(n_samples):
        orig = repr_.evaluate_constraint(0, np.array([xs[i], ys[i]]))
        x_full = _fill_aux(new_repr, np.array([xs[i], ys[i]]))
        new = new_repr.evaluate_constraint(0, x_full)
        diffs[i] = orig - new
    assert np.max(np.abs(diffs)) < 1e-9


@pytest.mark.regression
def test_cubic_objective_reformulation_preserves_value():
    """Objective with cubic monomial must evaluate identically after
    reformulation."""
    m = Model()
    x = m.continuous("x", lb=-1.5, ub=1.5)
    y = m.continuous("y", lb=-1.5, ub=1.5)
    m.minimize(x * x * y + y * y * x)
    m.subject_to(x + y <= 2.0)  # keep at least one constraint

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.reformulate_polynomial()
    assert stats["aux_variables_introduced"] >= 1

    rng = np.random.default_rng(1)
    n_samples = N_REGRESSION_SAMPLES
    xs = rng.uniform(-1.5, 1.5, n_samples)
    ys = rng.uniform(-1.5, 1.5, n_samples)
    for i in range(n_samples):
        orig = repr_.evaluate_objective(np.array([xs[i], ys[i]]))
        x_full = _fill_aux(new_repr, np.array([xs[i], ys[i]]))
        new = new_repr.evaluate_objective(x_full)
        assert abs(orig - new) < 1e-9


# ---------------------------------------------------------------------------
# 2. Aux-bound soundness (M5)
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_aux_bounds_enclose_product_at_every_feasible_point():
    """Every aux variable's declared [lb, ub] must enclose the value of
    its defining bilinear product over the entire original variable box."""
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x * x * x * y + 0.5 * y * y * y <= 1.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.reformulate_polynomial()
    n = new_repr.n_vars
    n_aux = n - 2
    lbs = [new_repr.var_lb(i) for i in range(n)]
    ubs = [new_repr.var_ub(i) for i in range(n)]
    # Coerce to scalars (var_lb returns [[lo]] for size-1 var blocks).
    aux_lbs = [float(np.asarray(lbs[2 + k]).ravel()[0]) for k in range(n_aux)]
    aux_ubs = [float(np.asarray(ubs[2 + k]).ravel()[0]) for k in range(n_aux)]

    rng = np.random.default_rng(2)
    n_samples = N_REGRESSION_SAMPLES
    xs = rng.uniform(-2.0, 2.0, n_samples)
    ys = rng.uniform(-2.0, 2.0, n_samples)
    for i in range(n_samples):
        x_full = _fill_aux(new_repr, np.array([xs[i], ys[i]]))
        for k in range(n_aux):
            v = float(x_full[2 + k])
            assert v >= aux_lbs[k] - 1e-9, f"aux{k}={v} < lb={aux_lbs[k]}"
            assert v <= aux_ubs[k] + 1e-9, f"aux{k}={v} > ub={aux_ubs[k]}"

    assert stats["aux_bounds_derived"] == n_aux


# ---------------------------------------------------------------------------
# 3. Idempotence
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_reformulation_is_idempotent():
    m = Model()
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x + y)
    m.subject_to(x * x * x + y * y * y <= 1.0)

    repr_ = model_to_repr(m)
    new_repr, stats1 = repr_.reformulate_polynomial()
    new_new_repr, stats2 = new_repr.reformulate_polynomial()

    assert stats1["aux_variables_introduced"] >= 1
    # Second pass: nothing left of degree > 2.
    assert stats2["constraints_rewritten"] == 0
    assert stats2["aux_variables_introduced"] == 0
    assert new_new_repr.n_vars == new_repr.n_vars
    assert new_new_repr.n_constraints == new_repr.n_constraints


# ---------------------------------------------------------------------------
# 4. Quadratic-only models are skipped
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_quadratic_model_not_rewritten():
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.minimize(x)
    m.subject_to(x * x <= 4.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.reformulate_polynomial()
    assert stats["constraints_rewritten"] == 0
    assert stats["aux_variables_introduced"] == 0
    assert new_repr.n_vars == repr_.n_vars
    assert new_repr.n_constraints == repr_.n_constraints


# ---------------------------------------------------------------------------
# 5. Aux variable sharing across monomials
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_repeated_bilinear_product_shares_aux_variable():
    """A model in which the same bilinear product appears in two
    distinct higher-degree monomials should produce only one aux
    variable for that product (canonical pair cache)."""
    m = Model()
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    z = m.continuous("z", lb=-1.0, ub=1.0)
    m.minimize(x + y + z)
    # Both cubics share the bilinear product x*y.
    m.subject_to(x * y * z <= 1.0)
    m.subject_to(x * y * x <= 1.0)

    repr_ = model_to_repr(m)
    _, stats = repr_.reformulate_polynomial()
    # 2 cubic monomials. With sharing of x*y: 1 aux for x*y, then
    # 2 more auxes for (x*y)*z and (x*y)*x — total 3, not 4.
    assert stats["aux_variables_introduced"] <= 3
