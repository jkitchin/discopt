"""Regression test: denominator clearing must not shrink violations below tol.

Sign-definite denominator clearing (issue #130) multiplies a constraint
``N/D sense 0`` through by ``D`` to expose a polynomial the relaxation can
bound.  Multiplying rescales the constraint by ``D(x)``, so when ``|D|`` can be
small over the box, a *gross* violation of the original constraint shrinks
proportionally in the cleared form.

Concretely, clearing ``6 - (x0 - 0.2458 x0**2 / x1) <= 0`` by ``x1 in
[1e-5, 30]`` turns an original violation of ``6.0`` into ``6.0 * x1 ~ 6.3e-5`` —
which slips *under* the absolute incumbent-feasibility tolerance (1e-4).  The
spatial-B&B then accepts a wildly infeasible near-origin point as a feasible
incumbent and certifies it: a false-optimal.

    before:  optimal 0.000311   (point x0~4e-6, x1~1e-5 violates the original
                                 constraint by ~6.0 — infeasible)
    after:   optimal 376.29     (the true optimum, valid dual bound)

The fix (``factorable_reform._clear_divisions``): divide the cleared body by
``dmin = min |D|`` so its magnitude is never smaller than the original, keeping
the fixed absolute feasibility tolerance sound.

st_e17 itself is classified convex (so production gates clearing off and never
hit this), but the bug is *not* about convexity — any nonconvex model with a
clearable small denominator triggers it on the production solve path.  The first
test reproduces exactly that.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest

# Brute-forced true optimum of the st_e17 ratio constraint (verified
# independently): min 29.4 x0 + 18 x1 s.t. x0 - 0.2458 x0**2/x1 >= 6.
_OPT = 376.291932
_OPT_X0 = 8.17003
_OPT_X1 = 7.56073


def _st_e17_core():
    """The st_e17 feasible region: a single ratio constraint, linear objective."""
    m = dm.Model("st_e17_core")
    x0 = m.continuous("x0", lb=0.0, ub=115.8)
    x1 = m.continuous("x1", lb=1e-5, ub=30.0)
    m.minimize(29.4 * x0 + 18 * x1)
    m.subject_to(6 - (x0 - 0.2458 * x0**2 / x1) <= 0)
    return m


@pytest.mark.correctness
def test_nonconvex_small_denominator_no_false_optimal():
    """A nonconvex model with a clearable small denominator must not false-optimize.

    Adds a bilinear constraint so the model is provably nonconvex — that routes
    it through the production denominator-clearing path (gated to nonconvex
    models), which is where the pre-fix false-optimal fired.
    """
    m = _st_e17_core()
    # Bilinear term -> nonconvex -> production applies denominator clearing.
    m.subject_to(m._variables[0] * m._variables[1] <= 2000)

    r = m.solve(time_limit=60, gap_tolerance=1e-4)

    # The headline soundness invariant: no dual bound below the true optimum may
    # be paired with a near-zero incumbent.  Pre-fix this certified obj ~ 3e-4.
    assert r.objective is not None
    assert r.objective >= _OPT - 1.0, (
        f"false-optimal: certified obj {r.objective} << true optimum {_OPT}"
    )
    # Valid dual bound never exceeds the incumbent.
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-3, f"invalid dual bound {r.bound} > obj {r.objective}"


@pytest.mark.correctness
def test_cleared_constraint_certifies_true_optimum():
    """Forcing the reformulation certifies the true optimum with a valid bound."""
    from discopt._jax.factorable_reform import factorable_reformulate

    m = factorable_reformulate(_st_e17_core())
    r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.status == "optimal", f"status={r.status}"
    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-1, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: dual bound never exceeds the optimum.
    assert r.bound is not None, "certified solve must report a dual bound"
    assert r.bound <= r.objective + 1e-2, f"invalid dual bound {r.bound} > obj {r.objective}"


def test_returned_point_satisfies_original_constraint():
    """The certified incumbent must satisfy the *original* ratio constraint.

    Directly guards the failure mode: the pre-fix solver returned a point that
    satisfied the *cleared* constraint within tolerance but violated the
    original division constraint by ~6.0.
    """
    from discopt._jax.factorable_reform import factorable_reformulate

    r = factorable_reformulate(_st_e17_core()).solve(time_limit=60, gap_tolerance=1e-4)
    assert r.x is not None
    x0 = float(r.x["x0"])
    x1 = float(r.x["x1"])
    # Original constraint: 6 - (x0 - 0.2458 x0**2/x1) <= 0.
    orig = 6 - (x0 - 0.2458 * x0**2 / x1)
    assert orig <= 1e-4, f"certified point violates original constraint: body={orig}"


def test_large_denominator_clearing_unchanged():
    """A denominator with ``min|D| >= 1`` is cleared without rescaling.

    The scale guard must only engage for sub-unit denominators so models that
    already certify (gear family, nvs: ``D`` a product of vars with ``lb >= 1``)
    are untouched.
    """
    from discopt._jax.factorable_reform import _clear_divisions

    m = dm.Model("big_denom")
    x0 = m.continuous("x0", lb=2.0, ub=10.0)
    x1 = m.continuous("x1", lb=3.0, ub=10.0)
    # denom = x0*x1 in [6, 100] -> dmin = 6 >= 1 -> no rescale.
    con_body = (5 / (x0 * x1)) - 1
    new_body, sense = _clear_divisions(con_body, "<=", m)
    # No leading constant-scale wrapper: the top node is the multiply-through,
    # not a ``Constant(1/dmin) * (...)``.
    from discopt.modeling.core import BinaryOp, Constant

    is_scaled = (
        isinstance(new_body, BinaryOp)
        and new_body.op == "*"
        and isinstance(new_body.left, Constant)
        and float(new_body.left.value) != 1.0
    )
    assert not is_scaled, "large denominator (dmin>=1) must not be rescaled"
