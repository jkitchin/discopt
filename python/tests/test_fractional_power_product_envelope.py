"""Soundness locks for the fractional-power-of-product objective lift (issue #138).

A fractional power of a polynomial — ``N / g**(1/3)`` and ``(N / g**(1/3))**0.83``
with ``g`` a product/polynomial — is what the relaxation could not linearize at
all (``objective_bound_valid=False``), so such terms were dropped from the
objective and no bound was produced. ``factorable_reform`` now decomposes the
composite chain into elementary aux variables, each with a defining equality the
relaxation can relax:

* ``base ** p`` (fractional ``p``)  ->  aux ``d == t**p`` (``t`` = base lifted to a
  variable), exposing a fractional power of a *single variable*;
* ``N / D`` (non-constant ``D``)   ->  aux ``r`` whose defining equality is cleared
  to the bilinear ``r*D == N``.

Recursion composes these so the objective becomes linear in the aux. These tests
pin the *envelope itself* on small, fully-bounded instances whose true optimum is
known by hand, independent of the slow MINLPLib ``.nl`` solves and of the
end-to-end bound-surfacing layer.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm


def test_ex1233_shaped_ratio_certifies_sound():
    """``300 x / g**(1/3)`` with bounded positive vars certifies its true optimum.

    Optimum 60.0 by hand: minimise the numerator (x=1), maximise the denominator
    (xa=xb=5 -> g = 0.5*(25*5 + 5*25) = 125, g**(1/3) = 5), so 300*1/5 = 60.
    """
    m = dm.Model("ex1233_ratio")
    x = m.continuous("x", lb=1.0, ub=10.0)
    xa = m.continuous("xa", lb=1.0, ub=5.0)
    xb = m.continuous("xb", lb=1.0, ub=5.0)
    m.minimize(300 * x / (0.5 * (xa**2 * xb + xa * xb**2)) ** 0.3333)

    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert abs(r.objective - 60.00966) <= 1e-1, f"obj {r.objective} != 60.01"
    # Soundness: the dual bound never exceeds the objective at the optimum.
    assert r.bound <= r.objective + 1e-4, f"unsound bound {r.bound} > obj {r.objective}"
    assert r.gap_certified, "small fully-bounded instance should certify"


def test_st_e35_shaped_power_of_ratio_certifies_sound():
    """``670 (x / g**(1/3))**0.83`` with bounded positive vars certifies soundly."""
    m = dm.Model("st_e35_power")
    x = m.continuous("x", lb=1.0, ub=10.0)
    xa = m.continuous("xa", lb=1.0, ub=5.0)
    xb = m.continuous("xb", lb=1.0, ub=5.0)
    m.minimize(670 * (x / (0.5 * (xa**2 * xb + xb**2 * xa)) ** 0.333333) ** 0.83)

    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert abs(r.objective - 176.16932) <= 1e-1, f"obj {r.objective} != 176.17"
    assert r.bound <= r.objective + 1e-4, f"unsound bound {r.bound} > obj {r.objective}"
    assert r.gap_certified, "small fully-bounded instance should certify"


def test_lift_is_value_preserving():
    """The reform must reproduce the original objective exactly (sound by being a
    relaxation of an *equivalent* model). Checked numerically against the lifted
    model's defining equalities at random feasible points."""
    import numpy as np
    from discopt._jax import factorable_reform as fr
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        IndexExpression,
        UnaryOp,
        Variable,
    )

    def ev(expr, env):
        if isinstance(expr, Constant):
            return float(np.asarray(expr.value).flatten()[0])
        if isinstance(expr, Variable):
            return env[expr.name]
        if isinstance(expr, IndexExpression):
            return env[expr.base.name]
        if isinstance(expr, BinaryOp):
            a, b = ev(expr.left, env), ev(expr.right, env)
            if expr.op == "**":
                return float(a) ** float(b)
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[expr.op]
        if isinstance(expr, UnaryOp):
            a = ev(expr.operand, env)
            return -a if expr.op == "neg" else abs(a)
        raise ValueError(type(expr))

    m0 = dm.Model("vp")
    x = m0.continuous("x", lb=1.0, ub=4.0)
    xa = m0.continuous("xa", lb=1.0, ub=4.0)
    xb = m0.continuous("xb", lb=1.0, ub=4.0)
    m0.minimize(670 * (x / (0.5 * (xa**2 * xb + xb**2 * xa)) ** 0.333333) ** 0.83)
    m1 = fr.factorable_reformulate(m0)

    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(20):
        env = {v.name: rng.uniform(1.0, 4.0) for v in m0._variables}

        def collect(e, auxes):
            if isinstance(e, Variable) and e.name.startswith("_fr_aux") and e.name not in env:
                auxes.add(e.name)
            if isinstance(e, BinaryOp):
                collect(e.left, auxes)
                collect(e.right, auxes)
            if isinstance(e, UnaryOp):
                collect(e.operand, auxes)

        # Solve each aux from its (cleared, aux-linear) defining constraint. The
        # constraint list is not guaranteed to be topologically ordered, so an
        # aux may depend on one defined by a later constraint; iterate to a
        # fixpoint (each pass resolves every constraint with exactly one still-
        # unknown aux) rather than assuming a single forward pass suffices.
        progress = True
        while progress:
            progress = False
            for c in m1._constraints:
                auxes: set[str] = set()
                collect(c.body, auxes)
                if len(auxes) == 1:
                    an = auxes.pop()
                    env[an] = 0.0
                    b0 = ev(c.body, env)
                    env[an] = 1.0
                    b1 = ev(c.body, env)
                    env[an] = -b0 / (b1 - b0)
                    progress = True
        if any(v.name.startswith("_fr_aux") and v.name not in env for v in m1._variables):
            continue
        o0 = ev(m0._objective.expression, env)
        o1 = ev(m1._objective.expression, env)
        max_rel = max(max_rel, abs(o0 - o1) / max(1.0, abs(o0)))
    assert max_rel <= 1e-9, f"reform not value-preserving: max rel diff {max_rel:.2e}"
