"""Nested powers inside a product keep their sign in the canonical DAG.

``_Canonicalizer._prod`` flattened ``(b^q)^e`` to ``b^(q*e)`` and
``(prod c_i^q_i)^e`` to ``prod c_i^(q_i*e)`` with no sign check. Both are
identities only for nonnegative bases: ``x*(x**2)**0.5`` became ``x**2`` (true
value ``-x**2`` on ``x < 0``), ``1/(x**2)**0.5`` became ``1/x``, and
``1/(x*y)**0.5`` became ``x**-0.5 * y**-0.5`` (undefined at ``x = y = -1``). The
uniform relaxation was built on the wrong function, and ``Model.solve`` certified
a wrong optimum. Found by the random expression-tree soundness fuzzer
(``scripts/fuzz/fuzz_relax_trees.py``).

The fold now uses ``|c_i|`` for a factor whose inner exponent is an integer when
the outer exponent is fractional (``(prod c_i^q_i)^e = prod |c_i|^(q_i*e)``,
exact wherever the expression is real).
"""

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.canonical_expr import canonicalize
from discopt._relax.uniform_relax import build_uniform_relaxation


def _relax_range(build, lo, hi):
    """(lower, upper) bound the uniform relaxation proves for ``build(x)``."""
    out = []
    for sense in ("min", "max"):
        m = dm.Model()
        x = m.continuous("x", lb=lo, ub=hi)
        (m.minimize if sense == "min" else m.maximize)(build(x))
        rel = build_uniform_relaxation(m)
        res = rel.model.solve(backend="simplex")
        assert res.status == "optimal"
        b = float(res.bound)
        out.append(-b if sense == "max" and rel.obj_sense_sign < 0 else b)
    return out


CASES = [
    # (label, builder, numpy evaluator)
    ("x*(x**2)**0.5", lambda x: x * (x**2) ** 0.5, lambda t: t * np.abs(t)),
    ("1/(x**2)**0.5", lambda x: 1 / (x**2) ** 0.5, lambda t: 1 / np.abs(t)),
    ("2/(x**2)**0.5", lambda x: 2 / (x**2) ** 0.5, lambda t: 2 / np.abs(t)),
    ("1/(x**4)**0.25", lambda x: 1 / (x**4) ** 0.25, lambda t: 1 / np.abs(t)),
    ("x*(x**2)**1.5", lambda x: x * (x**2) ** 1.5, lambda t: t * np.abs(t) ** 3),
    ("x*(x**4)**0.5", lambda x: x * (x**4) ** 0.5, lambda t: t**3),  # q*e = 2: no abs
    ("x*(x**3)**2", lambda x: x * (x**3) ** 2, lambda t: t**7),  # integer e: no abs
]


@pytest.mark.unit
@pytest.mark.parametrize("label, build, f", CASES, ids=[c[0] for c in CASES])
def test_relaxation_encloses_nested_power_on_negative_box(label, build, f):
    lo, hi = -3.0, -1.0
    lower, upper = _relax_range(build, lo, hi)
    ts = np.linspace(lo, hi, 2001)
    vals = f(ts)
    checks = 0
    for v in vals:
        tol = 1e-9 * max(1.0, abs(v))
        assert lower <= v + tol, (label, lower, v)
        assert upper >= v - tol, (label, upper, v)
        checks += 2
    assert checks == 2 * len(ts)


@pytest.mark.unit
def test_canonical_form_keeps_abs():
    def canon(build, **boxes):
        m = dm.Model()
        v = {k: m.continuous(k, lb=a, ub=b) for k, (a, b) in boxes.items()}
        m.minimize(build(**v))
        return canonicalize(m).objective

    node = canon(lambda x: x * (x**2) ** 0.5, x=(-3, -1))
    assert node.kind == "prod"
    assert ("call", "abs", ("var", 0)) in [ch.key for ch in node.children]

    node = canon(lambda x, y: 1 / (x * y) ** 0.5, x=(-3, -1), y=(-3, -1))
    assert sorted(ch.key for ch in node.children) == [
        ("call", "abs", ("var", 0)),
        ("call", "abs", ("var", 1)),
    ]

    # Integer outer exponent or an even merged exponent: plain fold, no abs.
    assert canon(lambda x: 1 / (x**2), x=(-3, -1)).key == ("pow", ("var", 0), -2.0)
    assert canon(lambda x: x * (x**4) ** 0.5, x=(-3, -1)).key == ("pow", ("var", 0), 3.0)


@pytest.mark.unit
def test_solve_certifies_true_optimum():
    """End to end: before the fix this returned ``optimal`` 0.7 with
    ``gap_certified=True``; ``z = 0, x = -3`` is feasible (``x*|x| = -9 <= -4``)
    with objective -0.3."""
    m = dm.Model()
    x = m.continuous("x", lb=-3, ub=-1)
    z = m.binary("z")
    m.subject_to(x * (x**2) ** 0.5 <= -4 + 20 * z)
    m.minimize(z + 0.1 * x)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert math.isclose(r.objective, -0.3, abs_tol=1e-5)
    assert r.bound <= -0.3 + 1e-6
