"""#1678 (b) / II.20 -- no certificate may rest on the default ``±9.999e19`` box.

A continuous variable declared without a bound gets ``DEFAULT_VARIABLE_BOUND`` on
that side. #850 had read that number as a declared finite bound, so the spatial tree
certified ``min x - z**2`` s.t. ``z*x <= 3`` (``x`` free, ``z in [-2, 2]``) as
``optimal`` with objective = bound = ``-9.999e19`` and ``gap_certified=True``. That
problem is unbounded (``z = 0``, ``x -> -inf``): a false certificate.

The contract now:

* the default box means "no bound". An LP/MILP that runs off along it is proved
  ``unbounded``. Any other result whose point sits on it is ``feasible`` with no
  dual bound, never ``optimal``, and a warning names the variable.
* a bound the user *wrote*, however large (``1e17``, ``9.9e19``), is honoured as
  posed, so the corner certificate of #850 still stands for it.
* models whose answer does not touch the default box are unaffected.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest

_MATCH = "sits on the default variable bound"


def _solve(m, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = m.solve(time_limit=20, **kw)
    return r, [str(w.message) for w in caught]


def _nlp_1678():
    m = dm.Model("nlp_1678")
    x = m.continuous("x")
    z = m.continuous("z", lb=-2, ub=2)
    m.subject_to(z * x <= 3)
    m.minimize(x - z**2)
    return m


def _minlp():
    m = dm.Model("minlp_free")
    x = m.continuous("x")
    k = m.integer("k", lb=0, ub=3)
    m.subject_to(k * x <= 3)
    m.minimize(x - k**2)
    return m


def _nonconvex_qp():
    m = dm.Model("qp_free")
    x = m.continuous("x")
    z = m.continuous("z", lb=-2, ub=2)
    m.minimize(x - z**2)
    return m


def _max_nlp():
    m = dm.Model("max_free")
    x = m.continuous("x")
    z = m.continuous("z", lb=-2, ub=2)
    m.subject_to(z * x >= -3)
    m.maximize(x + z**2)
    return m


@pytest.mark.parametrize("build", [_nlp_1678, _minlp, _nonconvex_qp, _max_nlp])
def test_no_certificate_on_the_default_box(build):
    r, msgs = _solve(build())
    assert r.status != "optimal", (r.status, r.objective, r.bound)
    assert not r.gap_certified
    assert r.bound is None and r.gap is None
    # The point is still published (it is feasible); only the claim is withdrawn.
    assert r.status == "feasible" and r.x is not None
    assert r.solver_stats.get("certificate/default_box_withheld") == 1.0
    assert any(_MATCH in m for m in msgs), msgs


def test_an_explicit_huge_bound_is_honoured():
    """The #850 reading survives for a bound the user wrote."""
    m = dm.Model("nlp_explicit")
    x = m.continuous("x", lb=-1e17)
    z = m.continuous("z", lb=-2, ub=2)
    m.subject_to(z * x <= 3)
    m.minimize(x - z**2)
    r, msgs = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-1e17, rel=1e-9)
    assert not any(_MATCH in m for m in msgs)


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_lp_and_milp_on_the_default_box_are_unbounded(monkeypatch, backend):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    m = dm.Model("lp")
    x = m.continuous("x", lb=0)
    m.minimize(-x)
    r, _ = _solve(m)
    assert r.status == "unbounded", (r.status, r.objective)

    m = dm.Model("milp")
    x = m.continuous("x", lb=0)
    b = m.binary("b")
    m.subject_to(x >= b)
    m.minimize(-x + b)
    r, _ = _solve(m)
    if backend == "highs":
        assert r.status == "unbounded", (r.status, r.objective)
    else:
        # The Rust MILP route has no ray proof; it must at least not certify.
        assert r.status != "optimal" and not r.gap_certified, (r.status, r.objective)


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_explicit_huge_lp_bound_is_optimal_at_the_corner(monkeypatch, backend):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    m = dm.Model("lp_explicit")
    x = m.continuous("x", lb=0, ub=1e18)
    m.minimize(-x)
    r, _ = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-1e18, rel=1e-9)


def _bounded_free_nlp():
    m = dm.Model("bounded_free")
    x = m.continuous("x")
    y = m.continuous("y")
    m.subject_to(x * y >= 1)
    m.subject_to(x >= 0.5)
    m.minimize(x**2 + y**2)
    return m, 2.0


def _convex_qp_free():
    m = dm.Model("convex_free")
    x = m.continuous("x")
    m.minimize((x - 3) ** 2)
    return m, 0.0


def _degenerate_lp_free():
    m = dm.Model("degenerate_free")
    m.continuous("x", lb=0)  # zero objective coefficient, nothing pushes it
    y = m.continuous("y", lb=1)
    m.minimize(y)
    return m, 1.0


@pytest.mark.parametrize("build", [_bounded_free_nlp, _convex_qp_free, _degenerate_lp_free])
def test_free_variables_away_from_the_box_keep_their_certificate(build):
    m, opt = build()
    r, msgs = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(opt, abs=1e-5)
    assert not any(_MATCH in m for m in msgs)
    assert all(np.all(np.abs(np.asarray(v)) < 1e19) for v in r.x.values())
