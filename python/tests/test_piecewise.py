"""Tests for the declared piecewise-linear construct ``Model.piecewise`` (#1482).

What is asserted, and why it is enough to call the construct *exact*:

* **Graph exactness** -- for every method, with ``x`` fixed at a point ``t``
  (breakpoints, midpoints, random interior points), both ``min y`` and ``max y``
  solve to ``f(t) = np.interp(t, b, v)``. If the rows admitted any ``y`` off the
  graph at some ``t``, one of the two would move off ``f(t)``.
* **Certified optima** -- models built on ``y`` solve to ``status="optimal"``
  with ``gap_certified=True`` and objective equal to an independent brute-force
  optimum (a PWL objective over a box attains its minimum at a breakpoint or a
  bound; for a separable pair coupled by one equality, at a point where one input
  sits on a breakpoint).
* **Domain agreement** -- an input whose declared domain escapes the breakpoint
  span is refused at declaration, and again at ``validate`` if widened later,
  including after a save/load round trip.
* **Log encoding soundness** -- for every binary pattern, the Gray-code rows the
  ``log`` method emits leave weight on at most one adjacent breakpoint pair, and
  every segment is reachable (checked exhaustively for 2..65 breakpoints).

Each test module-level probe counts its executed assertions where a loop could
otherwise degrade to a no-op (CLAUDE.md §6).
"""

from __future__ import annotations

import itertools
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling._piecewise import (
    PIECEWISE_METHODS,
    PiecewiseDomainError,
    PiecewiseLinear,
    normalize_piecewise_method,
)
from discopt.modeling.core import VarType, _SOSConstraint

B = [0.0, 1.0, 2.5, 3.0, 4.0, 6.0]
V = [1.0, 3.0, -0.5, 2.0, 2.0, -1.0]  # nonconvex, with a flat segment


def _f(t):
    return float(np.interp(t, B, V))


def _binaries(m):
    return sum(v.size for v in m._variables if v.var_type == VarType.BINARY)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_api_is_exposed():
    assert callable(dm.Model.piecewise)
    assert callable(dm.piecewise)
    assert dm.PiecewiseDomainError is PiecewiseDomainError
    assert issubclass(PiecewiseDomainError, ValueError)
    assert set(PIECEWISE_METHODS) == {"sos2", "log", "disaggregated", "incremental"}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("sos2", "sos2"),
        ("lambda", "sos2"),
        ("LOG", "log"),
        ("logarithmic", "log"),
        ("ebd", "log"),
        ("dcc", "disaggregated"),
        ("disaggregated", "disaggregated"),
        ("inc", "incremental"),
        ("delta", "incremental"),
    ],
)
def test_method_aliases(alias, canonical):
    assert normalize_piecewise_method(alias) == canonical


@pytest.mark.smoke
def test_unknown_method_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    with pytest.raises(ValueError, match="Unknown piecewise method"):
        m.piecewise(x, B, V, method="bigm")


@pytest.mark.smoke
def test_output_variable_and_bounds():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, name="f")
    assert isinstance(y, dm.Variable)
    assert y.name == "f"
    assert y.var_type == VarType.CONTINUOUS
    assert float(y.lb) == min(V) and float(y.ub) == max(V)


@pytest.mark.smoke
def test_free_function_finds_the_model():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = dm.piecewise(x, B, V, name="g")
    assert y.model is m and y.name == "g"
    with pytest.raises(TypeError):
        dm.piecewise(3.0, B, V)


@pytest.mark.smoke
@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_binary_counts_per_method(method):
    n = len(B)
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    m.piecewise(x, B, V, method=method)
    expected = {
        "disaggregated": n - 1,
        "incremental": n - 2,
        "log": math.ceil(math.log2(n - 1)),
        "sos2": 0,  # the SOS2 row carries the discrete choice until lowering
    }[method]
    assert _binaries(m) == expected
    sos = [c for c in m._constraints if isinstance(c, _SOSConstraint)]
    assert len(sos) == (1 if method == "sos2" else 0)
    if sos:
        assert sos[0].sos_type == 2 and len(sos[0].variables) == n


@pytest.mark.smoke
@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_two_breakpoints_add_no_binary(method):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.piecewise(x, [0, 1], [2, 5], method=method)
    assert _binaries(m) == 0
    assert not any(isinstance(c, _SOSConstraint) for c in m._constraints)


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bps,vals,match",
    [
        ([0.0], [1.0], "at least 2"),
        ([0, 1, 1, 2], [0, 1, 2, 3], "strictly increasing"),
        ([0, 2, 1], [0, 1, 2], "strictly increasing"),
        ([0, 1, 2], [0, 1], "one value per breakpoint"),
        ([0, np.nan, 2], [0, 1, 2], "finite"),
        ([0, 1, 2], [0, np.inf, 2], "finite"),
        ([[0, 1], [2, 3]], [[0, 1], [2, 3]], "one-dimensional"),
    ],
)
def test_malformed_table_refused(bps, vals, match):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    with pytest.raises(ValueError, match=match):
        m.piecewise(x, bps, vals)
    assert len(m._variables) == 1  # nothing added


@pytest.mark.smoke
def test_callable_values_are_sampled_at_breakpoints():
    t = PiecewiseLinear.from_table([1.0, 2.0, 4.0], np.sqrt)
    assert t.values == (1.0, math.sqrt(2.0), 2.0)
    assert t(3.0) == pytest.approx((math.sqrt(2.0) + 2.0) / 2)
    with pytest.raises(ValueError, match="outside its domain"):
        t(4.5)


# ---------------------------------------------------------------------------
# Domain agreement
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "lb,ub", [(-0.1, 6.0), (0.0, 6.5), (-1.0, 7.0), (None, 6.0), (0.0, None), (None, None)]
)
def test_domain_escaping_the_span_is_refused(lb, ub):
    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    n_vars, n_cons = len(m._variables), len(m._constraints)
    with pytest.raises(PiecewiseDomainError, match="not inside the breakpoint span"):
        m.piecewise(x, B, V)
    # A refused call leaves the model untouched.
    assert (len(m._variables), len(m._constraints)) == (n_vars, n_cons)
    assert m._piecewise_domains == []


@pytest.mark.smoke
def test_domain_inside_the_span_is_accepted():
    m = dm.Model()
    x = m.continuous("x", lb=1.5, ub=3.5)  # strictly inside [0, 6]
    m.piecewise(x, B, V)
    xf = m.continuous("xf", lb=6.0, ub=6.0)  # fixed on the last breakpoint
    m.piecewise(xf, B, V)


@pytest.mark.smoke
def test_array_input_reports_each_offending_element():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=[0.0, -1.0, 0.0], ub=[6.0, 6.0, 9.0])
    with pytest.raises(PiecewiseDomainError) as exc:
        m.piecewise(x, B, V)
    msg = str(exc.value)
    assert "element (1,)" in msg and "element (2,)" in msg and "element (0,)" not in msg


@pytest.mark.smoke
def test_composite_input_uses_a_rigorous_enclosure():
    m = dm.Model()
    s = m.continuous("s", lb=0.0, ub=1.0)
    w = m.continuous("w", lb=0.0, ub=2.0)
    # 2*s encloses as [0, 2] up to outward rounding: agrees with a [0, 2] span.
    m.piecewise(2 * s, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    # s + w ranges over [0, 3], which escapes [0, 2].
    with pytest.raises(PiecewiseDomainError, match="interval enclosure"):
        m.piecewise(s + w, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])


@pytest.mark.smoke
def test_input_must_belong_to_this_model():
    m1, m2 = dm.Model(), dm.Model()
    x = m1.continuous("x", lb=0, ub=6)
    with pytest.raises(ValueError, match="different Model"):
        m2.piecewise(x, B, V)


@pytest.mark.smoke
def test_widening_after_declaration_is_caught_by_validate():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, name="f")
    m.minimize(y)
    m.validate()
    x.ub = np.float64(8.0)
    with pytest.raises(PiecewiseDomainError, match="widened after declaration"):
        m.validate()
    with pytest.raises(PiecewiseDomainError):
        m.solve()
    x.ub = np.float64(6.0)
    m.validate()


@pytest.mark.smoke
def test_domain_guard_survives_serialization():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, name="f")
    m.minimize(y)
    m2 = dm.loads(dm.dumps(m))
    assert len(m2._piecewise_domains) == 1
    _, lo, hi, label = m2._piecewise_domains[0]
    assert (lo, hi, label) == (0.0, 6.0, "f")
    m2.validate()
    x2 = next(v for v in m2._variables if v.name == "x")
    x2.lb = np.float64(-1.0)
    with pytest.raises(PiecewiseDomainError):
        m2.validate()


# ---------------------------------------------------------------------------
# Log-encoding soundness (pure combinatorics on the emitted index sets)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_log_encoding_admits_exactly_the_adjacent_pairs():
    from discopt._relax.embedding import build_embedding_map

    patterns = 0
    for n in range(3, 66):
        emb = build_embedding_map(n, encoding="gray")
        reachable = set()
        for u in itertools.product((0, 1), repeat=emb.bit_count):
            allowed = set(range(n))
            for bit, ub in enumerate(u):
                allowed -= set(emb.negative_sets[bit] if ub else emb.positive_sets[bit])
            patterns += 1
            # Weight may sit on at most one adjacent pair of breakpoints.
            assert not allowed or max(allowed) - min(allowed) <= 1, (n, u, allowed)
            if len(allowed) == 2:
                reachable.add(min(allowed))
        # ...and every segment is selectable by some pattern.
        assert reachable == set(range(n - 1)), n
    assert patterns > 1000  # the probe ran


# ---------------------------------------------------------------------------
# Exactness: y is pinned to f(x) at and between breakpoints
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(1482)
_PROBES = sorted(
    set(B) | {(a + b) / 2 for a, b in zip(B, B[1:])} | set(np.round(_rng.uniform(0, 6, 4), 6))
)


def _y_range_at(method, t):
    out = []
    for sense in ("min", "max"):
        m = dm.Model()
        x = m.continuous("x", lb=t, ub=t)
        y = m.piecewise(x, B, V, method=method)
        (m.minimize if sense == "min" else m.maximize)(y)
        r = m.solve(time_limit=30)
        assert r.status == "optimal", (method, t, sense, r.status)
        out.append(float(r.value(y)))
    return out


@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_graph_is_exact_at_and_between_breakpoints(method):
    checked = 0
    for t in _PROBES:
        lo, hi = _y_range_at(method, float(t))
        assert lo == pytest.approx(_f(t), abs=1e-6), (method, t)
        assert hi == pytest.approx(_f(t), abs=1e-6), (method, t)
        checked += 1
    assert checked == len(_PROBES) >= 14


@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_array_input_is_elementwise(method):
    ts = np.array([0.0, 1.75, 5.2])
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=ts, ub=ts)
    y = m.piecewise(x, B, V, method=method, name="y")
    assert y.shape == (3,)
    m.maximize(dm.sum(y))
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.value(y), np.interp(ts, B, V), atol=1e-6)


# ---------------------------------------------------------------------------
# Certified optima against independent brute force
# ---------------------------------------------------------------------------


def _brute_1d(b, v, c, lo, hi):
    """min over [lo, hi] of c*x + f(x): attained at a breakpoint or a bound."""
    cands = [t for t in b if lo <= t <= hi] + [lo, hi]
    return min(c * t + float(np.interp(t, b, v)) for t in cands)


@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_random_1d_models_certify_the_brute_force_optimum(method):
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(8):
        n = int(rng.integers(3, 12))
        b = np.sort(rng.choice(np.arange(-20, 21), size=n, replace=False)).astype(float)
        v = rng.uniform(-10, 10, n).round(3)
        c = round(float(rng.uniform(-1, 1)), 3)
        lo = round(float(rng.uniform(b[0], b[n // 2])), 3)
        hi = round(float(rng.uniform(b[n // 2], b[-1])), 3)
        m = dm.Model()
        x = m.continuous("x", lb=lo, ub=hi)
        y = m.piecewise(x, b, v, method=method)
        m.minimize(c * x + y)
        r = m.solve(time_limit=30)
        assert r.status == "optimal" and r.gap_certified
        want = _brute_1d(b, v, c, lo, hi)
        assert float(r.objective) == pytest.approx(want, abs=1e-6)
        xs = float(r.value(x))
        assert float(r.value(y)) == pytest.approx(float(np.interp(xs, b, v)), abs=1e-6)
        checked += 1
    assert checked == 8


def test_separable_pair_with_coupling_certifies():
    """min f1(x1) + f2(x2) s.t. x1 + x2 == D; methods mixed across the two inputs."""
    b1, v1 = [0, 2, 5, 8, 10], [0, 6, 7, 15, 16]  # concave-ish cost
    b2, v2 = [0, 3, 4, 10], [0, 2, 9, 10]
    D = 11.0
    # Brute force: at an optimum one input sits on a breakpoint.
    cands = [t for t in b1 if 0 <= D - t <= 10] + [D - t for t in b2 if 0 <= D - t <= 10]
    want = min(np.interp(t, b1, v1) + np.interp(D - t, b2, v2) for t in cands)
    for m1, m2 in itertools.product(PIECEWISE_METHODS, repeat=2):
        m = dm.Model()
        x1 = m.continuous("x1", lb=0, ub=10)
        x2 = m.continuous("x2", lb=0, ub=10)
        f1 = m.piecewise(x1, b1, v1, method=m1)
        f2 = m.piecewise(x2, b2, v2, method=m2)
        m.subject_to(x1 + x2 == D)
        m.minimize(f1 + f2)
        r = m.solve(time_limit=30)
        assert r.status == "optimal" and r.gap_certified, (m1, m2)
        assert float(r.objective) == pytest.approx(float(want), abs=1e-6), (m1, m2)


@pytest.mark.parametrize("method", PIECEWISE_METHODS)
def test_pwl_inside_a_nonlinear_minlp_certifies(method):
    """y feeds a nonconvex quadratic; the spatial B&B path must certify it too."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, method=method)
    m.minimize((y - 1.2) ** 2 + 0.05 * (x - 4.5) ** 2)
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    # Brute force on a fine grid plus every breakpoint.
    grid = np.union1d(np.linspace(0, 6, 60001), B)
    want = np.min((np.interp(grid, B, V) - 1.2) ** 2 + 0.05 * (grid - 4.5) ** 2)
    assert float(r.objective) == pytest.approx(float(want), abs=1e-5)
    assert float(r.value(y)) == pytest.approx(_f(float(r.value(x))), abs=1e-6)


def test_composite_and_integer_inputs():
    m = dm.Model()
    a = m.integer("a", lb=0, ub=3)
    s = m.continuous("s", lb=0.0, ub=3.0)
    y1 = m.piecewise(a, B, V, name="fa")
    y2 = m.piecewise(a + s, B, V, name="fas")
    m.maximize(y1 + y2)
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    av, sv = float(r.value(a)), float(r.value(s))
    assert av == pytest.approx(round(av))
    assert float(r.value(y1)) == pytest.approx(_f(av), abs=1e-6)
    assert float(r.value(y2)) == pytest.approx(_f(av + sv), abs=1e-6)
    # Brute force: a in {0..3}, a + s at a breakpoint or a bound of [a, a + 3].
    want = max(
        _f(ai) + max(_f(t) for t in [ai, ai + 3] + [bk for bk in B if ai <= bk <= ai + 3])
        for ai in range(4)
    )
    assert float(r.objective) == pytest.approx(want, abs=1e-6)


def test_gallery_pump_example_certifies_the_brute_force_optimum():
    """``examples.example_piecewise_pumps`` solves to its exact optimum.

    Brute force: for fixed on/off the problem is a separable PWL program with one
    equality, so an optimum has at most one flow strictly inside a segment. The
    tabulated flows are multiples of 10 and the demand is 95, so every such
    vertex lies on the 5 m3/h grid -- enumerating that grid is exact.
    """
    import contextlib
    import io

    from discopt.modeling import examples

    with contextlib.redirect_stdout(io.StringIO()):
        m = examples.example_piecewise_pumps()
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified

    flows = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    tables = [
        [0.0, 6.0, 9.0, 11.5, 15.5, 22.0],
        [0.0, 5.0, 9.5, 12.0, 14.5, 19.0],
        [0.0, 7.0, 8.5, 10.0, 14.0, 21.0],
    ]
    grid = [0.0] + [float(g) for g in range(10, 51, 5)]
    best = math.inf
    checked = 0
    for q0 in grid:
        for q1 in grid:
            q2 = 95.0 - q0 - q1
            if not (q2 == 0.0 or 10.0 <= q2 <= 50.0):
                continue
            qs = (q0, q1, q2)
            cost = sum(np.interp(qk, flows, tables[k]) + 2.0 * (qk > 0) for k, qk in enumerate(qs))
            best = min(best, float(cost))
            checked += 1
    assert checked > 20
    assert float(r.objective) == pytest.approx(best, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.parametrize("method", ["incremental", "log", "disaggregated"])
def test_algebraic_methods_export_like_any_milp(method):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, method=method, name="f")
    m.minimize(y)
    lp = m.to_lp()
    assert "Binaries" in lp and "_pwl1_f_" in lp
    assert "SOS" not in lp


@pytest.mark.smoke
def test_sos2_method_is_refused_by_the_writers_not_dropped():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    y = m.piecewise(x, B, V, method="sos2")
    m.minimize(y)
    with pytest.raises(ValueError, match="SOS"):
        m.to_lp()


@pytest.mark.smoke
def test_default_method_is_incremental():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=6)
    m.piecewise(x, B, V)
    assert any(v.name.endswith("_delta") for v in m._variables)
