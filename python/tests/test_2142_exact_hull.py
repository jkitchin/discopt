"""#2142: ``DISCOPT_HULL_EXACT_PERSPECTIVE`` -- the division-free (homogenized) hull of a
nonconvex polynomial disjunct row.

For ``g(x) = sum_m c_m prod_{i in m} x_i <sense> rhs`` of degree ``d`` the hull emits
``sum_m c_m prod v_i * y^(d - |m|) <sense> rhs * y^d`` instead of the eps-perspective
``yhat g(v / yhat) - eps g(0) (1 - y)`` (``docs/dev/data/2142-exact-hull/proposal.md``).
Bound-changing on the ``"hull"`` route, so it is gated (CLAUDE.md §5). Pinned here:

* flag OFF: the lowering is byte-identical to the code before the flag existed (the
  fingerprints below were taken on the parent commit), and ON leaves a lowering with
  no nonconvex polynomial row byte-identical too;
* ``_polynomial_monomials``: expansion, cancellation, ``**``, and every refusal;
* flag ON: every nonconvex polynomial row is division-free, a convex row keeps the
  eps-perspective, a non-polynomial row keeps it too;
* exactness at fixed selectors: at ``y in {0, 1}`` with the copies at the disjunct's
  point the rows evaluate to the original row (``y = 1``) or to exactly ``0.0``
  (``y = 0``, the #1043 class);
* the differential battery (no feasible point cut, no infeasible point admitted) on
  the new shapes, and certified optima equal to big-M's;
* GDPlib cstr (slow, one shared deterministic solve): the kernel spec's largest
  bound <= 1e4 (it is 1e27 under the eps-perspective), and the 2,000-node bound is
  the prototype's (``exact_hull_proto.py`` arm ``homog``), bit for bit.
"""

from __future__ import annotations

import hashlib
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._point_eval import compile_point_fn
from discopt._relax import gdp_reformulate as G
from discopt._relax.gdp_reformulate import _is_linear, _polynomial_monomials, reformulate_gdp
from discopt.transformations import model_fingerprint
from gdp_differential import Shape, run_shape

FLAG = "DISCOPT_HULL_EXACT_PERSPECTIVE"
N_EXEC = {"n": 0}


def _check(cond, msg=""):
    N_EXEC["n"] += 1
    assert cond, msg


# --------------------------------------------------------------------------- shapes


def _bilinear_le():
    """A nonconvex bilinear inequality ``x z <= 1`` beside a linear row."""
    m = dm.Model("bilinear_le")
    x = m.continuous("x", lb=0, ub=4)
    z = m.continuous("z", lb=0, ub=4)
    m.either_or([[x * z <= 1, x + z >= 2.5], [x <= 0.5, z >= 3]], name="d")
    m.minimize(x - z)
    return m


def _cubic_eq():
    """A cubic equality written with ``**``: ``x**2 z - y == 0`` (degree 3)."""
    m = dm.Model("cubic_eq")
    x = m.continuous("x", lb=0, ub=2)
    z = m.continuous("z", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=8)
    m.either_or([[x**2 * z - y == 0], [x + z <= 0.5, y <= 0.5]], name="d")
    m.minimize(-x - z + 0.2 * y)
    return m


def _shared_product():
    """Two rows of one disjunct share ``x z``; one has a constant term (``g(0) != 0``)."""
    m = dm.Model("shared_product")
    x = m.continuous("x", lb=-2, ub=3)
    z = m.continuous("z", lb=-2, ub=3)
    y = m.continuous("y", lb=-1, ub=4)
    m.either_or(
        [[x * z <= 2, x * z + y - 1.5 >= 0], [x >= 2.5, y <= 0]],
        name="d",
    )
    m.minimize(x + z + y)
    return m


def _convex_quartic():
    """``x**4 + z**2 <= 3`` is provably convex and outside the cone class: it keeps the
    eps-perspective under the flag."""
    m = dm.Model("convex_quartic")
    x = m.continuous("x", lb=-2, ub=2)
    z = m.continuous("z", lb=-2, ub=2)
    m.either_or([[x**4 + z**2 <= 3], [x >= 1.8]], name="d")
    m.minimize(x + z)
    return m


def _exp_row():
    """A non-polynomial row (``exp``): unchanged under the flag."""
    m = dm.Model("exp_row")
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=8)
    m.either_or([[dm.exp(x) - y <= 0, x * y >= 1], [x <= 0.2]], name="d")
    m.minimize(-x + y)
    return m


def _linear_only():
    m = dm.Model("linear_only")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize(x)
    return m


SHAPES = [
    Shape(
        "bilinear_le",
        _bilinear_le,
        lambda v, t: (
            (v["x"] * v["z"] <= 1 + t and v["x"] + v["z"] >= 2.5 - t)
            or (v["x"] <= 0.5 + t and v["z"] >= 3 - t)
        ),
        points=[{"x": 0.4, "z": 2.5}, {"x": 2.5, "z": 0.4}, {"x": 0.5, "z": 3.0}],
    ),
    Shape(
        "cubic_eq",
        _cubic_eq,
        lambda v, t: (
            abs(v["x"] ** 2 * v["z"] - v["y"]) <= t
            or (v["x"] + v["z"] <= 0.5 + t and v["y"] <= 0.5 + t)
        ),
        points=[
            {"x": 1.0, "z": 1.0, "y": 1.0},
            {"x": 2.0, "z": 2.0, "y": 8.0},
            {"x": 1.5, "z": 0.5, "y": 1.125},
            {"x": 0.2, "z": 0.2, "y": 0.3},
        ],
    ),
    Shape(
        "shared_product",
        _shared_product,
        lambda v, t: (
            (v["x"] * v["z"] <= 2 + t and v["x"] * v["z"] + v["y"] >= 1.5 - t)
            or (v["x"] >= 2.5 - t and v["y"] <= t)
        ),
        points=[{"x": 1.0, "z": 1.0, "y": 0.5}, {"x": -2.0, "z": -1.0, "y": 0.0}],
    ),
]
BUILDERS = {
    "bilinear_le": _bilinear_le,
    "cubic_eq": _cubic_eq,
    "shared_product": _shared_product,
    "convex_quartic": _convex_quartic,
    "exp_row": _exp_row,
    "linear_only": _linear_only,
}

#: sha256 of ``repr(model_fingerprint(reformulate_gdp(build(), "hull")))`` on the
#: parent commit (before the flag existed), PYTHONHASHSEED-independent.
PRE_FLAG_FINGERPRINTS = {
    "bilinear_le": "5dae24244701ac35ac266c2d30d6e9ad6351f22a08bd664e9f076173a1aae83f",
    "cubic_eq": "0265d103438e264031250db0e5b261f379fce7a6f6d28c98fb59b0cadf3549e6",
    "shared_product": "ded8f69eeb4fb292f1a930935777363b4d7800a5f2e82ab14024578ca49ff4b3",
    "convex_quartic": "9c5b75462ff3635f99c4d2ef22e0f9a7d769bb28bc639a586c258baf2bfd4034",
    "exp_row": "6fcf0842f426f8affdb9cd86f22a5024dc16b08c99839ab6e69deb84a8210dec",
    "linear_only": "a5800b5c2303a0b42cc316d5fc8cf3ddd79aef27d065885c853d9cee37bb2463",
}


def _fp(model) -> str:
    return hashlib.sha256(repr(model_fingerprint(model)).encode()).hexdigest()


def _nonlinear_rows(low):
    return [c for c in low._constraints if not _is_linear(c.body)]


@pytest.fixture(autouse=True)
def _reset_stats():
    G._HULL_EXACT_STATS.update(homogenized=0, convex=0, not_polynomial=0)
    yield


# --------------------------------------------------------------------------- flag OFF


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("off", ["unset", "0"])
def test_flag_off_is_byte_identical_to_before(monkeypatch, name, off):
    if off == "unset":
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, "0")
    low = reformulate_gdp(BUILDERS[name](), "hull")
    assert _fp(low) == PRE_FLAG_FINGERPRINTS[name]
    assert G._HULL_EXACT_STATS == {"homogenized": 0, "convex": 0, "not_polynomial": 0}


@pytest.mark.parametrize("name", ["linear_only", "convex_quartic"])
def test_flag_on_without_a_nonconvex_polynomial_row_is_identical(monkeypatch, name):
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(BUILDERS[name](), "hull")
    assert _fp(low) == PRE_FLAG_FINGERPRINTS[name]
    assert G._HULL_EXACT_STATS["homogenized"] == 0


# --------------------------------------------------------------------------- monomials


def test_polynomial_monomials():
    m = dm.Model("mono")
    x = m.continuous("x", lb=-1, ub=1)
    y = m.continuous("y", lb=-1, ub=1)
    a = m.continuous("a", shape=(2,), lb=0, ub=1)
    p = m.parameter("p", 2.0)
    assert _polynomial_monomials(2 * x * y + 3) == {("x", "y"): 2.0, (): 3.0}
    assert _polynomial_monomials((x + y) * (x - y)) == {("x", "x"): 1.0, ("y", "y"): -1.0}
    assert _polynomial_monomials(x * y - y * x) == {}
    assert _polynomial_monomials(-(x**2) * y) == {("x", "x", "y"): -1.0}
    assert _polynomial_monomials((x + 1) ** 2) == {("x", "x"): 1.0, ("x",): 2.0, (): 1.0}
    assert _polynomial_monomials(x**0) == {(): 1.0}
    for bad in (x / y, dm.exp(x), x**0.5, x**-1, a[0] * x, p * x, x**9, dm.sqrt(x)):
        assert _polynomial_monomials(bad) is None, bad


# --------------------------------------------------------------------------- flag ON


@pytest.mark.parametrize(
    "name,homogenized,convex,declined",
    [
        ("bilinear_le", 1, 0, 0),
        ("cubic_eq", 1, 0, 0),
        ("shared_product", 2, 0, 0),
        ("convex_quartic", 0, 1, 0),
        ("exp_row", 1, 0, 1),
    ],
)
def test_flag_on_routes_rows(monkeypatch, name, homogenized, convex, declined):
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(BUILDERS[name](), "hull")
    assert G._HULL_EXACT_STATS == {
        "homogenized": homogenized,
        "convex": convex,
        "not_polynomial": declined,
    }
    rows = _nonlinear_rows(low)
    n_div = sum("/" in str(c.body) for c in rows)
    # every row that kept the eps-perspective divides by yhat; none of the others do
    eq_rows = 2 if name == "cubic_eq" else 1  # an equality is emitted as <= and >=
    assert len(rows) - n_div == homogenized * eq_rows
    assert n_div == (convex + declined) * (2 if name == "cubic_eq" else 1)


def _selected_point(low, declared_vals, k):
    """The lowered point with disjunct ``k`` selected: x and its copies at the
    declared values, every other disjunct's copies and selectors at 0."""
    vals = {}
    for v in low._variables:
        n = v.name
        if n in declared_vals:
            vals[n] = declared_vals[n]
        elif n.startswith("_hull_d_v_"):
            base, kk = n[len("_hull_d_v_") :].rsplit("_", 1)
            vals[n] = declared_vals[base] if int(kk) == k else 0.0
        elif "hull_d_" in n:  # selector
            vals[n] = 1.0 if n.endswith(f"_{k}") else 0.0
        else:
            raise AssertionError(f"unexpected lowered column {n!r}")
    x = np.zeros(sum(v.size for v in low._variables))
    for v in low._variables:
        x[low._flat_var_offset(v)] = vals[v.name]
    return x


@pytest.mark.parametrize(
    "name,point,k",
    [
        ("bilinear_le", {"x": 0.4, "z": 2.5}, 0),
        ("bilinear_le", {"x": 0.3, "z": 3.5}, 1),
        ("cubic_eq", {"x": 1.5, "z": 0.5, "y": 1.125}, 0),
        ("cubic_eq", {"x": 0.2, "z": 0.2, "y": 0.3}, 1),
        ("shared_product", {"x": 1.0, "z": 1.0, "y": 0.5}, 0),
        ("shared_product", {"x": 2.7, "z": -2.0, "y": -0.5}, 1),
    ],
)
def test_exact_at_fixed_selectors(monkeypatch, name, point, k):
    """At ``y_k = 1`` a homogenized row is the original row on the copies; at ``y_j = 0``
    it is exactly ``0.0`` in floating point (no eps, no constant: the #1043 class)."""
    monkeypatch.setenv(FLAG, "1")
    declared = BUILDERS[name]()
    low = reformulate_gdp(declared, "hull")
    x = _selected_point(low, point, k)
    n_rows = 0
    for con in low._constraints:
        if _is_linear(con.body):
            continue
        val = float(np.asarray(compile_point_fn(con.body, low)(x)).ravel()[0])
        disjunct = int(con.name.split("_d")[-1].split("_")[0])
        if disjunct != k:
            _check(val == 0.0, f"{con.name}: off-disjunct row = {val!r}, not exactly 0")
        else:
            ok = {"<=": val <= 1e-12, ">=": val >= -1e-12, "==": abs(val) <= 1e-12}
            _check(ok[con.sense], f"{con.name}: {val!r} {con.sense} 0 violated")
        n_rows += 1
    _check(n_rows > 0, "no nonlinear lowered row evaluated")


def test_off_disjunct_rows_are_exactly_zero(monkeypatch):
    """Every homogenized row at all copies 0 and every selector 0: exactly 0.0."""
    monkeypatch.setenv(FLAG, "1")
    n = 0
    for name in ("bilinear_le", "cubic_eq", "shared_product"):
        low = reformulate_gdp(BUILDERS[name](), "hull")
        x0 = np.zeros(sum(v.size for v in low._variables))
        for con in _nonlinear_rows(low):
            _check(float(np.asarray(compile_point_fn(con.body, low)(x0)).ravel()[0]) == 0.0)
            n += 1
    assert n == 5  # bilinear_le 1, cubic_eq 2 (== as <= and >=), shared_product 2


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_differential_battery(monkeypatch, shape):
    """No feasible point cut, no infeasible point admitted (sampled, solver-free)."""
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, "hull")
    assert rep.refused is None
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    assert rep.oracle_crosschecks == rep.comparisons, rep
    rep.assert_sound()
    assert G._HULL_EXACT_STATS["homogenized"] > 0


_OBJECTIVES = [(1, 1, 0), (-1, 1, 0), (1, -1, 0.5), (-1, -1, -0.3)]


@pytest.mark.parametrize("name", ["bilinear_le", "cubic_eq", "shared_product"])
def test_certified_optimum_matches_big_m(monkeypatch, name):
    monkeypatch.setenv(FLAG, "1")
    for cx, cz, cy in _OBJECTIVES:
        vals = {}
        for method in ("hull", "big-m"):
            m = BUILDERS[name]()
            v = {var.name: var for var in m._variables}
            obj = cx * v["x"] + cz * v["z"] + (cy * v["y"] if "y" in v else 0)
            m.minimize(obj)
            r = m.solve(gdp_method=method, time_limit=60)
            _check(r.status == "optimal", f"{name} {method}: {r.status}")
            vals[method] = r.objective
        _check(
            math.isclose(vals["hull"], vals["big-m"], rel_tol=1e-6, abs_tol=1e-6),
            f"{name} {(cx, cz, cy)}: hull {vals['hull']} vs big-m {vals['big-m']}",
        )
    assert G._HULL_EXACT_STATS["homogenized"] > 0


# --------------------------------------------------------------------------- GDPlib cstr

#: The prototype (``docs/dev/data/2142-exact-hull/exact_hull_proto.py``, arm ``homog``,
#: driver ``arm.py``) at 2,000 deterministic nodes: the implementation must reproduce
#: its tree bit for bit (bound-neutral against the prototype).
CSTR_PROTO_2000 = {"bound": 0.38595045554133645, "node_count": 2000}
CSTR_ORACLE = 3.0620073


@pytest.fixture(scope="module")
def cstr_flag_on():
    pytest.importorskip("pyomo")
    cstr = pytest.importorskip("gdplib.cstr")
    import discopt._rust as R
    from discopt.pyomo.gdp import convert_pyomo_gdp

    mp = pytest.MonkeyPatch()
    mp.setenv(FLAG, "1")
    mags: list[float] = []
    orig = R.solve_spatial_tree_py

    def wrapped(**k):
        lo, hi = np.asarray(k["global_lo"]), np.asarray(k["global_hi"])
        fin = np.isfinite(lo) & np.isfinite(hi)
        mags.append(float(np.max(np.abs(np.concatenate([lo[fin], hi[fin]])))))
        return orig(**k)

    mp.setattr(R, "solve_spatial_tree_py", wrapped)
    G._HULL_EXACT_STATS.update(homogenized=0, convex=0, not_polynomial=0)
    try:
        m = convert_pyomo_gdp(cstr.build_model()).model
        res = m.solve(gdp_method="hull", deterministic=True, max_nodes=2000, time_limit=1800)
        stats = dict(G._HULL_EXACT_STATS)
    finally:
        mp.undo()
    return res, mags, stats


@pytest.mark.slow
def test_cstr_spec_magnitude_bounded(cstr_flag_on):
    res, mags, stats = cstr_flag_on
    assert stats["homogenized"] == 5, stats
    assert mags, "the native spatial kernel never ran"
    assert max(mags) <= 1e4, max(mags)


@pytest.mark.slow
def test_cstr_2000_nodes_matches_prototype(cstr_flag_on):
    res, _, _ = cstr_flag_on
    assert res.node_count == CSTR_PROTO_2000["node_count"]
    assert res.bound == CSTR_PROTO_2000["bound"]
    assert res.bound <= CSTR_ORACLE + 1e-6
    assert res.objective is None or res.objective >= CSTR_ORACLE - 1e-5
