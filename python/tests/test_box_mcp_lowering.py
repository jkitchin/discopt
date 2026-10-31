"""Exact lowering of the box-bounded MCP relation ``F(z) ⊥ z in [l, u]`` (#2058).

The relation holds when ``z = l`` with ``F >= 0``, ``l < z < u`` with ``F = 0``, or
``z = u`` with ``F <= 0``. It is lowered by splitting it into at most two
symmetric NCP pairs on shifted slacks (``docs/dev/box-mcp-lowering-plan.md`` §3):

* both bounds finite: ``F = w⁺ - w⁻``, ``0 <= w⁺ ⊥ z - l``, ``0 <= w⁻ ⊥ u - z``;
* ``u = +inf``: ``0 <= F ⊥ z - l``; ``l = -inf``: ``0 <= -F ⊥ u - z``;
* both infinite: ``F == 0``; ``l == u``: ``z`` pinned, ``F`` free.

Each half goes to the unchanged NCP emitter of the method. Before #2058 every
non-NCP box was refused (``NotImplementedError``) by every lowering.

Expected values are hand-enumerated over the three branches in each docstring;
every solve also checks the **source** relation at the incumbent with the natural
map ``|z - mid(l, u, z - F)|``, so a test that only matched the objective could
not pass on a point the relation forbids.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import discopt.modeling.core as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt.mpec import (
    Complementarity,
    ComplementarityRole,
    _result_point,
    box_mcp,
    complementarity,
    generated_rows,
    reformulate_gdp,
    reformulate_scholtes,
    reformulate_sos1,
    refresh_lowering_bounds,
    solve_mpec,
    source_residuals,
)
from discopt.mpec_report import (
    ComplementarityKind,
    evaluate_at_point,
    point_from_flat,
    relation_residuals,
)

pytestmark = pytest.mark.smoke

METHODS = ("gdp", "sos1")
_TOL = 1e-6


def _natural_map(F: float, z: float, lo: float, hi: float) -> float:
    return abs(z - min(max(z - F, lo), hi))


def _assert_box_optimum(m, rels, *, objective, point=None):
    """Solve, require a certified optimum at ``objective`` (and ``point``), and
    require every **source** relation in ``rels`` to hold at the incumbent."""
    r = m.solve(time_limit=30.0)
    assert r.status == "optimal", r.status
    assert r.gap_certified
    assert r.objective == pytest.approx(objective, abs=_TOL), r.objective
    for name, v in (point or {}).items():
        assert float(r.x[name]) == pytest.approx(v, abs=1e-5), (name, r.x)
    rows = relation_residuals(m, list(rels), _result_point(m, r))
    assert rows, "the residual measured nothing"
    assert max(float(rw.complementarity.value) for rw in rows) < 1e-5, rows
    return r


def _aux(m):
    return [v.name for v in m._variables if "_wp_" in v.name or "_wm_" in v.name]


# ── §6.2 case matrix ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_guard_probe_z_at_lower_bound(method):
    """``F = z + 1`` on ``[-1, 1]``: ``z = 1`` needs ``F = 2 <= 0`` (no); the
    interior needs ``z = -1`` (not interior); ``z = -1`` has ``F = 0 >= 0``. The
    feasible set is ``{-1}``, so ``max z`` is ``-1``. This is the probe whose
    unlowered solve certified ``z = 1`` in #1149."""
    m = dm.Model("A")
    z = m.continuous("z", lb=-1, ub=1)
    m.maximize(z)
    rel = m.mcp(z + 1, z, method=method, name="c")
    assert rel.role is ComplementarityRole.BOX_MCP
    assert rel.lowering_in(m) == method
    _assert_box_optimum(m, [rel], objective=-1.0, point={"z": -1.0})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "target, objective, z_star",
    [(0.8, 0.04, 1.0), (-0.9, 0.01, -1.0), (0.1, 0.01, 0.0)],
)
def test_three_point_feasible_set(method, target, objective, z_star):
    """``F = -z`` on ``[-1, 1]``: ``z = -1`` has ``F = 1 >= 0``; the interior needs
    ``z = 0``; ``z = 1`` has ``F = -1 <= 0``. Feasible set ``{-1, 0, 1}``, so
    ``min (z - target)^2`` picks the nearest of the three."""
    m = dm.Model("B")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize((z - target) ** 2)
    rel = m.mcp(-z, z, method=method, name="c")
    _assert_box_optimum(m, [rel], objective=objective, point={"z": z_star})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sense, objective", [("maximize", 1.0), ("minimize", -1.0)])
def test_nonlinear_residual(method, sense, objective):
    """``F = z^2 - 1`` on ``[-1, 2]``: ``z = -1`` has ``F = 0``; the interior root
    is ``z = 1``; ``z = 2`` has ``F = 3 > 0`` (excluded). Feasible ``{-1, 1}``."""
    m = dm.Model("C")
    z = m.continuous("z", lb=-1, ub=2)
    getattr(m, sense)(z)
    rel = m.mcp(z * z - 1, z, method=method, name="c")
    _assert_box_optimum(m, [rel], objective=objective)


@pytest.mark.parametrize("method", METHODS)
def test_z_is_an_expression(method):
    """``F = x - y ⊥ x + y in [0.5, 1.5]``, ``min x + 3y`` over ``[0, 1]^2``. At the
    lower face ``x + y = 0.5`` with ``x >= y``: best ``(0.5, 0)``, 0.5. Interior
    needs ``x = y`` with ``x + y > 0.5`` (objective > 1); the upper face needs
    ``x <= y`` on ``x + y = 1.5`` (objective >= 3)."""
    m = dm.Model("D")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x + 3 * y)
    rel = m.mcp(x - y, x + y, lb=0.5, ub=1.5, method=method, name="c")
    _assert_box_optimum(m, [rel], objective=0.5, point={"x": 0.5, "y": 0.0})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sense, objective", [("maximize", 3.0), ("minimize", 1.0)])
def test_one_sided_upper_infinite(method, sense, objective):
    """``F = 3 - z ⊥ z in [1, +inf)``: ``z = 1`` has ``F = 2 >= 0``; interior
    ``z = 3``. Feasible ``{1, 3}``. One NCP half, no split auxiliaries."""
    m = dm.Model("E")
    z = m.continuous("z", lb=1, ub=50)
    getattr(m, sense)(z)
    rel = m.mcp(3 - z, z, lb=1, ub=np.inf, method=method, name="c")
    assert rel.role is ComplementarityRole.BOX_MCP
    assert _aux(m) == []
    _assert_box_optimum(m, [rel], objective=objective)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_one_sided_lower_infinite(method, sense):
    """``F = z - 1 ⊥ z in (-inf, 2]``: ``z = 2`` has ``F = 1 > 0`` (excluded);
    interior ``z = 1``. Feasible ``{1}``."""
    m = dm.Model("F")
    z = m.continuous("z", lb=-50, ub=2)
    getattr(m, sense)(z)
    rel = m.mcp(z - 1, z, lb=-np.inf, ub=2, method=method, name="c")
    assert _aux(m) == []
    _assert_box_optimum(m, [rel], objective=1.0, point={"z": 1.0})


@pytest.mark.parametrize("method", METHODS)
def test_both_infinite_is_an_equation(method):
    m = dm.Model("eq")
    z = m.continuous("z", lb=-50, ub=50)
    m.maximize(z)
    rel = m.mcp(z - 4, z, lb=-np.inf, ub=np.inf, method=method, name="c")
    assert [c.name for c in m._constraints] == ["c_eq"]
    assert [c.name for c in rel.rows_in(m)] == ["c_eq"]
    _assert_box_optimum(m, [rel], objective=4.0)


@pytest.mark.parametrize("method", METHODS)
def test_fixed_box_leaves_F_free(method):
    """``l == u``: ``z`` is pinned to 2 and ``F = x - 7`` is unconstrained, so
    ``min x`` is 0 although ``F = -7``."""
    m = dm.Model("fixed")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=-5, ub=5)
    m.minimize(x)
    rel = m.mcp(x - 7, z, lb=2, ub=2, method=method, name="c")
    assert [c.name for c in rel.rows_in(m)] == ["c_z_lb", "c_z_ub"]
    _assert_box_optimum(m, [rel], objective=0.0, point={"x": 0.0, "z": 2.0})


@pytest.mark.parametrize("method", METHODS)
def test_declared_bounds_narrower_than_box(method):
    """``z`` declared on ``[0, 1]``, box ``[-1, 2]``, ``F = 0.5 - z``. Neither box
    bound is reachable, so only the interior branch applies: ``z = 0.5``. Taking
    the box from ``z``'s declared bounds instead would also admit ``z = 0``
    (``F = 0.5 >= 0``) and ``z = 1`` (``F = -0.5 <= 0``), giving 0 and 1."""
    for sense in ("maximize", "minimize"):
        m = dm.Model("narrow")
        z = m.continuous("z", lb=0, ub=1)
        getattr(m, sense)(z)
        rel = m.mcp(0.5 - z, z, lb=-1, ub=2, method=method, name="c")
        _assert_box_optimum(m, [rel], objective=0.5)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("F, sense, z_star", [(2.0, "maximize", -1.0), (-2.0, "minimize", 1.0)])
def test_constant_residual_pins_z_to_a_bound(method, F, sense, z_star):
    """``F = 2 > 0`` can only sit at ``z = l``; ``F = -2`` only at ``z = u``."""
    m = dm.Model("const")
    z = m.continuous("z", lb=-1, ub=1)
    getattr(m, sense)(z)
    rel = m.mcp(F, z, method=method, name="c")
    _assert_box_optimum(m, [rel], objective=z_star, point={"z": z_star})


def _cournot(m, q, caps):
    """Marginal-profit residuals of a 2-firm Cournot game, ``a=10, b=1, c=1``:
    ``F_i = c - a + b (q_1 + q_2) + b q_i``. Uncapped equilibrium ``(3, 3)``."""
    return [1.0 - 10.0 + (q[0] + q[1]) + q[i] for i in range(2)]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_vector_relation_same_box(method, sense):
    """Caps ``(2, 2)``: ``F_i(2, 2) = -3 <= 0`` at the upper bound for both firms,
    and that is the unique equilibrium (plan Appendix C), so both senses give 4."""
    m = dm.Model("cournot")
    q = m.continuous("q", shape=(2,), lb=0, ub=2)
    getattr(m, sense)(q[0] + q[1])
    F = 1.0 - 10.0 + (q[0] + q[1]) + q
    rel = m.mcp(F, q, lb=0.0, ub=2.0, method=method, name="c")
    assert len(rel.elements(m)) == 2
    names = {c.name for c in m._constraints}
    assert {"c_0_z_lb", "c_1_z_ub", "c_0_split", "c_1_split"} <= names, names
    _assert_box_optimum(m, [rel], objective=4.0)


@pytest.mark.parametrize("method", METHODS)
def test_per_element_boxes_cournot(method):
    """Caps ``(2, 10)``: firm 1 at its cap needs ``F_1 = q_2 - 5 <= 0``; firm 2
    interior needs ``2 q_2 - 7 = 0``. Equilibrium ``(2, 3.5)``."""
    m = dm.Model("cournot2")
    q1 = m.continuous("q1", lb=0, ub=2)
    q2 = m.continuous("q2", lb=0, ub=10)
    m.maximize(q1 + q2)
    r1 = m.mcp(-9.0 + 2 * q1 + q2, q1, method=method, name="f1")
    r2 = m.mcp(-9.0 + q1 + 2 * q2, q2, method=method, name="f2")
    _assert_box_optimum(m, [r1, r2], objective=5.5, point={"q1": 2.0, "q2": 3.5})


@pytest.mark.parametrize("pmax, objective, p_star", [(3.0, 49.0, 3.0), (5.0, 36.0, 4.0)])
def test_price_cap_market(pmax, objective, p_star):
    """The ``Model.mcp`` docstring example. Excess supply ``F = 3p - 12``; uncapped
    the market clears at ``p = 4``. A cap ``3`` binds (``F(3) = -3 <= 0``); a cap
    ``5`` does not, and ``p = 5`` is excluded (``F(5) = 3 > 0``)."""
    m = dm.Model("price_cap")
    p = m.continuous("p", lb=0, ub=pmax)
    m.minimize((p - 10) ** 2)
    rel = m.mcp(3 * p - 12, p)
    _assert_box_optimum(m, [rel], objective=objective, point={"p": p_star})


def test_scholtes_on_a_box_is_local():
    pytest.importorskip("pounce")
    m = dm.Model("B_scholtes")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize((z - 0.8) ** 2)
    rel = box_mcp(-z, z, lb=-1.0, ub=1.0, name="c")
    r = solve_mpec(m, [rel], method="scholtes", t_min=1e-8)
    assert r.status in {"local_optimal", "local_limit"}, r.status
    assert r.bound is None
    assert r.gap_certified is False
    assert r.objective == pytest.approx(0.04, abs=1e-4)
    rep = r.mpec_report
    assert ComplementarityKind.formula(ComplementarityKind.NATURAL_MAP) in (
        rep.complementarity.definition
    )
    assert rep.complementarity.value < 1e-4
    # The Scholtes rows are a relaxation; a global solve over them still refuses.
    with pytest.raises(NotImplementedError, match="RELAXATION"):
        m.solve(time_limit=5.0)


def test_gdp_and_sos1_agree_on_random_linear_boxes():
    """20 seeded ``F = a z + b`` on random ``[l, u]``, ``min (z - c)^2``. The
    feasible set is ``{l if F(l) >= 0} ∪ {u if F(u) <= 0} ∪ {-b/a if interior}``;
    both lowerings must find the brute-force optimum over it."""
    rng = np.random.default_rng(2058)
    compared = 0
    for _ in range(20):
        lo = float(np.round(rng.uniform(-3, 0), 3))
        hi = float(np.round(lo + rng.uniform(0.5, 4), 3))
        a = float(np.round(rng.choice([-1, 1]) * rng.uniform(0.5, 2), 3))
        b = float(np.round(rng.uniform(-3, 3), 3))
        c = float(np.round(rng.uniform(lo - 1, hi + 1), 3))
        cand = []
        if a * lo + b >= 0:
            cand.append(lo)
        if a * hi + b <= 0:
            cand.append(hi)
        root = -b / a
        if lo < root < hi:
            cand.append(root)
        assert cand, "a box MCP on a bounded box always has a solution"
        want = min((zc - c) ** 2 for zc in cand)
        got = {}
        for method in METHODS:
            m = dm.Model("rand")
            z = m.continuous("z", lb=lo, ub=hi)
            m.minimize((z - c) ** 2)
            rel = m.mcp(a * z + b, z, method=method, name="c")
            got[method] = _assert_box_optimum(m, [rel], objective=want).objective
        assert got["gdp"] == pytest.approx(got["sos1"], abs=_TOL)
        compared += 1
    assert compared == 20


# ── §6.3 exactness of the decomposition, in closed form ──────────────────────


def test_box_decomposition_projection_matches_natural_map():
    """The B0/B1 rows admit ``(z, F)`` iff the relation holds. The case split
    ``(w⁺ = 0 ∨ z = l) × (w⁻ = 0 ∨ z = u)`` is done in closed form with the split
    auxiliaries bounded by an enclosure ``F in [-2, 2]``."""
    f_lo, f_hi = -2.0, 2.0
    compared = disagreements = 0
    for lo, hi in ((-1.0, 1.0), (0.0, 2.0), (-3.0, 0.0), (0.5, 0.5)):
        for z in np.linspace(lo - 0.5, hi + 0.5, 401):
            for F in (-2.0, -0.5, 0.0, 0.5, 2.0):
                in_box = lo - 1e-12 <= z <= hi + 1e-12
                mcp = in_box and _natural_map(F, z, lo, hi) <= 1e-12
                if lo == hi:  # B0: box rows only, F free
                    rows = in_box
                else:  # B1
                    rows = in_box and (
                        F == 0.0
                        or (abs(z - hi) <= 1e-12 and 0.0 <= -F <= max(0.0, -f_lo))
                        or (abs(z - lo) <= 1e-12 and 0.0 <= F <= max(0.0, f_hi))
                    )
                disagreements += mcp != rows
                compared += 1
    assert compared == 4 * 401 * 5 == 8020
    assert disagreements == 0


# ── §6.7 residual reports and row tracking ───────────────────────────────────


def test_box_residual_report_from_solve_mpec_gdp():
    m = dm.Model("B_report")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize((z - 0.8) ** 2)
    rel = box_mcp(-z, z, lb=-1.0, ub=1.0, name="c")
    r = solve_mpec(m, [rel], method="gdp")
    assert r.status == "optimal"
    rep = r.mpec_report
    assert rep is not None
    assert rep.n_scalar_relations == 1
    assert ComplementarityKind.formula(ComplementarityKind.NATURAL_MAP) in (
        rep.complementarity.definition
    )
    assert rep.complementarity.value < _TOL
    assert rep.lowered_row_residual is not None and rep.lowered_row_residual.value < _TOL
    assert rep.bound_violation.value == 0


def test_box_rows_are_tracked_on_the_parent():
    m = dm.Model("rows")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize(z)
    rel = m.mcp(-z, z, name="c")
    rows = rel.rows_in(m)
    assert rows is not None
    names = [c.name for c in rows]
    assert {
        "c_z_lb",
        "c_z_ub",
        "c_split",
        "c_lo_f_nonneg",
        "c_lo_g_nonneg",
        "c_hi_f_nonneg",
        "c_hi_g_nonneg",
        "c_lo",
        "c_hi",
    } == set(names), names
    assert generated_rows(m, [rel]) == rows
    assert m._complementarities == [rel], "the NCP halves are not registered"


def test_source_residuals_after_plain_solve():
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x + 3 * y)
    m.mcp(x - y, x + y, lb=0.5, ub=1.5, name="c")
    r = m.solve(time_limit=30.0)
    rep = source_residuals(m, r)
    assert rep is not None and rep.n_scalar_relations == 1
    assert rep.complementarity.value < _TOL


# ── §6.8 warm start ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_box_warm_start_lifts_the_split_auxiliaries(method):
    m = dm.Model("D_warm")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x + 3 * y)
    rel = box_mcp(x - y, x + y, lb=0.5, ub=1.5, name="c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = solve_mpec(m, [rel], method=method, initial_solution={x: 0.5, y: 0.0})
    assert not [w for w in caught if "could not lift" in str(w.message)]
    assert r.objective == pytest.approx(0.5, abs=_TOL)


def test_split_lifts_evaluate_to_the_positive_and_negative_parts():
    m = dm.Model("lifts")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    rel = box_mcp(x - y, x + y, lb=0.5, ub=1.5, name="c")
    lifts: list = []
    reformulate_gdp(m, [rel], _operand_lifts=lifts)
    split = {aux.name: expr for aux, expr in lifts if "_w" in aux.name}
    assert len(split) == 2, [a.name for a, _ in lifts]
    checked = 0
    for (xv, yv), (wp, wm) in (((0.5, 0.0), (0.5, 0.0)), ((0.2, 0.9), (0.0, 0.7))):
        flat = np.zeros(sum(v.size for v in m._variables))
        flat[0], flat[1] = xv, yv
        point = point_from_flat(m, flat)
        for name, expr in split.items():
            want = wp if "_wp_" in name else wm
            got = np.ravel(evaluate_at_point(m, expr, point))
            assert got.size == 1 and float(got[0]) == pytest.approx(want)
            checked += 1
    assert checked == 4


# ── §6.9 copies ──────────────────────────────────────────────────────────────


def _probe_b():
    m = dm.Model("B_copy")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize((z - 0.8) ** 2)
    m.mcp(-z, z, name="c")
    return m


@pytest.mark.parametrize("how", ["clone", "deepcopy", "pickle", "save_load"])
def test_box_lowered_model_survives_copies(how, tmp_path):
    m = _probe_b()
    if how == "clone":
        c = m.clone()
    elif how == "deepcopy":
        c = copy.deepcopy(m)
    elif how == "pickle":
        c = pickle.loads(pickle.dumps(m))
    else:
        from discopt.serialize import load

        m.save(tmp_path / "m.json")
        c = load(tmp_path / "m.json")
    assert len(c._constraints) == len(m._constraints)
    assert len(c._variables) == len(m._variables)
    (rel,) = c._complementarities
    assert rel.g_bounds == (-1.0, 1.0)
    assert rel.lowering_in(c) == "gdp"
    assert c.solve(time_limit=30.0).objective == pytest.approx(0.04, abs=_TOL)
    assert m.solve(time_limit=30.0).objective == pytest.approx(0.04, abs=_TOL)


@pytest.mark.parametrize("name", ["mpec.gdp", "mpec.sos1"])
def test_box_mcp_check_on_copy(name):
    m = dm.Model("B_check")
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize((z - 0.8) ** 2)
    rel = box_mcp(-z, z, lb=-1.0, ub=1.0, name="c")
    report = dt.check(name, m, pairs=[rel])
    assert m._constraints == [] and not rel.is_lowered_into(m)
    out = report.result
    (lowered,) = out._lowered_complementarities
    assert lowered.is_lowered_into(out)
    assert report.diff.added_constraints
    assert out.solve(time_limit=30.0).objective == pytest.approx(0.04, abs=_TOL)


# ── §6.10 refusals ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_refuses_unbounded_residual_in_two_sided_box(method):
    m = dm.Model("unbounded")
    z = m.continuous("z", lb=-1, ub=1)
    y = m.continuous("y", lb=-np.inf, ub=np.inf)
    m.minimize(z)
    with pytest.raises(ValueError, match="cannot bound the residual"):
        m.mcp(y, z, method=method, name="c")
    assert m._constraints == []
    assert [v.name for v in m._variables] == ["z", "y"]
    assert m._complementarities == []


def test_vector_refusal_leaves_no_rows_from_earlier_elements():
    """Element 1's residual is unbounded; element 0's is not. Every element is
    validated before any is emitted, so nothing from element 0 is left behind."""
    m = dm.Model("vec_refuse")
    q = m.continuous("q", shape=(2,), lb=0, ub=1)
    y = m.continuous("y", lb=-np.inf, ub=np.inf)
    m.minimize(q[0])
    F = q * 0.0 + np.array([1.0, 0.0]) + y * np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="cannot bound the residual"):
        m.mcp(F, q, lb=0.0, ub=1.0, name="c")
    assert m._constraints == []
    assert m._complementarities == []
    assert not _aux(m)


@pytest.mark.parametrize(
    "lower",
    [reformulate_gdp, reformulate_sos1, lambda m, p: reformulate_scholtes(m, p, 0.1)],
    ids=["gdp", "sos1", "scholtes"],
)
def test_refuses_inexpressible_residual_bounds(lower):
    m = dm.Model("inexpressible")
    z = m.continuous("z", lb=-1, ub=1)
    rel = Complementarity(z, z, "q", f_bounds=(1.0, np.inf), g_bounds=(-1.0, 1.0))
    with pytest.raises(ValueError, match="declares residual bounds"):
        lower(m, [rel])
    assert m._constraints == []


def test_refuses_empty_box_built_directly():
    m = dm.Model("empty")
    z = m.continuous("z", lb=-1, ub=5)
    rel = Complementarity(z, z, "q", f_bounds=(-np.inf, np.inf), g_bounds=(3.0, 1.0))
    with pytest.raises(ValueError, match="empty box"):
        reformulate_gdp(m, [rel])
    assert m._constraints == []


def test_box_mcp_refuses_array_bounds():
    m = dm.Model("arr")
    q = m.continuous("q", shape=(2,), lb=0, ub=2)
    with pytest.raises(ValueError, match="scalars"):
        box_mcp(q - 1.0, q, lb=np.array([0.0, 0.0]), ub=np.array([2.0, 2.0]), name="c")


@pytest.mark.parametrize("method, match", [("scholtes", "solve_mpec"), ("bigm", "unknown")])
def test_model_mcp_refuses_methods_it_cannot_apply(method, match):
    m = dm.Model("meth")
    z = m.continuous("z", lb=-1, ub=1)
    with pytest.raises(ValueError, match=match):
        m.mcp(-z, z, method=method)
    assert m._constraints == [] and m._complementarities == []


# ── #2061 review ─────────────────────────────────────────────────────────────


def _ub(m, prefix):
    return [float(v.ub) for v in m._variables if v.name.startswith(prefix)]


@pytest.mark.parametrize("method", METHODS)
def test_widening_after_lowering_does_not_cut_the_optimum(method):
    """Blocker: ``w⁺ <= 1`` was derived from ``x <= 1`` at declaration and kept after
    ``x.ub = 50``, so the solve certified -10 against a true -50. ``F = x - z`` with
    ``z in [0, 10]``: ``x = 50, z = 0`` gives ``F = 40 >= 0`` at ``z = l``."""
    m = dm.Model("widen")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=10)
    rel = box_mcp(x - z, z, 0, 10)
    (reformulate_gdp if method == "gdp" else reformulate_sos1)(m, [rel])
    x.ub = 50.0
    m.minimize(-x)
    _assert_box_optimum(m, [rel], objective=-50.0, point={"x": 50.0, "z": 0.0})


def test_widening_after_model_mcp_does_not_cut_the_optimum():
    """The reviewer's exact repro, through ``Model.mcp``."""
    m = dm.Model("widen_mcp")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=10)
    m.mcp(x - z, z, lb=0, ub=10)
    x.ub = 50.0
    m.minimize(-x)
    r = m.solve(time_limit=30.0)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-50.0, abs=_TOL)


@pytest.mark.parametrize("method", METHODS)
def test_ncp_operand_aux_bounds_follow_a_widening(method):
    """The same staleness in the NCP lowerings: the GDP lift / SOS1 aux of ``x*x``
    was bounded by ``x <= 1`` and stayed at 1 after ``x.ub = 3``. ``max x`` subject to
    ``0 <= x*x ⊥ y >= 0`` with ``y = 0`` admits ``x = 3``."""
    m = dm.Model("ncp_widen")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=10)
    rel = complementarity(x * x, y, name="p")
    (reformulate_gdp if method == "gdp" else reformulate_sos1)(m, [rel])
    x.ub = 3.0
    m.minimize(-x + y)
    r = m.solve(time_limit=30.0)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-3.0, abs=_TOL)


def test_refresh_is_a_no_op_on_an_unchanged_model():
    m = dm.Model("noop")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    z = m.continuous("z", shape=(2,), lb=-5, ub=5)
    m.mcp(x * x - z, z, lb=-1, ub=2)
    m.complementarity(x[0] * x[1] + 1, z[0] + 5)
    assert refresh_lowering_bounds(m, context="t") == 0


def test_refresh_tracks_a_narrowing_and_refuses_an_unbounded_widening():
    m = dm.Model("narrow")
    x = m.continuous("x", lb=0, ub=4)
    z = m.continuous("z", lb=0, ub=10)
    m.mcp(x - z, z, lb=0, ub=10)
    assert _ub(m, "_compl0_wp_") == [pytest.approx(4.0)]
    x.ub = 2.0
    assert refresh_lowering_bounds(m, context="t") == 1
    assert _ub(m, "_compl0_wp_") == [pytest.approx(2.0)]
    x.ub = np.inf
    m.minimize(-x)
    with pytest.raises(ValueError, match="cannot bound the residual"):
        m.solve()


@pytest.mark.parametrize("how", ["clone", "pickle"])
def test_widening_a_copy_refreshes_the_copy(how):
    m = dm.Model("copy_widen")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=10)
    m.mcp(x - z, z, lb=0, ub=10)
    m.minimize(-x)
    c = m.clone() if how == "clone" else pickle.loads(pickle.dumps(m))
    next(v for v in c._variables if v.name == "x").ub = 50.0
    r = c.solve(time_limit=30.0)
    assert r.status == "optimal" and r.objective == pytest.approx(-50.0, abs=_TOL)
    # the original is untouched by the copy's refresh
    assert _ub(m, "_compl0_wp_") == [pytest.approx(1.0)]


def test_save_writes_refreshed_bounds_and_load_refuses_a_widening():
    from discopt.serialize import dumps, loads

    m = dm.Model("saved")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=10)
    m.mcp(x - z, z, lb=0, ub=10)
    m.minimize(-x)
    x.ub = 50.0
    # widened in-session, then saved: the file carries the refreshed w+ bound
    r = loads(dumps(m)).solve(time_limit=30.0)
    assert r.status == "optimal" and r.objective == pytest.approx(-50.0, abs=_TOL)

    x.ub = 1.0
    doc = dumps(m)
    narrowed = loads(doc)
    next(v for v in narrowed._variables if v.name == "x").ub = 0.5
    assert narrowed.solve(time_limit=30.0).objective == pytest.approx(-0.5, abs=_TOL)

    widened = loads(doc)
    next(v for v in widened._variables if v.name == "x").ub = 50.0
    with pytest.raises(ValueError, match="was widened after this model was read"):
        widened.solve()
    with pytest.raises(ValueError, match="was widened after this model was read"):
        dumps(widened)


def test_refuses_residual_enclosure_on_default_sentinel_bounds():
    """``z`` on default bounds encloses ``y - z`` in ``±9.999e19`` -- finite to
    ``np.isfinite``, unbounded to the model. It was accepted as a real big-M."""
    m = dm.Model("sentinel")
    z = m.continuous("z")
    y = m.continuous("y", lb=0, ub=1)
    n = len(m._constraints)
    with pytest.raises(ValueError, match="cannot bound the residual"):
        m.mcp(y - z, y, lb=0, ub=1)
    assert len(m._constraints) == n


@pytest.mark.parametrize("pin", [np.inf, -np.inf])
def test_refuses_a_box_pinned_at_infinity(pin):
    """``l == u == ±inf`` read as B0 and emitted no rows at all; the solve then
    ignored the relation (-5 for ``min z`` over ``[-5, 5]``)."""
    m = dm.Model("pinned")
    z = m.continuous("z", lb=-5, ub=5)
    with pytest.raises(ValueError, match="pins the variable at infinity"):
        m.mcp(z - 1, z, lb=pin, ub=pin)
    direct = Complementarity(
        z - 1,
        z,
        "d",
        role=ComplementarityRole.BOX_MCP,
        f_bounds=(-np.inf, np.inf),
        g_bounds=(pin, pin),
    )
    with pytest.raises(ValueError, match="pins the variable at infinity"):
        reformulate_gdp(m, [direct])
    assert not m._constraints


@pytest.mark.parametrize("method", METHODS)
def test_multi_relation_refusal_leaves_no_rows_from_earlier_relations(method):
    m = dm.Model("multi")
    a = m.continuous("a", lb=0, ub=1)
    b = m.continuous("b")
    w = m.continuous("w", lb=0, ub=2)
    r1 = box_mcp(a * a - 0.5, a, 0, 1, name="r1")
    r2 = box_mcp(b - w, w, 0, 2, name="r2")
    nv = len(m._variables)
    with pytest.raises(ValueError, match="cannot bound the residual"):
        (reformulate_gdp if method == "gdp" else reformulate_sos1)(m, [r1, r2])
    assert not m._constraints and len(m._variables) == nv
    assert not r1.is_lowered_into(m)
    with pytest.raises(ValueError, match="cannot bound the residual"):
        solve_mpec(m, [r1, r2], method=method)
    assert not m._constraints and not m._complementarities


def test_split_enclosure_uses_the_relations_box_on_z():
    """``F = x - z`` with ``x in [0, 1]``, ``z`` declared on ``[-100, 100]`` but
    complementary on ``[0, 2]``: ``w⁺ <= 1``, ``w⁻ <= 2`` -- not 101 / 100."""
    m = dm.Model("zbox")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=-100, ub=100)
    rel = box_mcp(x - z, z, 0, 2)
    reformulate_gdp(m, [rel])
    assert _ub(m, f"_{rel.name}_wp_") == [pytest.approx(1.0)]
    assert _ub(m, f"_{rel.name}_wm_") == [pytest.approx(2.0)]
    m.minimize(-x)
    _assert_box_optimum(m, [rel], objective=-1.0)


def test_split_enclosure_uses_the_relations_box_on_an_element():
    m = dm.Model("zbox_elem")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    z = m.continuous("z", shape=(2,), lb=-100, ub=100)
    rel = box_mcp(x - z, z, 0, 2)
    reformulate_gdp(m, [rel])
    assert sorted(_ub(m, f"_{rel.name}_0_wm_") + _ub(m, f"_{rel.name}_1_wm_")) == [
        pytest.approx(2.0),
        pytest.approx(2.0),
    ]


def test_vector_residual_against_scalar_z_emits_box_rows_once():
    m = dm.Model("dedupe")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    rel = box_mcp(x - z, z, 0, 1)
    reformulate_gdp(m, [rel])
    names = [c.name for c in rel.rows_in(m) if c.name]
    assert sum(n.endswith("_z_lb") for n in names) == 1
    assert sum(n.endswith("_z_ub") for n in names) == 1
