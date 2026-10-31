"""Fix-and-solve differential for the box-MCP lowering (#2058, plan §6.4).

For each shape and each exact method, the **source** variables are fixed to a
sample point and the lowered model is solved:

* a point where the declared relation holds (natural map ``<= 1e-9``) must solve
  ``optimal`` -- the lowered rows, split-auxiliary bounds and big-M values cut no
  valid point;
* a point where it is violated (natural map ``> 1e-3``) must be ``infeasible`` --
  the rows admit no point the relation forbids.

Membership is decided from the relation in closed form, never from the rows under
test. Both arms must fire on every shape (CLAUDE.md §6).
"""

from __future__ import annotations

import discopt.modeling.core as dm
import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.correctness]

METHODS = ("gdp", "sos1")


def _nat(F, z, lo, hi):
    return abs(z - min(max(z - F, lo), hi))


# Each shape: (declared box per source variable, build(m, vars, method), residual(point),
# hand-picked boundary samples). ``residual`` is the max natural map over the
# shape's relations at a point {name: value}.
SHAPES = {
    "A": (
        {"z": (-1.0, 1.0)},
        lambda m, v, meth: m.mcp(v["z"] + 1, v["z"], method=meth),
        lambda p: _nat(p["z"] + 1, p["z"], -1.0, 1.0),
        [{"z": -1.0}, {"z": 1.0}, {"z": 0.0}],
    ),
    "B": (
        {"z": (-1.0, 1.0)},
        lambda m, v, meth: m.mcp(-v["z"], v["z"], method=meth),
        lambda p: _nat(-p["z"], p["z"], -1.0, 1.0),
        [{"z": -1.0}, {"z": 0.0}, {"z": 1.0}, {"z": 0.5}],
    ),
    "C": (
        {"z": (-1.0, 2.0)},
        lambda m, v, meth: m.mcp(v["z"] * v["z"] - 1, v["z"], method=meth),
        lambda p: _nat(p["z"] ** 2 - 1, p["z"], -1.0, 2.0),
        [{"z": -1.0}, {"z": 1.0}, {"z": 2.0}, {"z": 0.0}],
    ),
    "D": (
        {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        lambda m, v, meth: m.mcp(v["x"] - v["y"], v["x"] + v["y"], lb=0.5, ub=1.5, method=meth),
        lambda p: _nat(p["x"] - p["y"], p["x"] + p["y"], 0.5, 1.5),
        [{"x": 0.5, "y": 0.0}, {"x": 0.6, "y": 0.6}, {"x": 0.5, "y": 1.0}, {"x": 0.0, "y": 0.5}],
    ),
    "E": (
        {"z": (1.0, 50.0)},
        lambda m, v, meth: m.mcp(3 - v["z"], v["z"], lb=1.0, ub=np.inf, method=meth),
        lambda p: _nat(3 - p["z"], p["z"], 1.0, np.inf),
        [{"z": 1.0}, {"z": 3.0}, {"z": 50.0}],
    ),
    "G": (
        {"q1": (0.0, 2.0), "q2": (0.0, 10.0)},
        lambda m, v, meth: (
            m.mcp(-9.0 + 2 * v["q1"] + v["q2"], v["q1"], method=meth),
            m.mcp(-9.0 + v["q1"] + 2 * v["q2"], v["q2"], method=meth),
        ),
        lambda p: max(
            _nat(-9.0 + 2 * p["q1"] + p["q2"], p["q1"], 0.0, 2.0),
            _nat(-9.0 + p["q1"] + 2 * p["q2"], p["q2"], 0.0, 10.0),
        ),
        [{"q1": 2.0, "q2": 3.5}, {"q1": 0.0, "q2": 9.0}, {"q1": 2.0, "q2": 10.0}],
    ),
    "narrow": (
        {"z": (0.0, 1.0)},
        lambda m, v, meth: m.mcp(0.5 - v["z"], v["z"], lb=-1.0, ub=2.0, method=meth),
        lambda p: _nat(0.5 - p["z"], p["z"], -1.0, 2.0),
        [{"z": 0.0}, {"z": 0.5}, {"z": 1.0}],
    ),
}


def _samples(box, fixed, rng, n=40):
    out = list(fixed)
    while len(out) < n:
        out.append(
            {k: float(np.round(rng.uniform(lo, min(hi, 60.0)), 4)) for k, (lo, hi) in box.items()}
        )
    return out


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_fixed_source_point_solves_iff_the_relation_holds(shape, method):
    box, build, residual, fixed = SHAPES[shape]
    rng = np.random.default_rng([2058, sorted(SHAPES).index(shape), METHODS.index(method)])
    m = dm.Model(f"diff_{shape}")
    v = {k: m.continuous(k, lb=lo, ub=hi) for k, (lo, hi) in box.items()}
    m.minimize(sum(v.values()))
    build(m, v, method)

    n_feas = n_infeas = n_skipped = 0
    for point in _samples(box, fixed, rng):
        r_src = residual(point)
        if 1e-9 < r_src <= 1e-3:
            n_skipped += 1
            continue
        c = m.clone()
        by_name = {var.name: var for var in c._variables}
        for k, val in point.items():
            by_name[k].lb = val
            by_name[k].ub = val
        res = c.solve(time_limit=10.0)
        if r_src <= 1e-9:
            assert res.status == "optimal", (shape, method, point, res.status)
            n_feas += 1
        else:
            assert res.status == "infeasible", (shape, method, point, r_src, res.status)
            n_infeas += 1
    print(f"{shape}/{method}: feasible={n_feas} infeasible={n_infeas} skipped={n_skipped}")
    # Several shapes have a finite solution set (A, G and ``narrow`` a single point),
    # so random samples almost never land on it; the hand-picked boundary points
    # carry the feasible arm and every one that satisfies the relation must have run.
    n_fixed_feasible = sum(residual(pt) <= 1e-9 for pt in fixed)
    assert n_feas >= max(1, n_fixed_feasible), (shape, method, n_feas)
    assert n_infeas >= 3, (shape, method, n_infeas)
