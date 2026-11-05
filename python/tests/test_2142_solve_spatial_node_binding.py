"""#2142 item 3: the kernel's node LP on a fixed box, exposed for offline experiments.

``_rust.solve_spatial_node_py`` runs ``solve_spatial_node_with_cuts`` (the tree's node
solve, no warm start, OBBT off) on one box, optionally with extra ``<=`` rows appended as
the cut block, and returns the Neumaier-Shcherbina safe bound the tree would use. It is
how the fixed-box tangent experiment (``docs/dev/performance-plan.md`` §106) re-solved the
kernel's own LP like-for-like. Instrumentation only: no solve path calls it, which the
replay test pins by requiring it to reproduce the tree's own node LPs.
"""

from __future__ import annotations

import itertools
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _rust
from discopt._relax.spatial_producer import build_spatial_kernel_spec

pytestmark = pytest.mark.smoke

_NO_CUTS = dict(
    cut_ptr=np.zeros(1, dtype=np.int64),
    cut_cols=np.zeros(0, dtype=np.int64),
    cut_coeffs=np.zeros(0),
    cut_rhs=np.zeros(0),
)


def _model_and_spec():
    m = dm.Model("node2142")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.binary("z")
    m.minimize(-x * y + 0.5 * x**2 - y + 2.0 * z)
    m.subject_to(x + y <= 5.0)
    m.subject_to(x * y >= 1.0)
    m.subject_to(y <= 1.0 + 3.0 * z)
    spec = build_spatial_kernel_spec(m)
    assert spec is not None, "the kernel must take this model"
    return {k: v for k, v in spec.items() if not k.startswith("meta_")}


def _solve(spec, lo, hi, cuts=_NO_CUTS):
    r = _rust.solve_spatial_node_py(**spec, lo=lo, hi=hi, **cuts)
    if r["status"] == "optimal" and not math.isfinite(r["bound"]):
        r = _rust.solve_spatial_node_py(**spec, lo=lo, hi=hi, **cuts, cold_dual_start=True)
    return r


def test_binding_reproduces_the_trees_node_lps():
    """On every observed node the binding's LP optimum equals the tree's own (the tree
    may warm-start, so the objective is compared, not the vertex)."""
    spec = _model_and_spec()
    recs = []
    _rust.solve_spatial_tree_py(**spec, max_nodes=40, time_limit_s=60.0, node_observer=recs.append)
    n_cmp = 0
    for r in recs:
        if r["status"] != "optimal":
            continue
        out = _solve(spec, r["lo"], r["hi"])
        assert out["status"] == "optimal"
        assert abs(out["obj"] - r["obj"]) <= 1e-7 * max(1.0, abs(r["obj"])), (out["obj"], r["obj"])
        assert out["bound"] <= out["obj"] + 1e-9
        n_cmp += 1
    assert n_cmp > 0, "no optimal node was observed: the replay compared nothing"


def test_tangent_cut_raises_the_bound_and_stays_valid():
    """The x**2 term's LP-point tangent is a globally valid cut: appending it never lowers
    the safe bound, and the bound stays below every feasible point of the box (a grid of
    exactly feasible points; the true optimum is at most their minimum)."""
    spec = _model_and_spec()
    (i,), (s,) = spec["term_i"], spec["term_out"]
    lo = np.asarray(spec["global_lo"], dtype=float).copy()
    hi = np.asarray(spec["global_hi"], dtype=float).copy()
    base = _solve(spec, lo, hi)
    assert base["status"] == "optimal" and math.isfinite(base["bound"])
    cuts = {"ptr": [0], "cols": [], "co": [], "rhs": []}
    r, n_added = base, 0
    for _ in range(10):
        xp = np.asarray(r["x"])
        x0 = float(xp[i])
        if xp[s] >= x0 * x0 - 1e-9:
            break
        cuts["cols"] += [i, s]
        cuts["co"] += [2.0 * x0, -1.0]
        cuts["rhs"].append(x0 * x0)
        cuts["ptr"].append(len(cuts["cols"]))
        n_added += 1
        r = _solve(
            spec,
            lo,
            hi,
            dict(
                cut_ptr=np.array(cuts["ptr"], dtype=np.int64),
                cut_cols=np.array(cuts["cols"], dtype=np.int64),
                cut_coeffs=np.array(cuts["co"]),
                cut_rhs=np.array(cuts["rhs"]),
            ),
        )
        assert r["status"] == "optimal" and r["bound"] >= base["bound"] - 1e-9
    assert n_added > 0, "the root LP point never violated x**2: the cut path was not exercised"
    feas = [
        -xv * yv + 0.5 * xv**2 - yv + 2.0 * zv
        for xv, yv, zv in itertools.product(np.linspace(0, 4, 81), np.linspace(0, 4, 81), (0, 1))
        if xv + yv <= 5.0 and xv * yv >= 1.0 and yv <= 1.0 + 3.0 * zv
    ]
    assert feas
    assert r["bound"] <= min(feas) + 1e-9


def test_binding_rejects_malformed_input():
    spec = _model_and_spec()
    lo = np.asarray(spec["global_lo"], dtype=float)
    hi = np.asarray(spec["global_hi"], dtype=float)
    with pytest.raises(ValueError, match="length n_cols"):
        _rust.solve_spatial_node_py(**spec, lo=lo[:2], hi=hi, **_NO_CUTS)
    with pytest.raises(ValueError, match="cut_ptr"):
        _rust.solve_spatial_node_py(**spec, lo=lo, hi=hi, **dict(_NO_CUTS, cut_rhs=np.zeros(1)))
    bad = dict(
        cut_ptr=np.array([0, 1], dtype=np.int64),
        cut_cols=np.array([99], dtype=np.int64),
        cut_coeffs=np.array([1.0]),
        cut_rhs=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="bad ptr or column"):
        _rust.solve_spatial_node_py(**spec, lo=lo, hi=hi, **bad)
