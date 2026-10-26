"""#1995: the native kernel's cutoff propagation collapsed a node box to slivers.

``gdp.simplex`` lowers ``either_or([[x >= 4, y >= 1], [y >= 6, x >= 1]])`` (minimise
``x + y``) to 10 continuous columns. The seed incumbent is the optimum (5), so each
node's FBBT applies the objective cutoff ``x + y <= 5 + 1.5e-8`` against
``x >= 4, y >= 1`` and leaves ``x``/``y`` with widths ~2e-8. The McCormick LP over
that box ends ``Numerical``; the undecided arm branches a weight column and every
child inherits the same sliver, so the search ran ~60k nodes at 30 s without a
finite bound (the reported 4.9721 was the #933 root-relaxation fallback). The same
spec with ``run_propagation=False`` certifies at node 1. #1983 fixed this class for
the Python-side cutoff OBBT box only; this is the kernel's own per-node cutoff.

``DISCOPT_KERNEL_UNDECIDED_RETRY`` (``SpatialTreeConfig::undecided_unpropagated_retry``)
re-solves an undecided node over its pre-propagation box, which contains the node's
region, so that LP's verdict holds for the region.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import warnings

import discopt.transformations as dt
import numpy as np
import pytest
from discopt import _rust
from discopt._relax.spatial_producer import build_spatial_kernel_spec
from discopt.validation.feasibility import verify_point

sys.path.insert(0, os.path.dirname(__file__))
import serialize_builder_cases as cases  # noqa: E402

FLAG = "DISCOPT_KERNEL_UNDECIDED_RETRY"


def _lowered():
    return dt.create_using("gdp.simplex", cases._either_or("select_one")())


def _box_optimum(xl, xh, yl, yh):
    """min x + y over the source disjunction inside the box (closed form)."""
    best = math.inf
    for ax, ay in ((4.0, 1.0), (1.0, 6.0)):
        x, y = max(xl, ax), max(yl, ay)
        if x <= xh and y <= yh:
            best = min(best, x + y)
    return best


def _boxes():
    for d, (cx, cy) in itertools.product(
        [1e-3, 1e-2, 0.1, 1.0, 3.0], [(4.0, 1.0), (1.0, 6.0), (2.0, 3.0), (4.5, 1.5)]
    ):
        xl, xh = max(0.0, cx - d / 3), min(10.0, cx + d)
        yl, yh = max(0.0, cy - d / 3), min(10.0, cy + d)
        opt = _box_optimum(xl, xh, yl, yh)
        if math.isfinite(opt):
            yield (xl, xh, yl, yh), opt


def _root(model, box, incumbent, retry):
    xl, xh, yl, yh = box
    n_w = len(model._variables) - 2
    lb = np.array([xl, yl] + [0.0] * (2 * n_w))
    ub = np.array([xh, yh] + [1.0] * (2 * n_w))
    spec = build_spatial_kernel_spec(model, bounds=(lb, ub))
    assert spec is not None, "the kernel must take the lowered model"
    for k in [k for k in spec if k.startswith("meta_")]:
        spec.pop(k)
    return _rust.solve_spatial_tree_py(
        **spec,
        initial_incumbent=incumbent,
        max_nodes=1,
        time_limit_s=10.0,
        undecided_unpropagated_retry=retry,
    )


def test_the_lowered_either_or_certifies_with_the_retry(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _lowered().solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.node_count, r.bound)
    assert abs(r.objective - 5.0) <= 1e-6
    assert r.bound <= 5.0 + 1e-6
    stats = r.solver_stats or {}
    assert stats.get("tree/prop_retry_decided", 0) >= 1, stats  # the retry fired (§6)


def test_the_retry_is_off_by_default_and_reports_zero(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    from discopt.solver import _kernel_undecided_retry_enabled

    assert not _kernel_undecided_retry_enabled()
    monkeypatch.setenv(FLAG, "1")
    assert _kernel_undecided_retry_enabled()


def test_differential_bound_on_fixed_boxes():
    """ON bound >= OFF bound, and <= the true box optimum, at the root of fixed boxes
    with the incumbent at the box optimum (the trigger). Most roots must be undecided
    OFF, or the comparison would be vacuous."""
    model = _lowered()
    compared = undecided_off = 0
    for box, opt in _boxes():
        off = _root(model, box, opt + 1.5e-8, retry=False)
        on = _root(model, box, opt + 1.5e-8, retry=True)
        assert on["root_status"] != "undecided", (box, on["root_status"])
        assert on["bound"] >= off["bound"], (box, on["bound"], off["bound"])
        assert on["bound"] <= opt + 1e-6, (box, on["bound"], opt)
        undecided_off += off["root_status"] == "undecided"
        compared += 1
    assert compared >= 12
    assert undecided_off >= compared // 2, (undecided_off, compared)


def _lifted_point(model, x, y):
    """A feasible point of the lowered model over the source point ``(x, y)``: each
    clause's weight is a unit vector on one of its satisfied literals."""
    literal = {"4 - x": 4 - x, "6 - y": 6 - y, "1 - x": 1 - x, "1 - y": 1 - y}
    clauses = [("4 - x", "6 - y"), ("4 - x", "1 - x"), ("1 - y", "6 - y"), ("1 - y", "1 - x")]
    lam = []
    for a, b in clauses:
        assert literal[a] <= 1e-12 or literal[b] <= 1e-12
        lam += [1.0, 0.0] if literal[a] <= 1e-12 else [0.0, 1.0]
    return np.array([x, y] + lam)


def test_no_feasible_point_lies_below_the_retry_bound():
    """Feasible-point sampling: every sampled point of each box's feasible set,
    verified on the lowered model, has objective >= the ON root bound."""
    model = _lowered()
    rng = np.random.default_rng(1995)
    checked = 0
    for box, opt in _boxes():
        xl, xh, yl, yh = box
        bound = _root(model, box, opt + 1.5e-8, retry=True)["bound"]
        for _ in range(40):
            x, y = rng.uniform(xl, xh), rng.uniform(yl, yh)
            if not ((x >= 4 and y >= 1) or (y >= 6 and x >= 1)):
                continue
            point = _lifted_point(model, x, y)
            assert verify_point(model, point).ok
            assert x + y >= bound - 1e-9, (box, x, y, bound)
            checked += 1
        # the box optimum itself, the point a cut-off bound would exclude first
        for ax, ay in ((4.0, 1.0), (1.0, 6.0)):
            x, y = max(xl, ax), max(yl, ay)
            if x <= xh and y <= yh:
                assert verify_point(model, _lifted_point(model, x, y)).ok
                assert x + y >= bound - 1e-9, (box, x, y, bound)
                checked += 1
    assert checked >= 200, checked


@pytest.mark.parametrize("retry", [False, True])
def test_a_decided_root_is_unchanged_by_the_flag(retry):
    """Bound-neutral where the first LP decides: no incumbent, so no cutoff sliver."""
    model = _lowered()
    box = (0.0, 10.0, 0.0, 10.0)
    ref = _root(model, box, None, retry=False)
    got = _root(model, box, None, retry=retry)
    assert ref["root_status"] == "bounded"
    assert got["bound"] == ref["bound"] and got["n_prop_retries"] == 0
