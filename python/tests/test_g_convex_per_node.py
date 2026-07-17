"""End-to-end soundness + benefit tests for per-node G-convexity cuts (#181).

The per-node separator (`MccormickLPRelaxer._separate_g_convex`) adds box-local
transformation cuts at B&B nodes whose tightened box certifies a constraint
body G-convex. The non-negotiable property is **oracle soundness**: turning the
flag on must never change a certified optimum (a box-local cut wrongly reused
on a sub-box would cause a false optimum — the C-43/nvs22 hazard). These tests
solve branching G-convex models flag-OFF vs flag-ON and assert the certified
objective is identical, and that the node LP bound only tightens.

Marked slow (they run full B&B solves).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.slow


def _branching_g_convex_model():
    # log(x²+y²) is G-convex (not convex); the integer z + coupling force B&B,
    # so the box tightens across nodes and the detector fires deep in the tree.
    m = Model("t")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    z = m.integer("z", lb=0, ub=2)
    m.subject_to(dm.log(x**2 + y**2) - 1.6 <= 0)
    m.subject_to(z <= x)
    m.subject_to(x + y + z <= 5)
    m.maximize(x + y + z)
    return m


def _solve(flag, monkeypatch, model_fn, tl=30):
    monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", flag)
    r = model_fn().solve(time_limit=tl)
    return r


class TestPerNodeSoundness:
    def test_flag_preserves_certified_optimum(self, monkeypatch):
        off = _solve("0", monkeypatch, _branching_g_convex_model)
        on = _solve("1", monkeypatch, _branching_g_convex_model)
        assert off.status == "optimal" and on.status == "optimal"
        assert off.objective is not None and on.objective is not None
        # Soundness: identical certified objective (no false optimum).
        assert abs(float(on.objective) - float(off.objective)) <= 1e-6 * (
            1 + abs(float(off.objective))
        )

    def test_flag_does_not_increase_nodes(self, monkeypatch):
        # Where the cut fires it should help (or be neutral) — never blow up the
        # tree. This is a soft benefit check, not a soundness gate.
        off = _solve("0", monkeypatch, _branching_g_convex_model)
        on = _solve("1", monkeypatch, _branching_g_convex_model)
        no = getattr(off, "node_count", None)
        nn = getattr(on, "node_count", None)
        if no is not None and nn is not None:
            assert nn <= no + 1  # allow tie / off-by-one, forbid regression


class TestSeparatorTightening:
    def test_node_bound_tightens_on_certifying_box(self, monkeypatch):
        # On a tight sub-box that certifies G-convex, the flag-ON node LP bound
        # must be >= the flag-OFF bound (a valid tightening) and both must solve.
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        y = m.continuous("y", lb=1.0, ub=2.0)
        m.subject_to(dm.log(x**2 + y**2) - 1.6 <= 0)
        m.minimize(x + y)
        # A tight sub-box where the detector fires.
        lb = np.array([1.40, 1.60])
        ub = np.array([1.50, 1.70])

        monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", "0")
        off = MccormickLPRelaxer(m).solve_at_node(lb, ub)
        monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", "1")
        on = MccormickLPRelaxer(m).solve_at_node(lb, ub)

        assert off.status == "optimal" and on.status == "optimal"
        if off.lower_bound is not None and on.lower_bound is not None:
            # ON tightens (>=) or ties; never loosens.
            assert float(on.lower_bound) >= float(off.lower_bound) - 1e-6
