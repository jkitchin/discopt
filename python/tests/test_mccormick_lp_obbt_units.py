"""Coverage units for the McCormick LP relaxer and OBBT (#87).

Property-driven tests for the under-covered pockets of
``discopt._jax.mccormick_lp`` (separator loops, cut-pool inheritance /
remapping, incremental infeasibility re-verification, integer-ratio
partition combining, oversize decline) and ``discopt._jax.obbt``
(``run_obbt`` cutoff / conditioning branches, ``run_obbt_on_relaxation``
top-k scoring, DBBT, equality-propagation and LP-bootstrap finitization,
root OBBT with cutoff, reverse-FBBT integer rounding, and the linear
extractors).

Every bound claim is validated against dense sampling of the model over the
node box (a relaxer's lower bound must never exceed the sampled feasible
minimum); every tightening claim is validated by feasible-point retention
(a tightened box must keep every sampled feasible point and only ever
shrink). No "it ran" tests.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import (
    MccormickLPRelaxer,
    _pool_has_rows,
    _remap_pool_rows,
    build_milp_relaxation,
    column_identities,
)
from discopt.modeling.core import Model

SOUND_TOL = 1e-6


# ─────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────


def _sample_box(lb, ub, n=4000, seed=0):
    """Random interior points plus all box corners."""
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lb, ub, size=(n, lb.size))
    corners = np.array(list(itertools.product(*zip(lb, ub))))
    return np.vstack([pts, corners])


def _sampled_feasible_min(obj, feas, lb, ub, n=4000, seed=0):
    """Min of ``obj`` over sampled feasible points; asserts some point is feasible."""
    vals = [obj(p) for p in _sample_box(lb, ub, n, seed) if feas(p)]
    assert vals, "sampling found no feasible point — bad test setup"
    return min(vals)


def _assert_no_feasible_point(feas, lb, ub, n=4000, seed=0):
    assert not any(feas(p) for p in _sample_box(lb, ub, n, seed))


def _assert_retention(feas, lb, ub, new_lb, new_ub, n=4000, seed=0, tol=1e-6):
    """Every sampled feasible point of the ORIGINAL model stays inside the box."""
    kept = 0
    for p in _sample_box(lb, ub, n, seed):
        if feas(p):
            kept += 1
            assert np.all(p >= np.asarray(new_lb) - tol), (p, new_lb)
            assert np.all(p <= np.asarray(new_ub) + tol), (p, new_ub)
    assert kept > 0, "sampling found no feasible point — bad test setup"


# ─────────────────────────────────────────────────────────────
# Tiny models
# ─────────────────────────────────────────────────────────────


def _span_bilinear() -> Model:
    """min x*y s.t. x + y >= 1 — incremental fast path is in scope."""
    m = dm.Model("span")
    x = m.continuous("x", lb=-3, ub=4)
    y = m.continuous("y", lb=-2, ub=5)
    m.minimize(x * y)
    m.subject_to(x + y >= 1)
    return m


_SPAN_LB = np.array([-3.0, -2.0])
_SPAN_UB = np.array([4.0, 5.0])
_span_obj = lambda p: p[0] * p[1]  # noqa: E731
_span_feas = lambda p: p[0] + p[1] >= 1.0 - 1e-9  # noqa: E731


def _qcqp() -> Model:
    """Dense continuous QCQP: squares + bilinear (PSD/square separators fire)."""
    m = dm.Model("qcqp")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(x * x + y * y + x * y - 3 * x - 2 * y)
    m.subject_to(x + y >= 1)
    return m


_QCQP_LB = np.zeros(2)
_QCQP_UB = np.full(2, 3.0)
_qcqp_obj = lambda p: p[0] ** 2 + p[1] ** 2 + p[0] * p[1] - 3 * p[0] - 2 * p[1]  # noqa: E731
_qcqp_feas = lambda p: p[0] + p[1] >= 1.0 - 1e-9  # noqa: E731


def _constraint_bilinear(x_hi=2.0) -> Model:
    """min x + y s.t. x*y >= 2 — relaxation empty over [0,1]^2, feasible on [0,2]^2."""
    m = dm.Model("cbil")
    x = m.continuous("x", lb=0, ub=x_hi)
    y = m.continuous("y", lb=0, ub=x_hi)
    m.minimize(x + y)
    m.subject_to(x * y >= 2)
    return m


# ═════════════════════════════════════════════════════════════
# McCormick LP relaxer: separator soundness
# ═════════════════════════════════════════════════════════════


@pytest.mark.relaxation
class TestSeparatorSoundness:
    """Each separator's node LOWER BOUND must never exceed the sampled feasible
    minimum, and separating must never loosen the non-separated bound."""

    def test_multilinear_hull_separation_sound(self):
        # 5 distinct factors > DISCOPT_MULTILINEAR_RLT_MAX (4): the on-demand
        # multilinear envelope separator fires (loose recursive chain otherwise).
        m = dm.Model("p5")
        xs = [m.continuous(f"x{i}", lb=-1, ub=2) for i in range(5)]
        prod = xs[0]
        for h in xs[1:]:
            prod = prod * h
        m.minimize(prod)
        lb, ub = np.full(5, -1.0), np.full(5, 2.0)
        r = MccormickLPRelaxer(m)
        res = r.solve_at_node(lb, ub, out_cuts=[])  # cold, separating path
        assert res.status == "optimal" and res.lower_bound is not None
        smin = _sampled_feasible_min(lambda p: np.prod(p), lambda p: True, lb, ub)
        assert res.lower_bound <= smin + SOUND_TOL
        # The separator actually ran (per-family timer is pure instrumentation).
        assert r._sep_timers["multilinear"] > 0.0

    def test_multilinear_separation_never_loosens(self):
        m = dm.Model("p5b")
        xs = [m.continuous(f"x{i}", lb=-1, ub=2) for i in range(5)]
        prod = xs[0]
        for h in xs[1:]:
            prod = prod * h
        m.minimize(prod)
        lb, ub = np.full(5, -1.0), np.full(5, 2.0)
        off = MccormickLPRelaxer(m).solve_at_node(lb, ub, separate=False)
        on = MccormickLPRelaxer(m).solve_at_node(lb, ub, out_cuts=[])
        assert off.status == "optimal" and on.status == "optimal"
        assert on.lower_bound >= off.lower_bound - SOUND_TOL

    def test_multilinear_cut_emission_on_interior_point(self):
        # The sum == 2.5 hyperplane parks the LP point in the box interior where
        # the recursive-chain relaxation is strictly looser than the multilinear
        # hull, so the separator EMITS violated envelope cuts and the bound
        # measurably tightens — while staying below the sampled feasible min.
        def build():
            m = dm.Model("p5c")
            xs = [m.continuous(f"x{i}", lb=-1, ub=2) for i in range(5)]
            prod = xs[0]
            for h in xs[1:]:
                prod = prod * h
            m.minimize(prod)
            s = xs[0]
            for h in xs[1:]:
                s = s + h
            m.subject_to(s == 2.5)
            return m

        lb, ub = np.full(5, -1.0), np.full(5, 2.0)
        off = MccormickLPRelaxer(build()).solve_at_node(lb, ub, separate=False)
        on = MccormickLPRelaxer(build()).solve_at_node(lb, ub, out_cuts=[])
        assert off.status == "optimal" and on.status == "optimal"
        assert on.lower_bound >= off.lower_bound - SOUND_TOL
        # Sample the sum == 2.5 slice: draw 4 coords, solve the 5th, keep in-box.
        rng = np.random.default_rng(5)
        vals = []
        while len(vals) < 500:
            p4 = rng.uniform(-1.0, 2.0, size=4)
            p5 = 2.5 - p4.sum()
            if -1.0 <= p5 <= 2.0:
                vals.append(np.prod(p4) * p5)
        assert on.lower_bound <= min(vals) + SOUND_TOL

    def test_rlt_cuts_bound_sound_and_tightening(self):
        # rlt_cuts=True: targeted constraint-factor x bound-factor separation.
        # With x^2 / y^2 / x*y all lifted, the linearized products land on real
        # columns, so cuts are actually emitted (RLT tightens vs no-RLT).
        def build():
            m = dm.Model("rlt")
            x = m.continuous("x", lb=0, ub=3)
            y = m.continuous("y", lb=0, ub=3)
            m.minimize(x * x + y * y - 3 * (x * y))
            m.subject_to(x + y <= 4)
            return m

        lb, ub = np.zeros(2), np.full(2, 3.0)
        r = MccormickLPRelaxer(build(), rlt_cuts=True)
        res = r.solve_at_node(lb, ub, out_cuts=[])
        no_rlt = MccormickLPRelaxer(build()).solve_at_node(lb, ub, out_cuts=[])
        assert res.status == "optimal" and res.lower_bound is not None
        smin = _sampled_feasible_min(
            lambda p: p[0] ** 2 + p[1] ** 2 - 3 * p[0] * p[1],
            lambda p: p[0] + p[1] <= 4.0 + 1e-9,
            lb,
            ub,
        )
        assert res.lower_bound <= smin + SOUND_TOL
        assert r._sep_timers["rlt"] > 0.0
        # RLT cuts only ever tighten the separated (non-RLT) bound.
        assert res.lower_bound >= no_rlt.lower_bound - SOUND_TOL

    def test_edge_concave_separation_sound(self):
        # All-negative square coefficients + full bilinear coupling on 3 vars:
        # term-wise McCormick gives -6 while the joint vertex hull gives -4, so
        # the separator must EMIT cuts that close the gap without crossing the
        # true minimum (-4 at the (2,2,0)-type vertices).
        def build():
            m = dm.Model("ec3")
            x = m.continuous("x", lb=0, ub=2)
            y = m.continuous("y", lb=0, ub=2)
            z = m.continuous("z", lb=0, ub=2)
            m.minimize(-(x * x) - (y * y) - (z * z) + x * y + y * z + x * z)
            return m

        lb, ub = np.zeros(3), np.full(3, 2.0)
        r = MccormickLPRelaxer(build())
        res = r.solve_at_node(lb, ub, out_cuts=[])
        off = MccormickLPRelaxer(build()).solve_at_node(lb, ub, separate=False)
        assert res.status == "optimal" and res.lower_bound is not None
        assert getattr(r, "_ec_blocks", None), "edge-concave block was not detected"
        smin = _sampled_feasible_min(
            lambda p: (
                -(p[0] ** 2) - p[1] ** 2 - p[2] ** 2 + p[0] * p[1] + p[1] * p[2] + p[0] * p[2]
            ),
            lambda p: True,
            lb,
            ub,
        )
        assert res.lower_bound <= smin + SOUND_TOL
        # The vertex-hull cuts strictly tighten the term-wise bound here.
        assert res.lower_bound > off.lower_bound + 0.5

    def test_psd_moment_cuts_sound_and_tightening(self):
        m = _qcqp()
        r = MccormickLPRelaxer(m, psd_cuts=True)
        pool: list = []
        res = r.solve_at_node(_QCQP_LB, _QCQP_UB, out_cuts=pool)
        assert res.status == "optimal" and res.lower_bound is not None
        smin = _sampled_feasible_min(_qcqp_obj, _qcqp_feas, _QCQP_LB, _QCQP_UB)
        assert res.lower_bound <= smin + SOUND_TOL
        # The separated chain produced a non-empty, identity-tagged pool chunk.
        assert pool and pool[0][0].shape[0] > 0 and pool[0][2] is not None
        # PSD separation only tightens vs the base (non-separated) relaxation.
        base = MccormickLPRelaxer(m).solve_at_node(_QCQP_LB, _QCQP_UB, separate=False)
        assert res.lower_bound >= base.lower_bound - SOUND_TOL

    def test_g_convex_box_local_cuts_sound(self, monkeypatch):
        # DISCOPT_G_CONVEX_CUTS: box-local transformation cuts on a certifying
        # sub-box (out_cuts must be None — they may never enter the root pool).
        monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", "1")

        def build():
            m = dm.Model("gcx")
            x = m.continuous("x", lb=1.0, ub=2.0)
            y = m.continuous("y", lb=1.0, ub=2.0)
            m.subject_to(dm.log(x**2 + y**2) - 1.6 <= 0)
            m.minimize(x + y)
            return m

        lb, ub = np.array([1.40, 1.60]), np.array([1.50, 1.70])
        feas = lambda p: np.log(p[0] ** 2 + p[1] ** 2) <= 1.6 + 1e-9  # noqa: E731
        r_on = MccormickLPRelaxer(build())
        on = r_on.solve_at_node(lb, ub)
        monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", "0")
        off = MccormickLPRelaxer(build()).solve_at_node(lb, ub)
        assert on.status == "optimal" and off.status == "optimal"
        smin = _sampled_feasible_min(lambda p: p[0] + p[1], feas, lb, ub)
        assert on.lower_bound <= smin + SOUND_TOL
        # The separator ran and never loosens the flag-off bound.
        assert "gconvex" in r_on._sep_timers
        assert on.lower_bound >= off.lower_bound - SOUND_TOL


# ═════════════════════════════════════════════════════════════
# McCormick LP relaxer: incremental fast path + pool inheritance
# ═════════════════════════════════════════════════════════════


@pytest.mark.relaxation
class TestIncrementalAndPoolPaths:
    def test_incremental_declines_on_size_mismatch(self):
        r = MccormickLPRelaxer(_span_bilinear())
        assert r._inc is not None
        assert r._try_incremental_node(np.zeros(5), np.ones(5), None) is None

    def test_incremental_declines_on_near_unbounded_nonlinear_box(self):
        r = MccormickLPRelaxer(_span_bilinear())
        assert r._inc is not None
        assert r._try_incremental_node(np.array([-1e21, 0.0]), np.array([1e21, 1.0]), None) is None

    def test_untagged_valid_pool_row_is_appended_and_sound(self):
        # Legacy 2-tuple (untagged) pool: positional append on the fast path.
        # The row is the (redundant, globally valid over this box) McCormick
        # under-estimator w >= -2x - 3y - 6, so the bound must be unchanged.
        m = _span_bilinear()
        r = MccormickLPRelaxer(m)
        assert r._inc is not None and r._inc.ncol == 3
        valid = (np.array([[-2.0, -3.0, -1.0]]), np.array([6.0]))
        assert _pool_has_rows(valid)
        res = r.solve_at_node(_SPAN_LB, _SPAN_UB, inherited_cuts=valid)
        base = MccormickLPRelaxer(m).solve_at_node(_SPAN_LB, _SPAN_UB)
        assert res.status == "optimal"
        assert r._pool_stats["inherited_nodes"] == 1
        assert r._pool_stats["inherited_rows"] == 1
        smin = _sampled_feasible_min(_span_obj, _span_feas, _SPAN_LB, _SPAN_UB)
        assert res.lower_bound <= smin + SOUND_TOL
        # Redundant valid cut: identical bound (never looser, never above base+eps).
        assert abs(res.lower_bound - base.lower_bound) <= 1e-7 * (1 + abs(base.lower_bound))

    def test_c43_poisoned_pool_never_falsely_fathoms(self):
        # An INVALID pool row (x <= -10, empty on this box) flips the augmented
        # LP infeasible; the C-43 guard must recover the node pool-free instead
        # of trusting the false fathom.
        m = _span_bilinear()
        r = MccormickLPRelaxer(m)
        poison = (np.array([[1.0, 0.0, 0.0]]), np.array([-10.0]))
        res = r.solve_at_node(_SPAN_LB, _SPAN_UB, inherited_cuts=poison)
        assert res.status == "optimal", "poisoned pool must not fathom a feasible node"
        assert r._pool_stats["dropped_nodes"] >= 1
        smin = _sampled_feasible_min(_span_obj, _span_feas, _SPAN_LB, _SPAN_UB)
        assert res.lower_bound <= smin + SOUND_TOL

    def test_tagged_pool_inherits_on_fast_and_cold_paths(self):
        # Capture an identity-tagged root pool (PSD/square tangents), then replay
        # it through BOTH the incremental fast path and the cold build path.
        m = _qcqp()
        pool: list = []
        cap = MccormickLPRelaxer(m, psd_cuts=True).solve_at_node(_QCQP_LB, _QCQP_UB, out_cuts=pool)
        assert cap.status == "optimal" and pool
        chunk = pool[0]  # (A_rows, b_rows, idents)
        smin = _sampled_feasible_min(_qcqp_obj, _qcqp_feas, _QCQP_LB, _QCQP_UB)

        r_fast = MccormickLPRelaxer(m, psd_cuts=True)
        assert r_fast._inc is not None
        res_fast = r_fast.solve_at_node(_QCQP_LB, _QCQP_UB, inherited_cuts=chunk)
        assert res_fast.status == "optimal"
        assert r_fast._pool_stats["inherited_nodes"] >= 1
        assert res_fast.lower_bound <= smin + SOUND_TOL

        r_cold = MccormickLPRelaxer(m, psd_cuts=True, build_incremental=False)
        res_cold = r_cold.solve_at_node(_QCQP_LB, _QCQP_UB, inherited_cuts=chunk, separate=False)
        assert res_cold.status == "optimal"
        assert r_cold._pool_stats["inherited_nodes"] >= 1
        assert res_cold.lower_bound <= smin + SOUND_TOL
        # Inheriting valid rows never loosens the base (no-pool, no-sep) bound.
        base = MccormickLPRelaxer(m, build_incremental=False).solve_at_node(
            _QCQP_LB, _QCQP_UB, separate=False
        )
        assert res_cold.lower_bound >= base.lower_bound - SOUND_TOL

    def test_infeasible_node_fathomed_only_when_genuinely_empty(self):
        # x*y >= 2 over [0,1]^2 is empty (max product is 1): a rigorous fathom.
        m = _constraint_bilinear()
        r = MccormickLPRelaxer(m, build_incremental=False)
        res = r.solve_at_node(np.zeros(2), np.ones(2))
        assert res.status == "infeasible"
        _assert_no_feasible_point(lambda p: p[0] * p[1] >= 2.0 - 1e-9, np.zeros(2), np.ones(2))

    def test_warm_basis_then_infeasible_sibling_is_rigorous(self):
        # Solve a feasible node first (caches a warm basis), then an infeasible
        # sibling with the same row count: whatever internal path resolves it
        # (Farkas fast fathom, warm-basis cold re-solve, or equilibration
        # re-verify), the verdict must be a rigorous fathom of a genuinely
        # empty box — never a bound, never a false optimal.
        m = _span_bilinear()
        r = MccormickLPRelaxer(m)
        assert r._inc is not None
        first = r.solve_at_node(_SPAN_LB, _SPAN_UB)
        assert first.status == "optimal"
        assert r._inc_warm_basis is not None  # the warm basis is cached
        lb_i, ub_i = np.array([-3.0, -2.0]), np.array([-2.0, -1.0])
        _assert_no_feasible_point(_span_feas, lb_i, ub_i)
        second = r.solve_at_node(lb_i, ub_i)
        assert second.status == "infeasible"
        assert second.lower_bound is None

    def test_reverify_incremental_infeasible_direct(self):
        # Directly exercise the equilibration re-verify on a genuinely empty and
        # a genuinely feasible assembled system.
        m = _span_bilinear()
        r = MccormickLPRelaxer(m)
        inc = r._inc
        assert inc is not None
        # Empty: x+y >= 1 cannot hold on x in [-3,-2], y in [-2,-1].
        lb_i, ub_i = np.array([-3.0, -2.0]), np.array([-2.0, -1.0])
        _assert_no_feasible_point(_span_feas, lb_i, ub_i)
        A, b, bounds = inc.assemble(lb_i, ub_i)
        rv = r._reverify_incremental_infeasible(inc, A, b, bounds)
        # Either a Farkas-certified fathom or an abstention — never "optimal".
        assert rv is None or rv.status == "infeasible"
        # Feasible: the root box. Must never return a false "infeasible", and an
        # "optimal" result must carry a sound bound.
        A2, b2, bounds2 = inc.assemble(_SPAN_LB, _SPAN_UB)
        rv2 = r._reverify_incremental_infeasible(inc, A2, b2, bounds2)
        assert rv2 is None or rv2.status == "optimal"
        if rv2 is not None:
            smin = _sampled_feasible_min(_span_obj, _span_feas, _SPAN_LB, _SPAN_UB)
            assert rv2.lower_bound <= smin + SOUND_TOL

    def test_oversize_lift_declines_with_no_bound(self, monkeypatch):
        # Both the fast path and the cold path must decline (no bound claimed —
        # a decline is sound; a fabricated bound would not be).
        import discopt._jax.mccormick_lp as mlp

        monkeypatch.setattr(mlp, "_lp_lift_too_large", lambda *a, **k: True)
        fast = MccormickLPRelaxer(_span_bilinear()).solve_at_node(_SPAN_LB, _SPAN_UB)
        assert fast.status == "skipped_oversize" and fast.lower_bound is None
        cold = MccormickLPRelaxer(_span_bilinear(), build_incremental=False).solve_at_node(
            _SPAN_LB, _SPAN_UB
        )
        assert cold.status == "skipped_oversize" and cold.lower_bound is None

    def test_integer_ratio_partition_max_combines_soundly(self):
        # LP bound is loose here (-1) while the true feasible min (x == y) is 0,
        # so a valid partition bound strictly between them must be adopted.
        m = dm.Model("diag")
        x = m.continuous("x", lb=-1, ub=1)
        y = m.continuous("y", lb=-1, ub=1)
        m.minimize(x * y)
        m.subject_to(x - y == 0)
        lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        true_min = 0.0  # x == y  =>  x*y = x^2 >= 0

        class FakePartitioner:
            def __init__(self, v):
                self.v = v

            def node_bound(self, node_lb, node_ub, deadline=None):
                if isinstance(self.v, Exception):
                    raise self.v
                return self.v

        r = MccormickLPRelaxer(m)
        base = r.solve_at_node(lb, ub)
        assert base.status == "optimal" and base.lower_bound <= true_min + SOUND_TOL
        lifted_val = (base.lower_bound + true_min) / 2.0  # valid: <= true min
        assert lifted_val > base.lower_bound
        r.set_integer_ratio_partitioner(FakePartitioner(lifted_val))
        lifted = r.solve_at_node(lb, ub)
        assert abs(lifted.lower_bound - lifted_val) <= 1e-12
        assert lifted.lower_bound <= true_min + SOUND_TOL
        # Abstentions leave the LP bound untouched.
        r.set_integer_ratio_partitioner(FakePartitioner(None))
        assert r.solve_at_node(lb, ub).lower_bound == pytest.approx(base.lower_bound)
        r.set_integer_ratio_partitioner(FakePartitioner(RuntimeError("boom")))
        assert r.solve_at_node(lb, ub).lower_bound == pytest.approx(base.lower_bound)


# ═════════════════════════════════════════════════════════════
# Column identities and pool-row remapping (pure logic)
# ═════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestColumnIdentityRemap:
    def test_column_identities_structural_and_opaque(self):
        varmap = {
            "bilinear": {(0, 1): 2},
            "monomial": {(0, 2): 3},
            # square-of-a-lifted-aux: base 3 resolves to the monomial identity
            "univariate_square": {(3, 2): 4},
        }
        ident = column_identities(varmap, 6, 2)
        assert ident[0] == ("orig", 0) and ident[1] == ("orig", 1)
        assert ident[2] == ("bilinear", (0, 1))
        assert ident[3] == ("monomial", (0, 2))
        assert ident[4] == ("univariate_square", (("monomial", (0, 2)), 2))
        assert ident[5] == ("opaque", 5)

    def test_column_identities_ignores_out_of_range_and_nonint_base(self):
        # Out-of-range aux col is skipped; a non-integer square base is kept raw.
        varmap = {"univariate_square": {(0, 2): 99, ("z", 2): 1}}
        ident = column_identities(varmap, 2, 1)
        assert ident[0] == ("orig", 0)
        assert ident[1] == ("univariate_square", ("z", 2))

    def test_remap_moves_coefficients_by_identity(self):
        root = (("orig", 0), ("orig", 1), ("bilinear", (0, 1)))
        node = (("orig", 0), ("orig", 1), ("monomial", (0, 2)), ("bilinear", (0, 1)))
        a = np.array([[1.0, 0.0, 2.0]])
        b = np.array([5.0])
        A_rm, b_rm, kept, skipped = _remap_pool_rows(a, b, root, node, 4)
        assert kept == 1 and skipped == 0
        np.testing.assert_allclose(A_rm[0], [1.0, 0.0, 0.0, 2.0])
        assert b_rm[0] == 5.0
        # Semantics preserved: for any z consistent with the identity mapping the
        # remapped cut value equals the original cut value.
        rng = np.random.default_rng(3)
        for _ in range(50):
            z_node = rng.normal(size=4)
            z_root = np.array([z_node[0], z_node[1], z_node[3]])  # same identities
            assert abs(A_rm[0] @ z_node - a[0] @ z_root) < 1e-12

    def test_remap_skips_missing_opaque_and_out_of_range(self):
        root = (("orig", 0), ("monomial", (1, 2)), ("opaque", 2))
        node = (("orig", 0), ("bilinear", (0, 1)))
        rows = np.array(
            [
                [0.0, 1.0, 0.0],  # references monomial(1,2): absent at node -> skip
                [0.0, 0.0, 1.0],  # references an opaque root column -> skip
            ]
        )
        b = np.array([1.0, 2.0])
        A_rm, b_rm, kept, skipped = _remap_pool_rows(rows, b, root, node, 2)
        assert A_rm is None and b_rm is None and kept == 0 and skipped == 2
        # A row whose nonzero sits beyond the root identity map is also skipped.
        wide = np.zeros((1, 5))
        wide[0, 4] = 1.0
        A_rm2, _, kept2, skipped2 = _remap_pool_rows(wide, np.array([0.0]), root, node, 2)
        assert A_rm2 is None and kept2 == 0 and skipped2 == 1

    def test_remap_empty_pool_and_pool_has_rows(self):
        out = _remap_pool_rows(np.zeros((0, 3)), np.zeros(0), (), (), 3)
        assert out == (None, None, 0, 0)
        assert not _pool_has_rows(None)
        assert not _pool_has_rows((None, None))
        import scipy.sparse as sp

        assert _pool_has_rows((sp.csr_matrix(np.ones((1, 3))), np.ones(1)))
        assert not _pool_has_rows((sp.csr_matrix((0, 3)), np.zeros(0)))


# ═════════════════════════════════════════════════════════════
# OBBT: run_obbt option branches
# ═════════════════════════════════════════════════════════════


@pytest.mark.relaxation
class TestRunObbtBranches:
    def test_maximize_cutoff_retains_improving_points(self):
        # max a+b with incumbent 4: the cutoff row is a+b >= 4; every feasible
        # point with objective >= 4 must survive the tightening.
        from discopt._jax.obbt import run_obbt

        m = dm.Model("mx")
        a = m.continuous("a", lb=0, ub=10)
        b = m.continuous("b", lb=0, ub=10)
        m.maximize(a + b)
        m.subject_to(a + 2 * b <= 8)
        res = run_obbt(m, incumbent_cutoff=4.0)
        lb0, ub0 = np.zeros(2), np.full(2, 10.0)
        feas = lambda p: (p[0] + 2 * p[1] <= 8 + 1e-9) and (p[0] + p[1] >= 4 - 1e-9)  # noqa: E731
        _assert_retention(feas, lb0, ub0, res.tightened_lb, res.tightened_ub)
        # Only ever shrinks, and the cutoff genuinely tightened (a <= 8, b <= 4).
        assert np.all(res.tightened_lb >= lb0 - 1e-9)
        assert np.all(res.tightened_ub <= ub0 + 1e-9)
        assert res.tightened_ub[0] <= 8.0 + 1e-6
        assert res.tightened_ub[1] <= 4.0 + 1e-6

    def test_cutoff_is_only_row_when_constraints_nonlinear(self):
        # The bilinear constraint is skipped by the linear extractor, so the
        # cutoff row alone forms A_ub (the A_ub-is-None branch).
        from discopt._jax.obbt import run_obbt

        m = dm.Model("co")
        c = m.continuous("c", lb=0, ub=20)
        d = m.continuous("d", lb=0, ub=20)
        m.minimize(c + d)
        m.subject_to(c * d <= 25)
        res = run_obbt(m, incumbent_cutoff=5.0)
        assert res.tightened_ub[0] <= 5.0 + 1e-6
        assert res.tightened_ub[1] <= 5.0 + 1e-6
        feas = lambda p: (p[0] * p[1] <= 25 + 1e-9) and (p[0] + p[1] <= 5 + 1e-9)  # noqa: E731
        _assert_retention(feas, np.zeros(2), np.full(2, 20.0), res.tightened_lb, res.tightened_ub)

    def test_ill_conditioned_without_ns_abstains(self, monkeypatch):
        # Coefficients above _OBBT_COND_LIMIT with no dual (NS) oracle: OBBT must
        # return the box untouched (abstention is sound; a vertex bound is not).
        import discopt._jax.obbt as obbt_mod

        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: None)
        m = dm.Model("ill")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x + 1e12 * y <= 1e12)
        res = obbt_mod.run_obbt(m)
        assert res.n_lp_solves == 0 and res.n_tightened == 0
        np.testing.assert_allclose(res.tightened_lb, [0.0, 0.0])
        np.testing.assert_allclose(res.tightened_ub, [10.0, 10.0])

    def test_well_conditioned_without_ns_uses_raw_vertex(self, monkeypatch):
        # No dual oracle but a well-conditioned LP: the raw exact vertex is used
        # and the tightening stays sound (retention of the feasible region).
        import discopt._jax.obbt as obbt_mod

        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: None)
        m = dm.Model("wc")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)
        res = obbt_mod.run_obbt(m)
        assert np.isclose(res.tightened_ub[0], 10.0, atol=1e-6)
        feas = lambda p: p[0] + p[1] <= 10 + 1e-9  # noqa: E731
        _assert_retention(feas, np.zeros(2), np.full(2, 100.0), res.tightened_lb, res.tightened_ub)


# ═════════════════════════════════════════════════════════════
# OBBT: relaxation-level probes (top-k, cutoff, no-oracle, DBBT)
# ═════════════════════════════════════════════════════════════


def _bilinear_ge() -> Model:
    m = dm.Model("bge")
    u = m.continuous("u", lb=0.5, ub=4.0)
    v = m.continuous("v", lb=0.5, ub=4.0)
    m.minimize(u + v)
    m.subject_to(u * v >= 4.0)
    return m


_BGE_LB = np.array([0.5, 0.5])
_BGE_UB = np.array([4.0, 4.0])
_bge_feas = lambda p: p[0] * p[1] >= 4.0 - 1e-9  # noqa: E731


def _bge_relaxation():
    r = MccormickLPRelaxer(_bilinear_ge(), build_incremental=False)
    milp, varmap = build_milp_relaxation(
        r._model, r._terms, r._disc, bound_override=(_BGE_LB, _BGE_UB)
    )
    return milp, varmap


@pytest.mark.relaxation
class TestRelaxationObbtBranches:
    def test_no_exact_oracle_is_sound_noop(self, monkeypatch):
        import discopt._jax.obbt as obbt_mod

        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: None)
        monkeypatch.setattr(obbt_mod, "get_exact_lp_solver", lambda: None)
        milp, _ = _bge_relaxation()
        res = obbt_mod.run_obbt_on_relaxation(milp, 2)
        assert res.n_lp_solves == 0 and res.n_tightened == 0
        np.testing.assert_allclose(res.tightened_lb, _BGE_LB)
        np.testing.assert_allclose(res.tightened_ub, _BGE_UB)
        # run_obbt with no oracle at all is likewise a no-op over model bounds.
        m = dm.Model("noop")
        x = m.continuous("x", lb=0, ub=7)
        m.minimize(x)
        m.subject_to(x <= 3)
        r2 = obbt_mod.run_obbt(m)
        assert r2.n_lp_solves == 0
        np.testing.assert_allclose(r2.tightened_ub, [7.0])

    def test_require_ns_without_dual_oracle_abstains(self, monkeypatch):
        import discopt._jax.obbt as obbt_mod

        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: None)
        monkeypatch.setattr(obbt_mod, "_OBBT_COND_LIMIT", 1e-3)  # force require_ns
        milp, _ = _bge_relaxation()
        res = obbt_mod.run_obbt_on_relaxation(milp, 2)
        assert res.n_lp_solves == 0 and res.n_tightened == 0
        np.testing.assert_allclose(res.tightened_lb, _BGE_LB)
        np.testing.assert_allclose(res.tightened_ub, _BGE_UB)

    def test_cutoff_and_topk_tighten_soundly(self):
        from discopt._jax.obbt import run_obbt_on_relaxation

        milp, _ = _bge_relaxation()
        res = run_obbt_on_relaxation(milp, 2, incumbent_cutoff=4.5, top_k=1)
        # Shrink-only over the original columns.
        assert np.all(res.tightened_lb >= _BGE_LB - 1e-9)
        assert np.all(res.tightened_ub <= _BGE_UB + 1e-9)
        # Retention: every feasible point that beats the cutoff stays inside.
        feas = lambda p: _bge_feas(p) and (p[0] + p[1] <= 4.5 + 1e-9)  # noqa: E731
        _assert_retention(feas, _BGE_LB, _BGE_UB, res.tightened_lb, res.tightened_ub)
        # top_k=1 probes one candidate (2 LPs) plus at most one scoring LP.
        assert res.n_lp_solves <= 3

    def test_dbbt_reduced_cost_tightening_sound(self, monkeypatch):
        from discopt._jax.obbt import dbbt_on_relaxation

        milp, _ = _bge_relaxation()
        res = dbbt_on_relaxation(milp, 2, 4.5)
        assert res.n_lp_solves == 1
        assert np.all(res.tightened_lb >= _BGE_LB - 1e-9)
        assert np.all(res.tightened_ub <= _BGE_UB + 1e-9)
        feas = lambda p: _bge_feas(p) and (p[0] + p[1] <= 4.5 + 1e-9)  # noqa: E731
        _assert_retention(feas, _BGE_LB, _BGE_UB, res.tightened_lb, res.tightened_ub)
        # No finite cutoff -> sound no-op; no dual oracle -> sound no-op.
        assert dbbt_on_relaxation(milp, 2, float("inf")).n_lp_solves == 0
        import discopt._jax.obbt as obbt_mod

        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: None)
        assert obbt_mod.dbbt_on_relaxation(milp, 2, 4.5).n_lp_solves == 0

    def test_dbbt_tightens_from_nonzero_reduced_cost(self):
        # min x + y s.t. x >= 2 over [0,10]^2: at the LP optimum (2, 0) the
        # reduced cost of y is 1, so with cutoff 5 (gap 3) DBBT derives y <= 3.
        # Rigorous: every feasible point with objective <= 5 has y = obj - x <= 3.
        from types import SimpleNamespace

        from discopt._jax.obbt import dbbt_on_relaxation

        rel = SimpleNamespace(
            _bounds=[(0.0, 10.0), (0.0, 10.0)],
            _c=np.array([1.0, 1.0]),
            _A_ub=np.array([[-1.0, 0.0]]),  # -x <= -2  <=>  x >= 2
            _b_ub=np.array([-2.0]),
            _obj_offset=0.0,
        )
        res = dbbt_on_relaxation(rel, 2, 5.0)
        assert res.n_lp_solves == 1
        assert res.n_tightened >= 1
        assert res.tightened_ub[1] <= 3.0 + 1e-5
        feas = lambda p: (p[0] >= 2.0 - 1e-9) and (p[0] + p[1] <= 5.0 + 1e-9)  # noqa: E731
        _assert_retention(feas, [0.0, 0.0], [10.0, 10.0], res.tightened_lb, res.tightened_ub)


# ═════════════════════════════════════════════════════════════
# OBBT: finitization passes and root OBBT options
# ═════════════════════════════════════════════════════════════


@pytest.mark.relaxation
class TestFinitizationAndRootObbt:
    def test_propagate_equality_defined_bounds_finitizes_soundly(self):
        # v == x*y + 2 with x in [1,2], y in [1,3]: the true range of v is [3,8];
        # the pass must finitize v to an enclosing interval and touch nothing else.
        from discopt._jax.obbt import propagate_equality_defined_bounds

        m = dm.Model("prop")
        x = m.continuous("x", lb=1, ub=2)
        y = m.continuous("y", lb=1, ub=3)
        v = m.continuous("v", lb=-np.inf, ub=np.inf)
        m.minimize(x + y)
        m.subject_to(v == x * y + 2)
        lb = np.array([1.0, 1.0, -np.inf])
        ub = np.array([2.0, 3.0, np.inf])
        new_lb, new_ub, n_fin = propagate_equality_defined_bounds(m, lb, ub)
        assert n_fin >= 1
        assert np.isfinite(new_lb[2]) and np.isfinite(new_ub[2])
        np.testing.assert_allclose(new_lb[:2], [1.0, 1.0])
        np.testing.assert_allclose(new_ub[:2], [2.0, 3.0])
        # Every true point (x, y, x*y+2) stays inside the finitized box.
        for p in _sample_box([1.0, 1.0], [2.0, 3.0], n=500, seed=1):
            vv = p[0] * p[1] + 2.0
            assert new_lb[2] - 1e-9 <= vv <= new_ub[2] + 1e-9

    def test_bootstrap_finite_bounds_from_linear_rows(self):
        from discopt._jax.obbt import bootstrap_finite_bounds

        m = dm.Model("boot")
        a = m.continuous("a", lb=0, ub=np.inf)
        b = m.continuous("b", lb=0, ub=5)
        m.minimize(a)
        m.subject_to(a + b <= 10)
        lb = np.array([0.0, 0.0])
        ub = np.array([np.inf, 5.0])
        new_lb, new_ub, n_fin, _t = bootstrap_finite_bounds(m, lb, ub)
        assert n_fin == 1 and np.isfinite(new_ub[0])
        # Rigorous: every feasible a (i.e. a <= 10) is retained.
        assert new_ub[0] >= 10.0 - 1e-6
        # With an incumbent cutoff the finitized bound tightens further but must
        # keep every point with objective <= cutoff (a <= 3).
        cl, cu, n2, _ = bootstrap_finite_bounds(m, lb, ub, incumbent_cutoff=3.0)
        assert n2 == 1 and np.isfinite(cu[0])
        assert 3.0 - 1e-6 <= cu[0] <= 10.0
        assert cu[0] <= new_ub[0] + 1e-9

    def test_bootstrap_finitizes_open_lb_and_rounds_integers(self):
        # Open-below continuous (c >= -3) and open-above integer (k <= 7):
        # both directions finitize, the integer bound rounds inward, and the
        # true feasible ranges are fully retained.
        from discopt._jax.obbt import bootstrap_finite_bounds

        m = dm.Model("boot2")
        a = m.continuous("a", lb=0, ub=np.inf)
        c = m.continuous("c", lb=-np.inf, ub=5)
        k = m.integer("k", lb=0, ub=1000)
        m.minimize(a)
        m.subject_to(a + c <= 10)
        m.subject_to(c >= -3)
        m.subject_to(k <= 7)
        lb = np.array([0.0, -np.inf, 0.0])
        ub = np.array([np.inf, 5.0, np.inf])
        new_lb, new_ub, n_fin, _t = bootstrap_finite_bounds(m, lb, ub)
        assert n_fin == 3
        assert np.all(np.isfinite(new_lb)) and np.all(np.isfinite(new_ub))
        # Retention of the true feasible ranges: c in [-3, 5], a in [0, 13], k in {0..7}.
        assert new_lb[1] <= -3.0 <= 5.0 <= new_ub[1] + 1e-9
        assert new_ub[0] >= 13.0 - 1e-6
        assert new_ub[2] == 7.0  # integer bound rounded inward to an exact integer

    def test_root_obbt_with_cutoff_dbbt_and_min_improvement(self):
        from discopt._jax.obbt import obbt_tighten_root

        res = obbt_tighten_root(
            _bilinear_ge(),
            _BGE_LB.copy(),
            _BGE_UB.copy(),
            incumbent_cutoff=4.5,
            rounds=4,
            min_improvement=0.05,
        )
        assert not res.infeasible
        assert np.all(res.lb >= _BGE_LB - 1e-9) and np.all(res.ub <= _BGE_UB + 1e-9)
        assert res.n_tightened > 0
        feas = lambda p: _bge_feas(p) and (p[0] + p[1] <= 4.5 + 1e-9)  # noqa: E731
        _assert_retention(feas, _BGE_LB, _BGE_UB, res.lb, res.ub)
        # The optimum (2, 2) (objective 4 <= cutoff) must survive.
        assert res.lb[0] <= 2.0 <= res.ub[0]
        assert res.lb[1] <= 2.0 <= res.ub[1]

    def test_root_obbt_top_k_is_sound(self):
        from discopt._jax.obbt import obbt_tighten_root

        res = obbt_tighten_root(_bilinear_ge(), _BGE_LB.copy(), _BGE_UB.copy(), rounds=2, top_k=1)
        assert not res.infeasible
        assert np.all(res.lb >= _BGE_LB - 1e-9) and np.all(res.ub <= _BGE_UB + 1e-9)
        _assert_retention(_bge_feas, _BGE_LB, _BGE_UB, res.lb, res.ub)


# ═════════════════════════════════════════════════════════════
# OBBT: reverse FBBT integer rounding + linear extractors
# ═════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestReverseFbbtIntRounding:
    def test_monomial_root_bound_rounds_integer_inward(self):
        # w = x**2 with x integer in [0, 10] and w <= 5: x <= floor(sqrt(5)) = 2.
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0])
        ub = np.array([10.0])
        varmap = {"bilinear": {}, "monomial": {(0, 2): 1}}
        aux_lb = np.array([0.0, 0.0])
        aux_ub = np.array([0.0, 5.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap, is_int=np.array([True]))
        assert n >= 1
        assert ub[0] == 2.0
        # Retention: every integer x with x**2 <= 5 stays inside the box.
        for xv in (0, 1, 2):
            assert lb[0] - 1e-12 <= xv <= ub[0] + 1e-12


@pytest.mark.unit
class TestLinearExtractors:
    def test_constraint_extractor_edge_shapes(self):
        from discopt._jax.obbt import _extract_linear_constraints

        m = dm.Model("ex")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=1, ub=10)
        z = m.continuous("z", shape=(2,), lb=0, ub=10)
        m.minimize(x)
        m.subject_to((x + y) * 2 <= 6)  # composite * constant
        m.subject_to(-x <= -1)  # unary neg
        m.subject_to(z[0] + z[1] <= 3)  # indexed scalars
        m.subject_to(x / y <= 2)  # variable division -> skipped
        m.subject_to(dm.sin(x) <= 1)  # transcendental -> skipped
        m.subject_to(dm.sum(z) <= 12)  # sum over a vector variable -> skipped
        A_ub, b_ub, A_eq, b_eq, n_vars = _extract_linear_constraints(m)
        assert n_vars == 4
        assert A_ub is not None and A_ub.shape == (3, 4)  # only the linear rows
        np.testing.assert_allclose(A_ub[0], [2.0, 2.0, 0.0, 0.0])
        assert b_ub[0] == pytest.approx(6.0)
        np.testing.assert_allclose(A_ub[1], [-1.0, 0.0, 0.0, 0.0])
        assert b_ub[1] == pytest.approx(-1.0)
        np.testing.assert_allclose(A_ub[2], [0.0, 0.0, 1.0, 1.0])
        assert b_ub[2] == pytest.approx(3.0)
        assert A_eq is None and b_eq is None

    def test_objective_extractor_branches(self):
        from discopt._jax.obbt import _extract_linear_objective

        m = dm.Model("obj")
        z = m.continuous("z", shape=(2,), lb=0, ub=10)
        m.minimize(2 * z[0] - z[1] + 1)
        c = _extract_linear_objective(m, 2)
        assert c is not None
        np.testing.assert_allclose(c, [2.0, -1.0])

        m2 = dm.Model("objneg")
        x = m2.continuous("x", lb=0, ub=1)
        m2.minimize(-x)
        c2 = _extract_linear_objective(m2, 1)
        np.testing.assert_allclose(c2, [-1.0])

        m3 = dm.Model("objnl")
        a = m3.continuous("a", lb=0, ub=1)
        b = m3.continuous("b", lb=0, ub=1)
        m3.minimize(a * b)  # nonlinear -> None
        assert _extract_linear_objective(m3, 2) is None
