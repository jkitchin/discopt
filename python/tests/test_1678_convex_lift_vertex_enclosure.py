"""#1678 II.7: the hull root bound ignored the perspective OA the node LPs used.

``DISCOPT_PERSPECTIVE_OA`` (#1662) lifts a proven-convex GDP hull row
``yhat*g(v/yhat)`` and outer-approximates it, but ``_try_convex_lift`` sized
the lifted aux from the row's natural interval enclosure -- the dependency
product ``[eps, 1] * [0, v_ub/eps]``, about ``2e10`` -- and the #358
conditioning guard (``1e7``) declined the lift at the ROOT box, where ``y`` is
free. The root LP therefore carried no perspective OA and ``root_bound`` was the
McCormick value (-1898 on the synthesis witness) while every child, with ``y``
fixed, was lifted. Not a reporting bug: the root genuinely did not apply the cuts.

``DISCOPT_CONVEX_LIFT_VERTEX_ENCLOSURE`` gives such a node the enclosure its
certified curvature implies (vertex maximum + centre-tangent minimum for convex,
mirrored for concave, widened outward), so the root lifts and reaches the exact
continuous hull relaxation (-502.99).
"""

from __future__ import annotations

import discopt._relax.uniform_relax as ur
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.model_utils import flat_variable_bounds
from test_issue_1617_c12b_hull_perspective import (
    SYNTH_HULL_RELAXATION,
    SYNTH_OPT,
    build_synthesis_nl,
)

FLAG = "DISCOPT_CONVEX_LIFT_VERTEX_ENCLOSURE"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("DISCOPT_PERSPECTIVE_COMPOSITE", raising=False)
    monkeypatch.delenv("DISCOPT_PERSPECTIVE_OA", raising=False)


def test_flag_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    assert not ur.convex_lift_vertex_enclosure_enabled()
    monkeypatch.setenv(FLAG, "1")
    assert ur.convex_lift_vertex_enclosure_enabled()
    monkeypatch.delenv(FLAG)
    assert ur.convex_lift_vertex_enclosure_enabled()  # default ON since the panel


# ── the enclosure itself ────────────────────────────────────────────────────


def _convex(x):
    return (x[0] - 1.0) ** 2 + np.exp(x[1]) + x[0] * 0.5


def _convex_grad(x):
    g = np.zeros_like(x)
    g[0] = 2.0 * (x[0] - 1.0) + 0.5
    g[1] = np.exp(x[1])
    return g


@pytest.mark.parametrize("curv", ["convex", "concave"])
def test_enclosure_contains_every_sampled_value(curv):
    sign = 1.0 if curv == "convex" else -1.0
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        lo = rng.uniform(-3, 1, size=3)
        hi = lo + rng.uniform(0.01, 4, size=3)
        lo[2], hi[2] = 0.0, 0.0  # a column outside the support
        enc = ur._curvature_enclosure(
            lambda x: sign * _convex(x),
            lambda x: sign * _convex_grad(x),
            [0, 1],
            lo,
            hi,
            curv,
        )
        assert enc is not None
        pts = rng.uniform(lo, hi, size=(500, 3))
        vals = np.array([sign * _convex(p) for p in pts])
        assert np.all(vals >= enc[0]) and np.all(vals <= enc[1])
        checked += vals.size
        # The vertex side is the exact extremum (up to the outward margin).
        corners = [
            sign * _convex(np.array([a, b, 0.0])) for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
        ]
        exact = max(corners) if curv == "convex" else min(corners)
        side = enc[1] if curv == "convex" else enc[0]
        assert side == pytest.approx(exact, rel=1e-6, abs=1e-6)
    assert checked == 20 * 500


def test_enclosure_abstains():
    lo, hi = np.zeros(12), np.ones(12)
    big = list(range(ur._VERTEX_ENCLOSURE_MAX_VARS + 1))
    assert ur._curvature_enclosure(np.sum, np.ones_like, big, lo, hi, "convex") is None
    with np.errstate(divide="ignore"):  # log(0) at a vertex is not finite
        enc = ur._curvature_enclosure(
            lambda x: np.log(x[0]), lambda x: 1 / x, [0], lo, hi, "concave"
        )
    assert enc is None


# ── the hull row at the root box ────────────────────────────────────────────


def _root_relaxation(monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    h = dt.create_using("gdp.hull", build_synthesis_nl())
    lb, ub = flat_variable_bounds(h)
    return h, lb, ub, ur.build_uniform_relaxation(h, box=(lb, ub))


def test_root_box_lifts_the_hull_rows_only_with_the_flag(monkeypatch):
    *_, off = _root_relaxation(monkeypatch, "0")
    *_, on = _root_relaxation(monkeypatch, "1")
    assert not off.composite_multivar_specs
    assert len(on.composite_multivar_specs) == 3  # one per reactor disjunct


def test_refined_enclosure_contains_the_row_at_sampled_points(monkeypatch):
    """Feasible-point sampling: the enclosure that admitted each root lift contains
    the hull row's value everywhere in the box, and the aux box the LP carries
    does too."""
    got = []
    orig = ur._curvature_enclosure

    def record(f, grad_f, idxs, lo, hi, curvature):
        out = orig(f, grad_f, idxs, lo, hi, curvature)
        got.append((f, list(idxs), lo.copy(), hi.copy(), out))
        return out

    monkeypatch.setattr(ur, "_curvature_enclosure", record)
    _, lb, ub, rel = _root_relaxation(monkeypatch, "1")
    assert len(got) == 3
    rng = np.random.default_rng(1)
    checked = 0
    for (f, idx, lo, hi, enc), spec in zip(got, rel.composite_multivar_specs, strict=True):
        assert enc is not None
        assert max(abs(enc[0]), abs(enc[1])) < 1e4  # a true range, not 2e10
        lo_aux, hi_aux = rel.model._bounds[spec.aux_col]
        for _ in range(400):
            x = 0.5 * (lo + hi)
            x[idx] = rng.uniform(lo[idx], hi[idx])
            if rng.random() < 0.3:  # corners, including the eps end of yhat
                x[idx] = np.where(rng.random(len(idx)) < 0.5, lo[idx], hi[idx])
            v = float(np.asarray(f(x)).reshape(()))
            assert enc[0] <= v <= enc[1], (v, enc)
            assert lo_aux <= v <= hi_aux, (v, lo_aux, hi_aux)
            checked += 1
    assert checked == 3 * 400


def test_bound_tightening_relaxation_is_unchanged(monkeypatch):
    """OBBT/DBBT build under ``perspective_oa_suppressed`` and see the OFF build."""
    monkeypatch.setenv(FLAG, "1")
    h = dt.create_using("gdp.hull", build_synthesis_nl())
    lb, ub = flat_variable_bounds(h)
    fired = []
    orig = ur._curvature_enclosure
    monkeypatch.setattr(
        ur, "_curvature_enclosure", lambda *a, **k: fired.append(1) or orig(*a, **k)
    )
    with ur.perspective_oa_suppressed():
        rel = ur.build_uniform_relaxation(h, box=(lb, ub))
    assert not fired and not rel.composite_multivar_specs


# ── the solve ───────────────────────────────────────────────────────────────


def _solve(monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    return build_synthesis_nl().solve(gdp_method="hull", time_limit=120)


def test_root_bound_reaches_the_hull_relaxation(monkeypatch):
    """Differential bound test: ON root >= OFF root, never past the optimum, and at
    the exact continuous hull relaxation (min sense)."""
    off = _solve(monkeypatch, "0")
    on = _solve(monkeypatch, "1")
    for r in (off, on):
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(SYNTH_OPT, abs=1e-4)
    tol = 1e-6 * (1 + abs(SYNTH_OPT))
    assert off.root_bound < -1800  # the McCormick root the issue reported
    assert on.root_bound >= off.root_bound
    assert on.root_bound <= SYNTH_OPT + tol
    assert on.root_bound == pytest.approx(SYNTH_HULL_RELAXATION, abs=0.05)
    assert on.bound <= on.objective + tol
    assert on.node_count <= off.node_count
