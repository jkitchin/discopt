"""#1662: the perspective-composite proof feeds OA cuts on the spatial path only.

The GDP hull writes a convex disjunct row as the eps-perspective
``yhat * g(v / yhat)``. ``DISCOPT_PERSPECTIVE_COMPOSITE`` (#1617 C-12b) proves
it convex, but the proof reaches ``classify_model`` and re-routes the whole model
to NLP-BB, which lost on the ``rsyn*`` panel, so that flag stays OFF.
``DISCOPT_PERSPECTIVE_OA`` (default ON; ``=0`` opts out) enables the same rule
only inside the spatial relaxation's convex-lift certificate
(``uniform_relax._Builder._dcp``): the row is lifted and outer-approximated by
gradient cuts, and the route is unchanged.

The canonical DAG reconstructs ``v / yhat`` as ``yhat**-1 * v``, so the rule's
ratio matcher accepts that spelling too.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.rules import (
    classify_expr,
    classify_model,
    perspective_composite_enabled,
    perspective_proof_scope,
)
from discopt._relax.uniform_relax import perspective_oa_enabled

FLAG = "DISCOPT_PERSPECTIVE_OA"


@pytest.fixture(autouse=True)
def _composite_off(monkeypatch):
    # The route-switching rule must stay OFF: these tests are about the scope.
    monkeypatch.delenv("DISCOPT_PERSPECTIVE_COMPOSITE", raising=False)


def build_witness():
    # The #1616 comment's witness (2026-10-05): big-M certifies in 5 nodes; the
    # hull stopped uncertified after 179 spatial nodes.
    m = dm.Model("hull_convex")
    x = m.continuous("x", lb=0, ub=10)
    p = m.continuous("p", lb=0, ub=10)
    Y = m.boolean("Y", shape=2)
    y = Y.variable
    m.either_or(
        [[p <= 2 * dm.log(1 + x), y[0] == 1], [p <= 3 * dm.log(1 + x / 2), y[1] == 1]],
        name="unit",
    )
    m.maximize(p - 0.3 * x)
    return m


WITNESS_OPT = 2.4283137


def test_flag_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    assert not perspective_oa_enabled()
    monkeypatch.setenv(FLAG, "1")
    assert perspective_oa_enabled()
    monkeypatch.setenv(FLAG, "")
    assert not perspective_oa_enabled()  # as DISCOPT_PERSPECTIVE_OA_CUT reads ""
    monkeypatch.delenv(FLAG)
    assert perspective_oa_enabled()  # default ON since the section 5 panel


def test_scope_is_local_and_restores():
    assert not perspective_composite_enabled()
    with perspective_proof_scope(True):
        assert perspective_composite_enabled()
        with perspective_proof_scope(False):
            assert not perspective_composite_enabled()
        assert perspective_composite_enabled()
    assert not perspective_composite_enabled()


def test_scope_restores_after_an_exception():
    with pytest.raises(RuntimeError):
        with perspective_proof_scope(True):
            raise RuntimeError("boom")
    assert not perspective_composite_enabled()


def _unit():
    m = dm.Model("u")
    xp = m.continuous("xp", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=1)
    w = m.continuous("w", lb=0, ub=1)
    L = (1 - 1e-8) * y + 1e-8
    return m, xp, y, w, L


@pytest.mark.parametrize(
    "build, want",
    [
        # The canonical spelling of the ratio: L**-1 * A, either factor order.
        (lambda xp, y, w, L: -dm.log(1 + L**-1 * xp) * L, Curvature.CONVEX),
        (lambda xp, y, w, L: dm.log(1 + xp * L**-1) * L, Curvature.CONCAVE),
        # The written spelling still works.
        (lambda xp, y, w, L: -dm.log(1 + xp / L) * L, Curvature.CONVEX),
        # A different power of L is not a ratio.
        (lambda xp, y, w, L: -dm.log(1 + L**-2 * xp) * L, Curvature.UNKNOWN),
        # The reciprocal of a different divisor is not a ratio over L.
        (lambda xp, y, w, L: -dm.log(1 + (y + 1) ** -1 * xp) * L, Curvature.UNKNOWN),
        # A variable outside every ratio.
        (lambda xp, y, w, L: -dm.log(1 + L**-1 * xp + w) * L, Curvature.UNKNOWN),
        # A bilinear product stays unproved.
        (lambda xp, y, w, L: xp * y, Curvature.UNKNOWN),
    ],
)
def test_reciprocal_spelling_of_the_ratio(build, want):
    m, xp, y, w, L = _unit()
    e = build(xp, y, w, L)
    with perspective_proof_scope(True):
        assert classify_expr(e, m) == want
    if want in (Curvature.CONVEX, Curvature.CONCAVE):
        assert classify_expr(e, m) == Curvature.UNKNOWN  # outside the scope


def test_classify_model_does_not_see_the_proof(monkeypatch):
    """The flag must not reach the route decision: same verdict and mask."""
    h = dt.create_using("gdp.hull", build_witness())
    monkeypatch.setenv(FLAG, "0")
    off = classify_model(h)
    monkeypatch.setenv(FLAG, "1")
    on = classify_model(h)
    assert not off[0]
    assert (bool(on[0]), list(on[1])) == (bool(off[0]), list(off[1]))


def _solve(monkeypatch, flag, method="hull"):
    monkeypatch.setenv(FLAG, flag)
    return build_witness().solve(gdp_method=method, time_limit=120)


def test_witness_certifies_on_the_spatial_path(monkeypatch):
    on = _solve(monkeypatch, "1")
    assert on.status == "optimal" and on.gap_certified
    assert on.objective == pytest.approx(WITNESS_OPT, abs=1e-5)
    # Maximisation: the dual bound sits at or above the optimum.
    assert on.bound >= on.objective - 1e-6
    route = str(on.algorithm_route)
    assert "spatial" in route and "nlp-bb" not in route, route


def test_witness_differential_against_big_m(monkeypatch):
    """ON never cuts the optimum: its bound stays on the right side of big-M's
    certified objective, and it is no looser than OFF at the root."""
    oracle = _solve(monkeypatch, "0", method="big-m")
    assert oracle.status == "optimal" and oracle.gap_certified
    off = _solve(monkeypatch, "0")
    on = _solve(monkeypatch, "1")
    tol = 1e-6 * (1 + abs(oracle.objective))
    assert on.objective == pytest.approx(oracle.objective, abs=1e-5)
    assert on.bound >= oracle.objective - tol
    assert on.root_bound is not None and off.root_bound is not None
    assert on.root_bound >= oracle.objective - tol
    assert on.root_bound <= off.root_bound + tol  # never looser (max sense)
    assert on.node_count <= off.node_count


def test_bound_tightening_sees_the_off_relaxation(monkeypatch):
    """OBBT/DBBT build their relaxation with the perspective lift suppressed.

    With the OA rows inside the OBBT relaxation, each OBBT LP was larger, the
    sweep's budget covered fewer of them and its candidate filtering changed: on
    ``casctanks`` x121..x128 stayed at ub 1.4 instead of 1.066 and the root bound
    came out looser than OFF (6.041 vs 6.068). The flag must change the node LP
    and nothing else. Checked in two halves: every build inside root OBBT runs
    suppressed, and suppression removes the lift at a box where the node
    relaxation takes it.
    """
    import discopt._relax.mccormick_lp as mlp
    import discopt._relax.obbt as obbt
    import discopt._relax.uniform_relax as ur

    in_obbt = [0]
    obbt_builds: list[bool] = []
    lifted_at: list[tuple] = []

    orig_obbt = obbt.obbt_tighten_root

    def count_obbt(*a, **k):
        in_obbt[0] += 1
        try:
            return orig_obbt(*a, **k)
        finally:
            in_obbt[0] -= 1

    orig_build = mlp.build_milp_relaxation

    def record_build(*a, **k):
        if in_obbt[0]:
            obbt_builds.append(ur._PERSPECTIVE_OA_SUPPRESSED.get())
        return orig_build(*a, **k)

    orig_lift = ur._Builder._try_convex_lift

    def record_lift(self, node):
        out = orig_lift(self, node)
        if out is not None and node.kind == "prod" and not lifted_at:
            k = self.n_orig
            lifted_at.append((self.model, np.array(self.col_lb[:k]), np.array(self.col_ub[:k])))
        return out

    monkeypatch.setattr(obbt, "obbt_tighten_root", count_obbt)
    monkeypatch.setattr(mlp, "build_milp_relaxation", record_build)
    monkeypatch.setattr(ur._Builder, "_try_convex_lift", record_lift)
    on = _solve(monkeypatch, "1")
    assert on.status == "optimal" and on.gap_certified
    assert obbt_builds, "root OBBT built no relaxation, so nothing was checked"
    assert all(obbt_builds), "a root OBBT relaxation was built with the lift live"

    assert lifted_at, "the node relaxation never took the perspective lift"
    model, lb, ub = lifted_at[0]
    live = ur.build_uniform_relaxation(model, box=(lb, ub))
    with ur.perspective_oa_suppressed():
        off = ur.build_uniform_relaxation(model, box=(lb, ub))
    assert live.composite_multivar_specs
    assert not off.composite_multivar_specs
