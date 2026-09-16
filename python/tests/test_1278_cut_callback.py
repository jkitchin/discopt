"""Node cut callback — component D of #1248, via #1278.

``lazy_constraints`` fires only at integer-feasible nodes, so a plugin with
tangent-plane cuts of a nonconvex surface — the CALPHAD case #1249 is about — had
nowhere to put them. ``cut_callback`` fires at **every** node the search
evaluates, spatial ones included, and is handed the node's BOX, which is what a
spatial cut needs to choose a linearization point.

Soundness is the whole design problem here, and it is the opposite of component
A's. A's envelopes are DERIVED from the lowering, so there is nothing to be
unsound; a user cut is an ASSERTION the solver cannot check. discopt applies its
cut pool at every node through ``_AugmentedEvaluator``, so every accepted cut is
global, and an invalid one removes the optimum from the whole tree — a false
``optimal``. Hence two guards, neither optional:

* ``CutResult(scope="local")`` is refused at construction. There is no
  subtree-scoped pool, and silently promoting a local cut to global IS the false
  certificate.
* Every cut is checked against the points this solve has already verified
  feasible, and a violator RAISES rather than being dropped — a caller whose cut
  was silently discarded believes it applied.

The gate is a filter, not a proof, and says so: ``solver_stats`` carries
``cut_validation/witness_checks`` and ``cut_validation/unvalidated`` so a gate
that tested nothing cannot read as a pass (CLAUDE.md §6).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.callbacks import CutResult, CutValidationError, NodeCutContext

pytestmark = [pytest.mark.smoke]


def _hyperbola_model():
    """``min -(x+y)`` over ``{x, y in [0,3], x*y <= 1}``.

    The feasible set lies under a hyperbola, so McCormick over ``[0,3]^2`` is
    loose and the relaxation lets ``x + y`` run well past its true maximum of
    ``10/3`` (at ``x=3, y=1/3``). That makes ``x + y <= 10/3`` a valid global cut
    that the relaxation actually violates — the shape a real cut callback has.
    """
    m = dm.Model("hyperbola")
    x = m.continuous("x", lb=0.0, ub=3.0)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.subject_to(x * y <= 1.0)
    m.minimize(-(x + y))
    return m, x, y


TRUE_OPT = -(3.0 + 1.0 / 3.0)


def _spatial_model():
    """A model the search actually branches on: 39 nodes, 22 callback invocations
    over 12 distinct boxes (measured). The hyperbola model above closes in one
    batch of one node, so "fires at every node" cannot be tested on it."""
    m = dm.Model("spatial")
    v = [m.continuous(f"v{i}", lb=0.1, ub=3.0) for i in range(3)]
    m.subject_to(v[0] + v[1] + v[2] == 4.0)
    m.minimize(v[0] * v[1] * v[2] - dm.exp(v[0]) - dm.sin(3 * v[1]))
    return m, v


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #
def test_the_callback_fires_at_spatial_nodes_with_each_node_s_own_box():
    """``lazy_constraints`` fires only at integer-feasible nodes; this one fires
    at spatial nodes, which is the whole reason component D exists. The boxes must
    be genuinely different and genuinely inside the root's — a callback that only
    ever saw the root would pass a weaker assertion."""
    seen: list[NodeCutContext] = []
    m, _v = _spatial_model()
    r = m.solve(time_limit=90, cut_callback=lambda ctx, model: seen.append(ctx) or [])
    assert r.status == "optimal", r.status
    assert len(seen) >= 5, f"only {len(seen)} invocations — this is not a spatial test"

    root_lb = np.array([0.1, 0.1, 0.1])
    root_ub = np.array([3.0, 3.0, 3.0])
    boxes = set()
    strict_subboxes = 0
    for ctx in seen:
        assert ctx.node_lb.shape == ctx.node_ub.shape == (3,), ctx.node_lb.shape
        assert np.all(ctx.node_ub >= ctx.node_lb)
        assert np.all(ctx.node_lb >= root_lb - 1e-9)
        assert np.all(ctx.node_ub <= root_ub + 1e-9)
        assert ctx.x_relaxation.shape == (3,)
        assert np.isfinite(ctx.node_bound)
        boxes.add((tuple(ctx.node_lb), tuple(ctx.node_ub)))
        if np.any(ctx.node_lb > root_lb + 1e-9) or np.any(ctx.node_ub < root_ub - 1e-9):
            strict_subboxes += 1
    assert len(boxes) >= 3, f"only {len(boxes)} distinct boxes — not a spatial walk"
    assert strict_subboxes >= 2, f"only {strict_subboxes} boxes were strict subboxes"


def test_an_accepted_cut_reaches_the_pool_the_node_relaxations_use():
    """The proof that acceptance is not just bookkeeping.

    A bound comparison is the obvious test and is brittle here: this engine
    already closes these models exactly at the root, so there is no headroom for a
    valid cut to show up in ``root_bound`` (measured: -3.3333333333333885 with and
    without the cut). What can be asserted without ambiguity is the structural
    fact — the cut lands in ``_cut_pool``, which ``_AugmentedEvaluator`` applies at
    every node, with the coefficients, sense and rhs the callback returned.
    """
    from discopt._relax.cutting_planes import CutPool

    m, x, y = _hyperbola_model()
    pooled: list = []
    original_add = CutPool.add

    def recording_add(self, cut):
        pooled.append((np.asarray(cut.coeffs).copy(), float(cut.rhs), cut.sense))
        return original_add(self, cut)

    CutPool.add = recording_add
    try:
        r = m.solve(
            time_limit=60,
            cut_callback=lambda ctx, model: [
                CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=10.0 / 3.0)
            ],
        )
    finally:
        CutPool.add = original_add

    assert r.status == "optimal", r.status
    mine = [p for p in pooled if np.allclose(p[0], [1.0, 1.0]) and p[2] == "<="]
    assert mine, f"the callback's cut never reached the pool; pooled={pooled}"
    assert mine[0][1] == pytest.approx(10.0 / 3.0)
    assert (r.solver_stats or {}).get("cut_validation/cuts", 0) >= 1


def test_a_valid_cut_does_not_move_the_optimum():
    m, x, y = _hyperbola_model()

    def cb(ctx, model):
        return [CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=10.0 / 3.0)]

    with_cut = m.solve(time_limit=60, cut_callback=cb)
    m2, _x2, _y2 = _hyperbola_model()
    without = m2.solve(time_limit=60)
    assert with_cut.status == without.status == "optimal"
    assert with_cut.objective == pytest.approx(TRUE_OPT, abs=1e-4)
    assert with_cut.objective == pytest.approx(without.objective, abs=1e-4)


def test_the_gate_reports_how_many_points_it_actually_tested():
    """A validation that checked nothing must not read as a pass."""
    m, x, y = _hyperbola_model()

    def cb(ctx, model):
        return [CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=10.0 / 3.0)]

    r = m.solve(time_limit=60, cut_callback=cb)
    stats = r.solver_stats or {}
    assert stats.get("cut_validation/cuts", 0) > 0, stats
    assert "cut_validation/witness_checks" in stats, stats


# --------------------------------------------------------------------------- #
# The guards — the point of the component
# --------------------------------------------------------------------------- #
def test_a_cut_that_excludes_a_known_feasible_point_aborts_the_solve():
    """The failure this gate exists for. ``x + y <= 1`` is violated by the true
    optimum ``(3, 1/3)``; accepting it would make the solve report a false
    ``optimal`` at a worse point."""
    m, x, y = _hyperbola_model()

    def bad(ctx, model):
        return [CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=1.0)]

    with pytest.raises(CutValidationError, match="verified FEASIBLE"):
        m.solve(time_limit=60, cut_callback=bad)


def test_the_gate_is_not_bypassed_by_returning_the_bad_cut_late():
    """A cut generated deep in the search is judged against every witness collected
    earlier, not only against the node that produced it."""
    m, v = _spatial_model()
    state = {"n": 0}

    def late(ctx, model):
        state["n"] += 1
        if state["n"] < 4:
            return []
        # Every feasible point has v0+v1+v2 == 4, so this excludes all of them.
        return [CutResult(terms=[(v[0], 1.0), (v[1], 1.0), (v[2], 1.0)], sense="<=", rhs=1.0)]

    with pytest.raises(CutValidationError, match="verified FEASIBLE"):
        m.solve(time_limit=90, cut_callback=late)
    assert state["n"] >= 4, state


def test_a_local_scope_cut_is_refused_at_construction():
    """There is no subtree-scoped pool; treating a local cut as global is exactly
    the false-certificate case."""
    with pytest.raises(ValueError, match="scope='local'"):
        CutResult(terms=[], sense="<=", rhs=1.0, scope="local")


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="Invalid cut scope"):
        CutResult(terms=[], sense="<=", rhs=1.0, scope="subtree")


def test_the_default_scope_is_global():
    assert CutResult(terms=[], sense="<=", rhs=1.0).scope == "global"


def test_a_callback_that_raises_is_logged_and_the_solve_continues():
    """User code may fail softly; our own code around it may not (INT-1, #413)."""
    m, _x, _y = _hyperbola_model()
    calls = {"n": 0}

    def boom(ctx, model):
        calls["n"] += 1
        raise RuntimeError("user bug")

    r = m.solve(time_limit=60, cut_callback=boom)
    assert calls["n"] > 0
    assert r.status == "optimal"
    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-4)


def test_a_callback_returning_nothing_leaves_the_solve_byte_identical():
    """A cut callback that never cuts must not perturb the search — otherwise its
    presence alone would be a confound in any measurement using it."""
    m, _x, _y = _hyperbola_model()
    baseline = m.solve(time_limit=60)

    m2, _x2, _y2 = _hyperbola_model()
    inert = m2.solve(time_limit=60, cut_callback=lambda ctx, model: [])

    assert inert.status == baseline.status
    assert inert.objective == pytest.approx(baseline.objective, rel=1e-12, abs=1e-12)
    assert inert.node_count == baseline.node_count


def test_the_validator_is_callable_on_its_own():
    """It is public so a plugin can gate its own cuts before returning them."""
    from discopt.callbacks import validate_cut_against_witnesses

    m, x, y = _hyperbola_model()
    good = CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=10.0 / 3.0)
    bad = CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=1.0)
    witnesses = [np.array([3.0, 1.0 / 3.0]), np.array([1.0, 1.0])]

    n, worst = validate_cut_against_witnesses(good, m, witnesses)
    assert n == 2 and worst <= 1e-9, (n, worst)

    with pytest.raises(CutValidationError):
        validate_cut_against_witnesses(bad, m, witnesses)

    # No witnesses -> nothing checked. The count is the caller's signal that the
    # gate was vacuous; it must not silently read as a pass.
    n0, _ = validate_cut_against_witnesses(bad, m, [])
    assert n0 == 0


def test_the_retroactive_recheck_convicts_a_cut_accepted_with_no_evidence():
    """Raised in review on #1275: the FIRST cuts can arrive before any witness
    exists, so they are accepted unchecked, and the region a bad cut excludes can
    never afterwards produce a witness to convict it.

    Nothing recovers the excluded region, but such a cut can still be convicted by
    a witness found elsewhere, so every new witness is re-checked against every cut
    already accepted. This exercises that path directly — a solve-level version of
    this test passed both with and WITHOUT the re-check, because on every model
    tried the incumbent is recorded before the first cut is validated (measured:
    zero validations with an empty witness set over two models and 23
    validations). A test that cannot fail is not a test (CLAUDE.md §6).
    """
    from discopt.solver import _recheck_accepted_cuts

    m, v = _spatial_model()
    bad = CutResult(terms=[(v[0], 1.0), (v[1], 1.0), (v[2], 1.0)], sense="<=", rhs=1.0)
    good = CutResult(terms=[(v[0], 1.0), (v[1], 1.0), (v[2], 1.0)], sense="<=", rhs=10.0)
    witness = np.array([1.0, 1.5, 1.5])  # feasible: sums to 4

    tally: dict = {}
    _recheck_accepted_cuts([good], m, witness, tally)
    assert tally["retro_checks"] == 1, tally

    with pytest.raises(CutValidationError, match="verified FEASIBLE"):
        _recheck_accepted_cuts([good, bad], m, witness, tally)


def test_the_gate_is_not_vacuous_on_a_normal_solve():
    """The counterpart measurement: on an ordinary solve every cut IS judged
    against at least one witness, so the hole above is narrow rather than the
    common case. Pinned because a change that stopped recording the incumbent as a
    witness would widen it silently."""
    m, v = _spatial_model()
    r = m.solve(
        time_limit=90,
        cut_callback=lambda ctx, model: [
            CutResult(terms=[(x, 1.0) for x in v], sense="<=", rhs=100.0)
        ],
    )
    stats = r.solver_stats or {}
    assert stats.get("cut_validation/cuts", 0) > 0, stats
    assert stats.get("cut_validation/unvalidated", 0) == 0, stats
    assert stats.get("cut_validation/witness_checks", 0) >= stats["cut_validation/cuts"], stats


def test_the_retroactive_recheck_does_not_fire_on_a_valid_cut():
    """The re-check must not cost a correct callback its solve."""
    m, x, y = _hyperbola_model()
    r = m.solve(
        time_limit=60,
        cut_callback=lambda ctx, model: [
            CutResult(terms=[(x, 1.0), (y, 1.0)], sense="<=", rhs=10.0 / 3.0)
        ],
    )
    assert r.status == "optimal"
    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-4)
