"""Soundness lock: a non-rigorous NLP fathom must never yield a certified OPTIMAL.

Companion to the C-1 test (``test_c1_nlp_fathom_infeasible.py``), which pins the
false-*infeasible* verdict. This file pins the *other* false verdict off the same
mechanism: false-*optimal*.

Mechanism (confirmed SOUND by construction, previously UNtested):

  * A node whose local NLP merely fails / diverges / returns a constraint-violating
    iterate is sentinelled with ``result_lbs[i] = INFEASIBILITY_SENTINEL`` (1e30)
    and *no* rigorous infeasibility proof (``node_infeasible_mask[i] == False``).
  * The Rust tree still mechanically cuts that node's subtree by bound
    (``1e30 >= incumbent`` -> fathom-by-bound, no children — ``tree_manager.rs``).
    Nothing in the subtree is proven suboptimal; the search is only *mechanically*
    closed, not *certifiably* closed.
  * The Python guard (``solver.py`` nonconvex finalize, serial + batch twins) sets
    ``_gap_certified = False`` for exactly this case (sentinel bound AND not
    rigorously infeasible). The terminal verdict is
    ``status = "optimal" if search_closed and _gap_certified else "feasible"``, so a
    non-rigorous fathom forces "feasible", never "optimal".

The guard is a one-line predicate now extracted to
``solver._nonrigorous_sentinel_fathom`` so the *decision* is directly unit
testable. ``test_guard_*`` pin that predicate (fail-before: stubbing it to
always-True mislabels a sound rigorous prune; always-False mislabels a
non-rigorous fathom). ``test_integration_*`` exercises the whole path: a feasible
nonconvex model with a real incumbent whose one remaining node is fathomed
non-rigorously must report "feasible", not "optimal".
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as S
import numpy as np
import pytest
from discopt import Model
from discopt.constants import INFEASIBILITY_SENTINEL, SENTINEL_THRESHOLD
from discopt.solvers import NLPResult, SolveStatus

# ---------------------------------------------------------------------------
# Focused guard unit tests (deterministic; the primary lock)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_guard_nonrigorous_sentinel_fathom_decertifies():
    """A sentinel-bound node with NO rigorous infeasibility proof is a
    non-rigorous fathom -> the predicate is True (caller sets _gap_certified=False,
    downgrading the terminal verdict from 'optimal' to 'feasible')."""
    # bound at the failure sentinel, node_infeasible_mask[i] == False
    assert S._nonrigorous_sentinel_fathom(INFEASIBILITY_SENTINEL, False) is True
    # any bound at/above the threshold, still not rigorously infeasible
    assert S._nonrigorous_sentinel_fathom(SENTINEL_THRESHOLD, False) is True
    assert S._nonrigorous_sentinel_fathom(1e30, False) is True


@pytest.mark.smoke
def test_guard_rigorous_infeasible_does_not_decertify():
    """A RIGOROUSLY infeasible node (empty relaxation over the finite box ->
    node_infeasible_mask[i] == True) is a SOUND prune: the predicate is False, so
    the gap is NOT decertified. This is the branch that must never be swallowed by
    an over-eager guard."""
    assert S._nonrigorous_sentinel_fathom(INFEASIBILITY_SENTINEL, True) is False
    assert S._nonrigorous_sentinel_fathom(1e30, True) is False


@pytest.mark.smoke
def test_guard_finite_bound_is_not_a_fathom():
    """A node carrying a real finite relaxation bound is not being fathomed on the
    failure sentinel at all -> the predicate is False regardless of the mask."""
    assert S._nonrigorous_sentinel_fathom(5.0, False) is False
    assert S._nonrigorous_sentinel_fathom(-1.0e6, False) is False
    assert S._nonrigorous_sentinel_fathom(0.0, True) is False


@pytest.mark.smoke
def test_guard_fail_before_evidence_always_true_stub_is_wrong():
    """Fail-before witness. If the guard were stubbed to ALWAYS return True (the
    naive 'always decertify' over-guard), it would mislabel a sound rigorous prune
    as a taint. Pin that the real predicate distinguishes the two cases, so such a
    stub would flip this assertion."""

    def _always_true(_lb, _infeasible):
        return True

    # The real predicate separates rigorous (False) from non-rigorous (True);
    # an always-True stub collapses them and this assertion catches it.
    assert S._nonrigorous_sentinel_fathom(1e30, True) != _always_true(1e30, True)
    # Symmetric fail-before for an always-False stub (which would let a
    # non-rigorous fathom certify a false optimum):
    assert S._nonrigorous_sentinel_fathom(1e30, False) != (lambda *_: False)()


# ---------------------------------------------------------------------------
# End-to-end integration: whole path from a model to the terminal verdict
# ---------------------------------------------------------------------------


def _build_feasible_nonconvex_minlp():
    """A genuinely FEASIBLE nonconvex MINLP (integer var routes it onto the spatial
    B&B batch path). Feasible point x=y=1 (x*y=1>=1), z=1 -> objective 3."""
    m = Model("nonrigorous_optimal")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=1, ub=3)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y + z)
    return m


class _IncumbentThenExhaustTree:
    """Wraps the real B&B tree: holds a genuine feasible incumbent (so the terminal
    path takes the WITH-incumbent branch), then reports the search finished after
    the first batch import. This is the tree state produced when a real incumbent is
    in hand but every *remaining* open node carries a non-rigorous failure sentinel
    and is fathomed by bound — the false-optimal setup.

    The incumbent (x=y=1, z=1, obj=3) is genuinely feasible for the model
    (x*y=1>=1); it is not the certified global optimum, and it was NOT proven so
    because the only thing that closed the tree was the non-rigorous sentinel
    fathom. So the honest verdict is "feasible", never "optimal".
    """

    def __init__(self, real_cls, *a, **k):
        self._t = real_cls(*a, **k)
        self._imported = 0
        # A genuinely feasible point for _build_feasible_nonconvex_minlp:
        # x=1, y=1, z=1 (order matches variable creation) -> x*y=1>=1, obj=3.
        self._inc = (np.array([1.0, 1.0, 1.0], dtype=float), 3.0)

    def inject_incumbent(self, sol, obj):
        # A better feasible incumbent may legitimately replace the seed, but never
        # a bogus (sentinel) one.
        if np.isfinite(obj) and obj < SENTINEL_THRESHOLD and obj < self._inc[1]:
            self._inc = (np.asarray(sol, dtype=float).copy(), float(obj))
        return None

    def import_results(self, ids, lbs, sols, feas):
        self._imported += 1
        return self._t.import_results(ids, lbs, sols, feas)

    def incumbent(self):
        return self._inc

    def is_finished(self):
        return self._imported >= 1

    def __getattr__(self, name):
        return getattr(self._t, name)


def _nonrigorous_failing_node_nlp(
    evaluator, x0, node_lb, node_ub, constraint_bounds, options, nlp_solver="ipopt", convex=False
):
    """Non-rigorous failure: return an 'optimal'-status point at the lower corner
    that violates the model's x*y>=1 constraint (lower corner has x*y=0). The node
    is sentinelled with NO infeasibility proof — a classic non-rigorous fathom."""
    xbad = np.clip(np.asarray(node_lb, dtype=float), -1e6, 1e6)
    return NLPResult(
        status=SolveStatus.OPTIMAL, x=xbad, objective=float(np.sum(xbad)), iterations=1
    )


@pytest.mark.smoke
def test_integration_nonrigorous_fathom_reports_feasible_not_optimal(monkeypatch):
    """A feasible nonconvex model with a real incumbent, whose remaining node is
    fathomed non-rigorously (local NLP fails, no infeasibility proof, no rigorous
    relaxation bound), must report status 'feasible' with gap NOT certified —
    NEVER 'optimal'. A certified optimum here would be a false global-optimality
    claim off an unproven subtree."""
    m = _build_feasible_nonconvex_minlp()
    real_cls = S.PyTreeManager

    monkeypatch.setattr(
        S, "PyTreeManager", lambda *a, **k: _IncumbentThenExhaustTree(real_cls, *a, **k)
    )
    monkeypatch.setattr(S, "_solve_node_nlp", _nonrigorous_failing_node_nlp)
    # No rigorous relaxation bound available: the only thing removing a node from
    # the tree is the non-rigorous sentinel.
    monkeypatch.setattr(S, "_compute_interval_bound", lambda *a, **k: -np.inf)

    res = S.solve_model(
        m,
        time_limit=20.0,
        max_nodes=5000,
        mccormick_bounds="none",
        nlp_solver="ipopt",
        subnlp_enabled=False,
        rens=False,
        presolve=False,
        skip_convex_check=True,
    )

    # Headline soundness assertion: a non-rigorous fathom must NOT certify optimal.
    assert res.status != "optimal", (
        "non-rigorous fathom falsely certified global optimality "
        f"(status={res.status!r}, gap_certified={getattr(res, 'gap_certified', None)})"
    )
    assert res.status == "feasible", f"expected 'feasible', got {res.status!r}"
    assert not getattr(res, "gap_certified", False), (
        "gap must not be certified when a node was fathomed non-rigorously"
    )
    # A2 corollary: no sentinel escapes through the public bound/gap surface.
    if res.bound is not None:
        assert abs(res.bound) < SENTINEL_THRESHOLD, f"sentinel leaked into bound: {res.bound}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
