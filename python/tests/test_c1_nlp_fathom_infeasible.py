"""Soundness lock (C-1): a non-rigorous NLP fathom must never yield 'infeasible'.

Declaring a model INFEASIBLE is a *certificate claim*: it asserts the feasible set
is empty. That claim is only sound when every node that emptied the search tree was
fathomed on a RIGOROUS proof — an empty McCormick/LP relaxation over a finite box,
or a genuine ``SolveStatus.INFEASIBLE``. A node whose *local* NLP merely failed
(diverged, hit an iteration limit, or returned a constraint-violating iterate) is
sentinelled and pruned, but that is NOT a proof its subtree is empty: an
interior-point method can stall at an infeasible point on a perfectly feasible
nonconvex node.

Before the C-1 fix, ``solve_model``'s finalize else-branch declared
``status="infeasible"`` whenever the tree exhausted with no incumbent, checking only
``max_nodes`` / ``time_limit`` — it never consulted whether the fathoms were
rigorous. So a FEASIBLE model whose node NLPs all fail non-rigorously (and which has
no rigorous relaxation bound) was reported INFEASIBLE — the worst-class error
(a feasible problem told it has no solution).

The fix tracks a ``_nonrigorous_fathom`` flag (mirroring ``_solve_nlp_bb``'s
``_unconverged_fathom``) set whenever a node enters the tree with the failure
sentinel but WITHOUT a rigorous infeasibility certificate, and downgrades the
verdict to ``"unknown"`` (feasibility undetermined) instead of ``"infeasible"``.

These tests are @pytest.mark.smoke: they run sub-second and lock the class in CI on
every PR. ``test_nonrigorous_fathom_is_not_reported_infeasible`` reproduces the bug
(it returned ``"infeasible"`` on the pre-fix code); the rigorous negative test
confirms genuine infeasibility is preserved.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as S
import numpy as np
import pytest
from discopt import Model
from discopt.solvers import NLPResult, SolveStatus


def _failing_node_nlp(
    evaluator, x0, node_lb, node_ub, constraint_bounds, options, nlp_solver="ipopt", convex=False
):
    """A NON-rigorous NLP failure: return an 'optimal'-status point at the lower
    corner, which violates the model constraints. This is NOT a proof of
    infeasibility — the local solver simply failed to find a feasible point."""
    xbad = np.clip(np.asarray(node_lb, dtype=float), -1e6, 1e6)
    return NLPResult(
        status=SolveStatus.OPTIMAL, x=xbad, objective=float(np.sum(xbad)), iterations=1
    )


def _failing_multistart(*a, **k):
    node_lb = a[1] if len(a) > 1 else k["node_lb"]
    xbad = np.clip(np.asarray(node_lb, dtype=float), -1e6, 1e6)
    return NLPResult(
        status=SolveStatus.OPTIMAL, x=xbad, objective=float(np.sum(xbad)), iterations=1
    )


def _build_feasible_nonconvex_minlp():
    """A genuinely FEASIBLE nonconvex MINLP (integer var keeps it on the spatial
    B&B batch path). Feasible point: x=y=1 (x*y=1>=1), z=1 -> objective 3."""
    m = Model("c1_feasible")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=1, ub=3)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y + z)
    return m


class _ExhaustNoIncumbentTree:
    """Wraps the real B&B tree but reports the search finished with NO incumbent
    after the first batch of results is imported. This is exactly the tree state
    the Rust fathom-at-tight-box path produces when every leaf carries a
    non-rigorous failure sentinel: ``open_count()==0`` and no feasible incumbent."""

    def __init__(self, real_cls, *a, **k):
        self._t = real_cls(*a, **k)
        self._imported = 0

    def inject_incumbent(self, *a, **k):
        # No primal heuristic may smuggle in a feasible incumbent: we are
        # exercising the case where the ONLY thing removing nodes is non-rigorous
        # fathoming.
        return None

    def import_results(self, ids, lbs, sols, feas):
        self._imported += 1
        return self._t.import_results(ids, lbs, sols, feas)

    def incumbent(self):
        return None

    def is_finished(self):
        return self._imported >= 1

    def __getattr__(self, name):
        return getattr(self._t, name)


@pytest.mark.smoke
def test_nonrigorous_fathom_is_not_reported_infeasible(monkeypatch):
    """A feasible model whose nodes are all fathomed non-rigorously must NOT be
    reported infeasible. Pre-fix this returned ``status="infeasible"``."""
    m = _build_feasible_nonconvex_minlp()
    real_cls = S.PyTreeManager

    monkeypatch.setattr(
        S, "PyTreeManager", lambda *a, **k: _ExhaustNoIncumbentTree(real_cls, *a, **k)
    )
    monkeypatch.setattr(S, "_solve_node_nlp", _failing_node_nlp)
    monkeypatch.setattr(S, "_solve_root_node_multistart", _failing_multistart)
    # No rigorous relaxation bound available (mccormick 'none' + no interval save):
    # the only thing that removes a node from the tree is the non-rigorous sentinel.
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

    # Headline: the false certificate must be gone. A feasible model must never be
    # declared infeasible off non-rigorous fathoms.
    assert res.status != "infeasible", (
        f"feasible model falsely reported infeasible via non-rigorous fathoms (status={res.status})"
    )
    # It should report the sound 'undetermined' verdict, with no certified gap.
    assert res.status == "unknown", f"expected 'unknown', got {res.status!r}"
    assert not getattr(res, "gap_certified", False)


@pytest.mark.smoke
def test_rigorous_infeasibility_is_preserved():
    """A RIGOROUSLY infeasible model (empty box: x>=5 with x<=1) must still report
    'infeasible' — the fix must not downgrade a genuine infeasibility certificate."""
    m = Model("c1_rigorous_infeasible")
    x = m.continuous("x", lb=0.0, ub=1.0)
    z = m.integer("z", lb=0, ub=2)
    m.subject_to(x >= 5.0)  # contradicts x <= 1 -> empty feasible set
    m.subject_to(x * x + z >= 0.0)  # nonconvex term to reach the spatial path
    m.minimize(x + z)

    res = m.solve(time_limit=20.0, max_nodes=500)
    assert res.status == "infeasible", (
        f"rigorously infeasible model must stay infeasible, got {res.status!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
