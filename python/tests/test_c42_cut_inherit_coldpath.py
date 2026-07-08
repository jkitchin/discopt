"""Regression tests for C-42: the cut-inherit cold-path pool must never
truncate the B&B loop, and the lazy re-separation trigger.

Root cause pinned here (THRU-4 graduation blocker #552, fix in this change):
under ``DISCOPT_CUT_INHERIT`` the cold-path root pool probe (#551) captures a
valid root cut (nvs06: one PSD eigencut), but appending that row to a node's
relaxation can make the in-house warm-simplex solve fail *numerically*
(uncertified ``infeasible`` / ``iteration_limit`` — the C-38 failure class,
triggered by the extra row instead of a stale basis). The node then produced NO
bound; with the node NLP deliberately skipped on the LP-relaxer path, the
driver's failure sentinel pruned the ROOT non-rigorously, the pre-tree pump
chain never ran (its gate needs a root relaxation bound), and the B&B loop
exhausted after one node — nvs06 exited ``feasible 231.70`` at 1.5 s with 8.5 s
of budget unused instead of certifying ``optimal 1.7703125``.

The fix: the inherited pool is an ACCELERATOR, never a dependency. When the
pool-augmented cold solve yields no certified verdict, the pool rows are
stripped and the node re-solved without them — byte-identical to the no-pool
solve the default path performs — so the pool can perturb neither the incumbent
search nor loop termination (``solve_at_node`` in ``mccormick_lp.py``).

Part 2 (the tspn05-class blocker): lazy re-separation. Inheritance-only leaves
tspn05's nodes too loose to fathom (the tree's bound freezes, cert lost at
budget). The trigger is a driver-side GLOBAL-bound-stall governor (``solver.py``
``_LAZY_RESEP_*`` constants: stagnation window -> bounded re-separation probe ->
mute if the probe is bound-inert, reset on any improvement) plus a relaxer-side
stride safety net (``_LAZY_RESEP_STRIDE`` in ``mccormick_lp.py``). Per-node
signals (parent-bound stall, LP-gain productivity) were tried first and
falsified by measurement — see ``docs/dev/c42-cut-inherit-fix-2026-07-07.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver_tuning import SolverTuning

_DATA = Path(__file__).parent / "data" / "minlplib_nl"


def _mini_qp_relaxer_and_pool():
    """A dense integer QP whose root separation verifiably pools cuts, with the
    incremental engine disabled so node solves take the COLD path (the C-42
    surface — nvs06/nvs19/nvs24 are cold-path instances)."""
    import discopt.modeling.core as dm
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    m = dm.Model("mini_c42")
    x = m.integer("x", shape=(2,), lb=-3, ub=3)
    m.minimize(x[0] * x[0] + x[1] * x[1] - 2 * x[0] * x[1] + 0.5 * x[0] - 0.5 * x[1])
    relaxer = MccormickLPRelaxer(m, psd_cuts=True)
    lb = np.full(2, -3.0)
    ub = np.full(2, 3.0)
    chunks: list = []
    res = relaxer.solve_at_node(lb, ub, time_limit=30.0, out_cuts=chunks)
    assert res.status == "optimal" and chunks, "root pool capture failed"
    relaxer._inc = None  # force the cold path at node solves
    return relaxer, lb, ub, chunks[0]


@pytest.mark.smoke
def test_pool_drop_retry_recovers_the_node_bound(monkeypatch):
    """The mechanism test: when the pool-augmented cold solve fails with no
    certified verdict, the pool rows must be stripped and the node re-solved —
    the node keeps a valid bound instead of losing it (which is what truncated
    nvs06's loop at the root). Fails before the C-42 fix (no bound), passes
    after (bound recovered, ``pool/dropped_nodes`` fires)."""
    relaxer, lb, ub, pool = _mini_qp_relaxer_and_pool()

    baseline = relaxer.solve_at_node(lb, ub, time_limit=30.0)
    assert baseline.status == "optimal" and baseline.lower_bound is not None

    # Poison exactly the FIRST relaxation solve of the next node call (the
    # pool-augmented one) with an uncertified failure; delegate afterwards.
    # This reproduces deterministically what the nvs06 PSD pool row does to the
    # warm simplex numerically.
    from discopt._jax import milp_relaxation as _mr

    milp_cls = _mr.MilpRelaxationModel
    real_solve = milp_cls.solve
    state = {"fail_next": True}

    def flaky(self, *args, **kwargs):
        if state["fail_next"]:
            state["fail_next"] = False
            return SimpleNamespace(
                status="numerical", x=None, objective=None, bound=None, farkas_certified=False
            )
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(milp_cls, "solve", flaky)

    res = relaxer.solve_at_node(
        lb, ub, time_limit=30.0, inherited_cuts=pool, skip_pool_separators=True
    )
    assert res.status == "optimal", (
        f"node lost its bound after a failed pool-augmented solve: {res.status} "
        "(C-42: the pool must be dropped and the node re-solved)"
    )
    assert res.lower_bound is not None
    # Sound and no worse than the no-pool baseline (the retry IS the no-pool
    # solve; the lifted skip lets separation tighten it further).
    assert res.lower_bound >= baseline.lower_bound - 1e-6
    assert relaxer._pool_stats["dropped_nodes"] == 1


@pytest.mark.smoke
def test_pool_has_rows_is_sparse_safe():
    """C-43: the pool ``A_rows`` is a scipy CSR matrix whose ``len()`` is ambiguous
    and *raises*. The re-verify gate must count rows via ``.shape[0]`` (through
    ``_pool_has_rows``), not ``len()`` — otherwise it throws on every pool node,
    the driver swallows the exception and silently skips the node, and the pool
    infeasible is never re-verified (masking the bug behind a crash rather than
    fixing it). This asserts the helper is sparse-safe and truthful."""
    import scipy.sparse as sp
    from discopt._jax.mccormick_lp import _pool_has_rows

    A = sp.csr_matrix(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]))
    b = np.array([0.5, 0.5])
    # A naive len(A) would raise here — the whole point of the helper:
    with pytest.raises(TypeError):
        len(A)
    assert _pool_has_rows((A, b)) is True
    assert _pool_has_rows(None) is False
    assert _pool_has_rows((sp.csr_matrix((0, 3)), np.zeros(0))) is False


@pytest.mark.smoke
def test_pool_infeasible_reverify_recovers_false_fathom(monkeypatch):
    """C-43 mechanism: when the pool-augmented node solve fathoms a node as
    ``infeasible`` but the pool-free relaxation of the SAME box is feasible, the
    inherited pool row cut off a non-empty node (invalid on this sub-box). The
    re-verify must drop the pool and recover the node instead of trusting the false
    fathom. Fails before the C-43 fix (the false ``infeasible`` propagates); passes
    after (``pool/dropped_nodes`` fires and a valid bound is returned)."""
    relaxer, lb, ub, pool = _mini_qp_relaxer_and_pool()

    baseline = relaxer.solve_at_node(lb, ub, time_limit=30.0)
    assert baseline.status == "optimal" and baseline.lower_bound is not None

    # Poison the FIRST relaxation solve of the next node call (the pool-augmented
    # one) with a *Farkas-certified* infeasible — reproducing what an
    # invalid-on-sub-box pool row does: the pool-augmented polytope certifies empty
    # though the pool-free box is feasible. The pool-free re-solve then delegates.
    from discopt._jax import milp_relaxation as _mr

    milp_cls = _mr.MilpRelaxationModel
    real_solve = milp_cls.solve
    state = {"fail_next": True}

    def flaky(self, *args, **kwargs):
        if state["fail_next"]:
            state["fail_next"] = False
            r = SimpleNamespace(
                status="infeasible", x=None, objective=None, bound=None, farkas_certified=True
            )
            return r
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(milp_cls, "solve", flaky)

    res = relaxer.solve_at_node(
        lb, ub, time_limit=30.0, inherited_cuts=pool, skip_pool_separators=True
    )
    assert res.status == "optimal", (
        f"pool-induced false infeasible was trusted: {res.status} "
        "(C-43: a pool-augmented infeasible must be re-verified pool-free)"
    )
    assert res.lower_bound is not None
    assert res.lower_bound >= baseline.lower_bound - 1e-6
    assert relaxer._pool_stats["dropped_nodes"] == 1


@pytest.mark.smoke
def test_lazy_reseparation_stride_net_fires():
    """The relaxer-side safety net: every ``_LAZY_RESEP_STRIDE``-th
    skip-eligible node solve runs the full separation pass regardless of the
    driver's governor, so inheritance can never fully starve a class."""
    from discopt._jax.mccormick_lp import _LAZY_RESEP_STRIDE

    relaxer, lb, ub, pool = _mini_qp_relaxer_and_pool()
    for _ in range(_LAZY_RESEP_STRIDE):
        relaxer.solve_at_node(
            lb, ub, time_limit=30.0, inherited_cuts=pool, skip_pool_separators=True
        )
    assert relaxer._pool_stats["lazy_reseparations"] == 1
    assert relaxer._pool_stats["skipped_separations"] == _LAZY_RESEP_STRIDE - 1


@pytest.mark.slow
def test_nvs06_cut_inherit_certifies_like_default():
    """The #552 graduation blocker, end-to-end: nvs06 flag-ON must certify the
    oracle optimum 1.7703125 exactly like flag-OFF, instead of exiting after one
    node with ``feasible 231.70`` and most of the budget unused. Fails before
    the C-42 fix, passes after."""
    model = from_nl(str(_DATA / "nvs06.nl"))
    res = model.solve(time_limit=20, gap_tolerance=1e-4, tuning=SolverTuning(cut_inherit=True))
    assert str(res.status) == "optimal", f"nvs06 flag-ON lost its certificate: {res.status}"
    assert res.objective == pytest.approx(1.7703125, abs=1e-5)
    assert res.bound is not None and res.bound <= res.objective + 1e-6
    stats = res.solver_stats or {}
    # The cold-path pool populated (the #551 extension under test)...
    assert stats.get("pool/size", 0) >= 1, f"cold-path pool did not populate: {stats}"
    # ...and the destabilized root solve recovered by dropping it (the fix).
    assert stats.get("pool/dropped_nodes", 0) >= 1, f"pool-drop retry never fired: {stats}"


_CORPUS = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark" / "minlplib" / "nl"


@pytest.mark.slow
def test_nvs22_cut_inherit_no_false_optimal():
    """C-43 (#564) regression: the former CUT-INHERIT-GRAD graduation blocker.

    Before this fix, flag-ON ``nvs22`` certified a FALSE-OPTIMAL ``33.55`` against
    the oracle ``6.0582``: an inherited root cut-pool row is NOT valid on a
    tightened sub-box (the "root-valid ⇒ globally-valid" premise is false for the
    captured convex/square cut family here — a re-lifted child column changes what
    the pool row addresses), so a node whose box contains the true optimum became
    *Farkas-certified infeasible* once the pool was appended, falsely fathoming the
    region and closing the tree around ``33.55``. C-42's pool-drop-retry did not
    cover this: it fires only on the *no-certified-verdict* branch of the cold path,
    but here the pool-augmented solve SUCCEEDS with a (false) ``infeasible``.

    The fix re-verifies every pool-augmented ``infeasible`` against a pool-free
    solve (``solve_at_node`` in ``mccormick_lp.py``): the fathom is kept only if the
    pool-free relaxation is also infeasible; otherwise the node is recovered on its
    valid pool-free bound. So flag-ON is now sound — and in fact certifies the
    oracle optimum. This test was a strict-xfail (it went XPASS the moment the fix
    landed); it is now a plain passing regression. See
    ``docs/dev/c43-nvs22-fix-graduate-2026-07-08.md``.
    """
    nl = _CORPUS / "nvs22.nl"
    if not nl.exists():
        pytest.skip("nvs22 not in the local MINLPLib corpus")
    # Sanity: the default (force-off) path is SOUND — this must always hold.
    ref = from_nl(str(nl)).solve(time_limit=25, gap_tolerance=1e-4)
    assert ref.objective == pytest.approx(6.0582200, rel=5e-3), (
        f"default-path nvs22 regressed: {ref.objective}"
    )
    # Flag-ON must be SOUND. The hard invariant (C-43): no false certificate — the
    # dual bound is a valid lower bound (<= oracle + tol), and if the search
    # certifies, the objective is the oracle optimum (never the old 33.55).
    res = from_nl(str(nl)).solve(
        time_limit=25, gap_tolerance=1e-4, tuning=SolverTuning(cut_inherit=True)
    )
    assert res.bound is None or res.bound <= 6.0582200 + 1e-3, (
        f"nvs22 flag-ON dual bound {res.bound} crossed the oracle 6.0582 "
        "(false bound — the C-43 pool-infeasible re-verify regressed)"
    )
    if str(res.status) == "optimal":
        assert res.objective == pytest.approx(6.0582200, rel=5e-3), (
            f"nvs22 flag-ON FALSE-OPTIMAL: certified {res.objective} vs oracle 6.0582"
        )
    # The re-verify recovered at least one falsely-fathomed node (the mechanism
    # under test); if it never fired, the guard is inert and the test is vacuous.
    stats = res.solver_stats or {}
    assert stats.get("pool/dropped_nodes", 0) >= 1, (
        f"C-43 pool-infeasible re-verify never fired on nvs22: {stats}"
    )


@pytest.mark.slow
def test_tspn05_cut_inherit_certifies_via_lazy_reseparation():
    """The other #552 blocker: with inheritance-only, tspn05's bound stalls at
    190.28 and the certificate is lost at budget. The lazy re-separation
    trigger must break the stall so flag-ON certifies the oracle optimum
    191.25521 within the flag-OFF-class budget."""
    model = from_nl(str(_DATA / "tspn05.nl"))
    res = model.solve(time_limit=60, gap_tolerance=1e-4, tuning=SolverTuning(cut_inherit=True))
    assert str(res.status) == "optimal", f"tspn05 flag-ON lost its certificate: {res.status}"
    assert res.objective == pytest.approx(191.25521, rel=1e-5)
    assert res.bound is not None and res.bound <= res.objective + 1e-6
    stats = res.solver_stats or {}
    assert (
        stats.get("pool/stall_reseparations", 0) >= 1
        or stats.get("pool/lazy_reseparations", 0) >= 1
    ), f"no re-separation trigger fired on the stalling class: {stats}"
