"""C-13 (P0): the serial convex B&B path must not trust an under-converged NLP
objective as a rigorous node lower bound.

For a **convex** model the node NLP's objective is a valid lower bound *only* when
the solve converged to a KKT point (``SolveStatus.OPTIMAL``). An interior-point
iterate that stops at ``ITERATION_LIMIT`` can sit strictly ABOVE the true node
minimum (non-KKT, unconverged duals), so its objective is NOT a valid lower bound.
Trusting it can fathom the subtree holding the optimum while the optimality gap
stays certified — a false "optimal" certificate, the worst failure class.

The batch path already guards this via the ``_batch_trusted`` mask (roadmap P0.3),
and ``_solve_nlp_bb`` decertifies on ITERATION_LIMIT. This locks the same guarantee
onto the serial ``solve_model`` path (reached e.g. with ``nlp_bb=False``, or with
lazy constraints), which previously accepted ITERATION_LIMIT objectives as rigorous
bounds with no trust check and never decertified for convex models.

Contract under test (the false-certificate CLASS, not a named instance):

  * A convex serial node whose NLP returns ITERATION_LIMIT (non-KKT) must NOT
    produce ``status == "optimal"`` with ``gap_certified is True``. Either the true
    optimum is proven some other (rigorous) way, or the run decertifies to
    "feasible" — but it never *certifies* on the under-converged objective.
  * The reported dual bound is always a valid lower bound (``bound <= objective``
    for a minimize) — the dual never crosses the primal.
  * A normally-converging convex solve (all nodes OPTIMAL) is unaffected: it still
    certifies optimality with the correct objective (no regression, bound-neutral).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt.solvers import SolveStatus

# Convex MINLP (exp keeps it nonlinear, out of the MIQP class). The continuous
# relaxation optimum is fractional (x -> 4.5), so B&B genuinely branches on the
# serial path. True integer optimum: x=4, y=0.7 -> exp(0.6)+0.25+0 ~ 2.0721.
_OPT = float(np.exp(0.15 * 4) + (4 - 4.5) ** 2 + (0.7 - 0.7) ** 2)


def _convex_minlp():
    m = dm.Model("c13_cvx")
    x = m.integer("x", lb=0, ub=8)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(dm.exp(0.15 * x) + (x - 4.5) ** 2 + (y - 0.7) ** 2)
    m.subject_to(x + y >= 0)
    return m


def _force_iteration_limit(monkeypatch, inflate: float = 0.0):
    """Make every serial node/root NLP report ITERATION_LIMIT (non-KKT).

    ``inflate`` optionally pushes the objective ABOVE the true node minimum,
    emulating an interior-point iterate that stalled above the optimum — the
    precise condition under which the objective is not a valid lower bound.
    """
    orig_node = S._solve_node_nlp
    orig_root = S._solve_root_node_multistart

    def _degrade(res):
        if res is not None and res.status == SolveStatus.OPTIMAL:
            res.status = SolveStatus.ITERATION_LIMIT
            if inflate:
                res.objective = float(res.objective) + inflate
        return res

    monkeypatch.setattr(S, "_solve_node_nlp", lambda *a, **k: _degrade(orig_node(*a, **k)))
    monkeypatch.setattr(
        S, "_solve_root_node_multistart", lambda *a, **k: _degrade(orig_root(*a, **k))
    )


@pytest.mark.smoke
def test_serial_convex_iteration_limit_certifies_only_soundly(monkeypatch):
    """Under-converged convex serial nodes must never yield a FALSE certificate.

    #640: the uniform engine's node dual bound is the rigorous McCormick LP
    relaxation, NOT the (possibly non-KKT) NLP objective — so degrading every serial
    NLP to ITERATION_LIMIT can no longer inject an invalid bound (the companion
    ``inflated`` test confirms an above-optimum NLP objective never crosses the
    primal). The engine may therefore legitimately certify the TRUE optimum off the
    LP bound; the P0 guarantee this pins is the sound one — IF it certifies, the
    certificate is VALID (correct optimum, dual bound never above it), and it never
    certifies a false optimal off the degraded objective.
    """
    _force_iteration_limit(monkeypatch, inflate=0.0)
    # nlp_bb=False forces the solve_model serial loop (bypasses the convex NLP-BB
    # auto-select); batch_size=1 forces the serial per-node path (n_batch == 1).
    r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=1, nlp_bb=False)

    # No FALSE certificate: a certified optimal is the true optimum with a valid
    # (never-too-high) dual bound.
    if r.status == "optimal" and r.gap_certified is True:
        assert r.objective == pytest.approx(_OPT, abs=1e-3)
        assert r.bound is not None and r.bound <= _OPT + 1e-4
    # And the dual bound never crosses the primal, certified or not.
    if r.bound is not None and r.objective is not None:
        assert r.bound <= r.objective + 1e-6


@pytest.mark.smoke
def test_serial_convex_inflated_bound_never_crosses_primal(monkeypatch):
    """An inflated (above-optimum) non-KKT node objective must not fathom the
    optimum nor be reported as a certified dual bound above the true optimum."""
    _force_iteration_limit(monkeypatch, inflate=50.0)
    r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=1, nlp_bb=False)

    # No false optimal on the objective.
    if r.status == "optimal" and r.gap_certified:
        assert r.objective is not None and abs(r.objective - _OPT) < 1e-2, (
            f"false optimal: certified obj={r.objective} vs true {_OPT}"
        )

    # Certificate invariant: the reported dual bound never crosses the incumbent
    # (bound <= objective for a minimize), and never exceeds the true optimum.
    if r.bound is not None and np.isfinite(r.bound):
        if r.objective is not None and np.isfinite(r.objective):
            assert r.bound <= r.objective + 1e-6, (
                f"dual bound {r.bound} crossed primal {r.objective}"
            )
        assert r.bound <= _OPT + 1e-6, f"dual bound {r.bound} exceeds true optimum {_OPT}"


@pytest.mark.smoke
def test_serial_convex_converged_still_certifies():
    """Control: a normally-converging convex serial solve is unaffected — it still
    certifies optimality with the correct objective (fix is bound-neutral on the
    KKT-converged path)."""
    r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=1, nlp_bb=False)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective is not None and abs(r.objective - _OPT) < 1e-3
    # Valid dual bound meeting the incumbent.
    assert r.bound is not None and r.bound <= r.objective + 1e-6
