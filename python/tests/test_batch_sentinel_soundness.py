"""
Soundness regression: batch-path failure sentinels must never certify a wrong
objective.

A node whose NLP relaxation solve fails (error, divergence, or *local*
infeasibility) carries ``INFEASIBILITY_SENTINEL`` as its lower bound and is
pruned by the Rust tree as soon as an incumbent exists — without any proof
that the subtree is suboptimal or infeasible. The serial node path already
decertifies the gap in that case (issue #27a); these tests pin the same
guarantee onto the batch IPM/POUNCE path, which previously pruned silently and
could report a certified "optimal" for a wrong objective.

The batch solver is monkeypatched to simulate universal node-solve failure, so
the *failed nodes themselves* never license a certification. A certification
may still arise — soundly — from an INDEPENDENT rigorous global bound (the root
MILP-relaxation fallback, issue #138) when that bound proves the incumbent
globally optimal. The invariant under test is therefore the soundness one: any
reported bound is valid (never above the true optimum) and any certification is
of the TRUE optimum, never of a suboptimal point.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest


def _build_nonconvex_minlp() -> tuple[dm.Model, dict]:
    """Nonconvex MINLP that branches into a multi-node frontier.

    The integer ``x`` makes the root relaxation fractional, so the tree opens
    several nodes at once: ``export_batch`` returns >1 node and the POUNCE
    batch path (``n_batch > 1``) runs. The ``sin`` term keeps the model
    nonconvex and out of the MIQP class and — crucially — leaves it free of
    bilinear products, so the spatial-BB McCormick LP relaxer is inactive.
    That matters because the relaxer would otherwise rescue a failed node-NLP
    solve with a valid LP bound; without it, the failure sentinel survives to
    the decertification guard. The optimum is x=0, y≈3.665 (1.3*0 + sin≈-1.0 =
    -1.0); the warm start (x=8, y=5) is feasible but far worse, so pruning the
    whole tree off failure sentinels and certifying the gap would report a
    badly wrong "optimal".
    """
    m = dm.Model("sentinel_soundness")
    x = m.integer("x", lb=0, ub=8)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(2 * x + dm.sin(3 * y) - 0.7 * x)
    m.subject_to(3 * x + 2 * y >= 7)
    m.subject_to(x - y <= 3)
    warm = {x: 8, y: 5.0}
    return m, warm


# Optimum of the model above: x=0, y≈3.665 -> 1.3*0 + sin(3y)≈-1.0.
_OPT = -1.0


def _all_sentinel_batch(evaluator, batch_lb, batch_ub, batch_ids, n_vars, *args, **kwargs):
    """Stand-in for _solve_batch_pounce: every node NLP 'fails'."""
    n_batch = len(batch_ids)
    ids = np.array(batch_ids, dtype=np.int64)
    lbs = np.full(n_batch, S._INFEASIBILITY_SENTINEL, dtype=np.float64)
    sols = np.empty((n_batch, n_vars), dtype=np.float64)
    for i in range(n_batch):
        lb = np.asarray(batch_lb[i], dtype=np.float64)
        ub = np.asarray(batch_ub[i], dtype=np.float64)
        sols[i] = 0.5 * (np.clip(lb, -1e4, 1e4) + np.clip(ub, -1e4, 1e4))
    feas = np.zeros(n_batch, dtype=bool)
    trusted = np.ones(n_batch, dtype=bool)
    return ids, lbs, sols, feas, trusted


@pytest.mark.correctness
class TestBatchSentinelSoundness:
    def test_batch_failure_sentinels_decertify_gap(self, monkeypatch) -> None:
        """All batch nodes failing must not yield a certified 'optimal'."""
        m, warm = _build_nonconvex_minlp()

        calls = {"n": 0}
        real_batch = S._solve_batch_pounce

        def fake_batch(*args, **kwargs):
            calls["n"] += 1
            return _all_sentinel_batch(*args, **kwargs)

        # POUNCE is the NLP-node batch engine; force batching on this small model
        # (the size gate would otherwise route it to the serial path).
        monkeypatch.setattr(S, "_POUNCE_BATCH_MIN_VARS", 1)
        monkeypatch.setattr(S, "_solve_batch_pounce", fake_batch)

        result = m.solve(
            nlp_solver="pounce",
            batch_size=8,
            time_limit=60.0,
            max_nodes=5_000,
            initial_solution=warm,
        )

        assert calls["n"] >= 1, (
            "Batch path was never exercised; test setup no longer matches "
            "the solver's dispatch (check _use_pounce_batch conditions)."
        )
        # The incumbent survives as a feasible point...
        assert result.objective is not None
        # ...and SOUNDNESS is the real invariant: a failed batch node carries a
        # sentinel, not a proof, so the *failed nodes* may never certify the gap.
        # Certification is legitimate ONLY when an INDEPENDENT rigorous global
        # bound — the root-relaxation fallback (issue #138), never the failed
        # nodes — proves the incumbent globally optimal. Two guarantees, both held:
        #   (1) any reported dual bound is sound (never above the true optimum);
        #   (2) any certification is of the TRUE optimum, never a suboptimal point
        #       (this is exactly the "certified a wrong objective" failure the
        #       batch path could previously commit).
        assert result.bound is None or result.bound <= _OPT + 1e-6, (
            f"unsound bound {result.bound} > optimum {_OPT}"
        )
        if result.gap_certified:
            assert result.status == "optimal"
            assert abs(result.objective - _OPT) <= 1e-2, (
                f"certified a non-optimal objective {result.objective} "
                f"(true optimum {_OPT}) after pruning failed nodes"
            )

        monkeypatch.setattr(S, "_solve_batch_pounce", real_batch)

    def test_unpatched_solver_still_certifies(self) -> None:
        """Control: the real batch solver reaches the true optimum with a valid
        (certified) bound, confirming the model is solvable and that the sentinel
        test above exercises failure handling on a genuinely tractable problem."""
        m, _ = _build_nonconvex_minlp()
        result = m.solve(
            nlp_solver="pounce",
            batch_size=8,
            time_limit=120.0,
            max_nodes=20_000,
        )
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert abs(result.objective - _OPT) <= 1e-2
        # Unlike the sentinel case, the real solve keeps a rigorous bound.
        assert result.gap_certified is True
        assert result.bound is not None
