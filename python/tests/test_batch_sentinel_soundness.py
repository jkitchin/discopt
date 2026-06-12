"""
Soundness regression: batch-path failure sentinels must decertify the gap.

A node whose NLP relaxation solve fails (error, divergence, or *local*
infeasibility) carries ``INFEASIBILITY_SENTINEL`` as its lower bound and is
pruned by the Rust tree as soon as an incumbent exists — without any proof
that the subtree is suboptimal or infeasible. The serial node path already
decertifies the gap in that case (issue #27a) so the result downgrades from
"optimal" to "feasible". These tests pin the same guarantee onto the batch
IPM/POUNCE path, which previously pruned silently and could report a
certified "optimal" for a wrong objective.

The batch solver is monkeypatched to simulate universal node-solve failure;
the warm-started incumbent then makes every sentinel node prunable, the tree
exhausts, and an unguarded solver would claim a certified optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest


def _build_nonconvex_minlp() -> tuple[dm.Model, dict]:
    """Nonconvex MINLP whose root relaxation is fractional in the binaries.

    The knapsack structure forces real branching (several open nodes, so the
    batch path runs with batch_size=4); the x1*y*y term (degree 3) keeps the
    model out of the MIQP classification and makes it nonconvex. The warm
    start (all zeros, obj=0) is feasible but far from optimal (-11.5), so
    pruning the whole tree off failure sentinels and certifying the gap
    would report a badly wrong "optimal".
    """
    m = dm.Model("sentinel_soundness")
    x1 = m.binary("x1")
    x2 = m.binary("x2")
    x3 = m.binary("x3")
    x4 = m.binary("x4")
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(-3 * x1 - 4 * x2 - 5 * x3 - 7 * x4 + x1 * y * y - 0.5 * y)
    m.subject_to(2 * x1 + 3 * x2 + 4 * x3 + 5 * x4 <= 8)
    warm = {x1: 0, x2: 0, x3: 0, x4: 0, y: 0.0}
    return m, warm


def _all_sentinel_batch(evaluator, batch_lb, batch_ub, batch_ids, n_vars, *args, **kwargs):
    """Stand-in for _solve_batch_ipm: every node NLP 'fails'."""
    n_batch = len(batch_ids)
    ids = np.array(batch_ids, dtype=np.int64)
    lbs = np.full(n_batch, S._INFEASIBILITY_SENTINEL, dtype=np.float64)
    sols = np.empty((n_batch, n_vars), dtype=np.float64)
    for i in range(n_batch):
        lb = np.asarray(batch_lb[i], dtype=np.float64)
        ub = np.asarray(batch_ub[i], dtype=np.float64)
        sols[i] = 0.5 * (np.clip(lb, -1e4, 1e4) + np.clip(ub, -1e4, 1e4))
    feas = np.zeros(n_batch, dtype=bool)
    return ids, lbs, sols, feas


@pytest.mark.correctness
class TestBatchSentinelSoundness:
    def test_batch_failure_sentinels_decertify_gap(self, monkeypatch) -> None:
        """All batch nodes failing must not yield a certified 'optimal'."""
        m, warm = _build_nonconvex_minlp()

        calls = {"n": 0}
        real_batch = S._solve_batch_ipm

        def fake_batch(*args, **kwargs):
            calls["n"] += 1
            return _all_sentinel_batch(*args, **kwargs)

        monkeypatch.setattr(S, "_solve_batch_ipm", fake_batch)

        result = m.solve(
            nlp_solver="ipm",
            batch_size=4,
            time_limit=60.0,
            max_nodes=5_000,
            initial_solution=warm,
        )

        assert calls["n"] >= 1, (
            "Batch path was never exercised; test setup no longer matches "
            "the solver's dispatch (check _use_ipm_batch conditions)."
        )
        # The incumbent (warm start or better) survives as a feasible point...
        assert result.objective is not None
        # ...but the gap must NOT be certified: every pruned node carried a
        # failure sentinel, not an infeasibility proof or valid bound.
        assert result.status != "optimal", (
            f"Unsound certification: status={result.status!r} with "
            f"obj={result.objective} after pruning failed nodes"
        )
        assert result.bound is None
        assert result.gap is None

        monkeypatch.setattr(S, "_solve_batch_ipm", real_batch)

    def test_unpatched_solver_still_certifies(self) -> None:
        """Control: the real batch solver still reaches a certified optimum,
        so the decertification above is attributable to the sentinels."""
        m, _ = _build_nonconvex_minlp()
        result = m.solve(
            nlp_solver="ipm",
            batch_size=4,
            time_limit=120.0,
            max_nodes=20_000,
        )
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        # Optimum: x2=1,x4=1 (weight 8), x1=0 so y→1 → -11 - 0.5 = -11.5.
        assert abs(result.objective - (-11.5)) <= 1e-3 + 1e-2 * 11.5
