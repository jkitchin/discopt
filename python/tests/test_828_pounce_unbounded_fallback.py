"""#828: jit1 solves on the default POUNCE path (pounce#258 fixed the root cause).

jit1 is a convex reciprocal MINLP (objective ``Σ cᵢ/xᵢ`` for x>0 plus a badly
scaled linear tail, coefficients spanning ~10 to ~1e7). Its B&B node NLPs converge
to the node optimum and then returned Ipopt status 3
(``Search_Direction_Becomes_Too_Small``) because, under strong objective scaling, the
strict termination certificate was unreachable — and discopt's ``_IPOPT_STATUS_MAP``
mis-mapped status 3 onto ``UNBOUNDED``. The NLP-BB then fathomed those nodes
non-rigorously and reported ``status=unknown`` with no incumbent, while SCIP solves
jit1 in <0.1s.

Fixed end-to-end by:
  * **pounce#258** — makes the strict certificate reachable under objective scaling,
    so raw POUNCE returns OPTIMAL (not status 3) on jit1's nodes; and
  * discopt — ``_IPOPT_STATUS_MAP`` maps status 3 to ``ITERATION_LIMIT`` (a stalled
    limit), never ``UNBOUNDED`` (a local NLP cannot prove global unboundedness).

The interim cyipopt-retry-on-UNBOUNDED workaround (#834/#836) is therefore removed —
jit1 now solves on the pure POUNCE path with no retry and no cyipopt dependency.
Needs only the benchmark corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import cached_evaluator

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
_JIT1 = BENCH / "jit1.nl"


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _JIT1.exists(), reason="jit1.nl (benchmark corpus) absent")
def test_828_jit1_solves_on_default_path():
    """jit1 solves to its optimum on the DEFAULT (pure POUNCE) path — pounce#258 plus
    the status-3 remap remove the false-UNBOUNDED cascade. Before the fix it returned
    status='unknown' with no incumbent."""
    model = dm.from_nl(str(_JIT1))
    r = model.solve(time_limit=10)
    assert r.objective is not None, "#828: jit1 returned no incumbent (unknown) on the default path"
    # oracle optimum 173983.33 (SCIP)
    assert abs(r.objective - 173983.33) < 5.0, (
        f"#828: jit1 objective {r.objective} != optimum 173983.33"
    )
    # and the incumbent is genuinely feasible (never a false primal)
    if r.x is not None:
        from discopt._jax.primal_heuristics import _check_constraint_feasibility

        ev = cached_evaluator(model)
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(r.x[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        assert _check_constraint_feasibility(ev, flat, tol=1e-3), (
            "#828: reported incumbent is infeasible in the original model"
        )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _JIT1.exists(), reason="jit1.nl (benchmark corpus) absent")
def test_828_jit1_nodes_no_longer_spurious_unbounded():
    """Pins the mechanism: with the (removed) cyipopt retry unavailable, raw POUNCE
    must NOT return UNBOUNDED on jit1's node NLPs (pre-#258 it was 59/59 UNBOUNDED)."""
    import discopt.solvers.nlp_ipopt as NI
    import discopt.solvers.nlp_pounce as NP
    from discopt.solvers import SolveStatus

    # make any latent cyipopt retry a no-op so RAW POUNCE verdicts surface
    _orig_ip = NI.solve_nlp
    NI.solve_nlp = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("retry disabled for test"))
    _orig = NP.solve_nlp
    counts = {"unbounded": 0, "total": 0}

    def _wrap(*a, **k):
        r = _orig(*a, **k)
        counts["total"] += 1
        if r.status == SolveStatus.UNBOUNDED:
            counts["unbounded"] += 1
        return r

    NP.solve_nlp = _wrap
    try:
        r = dm.from_nl(str(_JIT1)).solve(time_limit=10)
    finally:
        NP.solve_nlp = _orig
        NI.solve_nlp = _orig_ip
    assert counts["unbounded"] == 0, (
        f"#828/pounce#258: {counts['unbounded']}/{counts['total']} jit1 node NLPs returned "
        "spurious UNBOUNDED on raw POUNCE"
    )
    assert r.objective is not None and abs(r.objective - 173983.33) < 5.0, (
        f"#828: jit1 did not solve on pure POUNCE (obj={r.objective})"
    )
