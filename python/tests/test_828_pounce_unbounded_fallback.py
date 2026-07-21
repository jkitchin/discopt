"""#828: a spurious POUNCE UNBOUNDED status is overturned by a cyipopt retry.

jit1 is a convex reciprocal MINLP (objective ``Σ cᵢ/xᵢ`` for x>0 plus a badly
scaled linear tail, coefficients spanning ~10 to ~1e7). POUNCE returns a *spurious*
UNBOUNDED on its bounded continuous relaxation (pounce#248) — cyipopt solves the
identical problem to OPTIMAL. Because the node relaxations "fail", the NLP-BB
fathoms them non-rigorously and reports ``status=unknown`` with no incumbent, while
SCIP solves jit1 in <0.1s.

`nlp_pounce.solve_nlp` now retries once with the KKT-valid cyipopt backend when
POUNCE reports UNBOUNDED and adopts cyipopt's result only if it converges to
OPTIMAL — so jit1 solves on the default path, while a genuinely unbounded
relaxation (where cyipopt also fails to converge) keeps POUNCE's verdict.

Needs the benchmark corpus AND cyipopt (the fallback backend).
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt.solvers.nlp_backend import available_backends

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
_JIT1 = BENCH / "jit1.nl"
_HAS_CYIPOPT = "cyipopt" in available_backends()


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _JIT1.exists(), reason="jit1.nl (benchmark corpus) absent")
@pytest.mark.skipif(not _HAS_CYIPOPT, reason="cyipopt backend absent (fallback unavailable)")
def test_828_jit1_solves_on_default_path():
    """jit1 solves to its optimum on the DEFAULT (pounce) path — the UNBOUNDED
    retry overturns pounce's spurious status. Before the fix it returned
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
