"""A node whose NLP point the feasibility gate refuses must not be fathomed silently.

Found while verifying #1284 on clay0303hfsg. ``_solve_nlp_bb`` excluded a node
whose converged NLP point the gate refused (the ``_INFEASIBILITY_SENTINEL``) and
set ``_unconverged_fathom``. That flag only guards the no-incumbent exit, so with
an incumbent from another branch the solve reported ``optimal`` over a subtree
nobody searched: 28862 against the recorded optimum 26669. A refusal is not an
infeasibility proof. On a convex model the node now abstains (inherited parent
bound, still branched); otherwise the gap is decertified.

The gate is patched to refuse every point at ``(y1, y2) = (0, 1)``, the optimum,
so the test exercises the exclusion arm whatever the gate's tolerances are. The
root relaxation is fractional, and the ``y1 = 1`` branch supplies an incumbent
(objective -1), which is what the refused branch was certified against.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest

TRUE_OPT = -1.1  # x = 0, y1 = 0, y2 = 1


def _model():
    m = dm.Model("refused")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y1 = m.binary("y1")
    y2 = m.binary("y2")
    m.subject_to(y1 + y2 <= 1.5)
    m.subject_to(x**2 <= 1.0)
    m.minimize(x**2 - y1 - 1.1 * y2)
    return m


@pytest.mark.parametrize("abstain", ["1", "0"])
def test_refused_node_point_never_yields_a_false_certificate(monkeypatch, abstain):
    monkeypatch.setenv("DISCOPT_CONVEX_STALL_ABSTAIN", abstain)
    real = solver_mod._check_constraint_feasibility
    refused = {"n": 0}

    def gate(evaluator, x, cl_list, cu_list, tol=1e-4):
        xv = np.asarray(x, dtype=np.float64).ravel()
        if xv.size == 3 and abs(xv[1]) < 1e-3 and abs(xv[2] - 1.0) < 1e-3:
            refused["n"] += 1
            return False
        return real(evaluator, x, cl_list, cu_list, tol)

    monkeypatch.setattr(solver_mod, "_check_constraint_feasibility", gate)
    res = _model().solve(time_limit=30, nlp_bb=True)
    assert refused["n"] > 0, "the patched gate never saw the optimal branch"
    # The refused branch holds the optimum, so no incumbent reaches it; the
    # solve may keep the y1 = 1 point (objective -1) but must not certify it.
    assert res.status != "infeasible", res.status
    if res.bound_valid:
        assert res.bound <= TRUE_OPT + 1e-6, (res.status, res.objective, res.bound)
    assert not (res.status == "optimal" and res.objective > TRUE_OPT + 1e-6), (
        res.status,
        res.objective,
        res.bound,
    )
