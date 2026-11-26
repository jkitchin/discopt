"""#2356: the incumbent repair accepts a step on inconsistent fixed-integer equalities.

With the integers frozen, the equality rows of a point can have no exact
completion. The least-squares Newton step of ``repair_point`` must then leave the
residual somewhere. On MINLPLib ``portfol_roundlot`` it took eight lot rows
(``-lot_k * x_k + n_k = 0``) from up to 1.165e-6 to ~1e-12, while the budget row
(``sum x = 1``) went from 1.16504e-7 to 1.16550e-7. The old row-by-row rule rejected
the step for that 4.6e-11 growth, so the 1.165e-6 point was published. It passes
``verify_point``'s scale-keyed allowance but fails ``examine()``'s unscaled
``primal_con_feas`` at 1e-6.

The point is the OA incumbent of ``Model.solve(solver="mip-nlp",
mip_nlp_method="oa")`` on the #2350 + #2355 stack, captured to full precision.
``main`` visits a different integer assignment, so the solve itself does not
reproduce there. The repair does.
"""

from __future__ import annotations

import os

import numpy as np
from discopt._affine_evaluator import post_solve_evaluator
from discopt.modeling.core import from_nl
from discopt.validation.feasibility import ABS_TOL, repair_point, verify_point

NL = os.path.join(os.path.dirname(__file__), "data", "minlplib", "portfol_roundlot.nl")

_X = np.array(
    [
        8.7420999998834958e-01,
        9.2737988451748935e-14,
        9.2737683062278648e-14,
        9.2737199412169306e-14,
        9.2737725957246980e-14,
        8.4139999988349604e-02,
        3.0509090897440520e-02,
        1.1141025629375252e-02,
        2.8291790555213712e-02,
        87421.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8414.0,
        1678.0,
        869.0,
    ]
)


def _unscaled_violations(evaluator, model, x):
    g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
    out = []
    for (start, stop, con), gi in zip(evaluator.constraint_row_map(), g):
        assert stop - start == 1
        r = gi - float(con.rhs or 0.0)
        out.append(abs(r) if con.sense == "==" else max(r if con.sense == "<=" else -r, 0.0))
    return np.array(out)


def test_repair_fixes_inconsistent_lot_rows():
    m = from_nl(NL)
    ev = post_solve_evaluator(m)
    before = _unscaled_violations(ev, m, _X)
    # The fixture still is the case under test: a lot row past examine's 1e-6, and
    # an exact lot completion that misses the budget row (no exact completion).
    assert before.max() > ABS_TOL, before
    lots = np.array([1e5, 32000, 78000, 56000, 43000, 1e5, 55000, 78000])
    assert abs((_X[9:17] / lots).sum() - 1.0) > 1e-8

    rep = repair_point(m, _X, evaluator=ev)
    assert rep.x is not None, rep.reason
    assert rep.excess_after < rep.excess_before

    after = _unscaled_violations(ev, m, rep.x)
    assert after.size == before.size == 11
    # examine()'s unscaled check, unchanged: every row within 1e-6 in model units.
    assert after.max() <= ABS_TOL, after
    # Integers untouched, and the declared-model gate vouches for the moved point.
    assert np.array_equal(rep.x[9:17], _X[9:17])
    verdict = verify_point(m, rep.x, with_objective=True)
    assert verdict.ok, verdict.reason
    assert abs(verdict.objective - _X[8]) <= 1e-9
