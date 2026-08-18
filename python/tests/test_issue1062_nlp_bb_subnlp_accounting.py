"""Regression test for issue #1062 — the NLP-BB path reported no sub-NLP work.

#1062 was opened on the observation that convexity-certified MINLPs come back
with ``subnlp_calls=0``, read as "the sub-NLP primal heuristic never fires on
convex models". The measurement that settled it: ``solve_model`` auto-selects
``_solve_nlp_bb`` for those models and returns from there, and ``_solve_nlp_bb``
built its ``SolveResult`` without ever setting the three sub-NLP counters. They
therefore defaulted to 0 on *every* solve routed to that path, no matter how
much sub-NLP work ran — on the syn/rsyn family the root ``one_hot_config_subnlp``
constructor calls ``subnlp`` 48 times per solve and the result still read 0.

So the headline metric was a §6 vacuous instrument: 0 could not distinguish "the
heuristic is disabled" from "the counter is not wired", and it was the second.
This pins the counters so that reading cannot silently return.

Gated on detected one-hot structure, never on a problem name (§2).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solver import solve_model


def _uneven_disjunction_model():
    """Two disjunctions of different sizes — the #823 constructor's own fixture.

    Nearest-rounding zeroes both groups, so plain ``subnlp`` declines and the
    point that reaches the tree comes from ``one_hot_config_subnlp``: exactly the
    sub-NLP work whose accounting this test pins.
    """
    m = dm.Model("uneven")
    y = m.binary("y", 5)
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="d1")
    m.subject_to(y[3] + y[4] == 1, name="d2")
    m.subject_to(x >= 5.0 - 4.0 * y[2], name="lo")
    m.minimize(x + y[3])
    return m


@pytest.mark.smoke
def test_nlp_bb_reports_the_subnlp_work_it_actually_did():
    """The NLP-BB SolveResult must carry the sub-NLP counters, not silent zeros."""
    res = solve_model(_uneven_disjunction_model(), nlp_bb=True, time_limit=30.0)

    assert res.nlp_bb is True, "test must exercise the NLP-BB path to be meaningful"
    assert res.objective is not None, "expected an incumbent from the disjunct constructor"

    # Before the #1062 fix all three were structurally 0 on this path.
    assert res.subnlp_calls > 0, (
        "NLP-BB ran the disjunct-selection sub-NLP constructor but reported "
        f"subnlp_calls={res.subnlp_calls}"
    )
    assert res.subnlp_feasible > 0
    assert res.subnlp_incumbent_updates > 0

    # The counters are counts of distinct things, and each is bounded by the
    # previous: a point cannot be injected without having been returned, and one
    # cannot be returned without a sub-NLP having been solved for it.
    assert res.subnlp_feasible <= res.subnlp_calls
    assert res.subnlp_incumbent_updates <= res.subnlp_feasible
