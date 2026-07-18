"""#772 — the reported incumbent must be a valid primal for the ORIGINAL problem.

Regression guard for the class of bug #770 introduced: a presolve pass
(`DISCOPT_COEF_TIGHTEN`) that mutated the model DAG *unsoundly* (enlarged the
fixed-charge feasible set) let ``Model.solve`` return an incumbent that was
feasible in the mutated model but **infeasible in the original problem**, with an
objective *past* the proven optimum — a false primal value (CLAUDE.md §1, worst
class). The merged soundness test for #770 missed it because it only checked one
direction (no *feasible* point removed), never that no *infeasible* point is
admitted.

This test closes that gap generally: for each instance it solves with the shipped
defaults, then verifies the returned incumbent against a **freshly parsed** model
(so any in-place presolve mutation cannot hide the violation) AND against the
proven optimum (sense-aware — a feasible point can never beat the optimum). It
would fail on #770 and catches any future unsound presolve/heuristic that surfaces
a false primal on these instances.

Oracles are hard-coded (the MINLPLib ``.solu`` snapshot is not available in CI).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _check_constraint_feasibility
from discopt.modeling.core import ObjectiveSense, from_nl

_DATA = "python/tests/data/minlplib_nl"

# (instance, =opt=, is_maximize) — vendored in-repo instances with proven optima,
# spanning both senses and including a gated-configuration synthesis model
# (syn05hfsg, the #770-affected class).
_CASES = [
    ("syn05hfsg", 837.7324009, True),
    ("nvs03", 16.0, False),
    ("alan", 2.925, False),
    ("gbd", 2.2, False),
    ("ex1221", 7.667180, False),
]


def _incumbent_feasible_in_fresh_model(instance: str, x: dict) -> bool:
    """Evaluate the incumbent against a pristine ``from_nl`` model, aligned by
    variable NAME (never by the solved model's possibly-mutated ordering)."""
    fresh = from_nl(f"{_DATA}/{instance}.nl")
    ev = cached_evaluator(fresh)
    flat = np.concatenate(
        [np.atleast_1d(np.asarray(x[v.name], dtype=np.float64)).ravel() for v in fresh._variables]
    )
    return bool(_check_constraint_feasibility(ev, flat))


@pytest.mark.slow
@pytest.mark.parametrize("instance,opt,is_max", _CASES)
def test_reported_incumbent_is_a_valid_primal(instance, opt, is_max):
    """The incumbent ``Model.solve`` returns must be feasible in the ORIGINAL model
    and must not beat the proven optimum. Guards against unsound presolve/heuristics
    surfacing a false primal (#772; regression for the #770 class)."""
    model = from_nl(f"{_DATA}/{instance}.nl")
    assert model._objective.sense == (
        ObjectiveSense.MAXIMIZE if is_max else ObjectiveSense.MINIMIZE
    ), f"{instance} sense assumption wrong"

    r = model.solve(time_limit=20)
    if r.x is None or r.objective is None:
        pytest.skip(f"{instance}: no incumbent within budget (nothing to verify)")

    tol = 1e-4 * (1 + abs(opt))

    # (1) The incumbent is feasible in a freshly parsed original model.
    assert _incumbent_feasible_in_fresh_model(instance, r.x), (
        f"{instance}: reported incumbent is INFEASIBLE in the original problem "
        f"(false primal — the #770 failure mode)"
    )

    # (2) A feasible incumbent can never beat the proven optimum.
    if is_max:
        assert r.objective <= opt + tol, (
            f"{instance}: incumbent obj {r.objective} exceeds =opt= {opt} (impossible "
            f"for a feasible max point — false primal)"
        )
    else:
        assert r.objective >= opt - tol, (
            f"{instance}: incumbent obj {r.objective} below =opt= {opt} (impossible "
            f"for a feasible min point — false primal)"
        )
