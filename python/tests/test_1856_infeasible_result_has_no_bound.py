"""#1856: an ``infeasible`` SolveResult carried ``bound=-10.058, bound_valid=True``.

The NLP-BB route built its infeasible result with the last frontier minimum as
``bound``, and ``__post_init__`` then derived ``bound_valid=True`` from the
certified status. The documented contract for an infeasibility certificate is
``bound=None`` / ``bound_valid=False``; it is now enforced at the chokepoint every
``SolveResult`` passes through, and by ``_set_bound``.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult


@pytest.mark.smoke
def test_constructor_normalises_an_infeasible_result():
    r = SolveResult(status="infeasible", bound=-10.058, gap=0.5, gap_certified=True)
    assert r.bound is None and r.gap is None
    assert r.bound_valid is False and r.bound_source is None
    assert r.gap_certified is True  # the infeasibility certificate itself stands


@pytest.mark.smoke
def test_set_bound_cannot_install_a_bound_on_infeasible():
    r = SolveResult(status="infeasible", gap_certified=True)
    r._set_bound(-3.0, valid=True, source="bnb_tree")
    assert r.bound is None and r.bound_valid is False and r.bound_source is None


@pytest.mark.smoke
def test_other_statuses_keep_their_bound():
    r = SolveResult(status="time_limit", objective=None, bound=-3.0, bound_valid=True)
    assert r.bound == -3.0 and r.bound_valid is True


@pytest.mark.slow
def test_issue_repro_nlp_bb_route():
    # 6 disjunctions of two radius-2 balls each, infeasible jointly (SciPy
    # enumeration of all 64 selections: min max-violation 22.41).
    rng = np.random.default_rng(0)
    n = 6
    m = dm.Model()
    x = m.continuous("x", shape=(n,), lb=0, ub=10)
    for _ in range(6):
        c1, c2 = rng.uniform(1, 9, n), rng.uniform(1, 9, n)
        m.either_or(
            [
                [sum((x[i] - c1[i]) ** 2 for i in range(n)) <= 4],
                [sum((x[i] - c2[i]) ** 2 for i in range(n)) <= 4],
            ]
        )
    cc = rng.integers(-3, 4, n)
    m.minimize(sum(float(cc[i]) * x[i] for i in range(n)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=120)
    if r.status == "infeasible":
        assert r.bound is None and r.bound_valid is False
    else:
        assert r.status in ("unknown", "time_limit", "node_limit")
