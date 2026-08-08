"""Issue #941, end to end: index spelling must not change the answer.

``v[-1]`` and ``v[3]`` name the same element of a shape-(4,) variable, so two
models differing only in that spelling are the *same model*. Before the fix the
structure layers resolved the negative form to the slot belonging to a different
variable, the McCormick envelope was built over a bilinear pair the model does
not contain, and the search returned ``status="optimal"`` with
``gap_certified=True`` on a bound above a demonstrably feasible point.

These are solve-level tests, so they are marked ``slow``. The unit-level
guarantee they rest on lives in ``test_941_flat_index.py``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _geom(last: int) -> dm.Model:
    """The reported reproducer. ``last`` spells the final element -1 or 3."""
    m = dm.Model("geom941")
    s = m.continuous("s", lb=0.2, ub=3.0)
    v = m.continuous("v", shape=(4,), lb=0.2, ub=3.0)
    m.minimize(sum(dm.exp(v[i]) / (1.0 + v[(i + 1) % 4]) for i in range(4)) + dm.log(s))
    m.subject_to(sum(v[i] ** 2 for i in range(4)) + s <= 9.0)
    m.subject_to(v[0] * v[last] >= 0.5)
    return m


def _geom_truth(x: np.ndarray) -> tuple[float, float, float]:
    """(objective, slack_1, slack_2) in plain numpy — independent of the solver."""
    s, v = float(x[0]), np.asarray(x[1:], dtype=float)
    obj = float(sum(np.exp(v[i]) / (1.0 + v[(i + 1) % 4]) for i in range(4)) + np.log(s))
    return obj, 9.0 - (float(np.sum(v**2)) + s), float(v[0] * v[3]) - 0.5


def _flat_solution(model: dm.Model, result) -> np.ndarray:
    return np.concatenate(
        [np.atleast_1d(np.asarray(result.x[v.name], dtype=float)).ravel() for v in model._variables]
    )


@pytest.mark.slow
def test_negative_index_does_not_certify_a_false_bound():
    """The reported failure: bound 3.4854 certified while 2.9619 is feasible.

    Asserted as the *certificate invariant* rather than as a fixed number, so the
    test keeps its meaning as the solver improves: a certified bound may never
    exceed an objective the model actually attains.
    """
    model = _geom(-1)
    res = model.solve(time_limit=60)

    assert res.objective is not None, "no incumbent — the reproducer should be feasible"
    x = _flat_solution(model, res)
    obj, slack_1, slack_2 = _geom_truth(x)

    # The returned point must really be feasible, re-checked outside the solver.
    assert slack_1 >= -1e-6, f"returned point violates sum(v^2)+s <= 9 by {-slack_1}"
    assert slack_2 >= -1e-6, f"returned point violates v0*v3 >= 0.5 by {-slack_2}"
    assert obj == pytest.approx(res.objective, rel=1e-6), "reported objective is not the point's"

    # The known-feasible point from the positive spelling. Any certified bound
    # above this is false: a minimization cannot prove a floor above a value it
    # can reach.
    known_feasible = 2.9618839111
    if res.gap_certified and res.bound is not None and np.isfinite(res.bound):
        assert res.bound <= known_feasible + 1e-5, (
            f"certified bound {res.bound} exceeds the feasible objective "
            f"{known_feasible} — a false certificate"
        )
    assert obj <= known_feasible + 1e-4, (
        f"negative-index spelling returned a worse incumbent ({obj}) than the "
        f"positive spelling ({known_feasible}); the two are the same model"
    )


@pytest.mark.slow
def test_both_spellings_agree():
    """`v[0]*v[-1]` and `v[0]*v[3]` are one model and must give one answer.

    Node counts are deliberately NOT compared: both arms are wall-clock limited
    here, so their node totals reflect machine speed, not the search.
    """
    results = {}
    for last in (-1, 3):
        model = _geom(last)
        res = model.solve(time_limit=60)
        x = _flat_solution(model, res)
        obj, slack_1, slack_2 = _geom_truth(x)
        assert slack_1 >= -1e-6 and slack_2 >= -1e-6, f"v[{last}] arm returned an infeasible point"
        results[last] = obj

    assert results[-1] == pytest.approx(results[3], rel=1e-4), (
        f"index spelling changed the optimum: v[-1] -> {results[-1]}, v[3] -> {results[3]}"
    )


@pytest.mark.slow
def test_negative_index_bilinear_is_actually_seen_by_the_structure_layer():
    """Anti-vacuity (CLAUDE.md §6): prove the bilinear term is catalogued.

    Without this, the solve-level tests above could pass for the wrong reason —
    a model whose nonconvexity is never detected has nothing to get wrong.
    """
    from discopt._jax.term_classifier import classify_nonlinear_terms

    model = _geom(-1)
    terms = classify_nonlinear_terms(model)

    # `s` is slot 0 and `v` occupies slots 1-4, so v[0]*v[-1] is the pair (1, 4).
    assert (1, 4) in terms.bilinear, (
        f"v[0]*v[-1] should be catalogued as the bilinear pair (1, 4); "
        f"got {terms.bilinear}. (0, 1) would be the #941 aliasing onto `s`."
    )
    assert (0, 1) not in terms.bilinear, "pair (0, 1) aliases `s` — the #941 defect"
