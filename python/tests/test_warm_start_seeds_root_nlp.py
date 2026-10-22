"""An infeasible ``initial_solution`` is still used as an NLP starting point.

A warm start that is not feasible is not injected as an incumbent, and spatial
B&B logged that it was "using it as an NLP starting point only" -- but no NLP was
ever started from it. On COPS ``pinene`` (5,005 columns, published start) the
root multistart's generic starts reached 27,101 at best in 120 s against the
reference 19.8721; a local solve from the published start reaches 19.87 in
0.8 s. The start now opens the root multistart, and where the multistart does
not run (the LP relaxer supplies the root bound) it seeds the root SubNLP.
"""

from __future__ import annotations

import discopt._relax.primal_heuristics as PH
import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest

START = np.array([1.37, -2.11, 0.5])  # x0*x1 + x2 = -2.39: violates the row


def _model():
    m = dm.Model("ws_seed")
    x = m.continuous("x", shape=(3,), lb=-4, ub=4)
    m.minimize(dm.sum(x**4 - 3 * x**2 + 0.5 * x))
    m.subject_to(x[0] * x[1] + x[2] >= 1.0)
    return m, x


@pytest.mark.parametrize("mc_mode", [None, "none"])
def test_infeasible_warm_start_starts_an_nlp(monkeypatch, mc_mode):
    seen: list[np.ndarray] = []
    node = S._solve_node_nlp
    sub = PH.subnlp

    def spy_node(ev, x0, *a, **k):
        seen.append(np.array(x0, dtype=float))
        return node(ev, x0, *a, **k)

    def spy_sub(model, x, *a, **k):
        seen.append(np.array(x, dtype=float))
        return sub(model, x, *a, **k)

    monkeypatch.setattr(S, "_solve_node_nlp", spy_node)
    monkeypatch.setattr(PH, "subnlp", spy_sub)
    m, x = _model()
    kw = {} if mc_mode is None else {"mccormick_bounds": mc_mode}
    # "none" (the Python tree's alphaBB route -- what the retired "nlp" mode was
    # demoted to on this nonconvex model) runs to the limit; the root seeds fire
    # in the first second either way.
    r = m.solve(time_limit=5, initial_solution={x: START}, **kw)
    assert seen, "no NLP ran -- the probe did not fire"
    assert any(p.size >= 3 and np.allclose(p[:3], START) for p in seen)
    assert r.objective is not None and np.isfinite(r.objective)


def test_root_multistart_tries_the_warm_start_first(monkeypatch):
    calls: list[np.ndarray] = []

    def fake(ev, x0, *a, **k):
        calls.append(np.array(x0, dtype=float))
        raise StopIteration

    monkeypatch.setattr(S, "_solve_node_nlp", fake)
    lb, ub = np.full(3, -4.0), np.full(3, 4.0)
    with pytest.raises(StopIteration):
        S._solve_root_node_multistart(
            None, lb, ub, [], {}, "pounce", initial_point=np.array([9.0, -1.0, 0.25])
        )
    np.testing.assert_array_equal(calls[0], [4.0, -1.0, 0.25])  # clipped into the box
    calls.clear()
    with pytest.raises(StopIteration):  # a start of the wrong length is skipped
        S._solve_root_node_multistart(None, lb, ub, [], {}, "pounce", initial_point=np.ones(2))
    np.testing.assert_array_equal(calls[0], np.zeros(3))  # the midpoint, as before
