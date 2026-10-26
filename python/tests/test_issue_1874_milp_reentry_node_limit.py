"""Issue #1874 (1): the native-MILP re-entry threw away a better incumbent on ``node_limit``.

``_solve_milp_simplex`` runs the Rust engine twice: run 1 on a capped clock slice,
then, from a ``feasible`` exit, a re-entry on the remainder. The engine reports
``NodeLimit`` *with* an incumbent when a run hits ``max_nodes`` (``decide_status``'s
``node_limit_hit`` arm), but the re-entry gate admitted only ``optimal``/``feasible``.
Measured on tln4 at ``time_limit=120``: run 2 ended ``node_limit`` holding 8.7 and the
solve reported run 1's 8.9 (optimum 8.3). With the fix it reports 8.7, which
``verify_point`` accepts on a freshly read model.

nvs/tln are not vendored and the real trigger needs a long run, so the engine's two
answers are scripted here around a REAL first call: run 1 is told to claim a worse
incumbent value, run 2 returns the true optimum with status ``node_limit``.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import _rust
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _legacy_lift_adoption(monkeypatch):
    """Pin the legacy integer-bilinear adoption rule (#1940's ``DISCOPT_IPX_RACE=0``).

    These tests script the native MILP engine that serves the LIFTED model. Under the
    #1940 race, the scripted uncertified lifted slice is followed by an un-lifted
    slice that certifies the model on its own, so the merged result no longer shows
    what the re-entry did. The race is covered by ``test_1940_lift_race.py``; here the
    lift must be the whole solve.
    """
    monkeypatch.setenv("DISCOPT_IPX_RACE", "0")


def _milp_route_model() -> Model:
    """Integer-bilinear, small ranges: linearised and served by the native MILP engine."""
    m = Model("reentry")
    x = [m.integer(f"x{i}", lb=0, ub=5) for i in range(4)]
    m.subject_to(x[0] + x[1] + x[2] + x[3] <= 9)
    m.subject_to(x[0] * x[1] - x[2] * x[3] <= 4)
    m.minimize(x[0] * x[2] - 3 * x[0] * x[1] + (x[3] - 2) ** 2 + x[1])
    return m


def _fixed_point_run(real, a, kw, point):
    """Run the REAL engine with the structural columns fixed to ``point``: a genuine
    feasible incumbent of the original model with a consistent lifted vector."""
    a = list(a)
    k = point.size  # the user's variables lead the (reformulated) column order
    xl = np.array(a[7], dtype=np.float64)
    xu = np.array(a[8], dtype=np.float64)
    xl[:k] = point
    xu[:k] = point
    a[7], a[8] = np.ascontiguousarray(xl), np.ascontiguousarray(xu)
    return real(*a, **kw)


ZEROS = np.zeros(4)  # feasible (0 <= 9, 0 - 0 <= 4), objective (0-2)**2 = 4 > optimum


def _script(monkeypatch, second):
    """Run 1 claims the worse ZEROS incumbent with the real run's (valid) bound; run 2
    is ``second(true_result)``."""
    real = _rust.solve_milp_csc_py
    seen = {"n": 0, "true": None, "worse": None}

    def scripted(*a, **kw):
        if a[12] <= 1000:  # auxiliary sub-MILPs keep their real answer
            return real(*a, **kw)
        seen["n"] += 1
        out = real(*a, **kw)
        status, x, obj, bound, nodes, iters = out
        assert np.isfinite(obj), "precondition: the real engine found an incumbent"
        if seen["n"] == 1:
            seen["true"] = obj
            w = _fixed_point_run(real, a, kw, ZEROS)
            assert w[0] == "optimal" and np.isfinite(w[2]) and w[2] > obj + 1e-6, w[:3]
            seen["worse"] = w[2]
            return ("feasible", w[1], w[2], bound, nodes, iters)
        return second(out)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", scripted)
    return seen


def test_reentry_node_limit_with_a_better_incumbent_is_adopted(monkeypatch):
    seen = _script(monkeypatch, lambda o: ("node_limit",) + tuple(o[1:]))
    r = _milp_route_model().solve(time_limit=60)
    assert (r.algorithm_route or "").startswith("native-milp"), r.algorithm_route
    assert seen["n"] == 2, "the re-entry did not run -- the probe measured nothing"
    assert r.objective == pytest.approx(seen["true"]) and r.objective < seen["worse"] - 1e-6
    # Still an uncertified exit: a node-limited run proves nothing.
    assert r.status == "feasible" and not r.gap_certified


def test_reentry_node_limit_without_an_incumbent_changes_nothing(monkeypatch):
    """No finite incumbent on the node_limit exit: run 1's point stands."""
    seen = _script(monkeypatch, lambda o: ("node_limit", o[1], float("inf")) + tuple(o[3:]))
    r = _milp_route_model().solve(time_limit=60)
    assert seen["n"] == 2
    assert r.objective == pytest.approx(seen["worse"])
