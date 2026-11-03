"""#2120: under ``deterministic=True`` a per-NLP wall cap must not decide the answer.

``continuous_multistart`` handed each start ``max_wall_time=min(3.0, remaining)``,
the node NLP a fixed 30 s, the seed relaxation 4 s. Those are role-2 budgets ("a
fixed number of seconds handed to a stage"). ``deterministic`` neutralized the
*stage* deadlines but not these per-solve caps, and POUNCE times them itself in
Rust, so no Python-side clock patch can reach them. On a stalled CI runner one cap
bound, the NLP returned a different point, and ``test_1513``'s dispersion solve
drifted by ~1e-8 at an identical node count.

The emulated stall below divides every ``max_wall_time`` POUNCE receives by
``K``, which is the same as a machine ``K`` times slower relative to the caps.
Before the fix the dispersion solve returned ``1.0717967060943734`` at ``K=1000``
against the reproducible ``1.071796754113688``. Shrinking only the 3 s caps also
reproduced ``1.0717967670184763`` bit for bit, the value CI's coverage lane reported
in run 41264123265.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt import solver_tuning as st

pounce = pytest.importorskip("pounce")

_CAP = dict(time_limit=600, deterministic=True, max_nodes=60)
_STALL = 1000.0


def _dispersion() -> dm.Model:
    """The ``test_1513`` model: 3 points in the unit square, maximize min sq-distance."""
    m = dm.Model("disp3")
    X = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(3)]
    Y = [m.continuous(f"y{i}", lb=0.0, ub=1.0) for i in range(3)]
    t = m.continuous("t", lb=0.0, ub=2.0)
    for i in range(3):
        for j in range(i + 1, 3):
            m.subject_to((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 >= t)
    m.maximize(t)
    return m


def _install_stall(monkeypatch, factor: float) -> dict:
    """Divide every ``max_wall_time`` POUNCE receives by ``factor``; count them."""
    seen = {"caps": 0}
    add_option = pounce.Problem.add_option

    def _add(self, key, value):
        if key == "max_wall_time":
            seen["caps"] += 1
            value = float(value) / factor
        return add_option(self, key, value)

    batch = pounce.solve_nlp_batch

    def _batch(*args, **kwargs):
        opts = dict(kwargs.get("options") or {})
        if "max_wall_time" in opts:
            seen["caps"] += 1
            opts["max_wall_time"] = float(opts["max_wall_time"]) / factor
            kwargs["options"] = opts
        return batch(*args, **kwargs)

    monkeypatch.setattr(pounce.Problem, "add_option", _add)
    monkeypatch.setattr(pounce, "solve_nlp_batch", _batch)
    return seen


def test_deterministic_solve_is_invariant_to_a_stalled_subsolve_clock(monkeypatch):
    """A 1000x stall relative to every POUNCE cap leaves the answer bit-identical."""
    ref = _dispersion().solve(**_CAP)
    seen = _install_stall(monkeypatch, _STALL)
    stalled = _dispersion().solve(**_CAP)
    # The probe fired: POUNCE was handed wall caps and every one was stalled.
    assert seen["caps"] > 0
    assert stalled.node_count == ref.node_count
    assert stalled.status == ref.status
    assert stalled.objective == ref.objective  # bit-identical, not approx (#2120)


def test_subsolve_wall_is_unchanged_off_deterministic(monkeypatch):
    monkeypatch.delenv("DISCOPT_DETERMINISTIC", raising=False)
    assert st.subsolve_wall_seconds(3.0) == 3.0
    opts = {"max_wall_time": 3.0}
    assert st.subsolve_wall_options(opts) == {"max_wall_time": 3.0}


def test_subsolve_wall_is_role1_remainder_under_deterministic(monkeypatch):
    import time

    monkeypatch.setenv("DISCOPT_DETERMINISTIC", "1")
    # No role-1 deadline: the cap is the only stall backstop left, so it stays.
    assert st.subsolve_wall_seconds(3.0) == 3.0
    opts = {"max_wall_time": 3.0, "max_iter": 50}
    assert st.subsolve_wall_options(opts) == {"max_wall_time": 3.0, "max_iter": 50}
    # A role-1 deadline 100 s out: the 3 s cap becomes the remainder.
    token = st.ROLE1_DEADLINE.set(time.perf_counter() + 100.0)
    try:
        wall = st.subsolve_wall_seconds(3.0)
        assert wall is not None and 90.0 < wall <= 100.0
        # An expired role-1 deadline still stops the sub-solve (floored, never 0).
        st.ROLE1_DEADLINE.set(time.perf_counter() - 5.0)
        assert st.subsolve_wall_seconds(3.0) == pytest.approx(1e-3)
    finally:
        st.ROLE1_DEADLINE.reset(token)
    # A dict without the key is left alone.
    assert st.subsolve_wall_options({"max_iter": 5}) == {"max_iter": 5}


def test_solver_role1_deadline_is_the_shared_contextvar():
    """``solver._ROLE1_DEADLINE`` is the object the backends read, not a copy."""
    import discopt.solver as S

    assert S._ROLE1_DEADLINE is st.ROLE1_DEADLINE
