"""#2091: under ``deterministic=True`` two root heuristics got a *duration* as an
absolute ``time.perf_counter()`` deadline.

``_role2_horizon`` returns the remaining role-1 time (seconds) in deterministic
mode. The native kernel seed and the root continuous multistart passed it an
absolute time and compared the result with the clock, so once the machine had been
up longer than the time limit the deadline was already in the past and neither
heuristic did any work -- whether they ran depended on machine uptime, which is
exactly what ``deterministic=True`` excludes. Both sites now use
``_role2_deadline``, the absolute form.
"""

from __future__ import annotations

import dataclasses
import time
from unittest import mock

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke


def _seed_with(deterministic: bool, role1_deadline):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    seen: list = []

    def candidates(model, lb, ub, n_orig, deadline):
        seen.append(deadline)
        yield np.array([0.5])

    token = S._ROLE1_DEADLINE.set(role1_deadline)
    tuning = dataclasses.replace(S._tuning(), deterministic=deterministic)
    try:
        with (
            mock.patch.object(S, "_native_kernel_seed_candidates", candidates),
            mock.patch.object(S, "_tuning", lambda: tuning),
        ):
            out = S._native_kernel_seed(m, np.array([0.0]), np.array([1.0]), 1.0, 0.0, 1)
    finally:
        S._ROLE1_DEADLINE.reset(token)
    return out, seen


@pytest.mark.parametrize("deterministic", [False, True])
def test_native_seed_deadline_is_absolute(deterministic):
    now = time.perf_counter()
    (obj, point), seen = _seed_with(deterministic, now + 45.0)
    assert len(seen) == 1, "the seed must reach its candidate generator"
    # An absolute perf_counter() deadline in the future, not a duration.
    assert seen[0] > now
    assert obj == pytest.approx(0.5)
    np.testing.assert_allclose(point, [0.5])


def test_native_seed_deterministic_without_role1_clock_runs():
    """No role-1 deadline: ``_role2_deadline`` is ``None`` -> no clock, still runs."""
    (obj, _), seen = _seed_with(True, None)
    assert len(seen) == 1 and seen[0] == float("inf")
    assert obj == pytest.approx(0.5)


def test_root_multistart_deadline_is_absolute_in_deterministic_mode():
    import discopt._relax.primal_heuristics as ph

    orig = ph.continuous_multistart
    calls: list = []

    def spy(model, *a, deadline=None, **k):
        calls.append((deadline, time.perf_counter()))
        return orig(model, *a, deadline=deadline, **k)

    m = dm.Model()
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.subject_to(x**2 + y**2 >= 1)
    m.minimize((x - 0.3) * (y + 0.2) + dm.sin(3 * x) * dm.cos(2 * y))
    with mock.patch.object(ph, "continuous_multistart", spy):
        m.solve(time_limit=20, threads=1, deterministic=True, max_nodes=50)
    assert calls, "root continuous multistart did not run on this model"
    for deadline, now in calls:
        assert deadline > now, f"deadline {deadline} already past at {now}"
