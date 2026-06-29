"""Tainted-tree-bound certification gap (#138).

On an uncertified *feasible* exit of the nonconvex spatial B&B, the tree's
frontier-minimum lower bound (``global_lower_bound``) is a valid global dual
bound, but it used to be discarded — dropped to ``None`` and replaced by the far
weaker root MILP-relaxation fallback (or nothing) — because the single
``_gap_certified`` flag conflated two distinct things: *the tree bound is
untainted* (no node fathomed without a soundness proof) vs *the gap is closed*
(optimality). A budget/node-limited feasible exit does not close the gap, yet its
untainted tree bound is still the best rigorous dual bound available.

Two changes fix this:

* The Rust tree (`import_results`) floors every node's imported bound at its
  inherited parent bound, so a node the Python orchestrator could not bound (the
  NLP objective is not a valid bound on a nonconvex node, or the per-node
  relaxation hit the deadline) retains its parent's valid bound instead of
  ``-inf`` — a single such open node no longer drags the whole dual bound to
  ``-inf``. Leaving a node unbounded fathoms nothing, so those cases no longer
  taint the tree.
* The finalize path keeps the untainted tree bound on a feasible exit (recomputing
  the gap) instead of dropping it, and re-earns ``optimal`` when it meets the
  incumbent.

The invariant under test is soundness: the reported bound is always a valid lower
bound (``<= true optimum`` for a minimize), never a false certificate.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.modeling.core import from_nl

_NVS17 = "python/tests/data/minlplib/nvs17.nl"
_NVS17_OPT = -1100.4  # best-known global optimum (minimize)


def _has(path: str) -> bool:
    return os.path.exists(path)


@pytest.mark.skipif(not _has(_NVS17), reason="nvs17.nl not vendored")
def test_uncertified_feasible_exit_reports_valid_finite_dual_bound():
    """A budget-limited nonconvex feasible exit surfaces the valid tree bound.

    Before #138 this reported ``bound=None`` (or the absurd root fallback, e.g.
    -5.5e5 vs an optimum of -1100). It must now be a finite, *valid* lower bound.
    """
    r = from_nl(_NVS17).solve(time_limit=4, gap_tolerance=1e-4)
    # nvs17 does not certify in a few seconds, so this exercises the feasible path.
    assert r.status in ("feasible", "optimal")
    if r.status == "feasible":
        assert r.bound is not None, "uncertified feasible exit dropped a valid bound"
        assert np.isfinite(r.bound), "reported a non-finite dual bound"
        # Soundness: a lower bound on a minimize is <= the true optimum.
        assert r.bound <= _NVS17_OPT + 1e-4 * max(1.0, abs(_NVS17_OPT))
        # And it must not exceed the incumbent it is paired with.
        assert r.objective is not None and r.bound <= r.objective + 1e-6
        # The gap is now a real, finite number (not None / inf).
        assert r.gap is not None and np.isfinite(r.gap)


@pytest.mark.skipif(not _has(_NVS17), reason="nvs17.nl not vendored")
def test_reported_bound_is_a_valid_lower_bound_not_a_false_certificate():
    """Whatever the exit status, a reported bound never exceeds the optimum and a
    *certified* gap is only claimed when the bound genuinely meets the incumbent."""
    r = from_nl(_NVS17).solve(time_limit=6, gap_tolerance=1e-4)
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= _NVS17_OPT + 1e-4 * max(1.0, abs(_NVS17_OPT))
    if r.gap_certified:
        # A certificate requires a finite bound that meets the incumbent.
        assert r.bound is not None and np.isfinite(r.bound)
        assert r.objective is not None
        assert abs(r.objective - r.bound) <= 1e-4 * max(1.0, abs(r.objective)) + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
