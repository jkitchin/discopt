"""Python-MILP-tree retirement step 6 (R4): an ``abs_gap_tolerance`` tighter than
``gap_tolerance`` keeps a MILP on the Rust engine.

It used to route to the Python MILP tree because the engine's relative gap floors its
denominator at 1.0 (#1243). The engine now applies the tree's own unfloored test,
``max(|ub|, |lb|, 1e-10)`` (``MilpOptions::rel_gap_max_abs``), whenever such a request
is made, so an ``optimal`` verdict still honours :func:`_gap_values_converged`.
"""

from __future__ import annotations

import discopt._rust as _rust
import discopt.modeling as dm
import pytest
from discopt.solver import _gap_values_converged


def _near_zero_milp():
    # min 0.001*(x - y) + 0.0013 * z over a small integer box: optimum near zero,
    # where the floored denominator and the unfloored one disagree by ~1000x.
    m = dm.Model("near_zero")
    x = m.integer("x", lb=0, ub=7)
    y = m.integer("y", lb=0, ub=7)
    z = m.integer("z", lb=0, ub=5)
    m.subject_to(3 * x - 2 * y + z >= 1.5)
    m.subject_to(x + y + z <= 9)
    m.minimize(0.001 * (x - y) + 0.0013 * z)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("abs_tol", [1e-9, 1e-12])
def test_tight_abs_gap_stays_on_the_engine_and_is_honoured(monkeypatch, abs_tol):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    seen: list[dict] = []
    real = _rust.solve_milp_csc_py

    def spy(*a, **kw):
        if "abs_gap_tol" in kw:  # the searches; not the integer-free root-LP probe
            seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", spy)
    r = _near_zero_milp().solve(time_limit=60, gap_tolerance=1e-4, abs_gap_tolerance=abs_tol)
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route
    assert "route/python_milp_tree_entry" not in (r.solver_stats or {})
    assert seen and all(kw.get("rel_gap_max_abs") is True for kw in seen), seen
    assert r.status == "optimal", r.status
    assert _gap_values_converged(r.objective, r.bound, 1e-4, abs_tol), (r.objective, r.bound)


@pytest.mark.smoke
def test_a_looser_abs_gap_keeps_the_floored_denominator(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    seen: list[dict] = []
    real = _rust.solve_milp_csc_py

    def spy(*a, **kw):
        if "abs_gap_tol" in kw:  # the searches; not the integer-free root-LP probe
            seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", spy)
    r = _near_zero_milp().solve(time_limit=60, gap_tolerance=1e-4, abs_gap_tolerance=1e-3)
    assert r.status == "optimal"
    assert seen and all(kw.get("rel_gap_max_abs") is False for kw in seen), seen
