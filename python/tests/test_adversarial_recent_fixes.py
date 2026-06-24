"""Adversarial regression suite for the soundness / crash / deadline fixes merged
in the week of 2026-06-18..24.

Two layers:

1. **Real triggering instances** (``_INSTANCES``) — the exact MINLPLib models that
   each bug was found on, so they provably exercise the fixed code path, checked
   against their BARON-confirmed optima (minlplib.solu). These are the gold
   standard: before the fix each returned an *unsound* result (false-feasible,
   false-infeasible, false-unbounded, or false-optimal); the suite asserts the
   sound outcome.

2. **Synthetic path-targeted problems** — constructed to hit a specific fixed code
   path that no small vendored instance covers (the OA maximize loop; the dense
   Jacobian XLA-compile guard on a > 1e6-entry model).

Soundness invariants (sense-aware), asserted everywhere:
  * not false-infeasible / not false-unbounded
  * dual bound on the correct side of the true optimum (a valid bound never
    crosses it)            -> catches false certificates (#277, #306)
  * incumbent never beats the true optimum   -> catches false-feasible (#310)
  * a gap=0 "optimal" sits at the true optimum -> catches false-optimal (#301)
  * the process survives (no native crash)     -> #313

Marked ``slow``; run with ``-m slow``.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import ObjectiveSense  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")

# (instance, true optimum, sense, time_limit, fix, what the bug did)
_INSTANCES = [
    (
        "ex1252a",
        128893.741,
        "min",
        40,
        "#310",
        "false-feasible 92117 (obj-only ints typed continuous)",
    ),
    (
        "hda",
        -5964.534084,
        "min",
        40,
        "#307",
        "false-infeasible (inverted fractional-power aux box)",
    ),
    (
        "ex8_5_4",
        -0.0004251471,
        "min",
        25,
        "#270",
        "false-infeasible (free var into undefined log domain)",
    ),
    (
        "carton7",
        191.7295481,
        "min",
        40,
        "#288/#289",
        "false-unbounded (dropped nonlinear bound in projection)",
    ),
    (
        "st_ph10",
        -10.5,
        "min",
        25,
        "#306",
        "false-optimal -28.06 (incumbent below its own dual bound)",
    ),
    (
        "nvs22",
        6.05822,
        "min",
        25,
        "#277",
        "false certificate (ill-conditioned OBBT pruned the optimum)",
    ),
    ("nvs12", -481.2, "min", 40, "#293", "unbounded hang (simplex MILP engine ignored time_limit)"),
]

_REL = 5e-3  # relative tolerance band for "beats the optimum" / "is the optimum"
_BND = 1e-2  # absolute slack for dual-bound-side checks (numerical)


def _band(opt: float) -> float:
    return max(_REL * abs(opt), 1e-4)


def _assert_sound(name, r, *, sense, opt, may_lack_incumbent=True):
    """Sense-aware soundness assertions against the oracle optimum ``opt``."""
    obj, bnd, status = r.objective, r.bound, r.status
    tol = _band(opt)

    assert status != "infeasible", f"{name}: FALSE-INFEASIBLE (instance is feasible)"
    assert status != "unbounded", f"{name}: FALSE-UNBOUNDED (instance is bounded)"

    if not may_lack_incumbent:
        assert obj is not None, f"{name}: no incumbent returned"

    if obj is not None:
        # No false-feasible: the incumbent must be a real feasible point, never
        # strictly better than the proven global optimum.
        if sense == "min":
            assert obj >= opt - tol, f"{name}: FALSE-FEASIBLE — {obj:.6g} < opt {opt:.6g}"
        else:
            assert obj <= opt + tol, f"{name}: FALSE-FEASIBLE — {obj:.6g} > opt {opt:.6g}"
        # No false-optimal: a certified optimum must sit at the true optimum.
        if status == "optimal":
            assert abs(obj - opt) <= tol, f"{name}: FALSE-OPTIMAL — {obj:.6g} != opt {opt:.6g}"

    # Sound dual bound: a valid lower (min) / upper (max) bound never crosses the
    # true optimum, and never crosses the incumbent.
    if bnd is not None:
        if sense == "min":
            assert bnd <= opt + tol + _BND, f"{name}: INVALID BOUND {bnd:.6g} > opt {opt:.6g}"
            if obj is not None:
                assert bnd <= obj + tol + _BND, (
                    f"{name}: UNSOUND CERT bound {bnd:.6g} > inc {obj:.6g}"
                )
        else:
            assert bnd >= opt - tol - _BND, f"{name}: INVALID BOUND {bnd:.6g} < opt {opt:.6g}"
            if obj is not None:
                assert bnd >= obj - tol - _BND, (
                    f"{name}: UNSOUND CERT bound {bnd:.6g} < inc {obj:.6g}"
                )


@pytest.mark.slow
@pytest.mark.parametrize("name,opt,sense,tl,fix,bug", _INSTANCES, ids=[i[0] for i in _INSTANCES])
def test_triggering_instance_is_sound(name, opt, sense, tl, fix, bug):
    """Each instance that triggered a recent bug must now return a sound result
    within the time limit (and the process must not crash)."""
    path = os.path.join(_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")
    t = time.perf_counter()
    r = dm.from_nl(path).solve(time_limit=tl, gap_tolerance=1e-4)
    wall = time.perf_counter() - t
    # Deadline honored (generous margin for one uninterruptible compile/solve that
    # straddles the deadline — the known diffuse residual, not a hang). #293/#311/#314.
    assert wall < tl + 60, f"{name}: ran {wall:.0f}s on a {tl}s limit (hang?) — {fix}"
    _assert_sound(name, r, sense=sense, opt=opt)


@pytest.mark.slow
def test_oa_maximize_is_sound():
    """OA on a MAXIMIZE convex MINLP must return the true maximum, not a negated /
    wrong point reported as 'optimal' (#301). Real trigger: syn05m (=max= 837.7324)."""
    from discopt.solvers.oa import solve_oa

    path = os.path.join(_DATA, "syn05m.nl")
    if not os.path.exists(path):
        pytest.skip("syn05m.nl not vendored")
    m = dm.from_nl(path)
    assert m._objective.sense == ObjectiveSense.MAXIMIZE
    r = solve_oa(m, time_limit=40, gap_tolerance=1e-4)
    _assert_sound("syn05m/OA", r, sense="max", opt=837.7324009)


@pytest.mark.slow
def test_oa_maximize_synthetic_concave():
    """Synthetic convex-MINLP MAXIMIZE that directly drives the OA loop (#301):
    max -(x-3)**2 - (y-2.5)**2, x in [0,5], y in {0..5}, x+y<=10  ->  -0.25 at y in {2,3}."""
    from discopt.solvers.oa import solve_oa

    m = dm.Model("oa_max")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(-((x - 3.0) ** 2) - (y - 2.5) ** 2)
    m.subject_to(x + y <= 10.0)
    r = solve_oa(m, time_limit=30, gap_tolerance=1e-5)
    _assert_sound("oa-synthetic", r, sense="max", opt=-0.25, may_lack_incumbent=False)


@pytest.mark.slow
def test_large_dense_jacobian_no_crash():
    """A sparse MINLP whose dense Jacobian (n_vars * n_constraints ~ 1.21e6) exceeds
    the 1e6 cap that crashed XLA's dense jacfwd compile (#313), with ~110 binaries
    (probing #313) and factorable equalities (#314). Must survive (no SIGBUS/SIGILL),
    honor the deadline within a margin, and return a sound certificate."""
    n = 1100
    m = dm.Model("big")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    bs = [m.binary(f"b{i}") for i in range(0, n, 10)]
    for i in range(n):
        m.subject_to(xs[i] * xs[i] + xs[(i + 1) % n] <= 10.0)
    for k, b in enumerate(bs):
        m.subject_to(xs[k] + 2.0 * b <= 6.0)
    m.minimize(dm.sum([xs[i] for i in range(n)]) - dm.sum(bs))
    budget = 8.0
    t = time.perf_counter()
    r = m.solve(time_limit=budget, gap_tolerance=1e-4)  # must not crash the process
    wall = time.perf_counter() - t
    assert wall < budget + 40, f"large model: deadline overrun wall={wall:.0f}s"
    assert r.status != "infeasible", "large model: FALSE-INFEASIBLE"
    # Sound certificate: a finite lower bound never exceeds the incumbent.
    if r.objective is not None and r.bound is not None:
        assert r.bound <= r.objective + 1e-3, "large model: UNSOUND CERT (bound > incumbent)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "slow", "-s"]))
