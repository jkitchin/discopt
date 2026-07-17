"""Regression lock for GitHub issue #267: nonconvex global-search gap on two
additional MINLPLib families, closed by the #188 continuous stratified
multistart lever.

Issue #267 recorded two sound capability gaps (valid feasible incumbent,
correctly *not* certified, but short of the global optimum) as additional
coverage for the same class as #188 (``kall_congruentcircles_c51``):

    | instance             | sense | discopt (issue) | global   |
    |----------------------|-------|-----------------|----------|
    | kall_diffcircles_5a  | min   | 7.1867          | 5.1162   |
    | inscribedsquare02    | max   | 0.50885         | 0.968017 |

The resolution is the *general, class-level* lever shipped in #188 (PR #668):
``primal_heuristics.continuous_multistart``, a budgeted stratified multistart
run once at the root for pure-continuous nonconvex models on the McCormick-LP
path (``SolverTuning.continuous_multistart`` / ``DISCOPT_CONTINUOUS_MULTISTART``,
default ON). #188 validated it on the *congruent*-circle packing family; #267
adds two families it did NOT exercise:

* ``kall_diffcircles_5a`` — a **heterogeneous-radii** packing (a sibling of the
  ``kall_*circles`` family, but with unequal circle radii, so the #188
  congruent lock does not cover it). Same trap mechanism: a wide/short box
  admits a two-row local packing that the LP-vertex-seeded first local solve
  parks in, while the single-row global basin is only reachable by
  diversifying the starting point.
* ``inscribedsquare02`` — a **different problem family entirely** (largest
  square inscribed in a parametric closed curve; a maximization with
  ``sin``/``cos``/polynomial curve constraints). This is the important
  coverage: it shows the lever generalizes past circle packing.

Reconstructions (the MINLPLib ``.nl`` instances are not vendored, #163, and the
network-restricted CI cannot fetch them):

* ``inscribedsquare02`` is reconstructed **faithfully** from its actual curve
  ``t -> (sin(t)*cos(t - t*t), sin(t)*t)`` and square conditions (the same
  model used by ``test_issue_267_univariate_product_lift.py`` for the dual
  bound). It reproduces the published landscape exactly: the global
  ``0.968017`` *and* the ``~0.509`` local basin the issue reported discopt
  parking in (n_starts=32/seed=123 lands on ``0.50885`` to all published
  digits pre-diversification).
* ``kall_diffcircles`` is a **representative heterogeneous-radii packing**
  built on the same Kallrath generator #188 reconstructed (pattern from the
  vendored ``kall_circles_c8a.nl``), with five distinct radii. It is not
  digit-faithful to ``kall_diffcircles_5a`` (whose radii are not published in
  the repo; its MINLPLib global is 5.11618 on a different box variant) — it
  reproduces the *gap mechanism*: the default path parks in a two-row local
  packing (obj ~1.841) while the continuous multistart reaches the single-row
  global basin (obj 1.525520, stable to 5 digits across 10 seeds).

Soundness (heuristic-policy regime, CLAUDE.md §5): ``continuous_multistart`` is
a primal finder only — every point is constraint-re-verified and
``inject_incumbent`` enforces strict improvement, so dual bounds and the
certificate are untouched. The assertions below lock that too: the incumbent
never crosses the global in the certifying direction, and the dual bound stays
a valid under/over-estimator (``bound <= incumbent`` for min, ``>=`` for max).

No solver code changes here: this file only *verifies and locks* that the
already-merged #188 lever closes the #267 coverage instances.
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest

# Every test here runs a multi-second global search (stratified multistart or a
# full solve to a time limit); none belong in the sub-second default suite.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Family 1: heterogeneous-radii circle packing (kall_diffcircles sibling)
# ---------------------------------------------------------------------------

# Five distinct radii + a box wide enough for the single-row global yet tall
# enough to admit the two-row local trap (2*max_radius fits twice under B_MAX).
_DIFF_RADII = [0.45, 0.48, 0.50, 0.52, 0.55]
_DIFF_A_MAX = 5.0
_DIFF_B_MAX = 2.2
# Global-basin objective (wasted area = a*b - sum pi*r_i^2), reproduced to this
# many digits by continuous_multistart on every one of 10 seeds tried.
_DIFF_GLOBAL = 1.525520
# The two-row local packing the LP-vertex-seeded default path parks in with the
# multistart disabled (reproducible; assert a robust margin below it).
_DIFF_LOCAL_PARK = 1.841340


def make_diffcircles() -> "dm.Model":
    """Heterogeneous-radii Kallrath circle-packing reconstruction (see module
    doc). Minimize wasted area packing 5 unequal circles into an ``a x b``
    box."""
    radii = _DIFF_RADII
    n = len(radii)
    m = dm.Model(name="diffcircles_5_recon")
    xs = [m.continuous(f"x{i}", lb=radii[i], ub=_DIFF_A_MAX - radii[i]) for i in range(n)]
    ys = [m.continuous(f"y{i}", lb=radii[i], ub=_DIFF_B_MAX - radii[i]) for i in range(n)]
    a = m.continuous("a", lb=0.0, ub=_DIFF_A_MAX)
    b = m.continuous("b", lb=0.0, ub=_DIFF_B_MAX)
    t = m.continuous("t", lb=0.0, ub=_DIFF_A_MAX * _DIFF_B_MAX)

    m.subject_to(t - a * b == 0.0)
    # Pairwise separation with UNEQUAL radii: (r_i + r_j)^2, not a shared 2R.
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to(
                (xs[i] - xs[j]) * (xs[i] - xs[j]) + (ys[i] - ys[j]) * (ys[i] - ys[j])
                >= (radii[i] + radii[j]) ** 2
            )
    for i in range(n):
        m.subject_to(xs[i] - a <= -radii[i])
        m.subject_to(ys[i] - b <= -radii[i])
    m.subject_to(xs[0] <= _DIFF_A_MAX / 2.0)
    m.subject_to(ys[0] <= _DIFF_B_MAX / 2.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to(xs[i] - xs[j] <= 0.0)

    m.minimize(t - sum(math.pi * r * r for r in radii))
    return m


def test_diffcircles_multistart_reaches_global_basin_unit():
    """The lever itself must reach the heterogeneous-packing global basin.

    Deterministic (fixed seed, fixed n_starts — no wall-clock dependence), so
    this is the machine-independent hard lock. The single LP-vertex seed and the
    deterministic anchors converge to the ~1.841 two-row local; only start
    diversification reaches 1.525520.
    """
    from discopt._jax.primal_heuristics import continuous_multistart

    result = continuous_multistart(make_diffcircles(), n_starts=96, seed=42)
    assert result is not None, "continuous multistart found no feasible packing at all"
    _x, obj = result
    assert obj <= _DIFF_GLOBAL + 1e-3, (
        f"multistart best {obj!r} did not reach the global basin {_DIFF_GLOBAL} "
        f"(default path parks in the two-row local ~{_DIFF_LOCAL_PARK})"
    )
    assert np.all(np.isfinite(_x))


def test_diffcircles_default_path_reaches_global():
    """End-to-end: the DEFAULT path (multistart ON) reaches the global basin,
    and the incumbent/bound stay sound."""
    r = make_diffcircles().solve(time_limit=20, gap_tolerance=1e-4)
    assert r.objective is not None, "expected a feasible incumbent"
    assert r.objective <= _DIFF_GLOBAL + 1e-3, (
        f"incumbent {r.objective!r} did not reach the global {_DIFF_GLOBAL} "
        f"(#267 regression: parked in the two-row local ~{_DIFF_LOCAL_PARK})"
    )
    # Soundness: minimization dual bound must stay a valid underestimator.
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-6, (
            f"dual bound {r.bound!r} exceeds incumbent {r.objective!r} — "
            f"certificate invariant violated"
        )


def test_diffcircles_flag_off_parks_causality(monkeypatch):
    """Causal control: with the #188 lever DISABLED the default path parks in
    the two-row local packing — proving the multistart is what closes the gap
    (and that flag=0 restores the prior behavior)."""
    monkeypatch.setenv("DISCOPT_CONTINUOUS_MULTISTART", "0")
    r = make_diffcircles().solve(time_limit=20, gap_tolerance=1e-4)
    assert r.objective is not None
    # Parks materially above the global basin (observed ~1.841; robust margin).
    assert r.objective >= _DIFF_GLOBAL + 0.15, (
        f"flag-off incumbent {r.objective!r} unexpectedly close to the global "
        f"{_DIFF_GLOBAL} — the multistart is supposed to be the escaping lever"
    )
    # Still sound even when parked.
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-6


# ---------------------------------------------------------------------------
# Family 2: inscribedsquare02 (largest square inscribed in a parametric curve)
# ---------------------------------------------------------------------------

# MINLPLib =best= oracle for inscribedsquare02 (maximize squared side length).
_ISQ_GLOBAL = 0.968017
# Feasibility slack: a feasible point may undershoot the analytic optimum only
# by constraint-tolerance, never materially.
_FEAS_SLACK = 1e-3


def make_inscribedsquare02() -> "dm.Model":
    """Faithful inscribedsquare02 reconstruction: curve
    ``t -> (sin(t)*cos(t - t*t), sin(t)*t)`` with the four-point square
    conditions, maximizing the squared side length ``x4^2 + x5^2``. Same model
    as ``test_issue_267_univariate_product_lift._inscribedsquare02_model`` (which
    locks the dual bound); here we lock the *primal* global-search gap."""
    m = dm.Model(name="inscribedsquare02_recon")
    xs = [m.continuous(f"x{i}", lb=-np.pi, ub=np.pi) for i in range(4)]
    x4 = m.continuous("x4", lb=0.0, ub=2.0)
    x5 = m.continuous("x5", lb=0.0, ub=2.0)
    x6 = m.continuous("x6", lb=-1.0, ub=1.0)
    x7 = m.continuous("x7", lb=-np.pi, ub=np.pi)
    m.maximize(x4**2 + x5**2)
    m.subject_to(dm.sin(xs[0]) * dm.cos(xs[0] - xs[0] * xs[0]) - x6 == 0)
    m.subject_to(dm.sin(xs[0]) * xs[0] - x7 == 0)
    m.subject_to(dm.sin(xs[1]) * dm.cos(xs[1] - xs[1] * xs[1]) - x4 - x6 == 0)
    m.subject_to(dm.sin(xs[1]) * xs[1] - x5 - x7 == 0)
    m.subject_to(dm.sin(xs[2]) * dm.cos(xs[2] - xs[2] * xs[2]) + x5 - x6 == 0)
    m.subject_to(dm.sin(xs[2]) * xs[2] - x4 - x7 == 0)
    m.subject_to(dm.sin(xs[3]) * dm.cos(xs[3] - xs[3] * xs[3]) - x4 + x5 - x6 == 0)
    m.subject_to(dm.sin(xs[3]) * xs[3] - x4 - x5 - x7 == 0)
    return m


def test_inscribedsquare02_multistart_reaches_global_basin_unit():
    """The lever reaches the inscribed-square global basin on a DIFFERENT family
    than #188's circle packing.

    Deterministic hard lock. ``continuous_multistart`` minimizes the internal
    objective, which for this maximization is ``-(x4^2 + x5^2)``; the maximize
    value is therefore ``-obj``. This isolated-optima landscape has many small
    inscribed squares (e.g. the ~0.509 basin the issue reported); reaching the
    0.968017 global needs enough diversified starts.
    """
    from discopt._jax.primal_heuristics import continuous_multistart

    result = continuous_multistart(make_inscribedsquare02(), n_starts=128, seed=42)
    assert result is not None, "continuous multistart found no inscribed square at all"
    _x, internal_obj = result
    max_value = -internal_obj  # undo the maximize -> minimize negation
    assert max_value >= _ISQ_GLOBAL - _FEAS_SLACK, (
        f"multistart best {max_value!r} did not reach the global {_ISQ_GLOBAL} "
        f"(issue #267 reported parking at ~0.509)"
    )
    # Soundness: a feasible square can only touch/undershoot the true optimum.
    assert max_value <= _ISQ_GLOBAL + _FEAS_SLACK, (
        f"multistart value {max_value!r} exceeds the true optimum {_ISQ_GLOBAL} "
        f"beyond feasibility slack — unsound incumbent"
    )
    assert np.all(np.isfinite(_x))


def test_inscribedsquare02_default_path_reaches_global():
    """End-to-end: the DEFAULT path reaches the inscribedsquare02 global 0.968017
    (issue #267: was parking at 0.50885), and stays sound.

    A generous time limit is used so the root multistart has ample budget; the
    deterministic unit test above carries the wall-clock-independent claim.
    """
    r = make_inscribedsquare02().solve(time_limit=35, gap_tolerance=1e-4)
    assert r.objective is not None, "expected a feasible incumbent"
    # Clears every reported local basin (0.509 / 0.62 / 0.84) into the global.
    assert r.objective >= _ISQ_GLOBAL - 5e-3, (
        f"incumbent {r.objective!r} did not reach the global {_ISQ_GLOBAL} "
        f"(#267 regression: parked at a small inscribed square, e.g. 0.50885)"
    )
    # Soundness: maximize incumbent must never cross the true optimum, and the
    # (upper) dual bound must never fall below the incumbent.
    assert r.objective <= _ISQ_GLOBAL + _FEAS_SLACK, (
        f"incumbent {r.objective!r} exceeds the true optimum {_ISQ_GLOBAL} "
        f"beyond feasibility slack — unsound incumbent"
    )
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound >= r.objective - 1e-6, (
            f"upper dual bound {r.bound!r} is below the incumbent {r.objective!r} — "
            f"certificate invariant violated"
        )
