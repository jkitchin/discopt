"""Integer-ratio partition bound (issue #309, ``DISCOPT_INTEGER_RATIO_PARTITION``).

The gear4 class: a constraint couples the objective to a quotient of products
of bounded integers, whose convex relaxation admits any target ratio at a
fractional point — the dual bound freezes at 0 and the tree near-enumerates.
The partition bound dives over the quotient's exactly-enumerable achievable
rational set; these tests pin its detection gates, its soundness (bound never
above the true optimum on brute-forceable boxes), the gear4 root-bound unlock
(the regression this feature exists for), and default-off byte-neutrality.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._jax.integer_ratio import (
    IntegerRatioPartitioner,
    detect_integer_ratio_specs,
)

GEAR4_TARGET = 144279.32477276
GEAR4_OPT = 1.643428


def _gear4() -> Model:
    m = Model("gear4")
    i1 = m.integer("i1", lb=12, ub=60)
    i2 = m.integer("i2", lb=12, ub=60)
    i3 = m.integer("i3", lb=12, ub=60)
    i4 = m.integer("i4", lb=12, ub=60)
    s = m.continuous("s", lb=0, ub=100000)
    t = m.continuous("t", lb=0, ub=100000)
    m.subject_to(1e6 * (i1 * i2) / (i3 * i4) + s - t == GEAR4_TARGET)
    m.minimize(s + t)
    return m


# ---------------------------------------------------------------------------
# Detection gates
# ---------------------------------------------------------------------------


def test_detects_gear4_ratio_spec():
    specs = detect_integer_ratio_specs(_gear4())
    assert [s.key for s in specs] == [((0, 1), (2, 3))]


def test_rejects_continuous_factor():
    m = Model("cont_num")
    x = m.continuous("x", lb=1, ub=10)
    y = m.integer("y", lb=1, ub=10)
    z = m.integer("z", lb=1, ub=10)
    s = m.continuous("s", lb=0, ub=100)
    m.subject_to((x * y) / z + s == 3.7)
    m.minimize(s)
    assert detect_integer_ratio_specs(m) == []


def test_rejects_sign_indefinite_denominator():
    m = Model("neg_den")
    x = m.integer("x", lb=1, ub=10)
    y = m.integer("y", lb=-5, ub=10)
    s = m.continuous("s", lb=0, ub=100)
    m.subject_to(x / y + s == 0.37)
    m.minimize(s)
    assert detect_integer_ratio_specs(m) == []


def test_rejects_unbounded_factor():
    m = Model("unbounded")
    x = m.integer("x", lb=1, ub=10)
    y = m.integer("y", lb=1)  # no upper bound
    s = m.continuous("s", lb=0, ub=100)
    m.subject_to(x / y + s == 0.37)
    m.minimize(s)
    assert detect_integer_ratio_specs(m) == []


def test_no_spec_on_ratio_free_model():
    m = Model("bilinear_only")
    x = m.integer("x", lb=1, ub=5)
    y = m.integer("y", lb=1, ub=5)
    m.subject_to(x * y >= 6)
    m.minimize(x + y)
    assert detect_integer_ratio_specs(m) == []


# ---------------------------------------------------------------------------
# The gear4 root-bound unlock (the #309 regression target)
# ---------------------------------------------------------------------------


def test_gear4_root_bound_lifts_off_zero():
    """Baseline root bound is ~0 (measured, #309); the partition bound must
    recover (nearly) the true optimum at the root box. The rigorous LP safe
    bound costs ~2.9e-4 on this 1e6-scaled row, so assert a 1e-3 margin."""
    m = _gear4()
    specs = detect_integer_ratio_specs(m)
    p = IntegerRatioPartitioner(m, specs)
    n = 6
    lb = np.array([12, 12, 12, 12, 0, 0], dtype=float)
    ub = np.array([60, 60, 60, 60, 100000, 100000], dtype=float)
    assert lb.size == n
    bound = p.node_bound(lb, ub)
    assert bound is not None
    assert bound > GEAR4_OPT - 1e-3
    # sound: never above the true optimum
    assert bound <= GEAR4_OPT + 1e-6


def test_gear4_node_bound_soundness_on_subboxes():
    """On every sampled sub-box the partition bound must not exceed the true
    minimum objective over the integer points of that box (brute force)."""
    m = _gear4()
    specs = detect_integer_ratio_specs(m)
    p = IntegerRatioPartitioner(m, specs)
    rng = np.random.default_rng(20260716)
    for _ in range(6):
        lo = rng.integers(12, 52, size=4)
        hi = np.minimum(60, lo + rng.integers(1, 9, size=4))
        lb = np.array([*lo, 0, 0], dtype=float)
        ub = np.array([*hi, 100000, 100000], dtype=float)
        bound = p.node_bound(lb, ub)
        if bound is None:
            continue  # abstained: nothing to check
        true_min = min(
            abs(1e6 * (a * b) / (c * d) - GEAR4_TARGET)
            for a in range(int(lo[0]), int(hi[0]) + 1)
            for b in range(int(lo[1]), int(hi[1]) + 1)
            for c in range(int(lo[2]), int(hi[2]) + 1)
            for d in range(int(lo[3]), int(hi[3]) + 1)
        )
        assert bound <= true_min + 1e-6, (lo, hi, bound, true_min)


def test_integer_infeasible_box_abstains():
    """A box with an empty integer range must abstain, never fabricate."""
    m = _gear4()
    specs = detect_integer_ratio_specs(m)
    p = IntegerRatioPartitioner(m, specs)
    lb = np.array([30.2, 12, 12, 12, 0, 0], dtype=float)
    ub = np.array([30.8, 60, 60, 60, 100000, 100000], dtype=float)
    assert p.node_bound(lb, ub) is None


# ---------------------------------------------------------------------------
# Default-off neutrality and flag-on solve wiring
# ---------------------------------------------------------------------------


def test_default_off_no_partitioner(monkeypatch):
    """Without the flag the relaxer never gets a partitioner (default path
    byte-identical)."""
    monkeypatch.delenv("DISCOPT_INTEGER_RATIO_PARTITION", raising=False)
    from discopt._jax.integer_ratio import enabled

    assert not enabled()


@pytest.mark.slow
def test_gear4_flag_on_certifies_with_fewer_nodes(monkeypatch):
    """Flag ON: gear4 certifies the known optimum with a materially smaller
    tree than the measured ~2.5k-node baseline (issue #309 acceptance)."""
    monkeypatch.setenv("DISCOPT_INTEGER_RATIO_PARTITION", "1")
    m = _gear4()
    res = m.solve(time_limit=150, gap_tolerance=1e-4)
    assert res.objective == pytest.approx(GEAR4_OPT, abs=1e-4)
    assert res.bound <= res.objective + 1e-6
    assert res.gap_certified
    assert res.node_count < 1500
    # the root bound must be unfrozen (baseline: ~0)
    assert res.root_bound > 1.0
