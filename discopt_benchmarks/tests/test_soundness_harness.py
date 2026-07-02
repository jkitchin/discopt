"""Tests for the reusable soundness harness (cert:T0.4).

Pins the two assertions Phases 2–4 consume: ``assert_bound_sound`` (a dual bound
never crosses the box optimum and never regresses vs a baseline) and
``assert_cut_valid`` (a cut removes no feasible point). The cut check is
exercised against the McCormick envelope of a bilinear product — the multilinear
separation whose soundness the harness must confirm — plus a deliberately
invalid cut that it must flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.soundness import (
    SoundnessError,
    assert_bound_sound,
    assert_cut_valid,
    known_optimum_oracle,
)

# ─────────────────────── assert_cut_valid ───────────────────────


def test_assert_cut_valid_flags_invalid_cut():
    """A cut that removes a known feasible point must raise (unit fixture)."""
    # Feasible points include the origin; the cut x0 + x1 <= -1 excludes it.
    cut = (np.array([1.0, 1.0]), -1.0)
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    with pytest.raises(SoundnessError):
        assert_cut_valid(cut, pts, tol=1e-9)


def test_assert_cut_valid_passes_valid_cut():
    cut = (np.array([1.0, 1.0]), 2.0)  # x0 + x1 <= 2
    pts = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5])]
    report = assert_cut_valid(cut, pts, tol=1e-9)
    assert len(report) == 3
    assert all(r["slack"] >= -1e-9 for r in report)


def _mccormick_cuts(xl, xu, yl, yu):
    """The four McCormick envelope inequalities for w = x*y over the box,
    each as (a, b) meaning a·[x, y, w] <= b."""
    return [
        (np.array([yl, xl, -1.0]), xl * yl),  # w >= xl*y + yl*x - xl*yl
        (np.array([yu, xu, -1.0]), xu * yu),  # w >= xu*y + yu*x - xu*yu
        (np.array([-yl, -xu, 1.0]), -xu * yl),  # w <= xu*y + yl*x - xu*yl
        (np.array([-yu, -xl, 1.0]), -xl * yu),  # w <= xl*y + yu*x - xl*yu
    ]


def test_multilinear_separation_soundness_sweep():
    """Sweep boxes and feasible bilinear points: every McCormick envelope cut is
    valid (removes no feasible point). This is the harness applied to genuine
    multilinear separation."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        xl, xu = sorted(rng.uniform(-3, 3, size=2))
        yl, yu = sorted(rng.uniform(-3, 3, size=2))
        if xu - xl < 1e-6 or yu - yl < 1e-6:
            continue
        cuts = _mccormick_cuts(xl, xu, yl, yu)
        # Feasible points of the graph w = x*y with (x, y) in the box.
        pts = []
        for _k in range(8):
            x = rng.uniform(xl, xu)
            y = rng.uniform(yl, yu)
            pts.append(np.array([x, y, x * y]))
        for cut in cuts:
            # Each cut is valid for every feasible point (tol absorbs fp noise).
            assert_cut_valid(cut, pts, tol=1e-7)


def test_mccormick_cut_catches_a_planted_violation():
    """Sanity: if we corrupt one envelope cut's rhs so it slices the interior,
    the harness catches it — the sweep above is not vacuously passing."""
    xl, xu, yl, yu = -1.0, 1.0, -1.0, 1.0
    a, b = _mccormick_cuts(xl, xu, yl, yu)[0]
    # Tighten the lower-envelope rhs past the true envelope: for this cut the
    # slack at (x, y) is (x-xl)(y-yl) - 0.5, so points near the lower-left corner
    # (small (x-xl)(y-yl)) now violate it.
    bad = (a, b - 0.5)
    pts = [np.array([x, y, x * y]) for x, y in [(-1.0, -1.0), (-0.9, -0.6), (-1.0, 0.5)]]
    with pytest.raises(SoundnessError):
        assert_cut_valid(bad, pts, tol=1e-7)


# ─────────────────────── assert_bound_sound ───────────────────────


def test_assert_bound_sound_validity_pass_and_fail():
    boxes = [(np.array([0.0]), np.array([1.0])), (np.array([1.0]), np.array([2.0]))]
    # Oracle: min of f(x) = x^2 over the box (attained at the lower endpoint here).
    def oracle(box):
        lo, hi = box
        return float(lo[0] ** 2)

    # A valid lower bound: convex envelope value <= true optimum.
    def good(box):
        lo, hi = box
        return float(lo[0] ** 2) - 0.1

    assert_bound_sound(good, boxes, oracle, tol=1e-6)

    # An invalid bound that crosses the optimum on the second box.
    def bad(box):
        lo, hi = box
        return float(lo[0] ** 2) + 0.5

    with pytest.raises(SoundnessError):
        assert_bound_sound(bad, boxes, oracle, tol=1e-6)


def test_assert_bound_sound_nonregression():
    boxes = [(np.array([0.0]), np.array([1.0]))]

    def oracle(_box):
        return 10.0

    def baseline(_box):
        return 5.0

    def improved(_box):
        return 7.0  # >= baseline and <= oracle: fine

    assert_bound_sound(improved, boxes, oracle, tol=1e-9, baseline_fn=baseline)

    def regressed(_box):
        return 3.0  # below baseline: non-regression violation

    with pytest.raises(SoundnessError):
        assert_bound_sound(regressed, boxes, oracle, tol=1e-9, baseline_fn=baseline)


def test_assert_bound_sound_max_sense():
    boxes = [(np.array([0.0]), np.array([1.0]))]

    def oracle(_box):
        return 5.0  # true max

    def upper(_box):
        return 6.0  # valid upper bound for max

    assert_bound_sound(upper, boxes, oracle, tol=1e-9, sense="max")

    def crossing(_box):
        return 4.0  # below the true max: invalid upper bound

    with pytest.raises(SoundnessError):
        assert_bound_sound(crossing, boxes, oracle, tol=1e-9, sense="max")


def test_known_optimum_oracle():
    oracle = known_optimum_oracle({"inst": -12.5}, "inst")
    assert oracle((np.array([0.0]), np.array([1.0]))) == -12.5
