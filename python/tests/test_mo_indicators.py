"""Tests for hypervolume, IGD, spread, and epsilon indicators."""

from __future__ import annotations

import numpy as np
import pytest
from discopt.mo import (
    ParetoFront,
    ParetoPoint,
    common_reference,
    epsilon_indicator,
    hypervolume,
    igd,
    spread,
)


def _front(objs, names=None, senses=None):
    objs = np.asarray(objs, dtype=np.float64)
    if objs.ndim == 1:
        objs = objs[None, :]
    k = objs.shape[1]
    names = list(names) if names else [f"f{i + 1}" for i in range(k)]
    senses = list(senses) if senses else ["min"] * k
    points = [
        ParetoPoint(
            x={},
            objectives=row.copy(),
            status="optimal",
            wall_time=0.0,
        )
        for row in objs
    ]
    return ParetoFront(
        points=points,
        method="test",
        objective_names=names,
        senses=senses,
    )


class TestHypervolume2D:
    def test_single_point(self):
        f = _front([[1.0, 1.0]])
        # Reference (3, 3): dominated region is 2 * 2 = 4.
        assert hypervolume(f, reference=np.array([3.0, 3.0])) == pytest.approx(4.0)

    def test_three_collinear_points(self):
        # Simple convex stair: (0, 2), (1, 1), (2, 0), reference (3, 3).
        # HV = (1-0)*(3-2) + (2-1)*(3-1) + (3-2)*(3-0) = 1 + 2 + 3 = 6.
        f = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        hv = hypervolume(f, reference=np.array([3.0, 3.0]))
        assert hv == pytest.approx(6.0)

    def test_dominated_points_ignored(self):
        # Same as above plus dominated (3, 3); HV should not change.
        f = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [2.5, 2.5]])
        hv = hypervolume(f, reference=np.array([3.0, 3.0]))
        assert hv == pytest.approx(6.0)

    def test_empty_front(self):
        f = _front(np.zeros((0, 2)))
        assert hypervolume(f, reference=np.array([1.0, 1.0])) == 0.0

    def test_default_reference(self):
        f = _front([[0.0, 2.0], [2.0, 0.0]])
        hv = hypervolume(f)  # default reference
        assert hv > 0.0


class TestHypervolume3D:
    def test_single_point(self):
        f = _front([[0.0, 0.0, 0.0]])
        hv = hypervolume(f, reference=np.array([2.0, 2.0, 2.0]))
        assert hv == pytest.approx(8.0)

    def test_two_parallel_points(self):
        # (0, 0, 1) and (1, 0, 0), reference (2, 2, 2).
        # Both dominate sub-boxes; compute by inclusion-exclusion.
        # Volume dominated by (0,0,1): (2-0)*(2-0)*(2-1) = 4
        # Volume dominated by (1,0,0): (2-1)*(2-0)*(2-0) = 4
        # Intersection is region x>=1, y>=0, z>=1: (2-1)*(2-0)*(2-1) = 2
        # Union = 4 + 4 - 2 = 6
        f = _front([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        hv = hypervolume(f, reference=np.array([2.0, 2.0, 2.0]))
        assert hv == pytest.approx(6.0, rel=1e-6)


class TestHypervolumeMonteCarlo:
    def test_agrees_with_exact_in_3d(self):
        rng = np.random.default_rng(42)
        # Three points on a concave surface.
        f = _front([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.array([2.0, 2.0, 2.0])
        exact = hypervolume(f, reference=ref, method="exact")
        approx = hypervolume(f, reference=ref, method="mc", n_samples=20_000, rng=rng)
        assert approx == pytest.approx(exact, rel=0.05)


class TestHypervolumeSenses:
    def test_maximize_sense(self):
        # Max-sense front (3, 1), (2, 2), (1, 3) with reference (0, 0)
        # (dominated by all points in max sense). After negation becomes
        # min-sense (-3, -1), (-2, -2), (-1, -3) vs reference (0, 0),
        # giving HV = 6. Sense-invariant.
        f_max = _front([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]], senses=("max", "max"))
        hv_max = hypervolume(f_max, reference=np.array([0.0, 0.0]))
        assert hv_max == pytest.approx(6.0)


class TestCommonReference:
    """MO4: the default HV reference is front-dependent (incomparable across
    fronts); ``common_reference`` builds one shared reference so two fronts for
    the same problem are comparable.
    """

    @pytest.mark.smoke
    def test_default_reference_is_front_dependent(self):
        # Two nested fronts. Under the default (front-derived) reference their
        # hypervolumes are computed against DIFFERENT references, so they are
        # not comparable -- the better (dominating) front can score LOWER.
        better = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        worse = _front([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]])
        hv_better_default = better.hypervolume()
        hv_worse_default = worse.hypervolume()
        # The pathology: the worse front scores strictly higher under defaults
        # (its self-derived reference box is larger). This is the MO4 bug.
        assert hv_worse_default > hv_better_default

    @pytest.mark.smoke
    def test_shared_reference_is_comparable(self):
        better = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        worse = _front([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]])
        ref = common_reference(better, worse)
        hv_better = better.hypervolume(reference=ref)
        hv_worse = worse.hypervolume(reference=ref)
        # Against a SHARED reference the dominating front scores at least as
        # high, restoring comparability.
        assert hv_better >= hv_worse

    @pytest.mark.smoke
    def test_reference_dominates_all_points(self):
        f1 = _front([[0.0, 2.0], [2.0, 0.0]])
        f2 = _front([[1.0, 3.0], [3.0, 1.0]])
        ref = common_reference(f1, f2)
        # Every point of every front must strictly dominate the reference so
        # each contributes positive hypervolume.
        for f in (f1, f2):
            assert f.hypervolume(reference=ref) > 0.0
            assert np.all(f.objectives() < ref[None, :])

    @pytest.mark.smoke
    def test_max_sense_shared_reference(self):
        f1 = _front([[3.0, 1.0], [1.0, 3.0]], senses=("max", "max"))
        f2 = _front([[2.0, 1.0], [1.0, 2.0]], senses=("max", "max"))
        ref = common_reference(f1, f2)
        # Max-sense: reference must be BELOW all points (worse in max sense).
        for f in (f1, f2):
            assert np.all(f.objectives() > ref[None, :])
            assert f.hypervolume(reference=ref) > 0.0

    @pytest.mark.smoke
    def test_incompatible_fronts_raise(self):
        f_min = _front([[0.0, 2.0]], senses=("min", "min"))
        f_max = _front([[0.0, 2.0]], senses=("min", "max"))
        with pytest.raises(ValueError):
            common_reference(f_min, f_max)
        with pytest.raises(ValueError):
            common_reference()
        empty = _front(np.zeros((0, 2)))
        with pytest.raises(ValueError):
            common_reference(empty)


class TestIGD:
    def test_zero_when_same_front(self):
        f = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        assert igd(f, f) == pytest.approx(0.0)

    def test_positive_distance(self):
        fa = _front([[0.0, 2.0], [2.0, 0.0]])
        fb = _front([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])  # denser
        # IGD from fb to fa: for (1, 1), nearest in fa is either (0, 2) or (2, 0),
        # both at distance sqrt(2); avg = (0 + sqrt(2) + 0)/3.
        expected = np.sqrt(2) / 3
        assert igd(fa, fb) == pytest.approx(expected)

    def test_incompatible_senses_raises(self):
        fa = _front([[0.0, 2.0]], senses=("min", "min"))
        fb = _front([[0.0, 2.0]], senses=("min", "max"))
        with pytest.raises(ValueError):
            igd(fa, fb)


class TestSpread:
    def test_uniform_spacing_low(self):
        # Uniform x_1 spacing with constant step -> std = 0 -> CoV = 0.
        f = _front(
            [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
            senses=("min", "min"),
        )
        assert spread(f) == pytest.approx(0.0, abs=1e-12)

    def test_nonuniform_spacing_positive(self):
        f = _front(
            [[0.0, 10.0], [1.0, 2.0], [2.0, 1.0], [10.0, 0.0]],
        )
        assert spread(f) > 0.0

    def test_single_point(self):
        f = _front([[0.0, 0.0]])
        assert spread(f) == 0.0


class TestEpsilonIndicator:
    def test_zero_when_same(self):
        f = _front([[0.0, 2.0], [2.0, 0.0]])
        assert epsilon_indicator(f, f) == pytest.approx(0.0)

    def test_positive_when_b_better(self):
        fa = _front([[2.0, 2.0]])
        fb = _front([[1.0, 1.0]])
        # Additive eps shifting A by -1 in each coordinate reaches B -> eps >= -1?
        # Actually: we need A shifted forward by eps to dominate B, so
        # I_eps = max over b of max over coord of min over a of (a_coord - b_coord).
        # Here a - b = (1, 1), so I_eps = 1.
        assert epsilon_indicator(fa, fb) == pytest.approx(1.0)

    def test_incompatible_senses_raises(self):
        fa = _front([[0.0, 2.0]], senses=("min", "min"))
        fb = _front([[0.0, 2.0]], senses=("min", "max"))
        with pytest.raises(ValueError):
            epsilon_indicator(fa, fb)
