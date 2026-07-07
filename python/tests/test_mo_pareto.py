"""Unit tests for ParetoPoint, ParetoFront, and dominance filtering."""

from __future__ import annotations

import numpy as np
import pytest
from discopt.mo import ParetoFront, ParetoPoint, filter_nondominated


def _make_front(objs, method="test", names=("f1", "f2"), senses=("min", "min")):
    """Build a ParetoFront from an (n, k) array of objective values."""
    points = [
        ParetoPoint(
            x={},
            objectives=np.asarray(row, dtype=np.float64),
            status="optimal",
            wall_time=0.0,
        )
        for row in objs
    ]
    return ParetoFront(
        points=points,
        method=method,
        objective_names=list(names),
        senses=list(senses),
    )


class TestFilterNondominated:
    def test_all_nondominated(self):
        objs = np.array([[0, 2], [1, 1], [2, 0]], dtype=float)
        mask = filter_nondominated(objs)
        assert mask.tolist() == [True, True, True]

    def test_one_dominated(self):
        objs = np.array([[0, 2], [1, 1], [2, 0], [3, 3]], dtype=float)
        mask = filter_nondominated(objs)
        assert mask.tolist() == [True, True, True, False]

    def test_duplicate_not_dominated(self):
        objs = np.array([[1, 1], [1, 1]], dtype=float)
        mask = filter_nondominated(objs)
        # Equal points do not strictly dominate each other -> both kept.
        assert mask.all()

    def test_empty_input(self):
        mask = filter_nondominated(np.zeros((0, 2)))
        assert mask.shape == (0,)

    def test_max_senses(self):
        # With a max-sense on the second column, bigger is better.
        objs = np.array([[0, 2], [1, 1], [2, 0]], dtype=float)
        senses = np.array([1.0, -1.0])  # min, max
        mask = filter_nondominated(objs, senses=senses)
        # (0, 2) dominates in both (smaller f1, larger f2 under max).
        # (1, 1) and (2, 0) are dominated by (0, 2) under these senses.
        assert mask.tolist() == [True, False, False]

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            filter_nondominated(np.array([1.0, 2.0, 3.0]))


class TestParetoFront:
    def test_n_and_k(self):
        front = _make_front([[0, 2], [1, 1], [2, 0]])
        assert front.n == 3
        assert front.k == 2

    def test_objectives_stack(self):
        front = _make_front([[0, 2], [1, 1], [2, 0]])
        arr = front.objectives()
        assert arr.shape == (3, 2)
        np.testing.assert_allclose(arr, [[0, 2], [1, 1], [2, 0]])

    def test_filtered_drops_dominated(self):
        front = _make_front([[0, 2], [1, 1], [2, 0], [3, 3]])
        kept = front.filtered()
        assert kept.n == 3
        assert kept.method == front.method
        # Dominated point (3, 3) must be dropped.
        assert not any(np.allclose(p.objectives, [3, 3]) for p in kept.points)

    def test_empty_filtered(self):
        front = _make_front([])
        kept = front.filtered()
        assert kept.n == 0

    def test_summary_is_string(self):
        front = _make_front([[0.0, 2.0], [2.0, 0.0]])
        s = front.summary()
        assert "Pareto Front" in s
        assert "2 points" in s

    def test_senses_array(self):
        front = _make_front([[0, 2]], senses=("min", "max"))
        np.testing.assert_allclose(front._senses_array(), [1.0, -1.0])


class TestFilteredDedup:
    """MO3: filtered() must collapse tolerance-equal duplicate objective vectors.

    Weak dominance keeps every copy of an identical objective vector (equal
    points do not strictly dominate one another), so without an explicit dedup
    a front with three identical anchors survives as three points -- inflating
    ``n`` and distorting spacing metrics.
    """

    @pytest.mark.smoke
    def test_exact_duplicates_collapsed(self):
        # 3 identical + 1 distinct + 1 dominated (MO3 repro: 3+1+1).
        objs = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 2.0], [3.0, 3.0]]
        front = _make_front(objs)
        kept = front.filtered()
        # Want exactly 2: one representative of the triple + the distinct
        # point; the dominated (3, 3) is dropped. (Pre-fix: 4.)
        assert kept.n == 2
        kept_objs = sorted(tuple(p.objectives) for p in kept.points)
        assert kept_objs == [(0.0, 2.0), (1.0, 1.0)]

    @pytest.mark.smoke
    def test_dedup_keeps_first_occurrence_params(self):
        pts = [
            ParetoPoint(
                x={},
                objectives=np.array([1.0, 1.0]),
                status="optimal",
                wall_time=0.0,
                scalarization_params={"weights": [0.2, 0.8]},
            ),
            ParetoPoint(
                x={},
                objectives=np.array([1.0, 1.0]),
                status="optimal",
                wall_time=0.0,
                scalarization_params={"weights": [0.5, 0.5]},
            ),
        ]
        front = ParetoFront(
            points=pts, method="test", objective_names=["f1", "f2"], senses=["min", "min"]
        )
        kept = front.filtered()
        assert kept.n == 1
        # First occurrence's params are preserved.
        assert kept.points[0].scalarization_params == {"weights": [0.2, 0.8]}

    @pytest.mark.smoke
    def test_genuinely_distinct_near_points_preserved(self):
        # Two points differing by 1e-4 must NOT be merged at the default 1e-8 tol.
        objs = [[1.0, 1.0], [1.0 + 1e-4, 1.0 - 1e-4], [0.0, 2.0]]
        front = _make_front(objs)
        kept = front.filtered()
        assert kept.n == 3

    @pytest.mark.smoke
    def test_dedup_tol_is_tunable(self):
        objs = [[1.0, 1.0], [1.0 + 1e-4, 1.0 - 1e-4]]
        front = _make_front(objs)
        # A looser tolerance collapses the near-duplicate pair.
        assert front.filtered(dedup_tol=1e-3).n == 1
        # The default keeps them distinct.
        assert front.filtered().n == 2
