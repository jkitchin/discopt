"""Per-B&B-node mask refresh tests.

A constraint that's indefinite on the root variable box but convex
once branching tightens bounds must have its mask entry flipped to
``True`` at that node. This enables OA cuts and the αBB skip at
nodes the root-level certificate could not reach.
"""

from __future__ import annotations

import numpy as np
from discopt._jax.convexity import (
    Curvature,
    certify_convex,
    classify_expr,
    refresh_convex_mask,
)
from discopt.modeling.core import Model


class TestRefreshSoundness:
    """Refresh must never demote a True mask entry."""

    def test_true_entries_preserved(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        m.minimize(x)
        m.subject_to(x**2 <= 5.0)  # always convex
        root_mask = [True]
        # Even an extreme degenerate box mustn't demote this entry.
        lb = np.array([-2.0])
        ub = np.array([2.0])
        refreshed = refresh_convex_mask(m, root_mask, lb, ub)
        assert refreshed == [True]

    def test_false_stays_false_when_no_proof(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(x)
        m.subject_to(x * y <= 1.0)  # genuinely indefinite
        root_mask = [False]
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        refreshed = refresh_convex_mask(m, root_mask, lb, ub)
        assert refreshed == [False]


class TestRefreshTightening:
    """Constraints UNKNOWN at root but provably convex on a tighter
    subtree box must flip to True under refresh."""

    def test_cubic_becomes_convex_on_nonneg_subtree(self):
        """``x^3 <= 10`` is UNKNOWN on ``[-5, 5]`` but CONVEX on ``[0, 5]``."""
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        m.minimize(x)
        m.subject_to(x**3 <= 10.0)
        # Root certificate cannot prove the whole box.
        assert classify_expr(x**3, m) == Curvature.UNKNOWN
        assert certify_convex(x**3, m) is None
        root_mask = [False]
        # After branching tightens to x >= 0:
        node_lb = np.array([0.0])
        node_ub = np.array([5.0])
        refreshed = refresh_convex_mask(m, root_mask, node_lb, node_ub)
        assert refreshed == [True]

    def test_quartic_shift_on_tighter_box(self):
        """``x^4 - 2 x^2 <= 10`` on root [-3, 3] is UNKNOWN; on a
        node with x in [1, 2] it is CONVEX."""
        m = Model("t")
        x = m.continuous("x", lb=-3.0, ub=3.0)
        m.minimize(x)
        m.subject_to(x**4 - 2.0 * x**2 <= 10.0)
        root_mask = [False]
        node_lb = np.array([1.0])
        node_ub = np.array([2.0])
        refreshed = refresh_convex_mask(m, root_mask, node_lb, node_ub)
        assert refreshed == [True]

    def test_mixed_mask_only_false_entries_reconsidered(self):
        """A True entry stays True; a refreshable False flips; an
        unrefreshable False stays False — all in one call."""
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(x + y)
        m.subject_to(x**2 <= 25.0)  # convex at root
        m.subject_to(x**3 <= 100.0)  # convex only on nonneg
        m.subject_to(x * y <= 1.0)  # indefinite anywhere
        root_mask = [True, False, False]
        node_lb = np.array([0.0, -1.0])
        node_ub = np.array([5.0, 1.0])
        refreshed = refresh_convex_mask(m, root_mask, node_lb, node_ub)
        assert refreshed == [True, True, False]


class TestRefreshEarlyExit:
    """Refresh must short-circuit gracefully on degenerate input."""

    def test_all_true_returns_copy_unchanged(self):
        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x)
        m.subject_to(x**2 <= 1.0)
        root_mask = [True]
        refreshed = refresh_convex_mask(m, root_mask, np.array([0.0]), np.array([1.0]))
        assert refreshed == [True]
        assert refreshed is not root_mask  # defensive copy

    def test_shape_mismatch_returns_root_mask_unchanged(self):
        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x)
        m.subject_to(x**2 <= 1.0)
        root_mask = [False]
        # Wrong-sized bounds — refresh must bail out safely.
        refreshed = refresh_convex_mask(m, root_mask, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert refreshed == root_mask

    def test_empty_constraints(self):
        m = Model("t")
        m.continuous("x", lb=0.0, ub=1.0)
        root_mask: list = []
        refreshed = refresh_convex_mask(m, root_mask, np.array([0.0]), np.array([1.0]))
        assert refreshed == []
