"""Unit tests for ``metrics.dual_bound_crosses_optimum`` (issue #759).

Pins the sense-aware dual-bound soundness predicate so soundness panels stop
applying a min-sense ``bound <= optimum`` check to MAXIMIZE instances (the
``syn05hfsg`` false positive that triggered #759).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import dual_bound_crosses_optimum


def test_minimize_lower_bound_soundness():
    opt = 100.0
    # A lower bound at/below the optimum is sound.
    assert not dual_bound_crosses_optimum(100.0, opt, minimize=True)
    assert not dual_bound_crosses_optimum(80.0, opt, minimize=True)
    # A lower bound strictly above the optimum is a crossing (too-tight, unsound).
    assert dual_bound_crosses_optimum(120.0, opt, minimize=True)


def test_maximize_upper_bound_soundness():
    opt = 837.7324
    # An upper bound at/above the optimum is sound — INCLUDING the syn05hfsg case
    # (bound ~1651 for a maximize whose optimum is ~837.73). This is the false
    # positive #759 was really about: NOT a violation.
    assert not dual_bound_crosses_optimum(1651.0, opt, minimize=False)
    assert not dual_bound_crosses_optimum(837.7324, opt, minimize=False)
    # An upper bound strictly below the optimum is a crossing (too-tight, unsound).
    assert dual_bound_crosses_optimum(700.0, opt, minimize=False)


def test_sense_matters_for_the_same_numbers():
    # bound above optimum: unsound for minimize, sound for maximize.
    bound, opt = 1651.0, 837.7324
    assert dual_bound_crosses_optimum(bound, opt, minimize=True)
    assert not dual_bound_crosses_optimum(bound, opt, minimize=False)


def test_missing_values_never_flag():
    assert not dual_bound_crosses_optimum(None, 1.0, minimize=True)
    assert not dual_bound_crosses_optimum(1.0, None, minimize=False)


def test_tolerance_absorbs_float_noise():
    opt = 0.0
    # Within tolerance of a zero optimum: not a crossing either way.
    assert not dual_bound_crosses_optimum(1e-9, opt, minimize=True)
    assert not dual_bound_crosses_optimum(-1e-9, opt, minimize=False)
