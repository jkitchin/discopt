"""Tests for the Wave-2 QCQP relaxation-strength scoreboard (W1).

Three guards:
  * the hardcoded ``known_optimum`` of every registered instance is reproduced by
    its independent reference computation (so the scoreboard's references are
    self-verifying, not magic numbers);
  * the baseline solver reaches each smoke instance's known optimum
    (``incorrect_count == 0`` — the correctness gate the cut work must preserve);
  * the scoreboard harness runs end-to-end and reports the expected fields,
    including a finite root gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

from benchmarks.problems.base import get_problems
from benchmarks.problems.qcqp_problems import (
    concave_boxqp,
    indefinite_qcqp,
    reference_optimum_multistart,
    reference_optimum_vertex,
)
from benchmarks.qcqp_scoreboard import run_scoreboard

# Map instance name -> (family, n, seed) to recompute references independently.
_GEN = {
    "qcqp_concave_n6": ("concave", 6, 6),
    "qcqp_concave_n5": ("concave", 5, 5),
    "qcqp_indef_n4_s1": ("indefinite", 4, 1),
    "qcqp_indef_n4_s2": ("indefinite", 4, 2),
    "qcqp_indef_n4_s3": ("indefinite", 4, 3),
    "qcqp_indef_n5_s11": ("indefinite", 5, 11),
    "qcqp_indef_n6_s12": ("indefinite", 6, 12),
}


def _reference(family: str, n: int, seed: int) -> float:
    if family == "concave":
        Q, c = concave_boxqp(n, seed)
        return reference_optimum_vertex(Q, c)
    Q, c = indefinite_qcqp(n, seed)
    return reference_optimum_multistart(Q, c)


@pytest.mark.correctness
def test_registered_optima_match_independent_reference():
    """Every hardcoded known_optimum reproduces its independent computation."""
    problems = {p.name: p for p in get_problems("qcqp", level="full")}
    assert set(problems) == set(_GEN), "instance set drifted from the reference map"
    for name, (family, n, seed) in _GEN.items():
        ref = _reference(family, n, seed)
        assert abs(problems[name].known_optimum - ref) < 1e-6, (
            f"{name}: registered {problems[name].known_optimum} vs reference {ref}"
        )


@pytest.mark.smoke
def test_baseline_reaches_known_optima():
    """Baseline solve hits every smoke instance's known optimum (gate)."""
    board = run_scoreboard(level="smoke")
    assert board.rows, "scoreboard produced no rows"
    assert board.incorrect_count == 0, board.format_table()
    for r in board.rows:
        assert r.status == "optimal"
        assert abs(r.objective - r.known_optimum) < 1e-4 + 1e-4 * abs(r.known_optimum)


@pytest.mark.smoke
def test_scoreboard_reports_root_gap_and_nodes():
    """The harness reports finite root gaps and node counts."""
    board = run_scoreboard(level="smoke")
    for r in board.rows:
        assert r.root_gap is not None and r.root_gap >= -1e-9
        assert r.node_count >= 1
    # The indefinite instances must exhibit a genuine root gap on at least one
    # instance (otherwise there is nothing for relaxation cuts to close).
    indef = [r for r in board.rows if "indef" in r.instance]
    assert any(r.root_gap > 1e-6 for r in indef), board.format_table()
