"""Verdict-honesty tests for the .nl cross-solver head-to-head harness.

The harness used to label *every* objective that didn't equal the known
optimum as ``VIOLATION`` (a binary correct/wrong flag with no GAP category).
That conflated honest convergence gaps — a feasible incumbent that simply
didn't close in the time budget — with the genuine red line: a solver that
*claims* a certified global at the wrong value, or returns an incumbent
strictly better than the proven global (an impossible bound).

These tests pin the three-way classification (``ok`` / ``GAP`` / ``VIOLATION``
/ ``n/a``) so a feasible-but-suboptimal result can never again be reported as a
correctness violation, while real violations still are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.global_opt_baron_vs_discopt import GAP, NA, OK, VIOLATION, classify
from scripts.global_opt_nl_solvers import write_report

pytestmark = pytest.mark.unit


def test_feasible_suboptimal_is_gap_not_violation():
    # discopt returns status=feasible at a worse-than-global value: honest GAP.
    assert classify("feasible", 6.9358, 5.4709, maximize=False) == GAP
    assert classify("feasible", 26.4358, 9.1635, maximize=False) == GAP


def test_wrong_certified_global_is_violation():
    # status=optimal asserts a certified global; a wrong value there is the red line.
    assert classify("optimal", 6.9358, 5.4709, maximize=False) == VIOLATION


def test_incumbent_better_than_proven_global_is_violation():
    # An incumbent strictly *below* the proven min (or above the max) is impossible.
    assert classify("feasible", 4.0, 5.4709, maximize=False) == VIOLATION
    assert classify("feasible", 9.0, 5.4709, maximize=True) == VIOLATION


def test_matching_optimum_is_ok_and_missing_oracle_is_na():
    assert classify("optimal", 5.4709, 5.4709, maximize=False) == OK
    assert classify("feasible", 5.4709, None, maximize=False) == NA
    assert classify("feasible", None, 5.4709, maximize=False) == NA


def test_report_distinguishes_gap_from_violation(tmp_path):
    """A feasible-but-suboptimal row renders as GAP; the summary counts 0 violations."""
    rows = [
        {
            "instance": "nvs05",
            "known": 5.4709,
            "maximize": False,
            "runs": {
                "discopt": {
                    "objective": 6.9358,
                    "verdict": classify("feasible", 6.9358, 5.4709, False),
                    "wall_time": 60.5,
                },
                "scip": {
                    "objective": 5.4709,
                    "verdict": classify("optimal", 5.4709, 5.4709, False),
                    "wall_time": 1.6,
                },
            },
        }
    ]
    md = write_report(rows, ["discopt", "scip"], tl=60, out_dir=tmp_path, ts="T")
    text = md.read_text()
    # discopt's honest gap is reported as GAP, not VIOLATION.
    assert "| discopt | 0/1 | 1 | 0 | 0 |" in text
    assert "| scip | 1/1 | 0 | 0 | 0 |" in text
    assert "VIOLATION" in text  # legend still documents the red line
