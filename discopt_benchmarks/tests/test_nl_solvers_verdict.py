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

from scripts.global_opt_baron_vs_discopt import (
    DISCOPT_WORKER,
    GAP,
    NA,
    OK,
    VIOLATION,
    bound_violates_oracle,
    classify,
)
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


# --------------------------------------------------------------------------- #
# A3: the dual bound must never cross the oracle (certificate invariant)
# --------------------------------------------------------------------------- #
def test_dual_bound_crossing_oracle_is_violation():
    # minimize: a lower bound ABOVE the true optimum is impossible (bound<=opt).
    assert classify("feasible", 5.4709, 5.4709, maximize=False, bound=5.60) == VIOLATION
    # maximize: an upper bound BELOW the true optimum is impossible (bound>=opt).
    assert classify("feasible", 5.4709, 5.4709, maximize=True, bound=5.35) == VIOLATION


def test_bound_violation_fires_even_without_incumbent():
    # No incumbent (obj=None) but the reported dual bound crosses the oracle:
    # still the red line, not n/a.
    assert classify("time_limit", None, 5.4709, maximize=False, bound=6.0) == VIOLATION


def test_valid_dual_bound_below_optimum_is_not_a_bound_violation():
    # A lower bound at/under the optimum with a suboptimal incumbent is an
    # honest GAP, never a bound violation.
    assert classify("feasible", 6.9358, 5.4709, maximize=False, bound=1.348) == GAP
    assert classify("optimal", 5.4709, 5.4709, maximize=False, bound=5.4709) == OK


def test_none_bound_preserves_incumbent_verdict():
    # bound=None (default) → classify falls through to the incumbent logic,
    # exactly as before A3 (backward-compatible).
    assert classify("feasible", 6.9358, 5.4709, maximize=False, bound=None) == GAP
    assert classify("optimal", 6.9358, 5.4709, maximize=False, bound=None) == VIOLATION


def test_bound_violates_oracle_predicate():
    assert bound_violates_oracle(6.0, 5.4709, maximize=False) is True
    assert bound_violates_oracle(5.0, 5.4709, maximize=False) is False  # valid LB
    assert bound_violates_oracle(5.0, 5.4709, maximize=True) is True  # UB below opt
    assert bound_violates_oracle(None, 5.4709, maximize=False) is False
    assert bound_violates_oracle(6.0, None, maximize=False) is False


def test_worker_reads_res_bound_not_lower_bound():
    # Regression for the A3 bug: SolveResult has `.bound`, not `.lower_bound`
    # (the old getattr silently read None on every one of the 61 rows).
    assert 'getattr(res, "bound"' in DISCOPT_WORKER
    assert 'getattr(res, "lower_bound"' not in DISCOPT_WORKER


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
