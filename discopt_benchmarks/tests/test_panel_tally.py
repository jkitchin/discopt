"""Tests for the panel tally that backs the manuscript's head-to-head table.

The manuscript quotes these counts, so the two properties that matter are:

1. the tally **re-derives** verdicts through the harness's own `classify`
   rather than reading the stored `verdict` field, so a correctness fix to
   `classify` propagates into the paper instead of leaving it quoting numbers
   frozen at the rule that existed when the panel ran; and
2. `certified` and `ok` stay distinct -- a certified-global *claim* is not the
   same measurement as *matching the oracle*, and conflating them is what
   produced the unsupported "BARON and SCIP each certify ~180 of ~200" claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.global_opt_baron_vs_discopt import NA, OK, VIOLATION  # noqa: E402
from scripts.panel_tally import tally  # noqa: E402

pytestmark = pytest.mark.unit


def _panel(tmp_path: Path, rows: list[dict], solvers: list[str] | None = None) -> Path:
    payload: dict = {"time_limit": 60.0, "timestamp": "T", "rows": rows}
    if solvers:
        payload["solvers"] = solvers
    p = tmp_path / "panel.json"
    p.write_text(json.dumps(payload))
    return p


def test_stored_verdict_is_ignored_in_favour_of_reclassification(tmp_path):
    """A panel written before the placeholder fix is re-scored, not replayed.

    This row is `prob10` as the 2026-07-18 panel recorded it: GAMS printed a
    placeholder `0.0` for a run that returned no point, and the panel froze
    `verdict: VIOLATION`. The tally must report `n/a`.
    """
    p = _panel(
        tmp_path,
        [
            {
                "instance": "prob10",
                "known": 3.445503794,
                "maximize": False,
                "runs": {
                    "baron": {
                        "status": "14 No Solution Returned",
                        "objective": 0.0,
                        "wall_time": 0.03,
                        "verdict": VIOLATION,  # the stale, now-superseded verdict
                    }
                },
            }
        ],
        solvers=["baron"],
    )
    res, compared, _ = tally(p)
    assert compared == 1
    assert res["baron"]["counts"][VIOLATION] == 0, "replayed the stored verdict"
    assert res["baron"]["counts"][NA] == 1
    assert res["baron"]["violations"] == []


def test_certified_and_ok_are_independent_columns(tmp_path):
    """`2 Locally Optimal` at the right value is `ok` but NOT certified."""
    rows = [
        # right answer, no global certificate
        {
            "instance": "a",
            "known": 10.0,
            "maximize": False,
            "runs": {"s": {"status": "2 Locally Optimal", "objective": 10.0, "wall_time": 1.0}},
        },
        # right answer, certified
        {
            "instance": "b",
            "known": 20.0,
            "maximize": False,
            "runs": {"s": {"status": "1 Optimal", "objective": 20.0, "wall_time": 2.0}},
        },
    ]
    res, compared, _ = tally(_panel(tmp_path, rows, solvers=["s"]))
    assert compared == 2
    assert res["s"]["counts"][OK] == 2, "both matched the oracle"
    assert res["s"]["certified"] == 1, "only one asserted a certified global"


def test_flat_two_solver_layout_is_read(tmp_path):
    """The older `global_opt_baron_vs_discopt` layout has no `runs` wrapper."""
    rows = [
        {
            "instance": "a",
            "known": 5.0,
            "maximize": False,
            "discopt": {"status": "optimal", "objective": 5.0, "wall_time": 3.0},
            "baron": {"status": "1 Optimal", "objective": 5.0, "wall_time": 0.5},
        }
    ]
    res, compared, meta = tally(_panel(tmp_path, rows))
    assert compared == 2, "flat layout not detected -- both solvers must be found"
    assert set(res) == {"discopt", "baron"}
    assert res["discopt"]["certified"] == 1 and res["baron"]["certified"] == 1
    assert meta["n_rows"] == 1


def test_timing_columns_split_solved_median_from_total_wall(tmp_path):
    """Total wall covers ALL instances; the median covers only solved ones.

    CLAUDE.md requires both, because a solved-only statistic flatters whichever
    solver times out most -- its slowest instances leave the population. Here
    the timeout is the expensive run, so the two columns must disagree.
    """
    rows = [
        {
            "instance": "fast",
            "known": 1.0,
            "maximize": False,
            "runs": {"s": {"status": "1 Optimal", "objective": 1.0, "wall_time": 1.0}},
        },
        {
            "instance": "slow",
            "known": 2.0,
            "maximize": False,
            "runs": {"s": {"status": "time_limit", "objective": None, "wall_time": 60.0}},
        },
    ]
    res, _, _ = tally(_panel(tmp_path, rows, solvers=["s"]))
    assert res["s"]["median_solved_s"] == 1.0, "timeout leaked into the solved median"
    assert res["s"]["total_wall_s"] == 61.0, "timeout dropped from total wall"


def test_a_real_violation_still_surfaces(tmp_path):
    """ANTI-VACUITY CONTROL for the reclassification test above.

    Without this, a tally that returned `n/a` for everything would satisfy
    `test_stored_verdict_is_ignored...` while reporting no violation ever.
    This is `ex1252`: `1 Optimal` asserts a certified global at the wrong value.
    """
    rows = [
        {
            "instance": "ex1252",
            "known": 128893.741,
            "maximize": False,
            "runs": {"s": {"status": "1 Optimal", "objective": 223191.3362, "wall_time": 0.1}},
        }
    ]
    res, _, _ = tally(_panel(tmp_path, rows, solvers=["s"]))
    assert res["s"]["counts"][VIOLATION] == 1
    assert res["s"]["violations"] == [("ex1252", "1 Optimal")]
