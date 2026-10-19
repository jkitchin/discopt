"""#1775 G0.6: scoring of the open-solver GDP baseline (``scripts/gdp_baseline.py``).

The scorer decides what the gap table calls solved, so it is pinned on the cases
the 300 s run actually produced: a GDPopt-GLOA "optimal" 6e-4 below the certified
optimum of a maximization (scored ``solved`` under the first, 0.1 % tolerance), an
"infeasible" claim on a feasible instance, and a SIGKILL whose worker had printed
warnings first (scored ``error`` instead of a kill).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmarks.gdplib_runner import OracleRecord

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gdp_baseline.py"


@pytest.fixture(scope="module")
def gb():
    spec = importlib.util.spec_from_file_location("gdp_baseline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MED_TERM = OracleRecord(6797.539701513444, "highs")


def _rec(**kw):
    return {"instance": "i", "arm": "a", "minimize": False, **kw}


def test_optimal_claim_outside_rel_1e4_is_incorrect(gb):
    """GDPopt-GLOA on med_term_purchasing (maximize): 'optimal' at 6793.458."""
    assert gb.score(_rec(status="optimal", objective=6793.4582021), MED_TERM) == "incorrect"


def test_optimal_claim_within_tolerance_is_solved(gb):
    assert gb.score(_rec(status="optimal", objective=6797.5397015134), MED_TERM) == "solved"
    near = 6797.539701513444 * (1 - 0.9e-4)
    assert gb.score(_rec(status="optimal", objective=near), MED_TERM) == "solved"


def test_a_better_than_optimal_incumbent_is_incorrect(gb):
    assert gb.score(_rec(status="feasible", objective=6800.0), MED_TERM) == "incorrect"
    assert gb.score(_rec(status="feasible", objective=6700.0), MED_TERM) == "unsolved"


def test_infeasible_claim_on_a_feasible_instance_is_incorrect(gb):
    assert gb.score(_rec(status="infeasible"), MED_TERM) == "incorrect"
    unproven = OracleRecord(None, "unproven", "gap open")
    assert gb.score(_rec(status="infeasible"), unproven) == "unsolved"
    assert gb.score(_rec(status="infeasible"), unproven, feasible_known=True) == "incorrect"


def test_sigkill_counts_as_a_kill_even_after_output(gb):
    rec = _rec(error="WARNING (W1002): Setting Var 'Ltotal[4,25]' ...", returncode=-9)
    assert gb.score(rec, MED_TERM) == "unsolved"
    assert gb.score(_rec(error="ValueError: boom", returncode=0), MED_TERM) == "error"


def test_a_point_infeasible_in_pyomo_is_incorrect_even_unproven(gb):
    unproven = OracleRecord(None, "unproven", "gap open")
    bad = {"max_rel_violation": 0.02, "bad_counts": 0, "bad_logic": 0}
    assert gb.score(_rec(status="optimal", objective=1.0, in_pyomo=bad), unproven) == "incorrect"
