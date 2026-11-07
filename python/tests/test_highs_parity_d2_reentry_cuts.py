"""highs-parity D2: the #698 re-entry run of the Rust MILP engine and the root cut budget.

``_solve_milp_simplex`` calls the Rust driver twice. Run 1 gets the graduated root cut
budget (``_milp_root_cut_budget``: 500 cuts over 50 rounds). Run 2 rebuilds the tree from
the root, but it was called without it, so it separated with the binding's single
16-cut pass. ``DISCOPT_MILP_REENTRY_CUTS=1`` passes the budget to run 2 as well. The
flag is default-OFF until its §5 panel passes, so the OFF arm must keep the legacy call.

The first call's status is forced to ``feasible`` so the re-entry fires
deterministically, as in ``test_1323_abs_gap_routes``.
"""

import numpy as np
import pytest
from discopt.solver import solve_model
from test_1323_abs_gap_routes import _knapsackish


def _bb_calls(monkeypatch, flag: str | None) -> list[dict]:
    import discopt._rust as _rust

    if flag is None:
        monkeypatch.delenv("DISCOPT_MILP_REENTRY_CUTS", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_MILP_REENTRY_CUTS", flag)
    monkeypatch.delenv("DISCOPT_MILP_ROOT_CUTS", raising=False)
    real = _rust.solve_milp_csc_py
    calls: list[dict] = []

    def recording(*args, **kwargs):
        calls.append(dict(kwargs))
        out = real(*args, **kwargs)
        if len([c for c in calls if "initial_incumbent" in c]) == 1 and out[0] == "optimal":
            return ("feasible", out[1], out[2], -np.inf, out[4], out[5])
        return out

    monkeypatch.setattr(_rust, "solve_milp_csc_py", recording)
    solve_model(_knapsackish(), nlp_solver="simplex", time_limit=20.0, verify_incumbent=False)
    bb = [c for c in calls if "initial_incumbent" in c]
    assert len(bb) >= 2, f"the re-entry never fired; recorded {len(calls)} driver calls"
    return bb


@pytest.mark.smoke
def test_reentry_gets_the_root_cut_budget_when_flagged(monkeypatch):
    run1, run2 = _bb_calls(monkeypatch, "1")[:2]
    assert run1.get("root_cuts") == 500 and run1.get("cut_rounds") == 50
    assert run2.get("root_cuts") == 500, f"run 2 got {run2.get('root_cuts')!r}"
    assert run2.get("cut_rounds") == 50 and run2.get("cut_select") is True
    assert run2.get("root_cut_time_s", 0.0) >= 0.5


@pytest.mark.smoke
def test_reentry_keeps_the_legacy_call_by_default(monkeypatch):
    run1, run2 = _bb_calls(monkeypatch, None)[:2]
    assert run1.get("root_cuts") == 500
    assert "root_cuts" not in run2 and "cut_rounds" not in run2
