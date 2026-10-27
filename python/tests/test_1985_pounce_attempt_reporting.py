"""#1985: ``solver="pounce"`` reporting across attempts, and the convex engine's ``qp_*`` options.

1. A solve that took several POUNCE calls (the #1768 continuation) reported only
   the published call's ``iteration_count``: a 331-iteration failure plus a
   14-iteration continuation read as 14. The published report now lists every
   call's counts (``statistics["attempts"]``) and their sum
   (``total_iteration_count``), flagged incomplete when POUNCE's own gh#884
   retry hid an attempt.
2. #1618's label asserted the answer was the base attempt's whenever
   ``dual_divergence_retry_promoted`` was False. On pounce-solver 0.13 that flag
   comes with the retry's answer, so the label contradicted the published
   answer. The attempt is now read from the document (does the trajectory end at
   the reported final values?).
3. ``qp_presolve``, ``qp_reg``, ``qp_hsde``, ``qp_equilibrate`` and
   ``qp_crossover`` were refused as "takes no such argument"; pounce-solver 0.13's
   ``pounce.qp.solve_qp`` takes them, and they are forwarded.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("pounce")

from discopt.solvers import _pounce_report as pr  # noqa: E402
from discopt.solvers import convex_ipm_pounce as cvx  # noqa: E402

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ── 2. the retried attempt is read from the document ─────────────────────────


def _retried_report(last_obj: float, last_du: float) -> dict:
    return {
        "statistics": {
            "iteration_count": 3,
            "final_objective": -5.0,
            "final_dual_inf": 1e-9,
            "num_obj_evals": 4,
            "dual_divergence_signature": True,
            "dual_divergence_retry_promoted": False,
        },
        "iterations": [
            {"iter": 0, "objective": 1.0, "inf_du": 1.0, "phase": "main"},
            {"iter": 1, "objective": last_obj, "inf_du": 9.0, "phase": "restoration"},
            {"iter": 3, "objective": last_obj, "inf_du": last_du, "phase": "main"},
        ],
    }


def test_retry_answer_is_labelled_retry_even_when_not_promoted():
    rep = _retried_report(-5.0, 1e-9)
    pr.label_retry_attempts(rep)
    st = rep["statistics"]
    assert st["counts_attempt"] == "retry"
    assert st["solution_attempt"] == "retry"
    assert st["total_counts_complete"] is False


def test_base_answer_is_labelled_base():
    rep = _retried_report(-4.0, 3e-2)  # the retry ended elsewhere
    pr.label_retry_attempts(rep)
    assert rep["statistics"]["solution_attempt"] == "base"


def test_label_matches_the_published_answer_end_to_end():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_issue_1618_remainder import _toll

    m, start = _toll()
    r = _quiet(m.solve, solver="pounce", initial_solution=start)
    st = r.solve_report["statistics"]
    if "counts_attempt" not in st:
        pytest.skip("POUNCE's gh#884 guard did not fire on this platform")
    rows = [x for x in r.solve_report["iterations"] if x.get("phase", "main") == "main"]
    ends_at_final = rows[-1]["inf_du"] == st["final_dual_inf"]
    assert st["solution_attempt"] == ("retry" if ends_at_final else "base")


# ── 1. every call's counts, and their total ─────────────────────────────────


def test_record_attempts_totals_every_call():
    first = {"statistics": {"iteration_count": 331, "num_obj_evals": 400}, "solution": {}}
    second = {"statistics": {"iteration_count": 14}, "solution": {"status": "SolveSucceeded"}}
    out = pr.record_attempts([first, second], 1)
    assert out is second
    st = out["statistics"]
    assert st["iteration_count"] == 14  # the published call's own count is kept
    assert st["total_iteration_count"] == 345
    assert st["total_counts_complete"] is True
    assert [a["published"] for a in st["attempts"]] == [False, True]
    assert st["attempts"][0]["iteration_count"] == 331
    assert st["attempts"][0]["num_obj_evals"] == 400


def test_record_attempts_flags_an_incomplete_total():
    retried = _retried_report(-5.0, 1e-9)
    pr.label_retry_attempts(retried)
    out = pr.record_attempts([retried, {"statistics": {"iteration_count": 2}}], 1)
    assert out["statistics"]["total_iteration_count"] == 5
    assert out["statistics"]["total_counts_complete"] is False
    out = pr.record_attempts([None, {"statistics": {"iteration_count": 2}}], 1)
    assert out["statistics"]["total_counts_complete"] is False
    assert out["statistics"]["attempts"][0]["report_missing"] is True


def test_continuation_report_covers_both_calls():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_issue_1618_remainder import _toll

    m, start = _toll()
    r = _quiet(m.solve, solver="pounce", initial_solution=start)
    st = r.solve_report["statistics"]
    if "attempts" not in st:
        pytest.skip("POUNCE did not stop mid-path on this platform; no continuation ran")
    assert len(st["attempts"]) == 2
    assert sum(a["published"] for a in st["attempts"]) == 1
    assert st["total_iteration_count"] == sum(a["iteration_count"] for a in st["attempts"])
    assert st["total_iteration_count"] > st["iteration_count"]


# ── 3. the convex engine's qp_* options are forwarded ───────────────────────


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.maximize(x[0] + x[1] + 0 * x[2])
    m.subject_to(x[0] + x[1] <= 4)
    m.subject_to(x[2] <= 3)
    return m


needs_013 = pytest.mark.skipif(not cvx._engine_takes("qp_crossover"), reason="pounce-solver < 0.13")


@needs_013
def test_qp_crossover_returns_a_vertex():
    interior = _quiet(_lp().solve, solver="pounce")
    vertex = _quiet(_lp().solve, solver="pounce", pounce_options={"qp_crossover": True})
    assert interior.status == vertex.status == "optimal"
    assert vertex.objective == pytest.approx(4.0, abs=1e-9)
    # The free column x[2] sits strictly inside its box at the IPM's answer and on
    # a bound at the vertex.
    assert 0.1 < interior.x["x"][2] < 2.9
    assert min(abs(vertex.x["x"][2]), abs(vertex.x["x"][2] - 3.0)) < 1e-9


@needs_013
@pytest.mark.parametrize(
    "opts",
    [
        {"qp_presolve": True},
        {"qp_reg": 1e-9},
        {"qp_hsde": False},
        {"qp_equilibrate": "no"},
    ],
)
def test_qp_options_are_accepted(opts):
    r = _quiet(_lp().solve, solver="pounce", pounce_options=opts)
    assert r.status == "optimal" and r.objective == pytest.approx(4.0, abs=1e-6)


@needs_013
def test_qp_options_reach_the_engine(monkeypatch):
    import functools

    import pounce.qp

    seen = {}
    real = pounce.qp.solve_qp

    @functools.wraps(real)  # keeps the signature _engine_takes reads
    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(pounce.qp, "solve_qp", spy)
    _quiet(_lp().solve, solver="pounce", pounce_options={"qp_reg": 1e-9, "qp_presolve": True})
    assert seen.get("qp_reg") == 1e-9 and seen.get("qp_presolve") is True


@needs_013
def test_presolve_or_crossover_with_a_warm_start_is_refused():
    Q = np.eye(2)
    c = np.array([-1.0, -2.0])
    with pytest.raises(ValueError, match="cannot be combined with a warm start"):
        cvx.solve_qp(Q, c, warm_start=np.zeros(2), options={"qp_crossover": True})


def test_old_pounce_refusal_names_the_version(monkeypatch):
    monkeypatch.setattr(cvx, "_engine_takes", lambda key: False)
    with pytest.raises(ValueError, match="only from pounce-solver 0.13"):
        cvx.convex_engine_options({"qp_crossover": True})


def test_unknown_qp_option_is_still_refused():
    with pytest.raises(ValueError, match="not options pounce.qp.solve_qp"):
        cvx.convex_engine_options({"qp_no_such_knob": 1})
