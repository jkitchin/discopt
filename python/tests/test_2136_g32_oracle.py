"""#2136: the G3.2 replay oracle judges against verified points, not HiGHS's status.

The first replay oracle trusted HiGHS's "Optimal" and never checked a Rust ``optimal``
against its own bound. On the clay0203 GDP master it flagged the wrong side (HiGHS's
false optimum, #2069) and missed the real defect (Rust ``optimal`` with bound -752904
against objective 41907). These tests pin the replacement rules on synthetic rows.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "discopt_benchmarks/scripts/g32")
)
import oracle  # noqa: E402


def _arm(status, obj, bound, viol=0.0, root=None):
    return {"status": status, "obj": obj, "bound": bound, "viol": viol, "root": root}


def test_clay0203_row_flags_the_rust_certificate_not_the_arm_against_highs():
    # The measured row: HiGHS "Optimal" 54489.4 at a verified point; Rust OFF
    # "optimal" 41907.46 (verified) with bound -752904; Rust ON feasible 55668.6.
    highs = {"status": "Optimal", "obj": 54489.415, "viol": 0.0}
    arms = {
        "arm0": _arm("optimal", 41907.460, -752903.98, viol=1.4e-14),
        "arm1": _arm("feasible", 55668.585, -np.inf),
    }
    v = oracle.judge(highs, arms, gap_tol=1e-4)
    assert v["best_primal"] == 41907.460 and v["best_from"] == "arm0"
    assert any("arm0 optimal with open gap" in b for b in v["bad"]), v
    # Nothing is charged to arm1 or to arm0's point against HiGHS's status ...
    assert not any("false optimal" in b for b in v["bad"]), v
    assert not any(b.startswith("arm1") for b in v["bad"]), v
    # ... and HiGHS's false optimum is recorded as a note, not a BAD row.
    assert any("highs false optimum" in n for n in v["notes"]), v
    assert v["checks"] > 0


def test_healthy_row_is_clean():
    highs = {"status": "Optimal", "obj": 41420.2719, "viol": 0.0}
    arms = {
        "arm0": _arm("optimal", 41420.2719, 41420.2719, root=-1.0),
        "arm1": _arm("optimal", 41420.2719, 41418.0, root=100.0),
    }
    v = oracle.judge(highs, arms, gap_tol=1e-4)
    assert v["bad"] == [] and v["notes"] == [], v


def test_false_optimal_and_crossing_bound_are_bad():
    highs = {"status": "Optimal", "obj": 10.0, "viol": 0.0}
    arms = {
        "arm0": _arm("optimal", 12.0, 12.0),  # beaten by HiGHS's verified 10
        "arm1": _arm("feasible", 11.0, 10.5, root=10.2),  # bound and root cross 10
    }
    v = oracle.judge(highs, arms, gap_tol=1e-4)
    assert any("arm0 false optimal" in b for b in v["bad"]), v
    assert any("arm1 bound" in b for b in v["bad"]), v
    assert any("arm1 root" in b for b in v["bad"]), v


def test_unverified_points_are_not_references():
    # HiGHS's point violates the rows: it must not become the reference, so the
    # arm's honest bound above it is not flagged.
    highs = {"status": "Optimal", "obj": 5.0, "viol": 1e-2}
    arms = {"arm0": _arm("optimal", 7.0, 7.0), "arm1": _arm("feasible", 8.0, 6.9, viol=0.5)}
    v = oracle.judge(highs, arms, gap_tol=1e-4)
    assert v["best_primal"] == 7.0 and v["best_from"] == "arm0"
    assert v["bad"] == ["arm1 returned point viol 0.5"], v


def test_infeasible_claim_against_a_verified_point():
    highs = {"status": "Infeasible", "obj": np.inf, "viol": None}
    arms = {
        "arm0": _arm("infeasible", np.inf, np.inf, viol=None),
        "arm1": _arm("feasible", 3.0, 1.0),
    }
    v = oracle.judge(highs, arms, gap_tol=1e-4)
    assert any("arm0 infeasible" in b for b in v["bad"]), v
    assert any("highs infeasible" in n for n in v["notes"]), v
