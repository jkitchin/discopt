"""Regression: regime-aware certified-objective neutrality (CUTOFF-SOUND-1).

The graduation gate's cert-neutrality check compares a flag-ON re-solve of the
cert panel against ``cert-baseline.jsonl``. Its objective tolerance (``OBJ_TOL``
1e-8) is a *byte-reproducibility* tolerance — correct for a **bound-neutral**
change (refactor/cache), where any objective drift is a bug.

For a **bound-changing** flag (a reduction/relaxation/cut behind a default-OFF env
flag) that tolerance is a *category error*: the flag legitimately alters the search
tree, so the certified objective may drift beyond 1e-8 while staying well within
correctness tolerance and — crucially — landing *closer to or exactly on* the true
optimum. The graduation gate flagged exactly this on the R2 cutoff-reduction:

* ``ex1225`` node_reduce: 30.999999951817372 -> **31.0** (the true optimum),
* ``st_e38`` root_fixpoint: 7197.727116839705 -> 7197.727148532429 (true optimum
  7197.727148524341 — the ON value is ~1e-8 from the true optimum, the OFF baseline
  was ~3e-5 *below* it).

Both drifts are TOWARD the true optimum and inside correctness tolerance, yet the
byte-reproducibility check flagged them as "objective" soundness violations — a
gate false-positive. This test pins the fix: in the ``bound_changing`` regime the
objective check brackets against the true optimum (``oracle``) with the correctness
tolerance, so a benign toward-optimum drift is NOT a violation, while a genuine
false certificate (a cross of the true optimum beyond correctness tolerance) STILL
is. The default ``bound_neutral`` regime keeps byte-reproducibility unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.cert_neutrality import check_neutrality  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.correctness]


def _row(obj: float, *, status: str = "optimal", nodes: int = 7) -> dict:
    return {"status": status, "objective": obj, "node_count": nodes}


# The two instances the graduation gate flagged, with their exact numbers.
EX1225_BASE = 30.999999951817372
EX1225_ON = 31.0  # node_reduce drifts ONTO the true optimum
EX1225_OPT = 31.0

ST_E38_BASE = 7197.727116839705
ST_E38_ON = 7197.727148532429  # root_fixpoint drifts toward opt
ST_E38_OPT = 7197.727148524341


def test_bound_neutral_still_byte_strict():
    """A bound-neutral flag must reproduce the objective to ~1e-8 — the toward-opt
    ex1225 drift (4.8e-8) IS a violation in this regime (unchanged behavior)."""
    base = {"ex1225": _row(EX1225_BASE)}
    new = {"ex1225": _row(EX1225_ON)}
    viol = check_neutrality(new, base, regime="bound_neutral")
    kinds = [v.kind for v in viol]
    assert "objective" in kinds, "bound-neutral drift beyond 1e-8 must flag (byte-strict)"


def test_bound_changing_toward_optimum_is_not_a_violation():
    """The CUTOFF-SOUND-1 fix: in the bound-changing regime, a drift that lands ON
    the true optimum (ex1225 -> 31.0) is NOT a soundness violation. This assertion
    FAILS before the fix (the old check flagged it as 'objective')."""
    base = {"ex1225": _row(EX1225_BASE)}
    new = {"ex1225": _row(EX1225_ON)}
    oracle = {"ex1225": EX1225_OPT}
    viol = check_neutrality(new, base, regime="bound_changing", oracle=oracle)
    assert viol == [], f"benign toward-optimum drift flagged as violation: {viol}"


def test_bound_changing_st_e38_toward_optimum_is_not_a_violation():
    """st_e38 root_fixpoint: ON is ~1e-8 from the true optimum, OFF baseline was
    ~3e-5 below it. The ON value is MORE accurate — not a violation."""
    base = {"st_e38": _row(ST_E38_BASE, nodes=3)}
    new = {"st_e38": _row(ST_E38_ON, nodes=3)}
    oracle = {"st_e38": ST_E38_OPT}
    viol = check_neutrality(new, base, regime="bound_changing", oracle=oracle)
    assert viol == [], f"toward-optimum st_e38 drift flagged as violation: {viol}"


def test_bound_changing_still_catches_a_real_false_certificate():
    """The fix must NOT weaken below true correctness: a certified objective that
    crosses the true optimum by more than correctness tolerance is STILL flagged as
    an 'objective' violation (a genuine false certificate)."""
    # Claim optimal at 25.0 when the true optimum is 31.0 (min) — a gross wrong cert.
    base = {"ex1225": _row(EX1225_BASE)}
    new = {"ex1225": _row(25.0)}
    oracle = {"ex1225": EX1225_OPT}
    viol = check_neutrality(new, base, regime="bound_changing", oracle=oracle)
    kinds = [v.kind for v in viol]
    assert "objective" in kinds, "a real false certificate must still be flagged"


def test_bound_changing_no_oracle_falls_back_to_correctness_drift():
    """With no oracle for the instance, the bound-changing check falls back to a
    correctness-tolerance drift bound vs baseline: a tiny drift passes, a gross one
    is still caught."""
    base = {"foo": _row(100.0)}
    # tiny benign drift (within correctness tol) — not a violation
    assert check_neutrality({"foo": _row(100.00001)}, base, regime="bound_changing") == []
    # gross drift (well beyond correctness tol) — still a violation
    viol = check_neutrality({"foo": _row(120.0)}, base, regime="bound_changing")
    assert [v.kind for v in viol] == ["objective"]


def test_bound_changing_status_and_missing_still_enforced():
    """Regime-awareness only relaxes the *objective reproducibility* tolerance; a
    lost optimal status or a missing instance is still a hard violation."""
    base = {"a": _row(1.0), "b": _row(2.0)}
    new = {"a": _row(1.0, status="feasible")}  # 'b' missing, 'a' not optimal
    oracle = {"a": 1.0, "b": 2.0}
    kinds = {v.kind for v in check_neutrality(new, base, regime="bound_changing", oracle=oracle)}
    assert "status" in kinds
    assert "missing" in kinds
