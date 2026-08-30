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


# --------------------------------------------------------------------------- #
# Reference-relative semantics: the checks must describe a REGRESSION against
# whatever reference they are given, not an absolute property of the new run.
#
# The graduation gate's CI subset compares each arm against a flag-OFF panel
# measured in the same session, instead of against the committed
# cert-baseline.jsonl snapshot (which had drifted a month behind `main`, failing
# all 7 arms identically and reporting it as `soundness=FAIL`). Both checks below
# were phrased absolutely, which is indistinguishable from reference-relative on
# the committed baseline but wrong against a live one.
# --------------------------------------------------------------------------- #


def test_committed_baseline_rows_are_all_optimal_with_an_objective():
    """The property that makes both changes below no-ops on the pre-existing path.

    ``gen_cert_baseline`` writes only the deterministically-certifying subset, so
    every committed row is ``optimal`` with a non-null objective. While that holds,
    `base.status == "optimal"` is always true and `nb is None` is always false, so
    the reference-relative checks behave exactly as the absolute ones did for every
    caller that passes the committed baseline (``check_cert_neutrality.main``, and
    the gate's full/nightly path). If this ever fails, those two changes stop being
    behaviour-preserving there and both need re-deriving.
    """
    from utils.cert_neutrality import load_baseline  # noqa: PLC0415

    from scripts.check_cert_neutrality import _CERT_BASELINE  # noqa: PLC0415

    baseline = load_baseline(_CERT_BASELINE)
    assert baseline, "committed cert baseline is empty"
    bad_status = {k: v.get("status") for k, v in baseline.items() if v.get("status") != "optimal"}
    bad_obj = [k for k, v in baseline.items() if v.get("objective") is None]
    assert not bad_status, f"committed baseline rows that are not optimal: {bad_status}"
    assert not bad_obj, f"committed baseline rows with a null objective: {bad_obj}"


def test_status_not_certified_in_both_arms_is_not_the_flags_fault():
    """clay0303hfsg / nvs05 / tanksize: not certified inside the 60 s budget with the
    flag OFF *or* ON. That is the budget and the machine, not the flag.

    Fails before the change (the check demanded ``new.status == "optimal"``
    outright, so it flagged an instance the reference could not certify either).
    """
    ref = {"clay0303hfsg": _row(None, status="time_limit")}
    new = {"clay0303hfsg": _row(None, status="time_limit")}
    viol = check_neutrality(new, ref, regime="bound_changing")
    assert [v.kind for v in viol] == [], f"charged the flag for a budget miss: {viol}"


def test_status_lost_relative_to_the_reference_is_still_a_violation():
    """The control for the test above: the case that actually matters — the arm
    lost a certification the reference had — must still be flagged."""
    ref = {"gbd": _row(2.2)}
    new = {"gbd": _row(None, status="time_limit")}
    kinds = {v.kind for v in check_neutrality(new, ref, regime="bound_changing")}
    assert "status" in kinds, "a certification lost relative to the reference must flag"


def test_no_certificate_on_either_side_is_not_an_objective_violation():
    """``objective None -> None``: neither side certified, so there is no
    certificate for either to be wrong about.

    Fails before the change — ``nb is None or no is None`` made this an
    ``objective`` violation, which is soundness-class and hard-fails the gate. It
    is what kept the CI subset red even once the status check was made relative.
    """
    ref = {"clay0303hfsg": _row(None, status="time_limit")}
    new = {"clay0303hfsg": _row(None, status="time_limit")}
    kinds = [v.kind for v in check_neutrality(new, ref, regime="bound_changing")]
    assert "objective" not in kinds, "None -> None reported as a false certificate"


def test_certifying_what_the_reference_could_not_is_not_a_violation():
    """``None -> value``: the arm certified an instance the reference could not.
    That is an improvement, not a false certificate."""
    ref = {"tanksize": _row(None, status="time_limit")}
    new = {"tanksize": _row(1.2686437598530085)}
    viol = check_neutrality(new, ref, regime="bound_changing")
    assert viol == [], f"an improvement reported as a violation: {viol}"


def test_losing_a_certificate_is_still_an_objective_violation():
    """The control: ``value -> None`` stays soundness-class. This is the only shape
    the pre-existing ``nb is None or no is None`` branch could ever take against the
    committed baseline, and it is unchanged."""
    ref = {"gbd": _row(2.2)}
    new = {"gbd": _row(None, status="time_limit")}
    kinds = [v.kind for v in check_neutrality(new, ref, regime="bound_changing")]
    assert "objective" in kinds, "a lost certificate must stay an objective violation"
