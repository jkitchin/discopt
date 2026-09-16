"""A certification lost at the wall is a perf fact, not a soundness fault (#1204).

#1187 established the rule for rows wall-limited in **both** arms: they are not
evidence either way, so they are excluded and reported UNMEASURED. It deliberately
left the *asymmetric* case alone — reference ``optimal``, arm out of clock — where
``check_neutrality`` raises a ``status`` violation and, when the objective goes to
``None``, the lost-certificate branch as well. Both are soundness-class, and both
hard-fail the arm.

The consequence is that whether an arm fails depends on the **control's** draw for
the edge rows, which on a shared CI runner is a coin flip. Measured over four
graduation-gate runs (#1204's own evidence): runs 21 and 24 failed on *different*
four-arm subsets from the same PR code, while run 25 on ``main`` drifted **more**
(5 violations vs 4) and passed — because its control failed to certify ``tanksize``
and so disarmed the tripwire for every arm. Run 29 (2026-09-14, ``main``) passed
the same way: its control lost ``nvs05`` and ``tanksize`` to the wall, which pushed
both rows into the both-arms exclusion where they could not fail anything.

The fix is NOT to stop looking at these rows. It is to classify the finding
correctly and keep reporting it:

* **soundness cannot be at stake.** ``optimal`` is in ``_SETTLED_STATUSES``, so
  ``_is_wall_limited`` is false for any certified row. A row excluded because *the
  arm* hit the wall is provably never certified, carries no certificate, and can
  therefore hide no false certificate.
* **what it does hide is a perf regression** — "this flag made the instance slower
  past the wall" — so it is reported as one (``wall_regression``), in the same
  perf-class bucket the gate already uses for ``node_regression``. Silently
  dropping it would be the weakening; that is what this file's last two tests
  exist to prevent.

The widening is *per check*, never per row, because one soundness question stays
live on an asymmetric row: when the **arm certified** and the reference was the
side that ran out of clock, the arm holds a certificate and it is still bracketed
against the oracle. A naive "exclude the row if either arm is wall-limited" would
drop exactly that check — it is pinned by
``test_a_certificate_is_bracketed_even_when_the_reference_lost_the_wall``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.cert_neutrality import (  # noqa: E402
    PERF_CLASS_KINDS,
    SOUNDNESS_CLASS_KINDS,
    check_neutrality,
    wall_limited_arms,
    wall_limited_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.correctness]

# tanksize's real numbers: it certifies in 31.0 s of its 60 s budget on the machine
# that generated cert-baseline.jsonl (0.52 of budget), which is why a runner ~2x
# slower tips it. The arm row is what run 24's control produced for it.
_BUDGETS = {"tanksize": 60.0}
_CERTIFIED = {
    "status": "optimal",
    "objective": 5.13,
    "node_count": 41,
    "wall_time": 31.0,
}
_LOST_AT_THE_WALL = {
    "status": "time_limit",
    "objective": None,
    "node_count": 96,
    "wall_time": 60.0,
}


def test_the_asymmetric_flip_is_not_a_soundness_fault():
    """The entry experiment: reference certifies, arm runs out of clock.

    Before #1204 this produced two soundness-class violations (``status`` and the
    lost-certificate ``objective``), either of which hard-fails the arm. After, it
    produces exactly one finding, perf-class, naming both walls.
    """
    base = {"tanksize": _CERTIFIED}
    new = {"tanksize": _LOST_AT_THE_WALL}

    # #1187's both-arms rule does not fire here — only one arm is wall-limited.
    assert wall_limited_rows(new, base, budgets=_BUDGETS) == {}

    arms = wall_limited_arms(new, base, budgets=_BUDGETS)
    assert set(arms) == {"tanksize"}, "the arm that hit the wall was not detected"
    assert arms["tanksize"].new is True
    assert arms["tanksize"].base is False

    viol = check_neutrality(new, base, wall_limited=arms, regime="bound_changing")
    kinds = [v.kind for v in viol]
    assert kinds == ["wall_regression"], (
        f"expected one perf-class wall_regression, got {kinds} — a certification "
        "lost at the wall must not be charged as soundness"
    )
    assert not (SOUNDNESS_CLASS_KINDS & set(kinds))
    assert set(kinds) <= PERF_CLASS_KINDS
    detail = viol[0].detail
    assert "60" in detail and "#1204" in detail, (
        f"the finding must name the budget it ran out of and its issue: {detail!r}"
    )


def test_the_finding_is_never_silent():
    """Reclassifying is not dropping.

    #1187's ``test_a_lost_certification_is_still_a_violation`` rejected excluding
    this row on the grounds that "the alternative silently accepts a lost
    certificate". That objection is answered by *reporting* it, not by charging it
    as soundness — so a run that loses a certification at the wall must still
    return a finding for it.
    """
    arms = wall_limited_arms(
        {"tanksize": _LOST_AT_THE_WALL}, {"tanksize": _CERTIFIED}, budgets=_BUDGETS
    )
    viol = check_neutrality(
        {"tanksize": _LOST_AT_THE_WALL},
        {"tanksize": _CERTIFIED},
        wall_limited=arms,
        regime="bound_changing",
    )
    assert len(viol) == 1, "a lost certification must still be reported"
    assert viol[0].instance == "tanksize"


def test_a_certification_lost_away_from_the_wall_is_still_soundness():
    """Only the wall buys the reclassification.

    An arm that stops being ``optimal`` while nowhere near its budget did not run
    out of clock — it changed behaviour, which is exactly what the guard is for.
    """
    early_loss = {"status": "feasible", "objective": 5.9, "node_count": 41, "wall_time": 4.0}
    base = {"tanksize": _CERTIFIED}
    new = {"tanksize": early_loss}

    arms = wall_limited_arms(new, base, budgets=_BUDGETS)
    assert arms == {}, "a row at 7 % of its budget is not wall-limited"

    kinds = {
        v.kind for v in check_neutrality(new, base, wall_limited=arms, regime="bound_changing")
    }
    assert "status" in kinds, "a lost certification away from the wall must still fail"
    assert "wall_regression" not in kinds


def test_a_certificate_is_bracketed_even_when_the_reference_lost_the_wall():
    """The check a naive row-wholesale exclusion would have dropped.

    Reference wall-limited, **arm certified**: the arm holds a certificate, and a
    certificate is bracketed against the oracle no matter what the other arm did.
    Excluding the row because *either* side hit the wall would delete a live
    false-certificate check — the one soundness question that is still answerable
    on an asymmetric row.
    """
    base = {
        "tanksize": {"status": "feasible", "objective": 9.9, "node_count": 12, "wall_time": 60.0}
    }
    # The arm certifies — and certifies something the oracle says is wrong.
    new = {"tanksize": {"status": "optimal", "objective": 1.0, "node_count": 41, "wall_time": 30.0}}

    arms = wall_limited_arms(new, base, budgets=_BUDGETS)
    assert arms["tanksize"].base is True and arms["tanksize"].new is False

    viol = check_neutrality(
        new, base, wall_limited=arms, regime="bound_changing", oracle={"tanksize": 5.13}
    )
    assert [v.kind for v in viol] == ["objective"], (
        "a false certificate must still fail even when the REFERENCE was the side "
        "that ran out of clock"
    )
    assert "FALSE CERTIFICATE" in viol[0].detail


def test_a_truncated_reference_is_not_a_yardstick():
    """The reference's own numbers stop being comparable when its clock ran out.

    A row the reference cut off at the wall did an amount of work set by the budget,
    so its node_count is not a bar the arm must clear, and its incumbent is not a
    value the arm must reproduce. Only the oracle bracket above survives.
    """
    base = {
        "tanksize": {"status": "feasible", "objective": 9.9, "node_count": 12, "wall_time": 60.0}
    }
    new = {
        "tanksize": {"status": "optimal", "objective": 5.13, "node_count": 96, "wall_time": 30.0}
    }

    arms = wall_limited_arms(new, base, budgets=_BUDGETS)
    viol = check_neutrality(new, base, wall_limited=arms, regime="bound_changing")
    assert viol == [], (
        "an 8x node_count over a clock-truncated reference, and a drift from its "
        "truncated incumbent, are both artifacts of the reference's budget"
    )
    # Proven to fire: without the wall context both artifacts are reported.
    kinds = {v.kind for v in check_neutrality(new, base, regime="bound_changing")}
    assert "node_regression" in kinds


def test_both_arms_wall_limited_still_yields_no_verdict():
    """#1187's rule is preserved, not replaced."""
    row = dict(_LOST_AT_THE_WALL)
    base = {"tanksize": dict(row, node_count=30, objective=8.0)}
    new = {"tanksize": dict(row, node_count=96, objective=7.0)}

    arms = wall_limited_arms(new, base, budgets=_BUDGETS)
    assert arms["tanksize"].base and arms["tanksize"].new
    assert check_neutrality(new, base, wall_limited=arms, regime="bound_changing") == []
    # and the #1187 reporting helper still names it, so the gate still says UNMEASURED
    assert set(wall_limited_rows(new, base, budgets=_BUDGETS)) == {"tanksize"}


def test_the_perf_and_soundness_buckets_are_exhaustive():
    """Every kind this module can emit must land in exactly one bucket.

    ``graduation_gate`` used to hardcode ``("objective", "status", "missing")`` as
    the fatal set, so a NEW kind would have defaulted to 'not fatal' silently by
    being in neither list. The buckets are now the module's own and a kind in
    neither is a bug this test catches.
    """
    assert not (SOUNDNESS_CLASS_KINDS & PERF_CLASS_KINDS)
    emitted = {
        "objective",
        "status",
        "missing",
        "oracle_bracket",
        "node_regression",
        "wall_regression",
    }
    assert emitted == (SOUNDNESS_CLASS_KINDS | PERF_CLASS_KINDS)


def test_the_gate_scripts_use_the_asymmetric_rule(  # noqa: D103
):
    """A rule the gate does not call is prose (#1187's own lesson, rule 6)."""
    checked = 0
    for rel in ("scripts/check_cert_neutrality.py", "scripts/graduation_gate.py"):
        text = (_BENCH_ROOT / rel).read_text()
        assert "wall_limited_arms" in text, f"{rel} still reads only the both-arms rule"
        assert "wall_limited=" in text, f"{rel} does not pass it to check_neutrality"
        checked += 1
    assert checked == 2, "the probe stopped reading files (rule 6)"
