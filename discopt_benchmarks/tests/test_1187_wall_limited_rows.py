"""A wall-limited row is not neutrality evidence (#1187, part 2).

``deterministic=True`` renders the **role-2** wall budgets inert — the sub-budgets
that decide *how much work* a stage does. It deliberately leaves ``time_limit``
alone, because ``time_limit`` is role 1: neutralizing it would let a solve run
without any wall bound, trading a reproducibility bug for a broken promise
(CLAUDE.md §1). The consequence is stated in the flag's own docstring and is easy
to forget when reading a panel:

    **``deterministic=True`` cannot equalise work on a run that terminates on the
    wall clock, because the terminating condition IS the wall clock.**

So two arms that both ended on ``time_limit`` did *different amounts of work*, and
any objective or ``node_count`` difference between them is a difference in work,
not in behaviour. #1180's corpus sweep read 13 of 66 rows that way and
manufactured a reproducible "0.516x regression" that re-measured, on the ordinary
wall budget, as a 5x-more-nodes, 30 %-tighter-bound *improvement*.

The measured second half of #1187 is the same fact from the other side: on
``beuster`` at ``time_limit=120`` with ``deterministic=True``, two builds that
differed only in Python marshaling cost issued **3858 OBBT probe LPs against
942** — 4.1x the work — for the same 3 nodes and the same bound, both ending
``status=time_limit``.

The fix is not to soften the check. It is to refuse to read a verdict off such a
row and to *say so*: ``wall_limited_rows`` names them, ``check_neutrality`` skips
exactly the named set, and both gate scripts print the count. A row excluded
without being reported would be the weakening; a row compared anyway is noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.cert_neutrality import (  # noqa: E402
    WALL_LIMIT_STATUSES,
    check_neutrality,
    wall_limited_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.correctness]


def _row(obj: float | None, *, status: str = "optimal", nodes: int = 7) -> dict:
    return {"status": status, "objective": obj, "node_count": nodes}


def test_both_arms_wall_limited_is_not_evidence():
    base = {"beuster": _row(100.0, status="time_limit", nodes=3)}
    new = {"beuster": _row(70.0, status="time_limit", nodes=15)}

    skipped = wall_limited_rows(new, base)
    assert set(skipped) == {"beuster"}
    assert "#1187" in skipped["beuster"], "the exclusion must carry its reason"

    # Compared anyway, this row reads as a 30 % objective drift AND a 400 % node
    # regression — two violations invented by the budget.
    assert len(check_neutrality(new, base)) >= 1
    # Excluded, it yields no verdict at all.
    assert check_neutrality(new, base, exclude=skipped) == []


def test_a_wall_cut_row_that_reports_feasible_is_caught():
    """Status alone is not enough, and this is the COMMON case.

    A run cut off by ``time_limit`` while holding an incumbent reports
    ``feasible``, not ``time_limit``. Measured on ``tls2``: every run ends
    ``feasible`` at the wall, and three *baseline* runs returned 245 / 217 / 179
    nodes with three different dual bounds — an instance that does not reproduce
    against itself. A status-only test compares it anyway and charges the
    difference to whatever change is under review.
    """
    # Two of tls2's own baseline runs, in the order that trips the node guard.
    budgets = {"tls2": 30.0}
    base = {"tls2": {"status": "feasible", "objective": 5.3, "node_count": 179, "wall_time": 31.1}}
    new = {"tls2": {"status": "feasible", "objective": 5.3, "node_count": 245, "wall_time": 30.6}}

    assert wall_limited_rows(new, base) == {}, "no budgets -> cannot tell, must not guess"
    skipped = wall_limited_rows(new, base, budgets=budgets)
    assert set(skipped) == {"tls2"}
    assert check_neutrality(new, base, exclude=skipped) == []
    # Without the exclusion this reads as a 37 % node regression invented by the wall
    # — and both rows are the SAME build.
    assert any(v.kind == "node_regression" for v in check_neutrality(new, base))


def test_a_settled_row_is_never_excluded_by_its_wall_time():
    """An instance that certified in its last second is a verdict, not a coincidence."""
    budgets = {"foo": 30.0}
    base = {"foo": {"status": "optimal", "objective": 1.0, "node_count": 5, "wall_time": 29.9}}
    new = {"foo": {"status": "optimal", "objective": 2.0, "node_count": 5, "wall_time": 29.9}}
    assert wall_limited_rows(new, base, budgets=budgets) == {}
    assert any(v.kind == "objective" for v in check_neutrality(new, base))


def test_a_lost_certification_is_still_a_violation():
    """The exclusion must not swallow a real regression.

    baseline ``optimal`` -> new ``time_limit`` is the flag losing a certificate the
    reference had. Only one arm is wall-limited, so the row is NOT excluded — and
    that is deliberate even for an instance known to be wall-flaky. A certified
    baseline is a verdict; refusing to compare against it because the new run ran
    out of clock would hide exactly the regression this check exists for. The cost
    is a false positive on an instance that does not reproduce against itself,
    which a re-run settles; the alternative silently accepts a lost certificate.
    """
    base = {"foo": _row(100.0, status="optimal")}
    new = {"foo": _row(None, status="time_limit")}

    assert wall_limited_rows(new, base) == {}
    kinds = {v.kind for v in check_neutrality(new, base)}
    assert "status" in kinds, "a lost certification must still fail the gate"


def test_node_limited_rows_stay_comparable():
    """``max_nodes`` is a DETERMINISTIC budget, so two runs that hit it did the
    same work and remain comparable. Excluding them would be the weakening this
    change is careful not to be."""
    assert "node_limit" not in WALL_LIMIT_STATUSES
    base = {"foo": _row(100.0, status="node_limit", nodes=20)}
    new = {"foo": _row(70.0, status="node_limit", nodes=20)}

    assert wall_limited_rows(new, base) == {}
    kinds = {v.kind for v in check_neutrality(new, base)}
    assert "objective" in kinds


def test_exclusion_does_not_hide_a_missing_row():
    """ "we chose not to read this row" and "the row is not there" are different
    facts, and only the first is excludable."""
    base = {"foo": _row(100.0, status="time_limit")}
    viol = check_neutrality({}, base, exclude={"foo"})
    assert [v.kind for v in viol] == ["missing"]


def test_default_is_unchanged():
    """No caller that does not pass ``exclude`` may see different behaviour."""
    base = {"foo": _row(100.0, status="time_limit")}
    new = {"foo": _row(70.0, status="time_limit")}
    assert len(check_neutrality(new, base)) >= 1


def test_both_gate_scripts_actually_exclude():
    """A helper no gate calls is a documented promise, not an enforced one.

    #1187 asks for the rule to be enforced in the harness, and the failure mode it
    guards against is precisely a rule that exists only in prose.
    """
    checked = 0
    for rel in ("scripts/check_cert_neutrality.py", "scripts/graduation_gate.py"):
        text = (_BENCH_ROOT / rel).read_text()
        assert "wall_limited_rows" in text, f"{rel} does not compute the excluded set"
        assert "budgets=budgets" in text, (
            f"{rel} does not pass the budgets, so it only catches an explicit "
            "time_limit status and misses the wall-cut feasible rows"
        )
        assert "exclude=" in text, f"{rel} does not pass it to check_neutrality"
        assert "UNMEASURED" in text, f"{rel} excludes rows without reporting them"
        checked += 1
    assert checked == 2, "the probe stopped reading files (rule 6)"


def test_graduation_gate_worker_source_still_compiles():
    """``graduation_gate`` builds its worker as a STRING and ``exec``s it in a
    subprocess, so a syntax error in the wiring above is invisible until a gate
    run fails an hour in. Build it here and compile it.
    """
    import types

    import scripts.graduation_gate as gg

    captured: dict[str, str] = {}

    def fake_run(args, **kw):
        captured["src"] = args[-1]
        return types.SimpleNamespace(stdout="", stderr="")

    real = gg.subprocess.run
    gg.subprocess.run = fake_run
    try:
        gg.run_cert_neutrality("test", {}, None)
    finally:
        gg.subprocess.run = real

    src = captured.get("src")
    assert src, "the worker source was never built — the probe measured nothing"
    compile(src, "<graduation_gate worker>", "exec")
    for needle in ("wall_limited_rows", "exclude=skipped", "SKIPJSON:"):
        assert needle in src, f"the worker lost its #1187 wiring: {needle}"


def test_a_failing_arm_names_the_rows_that_failed(capsys):
    """A gate that exits 1 must say WHICH rows did it, on stdout.

    ``graduation_gate`` printed only ``cert=FAIL`` per arm; the offending rows
    existed solely inside the uploaded JSON artifact. A CI failure whose detail is
    unreachable from the log is a failure nobody can act on — which is how run
    34072171566 reported four failing arms that could not be attributed to anything.

    Proven to fire rather than assumed to: the same call with an empty violation
    list must print no per-row line at all.
    """
    import json
    import types

    import scripts.graduation_gate as gg

    def _run_with(viol: list[dict]) -> str:
        def fake_run(args, **kw):
            return types.SimpleNamespace(
                stdout=(
                    "SKIPJSON:" + json.dumps({}) + "\n"
                    "ROWSJSON:" + json.dumps({}) + "\n"
                    "CERTJSON:" + json.dumps(viol) + "\n"
                ),
                stderr="",
            )

        real = gg.subprocess.run
        gg.subprocess.run = fake_run
        try:
            gg.run_cert_neutrality("lift_loose_products", {}, None)
        finally:
            gg.subprocess.run = real
        return capsys.readouterr().out

    noisy = _run_with(
        [
            {"instance": "st_e36", "kind": "objective", "detail": "objective 1.0 -> None"},
            {
                "instance": "st_e36",
                "kind": "status",
                "detail": "status=feasible (baseline optimal)",
            },
            {"instance": "tls2", "kind": "node_regression", "detail": "node_count 153 -> 255"},
        ]
    )
    assert "st_e36" in noisy, "the failing row was not named — the log still cannot be read"
    assert "objective 1.0 -> None" in noisy
    assert "status=feasible (baseline optimal)" in noisy
    assert "tls2" in noisy, "the perf note was dropped; it is context for the failure"
    assert "2 soundness-class violation(s), 1 node_count note(s)" in noisy

    quiet = _run_with([])
    assert "soundness-class violation(s)" not in quiet, (
        "the report fires on a clean arm — it would print on every green run and "
        "so prove nothing when it prints on a red one"
    )


def test_an_uncertified_incumbent_is_not_a_false_certificate():
    """An incumbent above the optimum is an open gap, not a wrong certificate.

    ``_objective_violation`` bracketed EVERY row's objective against the oracle
    and labelled a disagreement ``FALSE CERTIFICATE`` — including rows that never
    certified. A ``feasible`` row holds an incumbent, and an incumbent sitting
    above the true optimum is the *expected* shape of an unclosed gap.

    Measured on the vendored cert panel at half budget: ``nvs05`` came back
    ``feasible`` with objective ``1107.8904814191037`` in BOTH arms — the same
    number to the last bit, the flag having changed nothing — and the comparison
    reported a soundness-class violation against the true optimum ``5.47``. Same
    for ``nvs22`` (``33.55166`` in both arms). A guard that hard-fails on two
    identical numbers is not measuring the flag; it is measuring whether the row
    finished, which is what the runner decides.

    ``graduation_gate`` already states the correct rule for its own control-panel
    drift check — "Only a certified row carries a certificate ... so it is
    skipped" — and this function did not implement it.
    """
    oracle = {"nvs05": 5.470934108225147}
    incumbent = {
        "nvs05": {"status": "feasible", "objective": 1107.8904814191037, "node_count": 111}
    }
    # Both arms uncertified and IDENTICAL: nothing here is attributable to the flag.
    viol = check_neutrality(incumbent, incumbent, regime="bound_changing", oracle=oracle)
    kinds = [v.kind for v in viol]
    assert "objective" not in kinds, (
        f"an uncertified incumbent was called a false certificate: "
        f"{[(v.instance, v.kind, v.detail) for v in viol]}"
    )

    # The guard must KEEP full strength where a certificate really exists: the same
    # wrong number, but certified, is a genuine false certificate and must fail.
    certified = {"nvs05": {"status": "optimal", "objective": 1107.8904814191037, "node_count": 111}}
    viol_cert = check_neutrality(certified, certified, regime="bound_changing", oracle=oracle)
    assert any(v.kind == "objective" for v in viol_cert), (
        "a CERTIFIED objective disagreeing with the true optimum must still be a "
        "false certificate — the fix must not weaken the soundness check"
    )
    assert "FALSE CERTIFICATE" in " ".join(v.detail for v in viol_cert)


def test_a_lost_certificate_is_still_caught_after_the_incumbent_fix():
    """The fix must not swallow the case it superficially resembles.

    ``optimal`` -> ``feasible`` is a certification the run LOST, and stays a
    violation on both the status and the objective axis. Only the *labelling* of
    an uncertified incumbent as a false certificate is removed.
    """
    base = {"x": {"status": "optimal", "objective": 10.0, "node_count": 5}}
    lost_value = {"x": {"status": "feasible", "objective": None, "node_count": 5}}
    kinds = {v.kind for v in check_neutrality(lost_value, base, regime="bound_changing")}
    assert kinds == {"status", "objective"}, kinds

    # ... and when it still reports a number, the lost *status* is the violation.
    lost_status = {"x": {"status": "feasible", "objective": 10.0, "node_count": 5}}
    kinds2 = {v.kind for v in check_neutrality(lost_status, base, regime="bound_changing")}
    assert "status" in kinds2, kinds2
