"""The true optimum must lie between a row's dual bound and its incumbent.

Every other check in ``cert_neutrality`` is *differential* — it compares two runs,
and a budget can therefore invalidate it, which is the whole subject of #1187 and
#1204. This one is **absolute**: it compares ONE row against the model's own true
optimum, which no runner, budget or flag can change. So it applies to every row,
including the ones the wall rules decline to compare.

The gap it closes. #1195 correctly stopped bracketing *uncertified* incumbents
against the oracle: an incumbent ABOVE the optimum is the expected shape of an open
gap, not a wrong answer, and the old check hard-failed two graduation-gate arms on
``nvs05`` sitting 27 % above it. But "don't check uncertified incumbents" also
stopped catching an incumbent BELOW the optimum — a point better than the optimum
cannot be feasible, which is a wrong answer at any certification status — and left
the *dual bound* of an uncertified row unchecked entirely, though a bound past the
optimum would prune the optimum away.

Both directions are caught by one sense-free rule, which is why it can live in a
module with no solver dependency::

    minimization:  bound     <= opt <= incumbent
    maximization:  incumbent <= opt <= bound
    either:        min(bound, incumbent) - tol <= opt <= max(bound, incumbent) + tol

Falsified before shipping, on real solves rather than fixtures (CLAUDE.md §4): 96
runs over the 48 vendored panel instances at 2 s and 8 s budgets — small on purpose,
to force open gaps — gave **87 bracketable rows, 10 of them uncertified, and zero
violations**. The 9 skips were rows with no oracle, no incumbent yet, or no dual
bound at all (``alan`` and ``fac2`` at 2 s, where the relaxation layer produced
none). Had the rule fired on a legitimate open-gap row it would have been wrong and
would not have shipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.cert_neutrality import (  # noqa: E402
    SOUNDNESS_CLASS_KINDS,
    check_neutrality,
    oracle_bracket_coverage,
    wall_limited_arms,
)

pytestmark = [pytest.mark.unit, pytest.mark.correctness]

# A minimization with a true optimum of 10: a valid run brackets it.
_OPT = {"m": 10.0}


def _row(bound, obj, status="feasible", nodes=7, wall=1.0) -> dict:
    return {
        "status": status,
        "objective": obj,
        "bound": bound,
        "node_count": nodes,
        "wall_time": wall,
    }


def _kinds(new, base, **kw):
    return [v.kind for v in check_neutrality(new, base, oracle=_OPT, **kw)]


def test_an_open_gap_is_not_a_violation():
    """The #1195 case, which this must never re-break: an uncertified incumbent
    ABOVE the optimum is an unclosed gap, not a wrong answer."""
    base = {"m": _row(9.0, 12.0)}
    new = {"m": _row(9.5, 11.0)}
    assert _kinds(new, base, regime="bound_changing") == []


def test_an_incumbent_better_than_the_optimum_is_a_wrong_answer():
    """A point better than the true optimum cannot be feasible — so the run
    accepted an infeasible point, certified or not."""
    base = {"m": _row(9.0, 12.0)}
    new = {"m": _row(8.0, 9.0)}  # incumbent 9.0 < optimum 10.0
    viol = check_neutrality(new, base, oracle=_OPT, regime="bound_changing")
    assert [v.kind for v in viol] == ["oracle_bracket"]
    assert "cannot be feasible" in viol[0].detail
    assert viol[0].kind in SOUNDNESS_CLASS_KINDS


def test_a_dual_bound_past_the_optimum_is_a_wrong_answer():
    """A bound above the optimum would prune the optimum away — a false bound is a
    false certificate waiting for the search to close on it."""
    base = {"m": _row(9.0, 12.0)}
    new = {"m": _row(11.0, 12.0)}  # bound 11.0 > optimum 10.0
    viol = check_neutrality(new, base, oracle=_OPT, regime="bound_changing")
    assert [v.kind for v in viol] == ["oracle_bracket"]
    assert "prune it away" in viol[0].detail


def test_the_rule_needs_no_objective_sense():
    """A maximization brackets the other way round and must read the same.

    The module has no access to the model, so a rule that needed the sense could not
    live here at all — and guessing it would be the confidently-wrong answer.
    """
    base = {"m": _row(12.0, 9.0)}  # bound above incumbent: a maximization
    assert _kinds({"m": _row(11.5, 9.5)}, base, regime="bound_changing") == []
    # optimum 10 outside [9.5, 9.8]: the incumbent side, i.e. an infeasible point.
    assert _kinds({"m": _row(9.8, 9.5)}, base, regime="bound_changing") == ["oracle_bracket"]


def test_certified_rows_are_bracketed_too():
    base = {"m": _row(10.0, 10.0, status="optimal")}
    new = {"m": _row(8.0, 8.0, status="optimal")}
    assert "oracle_bracket" in _kinds(new, base, regime="bound_changing")


def test_tolerance_absorbs_ordinary_jitter():
    base = {"m": _row(10.0, 10.0, status="optimal")}
    new = {"m": _row(10.0 - 1e-9, 10.0 + 1e-9, status="optimal")}
    assert _kinds(new, base, regime="bound_changing") == []


# --------------------------------------------------------------------------- #
# no exclusion may switch an ABSOLUTE check off
# --------------------------------------------------------------------------- #
def test_an_excluded_row_is_still_bracketed():
    """``exclude`` means "not evidence about the flag", never "not evidence about
    itself"."""
    base = {"m": _row(9.0, 12.0)}
    new = {"m": _row(8.0, 9.0)}
    assert _kinds(new, base, regime="bound_changing", exclude={"m"}) == ["oracle_bracket"]


def test_a_wall_limited_row_is_still_bracketed():
    """The #1204 rules decline to COMPARE these rows. That is exactly why this check
    has to survive them: a wall-cut run still reports an incumbent and a bound, and
    an infeasible point does not become acceptable because the clock ran out."""
    budgets = {"m": 60.0}
    # both arms out of clock -> #1187 says no verdict at all...
    base = {"m": _row(9.0, 12.0, status="time_limit", wall=60.0)}
    new = {"m": _row(8.0, 9.0, status="time_limit", wall=60.0)}
    arms = wall_limited_arms(new, base, budgets=budgets)
    assert arms["m"].both
    assert _kinds(new, base, regime="bound_changing", wall_limited=arms) == ["oracle_bracket"]

    # ...and the asymmetric case, which is reported as perf, keeps it as well.
    base2 = {"m": _row(9.0, 12.0, status="optimal", wall=20.0)}
    kinds = _kinds(new, base2, regime="bound_changing", wall_limited=arms)
    assert "oracle_bracket" in kinds


def test_a_missing_row_reports_missing_and_nothing_else():
    assert _kinds({}, {"m": _row(9.0, 12.0)}, regime="bound_changing") == ["missing"]


# --------------------------------------------------------------------------- #
# the check must publish how much it read
# --------------------------------------------------------------------------- #
def test_coverage_counts_what_was_actually_bracketed():
    rows = {"m": _row(9.0, 12.0), "other": _row(1.0, 2.0)}
    checked, skipped = oracle_bracket_coverage(rows, _OPT)
    assert checked == 1
    assert skipped == {"other": "no true optimum in the oracle"}


def test_coverage_names_each_reason_it_could_not_read_a_row():
    rows = {
        "m": _row(None, None),
        "m2": _row(9.0, None),
        "m3": _row(None, 12.0),
        "m4": _row(1e20, 12.0),  # the Rust INF sentinel, not a number
        "m5": _row(float("nan"), 12.0),
    }
    oracle = dict.fromkeys(rows, 10.0)
    checked, skipped = oracle_bracket_coverage(rows, oracle)
    assert checked == 0
    assert set(skipped) == set(rows)
    assert "incumbent or dual bound" in skipped["m"]
    assert "no finite incumbent" in skipped["m2"]
    assert "no finite dual bound" in skipped["m3"]
    # 1e20 is INF in the Rust LP layer, and `isinf` is false for it (CLAUDE.md).
    assert "no finite dual bound" in skipped["m4"]
    assert "no finite dual bound" in skipped["m5"]


def test_an_empty_oracle_reads_nothing_and_says_so():
    """ "0 violations" over 0 comparisons is not a pass. A panel whose oracle file
    went missing would otherwise read exactly like a clean one (CLAUDE.md §6)."""
    rows = {"m": _row(9.0, 12.0)}
    checked, skipped = oracle_bracket_coverage(rows, {})
    assert checked == 0 and skipped == {"m": "no true optimum in the oracle"}
    assert check_neutrality(rows, rows, oracle={}, regime="bound_changing") == []


def test_both_gate_scripts_report_the_coverage():
    """A count nobody prints cannot be read, and this check's failure mode is
    reading nothing quietly."""
    checked = 0
    for rel in ("scripts/check_cert_neutrality.py", "scripts/graduation_gate.py"):
        text = (_BENCH_ROOT / rel).read_text()
        assert "oracle_bracket_coverage" in text, f"{rel} does not report the coverage"
        checked += 1
    assert checked == 2, "the probe stopped reading files (rule 6)"


def test_the_real_reference_panel_brackets_cleanly():
    """The committed reference must satisfy its own guard.

    If this ever fails, either a row in ``cert-baseline.jsonl`` disagrees with
    ``cert-optima.json`` beyond correctness tolerance — which is a finding in its own
    right — or the rule is wrong.
    """
    import json

    repo = _BENCH_ROOT.parent
    baseline = {}
    for line in (repo / "docs" / "dev" / "data" / "cert-baseline.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            baseline[row["instance"]] = row
    oracle = json.loads((repo / "docs" / "dev" / "data" / "cert-optima.json").read_text())
    checked, skipped = oracle_bracket_coverage(baseline, oracle)
    assert checked >= 50, f"only {checked} of {len(baseline)} reference rows bracketable"
    viol = check_neutrality(baseline, baseline, oracle=oracle, regime="bound_changing")
    assert [v for v in viol if v.kind == "oracle_bracket"] == [], (
        f"the committed reference violates its own oracle bracket: {viol}"
    )
    assert set(skipped) <= {"tspn05"}, f"unexpected unbracketable reference rows: {skipped}"
