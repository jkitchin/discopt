"""The Phase 0 baseline checker must prove itself, not assert itself.

``panel_baseline.py --check`` is the gate every Regime-N card in the consolidation
plan will lean on. A drift checker that silently compares nothing is strictly worse
than no checker: it prints "no drift" and is believed (CLAUDE.md §6). So the
headline test here does not check that ``--check`` passes on a good baseline — it
checks that ``--check`` **fails on a baseline that was deliberately corrupted**, and
that the number of comparisons it reports having executed is greater than zero.

The pair matters: the PASS arm establishes that a clean re-run is clean on this
machine, so the FAIL arm's failure is attributable to the injected one-node
perturbation and not to ambient flakiness.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SCRIPT = _BENCH_ROOT / "scripts" / "panel_baseline.py"

if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from scripts.panel_baseline import (  # noqa: E402
    _V_CONFIRMED,
    _V_NONDET,
    _V_TRANSIENT,
    _adjudicate,
    _annotate,
    _compare_hard,
    _obj_match,
    _resolve_subset,
    _signature,
    corpus_instances,
)

# A tiny instance that certifies in well under a second on the default path, so the
# end-to-end arms cost ~1 s each. Its identity is not load-bearing: any instance
# that reaches `optimal` deterministically would do.
_FAST_INSTANCE = "alan"
_BUDGET = "30"

# The end-to-end arms run under pytest, which is itself load. The load gate
# (item 15) is exercised by its own test; everywhere else it is waived explicitly
# so a busy CI box cannot turn these assertions into a REFUSED exit.
_ALLOW_LOAD = "--allow-load"


class _FakeOracle:
    """Minimal stand-in for ``reference_optima.Oracle``."""

    def __init__(self, value: float, source: str = "test", proven: bool = True):
        self.value = value
        self.source = source
        self.proven = proven


def _run(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )


def _comparison_count(stdout: str) -> int:
    """Pull the executed-comparison count out of the checker's own report."""
    m = re.search(r"comparisons executed:\s*(\d+)", stdout)
    assert m is not None, f"checker printed no comparison count:\n{stdout[-3000:]}"
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Pure-logic arms (no solve)                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_corpus_is_not_empty():
    """A panel over zero instances would make every later check vacuous."""
    assert len(corpus_instances()) >= 61


@pytest.mark.unit
def test_annotate_marks_certified_fast_optimal_comparable():
    row = {
        "instance": "x",
        "status": "optimal",
        "gap_certified": True,
        "wall": 5.0,
        "root_bound": 9.0,
    }
    out = _annotate(row, budget=60.0, oracle=lambda _n: _FakeOracle(10.0))
    assert out["comparable"] is True
    assert out["comparable_reason"] is None
    # root_gap_vs_reference = |ref - root_bound| / max(1, |ref|) = 1/10.
    assert out["root_gap_vs_reference"] == pytest.approx(0.1)
    assert out["reference_optimum"] == 10.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "needle"),
    [
        ({"status": "time_limit", "gap_certified": True, "wall": 5.0}, "budget-dependent"),
        ({"status": "feasible", "gap_certified": True, "wall": 5.0}, "budget-dependent"),
        ({"status": "errored", "gap_certified": False}, "status=errored"),
        ({"status": "optimal", "gap_certified": False, "wall": 1.0}, "gap_certified=False"),
        ({"status": "optimal", "gap_certified": True, "wall": 55.0}, "of the 60s budget"),
    ],
)
def test_annotate_excludes_non_reproducible_rows_with_a_stated_reason(row, needle):
    """Every excluded row says WHY. Silent narrowing is the failure this guards."""
    row = {"instance": "x", **row}
    out = _annotate(row, budget=60.0, oracle=lambda _n: None)
    assert out["comparable"] is False
    assert needle in out["comparable_reason"]


@pytest.mark.unit
def test_root_gap_vs_reference_is_none_without_an_oracle():
    """Coverage is never faked: no oracle => no reference-relative root gap."""
    row = {
        "instance": "x",
        "status": "optimal",
        "gap_certified": True,
        "wall": 1.0,
        "root_bound": 9.0,
    }
    out = _annotate(row, budget=60.0, oracle=lambda _n: None)
    assert out["root_gap_vs_reference"] is None
    assert out["reference_optimum"] is None


@pytest.mark.unit
def test_subset_refuses_unknown_names():
    """A typo'd --subset must refuse loudly, not shrink the panel to nothing."""
    with pytest.raises(SystemExit):
        _resolve_subset(["a", "b"], "a,does_not_exist")
    assert _resolve_subset(["a", "b", "c"], "2") == ["a", "b"]


@pytest.mark.unit
def test_obj_match_tolerance():
    assert _obj_match(1.0, 1.0 + 1e-12)
    assert not _obj_match(1.0, 1.0 + 1e-5)
    assert not _obj_match(1.0, None)


# --------------------------------------------------------------------------- #
# The headline arm: --check must DETECT a deliberate perturbation.             #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_check_detects_a_perturbed_node_count(tmp_path: Path):
    """Take a real one-instance baseline, bump one node_count by 1, re-check.

    Asserts three things, all of which have failed silently in this repo before:
      1. the clean re-check PASSES (so arm 2's failure is attributable);
      2. the perturbed re-check EXITS NON-ZERO;
      3. the checker reports having executed a POSITIVE number of comparisons in
         both arms — a checker that compares nothing cannot fail, and its "no
         drift" is a no-op that reads as a pass.
    """
    good = tmp_path / "baseline.json"
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good), _ALLOW_LOAD])
    assert gen.returncode == 0, f"baseline generation failed:\n{gen.stdout[-3000:]}\n{gen.stderr}"

    base = json.loads(good.read_text())
    assert len(base["rows"]) == 1
    row = base["rows"][0]
    assert row["comparable"], (
        f"{_FAST_INSTANCE} did not certify deterministically "
        f"({row.get('comparable_reason')}); the perturbation arm would prove nothing"
    )

    # ---- arm 1: clean re-check passes, with comparisons actually executed ----
    clean = _run(["--check", str(good), _ALLOW_LOAD])
    assert clean.returncode == 0, f"clean --check failed:\n{clean.stdout[-3000:]}"
    assert _comparison_count(clean.stdout) > 0
    assert "PASS: no node-count" in clean.stdout

    # ---- arm 2: one node more in the baseline => the check must fail ----------
    bad = tmp_path / "baseline_perturbed.json"
    base["rows"][0]["node_count"] = int(row["node_count"]) + 1
    bad.write_text(json.dumps(base))

    perturbed = _run(["--check", str(bad), _ALLOW_LOAD])
    assert perturbed.returncode != 0, (
        "--check ACCEPTED a baseline whose node_count was perturbed by 1. The Regime-N "
        f"gate is a no-op.\n{perturbed.stdout[-3000:]}"
    )
    n_cmp = _comparison_count(perturbed.stdout)
    assert n_cmp > 0, "the check failed having executed ZERO comparisons — it proved nothing"
    assert "NODE COUNT drift" in perturbed.stdout, (
        f"--check failed for some reason OTHER than the injected node-count drift:\n"
        f"{perturbed.stdout[-3000:]}"
    )
    # Item 15: the hardening must not be an escape hatch. A deterministic
    # one-node perturbation has to survive replicate-and-agree as CONFIRMED
    # drift, not be excused as environmental noise.
    assert _V_CONFIRMED in perturbed.stdout, (
        "the injected drift was NOT adjudicated as CONFIRMED — the replicate rule is "
        f"masking real drift, which plan §0.4 forbids:\n{perturbed.stdout[-4000:]}"
    )
    assert f"{_V_TRANSIENT} (" not in perturbed.stdout, (
        f"a deterministic perturbation was excused as TRANSIENT:\n{perturbed.stdout[-4000:]}"
    )


@pytest.mark.smoke
def test_check_refuses_a_baseline_with_no_comparable_rows(tmp_path: Path):
    """Zero executed comparisons is a FAILURE, not a pass (CLAUDE.md §6).

    Constructed by taking a real baseline and marking its only row
    non-comparable, which is exactly what an all-timeout panel would look like.
    """
    good = tmp_path / "baseline.json"
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good), _ALLOW_LOAD])
    assert gen.returncode == 0, gen.stdout[-3000:]

    base = json.loads(good.read_text())
    base["rows"][0]["comparable"] = False
    base["rows"][0]["comparable_reason"] = "synthetic: all rows non-comparable"
    empty = tmp_path / "baseline_nocmp.json"
    empty.write_text(json.dumps(base))

    res = _run(["--check", str(empty), _ALLOW_LOAD])
    assert res.returncode != 0, f"a zero-comparison check reported success:\n{res.stdout[-3000:]}"
    assert _comparison_count(res.stdout) == 0
    assert "ZERO comparisons executed" in res.stdout


# --------------------------------------------------------------------------- #
# Open-ledger item 15: replicate-and-agree adjudication.                       #
#                                                                             #
# The panel's own defect was that a real drift and a container flake produced  #
# identical output. These arms pin the three verdicts as PURE logic — no solve #
# — so the classification can be reasoned about without a 36-minute panel, and #
# so the "does not weaken the gate" property is asserted rather than argued.   #
# --------------------------------------------------------------------------- #
def _row(
    node_count: int, objective: float = 1.0, status: str = "optimal", certified: bool = True
) -> dict:
    return {
        "instance": "x",
        "status": status,
        "node_count": node_count,
        "objective": objective,
        "gap_certified": certified,
        "wall": 1.0,
        "comparable": True,
    }


@pytest.mark.unit
def test_compare_hard_executes_three_comparisons_per_row():
    """The executed count is a fact returned by the comparator, not an inference."""
    viol, n = _compare_hard("x", _row(10), _row(10))
    assert viol == []
    assert n == 3, "a comparable row must contribute status + node_count + objective"


@pytest.mark.unit
def test_compare_hard_catches_each_gated_quantity():
    assert any("NODE COUNT" in v for v in _compare_hard("x", _row(10), _row(11))[0])
    assert any("CERTIFIED OBJECTIVE" in v for v in _compare_hard("x", _row(10), _row(10, 2.0))[0])
    assert any(
        "STATUS drift" in v for v in _compare_hard("x", _row(10), _row(10, status="time_limit"))[0]
    )
    lost = _compare_hard("x", _row(10), _row(10, certified=False))[0]
    assert any("CERTIFICATION LOST" in v for v in lost)


@pytest.mark.unit
def test_adjudicate_confirms_deterministic_drift():
    """THE property that makes the hardening legal (plan §0.4).

    A real bound-neutrality violation is deterministic: the changed code runs on
    every replicate. Unanimous replicates that disagree with the baseline must
    FAIL, never be excused.
    """
    adj = _adjudicate("x", _row(10), [_row(11), _row(11), _row(11)])
    assert adj["verdict"] == _V_CONFIRMED
    assert adj["comparisons"] == 9
    assert any("NODE COUNT" in v for v in adj["violations"])


@pytest.mark.unit
def test_adjudicate_calls_a_reproducing_row_transient():
    adj = _adjudicate("x", _row(10), [_row(10), _row(10), _row(10)])
    assert adj["verdict"] == _V_TRANSIENT
    assert adj["comparisons"] == 9
    assert adj["violations"] == []


@pytest.mark.unit
def test_adjudicate_flags_self_disagreement_as_nondeterministic():
    """A row that will not reproduce ITSELF must fail under its own label.

    Averaging it into a pass is precisely the paper-over this hardening exists to
    avoid: it would hide solver-level nondeterminism behind a replicate rule.
    """
    adj = _adjudicate("x", _row(10), [_row(10), _row(91), _row(10)])
    assert adj["verdict"] == _V_NONDET
    assert "does not reproduce ITSELF" in adj["reason"]


@pytest.mark.unit
def test_adjudicate_refuses_to_pass_on_zero_replicates():
    """No replicates must never adjudicate a flagged row into a pass."""
    adj = _adjudicate("x", _row(10), [])
    assert adj["verdict"] == _V_CONFIRMED
    assert adj["comparisons"] == 0


@pytest.mark.unit
def test_signature_uses_exactly_the_gated_quantities():
    """Self-agreement is judged on what the gate tests, and nothing else."""
    assert _signature(_row(10)) == _signature(_row(10))
    assert _signature(_row(10)) != _signature(_row(11))
    assert _signature(_row(10)) != _signature(_row(10, status="feasible"))
    assert _signature(_row(10)) != _signature(_row(10, certified=False))
    # Two rows that both MATCH the baseline cannot be called disagreeing.
    assert _signature(_row(10, 1.0)) == _signature(_row(10, 1.0 + 1e-12))
    assert _signature(_row(10, 1.0)) != _signature(_row(10, 2.0))
    # wall time is not gated, so it must not make replicates look inconsistent
    a, b = _row(10), _row(10)
    b["wall"] = 44.0
    assert _signature(a) == _signature(b)


@pytest.mark.smoke
def test_check_refuses_to_run_above_the_load_gate(tmp_path: Path):
    """A gate run under contention is not a gate (CLAUDE.md §9).

    A negative threshold is unsatisfiable on any machine (a load average is never
    below zero), so this asserts the refusal path exists and exits non-zero — a
    refusal can never launder a FAIL into a PASS.
    """
    good = tmp_path / "baseline.json"
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good), _ALLOW_LOAD])
    assert gen.returncode == 0, gen.stdout[-3000:]

    res = _run(["--check", str(good), "--max-load=-1"])
    assert res.returncode == 4, f"expected a load refusal, got {res.returncode}:\n{res.stdout}"
    assert "REFUSED" in res.stdout
    assert "PASS" not in res.stdout


@pytest.mark.smoke
def test_replicates_zero_restores_the_single_shot_gate_and_says_so(tmp_path: Path):
    """The escape hatch must be loud, and must still fail on real drift."""
    good = tmp_path / "baseline.json"
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good), _ALLOW_LOAD])
    assert gen.returncode == 0, gen.stdout[-3000:]
    base = json.loads(good.read_text())
    base["rows"][0]["node_count"] = int(base["rows"][0]["node_count"]) + 1
    bad = tmp_path / "perturbed.json"
    bad.write_text(json.dumps(base))

    res = _run(["--check", str(bad), "--replicates", "0", _ALLOW_LOAD])
    assert res.returncode != 0
    assert "--replicates 0" in res.stdout
    assert "NODE COUNT drift" in res.stdout
    assert _comparison_count(res.stdout) > 0
