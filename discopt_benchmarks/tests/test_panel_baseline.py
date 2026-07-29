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
    _annotate,
    _obj_match,
    _resolve_subset,
    corpus_instances,
)

# A tiny instance that certifies in well under a second on the default path, so the
# end-to-end arms cost ~1 s each. Its identity is not load-bearing: any instance
# that reaches `optimal` deterministically would do.
_FAST_INSTANCE = "alan"
_BUDGET = "30"


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
    row = {"instance": "x", "status": "optimal", "gap_certified": True, "wall": 1.0,
           "root_bound": 9.0}
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
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good)])
    assert gen.returncode == 0, f"baseline generation failed:\n{gen.stdout[-3000:]}\n{gen.stderr}"

    base = json.loads(good.read_text())
    assert len(base["rows"]) == 1
    row = base["rows"][0]
    assert row["comparable"], (
        f"{_FAST_INSTANCE} did not certify deterministically "
        f"({row.get('comparable_reason')}); the perturbation arm would prove nothing"
    )

    # ---- arm 1: clean re-check passes, with comparisons actually executed ----
    clean = _run(["--check", str(good)])
    assert clean.returncode == 0, f"clean --check failed:\n{clean.stdout[-3000:]}"
    assert _comparison_count(clean.stdout) > 0
    assert "PASS: no node-count" in clean.stdout

    # ---- arm 2: one node more in the baseline => the check must fail ----------
    bad = tmp_path / "baseline_perturbed.json"
    base["rows"][0]["node_count"] = int(row["node_count"]) + 1
    bad.write_text(json.dumps(base))

    perturbed = _run(["--check", str(bad)])
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


@pytest.mark.smoke
def test_check_refuses_a_baseline_with_no_comparable_rows(tmp_path: Path):
    """Zero executed comparisons is a FAILURE, not a pass (CLAUDE.md §6).

    Constructed by taking a real baseline and marking its only row
    non-comparable, which is exactly what an all-timeout panel would look like.
    """
    good = tmp_path / "baseline.json"
    gen = _run(["--budget", _BUDGET, "--subset", _FAST_INSTANCE, "--out", str(good)])
    assert gen.returncode == 0, gen.stdout[-3000:]

    base = json.loads(good.read_text())
    base["rows"][0]["comparable"] = False
    base["rows"][0]["comparable_reason"] = "synthetic: all rows non-comparable"
    empty = tmp_path / "baseline_nocmp.json"
    empty.write_text(json.dumps(base))

    res = _run(["--check", str(empty)])
    assert res.returncode != 0, f"a zero-comparison check reported success:\n{res.stdout[-3000:]}"
    assert _comparison_count(res.stdout) == 0
    assert "ZERO comparisons executed" in res.stdout
