"""#1671: tripwire for the HiGHS presolve cycle behind the #1667 workaround.

The HiGHS MILP route switches off presolve rule 14 (sparsify) because HiGHS 1.15.1's
MIP presolve cycles forever on the #1667 unit-commitment hand-off -- in
``HPresolve::fastPresolveLoop`` -> ``rowPresolve``, a loop that polls neither
``time_limit`` nor the interrupt callbacks. That bit is a workaround for an upstream
bug, and #1671 says to re-measure and either drop it or keep it as a documented
opt-out once HiGHS fixes the bug. This test makes that step happen: it fails when the
installed HiGHS stops cycling.

``data/highs_presolve_cycle_1671.mps.gz`` is the route's hand-off itself, written by
``Highs.writeModel`` at the ``h.run()`` call in ``_solve_milp_std``: 2904 rows, 4344
columns, 240 binaries. It is a fixed HiGHS input, so later changes to discopt's standard
form do not change what it tests.

Measured 2026-10-06, ``Highs.presolve()`` on this file, ``time_limit=5``:

========================  =====================  =====================
``presolve_rule_off``     highspy 1.15.1         HiGHS ``latest``
                          (04024d7)              (8a5f9c1d8, unreleased)
========================  =====================  =====================
0 (defaults)              killed at 15 s         returns, 0.11 s
8192 (main's route)       killed at 15 s         returns, 0.09 s
16384 (sparsify off)      returns, 0.06 s        returns, 0.06 s
24576 (#1667's route)     returns, 0.05 s        returns, 0.06 s
========================  =====================  =====================

A full ``run()`` behaves the same way: on 1.15.1 it is still running long after its
30 s ``time_limit``, and with sparsify off it reaches ``Optimal 450623.7692`` in about
5 s. On ``latest``, every arm reaches that optimum in about 3 s, and main's route
(sparsify on) certifies ``optimal`` 450623.7692 in 11 s.

The upstream change that stops the cycle was found by a first-parent ``git bisect`` of
the HiGHS ``latest`` branch, 04024d7..8a5f9c1d8, running the ``highs`` CLI on this file
with ``--time_limit 20`` and killing it at 60 s. The first commit that is ``Optimal`` is
6c6282ba3, the merge of ERGO-Code/HiGHS PR 2962 ("Remove continuous singletons from
double-sided rows", 2026-08-25); its first parent f7b87ae01 is still killed. That PR
adds a presolve reduction and does not touch the loop's polling. It may remove the
rows the cycle needs on this model without fixing the loop itself. So the re-measure
this test asks for has to include the sparsify panel, and not only this file.

The probe is ``presolve()`` alone, which finishes in under 0.2 s whenever it
terminates. Each arm therefore runs in a subprocess that is killed after
``KILL_AFTER`` seconds, 50x that. The sparsify-off arm is the positive control: it
must return, which proves the probe ran and read the model.
"""

from __future__ import annotations

import gzip
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HANDOFF = Path(__file__).parent / "data" / "highs_presolve_cycle_1671.mps.gz"
#: Seconds before an arm is declared cycling. Presolve returns in < 0.2 s when it
#: returns at all (table above); interpreter start and ``import highspy`` add < 1 s.
KILL_AFTER = 10.0
_PARALLEL = 1 << 13  # kPresolveRuleParallelRowsAndCols, off on the route since #1634
_SPARSIFY = 1 << 14  # kPresolveRuleSparsify, the #1667 workaround

PROBE = textwrap.dedent(
    """\
    import sys, highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 5.0)
    h.setOptionValue("presolve_rule_off", int(sys.argv[2]))
    assert h.readModel(sys.argv[1]) == highspy.HighsStatus.kOk
    lp = h.getLp()
    n_int = sum(1 for t in lp.integrality_ if t != highspy.HighsVarType.kContinuous)
    st = h.presolve()
    print("RETURNED", lp.num_row_, lp.num_col_, n_int, st, h.githash(), flush=True)
    """
)


@pytest.fixture(scope="module")
def handoff_mps(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("issue1671") / "handoff.mps"
    out.write_bytes(gzip.decompress(HANDOFF.read_bytes()))
    return str(out)


def _presolve(path: str, rule_off: int) -> list[str] | None:
    """The probe's ``RETURNED`` fields, or ``None`` if it was still running at ``KILL_AFTER``."""
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", PROBE, path, str(rule_off)],
            capture_output=True,
            text=True,
            timeout=KILL_AFTER,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        return None
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RETURNED")]
    assert line, proc.stdout
    return line[0].split()[1:]


def test_highs_presolve_still_cycles_on_the_1667_handoff(handoff_mps):
    control = _presolve(handoff_mps, _PARALLEL | _SPARSIFY)
    assert control is not None, (
        f"presolve with sparsify OFF did not return within {KILL_AFTER} s; the control "
        "arm must return, so this probe cannot tell anything about the cycle"
    )
    rows, cols, n_int, status, githash = control
    assert (rows, cols, n_int) == ("2904", "4344", "240"), control
    assert status == "HighsStatus.kOk", control

    cycling = _presolve(handoff_mps, _PARALLEL)
    if cycling is not None:
        pytest.fail(
            f"HiGHS {githash} no longer cycles in presolve on the #1667 hand-off with "
            f"sparsify ON (returned {cycling}). The installed HiGHS no longer hits the "
            "upstream bug behind #1671 here (bisected to ERGO-Code/HiGHS PR 2962). "
            "Re-measure the sparsify panel from the #1667 PR on this HiGHS, then "
            "either drop _PRESOLVE_RULE_SPARSIFY from "
            "lp_milp_highs.MILP_PRESOLVE_RULE_OFF or keep it as a documented opt-out, "
            "and update or retire this test."
        )
