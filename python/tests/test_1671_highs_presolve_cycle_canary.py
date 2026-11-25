"""#1671: tripwire for the HiGHS presolve cycle behind the #1667 workaround.

The HiGHS MILP route switches off presolve rule 14 (sparsify) because HiGHS 1.15.1's
MIP presolve cycles forever on the #1667 unit-commitment hand-off -- in
``HPresolve::fastPresolveLoop`` -> ``rowPresolve``, a loop that polls neither
``time_limit`` nor the interrupt callbacks. That bit is a workaround for an upstream
bug, and #1671 says to re-measure and either drop it or keep it as a documented
opt-out once HiGHS fixes the bug. This test makes that step happen: it fails when the
installed HiGHS stops cycling without being a release known to carry the fix, and on
every version it checks that the route keeps sparsify off while the supported range still
admits a cycling release. highspy 1.16.0 is the first release with the fix (#2221). The
floor reached 1.16 in #2332; the bit stays as a documented workaround until the sparsify
re-measure of #2335 decides it (the final assertion below pins that state).

The cycle is a regression window. Measured 2026-11-18 (#2221) with this file's probe on
each release, sparsify on: 1.11.0 returns; 1.12.0, 1.13.0, 1.13.1, 1.14.0, 1.15.0 and
1.15.1 cycle; 1.16.0 returns. The defaults arm (``presolve_rule_off=0``) behaves the same.

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


def _pyproject_floor() -> str:
    """The highspy floor of ``pyproject.toml``'s core dependencies (``highspy>=X``)."""
    import re
    import tomllib

    root = Path(__file__).resolve().parents[2]
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    (spec,) = [d for d in deps if d.replace(" ", "").startswith("highspy")]
    m = re.match(r"highspy\s*>=\s*([0-9.]+)", spec)
    assert m, spec
    return m.group(1)


#: The oldest highspy ``pyproject.toml`` accepts (#2332: 1.16, past every cycling release).
_SUPPORTED_FLOOR = _pyproject_floor()
#: Releases measured to cycle (module docstring): ``[first, fixed)``.
_CYCLING_WINDOW = ((1, 12), (1, 16))

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
    """Sparsify stays off while any supported highspy cycles; only releases outside the
    measured 1.12-1.15 window may return."""
    control = _presolve(handoff_mps, _PARALLEL | _SPARSIFY)
    assert control is not None, (
        f"presolve with sparsify OFF did not return within {KILL_AFTER} s; the control "
        "arm must return, so this probe cannot tell anything about the cycle"
    )
    rows, cols, n_int, status, githash = control
    assert (rows, cols, n_int) == ("2904", "4344", "240"), control
    assert status == "HighsStatus.kOk", control

    cycling = _presolve(handoff_mps, _PARALLEL)
    from discopt.solvers import lp_milp_highs

    # The route keeps sparsify off until #2335's re-measure decides it. Dropping the bit
    # is a measured change (that panel), not a side effect of another one.
    assert lp_milp_highs.MILP_PRESOLVE_RULE_OFF & _SPARSIFY, (
        f"HiGHS {githash}: the MILP route no longer switches sparsify off; that needs the "
        "#2335 sparsify panel, and this canary re-scoped or deleted with its measurement"
    )
    if cycling is None:
        return
    # #2221: highspy 1.16.0 (f3f113d) ships ERGO-Code/HiGHS PR 2962 and no longer cycles.
    # Measured 2026-11-18, full run() on this file: sparsify on reaches Optimal 450623.7692
    # in 2.2 s, sparsify off the same optimum in 1.5 s. 1.11.0 predates the regression. A
    # release inside the measured window returning means this probe stopped testing.
    import highspy

    release = highspy.Highs().version()
    minor = tuple(int(p) for p in release.split(".")[:2])
    lo, hi = _CYCLING_WINDOW
    assert not (lo <= minor < hi), (
        f"HiGHS {githash} ({release}) returned from presolve with sparsify ON "
        f"({cycling}), but every release from {lo} up to {hi} was measured to cycle"
    )
    # The floor is past every cycling release (#2332), so the bit is a documented
    # workaround, not a cycle guard. Its docstring must say so and name the re-measure
    # that would retire it -- otherwise it reads as a guard that is still needed.
    assert tuple(int(p) for p in _SUPPORTED_FLOOR.split(".")) >= hi
    doc = Path(lp_milp_highs.__file__).read_text()
    assert "no supported release cycles" in doc and "#2335" in doc, (
        "lp_milp_highs: the sparsify bit's comment must record that the floor is past the "
        "cycling releases and point to the #2335 re-measure"
    )
