"""#1671: tripwire for the HiGHS presolve cycle on the #1667 unit-commitment hand-off.

HiGHS 1.12-1.15's MIP presolve cycles forever on the #1667 unit-commitment hand-off --
in ``HPresolve::fastPresolveLoop`` -> ``rowPresolve``, a loop that polls neither
``time_limit`` nor the interrupt callbacks. From #1667 the MILP route switched presolve
rule 14 (sparsify) off to avoid it. highspy 1.16.0 is the first release with the
upstream fix (#2221), the floor reached 1.16 in #2332, and #2335 re-measured sparsify
on vs off on 1.16 and turned it back ON for the route (the ``milp_highs`` masters keep
it off as a documented choice; record at ``lp_milp_highs._PRESOLVE_RULE_SPARSIFY``).

So this canary now guards the other direction: the route hands HiGHS sparsify on, so
the installed HiGHS must return from presolve on the hand-off with the route's own
``presolve_rule_off``. It fails if a supported release brings the cycle back, which
would hang ``Model.solve`` past its time limit on this class.

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


def test_route_presolve_returns_on_the_1667_handoff(handoff_mps):
    """The route's presolve configuration (sparsify on since #2335) must not cycle on the
    installed HiGHS. The sparsify-off arm is the positive control: it returned on every
    release measured, so it proves the probe ran and read the model."""
    control = _presolve(handoff_mps, _PARALLEL | _SPARSIFY)
    assert control is not None, (
        f"presolve with sparsify OFF did not return within {KILL_AFTER} s; the control "
        "arm must return, so this probe cannot tell anything about the cycle"
    )
    rows, cols, n_int, status, githash = control
    assert (rows, cols, n_int) == ("2904", "4344", "240"), control
    assert status == "HighsStatus.kOk", control

    from discopt.solvers import lp_milp_highs

    route = int(lp_milp_highs.MILP_PRESOLVE_RULE_OFF)
    assert not route & _SPARSIFY, "#2335: the route runs sparsify; this probe tests that"
    assert route & _PARALLEL  # #1634's rule 13 stays off
    returned = _presolve(handoff_mps, route)
    import highspy

    release = highspy.Highs().version()
    assert returned is not None, (
        f"HiGHS {githash} ({release}) cycles in presolve on the #1667 hand-off with the "
        f"route's presolve_rule_off={route} (sparsify on). The route would hang past its "
        "time limit on this class: switch sparsify off on the route again (#1667) and "
        "report the release upstream (#1671)"
    )
    # The floor is past every release measured to cycle (#2332).
    lo, hi = _CYCLING_WINDOW
    assert tuple(int(p) for p in _SUPPORTED_FLOOR.split(".")) >= hi
