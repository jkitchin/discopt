"""#2224: ``DISCOPT_MILP_PREINC_SCORE`` finds the first incumbent on a Lights-Out puzzle.

The fixture ``data/issue_2224/enlight8_perm3.mps.gz`` is MIPLIB 2017 ``enlight8`` (8x8
Lights-Out parity rows ``x_c + sum_{4-nbrs} x - 2 y_c = 1``, ``x`` binary, ``y`` integer)
with its columns and rows permuted by ``numpy.random.default_rng(3).permutation`` (the
performance-variability probe of the #2224 entry experiment). The optimum is 27 (MIPLIB
2017 ``.solu``; HiGHS 1.15.1 proves it on this copy).

Measured on the HiGHS-design search (default route), ``max_nodes=5000``,
``deterministic=True``: with HiGHS's ``1e-4`` cutoff/inference weights
(``DISCOPT_MILP_PREINC_SCORE=0``) the search has **no incumbent** at the node limit (bound
26); with the flag (``=1``, weight 1 while there is no incumbent; the default) it
certifies 27 in 1,352 nodes. The test runs ``=1``, ``=0`` and the default, so it fails if
the mechanism stops firing, stops being the default, *or* stops mattering. Work budget,
not a wall clock: the ``time_limit`` is a backstop well above the measured ~5 s per arm.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.correctness]

FIXTURE = Path(__file__).parent / "data" / "issue_2224" / "enlight8_perm3.mps.gz"
OPTIMUM = 27.0

_PROBE = r"""
import json, sys
import discopt._rust as R
import discopt.modeling as dm
m = dm.from_mps(sys.argv[1])
R.profile_reset_py()
r = m.solve(time_limit=600, gap_tolerance=1e-4, max_nodes=5000, deterministic=True)
c = {k: int(v) for k, v in R.profile_counters_py().items()}
print("ROW " + json.dumps({"status": r.status, "cert": bool(r.gap_certified),
                           "obj": r.objective, "bound": r.bound, "nodes": int(r.node_count),
                           "preinc": c.get("PreIncScoreBranchings", 0),
                           "runs": c.get("HighsMipSearchRuns", 0)}))
"""


def _run(flag: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_")}
    env["DISCOPT_PROFILE"] = "1"
    if flag is not None:
        env["DISCOPT_MILP_PREINC_SCORE"] = flag
    p = subprocess.run(
        [sys.executable, "-c", _PROBE, str(FIXTURE)],
        env=env, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
    assert p.returncode == 0 and len(rows) == 1, p.stderr[-3000:]
    return json.loads(rows[0][4:])


def test_preinc_score_certifies_the_permuted_lights_out_puzzle():
    on = _run("1")
    off = _run("0")
    default = _run(None)
    # The HiGHS-design search ran in both arms (the flag acts only there).
    assert on["runs"] > 0 and off["runs"] > 0, (on, off)
    # Flag on: the raised weight fired and the search certified the optimum.
    assert on["preinc"] > 0, on
    assert on["cert"], on
    assert on["obj"] == pytest.approx(OPTIMUM, abs=1e-6), on
    assert on["bound"] <= OPTIMUM + 1e-6, on
    # Flag off: HiGHS's weights, no raised-weight branching, and the search is worse
    # off than ON -- measured: no incumbent at the node limit (the case #2224 is about).
    # Asserted loosely (no incumbent, or more nodes than ON) so that an unrelated primal
    # improvement does not fail the test while the flag still matters.
    assert off["preinc"] == 0, off
    assert off["obj"] is None or (off["nodes"] > on["nodes"]), (off, on)
    assert off["bound"] <= OPTIMUM + 1e-6, off
    # The default is the flag ON (graduated): same search as the explicit `=1` arm.
    assert default["preinc"] == on["preinc"] and default["nodes"] == on["nodes"], (default, on)
    assert default["cert"] and default["obj"] == pytest.approx(OPTIMUM, abs=1e-6), default


def test_preinc_score_refuses_garbage():
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_")}
    env["DISCOPT_MILP_PREINC_SCORE"] = "maybe"
    p = subprocess.run(
        [sys.executable, "-c", _PROBE, str(FIXTURE)],
        env=env, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert "DISCOPT_MILP_PREINC_SCORE" in p.stderr and "not recognized" in p.stderr, (
        p.returncode,
        p.stderr[-2000:],
    )
