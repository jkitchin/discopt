"""``DISCOPT_HIGHS_SEARCH_SEPA``: the HiGHS-design search's separation is deterministic.

The search runs ``separationRound`` on the root's separation subsystem
(``RootSeparation``), whose separators carry state between calls: the tableau
separator's call/try counts gate its effort, and the path separator draws from a
seeded random generator. As in HiGHS (``HighsMipWorker`` owns a fresh
``HighsSeparation``), that state is reset when the search starts; the Rust test
``highs_search_starts_with_fresh_separators`` pins the reset itself. This test pins
the property the reset and the open-node / reduced-cost variable-bound tie-breaks must
preserve: two solves of the same model in two fresh processes (different hash seeds)
take the same tree.

bell5 (MIPLIB 3, ``data/milp_search_sepa/bell5.mps.gz``) is the instance the port was
measured on: with presolve off on the Rust engine the search separates on it (the test
asserts the rounds ran, so it cannot pass with the mechanism off), and a node budget
keeps it cheap.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "milp_search_sepa" / "bell5.mps.gz"

_CHILD = r"""
import json, sys
import discopt._rust as r
import discopt.modeling as dm
m = dm.from_mps(sys.argv[1])
res = m.solve(gap_tolerance=1e-6, deterministic=True, max_nodes=400, time_limit=600)
c = r.profile_counters_py()
print("RESULT " + json.dumps({
    "status": str(res.status),
    "objective": res.objective,
    "bound": res.bound,
    "nodes": res.node_count,
    "rounds": int(c.get("HighsMipSearchSepaRounds", 0)),
    "rows": int(c.get("HighsMipSearchSepaRows", 0)),
}))
"""


def _solve(hash_seed: str) -> dict:
    env = dict(os.environ)
    env.update(
        DISCOPT_HIGHS_SEARCH_SEPA="1",
        DISCOPT_MILP_PRESOLVE="0",
        DISCOPT_LP_ENGINE="rust",
        DISCOPT_PROFILE="1",
        RAYON_NUM_THREADS="1",
        PYTHONHASHSEED=hash_seed,
    )
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, str(DATA)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, p.stdout[-2000:]
    return json.loads(lines[0][len("RESULT ") :])


@pytest.mark.smoke
def test_search_sepa_is_deterministic_across_processes():
    a = _solve("1")
    b = _solve("12345")
    # The probe fired: the search separated and added rows, so the comparison below
    # is over a tree that the separation state shaped.
    assert a["rounds"] > 0 and a["rows"] > 0, a
    assert a == b
