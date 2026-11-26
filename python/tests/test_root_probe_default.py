"""`DISCOPT_HIGHS_SEPA_ROOT_PROBE` is default ON and `=0` opts out.

Root probing (HiGHS's presolve probing plus the per-round `separateImpliedBounds`)
runs in the Rust MILP root separation. It graduated to default ON on its §5 panel
(`docs/dev/flag-retirement-audit.md`, "Rows added 2026-11-26"). This test holds the
default and the opt-out to the same optimum, and checks that the probe counters fire
on the default arm and stay at zero with `=0`.

Before the graduation the default arm left every root-probe counter at zero.

The solve runs in a subprocess because the flag is read from the environment and the
profile counters are process-global.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "data" / "issue2069" / "clay0203_master.mps.gz"

_PROBE = r"""
import json, sys
import discopt._rust as R
import discopt.modeling as dm
m = dm.from_mps(sys.argv[1])
R.profile_reset_py()
r = m.solve(time_limit=120, deterministic=True, threads=1)
c = {k: int(v) for k, v in R.profile_counters_py().items()}
print("ROW " + json.dumps({"status": r.status, "obj": r.objective,
                           "presolve": c.get("MilpRootProbePresolve", 0),
                           "probed": c.get("MilpRootProbeProbed", 0),
                           "implied": c.get("MilpRootProbeImpliedCuts", 0)}))
"""


def _run(flag: str | None) -> dict:
    # The flag acts only in the Rust MILP engine; `auto` sends a pure MILP to HiGHS.
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_")}
    env.update(DISCOPT_LP_ENGINE="rust", DISCOPT_PROFILE="1", RAYON_NUM_THREADS="1")
    if flag is not None:
        env["DISCOPT_HIGHS_SEPA_ROOT_PROBE"] = flag
    p = subprocess.run(
        [sys.executable, "-c", _PROBE, str(FIXTURE)],
        env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr[-4000:]
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
    assert rows, p.stdout[-2000:] + p.stderr[-2000:]
    return json.loads(rows[-1][4:])


@pytest.mark.smoke
def test_root_probe_is_default_on_and_opts_out():
    on = _run(None)
    off = _run("0")
    assert on["status"] == "optimal" and off["status"] == "optimal", (on, off)
    assert on["obj"] == pytest.approx(off["obj"], rel=1e-6, abs=1e-6)
    assert on["presolve"] > 0 and on["probed"] > 0, on
    assert off["presolve"] == 0 and off["probed"] == 0 and off["implied"] == 0, off
