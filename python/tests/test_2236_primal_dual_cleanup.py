"""#2236: a cold primal solve refused on a bounds excursion is decided by the dual.

The fixture is the #2226 ex8_5_3 node LP (155 rows x 37 columns). On the Rust engine
the primal simplex reaches ``Optimal``, then its final audit finds a basic column
outside its bound and downgrades the verdict to ``numerical``; the equilibrated retry
does the same. HiGHS solves it to ``optimal`` -0.06756047745692995.

``DISCOPT_LP_PRIMAL_DUAL_CLEANUP=1`` re-optimizes such a refusal warm with the dual
simplex from the refused basis and keeps only an audited ``Optimal``. The flag is read
once per process (a ``OnceLock`` in Rust), so each arm runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)

#: HiGHS's certified optimum of the fixture LP (see ``test_2226_verified_lp_engine``).
NODE_OPT = -0.06756047745692995

_PROBE = r"""
import json, os, sys
sys.path.insert(0, {here!r})
from discopt import _rust
from discopt._relax import mccormick_lp as M
from test_2226_verified_lp_engine import _milp
_rust.profile_reset_py()
bound, x = M._verified_free_column_bound(_milp(), 60.0)
c = dict(_rust.profile_counters_py())
print("RESULT " + json.dumps({{
    "bound": bound,
    "engines": list(M._verified_lp_engines()),
    "attempts": int(c["PrimalDualCleanupAttempts"]),
    "rescues": int(c["PrimalDualCleanupRescues"]),
    "bounds_refusals": int(c["LpAuditBoundsFail"]),
}}))
"""


def _run(flag: str) -> dict:
    env = dict(os.environ)
    env.update(
        DISCOPT_LP_ENGINE="rust",
        DISCOPT_LP_PRIMAL_DUAL_CLEANUP=flag,
        DISCOPT_PROFILE="1",
    )
    env.pop("DISCOPT_LP_STD_ENGINE", None)
    cp = subprocess.run(
        [sys.executable, "-c", _PROBE.format(here=HERE)],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
    assert cp.returncode == 0 and lines, cp.stderr[-2000:]
    out = json.loads(lines[-1][len("RESULT ") :])
    # An explicit DISCOPT_LP_ENGINE makes the route defer to it (``(None,)``), and the
    # Rust audit counter proves the Rust primal really ran and refused (CLAUDE.md §6).
    assert "highs" not in out["engines"], out
    assert out["bounds_refusals"] >= 1, out
    return out


def test_flag_off_still_refuses_the_fixture():
    """The captured defect: with the flag OFF the Rust route returns no bound, and the
    clean-up never runs. If this starts passing a bound, the premise of #2236 changed."""
    out = _run("0")
    assert out["attempts"] == 0
    assert out["bound"] is None


def test_flag_on_decides_the_fixture_at_the_highs_optimum():
    """Fails before #2236: the refused basis was discarded. With the flag ON the warm
    dual decides the LP and ``solve_lp_std`` verifies the bound."""
    out = _run("1")
    assert out["attempts"] >= 1 and out["rescues"] >= 1, out
    assert out["bound"] == pytest.approx(NODE_OPT, abs=1e-9)


def test_unrecognized_flag_value_is_refused():
    """A typo must not silently mean OFF (the Rust parse table panics)."""
    env = dict(os.environ, DISCOPT_LP_ENGINE="rust", DISCOPT_LP_PRIMAL_DUAL_CLEANUP="yes")
    cp = subprocess.run(
        [sys.executable, "-c", _PROBE.format(here=HERE)],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    assert cp.returncode != 0
    assert "DISCOPT_LP_PRIMAL_DUAL_CLEANUP" in cp.stderr
