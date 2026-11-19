"""#2236: a cold primal solve refused on a bounds excursion is decided by the dual.

``DISCOPT_LP_PRIMAL_DUAL_CLEANUP`` (default ON since its §5 panel; ``=0`` opts out)
re-optimizes such a refusal warm with the dual simplex from the refused basis. It keeps
the answer only when it is ``Optimal``, its point passes the primal's audit, AND its
value is certified by weak duality on the same data (``dual_value_certifies_cols``, the
check the #2212 logical-dual rescue uses; added on the PR #2256 review). The flag is
read once per process (a ``OnceLock`` in Rust), so each arm runs in a subprocess.

The #2226 fixture (the ex8_5_3 node LP, 155 rows x 37 columns, HiGHS optimum
-0.06756047745692995) is the LP the issue was filed on. It has free columns: the plain
weak-duality certificate is minus infinity on a free column whose reduced cost does not
vanish to within rounding, so it refuses the fixture. The clean-up then projects the
dual onto ``{a_j'y = c_j}`` for those columns (``project.rs``, the MILP driver's
``DISCOPT_NS_PROJECT`` construction) and evaluates the enclosed Neumaier-Shcherbina
bound, which certifies it. The second positive case is a corpus solve (``nvs08``).
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
    "value_rejected": int(c["PrimalDualCleanupValueRejected"]),
    "projected": int(c["PrimalDualCleanupProjectedAccepts"]),
    "bounds_refusals": int(c["LpAuditBoundsFail"]),
}}))
"""


def _run(flag: str | None) -> dict:
    """Run the route in a fresh process; ``flag=None`` leaves the variable unset."""
    env = dict(os.environ)
    env.update(DISCOPT_LP_ENGINE="rust", DISCOPT_PROFILE="1")
    env.pop("DISCOPT_LP_STD_ENGINE", None)
    env.pop("DISCOPT_LP_PRIMAL_DUAL_CLEANUP", None)
    if flag is not None:
        env["DISCOPT_LP_PRIMAL_DUAL_CLEANUP"] = flag
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


def test_opt_out_still_refuses_the_fixture():
    """The captured defect: with ``=0`` the Rust route returns no bound, and the clean-up
    never runs. If this starts passing a bound, the premise of #2236 changed."""
    out = _run("0")
    assert out["attempts"] == 0
    assert out["bound"] is None


@pytest.mark.parametrize("flag", [None, "1"], ids=["default", "explicit_on"])
def test_projected_certificate_decides_the_free_column_fixture(flag):
    """Fails before #2236 (``=0``: no bound). By default the clean-up's warm dual answer
    is certified through the projected dual -- the plain weak-duality check alone refuses
    this free-column LP -- and ``solve_lp_std`` verifies the bound at HiGHS's optimum."""
    out = _run(flag)
    assert out["attempts"] >= 1 and out["rescues"] >= 1, out
    assert out["projected"] >= 1, out  # the free-column certificate is what decided it
    assert out["bound"] == pytest.approx(NODE_OPT, abs=1e-9)


_NVS08 = os.path.join(HERE, "data", "minlplib_nl", "nvs08.nl")
#: ``docs/dev/data/cert-optima.json``.
_NVS08_OPT = 23.449727347566373

_SOLVE = r"""
import json
import discopt.modeling as dm
from discopt import _rust
_rust.profile_reset_py()
r = dm.from_nl({path!r}).solve(time_limit=60, deterministic=True, max_nodes=200)
c = dict(_rust.profile_counters_py())
print("RESULT " + json.dumps({{
    "status": r.status, "objective": r.objective, "bound": r.bound,
    "certified": bool(r.gap_certified),
    "rescues": int(c["PrimalDualCleanupRescues"]),
    "value_rejected": int(c["PrimalDualCleanupValueRejected"]),
    "bounds_refusals": int(c["LpAuditBoundsFail"]),
}}))
"""


def _solve_nvs08(flag: str | None) -> dict:
    env = dict(os.environ, DISCOPT_PROFILE="1")
    env.pop("DISCOPT_LP_PRIMAL_DUAL_CLEANUP", None)
    if flag is not None:
        env["DISCOPT_LP_PRIMAL_DUAL_CLEANUP"] = flag
    cp = subprocess.run(
        [sys.executable, "-c", _SOLVE.format(path=_NVS08)],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
    assert cp.returncode == 0 and lines, cp.stderr[-2000:]
    out = json.loads(lines[-1][len("RESULT ") :])
    assert out["bounds_refusals"] >= 1, out  # the class the clean-up repairs occurred (§6)
    return out


@pytest.mark.parametrize("flag", [None, "1"], ids=["default", "explicit_on"])
def test_clean_up_rescues_a_certified_lp_on_nvs08(flag):
    """Fails before #2236 (``=0``: no rescue). By default the clean-up turns a bounds
    refusal on nvs08 into an ``Optimal`` whose point and value both certify, and the solve
    still certifies the known optimum."""
    out = _solve_nvs08(flag)
    assert out["rescues"] >= 1, out
    assert out["status"] == "optimal" and out["certified"], out
    assert out["objective"] == pytest.approx(_NVS08_OPT, rel=1e-6)
    assert out["bound"] <= _NVS08_OPT + 1e-6


def test_opt_out_never_rescues_on_nvs08():
    out = _solve_nvs08("0")
    assert out["rescues"] == 0 and out["value_rejected"] == 0, out
    assert out["status"] == "optimal" and out["certified"], out


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
