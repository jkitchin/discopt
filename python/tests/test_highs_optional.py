"""Which default paths still reach HiGHS (highs-parity P6.2)?

Mirrors ``test_1873_jax_optional.py``. Each check runs in a fresh interpreter whose import
system refuses ``highspy`` and *records the stack of every attempt*, so a path that
catches the ``ImportError`` and quietly degrades is seen as well as one that crashes.

``highspy`` is still a core dependency: decisions D2 (the pure-MILP route) and D3 (the
convex-route OA master) were reopened, not closed (``docs/dev/highs-parity-plan.md``).
So this guard does not demand zero attempts yet. It attributes every attempt to the
discopt function that made it and pins the set of **remaining roles**
(``KNOWN_ROLES``, each tied to a row of ``docs/dev/highs-optional-inventory.md``):

* an attempt from ``highs_available()`` is an availability probe, not a reach;
* an attempt from a known role is reported, never silently allowed to grow;
* an attempt from anywhere else FAILS: a new HiGHS reach on a default path.

When D2/D3 are bridged, the corresponding roles leave ``KNOWN_ROLES`` and this becomes
the zero-attempt guard the plan's P6.2 asks for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

#: function (innermost discopt frame below the blocker) -> inventory row it belongs to.
KNOWN_ROLES = {
    "require_highspy": "the caller's role is the next discopt frame (see ROLE_CALLERS)",
}
#: The discopt functions allowed to call ``require_highspy`` on a default path today,
#: with the inventory row (and the decision) that keeps each one there.
ROLE_CALLERS = {
    "_solve_milp_highs": "#1 pure-MILP route (D2)",
    "solve_milp_std": "#1 pure-MILP route (D2)",
    "solve_lp_std": "#3 solve_lp_std seam default",
    "solve": "#3 solve_lp_std seam default (lp_std_engine.HighsEngine.solve)",
    "_convex_route_oa_master": "#4 convex-route OA master (D3)",
    "_require_highspy": "#4 convex-route OA master (D3), milp_highs",
    # The frame is `_lp_dual` (the key used to read `_highs_lp_dual`, a name the function
    # no longer has, so the corpus guard reported this documented reach as new).
    "_lp_dual": "#8 convex node bound (HiGHS first, then the Rust simplex; P1.3 kill)",
}

_BLOCK_HIGHS = r"""
import sys, traceback
ATTEMPTS = []
CURRENT = [None]
class _NoHighs:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] == "highspy":
            ATTEMPTS.append((CURRENT[0], name, traceback.extract_stack(limit=40)))
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoHighs())

def _roles():
    out = []
    for inst, name, stack in ATTEMPTS:
        frames = [f for f in stack if "/discopt/" in f.filename.replace("\\\\", "/")]
        names = [f.name for f in frames]
        if "highs_available" in names:
            out.append((inst, "probe", "highs_available"))
            continue
        # innermost discopt frame outside the optional module, and its caller
        own = [f for f in frames if not f.filename.endswith("_optional_highs.py")]
        inner = own[-1].name if own else "?"
        caller = own[-2].name if len(own) > 1 else "?"
        out.append((inst, inner, caller, own[-1].filename.rsplit("/", 1)[-1] if own else "?"))
    return out
"""


def _run(body: str, timeout: float) -> dict:
    p = subprocess.run(
        [sys.executable, "-c", _BLOCK_HIGHS + body],
        capture_output=True,
        text=True,
        cwd="/",
        env=dict(os.environ),
        timeout=timeout,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert p.returncode == 0 and len(lines) == 1, p.stderr[-3000:]
    return json.loads(lines[0][len("RESULT ") :])


_SOLVE_ALL = r"""
import json, glob, os, sys
from discopt.modeling.core import from_nl
solved, errors = [], []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.nl")))[: int(sys.argv[2])]:
    name = os.path.basename(path)[:-3]
    CURRENT[0] = name
    try:
        r = from_nl(path).solve(time_limit=float(sys.argv[3]))
        solved.append((name, str(r.status), r.algorithm_route))
    except Exception as e:
        errors.append((name, type(e).__name__, str(e)[:300]))
print("RESULT " + json.dumps({"solved": solved, "errors": errors, "roles": _roles()}))
"""


def _classify(res: dict) -> tuple[list, list, list]:
    probes, known, unknown = [], [], []
    for row in res["roles"]:
        if row[1] == "probe":
            probes.append(row)
        elif row[1] == "require_highspy" and row[2] in ROLE_CALLERS:
            known.append(row)
        elif row[1] in ROLE_CALLERS:
            known.append(row)
        else:
            unknown.append(row)
    return probes, known, unknown


@pytest.mark.slow
def test_corpus_reaches_highs_only_through_known_roles():
    body = _SOLVE_ALL.replace("sys.argv[1]", repr(str(NL_DIR))).replace(
        "int(sys.argv[2])", "10_000"
    )
    body = body.replace("float(sys.argv[3])", "10.0")
    res = _run(body, timeout=3000)
    probes, known, unknown = _classify(res)
    assert res["solved"] or res["errors"], "nothing ran"
    assert not unknown, f"new HiGHS reach on a default path: {unknown[:10]}"
    # Reported, so a change in the remaining roles is visible in the log.
    summary = {
        "solved": len(res["solved"]),
        "errors": len(res["errors"]),
        "probe_attempts": len(probes),
        "role_attempts": sorted({(r[1], r[2]) for r in known}),
    }
    print(json.dumps(summary, indent=1))


@pytest.mark.smoke
def test_a_highs_only_call_raises_the_named_error():
    res = _run(
        r"""
import json
from discopt._optional_highs import MissingHighsError
from discopt.solvers.lp_milp_highs import require_highspy
try:
    require_highspy()
    out = {"raised": None}
except MissingHighsError as e:
    out = {"raised": type(e).__name__, "name": e.name, "msg": str(e)}
print("RESULT " + json.dumps(out))
""",
        timeout=120,
    )
    assert res["raised"] == "MissingHighsError" and res["name"] == "highspy", res
    assert "discopt[highs]" in res["msg"], res
