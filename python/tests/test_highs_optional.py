"""No default path reaches HiGHS (highs-parity P6.2 / P6.3).

Mirrors ``test_1873_jax_optional.py``. Each check runs in a fresh interpreter whose import
system refuses ``highspy`` and *records the stack of every attempt*, so a path that
catches the ``ImportError`` and quietly degrades is seen as well as one that crashes.

Since P6.3 ``highspy`` is the optional ``discopt[highs]`` extra. Without it the pure-MILP
route falls back to the in-house engine (``solver._lp_milp_backend``), and every other
default path is in-house already (``docs/dev/highs-optional-inventory.md``). So the guard
is the zero-attempt one: an attempt from ``highs_available()`` is an availability probe
and allowed; any other attempt FAILS, and so does any corpus instance that errors.
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


def _split(res: dict) -> tuple[list, list]:
    probes = [r for r in res["roles"] if r[1] == "probe"]
    reaches = [r for r in res["roles"] if r[1] != "probe"]
    return probes, reaches


@pytest.mark.slow
def test_corpus_never_reaches_highs():
    body = _SOLVE_ALL.replace("sys.argv[1]", repr(str(NL_DIR))).replace(
        "int(sys.argv[2])", "10_000"
    )
    body = body.replace("float(sys.argv[3])", "10.0")
    res = _run(body, timeout=3000)
    probes, reaches = _split(res)
    n = len(res["solved"]) + len(res["errors"])
    print(json.dumps({"instances": n, "solved": len(res["solved"]), "probes": len(probes)}))
    assert n >= 60, f"only {n} corpus instances ran"
    assert not reaches, f"a default path reached HiGHS: {reaches[:10]}"
    assert not res["errors"], f"instances failed without highspy: {res['errors'][:10]}"


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


_MILP = r"""
import discopt.modeling as dm
m = dm.Model("knap")
x = m.integer("x", shape=(3,), lb=0, ub=4)
m.maximize(5 * x[0] + 4 * x[1] + 3 * x[2])
m.subject_to(2 * x[0] + 3 * x[1] + x[2] <= 7)
"""


@pytest.mark.smoke
def test_milp_default_falls_back_to_the_in_house_engine_and_says_why():
    """P6.3: with no backend named and no highspy, a pure MILP runs on the in-house
    engine (never raises), certifies, and its route records why."""
    res = _run(
        _MILP
        + r"""
import json, os
os.environ.pop("DISCOPT_LP_MILP_BACKEND", None)
from discopt.solver import _lp_milp_backend
r = m.solve(time_limit=30)
print("RESULT " + json.dumps({"backend": _lp_milp_backend(), "status": r.status,
      "obj": r.objective, "cert": bool(r.gap_certified), "route": r.algorithm_route,
      "roles": _roles()}))
""",
        timeout=300,
    )
    assert res["backend"] == "rust", res
    assert res["status"] == "optimal" and res["cert"], res
    assert res["obj"] == pytest.approx(19.0), res  # x = (2, 0, 3): 10 + 9
    assert "highspy is not installed" in res["route"], res
    assert all(r[1] == "probe" for r in res["roles"]), res["roles"]


@pytest.mark.smoke
def test_highs_options_without_highs_are_refused():
    """P6.3: ``highs_options`` asks for HiGHS; without it the solve is refused with the
    named error, never run on another engine with the options dropped."""
    res = _run(
        _MILP
        + r"""
import json
from discopt._optional_highs import MissingHighsError
try:
    m.solve(time_limit=30, highs_options={"output_flag": True})
    out = {"raised": None}
except MissingHighsError as e:
    out = {"raised": type(e).__name__, "msg": str(e)}
print("RESULT " + json.dumps(out))
""",
        timeout=300,
    )
    assert res["raised"] == "MissingHighsError", res
    assert "highs_options" in res["msg"] and "discopt[highs]" in res["msg"], res
