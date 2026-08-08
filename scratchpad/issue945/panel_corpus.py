"""Issue #945 differential panel: seeding the NLP path from pounce_option_defaults.

CLAUDE.md §5 regime 2 (bound-changing): the arms are compared on certification
soundness first and search behaviour second. For every instance and arm it
records status / objective / bound / node_count, and checks against the
authoritative optima registry (``python/tests/data/known_optima.toml``):

  * ``bound_above_opt``  — a dual bound above the reference optimum (min sense)
                           has cut the optimum out of the search. Fatal.
  * ``super_optimal``    — an incumbent below the reference optimum is not a
                           feasible point. Fatal.
  * ``cert_inverted``    — bound > incumbent, the certificate invariant. Fatal.
  * certification regression — an instance that reported a *certified* gap in the
                           pre arm must not lose it in the post arm.

Both arms run INTERLEAVED in one process (§9), alternating which goes first, so
background load or cache warmth cannot land on one arm only. The pre arm is
reconstructed exactly as ``nlp_pounce.solve_nlp`` was before #945: ``print_level``
alone, leaving BOTH ``constr_viol_tol`` (Ipopt default 1e-4) and
``bound_relax_factor`` (1e-8) at Ipopt's values.

§6: this panel counts the ``solve_nlp`` calls each arm actually made. If that
count is zero the panel exercised nothing of the change under test and exits
non-zero — a green "0 differences" from a code path that never ran is the exact
failure mode this file exists to avoid.

Usage:  python -u scratchpad/issue945/panel_corpus.py out.json [time_limit]
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "tests"))

import discopt.solvers.nlp_pounce as NLPP  # noqa: E402
from _optima import known_optimum  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

# §8: assert we loaded the code under test, by file AND by a marker unique to it.
assert NLPP.__file__.startswith("/home/user/discopt/python/"), NLPP.__file__
assert "opts = pounce_option_defaults()" in inspect.getsource(NLPP.solve_nlp), (
    "post-#945 marker absent: nlp_pounce.solve_nlp is not seeded from the shared "
    "baseline, so there is no change here to panel"
)

_PRE_ARM = {"print_level": 0}
_REAL = pounce_option_defaults

ROOT = "python/tests/data/minlplib_nl"

# Counts solve_nlp entries per arm, so the panel can prove the path ran.
_CALLS = {"pre": 0, "post": 0}
_ARM = "post"
_ORIG_SOLVE_NLP = NLPP.solve_nlp


def _counting_solve_nlp(*args, **kwargs):
    _CALLS[_ARM] += 1
    return _ORIG_SOLVE_NLP(*args, **kwargs)


NLPP.solve_nlp = _counting_solve_nlp
# The callers import `solve_nlp` lazily inside functions, so patching the module
# attribute is enough; but solver.py's batch path reads the options directly.
import discopt.solver as SOLVER  # noqa: E402

_ORIG_BATCH_DEFAULTS = SOLVER.pounce_option_defaults


def set_arm(arm: str) -> None:
    global _ARM
    _ARM = arm
    if arm == "pre":
        NLPP.pounce_option_defaults = lambda: dict(_PRE_ARM)
        SOLVER.pounce_option_defaults = lambda: dict(_PRE_ARM)
    else:
        NLPP.pounce_option_defaults = _REAL
        SOLVER.pounce_option_defaults = _ORIG_BATCH_DEFAULTS


def solve_one(path: str, arm: str, time_limit: float) -> dict:
    set_arm(arm)
    t0 = time.perf_counter()
    try:
        res = from_nl(path).solve(time_limit=time_limit)
    except Exception as exc:  # recorded, never swallowed (§7)
        return {"error": f"{type(exc).__name__}: {exc}", "wall": time.perf_counter() - t0}
    return {
        "status": res.status,
        "objective": res.objective,
        "bound": res.bound,
        "node_count": res.node_count,
        "gap_certified": getattr(res, "gap_certified", None),
        "wall": time.perf_counter() - t0,
    }


# Reference optima are stored in the instance's own sense; the .nl files in this
# corpus are all minimizations, which the registry documents.
_BOUND_SLACK = 1e-6
_REL_SLACK = 1e-6


def _oracle_findings(name: str, r: dict) -> list[str]:
    """Soundness checks against the reference optimum. Empty list == clean."""
    if "error" in r:
        return []
    opt = known_optimum(name)
    if opt is None:
        return []
    out = []
    tol = _BOUND_SLACK + _REL_SLACK * abs(opt)
    b, o = r.get("bound"), r.get("objective")
    if b is not None and b > opt + tol:
        out.append(f"bound_above_opt({b:.12g} > {opt:.12g})")
    if o is not None and o < opt - tol:
        out.append(f"super_optimal({o:.12g} < {opt:.12g})")
    if b is not None and o is not None and b > o + tol:
        out.append(f"cert_inverted(bound {b:.12g} > incumbent {o:.12g})")
    return out


def main(out_path: str, time_limit: float) -> int:
    files = sorted(f for f in os.listdir(ROOT) if f.endswith(".nl"))
    rows: dict[str, dict] = {}
    comparisons = 0
    oracle_checks = 0
    diffs: list = []
    findings: list = []
    cert_regressions: list = []

    for i, f in enumerate(files, 1):
        name = f[:-3]
        p = os.path.join(ROOT, f)
        # Alternate which arm runs first so drift cannot systematically favour one.
        order = ("pre", "post") if i % 2 else ("post", "pre")
        got = {a: solve_one(p, a, time_limit) for a in order}
        rows[name] = got
        comparisons += 1

        for arm in ("pre", "post"):
            fs = _oracle_findings(name, got[arm])
            if known_optimum(name) is not None and "error" not in got[arm]:
                oracle_checks += 1
            for msg in fs:
                findings.append((name, arm, msg))

        # A certified gap must never be lost.
        if got["pre"].get("gap_certified") and not got["post"].get("gap_certified"):
            cert_regressions.append(name)

        keys = ("status", "objective", "bound", "node_count")
        same = all(got["pre"].get(k) == got["post"].get(k) for k in keys)
        if not same:
            diffs.append(
                (name, {k: (got["pre"].get(k), got["post"].get(k)) for k in keys
                        if got["pre"].get(k) != got["post"].get(k)})
            )
        print(
            f"[{i:3d}/{len(files)}] {name:28s} {'SAME' if same else 'DIFF'} "
            f"pre={got['pre'].get('status')}/{got['pre'].get('objective')!r}/"
            f"n{got['pre'].get('node_count')} "
            f"post={got['post'].get('status')}/{got['post'].get('objective')!r}/"
            f"n{got['post'].get('node_count')}",
            flush=True,
        )

    json.dump(
        {
            "rows": rows,
            "diffs": diffs,
            "findings": findings,
            "cert_regressions": cert_regressions,
            "solve_nlp_calls": dict(_CALLS),
            "time_limit": time_limit,
        },
        open(out_path, "w"),
        indent=1,
    )

    print(
        f"\nCOMPARISONS_EXECUTED={comparisons}  identical={comparisons - len(diffs)}  "
        f"differing={len(diffs)}"
    )
    print(f"ORACLE_CHECKS_EXECUTED={oracle_checks}  soundness_findings={len(findings)}")
    print(f"SOLVE_NLP_CALLS pre={_CALLS['pre']} post={_CALLS['post']}")
    print(f"CERT_REGRESSIONS={len(cert_regressions)} {cert_regressions}")
    for name, d in diffs:
        print(f"  DIFF {name}: {d}")
    for name, arm, msg in findings:
        print(f"  FINDING {name} [{arm}]: {msg}")

    if comparisons == 0:
        print("PANEL COMPARED NOTHING", file=sys.stderr)
        return 1
    if _CALLS["pre"] == 0 and _CALLS["post"] == 0:
        print(
            "PANEL NEVER ENTERED solve_nlp — it measured nothing about the change "
            "under test; do not read its 0 differences as evidence",
            file=sys.stderr,
        )
        return 1
    if findings or cert_regressions:
        return 2
    return 0


if __name__ == "__main__":
    out = sys.argv[1]
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    sys.exit(main(out, tl))
