"""Issue #945 differential panel: the NLP path stops returning out-of-box points.

CLAUDE.md §5 regime 2 (bound-changing): the arms are compared on certification
soundness first and search behaviour second. For every instance and arm it
records status / objective / bound / node_count, and checks against the
authoritative optima registry (``python/tests/data/known_optima.toml``):

  * ``cert_inverted``    — the bound crossed the incumbent (the certificate
                           invariant). Needs no oracle, so it runs on EVERY solved
                           instance, and is sense-aware: for a maximization the
                           bound is the UPPER side. Fatal.
  * ``bound_past_opt``   — a dual bound beyond the reference optimum has cut the
                           optimum out of the search. Fatal.
  * ``super_optimal``    — an incumbent better than the reference optimum is not a
                           feasible point. Fatal.
  * certification regression — an instance that reported a *certified* gap in the
                           pre arm must not lose it in the post arm.

Both arms run INTERLEAVED in one process (§9), alternating which goes first, so
background load or cache warmth cannot land on one arm only. The pre arm leaves
BOTH ``constr_viol_tol`` (Ipopt default 1e-4) and ``bound_relax_factor`` (1e-8)
at Ipopt's values everywhere, which is what the pre-#945 tree did — see the note
on ``set_arm`` for why that takes two seams rather than one.

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
from _optima import optima_registry  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

# §8: assert we loaded the code under test, by file AND by a marker unique to it.
# The marker tracks the CONTRACT, not a line of code: #945 requests the incumbent
# options at the point-consuming call sites and deliberately leaves the backend
# neutral. An earlier version of this assert named the superseded contract and
# refused to run once the branch rescoped — which is the guard working.
_REPO_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
assert NLPP.__file__.startswith(_REPO_PYTHON + os.sep), (
    f"loaded discopt from {NLPP.__file__}, not the tree under test at {_REPO_PYTHON} — "
    "a panel run against an installed copy measures the wrong code (§8)"
)

# The pre-#945 arm. Reconstructing it takes TWO seams, not one: #945 does NOT put
# bound_relax_factor in the backend default (that breaks the Benders dual LP — see
# solvers.pounce_incumbent_options), so the arm has to neutralize both the backend
# baseline AND the incumbent options requested by the call sites whose returned
# point becomes a solution. Patching only the backend would leave the incumbent
# requests live and silently mislabel the arm — the #940 lesson, one level up.
_PRE_BACKEND = {"print_level": 0}
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
import discopt._jax.primal_heuristics as PH  # noqa: E402
import discopt.solver as SOLVER  # noqa: E402
import discopt.solvers.gdpopt_loa as LOA  # noqa: E402
import discopt.solvers.oa as OA  # noqa: E402

# Every module that requests the incumbent options; the arm must cover all of them.
# ``PH`` joined the list when the class was closed at the two producers the first
# pass missed: ``feasibility_pump`` (and its five siblings, all routed through
# ``PH._heuristic_nlp_options``) and ``_solve_nlp_bb``'s terminal refine solve.
# Both read the module-global name, so patching the module attribute covers them.
_CONSUMERS = (SOLVER, OA, LOA, PH)
_REAL_INCUMBENT = SOLVER.pounce_incumbent_options

# Post-#945 markers, checked here rather than at import of NLPP because the change
# now lives at the CALL SITES. Absent => there is nothing to panel, and a green
# "0 differences" would be measuring the same tree twice.
for _fn in (
    SOLVER._solve_continuous,
    OA._solve_nlp_attempt,
    LOA._solve_nlp_subproblem,
    PH._heuristic_nlp_options,
):
    assert "pounce_incumbent_options()" in inspect.getsource(_fn), (
        f"post-#945 marker absent in {_fn.__name__}: the incumbent options are not "
        "requested there, so this panel would compare the pre arm against itself"
    )
assert "refine_opts.update(pounce_incumbent_options())" in inspect.getsource(
    SOLVER._solve_nlp_bb
), (
    "post marker absent in _solve_nlp_bb: its terminal refine solve is not "
    "requesting the incumbent options, so this panel would compare the pre arm "
    "against itself on the default MINLP path"
)
assert "pounce_incumbent_options()" not in inspect.getsource(NLPP.solve_nlp), (
    "bound_relax_factor leaked back into the NLP backend default; the pre arm "
    "reconstruction below assumes it is requested only at the call sites"
)


def set_arm(arm: str) -> None:
    global _ARM
    _ARM = arm
    pre = arm == "pre"
    NLPP.pounce_option_defaults = (lambda: dict(_PRE_BACKEND)) if pre else _REAL
    for mod in _CONSUMERS:
        mod.pounce_incumbent_options = (lambda: {}) if pre else _REAL_INCUMBENT
        if hasattr(mod, "pounce_option_defaults"):
            mod.pounce_option_defaults = (lambda: dict(_PRE_BACKEND)) if pre else _REAL


def solve_one(path: str, arm: str, time_limit: float) -> dict:
    set_arm(arm)
    t0 = time.perf_counter()
    try:
        model = from_nl(path)
        maximize = model._objective is not None and str(model._objective.sense).endswith("MAXIMIZE")
        res = model.solve(time_limit=time_limit)
    except Exception as exc:  # recorded, never swallowed (§7)
        return {"error": f"{type(exc).__name__}: {exc}", "wall": time.perf_counter() - t0}
    return {
        "maximize": maximize,
        "status": res.status,
        "objective": res.objective,
        "bound": res.bound,
        "node_count": res.node_count,
        "gap_certified": getattr(res, "gap_certified", None),
        "wall": time.perf_counter() - t0,
    }


# Reference optima are stored in the instance's own sense, and so are `bound` and
# `objective`; `maximize` per row is what keeps the comparisons oriented.
_BOUND_SLACK = 1e-6
_REL_SLACK = 1e-6

# `known_optimum` raises on an unrecorded instance by design; the registry covers
# 16 of this corpus's 66 instances, so look up rather than ask. Instances with no
# recorded optimum still get the certificate-ordering check and their pre/post
# diff — which is why the three counts are printed separately rather than as one
# reassuring total.
_REGISTRY = optima_registry()


def known_optimum(name: str):
    entry = _REGISTRY.get(name)
    return None if entry is None else entry["optimum"]


def _oracle_findings(name: str, r: dict, maximize: bool) -> list[str]:
    """Soundness checks. Empty list == clean.

    The certificate-ordering check needs no oracle and so is NOT gated on one —
    the first draft of this panel gated it, which silently skipped it on the 50 of
    66 instances with no recorded optimum. It is also sense-aware: for a
    maximization ``bound`` is an UPPER bound, so ``bound >= incumbent`` is the
    correct ordering there (``syn05hfsg`` is such an instance, and reads as a 2.76
    "inversion" if the sense is ignored).
    """
    if "error" in r:
        return []
    out = []
    b, o = r.get("bound"), r.get("objective")

    if b is not None and o is not None:
        scale = _BOUND_SLACK + _REL_SLACK * max(abs(b), abs(o))
        # In the model's own sense the bound is always the OPTIMISTIC side.
        crossed = (o - b) if maximize else (b - o)
        if crossed > scale:
            sense = "max" if maximize else "min"
            out.append(f"cert_inverted(bound {b:.12g} vs incumbent {o:.12g}, sense={sense})")

    opt = known_optimum(name)
    if opt is None:
        return out
    tol = _BOUND_SLACK + _REL_SLACK * abs(opt)
    if b is not None and (opt - b if maximize else b - opt) > tol:
        out.append(f"bound_past_opt({b:.12g} vs optimum {opt:.12g})")
    if o is not None and (o - opt if maximize else opt - o) > tol:
        out.append(f"super_optimal({o:.12g} vs optimum {opt:.12g})")
    return out


def main(out_path: str, time_limit: float) -> int:
    files = sorted(f for f in os.listdir(ROOT) if f.endswith(".nl"))
    rows: dict[str, dict] = {}
    comparisons = 0
    oracle_checks = 0
    cert_checks = 0
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
            r = got[arm]
            fs = _oracle_findings(name, r, bool(r.get("maximize")))
            if "error" not in r:
                # Certificate-ordering check fires on every solved instance; the
                # oracle comparisons additionally need a recorded optimum.
                if r.get("bound") is not None and r.get("objective") is not None:
                    cert_checks += 1
                if known_optimum(name) is not None:
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
                (
                    name,
                    {
                        k: (got["pre"].get(k), got["post"].get(k))
                        for k in keys
                        if got["pre"].get(k) != got["post"].get(k)
                    },
                )
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
    print(f"CERT_ORDER_CHECKS_EXECUTED={cert_checks}")
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
