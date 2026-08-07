"""Issue #940: replicate the corpus-panel instances that differed.

The panel runs under a 20s WALL budget, so node counts, the bound at
termination, and even whether an incumbent was found are throughput-dependent
and load-sensitive. A single differing run therefore proves nothing. This
applies the house replication protocol already used for the separation-LP panel
(see solver.py's note): re-run each decisive instance R times per arm with the
arms INTERLEAVED, then classify

    STABLE      — the difference reproduces in every replicate
    QUARANTINE  — replicates disagree; the run was noise, not a signal
    CLEAN       — the arms agree in every replicate

and, separately from any of that, check SOUNDNESS on every replicate against the
reference-optima registry: a dual bound may never exceed the true optimum (for a
minimize sense) and an incumbent may never beat it.

§6: prints an executed-comparison count and exits non-zero when it is zero.
"""

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, "python/tests")

import discopt.solvers.lp_pounce as LPP  # noqa: E402
import discopt.solvers.qp_pounce as QPP  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
assert hasattr(LPP, "_settle_ambiguous_unbounded"), "post-fix marker absent (§8)"
assert LPP._CONSTR_VIOL_TOL == 1e-8, LPP._CONSTR_VIOL_TOL

_POST_TOL = LPP._CONSTR_VIOL_TOL
_POST_SETTLE = LPP._settle_ambiguous_unbounded
_POST_RAY = QPP._certify_unbounded_ray
_IPOPT_DEFAULT_TOL = 1e-4

ROOT = "python/tests/data/minlplib_nl"
TIME_LIMIT = 20.0
REPS = 3

DECISIVE = [
    "bchoco06", "clay0303hfsg", "contvar", "gbd", "nvs05", "st_miqp4",
    "st_miqp5", "syn05hfsg", "tanksize", "tls2", "tspn12",
]


def set_arm(arm):
    if arm == "pre":
        LPP._CONSTR_VIOL_TOL = _IPOPT_DEFAULT_TOL
        QPP._CONSTR_VIOL_TOL = _IPOPT_DEFAULT_TOL
        LPP._settle_ambiguous_unbounded = lambda result, c, A, cl, cu, lb, ub, opts: result
        QPP._certify_unbounded_ray = lambda *a, **k: True
    else:
        LPP._CONSTR_VIOL_TOL = _POST_TOL
        QPP._CONSTR_VIOL_TOL = _POST_TOL
        LPP._settle_ambiguous_unbounded = _POST_SETTLE
        QPP._certify_unbounded_ray = _POST_RAY


def solve(name, arm):
    set_arm(arm)
    res = from_nl(os.path.join(ROOT, f"{name}.nl")).solve(time_limit=TIME_LIMIT)
    return {"status": res.status, "objective": res.objective,
            "bound": res.bound, "node_count": res.node_count}


def optimum(name):
    try:
        from _optima import known_optimum
    except ImportError:
        return None
    try:
        return float(known_optimum(name))
    except KeyError:
        return None


def main():
    comparisons = 0
    soundness_checks = 0
    violations = []
    out = {}

    for name in DECISIVE:
        opt = optimum(name)
        reps = []
        for r in range(REPS):
            got = {}
            for arm in (("pre", "post") if r % 2 == 0 else ("post", "pre")):
                t0 = time.perf_counter()
                got[arm] = solve(name, arm)
                got[arm]["wall"] = time.perf_counter() - t0
                # Soundness is checked on EVERY run of EVERY arm, independent of
                # whether the arms happen to agree (CLAUDE.md §1).
                if opt is not None and got[arm]["bound"] is not None:
                    soundness_checks += 1
                    tol = 1e-6 + 1e-4 * abs(opt)
                    if got[arm]["bound"] > opt + tol:
                        violations.append(
                            (name, arm, r, "bound above true optimum",
                             got[arm]["bound"], opt))
                if opt is not None and got[arm]["objective"] is not None:
                    soundness_checks += 1
                    tol = 1e-6 + 1e-4 * abs(opt)
                    if got[arm]["objective"] < opt - tol:
                        violations.append(
                            (name, arm, r, "incumbent beats true optimum",
                             got[arm]["objective"], opt))
            reps.append(got)
            comparisons += 1

        keys = ("status", "objective", "bound", "node_count")
        differs = [any(rep["pre"][k] != rep["post"][k] for k in keys) for rep in reps]
        if not any(differs):
            verdict = "CLEAN"
        elif all(differs):
            verdict = "STABLE-DIFF"
        else:
            verdict = "QUARANTINE"
        out[name] = {"verdict": verdict, "optimum": opt, "reps": reps}
        print(f"{name:14s} {verdict:12s} opt={opt}", flush=True)
        for r, rep in enumerate(reps):
            print(f"    r{r} pre ={rep['pre']['status']:11s} obj={rep['pre']['objective']!r} "
                  f"bound={rep['pre']['bound']!r} n={rep['pre']['node_count']}")
            print(f"    r{r} post={rep['post']['status']:11s} obj={rep['post']['objective']!r} "
                  f"bound={rep['post']['bound']!r} n={rep['post']['node_count']}")

    json.dump(out, open("scratchpad/issue940/panel_replicate.json", "w"), indent=1)
    print(f"\nCOMPARISONS_EXECUTED={comparisons}  SOUNDNESS_CHECKS={soundness_checks}")
    print(f"SOUNDNESS VIOLATIONS: {len(violations)}")
    for v in violations:
        print(f"   {v}")
    for verdict in ("CLEAN", "QUARANTINE", "STABLE-DIFF"):
        names = [n for n, d in out.items() if d["verdict"] == verdict]
        print(f"  {verdict:12s}: {len(names)}  {names}")
    if comparisons == 0:
        print("REPLICATION COMPARED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
