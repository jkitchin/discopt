"""Issue #940: the soundness gate, made to actually fire.

panel_replicate.py reported "0 violations" off SOUNDNESS_CHECKS=0 — every
decisive instance either had no reference optimum or returned bound=None under
the 20s budget, so the gate measured nothing (CLAUDE.md §6). This runs it where
it can execute:

  * the 16 corpus instances carrying a reference optimum in known_optima.toml,
  * with a budget generous enough to produce a bound and an incumbent,
  * both arms, interleaved, alternating order,

and asserts, per run:

  1. dual bound <= true optimum + tol      (the literal soundness invariant)
  2. incumbent >= true optimum - tol       (no super-optimal incumbent)
  3. bound <= incumbent + tol              (certificate invariant; oracle-free,
                                            so it fires on every run)

The run FAILS loudly if the executed-assertion count is zero.
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
from _optima import known_optimum, optima_registry  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
assert hasattr(LPP, "_settle_ambiguous_unbounded"), "post-fix marker absent (§8)"
assert LPP._CONSTR_VIOL_TOL == 1e-8, LPP._CONSTR_VIOL_TOL

_POST_TOL = LPP._CONSTR_VIOL_TOL
_POST_SETTLE = LPP._settle_ambiguous_unbounded
_POST_RAY = QPP._certify_unbounded_ray

ROOT = "python/tests/data/minlplib_nl"
TIME_LIMIT = 60.0


def set_arm(arm):
    if arm == "pre":
        LPP._CONSTR_VIOL_TOL = QPP._CONSTR_VIOL_TOL = 1e-4
        LPP._settle_ambiguous_unbounded = lambda result, c, A, cl, cu, lb, ub, opts: result
        QPP._certify_unbounded_ray = lambda *a, **k: True
    else:
        LPP._CONSTR_VIOL_TOL = QPP._CONSTR_VIOL_TOL = _POST_TOL
        LPP._settle_ambiguous_unbounded = _POST_SETTLE
        QPP._certify_unbounded_ray = _POST_RAY


def main():
    corpus = {f[:-3] for f in os.listdir(ROOT) if f.endswith(".nl")}
    names = sorted(corpus & set(optima_registry()))
    print(f"instances with a reference optimum: {len(names)}\n{names}\n", flush=True)

    checks = {"bound_vs_optimum": 0, "incumbent_vs_optimum": 0, "bound_vs_incumbent": 0}
    violations = []
    out = {}

    for i, name in enumerate(names):
        opt = float(known_optimum(name))
        got = {}
        for arm in (("pre", "post") if i % 2 == 0 else ("post", "pre")):
            set_arm(arm)
            t0 = time.perf_counter()
            r = from_nl(os.path.join(ROOT, f"{name}.nl")).solve(time_limit=TIME_LIMIT)
            got[arm] = {"status": r.status, "objective": r.objective, "bound": r.bound,
                        "node_count": r.node_count, "wall": time.perf_counter() - t0}
            tol = 1e-6 + 1e-4 * abs(opt)
            if r.bound is not None:
                checks["bound_vs_optimum"] += 1
                if r.bound > opt + tol:
                    violations.append((name, arm, "BOUND ABOVE TRUE OPTIMUM", r.bound, opt))
            if r.objective is not None:
                checks["incumbent_vs_optimum"] += 1
                if r.objective < opt - tol:
                    violations.append((name, arm, "INCUMBENT BEATS TRUE OPTIMUM",
                                       r.objective, opt))
            if r.bound is not None and r.objective is not None:
                checks["bound_vs_incumbent"] += 1
                if r.bound > r.objective + tol:
                    violations.append((name, arm, "BOUND ABOVE INCUMBENT",
                                       r.bound, r.objective))
        out[name] = {"optimum": opt, **got}
        agree = all(got["pre"][k] == got["post"][k]
                    for k in ("status", "objective", "bound", "node_count"))
        print(f"[{i + 1:2d}/{len(names)}] {name:22s} opt={opt:<14.8g} "
              f"{'SAME' if agree else 'DIFF'} "
              f"pre={got['pre']['status']}/{got['pre']['objective']!r}/b{got['pre']['bound']!r} "
              f"post={got['post']['status']}/{got['post']['objective']!r}/"
              f"b{got['post']['bound']!r}", flush=True)

    json.dump({"checks": checks, "violations": violations, "rows": out},
              open("scratchpad/issue940/panel_soundness.json", "w"), indent=1)

    total = sum(checks.values())
    print(f"\nEXECUTED_ASSERTIONS={total}  {checks}")
    print(f"SOUNDNESS VIOLATIONS: {len(violations)}")
    for v in violations:
        print(f"   {v}")
    if total == 0:
        print("GATE EXECUTED ZERO ASSERTIONS - result is meaningless", file=sys.stderr)
        return 1
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
