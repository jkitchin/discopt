"""#1153 decision panel: legacy vs the flat finder share, over the open subset.

One process, arms interleaved at every rung (CLAUDE.md §9), over the budgets
where #1153's harm actually lives (5/10/20/40 s). Reports, per arm, the
monotonicity violations (the issue's gate) and, across arms, the incumbent /
bound / node differential. Executed-comparison count; non-zero exit at zero.
"""
import json, os, statistics, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "heuristic_entry_share"), "marker absent — wrong tree"

corpus, ladder_s, out, subset_f = sys.argv[1:5]
ladder = [float(x) for x in ladder_s.split(",")]
names = json.load(open(subset_f))
print(f"# instances={len(names)} ladder={ladder} arms=legacy,flat", flush=True)

res = {}
for name in names:
    for tl in ladder:
        for arm, on in (("legacy", False), ("flat", True)):
            tok = solver_tuning.enter_scope(
                solver_tuning.SolverTuning(heuristic_entry_share=on)
            )
            t = time.perf_counter()
            try:
                r = from_nl(os.path.join(corpus, f"{name}.nl")).solve(
                    time_limit=tl, gap_tolerance=1e-4
                )
                row = dict(tl=tl, obj=None if r.objective is None else float(r.objective),
                           bound=None if r.bound is None else float(r.bound),
                           nodes=int(r.node_count or 0), status=str(r.status),
                           cert=bool(r.gap_certified), wall=time.perf_counter() - t)
            except Exception as exc:
                row = dict(tl=tl, error=repr(exc), wall=time.perf_counter() - t)
            finally:
                solver_tuning.reset_current(tok)
            res.setdefault(arm, {}).setdefault(name, []).append(row)
            print(f"[{arm:6s}] {name:22s} tl={tl:5.1f} obj={row.get('obj')!r} "
                  f"bound={row.get('bound')!r} nodes={row.get('nodes')} "
                  f"cert={row.get('cert')} {row.get('error','')}", flush=True)
    json.dump(res, open(out, "w"), indent=1)

n_cmp = {a: 0 for a in res}
n_viol = {a: 0 for a in res}
for arm in res:
    for name, rows in res[arm].items():
        for a, b in zip(rows, rows[1:]):
            if "error" in a or "error" in b or a["obj"] is None:
                continue
            n_cmp[arm] += 1
            if b["obj"] is None or b["obj"] > a["obj"] + 1e-6 * max(1.0, abs(a["obj"])):
                n_viol[arm] += 1
                print(f"!! VIOLATION [{arm}] {name}: tl={a['tl']} {a['obj']} -> "
                      f"tl={b['tl']} {b['obj']}", flush=True)
for arm in res:
    print(f"# arm={arm} comparisons={n_cmp[arm]} violations={n_viol[arm]}", flush=True)
raise SystemExit(1 if sum(n_cmp.values()) == 0 else 0)
