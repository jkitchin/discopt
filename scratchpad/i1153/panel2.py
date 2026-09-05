"""#1153 graduation panel: monotonicity + the ON/OFF differential in one run.

Usage: panel2.py <corpus> <ladder> <on|off> <out.json> <subset.json>

Runs the ladder on every instance in ``subset.json`` (instances that were still
OPEN at the largest phase-A rung — chosen by that measured property, never by
name) under one arm of ``budget_saturation``, and reports per-step monotonicity
violations. Prints per-solve progress and an executed-comparison count; exits
non-zero when nothing was compared (CLAUDE.md §6).
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
# CLAUDE.md §8: assert the marker unique to the version under test.
assert hasattr(solver_tuning, "saturate_role2"), "marker absent — wrong tree loaded"


def main(argv):
    corpus, ladder_s, arm, out, subset_f = argv[1:6]
    ladder = [float(x) for x in ladder_s.split(",")]
    on = arm == "on"
    names = json.load(open(subset_f))
    print(f"# arm={arm} ladder={ladder} instances={len(names)}", flush=True)
    results, n_cmp, n_viol = {}, 0, 0
    for name in names:
        path = os.path.join(corpus, f"{name}.nl")
        rows = []
        for tl in ladder:
            tok = solver_tuning.enter_scope(
                solver_tuning.SolverTuning(budget_saturation=on)
            )
            t = time.perf_counter()
            try:
                r = from_nl(path).solve(time_limit=tl, gap_tolerance=1e-4)
                row = dict(
                    tl=tl,
                    obj=None if r.objective is None else float(r.objective),
                    bound=None if r.bound is None else float(r.bound),
                    nodes=int(r.node_count or 0),
                    status=str(r.status),
                    cert=bool(r.gap_certified),
                    wall=time.perf_counter() - t,
                )
            except Exception as exc:
                row = dict(tl=tl, error=repr(exc), wall=time.perf_counter() - t)
            finally:
                solver_tuning.reset_current(tok)
            rows.append(row)
            print(
                f"[{arm}] {name:24s} tl={tl:6.1f} obj={row.get('obj')!r} "
                f"bound={row.get('bound')!r} nodes={row.get('nodes')} "
                f"cert={row.get('cert')} status={row.get('status')} "
                f"wall={row['wall']:.2f} {row.get('error','')}",
                flush=True,
            )
        results[name] = rows
        for a, b in zip(rows, rows[1:]):
            if "error" in a or "error" in b:
                continue
            n_cmp += 1
            ao, bo = a["obj"], b["obj"]
            if ao is None:
                continue
            worse = bo is None or bo > ao + 1e-6 * max(1.0, abs(ao))
            if worse:
                n_viol += 1
                print(f"!! VIOLATION [{arm}] {name}: tl={a['tl']} obj={ao} "
                      f"-> tl={b['tl']} obj={bo}", flush=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"# arm={arm} instances={len(names)} comparisons={n_cmp} violations={n_viol}",
          flush=True)
    return 1 if n_cmp == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
