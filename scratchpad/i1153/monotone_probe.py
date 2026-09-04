"""#1153 entry experiment: is incumbent quality monotone in ``time_limit``?

Prints one row per (instance, time_limit) and an EXECUTED-COMPARISON COUNT
(CLAUDE.md §6); exits non-zero when zero comparisons ran.
"""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
from discopt.modeling.core import from_nl

# CLAUDE.md §8: prove which code was loaded.
print(f"# discopt.__file__ = {discopt.__file__}", flush=True)

def run(path, tl):
    m = from_nl(path)
    t = time.perf_counter()
    r = m.solve(time_limit=tl, gap_tolerance=1e-4)
    wall = time.perf_counter() - t
    return dict(
        tl=tl,
        obj=None if r.objective is None else float(r.objective),
        bound=None if r.bound is None else float(r.bound),
        nodes=int(getattr(r, "node_count", 0) or 0),
        status=str(r.status),
        cert=bool(getattr(r, "gap_certified", False)),
        wall=wall,
    )

def main(argv):
    inst = argv[1]
    tls = [float(x) for x in argv[2].split(",")]
    reps = int(argv[3]) if len(argv) > 3 else 1
    rows = []
    n_cmp = 0
    # Interleave repetitions (CLAUDE.md §9), not budget-major.
    for rep in range(reps):
        for tl in tls:
            row = run(inst, tl)
            row["rep"] = rep
            rows.append(row)
            print(f"{os.path.basename(inst):24s} rep={rep} tl={tl:6.1f} "
                  f"obj={row['obj']!r} bound={row['bound']!r} nodes={row['nodes']} "
                  f"status={row['status']} cert={row['cert']} wall={row['wall']:.2f}",
                  flush=True)
    # Monotonicity comparisons over consecutive budgets within a rep.
    for rep in range(reps):
        r_rep = [r for r in rows if r["rep"] == rep]
        for a, b in zip(r_rep, r_rep[1:]):
            n_cmp += 1
            if a["obj"] is not None and b["obj"] is not None and b["obj"] > a["obj"] + 1e-6:
                print(f"!! NON-MONOTONE rep={rep}: tl={a['tl']} obj={a['obj']} "
                      f"-> tl={b['tl']} obj={b['obj']}", flush=True)
            elif a["obj"] is not None and b["obj"] is None:
                print(f"!! NON-MONOTONE rep={rep}: tl={a['tl']} obj={a['obj']} "
                      f"-> tl={b['tl']} obj=None", flush=True)
    print(f"# executed comparisons: {n_cmp}", flush=True)
    with open(os.environ.get("I1153_OUT", "/dev/null"), "w") as fh:
        json.dump(rows, fh, indent=1)
    if n_cmp == 0:
        print("PROBE MEASURED NOTHING", flush=True)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
