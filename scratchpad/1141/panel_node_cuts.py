"""Differential panel for the #1141 fractional-node hook: DISCOPT_OA_NODE_CUTS
OFF vs ON on the `lp_nlp_bb` / simplex master.

Runs each arm in the SAME process, interleaved per instance (CLAUDE.md §9), and
reports per-instance objective, dual bound, status, wall and the separator's
anti-vacuity counters. Exits non-zero if no comparison was executed (§6).
"""
import argparse, os, sys, time, importlib, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def load(name, nlpath):
    if nlpath is not None:
        from discopt.modeling.core import from_nl
        return from_nl(str(nlpath))
    import portfolio2
    kw = dict(x.split("=") for x in name.split(",") if "=" in x)
    return portfolio2.build(**{k: (float(v) if "." in v else int(v)) for k, v in kw.items()})


def run(model, node_cuts, time_limit):
    os.environ["DISCOPT_OA_NODE_CUTS"] = "1" if node_cuts else "0"
    t = time.perf_counter()
    r = model.solve(
        solver="mip-nlp",
        mip_nlp_method="lp_nlp_bb",
        milp_solver="simplex",
        time_limit=time_limit,
        gap_tolerance=1e-4,
    )
    wall = time.perf_counter() - t
    stats = getattr(r, "solver_stats", None) or {}
    return dict(
        status=str(r.status), obj=r.objective, bound=r.bound, wall=wall,
        nodes=getattr(r, "node_count", None),
        stats={k: v for k, v in stats.items() if "node" in str(k) or "lazy" in str(k)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="")
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--nl", default="")
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()

    items = []
    for n in filter(None, a.instances.split(";")):
        items.append((n, None))
    for n in filter(None, a.nl.split(",")):
        items.append((n, pathlib.Path(a.nl_dir) / f"{n}.nl"))

    compared = 0
    print(f"{'instance':26s} {'arm':4s} {'status':12s} {'objective':>16s} {'bound':>16s} "
          f"{'wall':>7s}  counters")
    for name, nlpath in items:
        res = {}
        for arm in ("off", "on"):
            model = load(name, nlpath)  # fresh model per arm: no leftover state
            try:
                res[arm] = run(model, arm == "on", a.time_limit)
            except Exception as exc:  # never swallowed (CLAUDE.md §7)
                print(f"{name:26s} {arm:4s} RAISED {type(exc).__name__}: {exc}", flush=True)
                raise
            r = res[arm]
            ob = "None" if r["obj"] is None else f"{r['obj']:.10g}"
            bd = "None" if r["bound"] is None else f"{r['bound']:.10g}"
            print(f"{name:26s} {arm:4s} {r['status']:12s} {ob:>16s} {bd:>16s} "
                  f"{r['wall']:7.2f}  {r['stats']}", flush=True)
        compared += 1
    print(f"\nEXECUTED COMPARISONS: {compared}")
    if compared == 0:
        print("PANEL MEASURED NOTHING", file=sys.stderr)
        sys.exit(1)


main()
