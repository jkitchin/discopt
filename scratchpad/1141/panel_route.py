"""#1141 route measurement: the certified-convex route currently targets the
HiGHS separate-and-restart master. Does the in-house simplex master WITH
fractional-node separation beat it?

Interleaved per instance (CLAUDE.md §9). Prints an executed-comparison count (§6).
"""
import argparse, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

ARMS = {
    "highs":        ("highs", "0"),
    "simplex":      ("simplex", "0"),
    "simplex+node": ("simplex", "1"),
}


def load(spec, nl_dir):
    if spec.endswith(".nl") or "/" in spec:
        from discopt.modeling.core import from_nl
        return from_nl(spec)
    if "=" in spec:
        import portfolio2
        kw = {}
        for part in spec.split(","):
            k, v = part.split("=")
            kw[k] = float(v) if "." in v else int(v)
        return portfolio2.build(**kw)
    from discopt.modeling.core import from_nl
    return from_nl(str(pathlib.Path(nl_dir) / f"{spec}.nl"))


ap = argparse.ArgumentParser()
ap.add_argument("--instances", required=True)
ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
ap.add_argument("--time-limit", type=float, default=60.0)
a = ap.parse_args()

compared = 0
print(f"{'instance':30s} {'arm':14s} {'status':10s} {'objective':>16s} {'bound':>16s} {'wall':>7s}")
for spec in a.instances.split(";"):
    spec = spec.strip()
    if not spec:
        continue
    for arm, (backend, nodecuts) in ARMS.items():
        os.environ["DISCOPT_OA_NODE_CUTS"] = nodecuts
        m = load(spec, a.nl_dir)
        t = time.perf_counter()
        try:
            r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver=backend,
                        time_limit=a.time_limit, gap_tolerance=1e-4)
        except Exception as exc:
            print(f"{spec:30s} {arm:14s} RAISED {type(exc).__name__}: {exc}", flush=True)
            raise
        w = time.perf_counter() - t
        ob = "None" if r.objective is None else f"{r.objective:.10g}"
        bd = "None" if r.bound is None else f"{r.bound:.10g}"
        print(f"{spec:30s} {arm:14s} {str(r.status):10s} {ob:>16s} {bd:>16s} {w:7.2f}", flush=True)
        compared += 1
print(f"\nEXECUTED RUNS: {compared}")
sys.exit(1 if compared == 0 else 0)
