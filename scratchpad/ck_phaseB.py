"""#1422 entry experiment, phase B: of the 39 convex-kernel-ELIGIBLE instances,
how many CERTIFY within a tight budget and how many DECLINE after spending it?

The decliners are the class #1422 item 1 reports: the attempt consumes the whole
budget (any `time_limit <= DISCOPT_CONVEX_KERNEL_BUDGET`, default 120) and the
default path is then handed ~0 s, so the solve returns neither incumbent nor
bound where a flag-off solve returns a sound bound.

Also records `first_incumbent_secs` on the certifiers. That is the number that
decides whether an "abandon the attempt if it still has no incumbent at f*budget"
policy is free on the instances the kernel wins: if every certifier finds its
first incumbent well before f*budget, the policy cannot cost one of them.

Exits non-zero if it solved nothing.
"""
import json, pathlib, sys, time
from discopt.modeling.core import from_nl
from discopt.solvers import _convex_kernel as ck

print("kernel __file__:", ck.__file__, flush=True)
BASE = pathlib.Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
names = json.load(open(sys.argv[1]))["eligible"]
assert len(names) > 10, f"eligible set too small: {len(names)}"
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
print(f"{len(names)} eligible instances, budget={BUDGET}s\n", flush=True)

rows, n = [], 0
for i, nm in enumerate(names):
    spec = ck.build_convex_spec(from_nl(str(BASE / f"{nm}.nl")))
    if spec is None:
        print(f"[{i+1}] {nm}: spec vanished (non-deterministic?) -- SKIPPED", flush=True)
        continue
    t0 = time.perf_counter()
    r = ck.solve_convex_tree(spec, time_limit_s=BUDGET, gap_tol=1e-4, initial_incumbent=None)
    w = time.perf_counter() - t0
    n += 1
    rows.append({"name": nm, "status": r["status"], "wall": w, "nodes": r["node_count"],
                 "bound": r["bound"], "incumbent": r["incumbent"],
                 "first_inc_secs": r["first_incumbent_secs"],
                 "first_inc_node": r["first_incumbent_node"]})
    print(f"[{i+1}/{len(names)}] {nm:26s} {r['status']:<11s} t={w:6.3f}s "
          f"nodes={r['node_count']:<6d} inc={r['incumbent']} "
          f"first_inc={r['first_incumbent_secs']}", flush=True)

pathlib.Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
cert = [r for r in rows if r["status"] == "optimal"]
decl = [r for r in rows if r["status"] != "optimal"]
waste = [r for r in decl if r["wall"] > 0.5 * BUDGET]
print(f"\nsolved            : {n}")
print(f"certifies         : {len(cert)}")
print(f"declines          : {len(decl)}")
print(f"  of which BURN >50% of the budget (the #1422 class): {len(waste)}")
print(f"  {[r['name'] for r in waste]}")
if cert:
    fi = [r["first_inc_secs"] for r in cert if r["first_inc_secs"] is not None]
    print(f"certifier first-incumbent secs: {sorted(round(x,4) for x in fi)}")
    print(f"certifier total walls        : {sorted(round(r['wall'],3) for r in cert)}")
print(f"\nEXECUTED SOLVES: {n}")
sys.exit(0 if n > 0 else 1)
