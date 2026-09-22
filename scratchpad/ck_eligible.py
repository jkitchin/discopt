"""#1422 entry experiment, step 1: which in-repo instances are convex-kernel
ELIGIBLE, and how long does the kernel need on each?

Eligible-and-certifies  -> the reserve's cost is bounded by its certify time.
Eligible-and-declines   -> the class #1422 reports: budget spent, nothing back.

Exits non-zero if it classified nothing.
"""
import pathlib, sys, time
import discopt
from discopt.modeling.core import from_nl
from discopt.solvers import _convex_kernel as ck

print("kernel __file__:", ck.__file__, flush=True)
assert "projects/discopt" in ck.__file__

CORPUS = pathlib.Path("python/tests/data/minlplib_nl")
nls = sorted(CORPUS.glob("*.nl"))
assert len(nls) > 20, f"corpus too small: {len(nls)}"
print(f"corpus: {len(nls)} .nl files\n", flush=True)

BUDGET = 60.0
eligible = []
n_built = 0
for p in nls:
    try:
        m = from_nl(str(p))
        spec = ck.build_convex_spec(m)
    except Exception as e:
        print(f"{p.stem:24s} SPEC-ERROR {type(e).__name__}: {e}", flush=True)
        continue
    n_built += 1
    if spec is None:
        continue
    t0 = time.perf_counter()
    r = ck.solve_convex_tree(spec, time_limit_s=BUDGET, gap_tol=1e-4, initial_incumbent=None)
    w = time.perf_counter() - t0
    eligible.append((p.stem, r["status"], w, r["node_count"], r["bound"], r["incumbent"]))
    print(f"{p.stem:24s} ELIGIBLE  status={r['status']:<11s} t={w:7.3f}s "
          f"nodes={r['node_count']:<7d} bound={r['bound']}", flush=True)

print(f"\nmodels built : {n_built}")
print(f"kernel-eligible: {len(eligible)}")
cert = [e for e in eligible if e[1] == "optimal"]
decl = [e for e in eligible if e[1] != "optimal"]
print(f"  certifies within {BUDGET}s : {len(cert)}")
if cert:
    print(f"    certify times: {sorted(round(e[2],3) for e in cert)}")
print(f"  declines                 : {len(decl)}  {[e[0] for e in decl]}")
print(f"\nEXECUTED CLASSIFICATIONS: {n_built}")
sys.exit(0 if n_built > 0 else 1)
