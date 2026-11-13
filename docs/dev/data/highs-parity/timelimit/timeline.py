import sys, time, os
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import lp_milp_loader as L
import discopt._rust as R
import discopt.modeling.core as core
from discopt.modeling.core import from_nl
import tempfile
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); mdl = from_nl(nl)
marks = []
real = R.solve_milp_csc_py
def spy(*a, **k):
    t = time.perf_counter(); marks.append(("rust_in", t, k.get("time_limit_s")))
    r = real(*a, **k); marks.append(("rust_out", time.perf_counter(), None)); return r
R.solve_milp_csc_py = spy
t0 = time.perf_counter()
r = mdl.solve(time_limit=float(sys.argv[2]))
t1 = time.perf_counter()
for m, t, x in marks:
    print(f"{m:9} {t - t0:7.2f}  budget={x}")
print(f"end       {t1 - t0:7.2f}  status={r.status}")
print("checks", len(marks))
sys.exit(0 if marks else 1)
