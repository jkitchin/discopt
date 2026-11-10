"""What does discopt's root presolve do to a MILP, and what reaches the Rust engine?
Prints per-pass stats and the model size before/after (CLAUDE.md §6: prints what it measured)."""
import os, sys, tempfile, json, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import lp_milp_loader as L
from discopt.modeling.core import from_nl
from discopt._relax.presolve_pipeline import run_root_presolve
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); mdl = from_nl(nl)
print("vars", len(mdl._variables), "constraints", len(mdl._constraints))
from discopt._rust import model_to_repr
rep = model_to_repr(mdl, getattr(mdl, "_builder", None))
t = time.time()
new_rep, stats = run_root_presolve(rep, eliminate=True, fbbt=True, time_limit_ms=30000)
print("presolve wall %.2fs" % (time.time() - t))
for k, v in stats.items():
    if isinstance(v, dict):
        v = {a: b for a, b in v.items() if not isinstance(b, list)}
    if k in ("deltas",):
        print(k, len(v)); continue
    print(k, v)
