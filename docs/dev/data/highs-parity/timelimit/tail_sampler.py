import sys, time, os, threading, traceback, collections
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import lp_milp_loader as L
from discopt.modeling.core import from_nl
import tempfile
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); mdl = from_nl(nl)
t0 = time.perf_counter(); samples = collections.Counter(); done = threading.Event(); n = [0]
me = threading.get_ident()
def sampler():
    while not done.is_set():
        time.sleep(0.25)
        if time.perf_counter() - t0 < 20.6: continue
        for tid, fr in sys._current_frames().items():
            if tid == threading.get_ident() or tid == me and False: continue
            st = traceback.extract_stack(fr)
            key = " <- ".join(f"{os.path.basename(f.filename)}:{f.name}:{f.lineno}" for f in reversed(st[-7:]))
            if "sampler" in key or "wait" in key.split(" <- ")[0]: continue
            samples[key] += 1; n[0] += 1
th = threading.Thread(target=sampler, daemon=True); th.start()
r = mdl.solve(time_limit=20.0)
done.set(); end = time.perf_counter() - t0
print("end", round(end, 2), r.status, "samples", n[0])
if end > 22:
    for k, c in samples.most_common(6): print(f"  {c:3d} {k[:600]}")
