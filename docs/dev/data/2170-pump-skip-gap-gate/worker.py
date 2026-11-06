"""One (instance, method) solve via gdp_baseline.worker, plus the root pump-gate log lines."""
import json, logging, os, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
name, method, tl = sys.argv[1], sys.argv[2], float(sys.argv[3])
gate = []
class H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if m.startswith("Feasibility pump (root):"):
            gate.append(m[:200])
lg = logging.getLogger("discopt.solver"); lg.setLevel(logging.INFO); lg.addHandler(H())
lg.propagate = False
import discopt.solver as S
import gdp_baseline as gb
rec = gb.worker(name, f"discopt-{method}", tl)
rec["method"] = method
rec["has_gate"] = hasattr(S, "_root_pump_skip_decision")
rec["gate_log"] = gate
rec["load1"] = os.getloadavg()[0]
print("ROW " + json.dumps(rec, default=str), flush=True)
os._exit(0)
