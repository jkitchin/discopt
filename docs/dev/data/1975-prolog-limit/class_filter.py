"""#1975: rebuild #1869's panel set: <=200 vars/cons, an unbounded continuous var after FBBT."""
import json, sys, os, signal
import numpy as np
import discopt.modeling as dm
from discopt.tightening import fbbt_box

D = sys.argv[1]
out = {}
def alarm(*a): raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
names = open(os.path.join(D, "..", "small.txt")).read().split()
for k, name in enumerate(names):
    signal.alarm(20)
    try:
        m = dm.from_nl(os.path.join(D, name + ".nl"))
        bt = fbbt_box(m)
        lb, ub = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
        cont = []
        off = 0
        for v in m._variables:
            sz = int(np.prod(v.shape)) if v.shape else 1
            if v.var_type.name.lower().startswith("cont"):
                cont.extend(range(off, off + sz))
            off += sz
        assert off == len(lb), (off, len(lb))
        nopen = int(sum(1 for j in cont if not (np.isfinite(lb[j]) and np.isfinite(ub[j]))))
        out[name] = {"n": len(lb), "open_cont": nopen}
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        signal.alarm(0)
    if k % 100 == 0:
        print(k, name, out[name], flush=True)
json.dump(out, open(os.path.join(D, "..", "class_filter.json"), "w"), indent=0)
sel = sorted(n for n, r in out.items() if r.get("open_cont", 0) > 0)
err = [n for n, r in out.items() if "error" in r]
print("checked", len(out), "open-continuous", len(sel), "errors", len(err))
open(os.path.join(D, "..", "class.txt"), "w").write("\n".join(sel))
assert len(out) == len(names)
