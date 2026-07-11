import os
import sys

import numpy as np

os.environ["DISCOPT_FBBT_BEFORE_ROOT_PROBE"] = sys.argv[2] if len(sys.argv) > 2 else "0"
import discopt.modeling as dm
from discopt._jax import mccormick_lp as M

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
name = sys.argv[1]

cap = {"first": None, "n": 0}
_orig = M.MccormickLPRelaxer.solve_at_node


def tap(self, lb, ub, *a, **k):
    r = _orig(self, lb, ub, *a, **k)
    if cap["first"] is None:
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        cap["first"] = dict(
            n=len(lb),
            inf_lb=int(np.sum(~np.isfinite(lb))),
            inf_ub=int(np.sum(~np.isfinite(ub))),
            status=getattr(r, "status", None),
            lb_val=None if getattr(r, "lower_bound", None) is None else float(r.lower_bound),
        )
    cap["n"] += 1
    return r


M.MccormickLPRelaxer.solve_at_node = tap

m = dm.from_nl(NL + name + ".nl")
res = m.solve(time_limit=8)
print(name, "flag=" + os.environ["DISCOPT_FBBT_BEFORE_ROOT_PROBE"], "PROBE_BOX:", cap["first"])
