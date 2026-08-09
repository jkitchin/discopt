"""Verify the review's claim: does the bare-container arm take the NLP path, and
does it still return a point BELOW its lower bound on this PR's branch?

The review stated the NLP-arm value under this PR as an inference (it was read
off the unpatched tree). This measures it directly on the branch.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.modeling as dm
import discopt.solvers.lp_pounce as LPP

# §8: on the branch, with the fix present.
assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
from discopt.solvers import pounce_option_defaults
assert pounce_option_defaults()["bound_relax_factor"] == 0.0, "not the branch under test"

CHECKS = 0
seen = set()
_orig_core = LPP._solve_core
def traced_core(*a, **k):
    seen.add("lp_pounce._solve_core")
    return _orig_core(*a, **k)
LPP._solve_core = traced_core

try:
    import discopt.solvers.nlp_pounce as NLPP
    _orig_nlp = NLPP.solve_nlp
    def traced_nlp(*a, **k):
        seen.add("nlp_pounce.solve_nlp")
        return _orig_nlp(*a, **k)
    NLPP.solve_nlp = traced_nlp
except ImportError:
    pass


def run(flat):
    seen.clear()
    m = dm.Model()
    s = m.set("S", [10, 20, 30])
    y = m.continuous("y", lb=1, ub=5, over=s)
    m.minimize(dm.sum(y.flat) if flat else dm.sum(y))
    r = m.solve()
    x = np.asarray(r.value(y), dtype=np.float64).ravel()
    return r.objective, x, sorted(seen)


print(f"true optimum = 3.0;  lb = 1.0 on every variable\n")
for flat in (True, False):
    obj, x, path = run(flat)
    CHECKS += 1
    below = float(np.max(1.0 - x))          # >0 means x sits BELOW its lb
    print(f"flat={str(flat):5s} path={path}")
    print(f"    obj={obj!r}  err_vs_3.0={obj - 3.0:+.4e}")
    print(f"    x={x}")
    print(f"    max lb violation = {below:.4e}  {'<-- BELOW ITS BOUND' if below > 1e-12 else 'ok'}\n")

print(f"CHECKS_EXECUTED={CHECKS}")
sys.exit(0 if CHECKS else 1)
