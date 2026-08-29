import os, sys
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
sys.path.insert(0, "python/tests")
from support import direct_testfuncs as tfs
from discopt.solvers.direct import _DirectSearch
n = 0
for name in ("six_hump_camel","branin","hartman_3","ackley_2","goldstein_price"):
    tf = tfs.get(name)
    s = _DirectSearch(tf.lb, tf.ub); hist=[]
    s.run(lambda x: (float(tf.np_body(x)), 0.0), 4000,
          on_iteration=lambda st: hist.append((st.stats.evals, st.best_feasible_value)))
    assert hist, f"{name}: DIRECT probe recorded no iterations"
    e = next((e for e,v in hist if v is not None and tf.relative_error(v) <= 1e-2), None)
    print(f"{name:<18}{e}"); n += 1
print(f"executed probes: {n}")
sys.exit(0 if n else 1)
