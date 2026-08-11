"""#966 entry experiment (CLAUDE.md §4): can an EAGER (no-jit) sparse-Hessian
evaluation bound the severe mode the first-time XLA compile causes?

Hypothesis: evaluating the colored-HVP Lagrangian Hessian with ``jax.disable_jit()``
avoids the uninterruptible whole-module XLA optimization (measured 124 s on
heatexch_gen3 in this container, 46-186 s in the F4 table) at an eager per-call
cost small enough to run several NLP iterations inside a 20 s budget.

Kill criterion: eager per-call wall > 5 s on either instance -> the fallback is
useless; the fix must instead be an entry-gate skip.

Prints executed-comparison counts (§6); exits non-zero if nothing was measured.
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import discopt  # noqa: E402
from discopt._relax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

measured = 0
for name in sys.argv[1:] or ["heatexch_gen1"]:
    m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
    ev = NLPEvaluator(m)
    n = ev.n_variables
    mcon = ev.n_constraints
    use_sparse = ev._use_sparse_hessian()
    print(f"{name}: n={n} m={mcon} sparse_hess_path={use_sparse}", flush=True)
    if not use_sparse:
        print(f"  {name}: not on the sparse compressed-HVP path; skipping", flush=True)
        continue
    x0 = np.zeros(n)
    lam = np.zeros(mcon)
    walls = []
    for k in range(3):
        t0 = time.perf_counter()
        with jax.disable_jit():
            vals = ev.evaluate_hessian_values(x0, 1.0, lam)
        walls.append(time.perf_counter() - t0)
        measured += 1
    # evaluate_hessian_values marks _hessian_compiled; reset honesty for print
    print(
        f"  eager walls: {['%.2f' % w for w in walls]} s  nnz={len(vals)} "
        f"(first call includes tracing; later calls are the steady state)",
        flush=True,
    )

print(f"executed-comparison count: {measured}")
if measured == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
