"""#966: is the cheap incremental node path in scope on the instances the
round-budget flag hurts?  (Entry experiment for "a declined round banks a floor":
the incremental patch+warm-start IS the cheap bound producer a declined round
would fall back to, so its availability decides whether that fallback exists.)

Counted output; exits non-zero if it examined nothing (CLAUDE.md §6).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from discopt._relax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

names = sys.argv[1:]
examined = 0
for name in names:
    path = f"python/tests/data/minlplib_nl/{name}.nl"
    m = from_nl(path)
    t0 = time.perf_counter()
    r = MccormickLPRelaxer(m)
    t_build = time.perf_counter() - t0
    lb, ub = flat_variable_bounds(m)
    inc_ok = r._inc is not None
    fast = None
    t_fast = None
    if inc_ok:
        t1 = time.perf_counter()
        fast = r._try_incremental_node(lb, ub, None)
        t_fast = time.perf_counter() - t1
    t2 = time.perf_counter()
    res = r.solve_at_node(lb, ub, time_limit=10.0)
    t_round = time.perf_counter() - t2
    examined += 1
    print(
        f"{name:16s} inc={inc_ok!s:5s} fast={'hit' if fast is not None else 'miss':4s} "
        f"t_fast={t_fast if t_fast is None else round(t_fast, 4)} "
        f"round={t_round:.3f}s status={res.status} lb={res.lower_bound} "
        f"(relaxer ctor {t_build:.2f}s)",
        flush=True,
    )

print(f"INSTANCES_EXAMINED={examined}")
if examined == 0:
    sys.exit(2)
