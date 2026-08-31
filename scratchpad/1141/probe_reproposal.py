"""#1141 item 4, at scale: does our master re-propose assignments it was told to cut?

tls2 through lp_nlp_bb: the HiGHS master needs 13 integer-feasible points to reach
optimal 5.3; the in-house master burns 1477 and never gets an incumbent. Same NLP
layer, same cut logic, so the difference is in what the master proposes.

The issue's item 4 reported "7 of 172 assignments re-proposed, one 6 times". This
measures the same thing on the in-house master and adds the check that matters:
for every REPEAT, was the cut returned the first time actually VIOLATED at the
re-proposed point? A cut that is violated there and re-proposed anyway means the
master is not honouring rows it was handed -- a defect in the driver, not the
separator. A cut that is satisfied means the separator handed back a row that
does not exclude its own assignment -- a defect in the separator.

Anti-vacuity (§6): prints the number of separations seen and exits non-zero if
zero, so "0 repeats" can never read as a pass when nothing was observed.
"""
import collections
import os
import sys

import numpy as np

os.environ.setdefault("DISCOPT_CONVEX_MINLP_ROUTE", "0")
import discopt.solvers.milp_simplex as ms  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "tls2"
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
_orig = ms.solve_milp_with_lazy_cuts
seen: dict[tuple, list] = {}
order: list[tuple] = []
stats = {"separations": 0, "repeats": 0, "cut_violated_at_repeat": 0,
         "cut_satisfied_at_repeat": 0, "no_cut_returned": 0}


def wrapper(*a, **kw):
    inner = kw["lazy_callback"]
    integrality = np.asarray(kw.get("integrality"), int)
    int_idx = np.flatnonzero(integrality != 0)

    def spy(x):
        stats["separations"] += 1
        key = tuple(int(round(v)) for v in np.asarray(x, float)[int_idx])
        rows = inner(x)
        rows = [] if rows is None else [(np.asarray(c, float), float(r)) for c, r in rows]
        if key in seen:
            stats["repeats"] += 1
            # Did the FIRST cut for this assignment exclude this very point?
            prev = seen[key]
            if not prev:
                stats["no_cut_returned"] += 1
            else:
                xa = np.asarray(x, float)
                worst = max(float(c[: len(xa)] @ xa - r) for c, r in prev)
                if worst > 1e-6:
                    stats["cut_violated_at_repeat"] += 1
                else:
                    stats["cut_satisfied_at_repeat"] += 1
        else:
            seen[key] = rows
            order.append(key)
        return rows

    kw["lazy_callback"] = spy
    return _orig(*a, **kw)


ms.solve_milp_with_lazy_cuts = wrapper
m = from_nl(f"python/tests/data/minlplib_nl/{NAME}.nl")
r = m.solve(time_limit=TL, solver="mip-nlp", mip_nlp_method="lp_nlp_bb",
            milp_solver="simplex")
print(f"{NAME}: status={r.status} obj={r.objective!r} bound={r.bound!r}")
print(f"separations           : {stats['separations']}")
print(f"distinct assignments  : {len(seen)}")
print(f"re-proposed           : {stats['repeats']}")
print(f"  ...where the earlier cut IS violated at the repeat : "
      f"{stats['cut_violated_at_repeat']}  <-- driver ignored a row it was given")
print(f"  ...where the earlier cut is NOT violated           : "
      f"{stats['cut_satisfied_at_repeat']}  <-- separator's row did not exclude it")
print(f"  ...where no cut was returned at all                : {stats['no_cut_returned']}")
counts = collections.Counter()
for k in order:
    counts[k] = 0
top = sorted(((v, k) for k, v in collections.Counter(
    [k for k in order]).items()), reverse=True)[:1]
if stats["separations"] == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
