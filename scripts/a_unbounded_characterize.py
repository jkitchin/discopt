#!/usr/bin/env python
"""A-UNBOUNDED (task #94) entry experiment: characterize unbounded vars + FBBT.

For nvs05/tanksize/casctanks: list which declared variables are unbounded (+-inf)
at model build, whether they appear in the objective, and whether FBBT
(``discopt.tightening.fbbt_box``) recovers finite bounds (i.e. the constraints
imply a bounded feasible region even though the declared box does not).

Reproduces docs/dev/a-unbounded-entry-2026-07-10.md sections 2-3 (FBBT column).
"""

from __future__ import annotations

import math
import os
import re

import discopt.modeling as dm
from discopt.tightening import fbbt_box

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
INSTS = ["nvs05", "tanksize", "casctanks"]
INF = 1e19


def is_inf(x: float) -> bool:
    return x is None or abs(float(x)) >= INF or math.isinf(float(x))


def main() -> None:
    for name in INSTS:
        m = dm.from_nl(NL + name + ".nl")
        vs = m._variables
        orig = [(float(v.lb), float(v.ub)) for v in vs]
        unb = [i for i, (lo, hi) in enumerate(orig) if is_inf(lo) or is_inf(hi)]

        obj = str(m._objective.expression)
        obj_vars = sorted(set(int(t[1:]) for t in re.findall(r"\bx\d+", obj)))
        obj_unb = sorted(set(unb) & set(obj_vars))

        res = fbbt_box(m, max_iter=50)
        finitized = sum(1 for i in unb if not is_inf(res.lb[i]) and not is_inf(res.ub[i]))
        print(f"\n===== {name}: {len(vs)} vars, {len(unb)} unbounded =====")
        print(f"  objective vars: {len(obj_vars)}; unbounded-in-objective: {obj_unb}")
        print(
            f"  FBBT: n_tightened={res.n_tightened}, "
            f"finitized {finitized}/{len(unb)} unbounded vars, "
            f"infeasible={res.infeasible}"
        )
        for i in unb[:6]:
            lo1, hi1 = float(res.lb[i]), float(res.ub[i])
            print(
                f"    v{i:<3} {vs[i].name:<6}: "
                f"[{orig[i][0]:.4g},{orig[i][1]:.4g}] -> [{lo1:.4g},{hi1:.4g}]"
            )


if __name__ == "__main__":
    main()
