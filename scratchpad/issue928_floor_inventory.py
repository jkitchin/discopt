"""What rigorous floor IS in hand when a round is cut short?

The entry probe (``issue928_round_cut_short_entry.py``) shows that a round whose
grant is already spent returns NO bound on several instances where an unclamped
round certifies one. This asks the follow-up the fix depends on: at that exit,
what sound floor did the round already own and throw away?

For each instance it re-runs the cut-short round with the build/solve seams
instrumented and reports, per round: whether the cold build was truncated, the
relaxation's ``_objective_bound_valid`` / ``_objective_floor`` (the rigorous
box-interval objective floor), the LP status, and whether ``MilpRelaxationResult``
carried a banked Neumaier-Shcherbina bound.

Prints ROUNDS_OBSERVED and exits non-zero if it observed none (CLAUDE.md §6).
Nothing is swallowed (§7).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._relax import mccormick_lp as _mc  # noqa: E402
from discopt._relax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver_tuning import SolverTuning  # noqa: E402
from discopt.solver_tuning import reset_current as _reset_tuning  # noqa: E402
from discopt.solver_tuning import set_current as _set_tuning  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "python/tests/data/minlplib_nl"
INSTANCES = (sys.argv[1] if len(sys.argv) > 1 else "bchoco06,bchoco07,bchoco08,tls2,hda").split(
    ","
)

_orig_build = _mc.build_milp_relaxation
seen: list[dict] = []


def _spy_build(*a, **kw):
    milp, varmap = _orig_build(*a, **kw)
    seen.append(
        {
            "truncated": bool(getattr(milp, "_build_truncated", False)),
            "rows_done": getattr(milp, "_build_constraints_done", None),
            "rows_total": getattr(milp, "_build_constraints_total", None),
            "obj_bound_valid": bool(milp._objective_bound_valid),
            "obj_floor": getattr(milp, "_objective_floor", None),
        }
    )
    return milp, varmap


_mc.build_milp_relaxation = _spy_build

rounds = 0
for name in INSTANCES:
    model = from_nl(str(CORPUS / f"{name}.nl"))
    lb, ub = flat_variable_bounds(model)
    for warm in (0, 1):
        os.environ["DISCOPT_LP_WARM_DEADLINE"] = str(warm)
        token = _set_tuning(SolverTuning())
        try:
            seen.clear()
            relaxer = MccormickLPRelaxer(model)
            res = relaxer.solve_at_node(
                lb, ub, time_limit=5.0, round_deadline=time.perf_counter() - 1.0
            )
        finally:
            _reset_tuning(token)
        rounds += 1
        b = seen[-1] if seen else {}
        print(
            f"{name:12s} warm={warm} status={res.status:13s} lb={res.lower_bound} | "
            f"build truncated={b.get('truncated')} rows={b.get('rows_done')}/{b.get('rows_total')} "
            f"obj_bound_valid={b.get('obj_bound_valid')} obj_floor={b.get('obj_floor')}",
            flush=True,
        )

print(f"ROUNDS_OBSERVED={rounds}")
raise SystemExit(0 if rounds else 1)
