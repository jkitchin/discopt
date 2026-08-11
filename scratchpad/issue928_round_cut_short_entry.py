"""Entry experiment (CLAUDE.md §4): what does a round cut short return?

§14b named the residual as an *interaction*: "LP yields on its deadline" x "round
yields on its deadline" compounding into a node result that carries no adoptable
bound. The corpus cell that exposed it (contvar @20 s) is wall-clock-shaped and
does not reproduce on every container, so this probes the mechanism DIRECTLY and
deterministically: hand ``solve_at_node`` a ``round_deadline`` that is already
spent, or nearly spent (the states the #966 clamp produces once the cold build
has eaten the round's grant), and record what comes back in each flag regime
against an unclamped control.

Hypothesis: a round cut short can return NO bound at all — either because the
starved LP yields before it can bank a dual (``time_limit``, the #928 arm) or
because the truncated build leaves a relaxation whose optimum no certification
route will certify (``uncertified``/``numerical``) — while the same box under an
unclamped round certifies a finite bound. Any such cell is a bound the round
threw away, and the analogue of what §14a did for an LP cut short is to floor
the round on a rigorous bound it already has.

Kill criterion: if every clamped cell returns a finite bound wherever the control
does, the hypothesis is wrong and the loss is elsewhere.

Prints CELLS_EXECUTED and exits non-zero if it measured nothing (§6). No
exception is swallowed (§7).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._relax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver_tuning import SolverTuning  # noqa: E402
from discopt.solver_tuning import reset_current as _reset_tuning  # noqa: E402
from discopt.solver_tuning import set_current as _set_tuning  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert "round_deadline" in MccormickLPRelaxer.solve_at_node.__code__.co_varnames

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "python/tests/data/minlplib_nl"
BINDING = (
    "4stufen,bchoco06,bchoco07,bchoco08,beuster,casctanks,clay0303hfsg,contvar,hda,"
    "heatexch_gen1,heatexch_gen2,heatexch_gen3,nvs05,syn05hfsg,tls2,tspn05,tspn08,"
    "tspn10,tspn12"
)
INSTANCES = ((sys.argv[1] if len(sys.argv) > 1 else "") or BINDING).split(",")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# (label, how many seconds of the round's grant are left when the round starts)
ROUNDS = (("ctrl", None), ("spent", -1.0), ("tight", 0.05))

cells = 0
rows: list[dict] = []
print(f"loadavg={[round(x, 2) for x in os.getloadavg()]}", flush=True)
for name in INSTANCES:
    model = from_nl(str(CORPUS / f"{name}.nl"))
    lb, ub = flat_variable_bounds(model)
    for warm in (0, 1):
        os.environ["DISCOPT_LP_WARM_DEADLINE"] = str(warm)
        for label, left in ROUNDS:
            token = _set_tuning(SolverTuning())
            try:
                relaxer = MccormickLPRelaxer(model)
                t = time.perf_counter()
                res = relaxer.solve_at_node(
                    lb,
                    ub,
                    time_limit=5.0,
                    round_deadline=None if left is None else time.perf_counter() + left,
                )
                wall = time.perf_counter() - t
            finally:
                _reset_tuning(token)
            cells += 1
            rows.append(
                {
                    "instance": name,
                    "warm": warm,
                    "round": label,
                    "status": res.status,
                    "lb": res.lower_bound,
                    "wall": wall,
                }
            )
            print(
                f"{name:14s} warm={warm} {label:5s} status={res.status:14s} "
                f"lb={res.lower_bound}  wall={wall:.2f}s",
                flush=True,
            )

ctrl = {(r["instance"], r["warm"]): r for r in rows if r["round"] == "ctrl"}
lost = [
    r
    for r in rows
    if r["round"] != "ctrl"
    and r["lb"] is None
    and any(
        c["lb"] is not None
        for c in rows
        if c["instance"] == r["instance"] and c["warm"] == r["warm"] and c["round"] == "ctrl"
    )
]
print(f"\nLOST_BOUND_CELLS={len(lost)}")
for r in lost:
    print(f"  {r['instance']:14s} warm={r['warm']} {r['round']:5s} status={r['status']}")

# Soundness control, counted (§6). A cut-short round's floor is a lower bound on
# the SAME box as the unclamped control's certified LP bound, and the LP feasible
# region is a subset of the column box, so the cut-short value must never exceed
# the control's. A violation would mean the truncated round reports a bound the
# full round disproves — the failure class this whole change must not create.
checked = 0
violations = []
for r in rows:
    if r["round"] == "ctrl" or r["lb"] is None:
        continue
    c = ctrl[(r["instance"], r["warm"])]
    if c["lb"] is None:
        continue
    checked += 1
    if r["lb"] > c["lb"] + 1e-6 * max(1.0, abs(c["lb"])):
        violations.append((r, c))
print(f"SOUNDNESS_COMPARISONS_EXECUTED={checked}  violations={len(violations)}")
for r, c in violations:
    print(f"  VIOLATION {r['instance']} warm={r['warm']} {r['round']}: {r['lb']} > ctrl {c['lb']}")
print(f"CELLS_EXECUTED={cells}")
if violations or not checked:
    raise SystemExit(2)
if OUT is not None:
    OUT.write_text(json.dumps(rows, indent=2))
raise SystemExit(0 if cells else 1)
