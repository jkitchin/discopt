"""#945: which of the two options costs nvs05 its incumbent?

Replication found nvs05's incumbent reproducibly worse post-#945 (8.7320 ->
12.5895, zero within-arm spread over 3 reps, both arms `feasible` at a 20 s
budget). nvs05 goes through the nonconvex B&B path, whose primal heuristics call
`solve_nlp` DIRECTLY — so the incumbent options (which only reach
`_solve_continuous`, OA and GDPopt-LOA) cannot be responsible, and the suspect is
`constr_viol_tol`, which IS a backend-wide default.

Four arms, interleaved, so the answer is attributed rather than assumed:

  main   print_level only                  (Ipopt: cvt 1e-4, brf 1e-8)
  cvt    constr_viol_tol=1e-8 only         (brf still Ipopt's)
  inc    incumbent options only            (cvt still Ipopt's)
  new    what ships

Kill criterion: if `cvt` matches `main`, constr_viol_tol is innocent and the
suspicion is wrong.

RESULT: it was guilty — `main` == `inc` at 8.7320 and `cvt` == `new` at 12.5895,
zero within-arm spread. `nvs05_feasibility.py` then falsified the follow-up
assumption that the cheaper incumbent was bought by a violated row: both points
are genuinely feasible (worst row 1.8e-12 / 3.6e-12, box 0, integrality 0), so it
was a real solution lost, not a false one rejected. #945 therefore does NOT route
constr_viol_tol to the NLP backend. Re-running this probe against the shipped tree
now shows all four arms identical at 8.7320 — the `cvt` arm has become a no-op
there, which is the fix, not a broken probe.

§6: prints an executed-run count and exits non-zero if it is zero.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as SOLVER  # noqa: E402
import discopt.solvers.gdpopt_loa as LOA  # noqa: E402
import discopt.solvers.nlp_pounce as NLPP  # noqa: E402
import discopt.solvers.oa as OA  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

_REAL = pounce_option_defaults
_REAL_INCUMBENT = SOLVER.pounce_incumbent_options
_CONSUMERS = (SOLVER, OA, LOA)

ARMS = {
    "main": ({"print_level": 0}, {}),
    "cvt": ({"print_level": 0, "constr_viol_tol": 1e-8}, {}),
    "inc": ({"print_level": 0}, None),
    "new": (None, None),
}


def set_arm(arm):
    backend, incumbent = ARMS[arm]
    NLPP.pounce_option_defaults = _REAL if backend is None else (lambda: dict(backend))
    for mod in _CONSUMERS:
        mod.pounce_incumbent_options = (
            _REAL_INCUMBENT if incumbent is None else (lambda: dict(incumbent))
        )
        if hasattr(mod, "pounce_option_defaults"):
            mod.pounce_option_defaults = _REAL if backend is None else (lambda: dict(backend))


PATH = "python/tests/data/minlplib_nl/nvs05.nl"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

rows = {a: [] for a in ARMS}
runs = 0
for _ in range(REPS):
    for arm in ARMS:  # interleaved, not four sequential blocks (§9)
        set_arm(arm)
        t0 = time.perf_counter()
        r = from_nl(PATH).solve(time_limit=TL)
        rows[arm].append((r.status, r.objective, r.bound, r.node_count))
        runs += 1
        print(
            f"  {arm:5s} status={r.status:11s} obj={r.objective!r} bound={r.bound!r} "
            f"n={r.node_count} wall={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
set_arm("new")

print(f"\n{'arm':6s} {'statuses':16s} {'objective range':>28s} {'bound range':>28s}")
print("-" * 82)
for arm, rs in rows.items():
    st = ",".join(sorted({x[0] for x in rs}))
    o = [x[1] for x in rs if x[1] is not None]
    b = [x[2] for x in rs if x[2] is not None]
    fo = f"{min(o):.10g}..{max(o):.10g}" if o else "—"
    fb = f"{min(b):.10g}..{max(b):.10g}" if b else "—"
    print(f"{arm:6s} {st:16s} {fo:>28s} {fb:>28s}")

print(f"\nEXECUTED_RUNS={runs}  reps={REPS}  time_limit={TL}")
if runs == 0:
    print("PROBE RAN NOTHING", file=sys.stderr)
    sys.exit(2)
