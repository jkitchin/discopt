"""#1013: does total primal infeasibility make progress during a degenerate run?

The dual loop's two progress measures are the dual objective (flat on a
degenerate pivot) and the primal infeasibility it drives to zero. If a long
degenerate run still reduces `pinf`, the loop is converging and must not be
interrupted; if `pinf` is flat too, the loop is making no progress in EITHER
measure — the stall. This reads the `DUALTRACE` stream and reports, per LP, the
longest window of pivots without a `pinf` improvement.

Prints a parsed-pivot count and exits non-zero at zero (CLAUDE.md §6).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
total = 0
for lp in sys.argv[1:]:
    env = dict(
        os.environ, DISCOPT_LP_DUAL_TRACE="1", DISCOPT_PROFILE="1", DISCOPT_LP_DUAL_STALL_MODE="off"
    )
    p = subprocess.run(
        [
            sys.executable,
            "-u",
            os.path.join(ROOT, "scratchpad/i1013/lprun.py"),
            os.path.join(ROOT, "scratchpad/i1013/lps", lp + ".npz"),
            "30",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    piv = []
    for ln in p.stderr.splitlines():
        if ln.startswith("DUALTRACE "):
            piv.append(dict(tok.split("=") for tok in ln.split()[1:]))
    status = [ln for ln in p.stdout.splitlines() if ln.startswith("RES ")][0][4:]
    total += len(piv)
    if not piv:
        print(f"{lp}: no pivots traced")
        continue
    best = float("inf")
    since = 0
    worst_gap = 0
    gap_at = 0
    run = 0
    worst_run = 0
    for k, d in enumerate(piv):
        v = float(d["pinf"])
        if v < best * (1 - 1e-9) - 1e-12:
            best = v
            since = 0
        else:
            since += 1
            if since > worst_gap:
                worst_gap, gap_at = since, k
        run = run + 1 if d["degen"] == "1" else 0
        worst_run = max(worst_run, run)
    print(
        f"{lp:26s} pivots={len(piv):6d} longest degenerate run={worst_run:6d} "
        f"longest no-pinf-progress window={worst_gap:6d} (ends at pivot {gap_at}) "
        f"final_pinf={float(piv[-1]['pinf']):.3e}"
    )
print("traced pivots:", total)
if total == 0:
    sys.exit(1)
