"""#1013: the no-progress window distribution, measured by the ENGINE's own test.

`pinf_report.py` reconstructed the progress test in Python and got it wrong: it
summed raw bound violations while the engine's `select_leaving` only counts
violations above `tol`, so sub-tolerance noise read as progress and the windows
came out far too short (`QPLIB_3871`: 24 by the probe, thousands by the engine).
This reads the engine's own `noprog=` counter out of the `DUALTRACE` stream, so
the threshold is chosen against the quantity the code actually tests.

Prints a parsed-pivot count and exits non-zero at zero (CLAUDE.md §6).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
total = 0
print(f"{'lp':26s} {'pivots':>7s} {'max_noprog':>10s} {'max_run':>8s}  status")
for lp in sys.argv[1:]:
    env = dict(
        os.environ, DISCOPT_LP_DUAL_TRACE="1", DISCOPT_PROFILE="1", DISCOPT_LP_DUAL_STALL_BAIL="0"
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
    noprog, run, n = 0, 0, 0
    for ln in p.stderr.splitlines():
        if not ln.startswith("DUALTRACE "):
            continue
        d = dict(tok.split("=") for tok in ln.split()[1:])
        noprog = max(noprog, int(d["noprog"]))
        run = max(run, int(d["run"]))
        n += 1
    res = [ln for ln in p.stdout.splitlines() if ln.startswith("RES ")]
    status = res[0].split('"status": "')[1].split('"')[0] if res else "?"
    total += n
    print(f"{lp:26s} {n:7d} {noprog:10d} {run:8d}  {status}")
print("traced pivots:", total)
if total == 0:
    sys.exit(1)
