"""hda @ time_limit=10 under the GRADUATED defaults — does the #928 xfail retire?

``test_time_limit_contract.py::[hda-10.0]`` is xfail(strict=False) against #928: the
#138 fallback's separated-relaxation phase ignored the budget it was handed, giving a
bimodal 10.6-13.0 s against a 12.5 s allowance. With DISCOPT_LP_WARM_DEADLINE now
default-ON the callee honours it. Measured here with the env UNSET (i.e. the shipped
default, not an opt-in), 3 reps, and the legacy path (=0) interleaved as the control.

Counted per §6: prints REPS_EXECUTED and exits non-zero if zero.
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = ROOT / "python/tests/data/minlplib_nl/hda.nl"
LIMIT = 10.0
ALLOWED = LIMIT * 1.10 + 1.5  # the contract test's own slack

RUNNER = """
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
from discopt.modeling.core import from_nl
from discopt._relax.milp_relaxation import _lp_warm_deadline_enabled
m = from_nl(sys.argv[1]); t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2])); wall = time.perf_counter() - t0
print(json.dumps({"wall": wall, "bound": r.bound, "status": r.status,
                  "flag": _lp_warm_deadline_enabled()}))
"""

rows = {"default": [], "legacy_optout": []}
executed = 0
for rep in range(3):
    for arm, env in (
        ("default", {}),
        ("legacy_optout", {"DISCOPT_LP_WARM_DEADLINE": "0", "DISCOPT_NODE_ROUND_BUDGET": "0"}),
    ):
        e = {**os.environ, **env}
        e.pop("DISCOPT_LP_WARM_DEADLINE", None) if not env else None
        p = subprocess.run(
            [sys.executable, "-u", "-c", RUNNER, str(NL), str(LIMIT)],
            capture_output=True,
            text=True,
            env=e,
            timeout=900,
        )
        if p.returncode != 0:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-4000:])
            raise SystemExit(f"run failed: {arm}")
        rec = json.loads(p.stdout.strip().splitlines()[-1])
        rows[arm].append(rec)
        executed += 1
        print(
            f"rep{rep + 1} {arm:14s} wall={rec['wall']:6.2f}s flag={rec['flag']} "
            f"bound={rec['bound']} status={rec['status']}",
            flush=True,
        )

for arm, recs in rows.items():
    w = [r["wall"] for r in recs]
    sd = statistics.stdev(w) if len(w) > 1 else 0.0
    print(
        f"{arm:14s} wall {statistics.fmean(w):.2f} +/- {sd:.2f}s  "
        f"allowed {ALLOWED:.2f}s  within={all(x <= ALLOWED for x in w)}"
    )
Path("scratchpad/issue928_grad_hda10.json").write_text(json.dumps(rows, indent=2))
print(f"REPS_EXECUTED={executed}")
sys.exit(0 if executed else 1)
