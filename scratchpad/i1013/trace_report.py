"""#1013 diagnosis: summarize the per-pivot `DUALTRACE` stream of one LP.

Answers the entry question — is the dual stall a run of near-zero steps on tiny
pivot elements, or genuine cycling? — by histogramming the chosen pivot magnitude
and the step lengths, split by whether the pivot was degenerate, and by reporting
the degenerate RUN-length distribution (the quantity the arming threshold keys on).

Prints a parsed-pivot count and exits non-zero when it is zero (CLAUDE.md §6).
"""

import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lp = sys.argv[1]
tl = sys.argv[2] if len(sys.argv) > 2 else "20"
env = dict(os.environ)
env["DISCOPT_LP_DUAL_TRACE"] = "1"
env["DISCOPT_PROFILE"] = "1"
env.setdefault("DISCOPT_LP_DUAL_STALL_HARRIS", "0")
path = os.path.join(ROOT, "scratchpad/i1013/lps", lp + ".npz")
p = subprocess.run(
    [sys.executable, "-u", os.path.join(ROOT, "scratchpad/i1013/lprun.py"), path, tl],
    capture_output=True,
    text=True,
    env=env,
)
assert p.returncode == 0, p.stderr[-3000:]

pivots = []
for ln in p.stderr.splitlines():
    if not ln.startswith("DUALTRACE "):
        continue
    d = {}
    for tok in ln.split()[1:]:
        k, _, v = tok.partition("=")
        d[k] = float(v) if ("." in v or "e" in v) else int(v)
    pivots.append(d)

res = [json.loads(ln[4:]) for ln in p.stdout.splitlines() if ln.startswith("RES ")]
print(f"lp={lp} harris={env['DISCOPT_LP_DUAL_STALL_HARRIS']} result={res[0] if res else None}")
print(f"traced pivots: {len(pivots)}")
if not pivots:
    sys.exit(1)


def q(xs, frac):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(frac * len(xs)))]


for label, sel in (("degenerate", lambda d: d["degen"]), ("productive", lambda d: not d["degen"])):
    sub = [d for d in pivots if sel(d)]
    if not sub:
        print(f"{label:11s}: none")
        continue
    piv = [d["piv"] for d in sub]
    tt = [d["t"] for d in sub]
    print(
        f"{label:11s}: n={len(sub):6d} ({100 * len(sub) / len(pivots):5.1f}%)  "
        f"|piv| p10={q(piv, 0.1):.2e} med={q(piv, 0.5):.2e} p90={q(piv, 0.9):.2e}  "
        f"|t| med={q(tt, 0.5):.2e}  tiny(|piv|<1e-4)={sum(1 for v in piv if v < 1e-4)}"
    )

# Degenerate RUN lengths: the quantity DEGEN_ARM_RUN keys on.
runs, cur = [], 0
for d in pivots:
    if d["degen"]:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
if runs:
    print(
        f"degenerate runs: n={len(runs)} max={max(runs)} med={statistics.median(runs):.0f} "
        f"mean={statistics.mean(runs):.1f}  runs>=32: {sum(1 for r in runs if r >= 32)}  "
        f"pivots in runs>=32: {sum(r for r in runs if r >= 32)}"
    )
else:
    print("degenerate runs: none")
print(
    f"flips/pivot med={q([d['flips'] for d in pivots], 0.5):.0f} ncand med={q([d['ncand'] for d in pivots], 0.5):.0f}"
)
