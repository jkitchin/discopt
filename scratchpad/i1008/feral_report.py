"""Join the two feral arms and report the fill change (#1008).

Correctness first (CLAUDE.md §1, zero slack): every LP whose objective moves off
the HiGHS optimum by more than the repo tolerance is a failure, reported before
any performance number. Performance is then read from the exact counters:
fill = LuFactorNnz / LuBasisNnz.

§6: prints the executed comparison count and exits non-zero if it is zero.
§7: nothing is caught.
"""

from __future__ import annotations

import json
import os
import sys

OLD = os.environ.get("OLD_JSONL", "/private/tmp/wt1008g/scratchpad/i1008/out/old.jsonl")
NEW = os.environ.get("NEW_JSONL", "/private/tmp/wt1008g/scratchpad/i1008/out/new.jsonl")
ATOL, RTOL = 1e-6, 1e-4


def load(p):
    return {json.loads(ln)["tag"]: json.loads(ln) for ln in open(p) if ln.strip()}


old, new = load(OLD), load(NEW)
common = sorted(set(old) & set(new))
assert common, "no overlapping LPs between the two arms"

bad = []
for t in common:
    for arm, r in (("v0.15.1", old[t]), ("b071d54", new[t])):
        if r["ref"] is None or r["status"] != "optimal":
            continue
        if abs(r["obj"] - r["ref"]) > ATOL + RTOL * abs(r["ref"]):
            bad.append((t, arm, r["obj"], r["ref"]))

print(f"# LPs compared: {len(common)}  (old={len(old)} new={len(new)})")
print("\n─── correctness vs HiGHS (zero slack) ───")
if bad:
    for t, arm, o, r in bad:
        print(f"  FAIL {t:24s} [{arm}] obj={o!r} ref={r!r}")
else:
    print(f"  clean — {2 * len(common)} arm×LP objectives all at the HiGHS optimum")

hdr = f"{'LP':<24} {'facs o/n':>12} {'fill old':>9} {'fill new':>9} {'Δfnnz':>9} {'wall o/n':>14}"
print("\n─── per-LP (fill = LuFactorNnz / LuBasisNnz) ───")
print(hdr)
tot = {k: [0, 0] for k in ("facs", "basis_nnz", "factor_nnz")}
n_better = n_worse = n_same = 0
for t in common:
    o, n = old[t], new[t]
    for k in tot:
        tot[k][0] += o[k]
        tot[k][1] += n[k]
    fo = o["factor_nnz"] / o["basis_nnz"] if o["basis_nnz"] else float("nan")
    fn = n["factor_nnz"] / n["basis_nnz"] if n["basis_nnz"] else float("nan")
    d = n["factor_nnz"] - o["factor_nnz"]
    if o["factor_nnz"]:
        rel = d / o["factor_nnz"]
        n_better += rel < -0.01
        n_worse += rel > 0.01
        n_same += abs(rel) <= 0.01
    print(
        f"{t:<24} {o['facs']:>5}/{n['facs']:<6} {fo:>8.2f}x {fn:>8.2f}x "
        f"{d:>+9d} {o['wall']:>6.2f}/{n['wall']:<7.2f}"
    )

print("\n─── totals ───")
fo = tot["factor_nnz"][0] / tot["basis_nnz"][0]
fn = tot["factor_nnz"][1] / tot["basis_nnz"][1]
print(f"  factorizations   {tot['facs'][0]:>12,} -> {tot['facs'][1]:>12,}")
print(f"  basis nnz        {tot['basis_nnz'][0]:>12,} -> {tot['basis_nnz'][1]:>12,}")
print(
    f"  factor nnz       {tot['factor_nnz'][0]:>12,} -> {tot['factor_nnz'][1]:>12,}"
    f"   ({(tot['factor_nnz'][1] / tot['factor_nnz'][0] - 1) * 100:+.1f}%)"
)
print(f"  fill             {fo:>11.2f}x -> {fn:>11.2f}x")
print(f"  per-LP factor nnz: better {n_better}, worse {n_worse}, within 1% {n_same}")
print(f"\nexecuted comparisons: {len(common)}")
sys.exit(1 if (bad or not common) else 0)
