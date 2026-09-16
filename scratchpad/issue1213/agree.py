#!/usr/bin/env python3
"""Cross-implementation agreement check (issue #1213): for every corpus .nl both
parsers accept, evaluate the objective at the SAME point and compare, and compare
the variable bounds. Divergence here would mean one of the two decoders is
mis-reading the format.

Point: x_j = clamp(0.5, x_l_j, x_u_j), taken from pounce's bounds; discopt's own
bounds are compared separately so the point is not silently different.

Prints an executed-comparison count; exits non-zero if zero, or on disagreement.
"""

import json
import math
import pathlib
import subprocess
import sys
import tempfile

import pounce

CORPUS = pathlib.Path(sys.argv[1])
BIN = "target/release/examples/nl_audit"
# Both readers use a finite "infinity" sentinel, but NOT the same one:
# pounce reports 1e19 (the Ipopt nlp_*_bound_inf convention), discopt 1e20.
# Treat anything at or beyond the smaller of the two as infinite on both sides.
INF = 1e19

cmp_obj = cmp_bnd = 0
bad = []
skipped = []

for f in sorted(CORPUS.glob("*.nl")):
    try:
        p = pounce.read_nl(str(f))
    except BaseException as e:
        skipped.append((f.stem, f"pounce: {type(e).__name__}"))
        continue
    xl, xu = [float(v) for v in p.x_l], [float(v) for v in p.x_u]
    x = [min(max(0.5, lo), hi) for lo, hi in zip(xl, xu)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(" ".join(repr(float(v)) for v in x))
        xfile = fh.name
    r = subprocess.run([BIN, "--eval", str(f), xfile], capture_output=True, text=True)
    if r.returncode != 0:
        skipped.append((f.stem, "discopt: " + r.stderr.strip().splitlines()[-1][:120]))
        continue
    d = json.loads(r.stdout)

    # bounds
    for j, (lo, hi, dlo, dhi) in enumerate(zip(xl, xu, d["lo"], d["hi"])):
        cmp_bnd += 1

        def same(a, b):
            if a <= -INF and b <= -INF:
                return True
            if a >= INF and b >= INF:
                return True
            return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))

        if not (same(lo, dlo) and same(hi, dhi)):
            bad.append((f.stem, f"bound var{j}: pounce=[{lo},{hi}] discopt=[{dlo},{dhi}]"))
            break

    # objective. pounce reports the minimize-sense objective and carries the
    # constant separately in obj_constant only when the reader splits it out.
    po = float(p.objective(x))
    do = d["obj"]
    if do is None:
        do = float("nan")
    cmp_obj += 1
    if math.isnan(po) and math.isnan(do):
        continue
    if not d["minimize"]:
        po = -po  # pounce normalizes to minimize; discopt keeps the source sense
    if not (abs(po - do) <= 1e-7 * max(1.0, abs(po), abs(do))):
        bad.append((f.stem, f"objective: pounce={po!r} discopt={do!r}"))

print(f"objective comparisons: {cmp_obj}")
print(f"bound comparisons    : {cmp_bnd}")
print(f"skipped              : {len(skipped)} {skipped}")
if bad:
    print(f"\nDISAGREEMENTS ({len(bad)}):")
    for n, m in bad:
        print(f"  {n}: {m}")
if cmp_obj == 0:
    sys.exit("PROBE DID NOT FIRE: zero objective comparisons")
sys.exit(1 if bad else 0)
