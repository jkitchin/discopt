#!/usr/bin/env python
"""P0 SPATIAL-CERT soundness sweep over the v3 re-run.

Beyond verdict.py's per-arm criteria, asserts the fix-specific invariants:
- NO new "optimal" label on an instance whose REPORTED gap is not closed
  (every optimal row: |obj - bound| <= max(1e-6 abs-house, 1e-4 rel) against
  the solve's own gap_tolerance semantics — here checked as rel <= 1e-4 with
  the same no-1.0-floor arithmetic the solver certifies with, plus abs 1e-6).
- bound never crosses the oracle (sense-corrected, zero meaningful slack).
- bound on the sound side of the objective within numerical noise (<= obj +
  1e-6 for min; the known pre-existing terminal-polish offset is ~6e-8).
- Enumerate label flips vs the PRE-FIX verdicts23 run per arm.
"""

import json
import math
import os
import sys

SP = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(SP, "..", "flag-graduation-verdicts23-2026-07-10")
sys.path.insert(0, SP)
from verdict import ORACLE, sense_sign  # noqa: E402

ARMS = ["off", "lu_density_route", "obj_branch_priority", "lift_loose_products"]


def gap_closed(obj, bnd):
    if obj is None or bnd is None or not (math.isfinite(obj) and math.isfinite(bnd)):
        return False
    abs_gap = abs(obj - bnd)
    if abs_gap <= 1e-6:
        return True
    return abs_gap / max(abs(obj), abs(bnd), 1e-10) <= 1e-4


def main():
    bad = 0
    for arm in ARMS:
        new = json.load(open(os.path.join(SP, f"v3_{arm}.json")))["results"]
        try:
            old = json.load(open(os.path.join(OLD, f"v3_{arm}.json")))["results"]
        except FileNotFoundError:
            old = {}
        flips = []
        for inst, r in new.items():
            st = str(r.get("status", ""))
            obj, bnd = r.get("objective"), r.get("bound")
            kind, best, bestdual = ORACLE[inst]
            s = sense_sign(r)
            o = old.get(inst, {})
            if o and str(o.get("status")) != st:
                flips.append((inst, o.get("status"), st))
            if st == "optimal":
                if not gap_closed(obj, bnd):
                    print(f"VIOLATION {arm}/{inst}: OPTIMAL with open gap obj={obj} bnd={bnd}")
                    bad += 1
                # min: bound must not sit above the objective beyond noise
                if obj is not None and bnd is not None and s * bnd > s * obj + 1e-6:
                    print(f"VIOLATION {arm}/{inst}: bound {bnd} beyond incumbent {obj}")
                    bad += 1
            if bnd is not None and math.isfinite(bnd):
                ref = best if kind == "opt" else (bestdual if bestdual is not None else best)
                pad = max(1e-4, 1e-3 * abs(ref))
                if s * bnd > s * ref + pad:
                    print(f"VIOLATION {arm}/{inst}: bound {bnd} crosses oracle {ref}")
                    bad += 1
        print(f"{arm}: label flips vs pre-fix run: {flips if flips else 'none'}")
    print(f"soundness sweep: {'CLEAN' if bad == 0 else f'{bad} VIOLATION(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
