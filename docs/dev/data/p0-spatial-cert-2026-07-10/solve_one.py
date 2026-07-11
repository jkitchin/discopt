#!/usr/bin/env python
"""Solve ONE instance in an isolated process; print a RESULT json line.
Env flags are read fresh at import (subprocess), so the parent sets e.g.
DISCOPT_NODE_REDUCE=1 before invoking this."""
import json, os, sys, time

CORPUS = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


def main():
    inst = sys.argv[1]
    tl = float(sys.argv[2])
    try:
        from discopt.modeling.core import from_nl
        m = from_nl(os.path.join(CORPUS, inst + ".nl"))
        sense = "min"
        try:
            s = m._objective.sense
            sense = "min" if str(s).lower().endswith("minimize") else "max"
        except Exception:
            pass
        t = time.time()
        r = m.solve(time_limit=tl, gap_tolerance=1e-4)
        wall = time.time() - t
        obj = getattr(r, "objective", None)
        bnd = getattr(r, "bound", None)
        out = {
            "instance": inst,
            "status": str(r.status),
            "objective": None if obj is None else float(obj),
            "bound": None if bnd is None else float(bnd),
            "node_count": int(getattr(r, "node_count", -1) or -1),
            "wall": round(wall, 3),
            "sense": sense,
        }
    except Exception as e:
        out = {"instance": inst, "status": "ERROR", "error": repr(e)[:400]}
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
