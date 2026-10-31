"""#2062: OA cut-mask diff and classification time over the in-repo MINLPLib corpus.

For every ``.nl`` in ``python/tests/data/minlplib_nl/`` runs
``classify_oa_cut_convexity`` with ``DISCOPT_OA_CONVEXITY_TAYLOR`` = 0 and = 1,
interleaved (0, 1, 0, 1), and reports rows whose mask entry changes and the
per-instance classification time (mean of the two runs per arm). Exits non-zero if
no instance was classified (CLAUDE.md §6) or if any row is True OFF and False ON
(the flag only adds a pre-test, so it can never lose a certificate).

Usage: python -u corpus_scan.py OUT.jsonl
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import discopt._relax.convexity.certificate as cert
from discopt._relax.convexity.rules import classify_oa_cut_convexity
from discopt.modeling.core import from_nl

FLAG = "DISCOPT_OA_CONVEXITY_TAYLOR"


def main() -> None:
    assert "_certify_univariate_taylor" in open(cert.__file__).read(), cert.__file__
    print("loaded", cert.__file__, flush=True)
    root = os.path.join(os.path.dirname(__file__), "../../../../python/tests/data/minlplib_nl")
    paths = sorted(glob.glob(os.path.join(root, "*.nl")))
    out = open(sys.argv[1], "w")
    classified = gained = lost = 0
    tot = {"0": 0.0, "1": 0.0}
    for p in paths:
        name = os.path.basename(p)[:-3]
        masks: dict = {}
        times: dict = {"0": [], "1": []}
        err = None
        for arm in ("0", "1", "0", "1"):
            os.environ[FLAG] = arm
            try:
                m = from_nl(p)
            except Exception as e:  # recorded: a model the reader refuses is not classified
                err = f"from_nl: {type(e).__name__}: {e}"
                break
            t0 = time.perf_counter()
            r = classify_oa_cut_convexity(m)
            times[arm].append(time.perf_counter() - t0)
            prev = masks.setdefault(arm, list(r.constraint_mask))
            assert prev == list(r.constraint_mask), f"{name}: nondeterministic mask"
        if err is not None:
            print(f"{name}: SKIPPED ({err})", flush=True)
            out.write(json.dumps({"model": name, "error": err}) + "\n")
            continue
        classified += 1
        off, on = masks["0"], masks["1"]
        g = sum(1 for a, b in zip(off, on) if b and not a)
        lo = sum(1 for a, b in zip(off, on) if a and not b)
        gained += g
        lost += lo
        t_off = sum(times["0"]) / 2
        t_on = sum(times["1"]) / 2
        tot["0"] += t_off
        tot["1"] += t_on
        row = {
            "model": name,
            "rows": len(off),
            "gained": g,
            "lost": lo,
            "t_off": t_off,
            "t_on": t_on,
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(
            f"{name}: rows={len(off)} gained={g} lost={lo} t_off={t_off:.3f}s t_on={t_on:.3f}s",
            flush=True,
        )
    print(
        f"classified {classified}/{len(paths)}; rows gained {gained}, lost {lost}; "
        f"time {tot['0']:.2f} -> {tot['1']:.2f} s",
        flush=True,
    )
    if classified == 0 or lost:
        sys.exit(1)


if __name__ == "__main__":
    main()
