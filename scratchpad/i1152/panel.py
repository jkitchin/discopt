"""#1152 §5 differential panel — root-setup budget/build deadline, ON vs OFF.

Runs the in-repo MINLPLib corpus at each requested ``time_limit`` with the flag
OFF and ON **interleaved** (per instance, per rep) so a load excursion hits both
arms, and writes one JSON record per (instance, T, arm, rep).

Checks are applied by ``panel_report.py``; this script only measures. It prints
each record as it lands (§10) and exits non-zero if it measured nothing (§6).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

CORPUS = "python/tests/data/minlplib_nl"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limits", default="5")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    # The flag is read from the environment at ``SolverTuning`` construction, so it
    # is set per solve through the typed tuning context instead (same precedence the
    # solver documents: ``tuning=`` kwarg > context > env).
    from discopt import solver_tuning
    from discopt.modeling.core import ObjectiveSense, from_nl

    names = sorted(f[:-3] for f in os.listdir(CORPUS) if f.endswith(".nl"))
    if args.only:
        want = set(args.only.split(","))
        names = [n for n in names if n in want]
    tls = [float(t) for t in args.time_limits.split(",")]

    n_rec = 0
    with open(args.out, "w") as fh:
        print(f"# marker=i1152-panel instances={len(names)} tls={tls} reps={args.reps}", flush=True)
        for rep in range(args.reps):
            for name in names:
                for T in tls:
                    for arm in ("off", "on"):
                        tuning = dataclasses.replace(
                            solver_tuning.SolverTuning(),
                            root_setup_build_deadline=(arm == "on"),
                        )
                        m = from_nl(os.path.join(CORPUS, name + ".nl"))
                        sense = (
                            "max"
                            if m._objective is not None
                            and m._objective.sense == ObjectiveSense.MAXIMIZE
                            else "min"
                        )
                        t0 = time.perf_counter()
                        r = m.solve(time_limit=T, tuning=tuning)
                        wall = time.perf_counter() - t0
                        rec = {
                            "instance": name,
                            "time_limit": T,
                            "arm": arm,
                            "rep": rep,
                            "sense": sense,
                            "wall": wall,
                            "ratio": wall / T,
                            "status": r.status,
                            "objective": r.objective,
                            "bound": r.bound,
                            "nodes": int(r.node_count),
                            "gap_certified": bool(getattr(r, "gap_certified", False)),
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_rec += 1
                        print(
                            f"{name:22s} T={T:5.1f} {arm:3s} rep={rep} wall={wall:7.2f} "
                            f"ratio={wall / T:5.2f} {r.status:11s} nodes={r.node_count:6d} "
                            f"bound={r.bound} obj={r.objective}",
                            flush=True,
                        )
    print(f"# records={n_rec}", flush=True)
    return 0 if n_rec else 1


if __name__ == "__main__":
    sys.exit(main())
