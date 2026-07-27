#!/usr/bin/env python3
"""Second-pass soundness scan: enumerate #779-guard-caught FALSE PRIMALs.

The main discopt-vs-SCIP sweep's VIOLATION classifier only sees the *returned*
incumbent; when the #779 verification guard catches an infeasible incumbent it
withholds it (returns obj=None), so the sweep records no violation even though an
unsound presolve mutation / heuristic fired (see #815, emfl050_3_3). This pass
re-runs a target set in a fresh subprocess per instance, captures **stderr**, and
records any ``FALSE PRIMAL`` / ``unsound`` / ``decertif`` emission — the guard's
tell.

Reads target instance names from the sweep JSONL (default: every flagged
instance) and writes results to a JSONL. Each instance runs with a short solve
time limit and a hard outer wall cap so #814-style overruns cannot stall the scan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BIG_NL = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark" / "minlplib" / "nl"

WORKER = r"""
import sys, discopt.modeling as dm
nl, tl = sys.argv[1], float(sys.argv[2])
try:
    r = dm.from_nl(nl).solve(time_limit=tl)
    print(f"STATUS {r.status} OBJ {r.objective}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"EXC {type(e).__name__}: {e}", flush=True)
"""

TELLS = ("FALSE PRIMAL", "unsound", "decertif", "INFEASIBLE in the original")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, required=True, help="sweep results JSONL")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--hard-cap", type=float, default=40.0, help="outer wall kill per instance")
    ap.add_argument("--only-flagged", action="store_true", default=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.sweep)]
    # Target: heuristic-heavy = any flagged instance (slowdown/violation/etc.) plus
    # any non-optimal terminal status (feasible/time_limit/unknown) where the primal
    # path ran hardest.
    targets: list[str] = []
    seen: set[str] = set()
    for r in rows:
        nm = r["name"]
        if nm in seen:
            continue
        flagged = bool(r["flags"])
        nonopt = r["discopt"]["status"] not in ("optimal", "infeasible")
        if flagged or nonopt:
            seen.add(nm)
            targets.append(nm)

    print(f"# false-primal scan: {len(targets)} target instances "
          f"(tl={args.time_limit}s, hard-cap={args.hard_cap}s)", flush=True)
    out = open(args.out, "w")
    hits = []
    for i, nm in enumerate(targets, 1):
        nl = str(BIG_NL / f"{nm}.nl")
        if not Path(nl).exists():
            continue
        try:
            proc = subprocess.run(
                [sys.executable, "-c", WORKER, nl, str(args.time_limit)],
                capture_output=True, text=True, timeout=args.hard_cap,
            )
            err = proc.stdout + proc.stderr
            killed = False
        except subprocess.TimeoutExpired as e:
            err = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
            # TimeoutExpired.stdout/stderr may be bytes
            if isinstance(e.stdout, (bytes, bytearray)):
                err = (e.stdout or b"").decode("utf-8", "replace") + \
                      (e.stderr or b"").decode("utf-8", "replace")
            killed = True
        tell = next((t for t in TELLS if t in err), None)
        rec = {"name": nm, "false_primal": tell is not None, "tell": tell,
               "killed_at_cap": killed}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        mark = f"  <<FALSE_PRIMAL ({tell})" if tell else ("  (cap)" if killed else "")
        print(f"  [{i}/{len(targets)}] {nm:24}{mark}", flush=True)
        if tell:
            hits.append(nm)

    out.close()
    print("\n================ FALSE-PRIMAL SCAN SUMMARY ================", flush=True)
    print(f"targets scanned: {len(targets)}", flush=True)
    print(f"FALSE PRIMAL emissions (guard-caught, incumbent withheld): {len(hits)}", flush=True)
    for nm in hits:
        print(f"  {nm}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
