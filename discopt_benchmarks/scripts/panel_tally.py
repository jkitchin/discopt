"""Tally a head-to-head panel JSON into the table the manuscript cites.

Why this exists
---------------
The manuscript's head-to-head paragraph carried a `TODO(#466)` and a set of
round numbers ("BARON and SCIP each certify on the order of 180 of the ~200
instances, while discopt certifies roughly 120") that matched no artifact in
the repository. Round numbers in a paper are read as measurements, so they need
a file behind them.

This script is that file's reader. It takes a panel JSON written by
`global_opt_baron_vs_discopt.py` or `global_opt_nl_solvers.py` and re-derives
every verdict by calling the harness's own `classify` -- it never reads the
stored `verdict` field. That is deliberate: a stored verdict is frozen at the
rule that existed when the panel ran, so a later correctness fix to `classify`
would leave the paper quoting superseded numbers. Re-deriving means the table
always reflects the current rule, and a rule change shows up as a diff here.

Two distinctions the table keeps separate, because conflating them is what
produced the unsupported claim in the first place:

  certified  -- the solver ASSERTED a certified global (`_claims_global`).
                This is a claim about the solver's own certificate.
  ok         -- the incumbent MATCHES the published optimum within tolerance.
                This is a claim about the answer, and says nothing about
                whether the solver proved it.

A solver can be `ok` without being certified (it found the right answer but
did not prove it) and certified without being `ok` (which is a VIOLATION).

Timing columns follow CLAUDE.md's rule for the 3-way comparison: median over
SOLVED instances *and* total wall over ALL instances, because a solved-only
statistic flatters whichever solver times out most -- its slowest instances
leave the population.

Usage
-----
    python -m discopt_benchmarks.scripts.panel_tally <panel.json> [--org]

`--org` emits an org-mode table for direct inclusion in the manuscript.
Prints an executed-comparison count and exits non-zero if it is zero (CLAUDE.md
section 6).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.global_opt_baron_vs_discopt import (  # noqa: E402
    GAP,
    NA,
    OK,
    VIOLATION,
    _claims_global,
    _no_solution_returned,
    classify,
)


def _runs_by_solver(row: dict[str, Any], solvers: list[str]) -> dict[str, dict]:
    """Both panel layouts: the N-solver ``runs`` dict and the flat 2-solver form."""
    if isinstance(row.get("runs"), dict):
        return {s: (row["runs"].get(s) or {}) for s in solvers}
    return {s: (row.get(s) or {}) for s in solvers}


def _solver_names(panel: dict[str, Any], rows: list[dict]) -> list[str]:
    if isinstance(panel.get("solvers"), list):
        return list(panel["solvers"])
    # flat layout: any top-level key whose value is a run dict carrying a status
    names = []
    for k, v in (rows[0] if rows else {}).items():
        if isinstance(v, dict) and "status" in v:
            names.append(k)
    return names


def tally(path: Path) -> tuple[dict[str, dict], int, dict]:
    panel = json.loads(path.read_text())
    rows = panel["rows"] if isinstance(panel, dict) and "rows" in panel else panel
    meta = {
        "file": path.name,
        "time_limit": panel.get("time_limit") if isinstance(panel, dict) else None,
        "timestamp": panel.get("timestamp") if isinstance(panel, dict) else None,
        "n_rows": len(rows),
    }
    solvers = _solver_names(panel if isinstance(panel, dict) else {}, rows)

    out: dict[str, dict] = {}
    compared = 0
    for s in solvers:
        counts = {OK: 0, GAP: 0, VIOLATION: 0, NA: 0}
        certified = 0
        solved_times: list[float] = []
        total_wall = 0.0
        flagged: list[tuple[str, str]] = []
        for r in rows:
            run = _runs_by_solver(r, solvers)[s]
            status = run.get("status", "")
            known = r.get("known")
            # Same placeholder rule the parser now applies, so a panel written
            # BEFORE that fix is re-scored under the corrected rule rather than
            # reproducing its false violations.
            obj = None if _no_solution_returned(status) else run.get("objective")
            verdict = classify(status, obj, known, bool(r.get("maximize")), run.get("lower_bound"))
            counts[verdict] += 1
            compared += 1
            if _claims_global(status):
                certified += 1
            wall = run.get("wall_time") or 0.0
            total_wall += wall
            if verdict == OK:
                solved_times.append(wall)
            if verdict == VIOLATION:
                flagged.append((r["instance"], status))
        out[s] = {
            "certified": certified,
            "counts": counts,
            "median_solved_s": statistics.median(solved_times) if solved_times else None,
            "total_wall_s": total_wall,
            "violations": flagged,
        }
    return out, compared, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel", type=Path)
    ap.add_argument("--org", action="store_true", help="emit an org-mode table")
    args = ap.parse_args()

    res, compared, meta = tally(args.panel)

    print(f"panel     : {meta['file']}")
    print(f"instances : {meta['n_rows']}")
    print(f"time limit: {meta['time_limit']} s")
    print()

    if args.org:
        print(
            "| solver | certified | matches oracle | gap | violations | n/a | "
            "median solved (s) | total wall (s) |"
        )
        print(
            "|--------+-----------+----------------+-----+------------+-----+"
            "-------------------+----------------|"
        )
        for s, d in res.items():
            c = d["counts"]
            med = f"{d['median_solved_s']:.2f}" if d["median_solved_s"] is not None else "--"
            print(
                f"| {s} | {d['certified']} | {c[OK]} | {c[GAP]} | {c[VIOLATION]} | "
                f"{c[NA]} | {med} | {d['total_wall_s']:.1f} |"
            )
    else:
        hdr = (
            f"{'solver':<10} {'cert':>5} {'ok':>5} {'GAP':>5} {'VIOL':>5} {'n/a':>5} "
            f"{'med(s)':>8} {'wall(s)':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for s, d in res.items():
            c = d["counts"]
            med = f"{d['median_solved_s']:.2f}" if d["median_solved_s"] is not None else "--"
            print(
                f"{s:<10} {d['certified']:>5} {c[OK]:>5} {c[GAP]:>5} {c[VIOLATION]:>5} "
                f"{c[NA]:>5} {med:>8} {d['total_wall_s']:>9.1f}"
            )
        for s, d in res.items():
            for inst, status in d["violations"]:
                print(f"  VIOLATION  {s:<10} {inst:<28} {status}")

    print(f"\nEXECUTED COMPARISONS: {compared}")
    if compared == 0:
        print("PANEL TALLIED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
