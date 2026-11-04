"""Summarise ``lp_milp_panel.py run`` JSON into the highs-parity §6 baseline table.

Usage::

    python discopt_benchmarks/scripts/panel_report.py RUN.json [RUN.json ...] [--md]

One line per (panel, arm). Definitions (``docs/dev/highs-parity-plan.md`` P0.3):

* **solved** -- every round returned the oracle's verdict: ``optimal`` on an instance with an
  oracle optimum, the oracle's ``infeasible``/``unbounded`` on an INF instance;
* **certified** -- every round ``gap_certified`` (R/H only; raw HiGHS issues no discopt
  certificate, so H0 shows ``--``);
* **total wall** -- the sum over *all* instances of the per-instance median wall (a
  solved-only total flatters the arm that times out most);
* **median (solved)** -- the median of those per-instance medians over the solved ones;
* **nodes** -- the sum over instances of the per-instance median node count;
* **incorrect** -- instances with any incorrect round (the run's oracle check);
* **timing-valid** -- instances whose load stayed at or under the run's gate; when it is
  below the instance count the walls are *indicative* (the plan's wording).

The script counts every (instance, arm) cell it summarises and exits non-zero when that
count is 0 (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _solved(cell: dict, oracle_status: str) -> bool:
    want = "optimal" if oracle_status == "Optimal" else oracle_status.lower()
    sts = [str(s).lower() for s in cell.get("status_all", [])]
    return bool(sts) and all(s == want for s in sts)


def summarise(runs: list[dict]) -> tuple[list[dict], int]:
    """``(lines, cells)``: one dict per (panel, arm) in first-seen order."""
    acc: dict[tuple[str, str], dict] = {}
    cells = 0
    for run in runs:
        if not run.get("complete", True):
            raise SystemExit("refusing an incomplete run (complete=false); finish or --resume it")
        arms = run["args"]["arms"].split(",")
        for row in run["rows"]:
            for arm in arms:
                cell = row.get(arm)
                if not cell or "status_all" not in cell:
                    raise SystemExit(f"{row['name']} [{arm}]: no per-round data ({cell})")
                cells += 1
                a = acc.setdefault(
                    (row["panel"], arm),
                    {"panel": row["panel"], "arm": arm, "n": 0, "solved": 0, "certified": 0,
                     "walls": [], "solved_walls": [], "nodes": 0, "incorrect": 0, "valid": 0},
                )  # fmt: skip
                a["n"] += 1
                ok = _solved(cell, row["oracle_status"])
                a["solved"] += ok
                a["certified"] += all(c is True for c in cell.get("certified_all", [None]))
                a["walls"].append(cell["wall_med"])
                if ok:
                    a["solved_walls"].append(cell["wall_med"])
                nodes = [x for x in cell.get("nodes_all", []) if x is not None]
                a["nodes"] += statistics.median(nodes) if nodes else 0
                a["incorrect"] += bool(cell.get("incorrect"))
                a["valid"] += bool(row["wall_valid"])
    lines = []
    for a in acc.values():
        lines.append(
            {
                "panel": a["panel"],
                "arm": a["arm"],
                "n": a["n"],
                "solved": a["solved"],
                "certified": None if a["arm"] == "H0" else a["certified"],
                "total_wall": sum(a["walls"]),
                "median_solved": statistics.median(a["solved_walls"])
                if a["solved_walls"]
                else None,
                "nodes": a["nodes"],
                "incorrect": a["incorrect"],
                "timing_valid": a["valid"],
            }
        )
    return lines, cells


def _fmt(x, nd=2):
    return "--" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--md", action="store_true", help="print a markdown table")
    a = p.parse_args(argv)
    runs = [json.loads(Path(f).read_text()) for f in a.runs]
    lines, cells = summarise(runs)
    head = ["panel", "arm", "n", "solved", "certified", "total wall s", "median (solved) s",
            "nodes", "incorrect", "timing-valid"]  # fmt: skip
    rows = [
        [ln["panel"], ln["arm"], ln["n"], ln["solved"], _fmt(ln["certified"]),
         _fmt(ln["total_wall"]), _fmt(ln["median_solved"], 3), _fmt(int(ln["nodes"])),
         ln["incorrect"], f"{ln['timing_valid']}/{ln['n']}"]
        for ln in lines
    ]  # fmt: skip
    if a.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(str(c) for c in r) + " |")
    else:
        for r in [head, *rows]:
            print("  ".join(f"{str(c):>14s}" for c in r))
    print(f"report: cells={cells} runs={len(runs)}", flush=True)
    return 0 if cells > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
