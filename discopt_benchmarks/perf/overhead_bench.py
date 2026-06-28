"""Per-solve orchestration-overhead micro-bench (issue #330).

On easy MINLP/MIQP instances discopt's wall time is dominated by **fixed
Python + JAX per-solve overhead**, not the Rust solver core — every instance
below certifies in ≤21 nodes with ``rust% ≈ 0``, so a faster core does nothing
for them. This micro-bench is the committed reproduction of that measurement:
it solves the flagged instances via ``from_nl(...).solve(...)``, reads the
``rust_time / jax_time / python_time`` split (plus XLA compile count and node
count) straight off the :class:`~discopt_benchmarks.perf.measure.PerfRecord`,
and prints a table with the slowdown vs SCIP.

It is *not* a CI gate (the slow-search panel + soundness tripwire in
``perf/gate.py`` own that). It exists so the overhead regime stays measurable
and a before/after table can be regenerated on demand::

    python -m discopt_benchmarks.perf.overhead_bench
    python -m discopt_benchmarks.perf.overhead_bench --json out.jsonl

The SCIP wall times are the reference numbers reported in issue #330 (SCIP
10.0.2). They are baked in as a reference column so the slowdown ratio is
reproducible without a SCIP install; pass ``--no-scip`` to drop the column.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from discopt_benchmarks.perf.measure import measure_solve
from discopt_benchmarks.perf.panel import DATA_DIR


@dataclass(frozen=True)
class OverheadInstance:
    """One overhead-regime instance: a ≤21-node MINLP/MIQP whose wall time is
    fixed Python/JAX overhead. ``scip_wall`` is the SCIP 10.0.2 reference from
    issue #330 (seconds)."""

    name: str
    time_limit: float
    scip_wall: float


# The instance set from issue #330 — each certifies in ≤21 nodes with rust ≈ 0%,
# so the wall time is fixed Python + JAX per-solve overhead. SCIP walls are the
# issue's measured reference (SCIP 10.0.2).
INSTANCES: list[OverheadInstance] = [
    OverheadInstance("ex1221", 60, 0.09),
    OverheadInstance("ex1226", 60, 0.04),
    OverheadInstance("nvs03", 60, 0.03),
    OverheadInstance("nvs06", 60, 0.04),
    OverheadInstance("alan", 60, 0.09),
    OverheadInstance("dispatch", 60, 0.04),
    OverheadInstance("ex1225", 60, 0.03),
]


def scip_wall(nl_path: str, time_limit: float, gap_tolerance: float = 1e-4) -> float | None:
    """Solve a ``.nl`` instance with SCIP (PySCIPOpt) and return its wall time, or
    ``None`` when PySCIPOpt is not installed. Lets the slowdown ratio be measured
    on the *same* machine instead of relying on the baked-in #330 reference."""
    try:
        from pyscipopt import Model as _ScipModel
    except Exception:
        return None
    import time

    m = _ScipModel()
    m.hideOutput()
    m.readProblem(nl_path)
    m.setRealParam("limits/gap", gap_tolerance)
    m.setRealParam("limits/time", time_limit)
    t0 = time.perf_counter()
    m.optimize()
    return time.perf_counter() - t0


def run(gap_tolerance: float = 1e-4, live_scip: bool = False) -> list[dict]:
    """Solve every overhead instance and return one result dict per instance.

    When ``live_scip`` is set and PySCIPOpt is installed, the SCIP wall is
    measured on this machine; otherwise the #330 reference (SCIP 10.0.2) is used.
    """
    # Warm the JAX/XLA stack once on a throwaway solve so the first measured
    # instance is not charged the one-time import + cold-compile cost (which
    # otherwise shows up as a ~15 s outlier on whichever instance runs first).
    _first = next(
        (i for i in INSTANCES if os.path.exists(os.path.join(DATA_DIR, f"{i.name}.nl"))), None
    )
    if _first is not None:
        measure_solve(
            os.path.join(DATA_DIR, f"{_first.name}.nl"),
            time_limit=_first.time_limit,
            gap_tolerance=gap_tolerance,
        )

    rows: list[dict] = []
    for inst in INSTANCES:
        path = os.path.join(DATA_DIR, f"{inst.name}.nl")
        if not os.path.exists(path):
            continue
        rec = measure_solve(path, time_limit=inst.time_limit, gap_tolerance=gap_tolerance)
        wall = rec.wall_time or 0.0
        row = rec.to_json()
        _scip = None
        if live_scip:
            _scip = scip_wall(path, inst.time_limit, gap_tolerance)
        row["scip_wall"] = inst.scip_wall if _scip is None else _scip
        row["scip_source"] = "reference" if _scip is None else "live"
        row["rust_pct"] = 100.0 * rec.rust_time / wall if wall else 0.0
        row["jax_pct"] = 100.0 * rec.jax_time / wall if wall else 0.0
        row["python_pct"] = 100.0 * rec.python_time / wall if wall else 0.0
        row["slowdown_vs_scip"] = wall / row["scip_wall"] if row["scip_wall"] else None
        rows.append(row)
    return rows


def format_table(rows: list[dict], with_scip: bool = True) -> str:
    """Render the rows as the fixed-width before/after table used in the PR."""
    head = (
        f"{'instance':10} {'wall(s)':>8} {'rust%':>6} {'jax%':>6} {'py%':>6} "
        f"{'nodes':>6} {'compiles':>9}"
    )
    if with_scip:
        head += f" {'scip(s)':>8} {'slowdown':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        line = (
            f"{r['instance']:10} {r['wall_time']:8.3f} {r['rust_pct']:6.1f} "
            f"{r['jax_pct']:6.1f} {r['python_pct']:6.1f} {r['node_count']:6d} "
            f"{r['xla_compile_count']:9d}"
        )
        if with_scip:
            sd = r.get("slowdown_vs_scip")
            line += f" {r['scip_wall']:8.3f} {('' if sd is None else f'{sd:.1f}x'):>9}"
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=1e-4, help="relative gap tolerance")
    ap.add_argument("--json", type=str, default=None, help="write per-instance records as JSONL")
    ap.add_argument("--no-scip", action="store_true", help="drop the SCIP reference column")
    ap.add_argument(
        "--live-scip",
        action="store_true",
        help="measure SCIP wall on this machine via PySCIPOpt (else use #330 reference)",
    )
    args = ap.parse_args()

    rows = run(gap_tolerance=args.gap, live_scip=args.live_scip)
    print(format_table(rows, with_scip=not args.no_scip))
    if rows and any(r.get("scip_source") == "live" for r in rows):
        print("(SCIP wall measured live on this machine)")

    if args.json:
        with open(args.json, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} records to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
