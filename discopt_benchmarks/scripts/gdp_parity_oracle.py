"""Fill the ``unproven`` rows of the #1775 GDP parity panel from the GAMS oracle.

Runs, for every :data:`gdplib_runner.GDP_PARITY` instance whose
:func:`gdplib_runner.panel_oracle` row is ``unproven`` (or every instance with
``--all``), GAMS/SCIP and GAMS/BARON on the ``gdp.bigm`` reformulation through
:func:`gdplib_runner._solve_with_gams`. That function trusts a value only when the
gap closed (``lb == ub``) and the incumbent GAMS loads back is feasible in the Pyomo
model, with its integer variables rounded (#1802 S2). As in ``_attach_oracle``, two
certifying solvers must **agree** (a disagreement leaves the row unproven); one
solver certifying while the other *declines* (an open gap, a point infeasible at
its integral rounding, a GAMS write failure -- each printed) certifies the row with
that solver alone as its source, as ``batch_processing`` already is. Needs GAMS
with SCIP and BARON licensed; a BARON demo license (10 variables / 10 constraints)
is detected in the ``.lst`` and declines.

Prints one line per instance (flushed) and the rows to paste into
``reference_optima()``; exits 2 when GAMS is unavailable and 1 when nothing was
certified.

    python -u discopt_benchmarks/scripts/gdp_parity_oracle.py --time-limit 3600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    import pyomo.environ as pyo

    from benchmarks import gdplib_runner as gr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--all", action="store_true", help="re-certify every panel instance")
    ap.add_argument("--include", default="", help="comma-separated instance names")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if not pyo.SolverFactory("gams").available(exception_flag=False):
        print("GAMS is not available; this oracle needs GAMS with SCIP and BARON.", flush=True)
        return 2

    table = gr.panel_oracle()
    if args.include:
        names = [n for n in args.include.split(",") if n]
    elif args.all:
        names = list(gr.GDP_PARITY)
    else:
        names = [n for n, rec in table.items() if rec.source == "unproven"]
    specs = {s.name: s for s in gr.discover_models(include=names)}

    rows, certified = {}, 0
    for name in names:
        spec = specs.get(name)
        if spec is None:
            print(f"{name:36s} NOT BUILDABLE here (missing builder or data)", flush=True)
            continue
        t0 = time.perf_counter()
        why: list[str] = []
        scip = gr._solve_with_gams(spec, "bigm", args.time_limit, "scip", why=why)
        baron = gr._solve_with_gams(spec, "bigm", args.time_limit, "baron", why=why)
        wall = round(time.perf_counter() - t0, 1)
        value, source = None, "unproven"
        if scip is not None and baron is not None:
            tol = 1e-4 + 1e-3 * max(abs(scip), abs(baron))
            value, source = (None, "DISAGREE") if abs(scip - baron) > tol else (baron, "baron+scip")
        elif baron is not None:
            value, source = baron, "baron"
        elif scip is not None:
            value, source = scip, "scip"
        rows[name] = {
            "scip": scip,
            "baron": baron,
            "value": value,
            "source": source,
            "declines": why,
            "wall": wall,
            "time_limit": args.time_limit,
            "method": "bigm",
            "date": time.strftime("%Y-%m-%d"),
        }
        print(f"{name:36s} scip={scip} baron={baron} -> {source} ({wall}s)", flush=True)
        for reason in why:
            print(f"{'':36s}   {reason}", flush=True)
        certified += value is not None

    print("\n# rows for _GAMS_OPTIMA (gap closed, point feasible at its integral rounding):")
    for name, r in rows.items():
        if r["value"] is not None:
            print(f'    "{name}": ({r["value"]!r}, "{r["source"]}"),')
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"certified {certified} of {len(rows)}", flush=True)
    return 0 if certified else 1


if __name__ == "__main__":
    sys.exit(main())
