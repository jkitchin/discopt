"""Fill the ``unproven`` rows of the #1775 GDP parity panel from the GAMS oracle.

Runs, for every :data:`gdplib_runner.GDP_PARITY` instance whose
:func:`gdplib_runner.panel_oracle` row is ``unproven`` (or every instance with
``--all``), GAMS/SCIP and GAMS/BARON on the ``gdp.bigm`` reformulation through
:func:`gdplib_runner._solve_with_gams`. That function trusts a value only when the
gap closed (``lb == ub``) and the incumbent GAMS loads back is feasible in the Pyomo
model. As in ``_attach_oracle``, the two solvers must **both** certify and
**agree**; one certifying alone, or a disagreement, leaves the row unproven and says
so. Needs GAMS with SCIP and BARON licensed (BARON demo licenses cap at 10
variables / 10 constraints and certify nothing here -- check the ``.lst``).

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
        scip = gr._solve_with_gams(spec, "bigm", args.time_limit, "scip")
        baron = gr._solve_with_gams(spec, "bigm", args.time_limit, "baron")
        wall = round(time.perf_counter() - t0, 1)
        verdict = "unproven"
        if scip is not None and baron is not None:
            tol = 1e-4 + 1e-3 * max(abs(scip), abs(baron))
            verdict = "DISAGREE" if abs(scip - baron) > tol else "certified"
        rows[name] = {"scip": scip, "baron": baron, "verdict": verdict, "wall": wall}
        print(f"{name:36s} scip={scip} baron={baron} -> {verdict} ({wall}s)", flush=True)
        certified += verdict == "certified"

    print("\n# rows for reference_optima() (BARON+SCIP agreed, both feasibility-verified):")
    for name, r in rows.items():
        if r["verdict"] == "certified":
            print(f'        "{name}": {r["baron"]!r},')
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"certified {certified} of {len(rows)}", flush=True)
    return 0 if certified else 1


if __name__ == "__main__":
    sys.exit(main())
