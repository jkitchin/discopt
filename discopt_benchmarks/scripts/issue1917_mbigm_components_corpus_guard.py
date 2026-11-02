#!/usr/bin/env python3
"""#1917 V7: the in-repo ``.nl`` corpus cannot see ``DISCOPT_MBIGM_LP_COMPONENTS``.

The pre-registration (``docs/dev/data/1917-flag-panel/
DISCOPT_MBIGM_LP_COMPONENTS.prereg.md``) leaves the corpus out of the panel because
the flag acts only inside ``_compute_big_m_lp``, which only ``reformulate_gdp``'s
mbigm branches call. This script checks that claim instead of assuming it:

1. Every ``python/tests/data/minlplib_nl/*.nl`` parses, and none carries a
   constraint ``reformulate_gdp`` would lower (``solver._model_has_gdp_constraints``).
   The script fails if it parsed no file or examined no constraint.
2. ``--n-solve`` of them (the smallest files, deterministic) are solved with the
   flag ON and ``gdp_method="mbigm"`` while a counter wraps ``_compute_big_m_lp``.
   It must record 0 calls.
3. Positive control: one ``gdp_parity`` instance solved the same way must record
   more than 0 calls. Otherwise the counter cannot see a call, and steps 2 and 3
   prove nothing.

Exit 0 when every check passes, 1 on a failed check, 2 when nothing executed.

    DISCOPT_MBIGM_LP_COMPONENTS=1 python -u \\
        discopt_benchmarks/scripts/issue1917_mbigm_components_corpus_guard.py \\
        --out <V7>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "discopt_benchmarks"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-solve", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--control", default="pyomo.eight_process")
    args = ap.parse_args(argv)

    assert os.environ.get("DISCOPT_MBIGM_LP_COMPONENTS") == "1", "run with the flag ON"
    import discopt
    import discopt._relax.gdp_reformulate as G  # noqa: N812
    from discopt.modeling.core import from_nl
    from discopt.solver import _model_has_gdp_constraints

    assert hasattr(G, "_component_sub_lp"), "flag code absent"
    assert discopt.__file__.startswith(str(REPO)), discopt.__file__

    calls = [0]
    real = G._compute_big_m_lp

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    G._compute_big_m_lp = counted
    out: dict = {"discopt": discopt.__file__, "failures": []}

    # 1. parse every corpus file; none may carry a GDP constraint.
    files = sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl"))
    parsed = cons = 0
    gdp = []
    for p in files:
        m = from_nl(str(p))
        parsed += 1
        cons += len(m._constraints)
        if _model_has_gdp_constraints(m):
            gdp.append(p.name)
    out.update(parsed=parsed, constraints=cons, with_gdp=gdp)
    print(f"parsed {parsed} .nl files, {cons} constraints, {len(gdp)} with GDP", flush=True)
    if parsed == 0 or cons == 0:
        out["failures"].append("parsed no file or no constraint")
    if gdp:
        out["failures"].append(f"GDP constraints in {gdp}")

    # 2. solve the smallest few under mbigm, flag ON: 0 calls.
    small = sorted(files, key=lambda p: (p.stat().st_size, p.name))[: args.n_solve]
    solves = []
    for p in small:
        before = calls[0]
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = from_nl(str(p)).solve(gdp_method="mbigm", time_limit=args.time_limit)
        n = calls[0] - before
        solves.append(
            {
                "instance": p.name,
                "status": str(r.status),
                "calls": n,
                "wall": round(time.perf_counter() - t0, 2),
            }
        )
        print(f"corpus {p.name:28s} {r.status!s:12s} _compute_big_m_lp calls={n}", flush=True)
        if n:
            out["failures"].append(f"{p.name}: {n} _compute_big_m_lp calls")
    out["corpus_solves"] = solves
    if not solves:
        out["failures"].append("no corpus solve executed")

    # 3. positive control: a gdp_parity instance under mbigm must call it.
    from discopt.pyomo.gdp import convert_pyomo_gdp

    from benchmarks import gdplib_runner as gr

    specs = [s for s in gr.discover_models(include=[args.control]) if s.name == args.control]
    assert len(specs) == 1, (args.control, [s.name for s in specs])
    before = calls[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = convert_pyomo_gdp(specs[0].builder()).model.solve(
            gdp_method="mbigm", time_limit=args.time_limit
        )
    n = calls[0] - before
    out["control"] = {"instance": args.control, "status": str(r.status), "calls": n}
    print(f"control {args.control} {r.status!s} _compute_big_m_lp calls={n}", flush=True)
    if n == 0:
        out["failures"].append(f"positive control {args.control}: 0 calls")

    executed = parsed + len(solves) + 1
    out["executed"] = executed
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    verdict = "PASS" if not out["failures"] else "FAIL: " + "; ".join(out["failures"])
    print(f"executed {executed} checks; V7 {verdict}", flush=True)
    sys.stdout.flush()
    os._exit(2 if executed == 0 else (1 if out["failures"] else 0))


if __name__ == "__main__":
    main()
