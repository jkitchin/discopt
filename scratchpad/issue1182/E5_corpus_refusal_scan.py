"""#1182 entry experiment, part 4: how often does the refusal class occur on a REAL corpus?

E3 found a class where BOTH classical lowerings refuse and the Theorem-1
lowering certifies: a disjunct row whose body has an **unbounded interval
enclosure** (so discopt's big-M pass raises ``cannot bound the body ... from
above``) and which is **not finite at the origin** (so the
Furman-Sawaya-Grossmann perspective raises ``HullPerspectiveOriginError``).

That fixture is synthetic. Per the #727 RLT lesson a mechanism validated only on
a synthetic proxy can be a no-op on the real class, so this probe measures how
often each refusal condition actually holds on the **GDPlib** corpus -- the
reference GDP library discopt already benchmarks against
(``benchmarks.gdplib_runner``).

The two conditions are evaluated on the Pyomo source models with Pyomo's own
interval propagation (``compute_bounds_on_expr``) and an origin evaluation, which
are the same two questions discopt's two lowerings ask. Refusal counts are
reported per model and per condition, never summed into one "hard" number.

Prints the executed-assertion count and exits non-zero if it is zero.
"""

from __future__ import annotations

import importlib
import math
import sys
import traceback

MODELS = [
    "batch_processing", "biofuel", "cstr", "disease_model", "ex1_linan_2023",
    "gdp_col", "hda", "jobshop", "kaibel", "med_term_purchasing", "methanol",
    "mod_hens", "modprodnet", "positioning", "small_batch", "spectralog",
    "stranded_gas", "syngas", "water_network",
]


def _origin_value(body, variables):
    """Evaluate ``body`` with every variable at 0. Non-finite => hull refuses."""
    from pyomo.environ import value

    saved = [(v, v.value, v.fixed) for v in variables]
    try:
        for v in variables:
            v.fixed = False
            v.set_value(0.0, skip_validation=True)
        try:
            val = value(body)
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
        return val if isinstance(val, (int, float)) and math.isfinite(val) else None
    finally:
        for v, val0, fixed0 in saved:
            v.set_value(val0, skip_validation=True)
            v.fixed = fixed0


def scan_model(name):
    from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.environ import Block, Constraint
    from pyomo.gdp import Disjunct

    mod = importlib.import_module(f"gdplib.{name}")
    build = getattr(mod, "build_model", None) or getattr(mod, "build_" + name, None)
    if build is None:
        cands = [a for a in dir(mod) if a.startswith("build")]
        if not cands:
            raise AttributeError(f"gdplib.{name} exposes no build* entry point")
        build = getattr(mod, cands[0])
    m = build()

    rows = unbounded = origin_bad = both = 0
    for d in m.component_data_objects(Disjunct, active=True, descend_into=(Block, Disjunct)):
        for c in d.component_data_objects(Constraint, active=True, descend_into=True):
            rows += 1
            body = c.body
            variables = list(identify_variables(body, include_fixed=False))
            try:
                lo, hi = compute_bounds_on_expr(body)
            except Exception:
                lo, hi = None, None
            unb = (
                lo is None or hi is None
                or not math.isfinite(lo) or not math.isfinite(hi)
            )
            org = _origin_value(body, variables)
            if unb:
                unbounded += 1
            if org is None:
                origin_bad += 1
            if unb and org is None:
                both += 1
    return rows, unbounded, origin_bad, both


def main() -> int:
    assertions = 0
    table = []
    for name in MODELS:
        try:
            rows, unb, org, both = scan_model(name)
        except Exception as exc:
            print(f"[{name}] SKIPPED {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            traceback.print_exc(limit=1)
            continue
        assertions += rows
        table.append((name, rows, unb, org, both))
        print(
            f"[{name}] disjunct_rows={rows} unbounded_enclosure={unb} "
            f"nonfinite_at_origin={org} both={both}",
            flush=True,
        )

    print("\n=== summary (disjunct rows by refusal condition) ===")
    hdr = ("model", "rows", "big-m refuses", "hull refuses", "both refuse")
    print(" | ".join(f"{h:>22}" for h in hdr))
    tot = [0, 0, 0, 0]
    for name, rows, unb, org, both in table:
        print(" | ".join(f"{c:>22}" for c in (name, str(rows), str(unb), str(org), str(both))))
        tot[0] += rows
        tot[1] += unb
        tot[2] += org
        tot[3] += both
    print(" | ".join(f"{c:>22}" for c in ("TOTAL", *map(str, tot))))

    print(f"\nexecuted assertions (disjunct rows examined): {assertions}")
    if assertions == 0:
        print("FAIL: the probe measured nothing (CLAUDE.md section 6)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
