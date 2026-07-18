#!/usr/bin/env python3
r"""#282 Workstream A — re-confirm the ``DISCOPT_ROOT_LP_PROBE_TIGHT`` root A/B.

Reproduces ``results/issue282/root_lp_probe_ab_20260717T183044.json``: for each of
the seven ``syn*``/``rsyn*`` panel instances, solve with the flag OFF and ON at a
fixed budget and record the **root-bound** and **dual-bound** excess vs the proven
``=opt=`` optimum. The root/dual numbers are *value-based* (load-independent): the
tightened-box probe keeps or discards the McCormick LP relaxer, a decision made at
the root and independent of wall-clock contention. They must reproduce the prior run.

    reported dual excess = |bound - opt| / max(1, |opt|) * 100   (min: opt - bound)
    root  excess         = |root_bound - opt| / max(1, |opt|) * 100

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=python \
      python discopt_benchmarks/scripts/issue282_root_lp_probe_ab.py --budget 30
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
NL_DIR = SNAPSHOT / "minlplib" / "nl"
SOLU = SNAPSHOT / "minlplib.solu"

INSTANCES = [
    "rsyn0805m",
    "rsyn0810m",
    "rsyn0815m",
    "syn15m02hfsg",
    "syn30hfsg",
    "syn40hfsg",
    "syn40m",
]

_FLAG = "DISCOPT_ROOT_LP_PROBE_TIGHT"


def load_opt(names: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(SOLU) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "=opt=" and parts[1] in names:
                with contextlib.suppress(ValueError):
                    out[parts[1]] = float(parts[2])
    return out


def _f(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if abs(xf) < 1e29 else None


def solve_one(name: str, budget: float, flag_on: bool) -> dict:
    from discopt.modeling.core import ObjectiveSense, from_nl

    if flag_on:
        os.environ[_FLAG] = "1"
    else:
        os.environ.pop(_FLAG, None)

    nl = NL_DIR / f"{name}.nl"
    model = from_nl(str(nl))
    sense = getattr(getattr(model, "_objective", None), "sense", None)
    if sense not in (ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE):
        raise RuntimeError(f"{name}: unresolved sense {sense!r}")
    is_max = sense == ObjectiveSense.MAXIMIZE

    res = model.solve(time_limit=budget, gap_tolerance=1e-4)
    return {
        "is_max": bool(is_max),
        "status": getattr(res, "status", None),
        "objective": _f(getattr(res, "objective", None)),
        "bound": _f(getattr(res, "bound", None)),
        "root_bound": _f(getattr(res, "root_bound", None)),
        "node_count": getattr(res, "node_count", None),
        "nlp_bb": getattr(res, "nlp_bb", None),
        "gap_certified": getattr(res, "gap_certified", None),
    }


def excess(bound, opt, is_max):
    """Relative % excess of a dual bound over the optimum (>=0 when sound)."""
    if bound is None or opt is None:
        return None
    raw = (bound - opt) if is_max else (opt - bound)
    return raw / max(1.0, abs(opt)) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--instances", default=",".join(INSTANCES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = [s for s in args.instances.split(",") if s.strip()]
    opts = load_opt(set(names))
    rows = []
    for name in names:
        opt = opts.get(name)
        print(f"=== {name} (opt={opt}) ===", flush=True)
        off = solve_one(name, args.budget, flag_on=False)
        on = solve_one(name, args.budget, flag_on=True)
        path = "nlp_bb" if off.get("nlp_bb") else "spatial"
        rec = {
            "name": name,
            "opt": opt,
            "path": path,
            "root_excess_off": excess(off["root_bound"], opt, off["is_max"]),
            "root_excess_on": excess(on["root_bound"], opt, on["is_max"]),
            "dual_excess_off": excess(off["bound"], opt, off["is_max"]),
            "dual_excess_on": excess(on["bound"], opt, on["is_max"]),
            "nodes_off": off["node_count"],
            "nodes_on": on["node_count"],
            "sound": True,
            "obj_off": off["objective"],
            "obj_on": on["objective"],
        }
        # Soundness: a valid dual bound never lands on the wrong side of the
        # proven optimum. For MAX it is an UPPER bound (>= opt); for MIN a LOWER
        # bound (<= opt). Unsound iff it crosses to the *incumbent* side.
        tol = 1e-4 * max(1.0, abs(opt or 0.0))
        for tag, r in (("off", off), ("on", on)):
            if r["bound"] is not None and opt is not None:
                crossed = (r["bound"] < opt - tol) if r["is_max"] else (r["bound"] > opt + tol)
                if crossed:
                    rec["sound"] = False
                    rec[f"unsound_{tag}"] = f"bound {r['bound']} crosses opt {opt}"
        rows.append(rec)
        print(
            f"    root %: {rec['root_excess_off']:.1f} -> {rec['root_excess_on']:.1f} | "
            f"dual %: {rec['dual_excess_off']:.1f} -> {rec['dual_excess_on']:.1f} | "
            f"nodes {rec['nodes_off']}->{rec['nodes_on']} | sound={rec['sound']}",
            flush=True,
        )

    out = args.out or (
        Path(__file__).resolve().parents[1]
        / "results"
        / "issue282"
        / f"root_lp_probe_ab_reconfirm_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
