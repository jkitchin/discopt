"""cert:P3.1d — native {0,½}-CG (zero-half) separator: ON vs OFF measurement.

Measures the lever the zero-half separator is built to move (certification-gap-
plan.md §7 "Phase 3 1d"): the ROOT dual bound and root gap closed with
``DISCOPT_ZEROHALF`` ON vs OFF on the graphpart panel, routed through the
cut-enabled ``_solve_milp_bb`` path via ``DISCOPT_P3_FORCE_CUT_PATH=1`` (the 1c
reachability toggle). 1d predicted zero-half alone closes ~0.6–0.9 of the
reachable root gap on this class (every other SCIP family closes 0).

Root gap closed, anchored on discopt's OWN separators-off root bound so ON/OFF is
comparable:
    gap_closed = (root_bound_on - root_bound_off) / (opt - root_bound_off)
with ``opt`` from ``minlplib.solu``. The SCIP-attribution anchor (``scip_all_off``
from the 1d JSON) is printed alongside so the discopt LP floor can be compared to
SCIP's raw-LP floor.

Correctness gate (non-negotiable): ``incorrect_count == 0`` — with the flag ON the
reported objective must never beat the oracle and the dual bound must never cross
it. Guardrails mirror p3_cmir_aggregation_onoff: panel small; hard per-solve cap;
results persisted to ``results/`` as JSON; errored instances LABELED, never dropped.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_zerohalf_onoff.py
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Route the integer-product / graphpart class through the cut-enabled
# _solve_milp_bb path so the root cut loop (and thus the zero-half separator) is
# reachable — the 1c toggle. Set before importing discopt.
os.environ.setdefault("DISCOPT_P3_FORCE_CUT_PATH", "1")

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_SOLU = _SNAPSHOT / "minlplib.solu"
_CERT_OPTIMA = _REPO_ROOT / "docs/dev/data/cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

PANEL = [
    "graphpart_2pm-0044-0044",
    "graphpart_2g-0044-1601",
    "graphpart_2pm-0055-0055",
    "ex1263a",
]

# SCIP separators-off root LP floor from the 1d attribution JSON (for comparison
# only — shows where discopt's LP floor sits vs SCIP's raw LP).
SCIP_ALL_OFF = {
    "graphpart_2pm-0044-0044": -16.0,
    "graphpart_2g-0044-1601": -1025886.0,
    "graphpart_2pm-0055-0055": -24.999999999999996,
}

CAP_SECONDS = 45.0


def _load_oracle(name: str) -> float | None:
    if _SOLU.exists():
        best = None
        with _SOLU.open() as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == name:
                    try:
                        val = float(parts[2])
                    except ValueError:
                        continue
                    if parts[0] == "=opt=":
                        return val
                    if parts[0] == "=best=" and best is None:
                        best = val
        if best is not None:
            return best
    if _CERT_OPTIMA.exists():
        data = json.loads(_CERT_OPTIMA.read_text())
        if name in data:
            return float(data[name])
    return None


def _solve(nl_path: Path, zerohalf_on: bool, max_nodes: int) -> dict:
    os.environ["DISCOPT_ZEROHALF"] = "1" if zerohalf_on else "0"
    import discopt.modeling as dm

    t0 = time.monotonic()
    model = dm.from_nl(str(nl_path))
    res = model.solve(time_limit=CAP_SECONDS, max_nodes=max_nodes)
    wall = time.monotonic() - t0
    stats = getattr(res, "solver_stats", None) or {}
    zh_cuts = float(stats.get("cuts/zerohalf", 0.0))
    return {
        "status": str(res.status),
        "objective": res.objective,
        "bound": res.bound,
        "root_bound": getattr(res, "root_bound", None),
        "root_gap": getattr(res, "root_gap", None),
        "node_count": res.node_count,
        "zerohalf_cuts": zh_cuts,
        "wall": wall,
    }


def _incorrect(objective, oracle, sense_min=True) -> bool:
    if objective is None or oracle is None:
        return False
    tol = 1e-4 * (1.0 + abs(oracle))
    return objective < oracle - tol if sense_min else objective > oracle + tol


def _gap_closed(root_off, root_on, opt) -> float | None:
    if root_off is None or root_on is None or opt is None:
        return None
    denom = opt - root_off
    if abs(denom) < 1e-9:
        return None  # LP floor already at the optimum: no reachable gap
    return (root_on - root_off) / denom


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    incorrect_count = 0

    print(f"cert:P3.1d — zero-half ON/OFF over {len(PANEL)} instances")
    print("(DISCOPT_P3_FORCE_CUT_PATH=1; root bound at max_nodes=1)\n")
    hdr = (
        f"{'instance':<24} {'opt':>12} {'scip_off':>12} "
        f"{'root_off':>12} {'root_on':>12} {'zh_cuts':>8} {'gap_closed':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in PANEL:
        nl_path = _NL_DIR / f"{name}.nl"
        row: dict = {"instance": name}
        if not nl_path.exists():
            row.update(status="skipped", note="nl-missing")
            rows.append(row)
            print(f"{name:<24} SKIP (nl missing)")
            continue

        oracle = _load_oracle(name)
        try:
            # Root bound at max_nodes=1 (the reachable root LP + cut loop).
            off = _solve(nl_path, zerohalf_on=False, max_nodes=1)
            on = _solve(nl_path, zerohalf_on=True, max_nodes=1)
            # Correctness: a bounded-node ON solve must never certify below the
            # oracle (an uncapped solve is too slow on the larger graphparts; the
            # root-bound-vs-oracle check below is the primary false-certificate
            # guard, and a bounded tree still exercises the ON cut path in-tree).
            on_full = _solve(nl_path, zerohalf_on=True, max_nodes=2000)
        except Exception as exc:  # noqa: BLE001 — label, never drop
            row.update(status="skipped", note=f"error:{type(exc).__name__}:{exc}")
            rows.append(row)
            print(f"{name:<24} SKIP (error: {exc})")
            continue

        bad_on = _incorrect(on["objective"], oracle) or _incorrect(on_full["objective"], oracle)
        bad_bound = (
            on["root_bound"] is not None
            and oracle is not None
            and math.isfinite(on["root_bound"])
            and on["root_bound"] > oracle + 1e-4 * (1.0 + abs(oracle))
        )
        if bad_on or bad_bound:
            incorrect_count += 1

        gc = _gap_closed(off["root_bound"], on["root_bound"], oracle)
        row.update(
            status="run",
            oracle=oracle,
            scip_all_off=SCIP_ALL_OFF.get(name),
            off=off,
            on=on,
            on_full=on_full,
            gap_closed=gc,
            incorrect_on=bad_on,
            bad_bound=bad_bound,
        )
        rows.append(row)

        def _f(x):
            return f"{x:.4g}" if isinstance(x, (int, float)) and math.isfinite(x) else "—"

        print(
            f"{name:<24} {_f(oracle):>12} {_f(SCIP_ALL_OFF.get(name)):>12} "
            f"{_f(off['root_bound']):>12} {_f(on['root_bound']):>12} "
            f"{_f(on['zerohalf_cuts']):>8} {_f(gc):>11}",
            flush=True,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_zerohalf_onoff_{ts}.json"
    out.write_text(json.dumps({"cap_seconds": CAP_SECONDS, "rows": rows}, indent=2, default=str))
    print(f"\nincorrect_count = {incorrect_count}")
    print(f"raw JSON -> {out}")
    return 1 if incorrect_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
