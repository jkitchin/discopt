"""Phase 3 entry experiment 0b (cert:P3.0b) — SCIP root-bound proxy.

A LEAN, cheap proxy for the plan's "inject SCIP's cuts into discopt's McCormick
LP" spike (``certification-gap-plan.md`` §7, Phase 0b). Rather than build full
cut-injection plumbing (too expensive), it measures three root dual bounds per
panel instance and compares the *root gap closed* by discopt vs by SCIP-with-cuts:

  1. discopt root bound   — ``SolveResult.root_bound`` from a short discopt solve
                            (populated by Phase 0's T0.1 producer).
  2. SCIP root bound      — pyscipopt reads the ``.nl`` directly, solves with a
                            node limit of 1 so SCIP's default root cut loop runs
                            (aggregation / c-MIR etc.), then ``getDualbound()``.
  3. oracle optimum       — ``minlplib.solu`` (``=opt=`` / ``=best=`` lines),
                            with ``docs/dev/data/cert-optima.json`` as a fallback.

Root gap closed by a bound B (min sense):
    gap_closed = (B - trivial) / (opt - trivial)
where ``trivial`` is a cheap LP/continuous root lower bound shared by both
(discopt's own root_bound floor is not comparable across solvers, so the trivial
anchor is SCIP's *first-LP* root relaxation bound before its cut loop — read via
a second SCIP solve with separating/propagation rounds disabled). This isolates
the question the experiment must answer:

  * If SCIP closes a LOT more root gap than discopt across the panel
    -> the weak per-node bound is a SEPARATOR-QUALITY problem -> build native
    c-MIR (Phase 3 part 2).
  * If SCIP's root gap is also poor (SCIP ~ discopt, both far from opt)
    -> it is a RELAXATION / BRANCHING problem -> re-scope, do NOT build c-MIR.

Hard guardrails (learned from T2.1): panel <= 8 instances; per-instance 60s hard
cap on both the discopt solve and each SCIP root solve; results persisted to
``discopt_benchmarks/results/`` as JSON; errored / over-cap instances are LABELED
skipped, never silently dropped.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_0b_scip_rootbound.py
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# JAX config must be set before discopt is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_SOLU = _SNAPSHOT / "minlplib.solu"
_CERT_OPTIMA = _REPO_ROOT / "docs/dev/data/cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

# Small integer-product / MILP-relaxation panel. graphpart is the plan's Phase 3
# gate probe; the three smallest graphpart .nl instances by file size are chosen.
PANEL = [
    "ex1263",
    "ex1263a",
    "fac1",
    "fac2",
    "fac3",
    "graphpart_2pm-0044-0044",
    "graphpart_2g-0044-1601",
    "graphpart_2pm-0055-0055",
]

CAP_SECONDS = 60.0


def _load_oracle(name: str) -> float | None:
    """Return the known optimum for *name* from solu, else cert-optima.json."""
    # minlplib.solu: "=opt=  NAME  VALUE" (also =best= as a fallback).
    if _SOLU.exists():
        best = None
        with _SOLU.open() as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == name:
                    tag = parts[0]
                    try:
                        val = float(parts[2])
                    except ValueError:
                        continue
                    if tag == "=opt=":
                        return val
                    if tag == "=best=" and best is None:
                        best = val
        if best is not None:
            return best
    if _CERT_OPTIMA.exists():
        data = json.loads(_CERT_OPTIMA.read_text())
        if name in data:
            return float(data[name])
    return None


def _discopt_root_bound(nl_path: Path) -> tuple[str, float | None, float | None, float | None]:
    """(status, objective, root_bound, wall) from a short discopt solve."""
    import discopt.modeling as dm

    t0 = time.monotonic()
    model = dm.from_nl(str(nl_path))
    res = model.solve(time_limit=CAP_SECONDS)
    wall = time.monotonic() - t0
    return str(res.status), res.objective, res.root_bound, wall


def _scip_bounds(nl_path: Path) -> tuple[float | None, float | None, str]:
    """(trivial_root_lp_bound, scip_root_bound_with_cuts, status).

    trivial: SCIP's root relaxation bound with separation *and* propagation
             disabled and a node limit of 1 — the raw LP/relaxation floor before
             any cut loop.
    scip_root: SCIP's root bound with its default cut loop active (node limit 1).
    """
    from pyscipopt import Model

    # --- trivial: cuts + propagation off ---
    trivial = None
    m0 = Model()
    m0.hideOutput()
    m0.readProblem(str(nl_path))
    m0.setParam("limits/nodes", 1)
    m0.setParam("limits/time", CAP_SECONDS)
    m0.setParam("separating/maxrounds", 0)
    m0.setParam("separating/maxroundsroot", 0)
    m0.setParam("presolving/maxrounds", 0)
    m0.setParam("propagating/maxrounds", 0)
    m0.setParam("propagating/maxroundsroot", 0)
    m0.optimize()
    try:
        trivial = float(m0.getDualbound())
    except Exception:
        trivial = None

    # --- scip_root: default cut loop, one node ---
    m1 = Model()
    m1.hideOutput()
    m1.readProblem(str(nl_path))
    m1.setParam("limits/nodes", 1)
    m1.setParam("limits/time", CAP_SECONDS)
    m1.optimize()
    status = m1.getStatus()
    try:
        scip_root = float(m1.getDualbound())
    except Exception:
        scip_root = None
    return trivial, scip_root, status


def _gap_closed(bound: float | None, trivial: float | None, opt: float | None) -> float | None:
    """Fraction of the root gap closed by *bound* (min sense).

    (bound - trivial) / (opt - trivial). None if inputs missing or the
    denominator is ~0 (root already tight / degenerate).
    """
    if bound is None or trivial is None or opt is None:
        return None
    denom = opt - trivial
    if abs(denom) < 1e-9:
        return 1.0  # trivial bound already at the optimum -> nothing to close
    return (bound - trivial) / denom


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_run = 0
    n_skip = 0

    print(f"cert:P3.0b — SCIP root-bound proxy over {len(PANEL)} instances\n")
    hdr = (
        f"{'instance':<26} {'opt':>13} {'triv':>13} "
        f"{'disc_root':>13} {'scip_root':>13} {'disc_gc':>8} {'scip_gc':>8}  note"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in PANEL:
        nl_path = _NL_DIR / f"{name}.nl"
        row: dict = {"instance": name}
        if not nl_path.exists():
            row["status"] = "skipped"
            row["note"] = "nl-missing"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} {'SKIP (nl missing)':>60}")
            continue

        opt = _load_oracle(name)
        row["oracle_opt"] = opt

        # discopt root bound
        try:
            d_status, d_obj, d_root, d_wall = _discopt_root_bound(nl_path)
            row["discopt_status"] = d_status
            row["discopt_objective"] = d_obj
            row["discopt_root_bound"] = d_root
            row["discopt_wall"] = d_wall
        except Exception as exc:  # noqa: BLE001 - label + continue per guardrail
            row["status"] = "skipped"
            row["note"] = f"discopt-error: {type(exc).__name__}: {exc}"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP  discopt-error: {type(exc).__name__}: {exc}")
            continue

        # SCIP root bounds
        try:
            trivial, scip_root, scip_status = _scip_bounds(nl_path)
            row["scip_trivial_bound"] = trivial
            row["scip_root_bound"] = scip_root
            row["scip_status"] = scip_status
        except Exception as exc:  # noqa: BLE001
            row["status"] = "skipped"
            row["note"] = f"scip-error: {type(exc).__name__}: {exc}"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP  scip-error: {type(exc).__name__}: {exc}")
            continue

        d_gc = _gap_closed(d_root, trivial, opt)
        s_gc = _gap_closed(scip_root, trivial, opt)
        row["discopt_gap_closed"] = d_gc
        row["scip_gap_closed"] = s_gc
        row["status"] = "run"
        note = ""
        if opt is None:
            note = "no-oracle"
        elif not _finite(trivial):
            note = "no-trivial"
        row["note"] = note
        rows.append(row)
        n_run += 1

        def _f(x: float | None, w: int = 13) -> str:
            return f"{x:>{w}.4g}" if _finite(x) else f"{'--':>{w}}"

        def _p(x: float | None) -> str:
            return f"{x:>8.3f}" if _finite(x) else f"{'--':>8}"

        print(
            f"{name:<26} {_f(opt)} {_f(trivial)} {_f(d_root)} {_f(scip_root)} "
            f"{_p(d_gc)} {_p(s_gc)}  {note}"
        )

    # Aggregate signal: median gap-closed for each solver over the run rows.
    d_vals = [
        r["discopt_gap_closed"]
        for r in rows
        if r.get("status") == "run" and _finite(r.get("discopt_gap_closed"))
    ]
    s_vals = [
        r["scip_gap_closed"]
        for r in rows
        if r.get("status") == "run" and _finite(r.get("scip_gap_closed"))
    ]

    def _median(xs: list[float]) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        n = len(xs)
        mid = n // 2
        return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    d_med = _median(d_vals)
    s_med = _median(s_vals)

    print("\n--- aggregate (median root gap closed) ---")
    print(f"  discopt: {d_med if d_med is None else round(d_med, 3)}  (n={len(d_vals)})")
    print(f"  scip   : {s_med if s_med is None else round(s_med, 3)}  (n={len(s_vals)})")
    print(f"\nRun: {n_run}   Skipped: {n_skip}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_0b_scip_rootbound_{stamp}.json"
    payload = {
        "task": "cert:P3.0b",
        "generated": stamp,
        "panel": PANEL,
        "cap_seconds": CAP_SECONDS,
        "n_run": n_run,
        "n_skipped": n_skip,
        "median_discopt_gap_closed": d_med,
        "median_scip_gap_closed": s_med,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nPersisted -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
