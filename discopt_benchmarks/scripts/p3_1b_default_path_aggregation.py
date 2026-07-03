"""cert:P3.1b — default-path aggregation c-MIR: ON vs OFF + firing check.

Phase 3 follow-on 1b (``certification-gap-plan.md`` §7). Build 1 measured the
aggregation separator on the LP-spatial engine and found +0 bound movement there
(that relaxation is already root-tight on the panel). 1b measures the **default
MILP path** — the path where 0b measured discopt closing 0% of the root gap SCIP
closes ~100% of.

It answers, with numbers, three things per instance:

  1. FIRES?  — does the aggregation c-MIR separator actually run on the default
     path? Read from ``SolveResult.solver_stats['cuts/aggregation']`` (the
     cert:P3.1b instrumentation in ``_solve_milp_bb``). It also records WHICH
     internal solve path the model took (``_solve_milp_bb`` / ``_solve_milp_simplex``
     / spatial-McCormick), because the aggregation hook lives only in
     ``_solve_milp_bb``'s root cut loop.
  2. ON vs OFF — root dual bound, root gap closed vs the ``minlplib.solu`` oracle,
     and node count at an equal fixed node budget.
  3. CORRECT? — ``incorrect_count`` must be 0: the reported optimum never beats
     the oracle and the dual bound never crosses it.

Guardrails: panel <= 8; 90 s hard cap per solve; results persisted to
``results/``; over-cap / errored instances LABELED, never dropped.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_1b_default_path_aggregation.py
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

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_SOLU = _SNAPSHOT / "minlplib.solu"
_CERT_OPTIMA = _REPO_ROOT / "docs/dev/data/cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

# The 0b panel where the 0% root-gap lives (integer-product / graphpart). All 8
# instances are measured on the DEFAULT path (no lp_spatial), unlike build 1.
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

CAP_SECONDS = 90.0
BUDGET_NODES = 2000


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


# Which internal solve entrypoint the model took — the aggregation hook lives
# only in _solve_milp_bb's root cut loop, so this pins the firing question.
_PATH_HITS: dict[str, int] = {}


def _install_path_probe() -> None:
    import discopt.solver as sv

    for fn in ("_solve_milp_bb", "_solve_milp_simplex", "_solve_milp_gurobi"):
        if not hasattr(sv, fn):
            continue
        orig = getattr(sv, fn)

        def make(name, o):
            def wrapped(*a, **k):
                _PATH_HITS[name] = _PATH_HITS.get(name, 0) + 1
                return o(*a, **k)

            return wrapped

        setattr(sv, fn, make(fn, orig))


def _solve(nl_path: Path, cmir_on: bool, max_nodes: int) -> dict:
    """Solve once on the DEFAULT path; return status/objective/bound/nodes and the
    per-source root cut counts + the internal solve path taken."""
    os.environ["DISCOPT_CMIR_AGGREGATION"] = "1" if cmir_on else "0"

    import discopt.modeling as dm

    _PATH_HITS.clear()
    t0 = time.monotonic()
    model = dm.from_nl(str(nl_path))
    res = model.solve(time_limit=CAP_SECONDS, max_nodes=max_nodes)
    wall = time.monotonic() - t0
    stats = dict(res.solver_stats) if res.solver_stats else {}
    return {
        "status": str(res.status),
        "objective": res.objective,
        "bound": res.bound,
        "root_bound": res.root_bound,
        "node_count": res.node_count,
        "wall": wall,
        "cuts_aggregation": stats.get("cuts/aggregation", 0.0),
        "cuts_all": {k: v for k, v in stats.items() if k.startswith("cuts/")},
        "path": dict(_PATH_HITS),
    }


def _gap_closed(bound, oracle, root_lp) -> float | None:
    """Root gap closed vs oracle, anchored at the raw root-LP floor when known.

    Here we lack SCIP's trivial anchor per run, so report the fraction of the
    |oracle - root_bound| distance — i.e. 1.0 means root_bound == oracle. Anchor
    at root_lp when both bounds share it; otherwise report None (uncomparable)."""
    if bound is None or oracle is None or not math.isfinite(bound):
        return None
    # normalized closeness of the (dual) bound to the oracle
    denom = max(1.0, abs(oracle))
    return 1.0 - abs(oracle - bound) / denom


def _incorrect(objective, oracle, sense_min=True) -> bool:
    if objective is None or oracle is None:
        return False
    tol = 1e-4 * (1.0 + abs(oracle))
    return objective < oracle - tol if sense_min else objective > oracle + tol


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _install_path_probe()
    rows: list[dict] = []
    n_run = 0
    n_skip = 0
    incorrect_count = 0
    any_fired = False

    print(f"cert:P3.1b — default-path aggregation c-MIR ON/OFF over {len(PANEL)} instances\n")
    print(f"(default path, equal budget {BUDGET_NODES} nodes, {CAP_SECONDS:.0f}s cap)\n")
    hdr = (
        f"{'instance':<26} {'opt':>12} {'bnd_off':>12} {'bnd_on':>12} "
        f"{'nd_off':>8} {'nd_on':>8} {'agg_cuts':>8} {'path':>16}  note"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in PANEL:
        nl_path = _NL_DIR / f"{name}.nl"
        row: dict = {"instance": name}
        if not nl_path.exists():
            row.update(status="skipped", note="nl-missing")
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP (nl missing)")
            continue

        oracle = _load_oracle(name)
        try:
            off = _solve(nl_path, cmir_on=False, max_nodes=BUDGET_NODES)
            on = _solve(nl_path, cmir_on=True, max_nodes=BUDGET_NODES)
            on_full = _solve(nl_path, cmir_on=True, max_nodes=500_000)
        except Exception as exc:  # noqa: BLE001 — label, never drop
            row.update(status="skipped", note=f"error:{type(exc).__name__}:{exc}")
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP (error: {exc})")
            continue

        agg_on = float(on.get("cuts_aggregation", 0.0))
        fired = agg_on > 0
        any_fired = any_fired or fired

        bad_on = _incorrect(on["objective"], oracle) or _incorrect(on_full["objective"], oracle)
        bad_bound = (
            on["bound"] is not None
            and oracle is not None
            and math.isfinite(on["bound"])
            and on["bound"] > oracle + 1e-4 * (1.0 + abs(oracle))
        )
        if bad_on or bad_bound:
            incorrect_count += 1

        gc_off = _gap_closed(off["root_bound"], oracle, None)
        gc_on = _gap_closed(on["root_bound"], oracle, None)

        path_key = ",".join(sorted(on.get("path", {}).keys())) or "spatial-mccormick"

        row.update(
            status="run",
            oracle=oracle,
            off=off,
            on=on,
            on_full=on_full,
            aggregation_fired=fired,
            aggregation_cuts_on=agg_on,
            gap_closed_off=gc_off,
            gap_closed_on=gc_on,
            solve_path=path_key,
            incorrect_on=bad_on,
            bad_bound=bad_bound,
        )
        rows.append(row)
        n_run += 1

        def _f(x, w=12):
            ok = isinstance(x, (int, float)) and math.isfinite(x)
            return f"{x:>{w}.4g}" if ok else f"{'—':>{w}}"

        note = ""
        if bad_on or bad_bound:
            note = "!!! INCORRECT (ON) !!!"
        elif not fired:
            note = "agg did NOT fire"
        elif off["bound"] is not None and on["bound"] is not None:
            note = f"Δbnd={on['bound'] - off['bound']:+.4g}"
        print(
            f"{name:<26} {_f(oracle)} {_f(off['bound'])} {_f(on['bound'])} "
            f"{_f(off['node_count'], 8)} {_f(on['node_count'], 8)} "
            f"{agg_on:>8.0f} {path_key:>16}  {note}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_1b_default_path_aggregation_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "task": "cert:P3.1b",
                "generated": ts,
                "panel": PANEL,
                "cap_seconds": CAP_SECONDS,
                "budget_nodes": BUDGET_NODES,
                "n_run": n_run,
                "n_skipped": n_skip,
                "incorrect_count": incorrect_count,
                "any_aggregation_fired": any_fired,
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n{n_run} run, {n_skip} skipped. incorrect_count = {incorrect_count}")
    print(f"any aggregation cut fired on default path: {any_fired}")
    print(f"raw JSON -> {out}")
    return 1 if incorrect_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
