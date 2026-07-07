"""cert:P3 — Marchand-Wolsey aggregation c-MIR: ON vs OFF measurement.

Measures the lever the aggregation c-MIR separator is built to move: the root
dual bound and the B&B node count with ``DISCOPT_CMIR_AGGREGATION`` ON vs OFF, on
the integer-product / graphpart panel from ``certification-gap-plan.md`` §7's 0b
verdict (where discopt closes ~0% of the root gap SCIP's cut loop closes ~100%
of). The correctness invariant (`incorrect_count == 0`) is checked per instance
against the ``minlplib.solu`` oracle: with the flag ON the reported dual bound
must never cross the known optimum.

Guardrails mirror p3_0b: panel <= 8; a 60 s hard cap per solve; results persisted
to ``results/`` as JSON; over-cap / errored instances LABELED, never dropped.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_cmir_aggregation_onoff.py
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

# The native aggregation c-MIR runs in the LP-spatial B&B engine's node-cut
# separator (`_separate_node_cuts`), which is scoped to *pure-integer* models.
# The panel is therefore the pure-integer, in-scope subset of §7's 0b panel —
# exactly the graphpart probes (plus ex1263a) where 0b measured discopt closing
# 0% of the root gap SCIP's cut loop closes 100% of. (ex1263/fac1-3 carry
# continuous vars → out of the spatial engine's scope; left to the follow-on
# default-path wiring.)
PANEL = [
    "ex1263a",
    "graphpart_2pm-0044-0044",
    "graphpart_2g-0044-1601",
    "graphpart_2pm-0055-0055",
]

CAP_SECONDS = 60.0
# Cut rounds must be > 0 for the spatial engine's root cut loop (and thus the
# aggregation separator) to run at all; keep modest so the measurement is cheap.
CUT_ROUNDS = 3
# Equal-node-budget bound comparison (the §1.5 / §7 gate): a tighter bound with
# cuts ON at the SAME budget is the lever. Uncapped ON solve checks the oracle.
BUDGET_NODES = 400


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


def _solve(nl_path: Path, cmir_on: bool, max_nodes: int) -> dict:
    """Solve once via the LP-spatial engine; return status/objective/bound/nodes.

    The bound is the engine's final dual bound. At an equal `max_nodes` budget a
    *higher* (tighter) bound with cuts ON is the lever §7 measures. Also runs an
    uncapped solve (separate call) for the certified-optimum correctness check."""
    os.environ["DISCOPT_CMIR_AGGREGATION"] = "1" if cmir_on else "0"

    import discopt.modeling as dm

    t0 = time.monotonic()
    model = dm.from_nl(str(nl_path))
    res = model.solve(
        time_limit=CAP_SECONDS,
        lp_spatial=True,
        lp_spatial_cut_rounds=CUT_ROUNDS,
        max_nodes=max_nodes,
    )
    wall = time.monotonic() - t0
    return {
        "status": str(res.status),
        "objective": res.objective,
        "bound": res.bound,
        "node_count": res.node_count,
        "wall": wall,
    }


def _incorrect(objective, oracle, sense_min=True) -> bool:
    """A reported optimum that beats the oracle by > tol is a false certificate."""
    if objective is None or oracle is None:
        return False
    tol = 1e-4 * (1.0 + abs(oracle))
    # min sense: objective should be >= oracle - tol (never below true optimum).
    return objective < oracle - tol if sense_min else objective > oracle + tol


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_run = 0
    n_skip = 0
    incorrect_count = 0

    print(f"cert:P3 — aggregation c-MIR ON/OFF over {len(PANEL)} instances\n")
    print(f"(LP-spatial engine, cut_rounds={CUT_ROUNDS}, equal budget {BUDGET_NODES} nodes)\n")
    hdr = (
        f"{'instance':<26} {'opt':>13} "
        f"{'bound_off':>13} {'bound_on':>13} {'nodes_off':>10} {'nodes_on':>10}  note"
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
            # Correctness: uncapped solve with the flag ON must certify the oracle.
            on_full = _solve(nl_path, cmir_on=True, max_nodes=500_000)
        except Exception as exc:  # noqa: BLE001 — label, never drop
            row.update(status="skipped", note=f"error:{type(exc).__name__}:{exc}")
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP (error: {exc})")
            continue

        # Correctness gate: the flag ON must never certify below the true optimum.
        bad_on = _incorrect(on["objective"], oracle) or _incorrect(on_full["objective"], oracle)
        # Dual bound must also never cross the oracle (min sense: bound <= opt).
        bad_bound = (
            on["bound"] is not None
            and oracle is not None
            and math.isfinite(on["bound"])
            and on["bound"] > oracle + 1e-4 * (1.0 + abs(oracle))
        )
        if bad_on or bad_bound:
            incorrect_count += 1

        row.update(
            status="run",
            oracle=oracle,
            off=off,
            on=on,
            on_full=on_full,
            incorrect_on=bad_on,
            bad_bound=bad_bound,
        )
        rows.append(row)
        n_run += 1

        def _f(x):
            return f"{x:.4g}" if isinstance(x, (int, float)) and math.isfinite(x) else "—"

        note = ""
        if bad_on or bad_bound:
            note = "!!! INCORRECT (ON) !!!"
        elif off["bound"] is not None and on["bound"] is not None:
            d = on["bound"] - off["bound"]
            note = f"Δbound={d:+.4g}"
        print(
            f"{name:<26} {_f(oracle):>13} "
            f"{_f(off['bound']):>13} {_f(on['bound']):>13} "
            f"{_f(off['node_count']):>10} {_f(on['node_count']):>10}  {note}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_cmir_aggregation_onoff_{ts}.json"
    out.write_text(
        json.dumps(
            {"cap_seconds": CAP_SECONDS, "cut_rounds": CUT_ROUNDS, "rows": rows},
            indent=2,
            default=str,
        )
    )
    print(f"\n{n_run} run, {n_skip} skipped. incorrect_count = {incorrect_count}")
    print(f"raw JSON -> {out}")
    # Hard gate: any incorrect certificate is a failure.
    return 1 if incorrect_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
