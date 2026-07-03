"""cert:P3.1c — cut-reachability entry experiment: force the cut-enabled path.

Phase 3 follow-on 1c (``certification-gap-plan.md`` §7). The 1b measurement
found the aggregation c-MIR / Gomory / cover separators fire **0 times** on the
integer-product / graphpart class because the default dispatch routes those
instances to the monolithic Rust ``_solve_milp_simplex`` engine (no cut seam),
and the ``_solve_milp_bb`` fallback gates integer cuts off when
``prefer_pounce=False``.

This experiment answers the decisive GO/NO-GO question, with numbers, BEFORE any
invasive Rust cut-seam refactor: if the class is made to run the **cut-enabled**
path (so the aggregation c-MIR + existing Gomory/cover cuts actually fire at the
root), does the root dual bound close a material fraction of the 0b gap toward
SCIP's ~100%?

Lever (least-invasive, experiment-scoped, default-OFF): env
``DISCOPT_P3_FORCE_CUT_PATH=1`` SKIPS the ``nlp_solver->"simplex"`` reroute at
``solver.py:~3327``, keeping the reformulated MILP on the self-hosted
``_solve_milp_bb`` path with ``prefer_pounce=True`` (which enables the integer
cut loop). ``DISCOPT_CMIR_AGGREGATION=1`` additionally arms the aggregation
c-MIR separator. Both default-off; math-neutral when off.

Per instance it records:

  1. FIRES?  — do cuts run on this path? Read from
     ``SolveResult.solver_stats['cuts/{gomory,mir,aggregation,cover_clique}']``
     (the cert:P3.1b per-source counter). Also records WHICH internal solve path
     the model took (``_solve_milp_bb`` vs ``_solve_milp_simplex``).
  2. ON vs OFF — root dual bound, root gap closed vs the ``minlplib.solu``
     oracle (anchored at the OFF root-LP floor), and node count at an equal
     fixed node budget.
  3. CORRECT? — ``incorrect_count`` must be 0: reported optimum never beats the
     oracle and the dual bound never crosses it.

Guardrails: panel <= 6; 90 s hard cap per solve; results persisted to
``results/``; over-cap / errored instances LABELED, never dropped; synchronous.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_1c_cut_reachability.py
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


def _bootstrap_discopt_from_this_worktree() -> None:
    """Make ``discopt`` resolve to THIS worktree's ``python/`` source (so the
    cert:P3.1c ``solver.py`` toggle under test is the one that runs), while the
    compiled Rust extension resolves from the installed maturin package.

    The shared venv's ``discopt.pth`` may point at a since-removed sibling
    worktree, so ``discopt._rust`` is not otherwise importable here; we alias it
    to the standalone installed ``_rust`` extension (same ABI, same symbols).
    Both steps are inert if the environment is already consistent."""
    import contextlib
    import importlib
    import sys

    repo_python = str(Path(__file__).resolve().parent.parent.parent / "python")
    if repo_python not in sys.path:
        sys.path.insert(0, repo_python)
    # Alias discopt._rust -> the installed compiled extension, before any
    # discopt.solver import triggers `from discopt._rust import ...`.
    if "discopt._rust" not in sys.modules:
        with contextlib.suppress(ImportError):
            sys.modules["discopt._rust"] = importlib.import_module("_rust")


_bootstrap_discopt_from_this_worktree()

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_SOLU = _SNAPSHOT / "minlplib.solu"
_CERT_OPTIMA = _REPO_ROOT / "docs/dev/data/cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

# The 0b/1b integer-product / graphpart panel where the 0% root gap lives.
# Small subset only (<=6): ex1263/ex1263a + 3 small graphparts. fac1-3 skipped
# (already ~1.0 gap-closed by discopt, and they route to spatial-McCormick which
# this lever does not touch).
PANEL = [
    "ex1263",
    "ex1263a",
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


# Which internal solve entrypoint the model took — the cut loop lives only in
# _solve_milp_bb, so this pins the reachability question.
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


def _solve(nl_path: Path, force_cut_path: bool, max_nodes: int) -> dict:
    """Solve once; return status/objective/bound/nodes + per-source root cut
    counts + the internal solve path taken. ``force_cut_path`` toggles the
    cert:P3.1c reroute-skip lever AND arms the aggregation c-MIR separator."""
    if force_cut_path:
        os.environ["DISCOPT_P3_FORCE_CUT_PATH"] = "1"
        os.environ["DISCOPT_CMIR_AGGREGATION"] = "1"
    else:
        os.environ["DISCOPT_P3_FORCE_CUT_PATH"] = "0"
        os.environ["DISCOPT_CMIR_AGGREGATION"] = "0"

    import discopt.modeling as dm

    _PATH_HITS.clear()
    t0 = time.monotonic()
    model = dm.from_nl(str(nl_path))
    res = model.solve(time_limit=CAP_SECONDS, max_nodes=max_nodes)
    wall = time.monotonic() - t0
    stats = dict(res.solver_stats) if res.solver_stats else {}
    cuts = {k: v for k, v in stats.items() if k.startswith("cuts/")}
    return {
        "status": str(res.status),
        "objective": res.objective,
        "bound": res.bound,
        "root_bound": res.root_bound,
        "node_count": res.node_count,
        "wall": wall,
        "cuts_all": cuts,
        "cuts_total": float(sum(cuts.values())),
        "path": dict(_PATH_HITS),
    }


def _gap_closed(bound, oracle, trivial) -> float | None:
    """Root gap closed by ``bound`` vs the oracle, anchored at the OFF root-LP
    floor (``trivial``): (bound - trivial) / (oracle - trivial). 1.0 == bound at
    oracle; 0.0 == bound at the trivial floor. This mirrors the 0b definition
    but uses discopt's own OFF root bound as the shared anchor (no SCIP here)."""
    if bound is None or oracle is None or trivial is None:
        return None
    if not (math.isfinite(bound) and math.isfinite(oracle) and math.isfinite(trivial)):
        return None
    denom = oracle - trivial
    if abs(denom) < 1e-9:
        return None
    return (bound - trivial) / denom


def _incorrect(objective, oracle, sense_min=True) -> bool:
    if objective is None or oracle is None:
        return False
    tol = 1e-4 * (1.0 + abs(oracle))
    return objective < oracle - tol if sense_min else objective > oracle + tol


def main() -> int:
    import sys

    panel = [a for a in sys.argv[1:] if not a.startswith("-")] or PANEL

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _install_path_probe()
    rows: list[dict] = []
    n_run = 0
    n_skip = 0
    incorrect_count = 0
    any_fired = False

    print(f"cert:P3.1c — cut-reachability ON/OFF over {len(panel)} instances\n")
    print(f"(equal budget {BUDGET_NODES} nodes, {CAP_SECONDS:.0f}s cap)\n")
    hdr = (
        f"{'instance':<26} {'opt':>12} {'bnd_off':>12} {'bnd_on':>12} "
        f"{'gc_off':>7} {'gc_on':>7} {'nd_off':>7} {'nd_on':>7} "
        f"{'cuts_on':>7} {'path_on':>18}  note"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in panel:
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
            off = _solve(nl_path, force_cut_path=False, max_nodes=BUDGET_NODES)
            on = _solve(nl_path, force_cut_path=True, max_nodes=BUDGET_NODES)
            on_full = _solve(nl_path, force_cut_path=True, max_nodes=500_000)
        except Exception as exc:  # noqa: BLE001 — label, never drop
            row.update(status="skipped", note=f"error:{type(exc).__name__}:{exc}")
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP (error: {exc})")
            continue

        cuts_on = float(on.get("cuts_total", 0.0))
        fired = cuts_on > 0
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

        # Anchor gap-closed at the OFF root-LP floor (the shared "trivial" bound).
        trivial = off["root_bound"]
        gc_off = _gap_closed(off["root_bound"], oracle, trivial)  # ~0.0 by construction
        gc_on = _gap_closed(on["root_bound"], oracle, trivial)

        path_key = ",".join(sorted(on.get("path", {}).keys())) or "spatial-mccormick"

        row.update(
            status="run",
            oracle=oracle,
            off=off,
            on=on,
            on_full=on_full,
            cuts_fired=fired,
            cuts_total_on=cuts_on,
            gap_closed_off=gc_off,
            gap_closed_on=gc_on,
            solve_path_on=path_key,
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
            note = "cuts did NOT fire"
        elif off["root_bound"] is not None and on["root_bound"] is not None:
            note = f"Δroot={on['root_bound'] - off['root_bound']:+.4g}"
        print(
            f"{name:<26} {_f(oracle)} {_f(off['root_bound'])} {_f(on['root_bound'])} "
            f"{_f(gc_off, 7)} {_f(gc_on, 7)} "
            f"{_f(off['node_count'], 7)} {_f(on['node_count'], 7)} "
            f"{cuts_on:>7.0f} {path_key:>18}  {note}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_1c_cut_reachability_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "task": "cert:P3.1c",
                "generated": ts,
                "panel": panel,
                "cap_seconds": CAP_SECONDS,
                "budget_nodes": BUDGET_NODES,
                "n_run": n_run,
                "n_skipped": n_skip,
                "incorrect_count": incorrect_count,
                "any_cut_fired": any_fired,
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n{n_run} run, {n_skip} skipped. incorrect_count = {incorrect_count}")
    print(f"any cut fired on forced cut path: {any_fired}")
    print(f"raw JSON -> {out}")
    return 1 if incorrect_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
