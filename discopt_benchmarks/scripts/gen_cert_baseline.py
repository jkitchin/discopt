"""Generate the certification baseline (cert:T0.5).

Runs discopt over the global50 panel (``config/baron_global50.txt``) plus the
perf panel (``perf/panel.py``) with the T0.1–T0.3 instrumentation on, and writes:

  * ``reports/cert0_<timestamp>.json`` — a ``BenchmarkResults`` the
    ``run_benchmarks.py --gate cert0`` check consumes; and
  * ``docs/dev/data/cert-baseline.jsonl`` — the committed §0.2.5 bound-neutrality
    reference, restricted to the **deterministic certifying** subset: each
    instance is solved twice and included only if both runs reach OPTIMAL with a
    bit-identical node_count and objective. Time-limited / non-deterministic rows
    are excluded, so the reference is reproducible by construction; and
  * ``docs/dev/data/cert-baseline-meta.json`` — the reference's **provenance**
    (generating commit, timestamp, budget, host) and the **drop record**: every
    instance the run refused to admit, with the reason, and whether the previous
    committed reference covered it.

Per-instance time limits use the perf panel's own budgets where defined, else
``--time-limit``. The baseline is the frozen reference, so re-generate it
deliberately (not in CI).

**Coverage may not shrink silently** (issue #1134). The admission filter is
machine-sensitive — it drops anything that does not certify inside
``_MARGIN_FRAC`` of its budget — so a regeneration on a slower box quietly
deletes rows rather than recording a regression, leaving a *smaller* panel that
still reads as a green reference. A run that would drop an instance the previous
committed reference covered therefore refuses to overwrite the baseline and exits
non-zero; ``--allow-shrink`` performs the write, and the meta file records exactly
what was lost either way. The full-panel ``reports/cert0_*.json`` and the meta
file are always written, so a refused run still leaves its evidence behind.

Usage:
    python discopt_benchmarks/scripts/gen_cert_baseline.py [--time-limit 60]
                                                           [--allow-shrink]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable  # noqa: TC003
from datetime import datetime
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
# Both roots: ``benchmarks.*`` resolves from _BENCH_ROOT; the ``perf`` package's
# __init__ imports the fully-qualified ``discopt_benchmarks.perf.*`` (needs the
# repo root).
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.metrics import (  # noqa: E402
    BenchmarkResults,
    SolveStatus,
    incorrect_count,
    root_gap_populated_fraction,
)
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from perf.panel import PANEL  # noqa: E402
from utils.cert_neutrality import load_baseline  # noqa: E402

_GLOBAL50 = _BENCH_ROOT / "config" / "baron_global50.txt"
_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"
_CERT_BASELINE_META = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline-meta.json"
_CERT_OPTIMA = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"

# Objective reproducibility tolerance (absolute + relative). A certified optimum
# reproduces only to ~1e-10 across independent runs, so this is the tolerance the
# §0.2.5 neutrality check compares objectives at (node_count stays bit-exact).
_OBJ_TOL = 1e-8
_OBJ_RTOL = 1e-9
# Determinism filter: solve each instance this many times, and require it to
# certify within this fraction of its budget (margin against boundary-flakiness).
_N_DET = 3
_MARGIN_FRAC = 0.6


def coverage_loss(previous: dict[str, dict], certifying: Iterable[str]) -> list[str]:
    """Instances the previous committed reference covered that this run will not.

    This is the quantity a regeneration must never lose silently (#1134): the
    admission filter drops an instance for *any* of not-optimal / node drift /
    objective drift / near-budget, and three of those four are properties of the
    machine and the budget rather than of the tree. Dropping such a row shrinks the
    panel that later changes are held against, which reads as a green reference
    while covering less.
    """
    return sorted(set(previous) - set(certifying))


def _generating_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD`` for the provenance record.

    Deliberately narrow (`CalledProcessError`/`FileNotFoundError` only): a broad
    ``except`` here would turn "this tree is not a checkout" into a silent null and
    the provenance field is the whole point of the record (CLAUDE.md §7).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  provenance: git rev-parse failed ({exc}); commit recorded as null")
        return None
    return out.stdout.strip() or None


def build_meta(
    *,
    certifying: list[str],
    dropped: list[tuple[str, str]],
    previous: dict[str, dict],
    time_limit: float,
    attempted: int,
    allow_shrink: bool,
) -> dict:
    """The committed provenance + drop record for this regeneration.

    ``dropped`` is ``(instance, reason)`` as the admission filter produced it. Rows
    the *previous* reference covered are marked ``in_previous_baseline`` and carry
    that row's status/nodes/objective, so the record says what was lost, not merely
    that something was.
    """
    lost = set(coverage_loss(previous, certifying))
    return {
        "generated_at": datetime.now().isoformat(),
        "commit": _generating_commit(),
        "time_limit": time_limit,
        "instances_attempted": attempted,
        "instances_certifying": len(certifying),
        "previous_instances_certifying": len(previous),
        "coverage_lost": sorted(lost),
        "allow_shrink": allow_shrink,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "dropped": [
            {
                "instance": name,
                "reason": reason,
                "in_previous_baseline": name in previous,
                "previous": (
                    {
                        "status": previous[name].get("status"),
                        "node_count": previous[name].get("node_count"),
                        "objective": previous[name].get("objective"),
                    }
                    if name in previous
                    else None
                ),
            }
            for name, reason in dropped
        ],
    }


def _instance_budgets(default_tl: float) -> dict[str, float]:
    """Per-instance time limits: global50 at the default, perf-panel instances at
    their own (usually larger) panel budget so the flagships still expose signal."""
    names = _GLOBAL50.read_text().split()
    budgets = dict.fromkeys(names, default_tl)
    for inst in PANEL:
        budgets[inst.name] = max(budgets.get(inst.name, 0.0), float(inst.time_limit))
    return budgets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0, help="default per-instance seconds")
    ap.add_argument(
        "--allow-shrink",
        action="store_true",
        help=(
            "overwrite cert-baseline.jsonl even when this run drops an instance the "
            "committed reference covered (#1134). Without it such a run refuses the "
            "write and exits non-zero; the drop record is written either way."
        ),
    )
    args = ap.parse_args()

    # The reference this run would replace: read BEFORE the solves so the shrink
    # guard has something to compare against even if the write is refused.
    previous = load_baseline(_CERT_BASELINE) if _CERT_BASELINE.exists() else {}

    budgets = _instance_budgets(args.time_limit)
    order = sorted(budgets)
    print(f"Certification baseline: {len(order)} instances (global50 + perf panel)")

    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    # Full-panel results feed the cert0 coverage gate (root_gap on all of
    # global50). The committed neutrality reference (cert-baseline.jsonl) is the
    # stricter *deterministic certifying* subset built below.
    results = BenchmarkResults(suite="cert0", timestamp=datetime.now().isoformat())
    certifying: list = []
    dropped: list[tuple[str, str]] = []

    def _solve(name: str):
        cfg = BenchmarkConfig(
            suite_name="cert0", time_limit=int(budgets[name]), num_runs=1, solvers=[solver]
        )
        return BenchmarkRunner(cfg)._run_discopt(solver, name, 0)

    for i, name in enumerate(order, 1):
        cfg = BenchmarkConfig(suite_name="cert0", time_limit=1, num_runs=1, solvers=[solver])
        if BenchmarkRunner(cfg)._find_nl_file(name) is None:
            print(f"  [{i}/{len(order)}] SKIP {name} (not vendored)", flush=True)
            continue
        # Solve _N_DET times. An instance qualifies for the neutrality reference
        # only if EVERY run certifies to OPTIMAL with a bit-identical node_count
        # and an objective agreeing to tolerance, AND it certifies with a
        # comfortable time margin (max wall <= _MARGIN_FRAC * budget). The margin
        # guard excludes *boundary-flaky* instances — ones that certify right at
        # the time limit and flip to `feasible` under tiny timing differences
        # (nvs17 at 60s), which the twice-solve check let slip through. The
        # objective uses a tolerance because a certified optimum jitters ~1e-10
        # across runs (JAX/BLAS) — bit-exact equality is not meaningful; this is
        # the same tolerance the §0.2.5 neutrality check uses.
        runs = [_solve(name) for _ in range(_N_DET)]
        results.add_result(runs[0])  # full panel = first run
        r0 = runs[0]
        budget = float(budgets[name])
        all_opt = all(r.status == SolveStatus.OPTIMAL for r in runs)
        nodes_same = all(r.node_count == r0.node_count for r in runs)
        objs = [r.objective for r in runs]
        obj_ok = all(o is not None for o in objs) and all(
            abs(o - objs[0]) <= _OBJ_TOL + _OBJ_RTOL * abs(objs[0]) for o in objs
        )
        max_wall = max(r.wall_time for r in runs)
        margin_ok = max_wall <= _MARGIN_FRAC * budget
        det = all_opt and nodes_same and obj_ok and margin_ok
        if det:
            certifying.append(r0)
            tag = "CERTIFY"
        else:
            if not all_opt:
                reason = "not-optimal(" + "/".join(r.status.value for r in runs) + ")"
            elif not nodes_same:
                reason = "node_count " + "/".join(str(r.node_count) for r in runs)
            elif not obj_ok:
                reason = "obj drift"
            else:
                reason = f"near-limit(wall {max_wall:.0f}s/{budget:.0f}s)"
            dropped.append((name, reason))
            tag = f"drop:{reason}"
        rg = "None" if r0.root_gap is None else f"{r0.root_gap:.3g}"
        print(
            f"  [{i}/{len(order)}] {name:20s} {r0.status.value:10s} "
            f"nodes={r0.node_count} root_gap={rg} {tag}",
            flush=True,
        )

    # reports/cert0_<ts>.json for the --gate cert0 consumer (full panel).
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    reports_dir = _REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    results_path = reports_dir / f"cert0_{ts}.json"
    results.save(results_path)

    # docs/dev/data/cert-baseline.jsonl — the §0.2.5 neutrality reference: the
    # deterministic-certifying subset only.
    cert_rows = sorted(certifying, key=lambda r: r.instance)
    os.makedirs(_CERT_BASELINE.parent, exist_ok=True)

    # Provenance + drop record. Written unconditionally, including on a refused
    # write: a run that shrank coverage is exactly the run whose evidence matters.
    lost = coverage_loss(previous, [r.instance for r in cert_rows])
    meta = build_meta(
        certifying=[r.instance for r in cert_rows],
        dropped=dropped,
        previous=previous,
        time_limit=float(args.time_limit),
        attempted=len(order),
        allow_shrink=bool(args.allow_shrink),
    )
    _CERT_BASELINE_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    if not lost or args.allow_shrink:
        with open(_CERT_BASELINE, "w") as fh:
            for r in cert_rows:
                fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    # Summary + self-check.
    full_rows = results.get_results("discopt")
    coverage = root_gap_populated_fraction(full_rows)
    optima = json.loads(_CERT_OPTIMA.read_text()) if _CERT_OPTIMA.exists() else {}
    incorrect = incorrect_count(full_rows, optima) if optima else None
    print("\n─── summary ───")
    print(f"  full panel rows: {len(full_rows)}  (cert0 gate)")
    print(f"  root_gap coverage: {coverage:.3f} (gate: >= 0.90)")
    print(f"  incorrect_count (vs {len(optima)} oracles): {incorrect}")
    print(f"  deterministic-certifying subset: {len(cert_rows)}  (neutrality reference)")
    print(f"  dropped ({len(dropped)}): " + ", ".join(f"{n}[{r}]" for n, r in dropped))
    print(f"  results: {results_path}")
    print(f"  meta: {_CERT_BASELINE_META}")
    if not lost:
        print(f"  coverage: {len(previous)} -> {len(cert_rows)} instances, none lost")
        print(f"  baseline: {_CERT_BASELINE} (written)")
        return 0
    reasons = dict(dropped)
    print(
        f"\n  COVERAGE LOSS: {len(lost)} instance(s) the committed reference covered "
        "are not admitted by this run:"
    )
    for name in lost:
        print(f"    {name:20s} {reasons.get(name, 'absent from this run')}")
    if args.allow_shrink:
        print(f"  baseline: {_CERT_BASELINE} (written under --allow-shrink)")
        return 0
    print(
        f"  baseline: {_CERT_BASELINE} NOT overwritten. Refusing to shrink the panel\n"
        "  silently (#1134): a smaller reference still reads as green while covering\n"
        "  less. Establish why each instance above dropped -- a genuine regression, or\n"
        "  a slower box than the one that generated the reference -- then re-run with\n"
        "  --allow-shrink to record the loss deliberately."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
