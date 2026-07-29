"""Phase 0 — the frozen Regime-N baseline over the in-repo ``.nl`` corpus.

Consolidation plan (``docs/dev/consolidation-plan-2026-07-28.md``) §0.1 defines
**Regime N** (bound-neutral changes: refactors, dead-code deletion, rewiring that
must not change math) and gates it on ``node_count`` and the certified
``objective`` being *exactly* unchanged on a certifying panel. Until this script
existed there was no such panel artifact: every Regime-N card was asked to prove
"nothing drifted" against nothing.

This is that artifact's producer, and its checker.

Two modes
---------

``panel_baseline.py``
    Solve every instance in the corpus on **default solver settings** at a fixed
    per-instance budget and write ``reports/panel_baseline_<short-sha>.json``.

``panel_baseline.py --check reports/panel_baseline_<sha>.json``
    Re-run the same instances at the *baseline's own budget* and fail non-zero on
    any ``node_count`` or certified-``objective`` drift, printing the number of
    comparisons actually executed. **Zero executed comparisons is itself a
    failure** (CLAUDE.md §6): a checker that compares nothing and prints "0
    violations" reads exactly like a pass, and this repo has shipped that mistake
    more than once.

What is compared, and what is not
---------------------------------

A wall-clock budget makes some rows non-reproducible *by construction*: an
instance that hits the time limit explores however many nodes the machine gave it
that minute, so demanding an exact node count there would make the checker fail on
ambient load rather than on drift — a gate that cries wolf gets disabled, which is
worse than no gate. So each row carries an explicit ``comparable`` flag:

* ``comparable = true``  — the baseline run reached a **budget-independent**
  terminal status (``optimal`` / ``infeasible``) with a **certified** gap, in less
  than ``MARGIN_FRAC`` (0.6) of the budget. These rows are compared **hard**:
  node-count drift, certified-objective drift, or loss of that terminal status is
  a FAILURE. The 40 % margin is the same determinism filter
  ``gen_cert_baseline.py`` uses for the committed §0.2.5 neutrality reference.
* ``comparable = false`` — time-limited, uncertified, or errored rows. They are
  still solved, still recorded, and drift on them is still **reported** (with
  counts), but it cannot fail the check. ``comparable_reason`` says why for every
  such row, so the narrowing is visible rather than silent.

The two populations are printed separately and both counts appear in the exit
summary. A run whose comparable population collapsed to zero fails.

Root-gap instrumentation (plan task 0.3)
----------------------------------------

``certification-gap-plan.md``'s gap table carries the row *"Root-gap
instrumentation — schema exists, never populated"*. ``SolveResult.root_bound`` /
``root_gap`` / ``root_time`` and ``benchmarks.metrics.SolveResult.root_gap`` are
that schema. This panel populates it per instance, and adds the reference-relative
form the later cuts/propagation cards actually need::

    root_gap_vs_reference = |ref - root_bound| / max(1, |ref|)

which is measured against the **oracle optimum** rather than against whatever
incumbent this particular run happened to find, so a root-bound improvement is not
masked by a primal-heuristic improvement. It is populated only where a reference
optimum exists (``utils.reference_optima``); ``root_gap_reference_source`` records
which oracle supplied it, and the summary prints the coverage fraction so a reader
can never mistake partial coverage for full coverage.

Usage
-----

::

    python -u discopt_benchmarks/scripts/panel_baseline.py                  # full corpus
    python -u discopt_benchmarks/scripts/panel_baseline.py --budget 30
    python -u discopt_benchmarks/scripts/panel_baseline.py --subset alan,ex1221
    python -u discopt_benchmarks/scripts/panel_baseline.py --subset 5       # first 5
    python -u discopt_benchmarks/scripts/panel_baseline.py \
        --check reports/panel_baseline_abc1234.json

Internal child mode: ``--solve <instance> <budget>`` (one instance, one JSON line).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

# BOTH in-repo corpora, unioned. Neither is a superset of the other (66 + 81
# files, 119 distinct stems), and panelling one alone silently omits whole
# families — that is how #764 graduated a flag while missing the nvs17/19/24
# regression (#902). ``minlplib_nl`` alone is the "61-instance corpus" the
# consolidation plan names; the union is the 147-file set #865 used.
_CORPUS_DIRS = (
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl",
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib",
)

_REPORTS_DIR = _REPO_ROOT / "reports"

# Regime-N comparison tolerances. ``node_count`` is compared BIT-EXACT (it is an
# integer and a bound-neutral change cannot move it). The certified objective
# reproduces only to ~1e-10 across independent runs of the same build, so it is
# compared at the same tolerance ``gen_cert_baseline.py`` uses for the §0.2.5
# reference rather than bit-exact.
_OBJ_TOL = 1e-8
_OBJ_RTOL = 1e-9

# Determinism filter for the hard-compared population: a row counts as comparable
# only if it certified in under this fraction of its budget, so a boundary row
# cannot flip to ``time_limit`` under mild load and manufacture a failure.
_MARGIN_FRAC = 0.6

# Statuses whose node count does not depend on the wall budget.
_TERMINAL_STATUSES = ("optimal", "infeasible")
_BAD_STATUSES = ("errored", "child_crashed", "child_timeout")

_DEFAULT_BUDGET = 60.0
# Subprocess wall guard. The solve itself is bounded by the budget; this only
# catches a child that wedges outside the solver's own clock.
_CHILD_TIMEOUT_SLACK = 120.0


# --------------------------------------------------------------------------- #
# Corpus resolution                                                           #
# --------------------------------------------------------------------------- #
def corpus_instances() -> list[str]:
    """Sorted union of instance stems across every in-repo corpus directory."""
    names: set[str] = set()
    for d in _CORPUS_DIRS:
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.nl"))
    return sorted(names)


def instance_path(instance: str) -> Path:
    """Resolve an instance stem to whichever corpus directory holds it."""
    for d in _CORPUS_DIRS:
        p = d / f"{instance}.nl"
        if p.exists():
            return p
    raise FileNotFoundError(f"{instance}.nl not found in {[str(d) for d in _CORPUS_DIRS]}")


def _short_sha() -> str:
    """Short git SHA of the tree being measured, or ``nogit``.

    Recorded in the artifact name AND inside it: a baseline is only meaningful
    against the commit it was taken on, and a file named after a commit that is
    not the commit measured is worse than an unnamed one.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "nogit"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def _load1() -> float:
    """1-minute load average, or ``nan`` where the platform has none."""
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # pragma: no cover - platform without loadavg
        return float("nan")


# --------------------------------------------------------------------------- #
# Child: solve ONE instance on DEFAULT settings, print a single JSON line.     #
#                                                                             #
# One subprocess per instance so env / JAX / module-global counter state is    #
# fully isolated: a baseline in which instance N's numbers depend on whether   #
# instance N-1 ran first is not a baseline. NOTHING here sets a DISCOPT_* flag #
# — the whole point of this panel is "default settings".                       #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: PLC0415

    nl = str(instance_path(instance))
    out: dict = {
        "instance": instance,
        # CLAUDE.md §8: record WHICH discopt was loaded. A baseline taken against
        # a stale site-packages install and compared against a worktree is a
        # measurement of nothing, and it has happened here.
        "discopt_file": discopt.__file__,
        "budget": float(budget),
    }
    try:
        model = from_nl(nl)
        out["sense"] = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        t0 = time.perf_counter()
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["gap"] = None if r.gap is None else float(r.gap)
        out["gap_certified"] = bool(r.gap_certified)
        out["node_count"] = int(r.node_count)
        # Root-gap instrumentation (plan task 0.3). Straight off SolveResult; the
        # reference-relative form is computed in the parent, which owns the oracle.
        out["root_bound"] = None if r.root_bound is None else float(r.root_bound)
        out["root_gap"] = None if r.root_gap is None else float(r.root_gap)
        out["root_time"] = None if r.root_time is None else float(r.root_time)
        out["convex_fast_path"] = bool(r.convex_fast_path)
        out["nlp_bb"] = bool(r.nlp_bb)
    except Exception as exc:
        # Errors are LABELLED, never dropped: an instance that stopped parsing is
        # a regression, and a panel that quietly shrinks its corpus hides it.
        out["status"] = "errored"
        out["error"] = repr(exc)

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _solve_one(instance: str, budget: float) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance, str(budget)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=budget + _CHILD_TIMEOUT_SLACK,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "status": "child_timeout", "budget": float(budget)}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "status": "child_crashed",
        "budget": float(budget),
        "stderr_tail": proc.stderr[-800:],
    }


def _annotate(row: dict, budget: float, oracle) -> dict:
    """Add the derived Regime-N and root-gap fields to a raw child row.

    ``oracle`` is the ``reference_optima.reference_oracle`` callable (injected so
    the annotation is testable without the corpus).
    """
    status = str(row.get("status"))
    wall = row.get("wall")
    certified = bool(row.get("gap_certified"))

    reason = None
    if status in _BAD_STATUSES:
        reason = f"status={status}"
    elif status not in _TERMINAL_STATUSES:
        reason = f"status={status} is budget-dependent"
    elif not certified:
        reason = "gap_certified=False (bound is heuristic; objective is not a certificate)"
    elif not isinstance(wall, (int, float)):
        reason = "no wall time recorded"
    elif wall >= _MARGIN_FRAC * budget:
        reason = (
            f"certified at {wall:.1f}s = {wall / budget:.0%} of the {budget:.0f}s budget "
            f"(>= {_MARGIN_FRAC:.0%} margin; too close to the limit to be reproducible)"
        )
    row["comparable"] = reason is None
    row["comparable_reason"] = reason

    # ---- root-gap instrumentation (plan task 0.3) ---------------------------
    ref = oracle(row["instance"])
    row["reference_optimum"] = None if ref is None else float(ref.value)
    row["reference_source"] = None if ref is None else ref.source
    row["reference_proven"] = None if ref is None else bool(ref.proven)
    rb = row.get("root_bound")
    if ref is not None and isinstance(rb, (int, float)) and math.isfinite(rb):
        # Same floored-relative convention as SolveResult.root_gap, but measured
        # against the ORACLE rather than this run's incumbent, so a root-bound
        # change is not confounded with a primal-heuristic change.
        row["root_gap_vs_reference"] = abs(float(ref.value) - rb) / max(1.0, abs(float(ref.value)))
        row["root_gap_reference_source"] = ref.source
    else:
        row["root_gap_vs_reference"] = None
        row["root_gap_reference_source"] = None
    return row


def _print_row(idx: int, total: int, row: dict) -> None:
    """One line per instance, flushed (CLAUDE.md §10: never silent for long stretches)."""
    rg = row.get("root_gap_vs_reference")
    rg_s = "-" if rg is None else f"{rg:.4g}"
    wall = row.get("wall")
    wall_s = "   -  " if not isinstance(wall, (int, float)) else f"{wall:6.1f}"
    print(
        f"  [{idx:3d}/{total}] {row['instance']:24s} "
        f"{str(row.get('status')):11s} "
        f"nodes={str(row.get('node_count', '-')):>8s} "
        f"obj={str(row.get('objective'))[:14]:>14s} "
        f"bound={str(row.get('bound'))[:14]:>14s} "
        f"cert={'Y' if row.get('gap_certified') else '.'} "
        f"rootgap*={rg_s:>10s} "
        f"w={wall_s} "
        f"{'CMP' if row.get('comparable') else '   '}",
        flush=True,
    )


def _resolve_subset(instances: list[str], subset: str | None) -> list[str]:
    """``--subset`` is either an integer count or a comma-separated name list."""
    if not subset:
        return instances
    s = subset.strip()
    if s.isdigit():
        return instances[: int(s)]
    wanted = [w.strip() for w in s.split(",") if w.strip()]
    known = set(instances)
    missing = [w for w in wanted if w not in known]
    if missing:
        # Loud refusal (CLAUDE.md §3): a typo'd name must not silently shrink the
        # panel to nothing and then report a clean pass.
        raise SystemExit(f"ERROR: --subset names not in the corpus: {', '.join(missing)}")
    return [i for i in instances if i in set(wanted)]


def _oracle_fn():
    from utils.reference_optima import reference_oracle  # noqa: PLC0415

    return reference_oracle


def _run_panel(instances: list[str], budget: float, label: str) -> tuple[list[dict], dict]:
    """Solve every instance once; return (rows, run metadata)."""
    oracle = _oracle_fn()
    load_start = _load1()
    load_peak = 0.0 if math.isnan(load_start) else load_start
    t_start = time.perf_counter()

    print(
        f"{label}: {len(instances)} instance(s), {budget:.0f}s budget, DEFAULT settings, "
        f"one subprocess per instance.",
        flush=True,
    )
    print(f"1-min load at start: {load_start:.2f} (recorded, not gated).\n", flush=True)

    rows: list[dict] = []
    for i, inst in enumerate(instances, 1):
        row = _annotate(_solve_one(inst, budget), budget, oracle)
        rows.append(row)
        lv = _load1()
        if not math.isnan(lv):
            load_peak = max(load_peak, lv)
        _print_row(i, len(instances), row)

    meta = {
        "budget_seconds": budget,
        "total_wall_seconds": time.perf_counter() - t_start,
        "load_start": load_start,
        "load_peak": load_peak,
        "python": sys.version.split()[0],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return rows, meta


def _root_gap_summary(rows: list[dict]) -> dict:
    """Coverage + distribution of the root-gap instrumentation (plan task 0.3)."""
    solver_pop = [r for r in rows if r.get("root_gap") is not None]
    ref_pop = [r for r in rows if r.get("root_gap_vs_reference") is not None]
    with_ref = [r for r in rows if r.get("reference_optimum") is not None]
    vals = sorted(float(r["root_gap_vs_reference"]) for r in ref_pop)
    median = None
    if vals:
        n = len(vals)
        median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {
        "rows": len(rows),
        "with_reference_optimum": len(with_ref),
        "root_gap_populated": len(solver_pop),
        "root_gap_populated_fraction": (len(solver_pop) / len(rows)) if rows else 0.0,
        "root_gap_vs_reference_populated": len(ref_pop),
        "root_gap_vs_reference_fraction": (len(ref_pop) / len(rows)) if rows else 0.0,
        "root_gap_vs_reference_median": median,
    }


def _status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get("status"))] = counts.get(str(r.get("status")), 0) + 1
    return dict(sorted(counts.items()))


def _print_summary(rows: list[dict], meta: dict) -> None:
    counts = _status_counts(rows)
    n_cmp = sum(1 for r in rows if r.get("comparable"))
    rg = _root_gap_summary(rows)
    rg_med = rg["root_gap_vs_reference_median"]
    rg_med_s = "-" if rg_med is None else f"{rg_med:.4g}"
    print("\n" + "=" * 78, flush=True)
    print(f"instances       : {len(rows)}", flush=True)
    print(f"statuses        : {counts}", flush=True)
    print(
        f"comparable rows : {n_cmp}/{len(rows)} "
        f"(certified terminal within {_MARGIN_FRAC:.0%} of budget — the Regime-N population)",
        flush=True,
    )
    print(
        f"root_gap        : {rg['root_gap_populated']}/{rg['rows']} populated "
        f"({rg['root_gap_populated_fraction']:.0%}); "
        f"vs reference {rg['root_gap_vs_reference_populated']}/{rg['rows']} "
        f"({rg['root_gap_vs_reference_fraction']:.0%}, "
        f"{rg['with_reference_optimum']} rows have an oracle); "
        f"median={rg_med_s}",
        flush=True,
    )
    print(
        f"wall            : {meta['total_wall_seconds']:.1f}s total; "
        f"load start {meta['load_start']:.2f} peak {meta['load_peak']:.2f}",
        flush=True,
    )
    print("=" * 78, flush=True)


def cmd_baseline(args: argparse.Namespace) -> int:
    instances = _resolve_subset(corpus_instances(), args.subset)
    if not instances:
        print("ERROR: corpus resolved to zero instances — nothing was measured.", flush=True)
        return 2

    rows, meta = _run_panel(instances, args.budget, "panel_baseline")
    sha = _short_sha()
    artifact = {
        "schema": "panel_baseline/1",
        "git_sha": sha,
        "git_dirty": _git_dirty(),
        "corpus_dirs": [str(d.relative_to(_REPO_ROOT)) for d in _CORPUS_DIRS],
        "margin_frac": _MARGIN_FRAC,
        "obj_tol": _OBJ_TOL,
        "obj_rtol": _OBJ_RTOL,
        **meta,
        "status_counts": _status_counts(rows),
        "comparable_count": sum(1 for r in rows if r.get("comparable")),
        "root_gap_summary": _root_gap_summary(rows),
        "rows": rows,
    }

    out = Path(args.out) if args.out else _REPORTS_DIR / f"panel_baseline_{sha}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=False) + "\n")

    _print_summary(rows, meta)
    if artifact["git_dirty"]:
        print(
            "WARNING: the working tree was DIRTY when this baseline was taken; the "
            f"artifact is named for {sha} but does not necessarily describe it.",
            flush=True,
        )
    print(f"\nBaseline written: {out}", flush=True)
    if artifact["comparable_count"] == 0:
        print(
            "ERROR: zero comparable rows — this baseline can never gate anything "
            "(every row is time-limited, uncertified, or errored).",
            flush=True,
        )
        return 1
    return 0


def _obj_match(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= _OBJ_TOL + _OBJ_RTOL * max(abs(a), abs(b))


def cmd_check(args: argparse.Namespace) -> int:
    """Re-run the baseline's instances and fail on any Regime-N drift."""
    path = Path(args.check)
    base = json.loads(path.read_text())
    if base.get("schema") != "panel_baseline/1":
        raise SystemExit(f"ERROR: {path} is not a panel_baseline/1 artifact.")
    base_rows = {r["instance"]: r for r in base["rows"]}
    budget = float(args.budget) if args.budget_explicit else float(base["budget_seconds"])
    if budget != float(base["budget_seconds"]):
        print(
            f"WARNING: re-running at {budget:.0f}s against a baseline taken at "
            f"{float(base['budget_seconds']):.0f}s. Node counts on budget-dependent rows "
            f"are not comparable across budgets.",
            flush=True,
        )

    instances = _resolve_subset(sorted(base_rows), args.subset)
    print(
        f"panel_baseline --check against {path.name} "
        f"(baseline sha {base.get('git_sha')}, current sha {_short_sha()})",
        flush=True,
    )
    rows, meta = _run_panel(instances, budget, "panel_baseline --check")
    new_rows = {r["instance"]: r for r in rows}

    # ---- comparisons -------------------------------------------------------
    # Counted, not assumed. A checker that executes zero comparisons and prints
    # "no drift" is the exact failure CLAUDE.md §6 is about, so the counts are
    # part of the verdict and zero is a FAILURE.
    n_node_cmp = 0
    n_obj_cmp = 0
    n_status_cmp = 0
    failures: list[str] = []
    soft: list[str] = []
    missing: list[str] = []

    for inst in instances:
        b = base_rows[inst]
        n = new_rows.get(inst)
        if n is None:  # pragma: no cover - _run_panel always emits a row
            missing.append(inst)
            continue
        if not b.get("comparable"):
            # Soft population: reported, never fatal. Reason is carried through so
            # a reader can see WHY this row is not gating.
            if int(b.get("node_count", -1)) != int(n.get("node_count", -2)) or not _obj_match(
                b.get("objective"), n.get("objective")
            ):
                soft.append(
                    f"{inst}: [non-comparable: {b.get('comparable_reason')}] "
                    f"nodes {b.get('node_count')}->{n.get('node_count')}, "
                    f"obj {b.get('objective')}->{n.get('objective')}"
                )
            continue

        # Hard population.
        n_status_cmp += 1
        if str(n.get("status")) != str(b.get("status")):
            failures.append(
                f"{inst}: STATUS drift {b.get('status')} -> {n.get('status')} "
                f"(baseline certified in {b.get('wall', float('nan')):.1f}s)"
            )
        if not n.get("gap_certified", False):
            failures.append(
                f"{inst}: CERTIFICATION LOST — baseline gap_certified=True, now False"
            )

        n_node_cmp += 1
        if int(b["node_count"]) != int(n.get("node_count", -1)):
            failures.append(
                f"{inst}: NODE COUNT drift {b['node_count']} -> {n.get('node_count')} "
                f"(Regime N requires exactly unchanged, improvement included)"
            )

        n_obj_cmp += 1
        if not _obj_match(b.get("objective"), n.get("objective")):
            failures.append(
                f"{inst}: CERTIFIED OBJECTIVE drift {b.get('objective')} -> "
                f"{n.get('objective')} (tol {_OBJ_TOL:g} + {_OBJ_RTOL:g}·|obj|)"
            )

    total_cmp = n_node_cmp + n_obj_cmp + n_status_cmp
    _print_summary(rows, meta)
    print(
        f"\ncomparisons executed: {total_cmp} "
        f"(node_count {n_node_cmp}, certified objective {n_obj_cmp}, status {n_status_cmp}) "
        f"over {sum(1 for i in instances if base_rows[i].get('comparable'))} comparable "
        f"of {len(instances)} baseline row(s)",
        flush=True,
    )
    if soft:
        print(f"\nNON-COMPARABLE drift ({len(soft)}) — reported, not gating:", flush=True)
        for s in soft:
            print(f"  - {s}", flush=True)
    if missing:
        failures.append(f"instances present in baseline but not re-run: {', '.join(missing)}")

    if total_cmp == 0:
        print(
            "\nFAIL: ZERO comparisons executed. This check proved nothing — a probe that "
            "compares nothing and reports no drift is a no-op that reads as a pass "
            "(CLAUDE.md §6).",
            flush=True,
        )
        return 3

    if failures:
        print(f"\nFAIL: {len(failures)} Regime-N violation(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1

    print("\nPASS: no node-count or certified-objective drift.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == "--solve":
        return _run_child(argv[1], float(argv[2]))

    p = argparse.ArgumentParser(
        description="Phase 0 Regime-N baseline over the in-repo .nl corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--check",
        metavar="BASELINE.json",
        help="re-run and fail non-zero on any node-count / certified-objective drift",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=_DEFAULT_BUDGET,
        help=f"per-instance wall budget in seconds (default {_DEFAULT_BUDGET:.0f}); "
        "in --check mode the baseline's own budget is used unless this is given",
    )
    p.add_argument(
        "--subset",
        help="an integer (first N instances) or a comma-separated list of instance names",
    )
    p.add_argument("--out", help="output path (default reports/panel_baseline_<sha>.json)")
    args = p.parse_args(argv)
    args.budget_explicit = any(a == "--budget" or a.startswith("--budget=") for a in argv)

    if args.check:
        return cmd_check(args)
    return cmd_baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
