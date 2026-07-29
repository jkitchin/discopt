"""Phase 5.4 entry experiment — does a non-certifying convex-kernel attempt add wall?

Consolidation plan Phase 5.4 names one measured hazard as the thing that must be made
"strictly safe" before ``DISCOPT_CONVEX_KERNEL`` can graduate default-ON:
``watercontamination0202`` classifies convex in 2.9 s and then runs **2001 s with no
bound** against 49 s on the spatial path (`sota-parity-analysis-2026-07-27.md` §3 G-C).

**Hypothesis under test.** That is not an instance quirk — it is the *budget
arithmetic*. ``Model.solve`` gives the convex kernel ``min(time_limit,
DISCOPT_CONVEX_KERNEL_BUDGET=120)`` seconds (`_convex_kernel.py:717`), adopts the
result **only** when it certifies, and then calls ``solve_model`` with the caller's
**full** ``time_limit`` again (`modeling/core.py:4201-4207`) — the elapsed kernel time
is never deducted. So any eligible-but-uncertifiable model pays its whole default-path
budget **plus** the kernel attempt.

**Falsifying experiment (this script).** Run every convex-kernel-eligible in-repo
instance with the flag OFF and ON, arms interleaved, N replicates, subprocess-isolated,
and compare *total* wall against the requested budget.

**Kill criterion.** If no eligible instance's ON-arm wall materially exceeds its budget
(> ``budget + 5 s``) while OFF's does not, the additive-budget hazard is NOT reproducible
on the in-repo corpus and the fix must be re-scoped rather than built.

Eligibility is measured, not assumed: ``build_convex_spec`` is run over the whole corpus
and only the instances it accepts are panelled. Executed counts are printed and a run
that measured nothing exits non-zero (CLAUDE.md §6).

Usage::

    python -u discopt_benchmarks/scripts/phase5_convex_kernel_budget_entry.py
    python -u discopt_benchmarks/scripts/phase5_convex_kernel_budget_entry.py --budget 30 --reps 2

Internal child mode: ``--solve <instance> <0|1> <budget>``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "discopt_benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.panel_baseline import (  # noqa: E402
    _load1,
    _short_sha,
    corpus_instances,
    instance_path,
)

_REPORTS_DIR = _REPO_ROOT / "reports"
_CHILD_SLACK = 240.0


def _run_child(instance: str, flag: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Both arms set EXPLICITLY: inferring an arm from a default the harness does not
    # control is how a panel silently compares OFF against OFF (#902).
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1" if flag == "1" else "0"

    import discopt  # noqa: PLC0415
    from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: PLC0415
    from discopt.solvers import _convex_kernel  # noqa: PLC0415

    out: dict = {
        "instance": instance,
        "flag": flag,
        "budget": float(budget),
        "discopt_file": discopt.__file__,
        # CLAUDE.md §8: record the arm the loaded module actually reports, not the
        # one we asked for.
        "kernel_enabled": bool(_convex_kernel.convex_kernel_enabled()),
    }
    try:
        model = from_nl(str(instance_path(instance)))
        out["sense"] = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        t0 = time.perf_counter()
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["node_count"] = int(r.node_count)
        out["gap_certified"] = bool(r.gap_certified)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _solve(instance: str, flag: str, budget: float) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve",
        instance,
        flag,
        str(budget),
    ]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget * 3 + _CHILD_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "flag": flag, "status": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "flag": flag,
        "status": "child_crashed",
        "stderr_tail": proc.stderr[-600:],
    }


def _eligible() -> list[str]:
    """Instances ``build_convex_spec`` actually accepts — measured, not assumed."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--eligible"]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["DISCOPT_CONVEX_KERNEL"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    for line in proc.stdout.splitlines():
        if line.startswith("ELIGIBLE_JSON "):
            return json.loads(line[len("ELIGIBLE_JSON ") :])
    raise RuntimeError(f"eligibility sweep produced nothing: {proc.stderr[-800:]}")


def _run_eligibility() -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1"
    from discopt.modeling.core import from_nl  # noqa: PLC0415
    from discopt.solvers._convex_kernel import build_convex_spec  # noqa: PLC0415

    ok: list[str] = []
    for stem in corpus_instances():
        try:
            if build_convex_spec(from_nl(str(instance_path(stem)))) is not None:
                ok.append(stem)
        except Exception as exc:  # labelled, never silently dropped
            print(f"# eligibility error {stem}: {exc!r}", file=sys.stderr, flush=True)
    print("ELIGIBLE_JSON " + json.dumps(ok), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", nargs=3, metavar=("INSTANCE", "FLAG", "BUDGET"))
    ap.add_argument("--eligible", action="store_true")
    ap.add_argument("--budget", type=float, default=45.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated instance subset")
    args = ap.parse_args(argv)
    if args.solve:
        return _run_child(args.solve[0], args.solve[1], float(args.solve[2]))
    if args.eligible:
        return _run_eligibility()

    load_start = _load1()
    if args.only:
        insts = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        print("measuring convex-kernel eligibility over the in-repo corpus ...", flush=True)
        insts = _eligible()
    print(f"eligible instances ({len(insts)}): {insts}", flush=True)
    if not insts:
        print("FAIL: no eligible instance — nothing to measure", file=sys.stderr)
        return 2

    rows: list[dict] = []
    executed = 0
    for rep in range(args.reps):
        for inst in insts:
            # Interleave the arms within a replicate (CLAUDE.md §9): a sequential
            # all-OFF-then-all-ON layout attributes any load drift to the flag.
            for flag in ("0", "1"):
                r = _solve(inst, flag, args.budget)
                r["rep"] = rep
                rows.append(r)
                executed += 1
                print(
                    f"  rep{rep} {inst:<22} flag={flag} status={r.get('status'):<12} "
                    f"wall={r.get('wall', float('nan')):.2f}s "
                    f"obj={r.get('objective')} enabled={r.get('kernel_enabled')}",
                    flush=True,
                )
    load_end = _load1()

    # ---- verdict ---------------------------------------------------------- #
    hazard: list[str] = []
    summary: list[dict] = []
    comparisons = 0
    for inst in insts:
        off = [r for r in rows if r["instance"] == inst and r.get("flag") == "0"]
        on = [r for r in rows if r["instance"] == inst and r.get("flag") == "1"]
        offw = [r["wall"] for r in off if r.get("wall") is not None]
        onw = [r["wall"] for r in on if r.get("wall") is not None]
        if not offw or not onw:
            continue
        comparisons += 1
        s = {
            "instance": inst,
            "off_wall_mean": statistics.fmean(offw),
            "on_wall_mean": statistics.fmean(onw),
            "off_wall_sd": statistics.pstdev(offw) if len(offw) > 1 else 0.0,
            "on_wall_sd": statistics.pstdev(onw) if len(onw) > 1 else 0.0,
            "off_status": [r.get("status") for r in off],
            "on_status": [r.get("status") for r in on],
            "off_obj": [r.get("objective") for r in off],
            "on_obj": [r.get("objective") for r in on],
        }
        summary.append(s)
        # The hazard: ON overruns the REQUESTED budget while OFF respects it.
        if s["on_wall_mean"] > args.budget + 5.0 and s["off_wall_mean"] <= args.budget + 5.0:
            hazard.append(
                f"{inst}: ON {s['on_wall_mean']:.1f}s (sd {s['on_wall_sd']:.2f}) vs "
                f"OFF {s['off_wall_mean']:.1f}s (sd {s['off_wall_sd']:.2f}) "
                f"on a {args.budget:.0f}s budget"
            )

    print("\n" + "=" * 92)
    print("ENTRY EXPERIMENT — convex-kernel budget additivity")
    print("=" * 92)
    print(f"{'instance':<24}{'OFF wall':>12}{'ON wall':>12}{'delta':>10}   statuses")
    for s in summary:
        print(
            f"{s['instance']:<24}{s['off_wall_mean']:>10.2f}s{s['on_wall_mean']:>10.2f}s"
            f"{s['on_wall_mean'] - s['off_wall_mean']:>9.2f}s   "
            f"OFF={s['off_status']} ON={s['on_status']}"
        )
    print(f"\nexecuted solves: {executed};  paired comparisons: {comparisons}")
    print(f"load {load_start:.2f} -> {load_end:.2f};  budget {args.budget}s;  reps {args.reps}")
    verdict = "CONFIRMED" if hazard else "KILLED"
    print(f"\nVERDICT: additive-budget hazard {verdict}")
    for h in hazard:
        print(f"  - {h}")
    if not hazard:
        print(
            "  no eligible instance overran its budget under ON — the hazard is not "
            "reproducible on the in-repo corpus; re-scope rather than build."
        )

    _REPORTS_DIR.mkdir(exist_ok=True)
    out = (
        _REPORTS_DIR / f"phase5_convex_kernel_budget_entry_{_short_sha()}_b{int(args.budget)}.json"
    )
    out.write_text(
        json.dumps(
            {
                "schema": "phase5_convex_budget_entry/1",
                "git_sha": _short_sha(),
                "budget": args.budget,
                "reps": args.reps,
                "eligible": insts,
                "executed_solves": executed,
                "comparisons": comparisons,
                "load_start": load_start,
                "load_end": load_end,
                "verdict": verdict,
                "hazard": hazard,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifact: {out.relative_to(_REPO_ROOT)}")
    if executed == 0 or comparisons == 0:
        print("FAIL: zero executed comparisons", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
