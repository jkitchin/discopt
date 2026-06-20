#!/usr/bin/env python3
"""Global-optimization head-to-head: discopt vs the .nl-native solvers.

Runs every vendored MINLPLib ``.nl`` instance
(``python/tests/data/minlplib_nl/*.nl``) through **discopt, HiGHS, SCIP and
Couenne** under one per-problem budget, checks each against the MINLPLib
``primalbound`` oracle, and writes a markdown + JSON report.

These three external solvers all read the ``.nl`` directly, so unlike the
BARON-via-GAMS path (see ``global_opt_baron_vs_discopt.py``) they need no
format conversion. We reuse the benchmark runner's command-builder and
solver-specific output parsers (``_build_command`` / ``_parse_external_output``)
rather than reinventing them, and reuse the discopt subprocess worker and the
correctness oracle from the BARON harness so both reports are comparable.

Note on HiGHS: the AMPL-ASL HiGHS is an LP/MILP solver — it cannot handle the
nonlinear instances and will report ERROR/INFEASIBLE on them. That is expected
and shows up plainly in the report (it is a reference for the linear subset).

Usage:
    python -m discopt_benchmarks.scripts.global_opt_nl_solvers \
        [--time-limit 60] [--solvers highs,scip,couenne] [--instances a,b]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# runner.py uses top-level `benchmarks.*` / `utils.*` imports, so the
# discopt_benchmarks/ directory itself must be on sys.path (the same way
# run_benchmarks.py is invoked).
_DB = Path(__file__).resolve().parents[1]
if str(_DB) not in sys.path:
    sys.path.insert(0, str(_DB))

from discopt_benchmarks.scripts.global_opt_baron_vs_discopt import (  # noqa: E402
    GAP,
    NA,
    OK,
    REPO,
    VIOLATION,
    SolverRun,
    all_instances,
    classify,
    fmt,
    load_known_optima,
    nl_is_maximize,
    run_discopt,
)

from benchmarks.runner import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    SolverConfig,
)

CONFIG_TOML = REPO / "discopt_benchmarks" / "config" / "benchmarks.toml"
DEFAULT_EXTERNAL = ["highs", "scip", "couenne"]


def load_solver_commands() -> dict[str, str]:
    with open(CONFIG_TOML, "rb") as f:
        cfg = tomllib.load(f)
    return {name: s["command"] for name, s in cfg.get("solvers", {}).items() if "command" in s}


def make_runner(tl: float) -> BenchmarkRunner:
    """A runner instance used purely for its command-builder + parsers."""
    return BenchmarkRunner(BenchmarkConfig(suite_name="global_opt_nl", time_limit=int(tl)))


def run_external(runner: BenchmarkRunner, solver: SolverConfig, name: str, tl: float) -> SolverRun:
    """Drive one .nl-native solver via the runner's command + parse logic."""
    nl = runner._find_nl_file(name)
    if nl is None:
        return SolverRun(status="NO_NL", error="no .nl file resolved")
    cmd = runner._build_command(solver, name)
    # AMPL-ASL solvers (couenne/highs) write `<stub>.sol` next to the input .nl;
    # snapshot the dir so we can remove any solver-written .sol and not pollute
    # the vendored test-data directory.
    nl_dir = Path(nl).resolve().parent
    before = {p.name for p in nl_dir.glob("*.sol")}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=tl + 30)
    except subprocess.TimeoutExpired:
        return SolverRun(status="TIME_LIMIT", wall_time=tl + 30, error="outer-timeout")
    except FileNotFoundError:
        return SolverRun(status="NO_BINARY", error=f"missing executable: {cmd[0]}")
    finally:
        for p in nl_dir.glob("*.sol"):
            if p.name not in before:
                p.unlink(missing_ok=True)
    elapsed = time.perf_counter() - t0
    res = runner._parse_external_output(solver.name, name, proc.stdout, proc.stderr, elapsed)
    return SolverRun(
        status=str(getattr(res.status, "value", res.status)),
        objective=(None if res.objective is None else float(res.objective)),
        lower_bound=(None if res.bound is None else float(res.bound)),
        node_count=int(getattr(res, "node_count", 0) or 0),
        wall_time=elapsed,
    )


def write_report(
    rows: list[dict], solver_order: list[str], tl: float, out_dir: Path, ts: str
) -> Path:
    md = out_dir / f"global_opt_nl_solvers_{ts}.md"
    n_oracle = sum(r["known"] is not None for r in rows)

    # per-solver verdict tallies (honest OK / GAP / VIOLATION / n/a)
    tallies = {s: {OK: 0, GAP: 0, VIOLATION: 0, NA: 0} for s in solver_order}
    for r in rows:
        for s in solver_order:
            tallies[s][r["runs"][s]["verdict"]] += 1

    lines = [
        "# Global Optimization Benchmark — discopt vs HiGHS / SCIP / Couenne",
        "",
        f"- Instances: **{len(rows)}** vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`)",
        f"- Per-problem time limit: **{int(tl)} s**; correctness tol abs=1e-6 "
        "rel=1e-4 vs MINLPLib `primalbound`",
        "- discopt: isolated subprocess `Model.solve`; HiGHS/SCIP/Couenne read "
        "the `.nl` directly (runner command-builder + parsers)",
        f"- Instances with a known optimum (oracle): **{n_oracle}/{len(rows)}**",
        f"- Generated: {ts}",
        "",
        "> HiGHS is LP/MILP-only (AMPL-ASL build); it errors on nonlinear "
        "instances by design — a reference for the linear subset, not a global "
        "MINLP solver.",
        "",
        "## Correctness summary (vs known optimum)",
        "",
        "| verdict | meaning |",
        "|---|---|",
        "| `ok` | incumbent matches the known global within tolerance |",
        "| `GAP` | honest feasible incumbent **worse** than the global — a "
        "convergence gap in the time budget, *not* a correctness bug |",
        "| `VIOLATION` | **the red line**: claimed a certified global with the "
        "wrong value, or returned an incumbent strictly *better* than the proven "
        "global (impossible → bug) |",
        "| `n/a` | no oracle, or no incumbent returned |",
        "",
        "| solver | ok | GAP | VIOLATION | n/a |",
        "|---|---|---|---|---|",
    ]
    for s in solver_order:
        t = tallies[s]
        lines.append(f"| {s} | {t[OK]}/{n_oracle} | {t[GAP]} | {t[VIOLATION]} | {t[NA]} |")

    # per-instance: obj+verdict+time per solver
    head = "| instance | known |"
    sep = "|---|---|"
    for s in solver_order:
        head += f" {s} obj | v | t |"
        sep += "---|---|---|"
    lines += ["", "## Per-instance results", "", head, sep]
    for r in sorted(rows, key=lambda x: x["instance"]):
        cells = f"| {r['instance']} | {fmt(r['known'])} |"
        for s in solver_order:
            run = r["runs"][s]
            cells += f" {fmt(run['objective'])} | {run['verdict']} | {run['wall_time']:.2f} |"
        lines.append(cells)

    md.write_text("\n".join(lines) + "\n")
    import json

    (out_dir / f"global_opt_nl_solvers_{ts}.json").write_text(
        json.dumps(
            {"time_limit": tl, "timestamp": ts, "solvers": solver_order, "rows": rows}, indent=2
        )
    )
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument(
        "--solvers",
        type=str,
        default=",".join(DEFAULT_EXTERNAL),
        help="external .nl solvers to include (default: highs,scip,couenne)",
    )
    ap.add_argument("--instances", type=str, default="")
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = load_known_optima()
    insts = (
        [s.strip() for s in args.instances.split(",") if s.strip()]
        if args.instances
        else all_instances()
    )
    ext_names = [s.strip() for s in args.solvers.split(",") if s.strip()]
    commands = load_solver_commands()
    solver_order = ["discopt"] + ext_names

    # validate external binaries up front; warn (don't abort) if missing
    runner = make_runner(args.time_limit)
    ext_cfgs: dict[str, SolverConfig] = {}
    for nm in ext_names:
        cmd = commands.get(nm, nm)
        if not Path(cmd).exists():
            print(f"# WARNING: {nm} binary not found at {cmd} — will report NO_BINARY", flush=True)
        ext_cfgs[nm] = SolverConfig(name=nm, command=cmd, solver_type="external", nl_interface=True)

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    print(
        f"# global-opt .nl-solver head-to-head: {len(insts)} instances, "
        f"solvers={solver_order}, time_limit={int(args.time_limit)}s",
        flush=True,
    )

    rows: list[dict] = []
    for i, name in enumerate(insts, 1):
        kn = known.get(name)
        mx = nl_is_maximize(name)
        runs: dict[str, dict] = {}
        d = run_discopt(name, args.time_limit)
        runs["discopt"] = {**vars(d), "verdict": classify(d.status, d.objective, kn, mx)}
        for nm in ext_names:
            r = run_external(runner, ext_cfgs[nm], name, args.time_limit)
            runs[nm] = {**vars(r), "verdict": classify(r.status, r.objective, kn, mx)}
        rows.append({"instance": name, "known": kn, "maximize": mx, "runs": runs})
        cells = " | ".join(
            f"{s}:{fmt(runs[s]['objective'], 8)} {runs[s]['verdict']:9} {runs[s]['wall_time']:.1f}s"
            for s in solver_order
        )
        print(f"[{i:2}/{len(insts)}] {name:20} {cells}", flush=True)

    md = write_report(rows, solver_order, args.time_limit, out_dir, ts)
    # The red line is a VIOLATION (wrong certified value / impossible bound), NOT
    # an honest GAP (feasible incumbent that simply didn't converge in budget).
    violations = sum(r["runs"]["discopt"]["verdict"] == VIOLATION for r in rows)
    print(f"\n# DONE. discopt VIOLATIONS {violations}. Report: {md}", flush=True)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
