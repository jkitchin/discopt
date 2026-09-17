#!/usr/bin/env python3
"""Run a CUTEst NLP benchmark suite: discopt's local NLP backends vs standalone Ipopt.

This is the *local-NLP quality* tier of the NLP benchmark. CUTEst problems carry
no global certificate, so the comparison is:

* **converged** -- the backend reported a KKT point (Ipopt/POUNCE ``OPTIMAL``);
* **wrong** -- converged, but to an objective that disagrees with a known
  optimum from ``[known_optima]`` in ``config/cutest_suites.toml`` (a hard
  failure: exit status 1);
* **differs** -- converged to a different objective than ``ipopt_standalone``
  (reported, not failed: two local solvers may find different local minima);
* **lost** -- ``ipopt_standalone`` converged and the discopt arm did not.

``ipopt_standalone`` drives cyipopt straight from PyCUTEst callbacks, so it
isolates discopt's evaluator/backend layer from the problem itself.

The global-certification NLP tier is the MINLPLib panel
(``run_benchmarks.py --suite nlp_smoke|nlp --use-cache``).

Usage:
    python -u discopt_benchmarks/scripts/run_cutest_benchmarks.py --suite cutest_smoke
    python -u discopt_benchmarks/scripts/run_cutest_benchmarks.py --suite cutest_nlp \\
        --output results/cutest_nlp.json
    python discopt_benchmarks/scripts/run_cutest_benchmarks.py --list

Requires CUTEst (``make setup-cutest`` then ``source ~/.local/cutest/env.sh``).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import statistics
import sys
import tomllib
from datetime import datetime
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

CONFIG_PATH = _BENCH_ROOT / "config" / "cutest_suites.toml"
REFERENCE_SOLVER = "ipopt_standalone"
# Objective agreement: |a - b| <= REL_TOL * max(1, |a|, |b|).
REL_TOL = 1e-4
# Shift for the shifted geometric mean. CUTEst solves are mostly milliseconds,
# so the MINLPLib 1 s shift would flatten every ratio to ~1.
SGM_SHIFT = 0.01


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def suite_config(config: dict, name: str, **overrides):
    """Build a ``CUTEstSuiteConfig`` from a ``[suites.<name>]`` table."""
    from benchmarks.cutest_runner import CUTEstSuiteConfig

    suites = config.get("suites", {})
    if name not in suites:
        raise SystemExit(f"unknown CUTEst suite {name!r}; have: {', '.join(sorted(suites))}")
    fields = {f.name for f in dataclasses.fields(CUTEstSuiteConfig)}
    table = dict(suites[name])
    unknown = set(table) - fields
    if unknown:
        raise SystemExit(f"suite {name!r} has unknown keys: {sorted(unknown)}")
    table.update({k: v for k, v in overrides.items() if v is not None})
    return CUTEstSuiteConfig(name=name, **table)


def objectives_agree(a: float, b: float, rel_tol: float = REL_TOL) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))


def shifted_geomean(times: list[float], shift: float = SGM_SHIFT) -> float | None:
    if not times:
        return None
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def summarize(results, known_optima: dict[str, float]) -> dict:
    """Score every solver arm. Pure function of a ``BenchmarkResults``."""
    from benchmarks.metrics import SolveStatus

    by_solver = {s: {r.instance: r for r in results.get_results(s)} for s in results.get_solvers()}
    ref = by_solver.get(REFERENCE_SOLVER, {})
    summary: dict = {"solvers": {}, "comparisons": 0, "problems": len(results.get_instances())}

    for solver, rows in by_solver.items():
        converged = [r for r in rows.values() if r.status == SolveStatus.OPTIMAL]
        limited = [r for r in rows.values() if r.status == SolveStatus.TIME_LIMIT]
        errors = [r for r in rows.values() if r.status == SolveStatus.ERROR]
        finite = [r.wall_time for r in rows.values() if math.isfinite(r.wall_time)]
        conv_times = [r.wall_time for r in converged]

        wrong, differs, lost = [], [], []
        for name, r in rows.items():
            if r.status == SolveStatus.OPTIMAL and name in known_optima:
                summary["comparisons"] += 1
                if r.objective is None or not objectives_agree(r.objective, known_optima[name]):
                    wrong.append((name, r.objective, known_optima[name]))
            if solver == REFERENCE_SOLVER or name not in ref:
                continue
            rr = ref[name]
            if rr.status != SolveStatus.OPTIMAL:
                continue
            summary["comparisons"] += 1
            if r.status != SolveStatus.OPTIMAL:
                lost.append((name, r.status.value, rr.objective))
            elif r.objective is None or not objectives_agree(r.objective, rr.objective):
                differs.append((name, r.objective, rr.objective))

        summary["solvers"][solver] = {
            "n": len(rows),
            "converged": len(converged),
            "limit": len(limited),
            "error": len(errors),
            "other": len(rows) - len(converged) - len(limited) - len(errors),
            "sgm_converged": shifted_geomean(conv_times),
            "median_converged": statistics.median(conv_times) if conv_times else None,
            "total_wall": sum(finite),
            "wrong": wrong,
            "differs": differs,
            "lost": lost,
        }
    return summary


def _fmt(t: float | None, unit: str = "s") -> str:
    return "—" if t is None else f"{t:.4f}{unit}"


def render_markdown(suite_name: str, summary: dict, errors: dict, meta: dict) -> str:
    lines = [
        f"# CUTEst NLP benchmark — `{suite_name}`",
        "",
        f"- run: {meta['timestamp']}",
        f"- problems: {summary['problems']}, time limit {meta['time_limit']} s",
        f"- reference: `{REFERENCE_SOLVER}`; agreement tol {REL_TOL:g} (rel, floor 1)",
        f"- executed comparisons: {summary['comparisons']}",
        f"- excluded as over-determined (more equalities than free variables): "
        f"{len(meta.get('excluded', {}))}",
        "",
        "| solver | converged | limit (time/iter/stall) | error | other | wrong | differs | lost "
        f"| SGM conv (shift {SGM_SHIFT:g}s) | median conv | total wall |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for solver, s in summary["solvers"].items():
        lines.append(
            f"| {solver} | {s['converged']}/{s['n']} | {s['limit']} | {s['error']} | "
            f"{s['other']} | {len(s['wrong'])} | {len(s['differs'])} | {len(s['lost'])} | "
            f"{_fmt(s['sgm_converged'])} | {_fmt(s['median_converged'])} | "
            f"{s['total_wall']:.2f}s |"
        )
    for solver, s in summary["solvers"].items():
        for key, title in (
            ("wrong", "WRONG vs known optimum (obj, known)"),
            ("lost", f"converged in {REFERENCE_SOLVER} only (status, ref obj)"),
            ("differs", f"different objective from {REFERENCE_SOLVER} (obj, ref obj)"),
        ):
            if s[key]:
                lines += ["", f"### {solver}: {title}", ""]
                lines += [f"- `{row[0]}`: {row[1]!r}, {row[2]!r}" for row in s[key]]
    if meta.get("excluded"):
        lines += ["", "### Excluded problems", ""]
        lines += [f"- `{name}`: {why}" for name, why in sorted(meta["excluded"].items())]
    if errors:
        lines += ["", "### Errors", ""]
        lines += [f"- `{solver}` `{prob}`: {msg}" for (solver, prob), msg in sorted(errors.items())]
    return "\n".join(lines) + "\n"


def _prepare_environment() -> None:
    """pycutest compiles into $PYCUTEST_CACHE; unset, it litters the CWD.

    A ``pycutest_cache_holder/`` left in the working directory also shadows the
    real cache on ``sys.path`` and makes later imports fail with
    ``ModuleNotFoundError``, so always pin it before pycutest is imported.
    """
    cache = Path(os.environ.setdefault("PYCUTEST_CACHE", str(Path.home() / ".cache/pycutest")))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LDFLAGS", "-Wl,-w")
    os.environ.setdefault("FFLAGS", "-w")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", default="cutest_smoke")
    p.add_argument("--list", action="store_true", help="list suites and exit")
    p.add_argument(
        "--solvers",
        default=None,
        help="comma-separated arms (default: [general].default_solvers); "
        "choices: discopt_pounce, discopt_ipopt, discopt_ipm, scipy, ipopt_standalone",
    )
    p.add_argument("--max-n", type=int, default=None, help="override max_variables")
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--time-limit", type=int, default=None, help="per-solve seconds")
    p.add_argument("--output", type=Path, default=None, help="JSON path (.md written beside)")
    args = p.parse_args(argv)

    config = load_config()
    if args.list:
        for name, table in config.get("suites", {}).items():
            print(f"  {name:28s} {table.get('description', '')}")
        return 0

    _prepare_environment()
    solvers = (
        args.solvers.split(",")
        if args.solvers
        else list(config.get("general", {}).get("default_solvers", []))
    )
    cfg = suite_config(
        config,
        args.suite,
        max_variables=args.max_n,
        max_instances=args.max_instances,
        time_limit_seconds=args.time_limit,
    )

    try:
        import pycutest

        pycutest.find_problems(constraints="unconstrained", n=[2, 2])
    except Exception as exc:  # report and stop: every solve would error the same way
        print(
            f"ERROR: CUTEst is not usable ({type(exc).__name__}: {exc}).\n"
            "Run `make setup-cutest` once, then `source ~/.local/cutest/env.sh`.",
            file=sys.stderr,
        )
        return 2

    from benchmarks.cutest_runner import CUTEstBenchmarkRunner

    runner = CUTEstBenchmarkRunner(cfg)
    problems = runner.discover_problems()
    print(f"[cutest] suite={cfg.name} problems={len(problems)} solvers={solvers}", flush=True)
    if not problems:
        print("ERROR: suite resolved to zero problems", file=sys.stderr)
        return 2
    runner.load_problem_info()
    results = runner.run_all(solvers=solvers, verbose=True)

    known = {k: float(v) for k, v in config.get("known_optima", {}).items()}
    summary = summarize(results, known)
    meta = {
        "timestamp": results.timestamp,
        "time_limit": cfg.time_limit_seconds,
        "excluded": runner.excluded,
    }
    report = render_markdown(cfg.name, summary, runner.errors, meta)
    print("\n" + report, flush=True)

    out = args.output or (
        _BENCH_ROOT / "reports" / f"cutest_{cfg.name}_{datetime.now():%Y%m%dT%H%M%S}.json"
    )
    results.save(out)
    out.with_suffix(".md").write_text(report)
    print(f"[cutest] wrote {out} and {out.with_suffix('.md')}")

    n_rows = sum(s["n"] for s in summary["solvers"].values())
    print(f"[cutest] executed solves: {n_rows}; executed comparisons: {summary['comparisons']}")
    n_errors = sum(s["error"] for s in summary["solvers"].values())
    if n_rows == 0 or n_errors == n_rows:
        print(f"ERROR: {n_errors}/{n_rows} solves errored; nothing was measured", file=sys.stderr)
        return 2
    if any(s["wrong"] for s in summary["solvers"].values()):
        print("FAIL: a converged objective disagrees with a known optimum", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
