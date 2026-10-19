"""G0.6 baseline: discopt's GDP routes vs Pyomo's GDP strategies (#1775, spec §1.2 / §8).

Runs every ``gdp_parity`` instance (:data:`gdplib_runner.GDP_PARITY`) under

* the **Pyomo arms**, profile O (open solvers: HiGHS MIP, Ipopt NLP, SCIP MINLP /
  global; ``scripts/install_open_gdp_solvers.sh``): ``gdp.bigm`` + SCIP,
  ``gdp.hull`` + SCIP, ``gdpopt.loa``, ``gdpopt.gloa`` (SCIP as the global NLP
  solver), ``gdpopt.lbb`` (SCIP as the MINLP solver), ``gdpopt.ric`` and
  ``gdpopt.enumerate``. GDPlib's ``benchmark.py`` has GAMS profiles only, so this
  script is profile O's runner;
* the **discopt arms**: ``from_pyomo_gdp`` then ``Model.solve(gdp_method=...)`` for
  ``big-m`` (the default route), ``hull``, ``mbigm``, ``auto`` and ``loa``.

Every (instance, arm) runs in a fresh subprocess, hard-killed at the time limit plus
a grace period, and appends one JSON line to ``--out`` (flushed): a restart skips the
lines already there. Arms are interleaved within each instance, the 1-minute load
average is recorded per run, and a run started above ``--max-load`` is marked
``load_gated`` (its wall time is not a timing claim, CLAUDE.md §9).

Each run is scored against :func:`gdplib_runner.panel_oracle`:

* ``solved`` -- claims optimal and agrees with the oracle;
* ``incorrect`` -- claims optimal and the oracle refutes it, or reports an
  incumbent strictly better than the oracle optimum (a false primal);
* ``claimed_unverified`` -- claims optimal on an instance whose oracle is unproven;
* ``unsolved`` -- anything else.

discopt incumbents are additionally checked in the original Pyomo GDP model, under
Pyomo's semantics (``gdp_convert_parity._pyomo_violation``).

    python -u discopt_benchmarks/scripts/gdp_baseline.py --out base.jsonl
    python -u discopt_benchmarks/scripts/gdp_baseline.py --out base.jsonl --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))

PYOMO_ARMS = (
    "pyomo-bigm",
    "pyomo-hull",
    "pyomo-loa",
    "pyomo-gloa",
    "pyomo-lbb",
    "pyomo-ric",
    "pyomo-enumerate",
)
DISCOPT_ARMS = ("discopt-big-m", "discopt-hull", "discopt-mbigm", "discopt-auto", "discopt-loa")
ARMS = PYOMO_ARMS + DISCOPT_ARMS

#: The spec §1.2 family pairing (discopt route -> Pyomo strategy).
FAMILIES = {
    "big-m": ("discopt-big-m", "pyomo-bigm"),
    "hull": ("discopt-hull", "pyomo-hull"),
    "loa": ("discopt-loa", "pyomo-loa"),
}

#: Profile O solver roles. The spec names HiGHS for MIP; GDPopt cannot use it here.
#: GDPopt solves every fixed-configuration subproblem with ``mip_solver`` and a
#: ``dual`` suffix attached, and on a subproblem that still has integer columns
#: (GDPlib ``ex1_linan_2023``) both Pyomo HiGHS interfaces raise instead of
#: returning no duals (appsi: "Solver does not currently have valid duals";
#: ``highs``: ``NoDualsError``), as does ``scip_direct`` ("failed to implement
#: required method 'get_duals'"). CBC, an open MIP solver driven through Pyomo's
#: classic shell interface, does not, so GDPopt's MIP role is CBC. The gdp.bigm /
#: gdp.hull arms attach no dual suffix and use SCIP through ``scip_direct``.
#: GLOA's global NLP solver must also tolerate the dual suffix, so it is the SCIP
#: *executable* (Pyomo ``scip``, built by install_open_gdp_solvers.sh), not
#: ``scip_direct``.
_MIP, _NLP, _GLOBAL, _GLOBAL_EXE = "cbc", "ipopt", "scip_direct", "scip"


# ---------------------------------------------------------------------------
# worker: one (instance, arm), run in its own process
# ---------------------------------------------------------------------------


def _spec(name: str):
    from benchmarks import gdplib_runner as gr

    specs = gr.discover_models(include=[name])
    if not specs:
        raise LookupError(f"no builder for {name!r}")
    return specs[0]


def _run_pyomo(model, arm: str, time_limit: float) -> dict:
    import pyomo.environ as pyo

    obj = next(model.component_data_objects(pyo.Objective, active=True))
    if arm in ("pyomo-bigm", "pyomo-hull"):
        pyo.TransformationFactory("gdp." + arm.split("-")[1]).apply_to(model)
        res = pyo.SolverFactory(_GLOBAL).solve(model, timelimit=time_limit, load_solutions=False)
        tc = str(res.solver.termination_condition)
        if len(res.solution) > 0:  # an incumbent exists (none at a bare time limit)
            model.solutions.load_from(res)
    else:
        strategy = arm.split("-", 1)[1]
        # A subproblem that keeps discrete columns goes to GDPopt's (local) MINLP
        # solver, whose default is BARON; give it SCIP (the executable, which
        # tolerates the dual suffix) so profile O stays open.
        kwargs = {
            "time_limit": time_limit,
            "mip_solver": _MIP,
            "minlp_solver": _GLOBAL_EXE,
            "local_minlp_solver": _GLOBAL_EXE,
        }
        if strategy == "gloa":
            kwargs["nlp_solver"] = _GLOBAL_EXE
        elif strategy == "lbb":
            kwargs.pop("mip_solver")
        else:
            kwargs["nlp_solver"] = _NLP
        try:
            res = pyo.SolverFactory("gdpopt." + strategy).solve(model, **kwargs)
        except AttributeError as err:
            # Pyomo 6.10 gdpopt.lbb: the time-limit exit calls a method that does
            # not exist (``_get_final_results_object``; the real one is
            # ``_get_final_pyomo_results_object``). Only that exact failure is a
            # time-limit outcome; anything else propagates as an error.
            if strategy != "lbb" or "_get_final_results_object" not in str(err):
                raise
            return {
                "status": "maxTimeLimit",
                "objective": None,
                "bound": None,
                "minimize": obj.sense == pyo.minimize,
                "note": "pyomo gdpopt.lbb time-limit path bug (_get_final_results_object)",
            }
        tc = str(res.solver.termination_condition)
    prob = res.problem[0] if hasattr(res.problem, "__getitem__") else res.problem
    minimize = obj.sense == pyo.minimize
    lb = getattr(prob, "lower_bound", None)
    ub = getattr(prob, "upper_bound", None)
    bound = lb if minimize else ub
    value = pyo.value(obj, exception=False)
    return {
        "status": tc,
        "objective": None if value is None else float(value),
        "bound": None if bound is None or not math.isfinite(float(bound)) else float(bound),
        "minimize": minimize,
    }


def _run_discopt(model, arm: str, time_limit: float) -> dict:
    import warnings

    from discopt.pyomo.gdp import convert_pyomo_gdp

    sys.path.insert(0, str(HERE.parent))
    from gdp_convert_parity import _pyomo_violation

    conv = convert_pyomo_gdp(model)
    method = arm.split("-", 1)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = conv.model.solve(gdp_method=method, time_limit=time_limit)
    out = {
        "status": r.status,
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None or not math.isfinite(r.bound) else float(r.bound),
        "minimize": conv.model._objective.sense.name == "MINIMIZE",
        "node_count": getattr(r, "node_count", None),
    }
    if r.x is not None and r.objective is not None:
        out["in_pyomo"] = _pyomo_violation(model, conv, r)
    return out


def worker(name: str, arm: str, time_limit: float) -> dict:
    import discopt

    rec: dict = {"instance": name, "arm": arm, "discopt": discopt.__file__}
    t_build = time.perf_counter()
    model = _spec(name).builder()
    rec["build_s"] = round(time.perf_counter() - t_build, 2)
    t0 = time.perf_counter()
    if arm in PYOMO_ARMS:
        rec.update(_run_pyomo(model, arm, time_limit))
    else:
        rec.update(_run_discopt(model, arm, time_limit))
    rec["wall"] = round(time.perf_counter() - t0, 3)
    return rec


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score(rec: dict, oracle) -> str:
    """``solved`` / ``incorrect`` / ``claimed_unverified`` / ``unsolved`` / ``error``."""
    if "error" in rec:
        return "error"
    claimed = str(rec.get("status", "")).lower() == "optimal"
    obj = rec.get("objective")
    bad_point = rec.get("in_pyomo") and (
        rec["in_pyomo"]["max_rel_violation"] > 1e-5
        or rec["in_pyomo"]["bad_counts"]
        or rec["in_pyomo"]["bad_logic"]
    )
    if oracle is None or oracle.value is None:
        if bad_point:
            return "incorrect"
        return "claimed_unverified" if claimed else "unsolved"
    opt = oracle.value
    tol = 1e-4 + 1e-3 * abs(opt)
    better = obj is not None and (obj < opt - tol if rec.get("minimize", True) else obj > opt + tol)
    if bad_point or better:
        return "incorrect"
    if claimed:
        return "solved" if obj is not None and abs(obj - opt) <= tol else "incorrect"
    return "unsolved"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["instance"], r["arm"]))
    return out


def drive(args) -> int:
    from benchmarks import gdplib_runner as gr

    names = [n for n in (args.include.split(",") if args.include else gr.GDP_PARITY) if n]
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS) if a]
    out = Path(args.out)
    done = _done(out)
    table = gr.panel_oracle()
    n_new = 0
    for i, name in enumerate(names):
        # Interleave: rotate the arm order per instance so no arm always runs first.
        order = arms[i % len(arms) :] + arms[: i % len(arms)]
        for arm in order:
            if (name, arm) in done:
                continue
            load = os.getloadavg()[0]
            t0 = time.perf_counter()
            cmd = [sys.executable, "-u", str(HERE), "--worker", name, arm, str(args.time_limit)]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.time_limit + args.grace,
                    cwd=str(BENCH.parent),
                )
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT ")]
                if lines:
                    rec = json.loads(lines[-1][len("@@RESULT ") :])
                else:
                    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                    why = " | ".join(tail)[:400]
                    if proc.returncode == -9 and not why:
                        # SIGKILL with no output: the kernel's OOM killer (the
                        # container's dmesg shows it), not a crash in Python.
                        why = "killed by SIGKILL (out of memory)"
                    rec = {
                        "instance": name,
                        "arm": arm,
                        "error": why,
                        "returncode": proc.returncode,
                    }
            except subprocess.TimeoutExpired:
                rec = {
                    "instance": name,
                    "arm": arm,
                    "status": "killed",
                    "objective": None,
                    "bound": None,
                    "wall": args.time_limit + args.grace,
                }
            rec["elapsed"] = round(time.perf_counter() - t0, 2)
            rec["load1"] = round(load, 2)
            rec["load_gated"] = load > args.max_load
            rec["time_limit"] = args.time_limit
            rec["verdict"] = score(rec, table.get(name))
            with out.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            n_new += 1
            print(
                f"{name:34s} {arm:16s} {rec['verdict']:18s} status={rec.get('status')} "
                f"obj={rec.get('objective')} bound={rec.get('bound')} "
                f"wall={rec.get('wall')} load={rec['load1']}",
                flush=True,
            )
    print(f"ran {n_new} new runs; {len(_done(out))} recorded in {out}", flush=True)
    return 0 if (n_new or done) else 1


def report(path: Path) -> int:
    """Per-arm and per-family tables from a results file (spec §1.2 metrics)."""
    from benchmarks import gdplib_runner as gr

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    table = gr.panel_oracle()
    by_arm: dict[str, list] = {}
    for r in rows:
        r["verdict"] = score(r, table.get(r["instance"]))
        by_arm.setdefault(r["arm"], []).append(r)

    def sgm(walls, shift=1.0):
        return math.exp(sum(math.log(w + shift) for w in walls) / len(walls)) - shift

    print(
        f"{'arm':18s} {'n':>3s} {'solved':>6s} {'incorr':>6s} {'unver':>5s} "
        f"{'SGM(s)':>8s} {'median_solved':>13s} {'total(s)':>9s}"
    )
    stats = {}
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        # A run that is not solved counts at the time limit (spec §1.2 SGM).
        walls = [
            min(float(r["wall"]), r["time_limit"]) if r["verdict"] == "solved" else r["time_limit"]
            for r in rs
        ]
        solved = [r for r in rs if r["verdict"] == "solved"]
        med = sorted(float(r["wall"]) for r in solved)
        median = med[len(med) // 2] if med else float("nan")
        stats[arm] = (len(solved), sgm(walls))
        print(
            f"{arm:18s} {len(rs):3d} {len(solved):6d} "
            f"{sum(r['verdict'] == 'incorrect' for r in rs):6d} "
            f"{sum(r['verdict'] == 'claimed_unverified' for r in rs):5d} "
            f"{sgm(walls):8.2f} {median:13.2f} {sum(walls):9.1f}"
        )
    print("\nfamily   discopt solved/SGM   pyomo solved/SGM   SGM ratio")
    for fam, (d, p) in FAMILIES.items():
        if d in stats and p in stats:
            print(
                f"{fam:8s} {stats[d][0]:6d} / {stats[d][1]:7.2f}   {stats[p][0]:6d} / "
                f"{stats[p][1]:7.2f}   {stats[d][1] / max(stats[p][1], 1e-9):6.2f}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        name, arm, tl = argv[1], argv[2], float(argv[3])
        try:
            rec = worker(name, arm, tl)
        except Exception as e:  # noqa: BLE001  -- recorded, never silent
            rec = {"instance": name, "arm": arm, "error": f"{type(e).__name__}: {e}"[:400]}
        print("@@RESULT " + json.dumps(rec, default=str), flush=True)
        os._exit(0)  # do not wait on solver threads at interpreter exit
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--max-load", type=float, default=16.0)
    ap.add_argument("--include", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report(Path(args.out))
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
