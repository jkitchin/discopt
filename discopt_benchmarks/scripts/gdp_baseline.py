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
* ``incorrect`` -- claims optimal and the oracle refutes it, reports an
  incumbent strictly better than the oracle optimum (a false primal), or claims
  infeasible on an instance with a known feasible point (a certified oracle, or a
  discopt incumbent verified feasible in the Pyomo model, :func:`verified_feasible`);
* ``claimed_unverified`` -- claims optimal on an instance whose oracle is unproven;
* ``unsolved`` -- anything else, a run killed at the time limit or by SIGKILL
  (exit -9, presumed the kernel's OOM killer) included.

"Agrees" is the repository's tolerance (CLAUDE.md: abs 1e-6, rel 1e-4), the same
for every arm: a claim of optimality further from the oracle is a false
certificate whatever the solver's own gap setting.

discopt incumbents are additionally checked in the original Pyomo GDP model, under
Pyomo's semantics (``gdp_convert_parity._pyomo_violation``, with its variable-bound
and integrality checks on).

Before any row is written, :func:`preflight` checks that every requested arm's
solvers are available and refuses the run otherwise. Error rows are retried on a
restart (``--skip-errors`` keeps them). Each row records the git ``HEAD`` and host
it ran on.

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
#: LOA on the framework with its options (#1775 G4.1): arm -> gdp_options.
LOA_OPTION_ARMS: dict[str, dict] = {
    "discopt-loa-global": {"subproblem": "global"},
    "discopt-loa-setcover": {"init_algorithm": "set_covering"},
    "discopt-loa-global-setcover": {"subproblem": "global", "init_algorithm": "set_covering"},
}
DISCOPT_ARMS = (
    "discopt-big-m",
    "discopt-hull",
    "discopt-mbigm",
    "discopt-auto",
    "discopt-loa",
    *LOA_OPTION_ARMS,
)
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
        transform = arm.split("-")[1]
        opts = {}
        if transform == "hull":
            # A disjunct whose functions are not finite at the origin makes Pyomo's
            # hull search for a well-defined point with a nonlinear solver whose
            # default is gurobi_direct_minlp (GDPlib gdp_col); keep it open.
            opts["well_defined_points_heuristic_solver"] = _GLOBAL
        pyo.TransformationFactory("gdp." + transform).apply_to(model, **opts)
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
    options = LOA_OPTION_ARMS.get(arm)
    method = "loa" if options is not None else arm.split("-", 1)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if options is not None:
            r = conv.model.solve(gdp_method=method, gdp_options=options, time_limit=time_limit)
        else:
            r = conv.model.solve(gdp_method=method, time_limit=time_limit)
    out = {
        "status": r.status,
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None or not math.isfinite(r.bound) else float(r.bound),
        "minimize": conv.model._objective.sense.name == "MINIMIZE",
        "node_count": getattr(r, "node_count", None),
    }
    if r.x is not None and r.objective is not None:
        out["in_pyomo"] = _pyomo_violation(model, conv, r, domains=True)
    return out


def _provenance(discopt_file: str) -> dict:
    """The git ``HEAD`` of the tree ``discopt`` was imported from, and the host.

    ``git_head`` is None, with ``git_head_error`` saying why, when it cannot be read
    (an installed wheel, no git): a row never claims a commit it did not check."""
    import socket

    out: dict = {"hostname": socket.gethostname(), "git_head": None}
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(discopt_file).resolve().parent),
        )
    except (OSError, subprocess.SubprocessError) as err:
        out["git_head_error"] = f"{type(err).__name__}: {err}"[:200]
        return out
    if proc.returncode == 0 and proc.stdout.strip():
        out["git_head"] = proc.stdout.strip()
    else:
        out["git_head_error"] = (proc.stderr.strip() or f"exit {proc.returncode}")[:200]
    return out


def worker(name: str, arm: str, time_limit: float) -> dict:
    import discopt

    rec: dict = {"instance": name, "arm": arm, "discopt": discopt.__file__}
    rec.update(_provenance(discopt.__file__))
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


#: CLAUDE.md's numerical tolerances: an objective "agrees" within abs + rel*|opt|.
ABS_TOL, REL_TOL = 1e-6, 1e-4
#: A point's relative row / bound violation, and its distance from integrality.
FEAS_TOL, INT_TOL = 1e-5, 1e-5
_DOMAIN_KEYS = ("max_bound_violation", "max_integrality_violation")


def _violates(inp: dict | None) -> bool:
    """The Pyomo check *measured* a violation (rows, counts, logic, or -- when it
    was asked for -- bounds or integrality)."""
    if not inp:
        return False
    return bool(
        inp["max_rel_violation"] > FEAS_TOL
        or inp["bad_counts"]
        or inp["bad_logic"]
        or inp.get("max_bound_violation", 0.0) > FEAS_TOL
        or inp.get("max_integrality_violation", 0.0) > INT_TOL
    )


def verified_feasible(inp: dict | None) -> bool:
    """The point is *verified* feasible in the Pyomo model: at least one check ran
    (CLAUDE.md §6), the bound and integrality checks were run, and nothing is
    violated. A row from before those checks existed is not verified."""
    return bool(
        inp
        and inp.get("checked", 0) > 0
        and all(k in inp for k in _DOMAIN_KEYS)
        and not _violates(inp)
    )


def score(rec: dict, oracle, feasible_known: bool = False) -> str:
    """``solved`` / ``incorrect`` / ``claimed_unverified`` / ``unsolved`` / ``error``.

    ``feasible_known``: the instance has a point verified feasible in the Pyomo model
    even when its oracle value is unproven (see :func:`report`).
    """
    if rec.get("returncode") == -9:
        return "unsolved"  # SIGKILL: the OOM killer, whatever it printed first
    if "error" in rec:
        return "error"
    status = str(rec.get("status", "")).lower()
    claimed = status == "optimal"
    if status == "infeasible" and (
        feasible_known or (oracle is not None and oracle.value is not None)
    ):
        return "incorrect"
    obj = rec.get("objective")
    bad_point = _violates(rec.get("in_pyomo"))
    if oracle is None or oracle.value is None:
        if bad_point:
            return "incorrect"
        return "claimed_unverified" if claimed else "unsolved"
    opt = oracle.value
    tol = ABS_TOL + REL_TOL * abs(opt)
    better = obj is not None and (obj < opt - tol if rec.get("minimize", True) else obj > opt + tol)
    if bad_point or better:
        return "incorrect"
    if claimed:
        return "solved" if obj is not None and abs(obj - opt) <= tol else "incorrect"
    return "unsolved"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _done(path: Path, skip_errors: bool = False) -> set[tuple[str, str]]:
    """The (instance, arm) pairs with a recorded outcome. An error row (a worker
    exception, or a crash with no result) is not an outcome and is retried unless
    ``skip_errors``; a SIGKILL (exit -9) is scored as a kill, so it counts as done.
    :func:`latest_rows` keeps the newest row per pair."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" in r and r.get("returncode") != -9 and not skip_errors:
                continue
            out.add((r["instance"], r["arm"]))
    return out


def latest_rows(rows: list[dict]) -> list[dict]:
    """The newest row per (instance, arm), in first-seen order (a retried error
    row is superseded by its retry)."""
    last: dict[tuple[str, str], dict] = {}
    for r in rows:
        last.pop((r["instance"], r["arm"]), None)
        last[(r["instance"], r["arm"])] = r
    return list(last.values())


#: The solvers each arm needs (``mcpp`` is GLOA's McCormick library, not a solver).
ARM_SOLVERS = {
    "pyomo-bigm": (_GLOBAL,),
    "pyomo-hull": (_GLOBAL,),
    "pyomo-loa": (_MIP, _NLP, _GLOBAL_EXE),
    "pyomo-gloa": (_MIP, _GLOBAL_EXE, "mcpp"),
    "pyomo-lbb": (_NLP, _GLOBAL_EXE),
    "pyomo-ric": (_MIP, _NLP, _GLOBAL_EXE),
    "pyomo-enumerate": (_MIP, _NLP, _GLOBAL_EXE),
    **dict.fromkeys(DISCOPT_ARMS, ("discopt",)),
}


def _solver_available(name: str) -> bool:
    if name == "mcpp":
        from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available

        return bool(mcpp_available())
    if name == "discopt":
        import discopt._rust  # noqa: F401  -- raises when the extension is missing
        from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: F401

        return True
    import pyomo.environ as pyo

    return bool(pyo.SolverFactory(name).available(exception_flag=False))


def preflight(arms: list[str]) -> None:
    """Refuse the run, before any row is written, when a requested arm is unknown or
    one of its solvers is unavailable (otherwise every run of that arm is an error
    row: e.g. Pyomo 6.10 on macOS without ``scip_direct`` or mcpp)."""
    unknown = [a for a in arms if a not in ARM_SOLVERS]
    if unknown:
        raise ValueError(f"unknown arm(s): {', '.join(unknown)}")
    cache: dict[str, bool] = {}
    missing = []
    for arm in arms:
        for solver in ARM_SOLVERS[arm]:
            if solver not in cache:
                cache[solver] = _solver_available(solver)
            if not cache[solver]:
                missing.append(f"{arm} needs {solver}")
    if missing:
        raise RuntimeError("solver preflight failed: " + "; ".join(missing))


def _run_one(name: str, arm: str, args) -> dict:
    """One run in its own subprocess, hard-killed at ``time_limit + grace``."""
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
            if proc.returncode == -9:
                # SIGKILL: presumed the kernel's OOM killer, not a crash in
                # Python -- even when the solver printed warnings first.
                why = "killed (SIGKILL; presumed OOM)" + (f": {why}" if why else "")
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
    rec["jobs"] = args.jobs
    return rec


def drive(args) -> int:
    """Run every pending (instance, arm) pair, ``--jobs`` at a time.

    With ``--jobs N > 1`` the runs share the machine, so their wall times carry
    contention: each row records ``jobs`` and its start load, and a timing claim
    from a parallel run needs a serial control on a sample of the same runs. Two
    arms of the same instance never run at once: they have the same memory
    footprint, so co-scheduling them multiplies one instance's peak (the jobs=3
    stranded_gas rows were all SIGKILLed).
    """
    import threading

    from benchmarks import gdplib_runner as gr

    names = [n for n in (args.include.split(",") if args.include else gr.GDP_PARITY) if n]
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS) if a]
    preflight(arms)
    out = Path(args.out)
    done = _done(out, getattr(args, "skip_errors", False))
    table = gr.panel_oracle()
    pending = []
    for i, name in enumerate(names):
        # Interleave: rotate the arm order per instance so no arm always runs first.
        order = arms[i % len(arms) :] + arms[: i % len(arms)]
        pending.extend((name, arm) for arm in order if (name, arm) not in done)
    lock = threading.Lock()
    cond = threading.Condition()
    queue, running = list(pending), set()

    def next_task() -> tuple[str, str] | None:
        # The first queued run whose instance is not running; wait when every
        # queued run's instance is busy.
        with cond:
            while queue:
                for k, (name, _arm) in enumerate(queue):
                    if name not in running:
                        running.add(name)
                        return queue.pop(k)
                cond.wait()
            return None

    def one(task: tuple[str, str]) -> None:
        name, arm = task
        try:
            rec = _run_one(name, arm, args)
        finally:
            with cond:
                running.discard(name)
                cond.notify_all()
        rec["verdict"] = score(rec, table.get(name))
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            print(
                f"{name:34s} {arm:16s} {rec['verdict']:18s} status={rec.get('status')} "
                f"obj={rec.get('objective')} bound={rec.get('bound')} "
                f"wall={rec.get('wall')} load={rec['load1']}",
                flush=True,
            )

    failures: list[BaseException] = []

    def loop() -> None:
        try:
            while (task := next_task()) is not None:
                one(task)
        except BaseException as err:  # re-raised below, never swallowed
            failures.append(err)

    threads = [threading.Thread(target=loop) for _ in range(max(1, args.jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise failures[0]
    print(f"ran {len(pending)} new runs; {len(_done(out))} recorded in {out}", flush=True)
    return 0 if (pending or done) else 1


def score_rows(rows: list[dict], table: dict) -> list[dict]:
    """Score the newest row per (instance, arm) in place, and return those rows.

    An instance with a point :func:`verified_feasible` in the Pyomo model refutes an
    infeasibility claim there even when its oracle value is unproven."""
    rows = latest_rows(rows)
    feasible = {
        r["instance"]
        for r in rows
        if r.get("objective") is not None and verified_feasible(r.get("in_pyomo"))
    }
    for r in rows:
        r["verdict"] = score(r, table.get(r["instance"]), r["instance"] in feasible)
    return rows


def report(path: Path) -> int:
    """Per-arm and per-family tables from a results file (spec §1.2 metrics)."""
    from benchmarks import gdplib_runner as gr

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    by_arm: dict[str, list] = {}
    for r in score_rows(rows, gr.panel_oracle()):
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
    ap.add_argument(
        "--jobs", type=int, default=1, help="runs at once (rows record it; see drive())"
    )
    ap.add_argument("--max-load", type=float, default=16.0)
    ap.add_argument("--include", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--skip-errors", action="store_true", help="treat recorded error rows as done")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report(Path(args.out))
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
