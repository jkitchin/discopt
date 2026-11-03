#!/usr/bin/env python3
"""#1917: the CLAUDE.md section 5 graduation panel for ``DISCOPT_HULL_CONIC_QUADRATIC``.

The flag's first panels (``issue1917_gdp_flag_panel.py`` on the 24-instance
``gdp_parity`` panel) found it cert-clean and net-positive where it fires, but it
fired on 2 lowerings (clay0203: 53 -> 12 s; syngas: neutral), so "broadly" rested
on n = 1. Its audit row names what would graduate it: a wider panel of
convex-quadratic GDP models. This script runs that panel.

Corpus (``--corpus``; the fire phase decides membership, not a name list)
-------------------------------------------------------------------------
* ``generated`` -- :data:`benchmarks.convex_gdp_generator.PANEL`: seeded
  constrained-layout, coverage-clustering and waypoint-through-ellipsoids GDPs
  (see that module).
* ``pyomo`` -- every instance of the vendored Pyomo ``examples/gdp`` catalog.
* ``gdplib`` -- every discoverable GDPlib builder.

DECISION RULE (pre-registered 2026-11-01, before any solve row existed)
-----------------------------------------------------------------------
Over the instances where the flag FIRES (OFF and ON lowerings differ under
``PYTHONHASHSEED=0``), ``gdp_method="hull"``, 120 s, 3 reps, OFF/ON interleaved:

GRADUATE iff BOTH

(i) **cert-clean**:
    * 0 ``incorrect`` verdicts, scored against the instance's reference: SCIP's
      certified optimum on Pyomo's own ``gdp.bigm`` (the ``reference`` phase,
      300 s; big-M is exact at every integer point, so its optimum is the GDP's),
      else the ``gdp_parity`` oracle, else the best certified value of
      either arm (all certified values of all arms must agree within
      abs 1e-6 + rel 1e-4, or every disagreeing claim counts as incorrect);
    * no dual bound past the reference (sense-aware, same tolerance);
    * no certification lost OFF -> ON (no instance certified in more reps OFF
      than ON);
    * every incumbent of either arm verified feasible in the original Pyomo GDP
      (``gdp_baseline``'s ``in_pyomo`` check: rows, bounds, integrality, logic);

(ii) **net-positive**, across the fired instances:
    * ON certifies at least as many instances as OFF (an instance counts as
      certified for an arm when >= 2 of its 3 reps are ``solved``);
    * the shifted geometric mean (shift 10 s) of the per-instance median wall,
      capped at the time limit, satisfies SGM_ON / SGM_OFF <= 0.9;
    * the SGM (shift 10) of the per-instance median node count is not worse:
      SGM_ON <= SGM_OFF.

Otherwise the flag stays OFF and the measurement is recorded in its audit row.
Wall-clock SD across reps and each row's ``load1`` are reported; with other jobs
on the machine, node counts and certification status are the primary signals.

    python -u issue1917_conic_panel.py --phase fire --corpus all --out conic.jsonl
    python -u issue1917_conic_panel.py --phase reference --out conic.jsonl
    python -u issue1917_conic_panel.py --phase solve --workers 2 --out conic.jsonl
    python issue1917_conic_panel.py --report --out conic.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
BENCH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE.parent))

import gdp_baseline as gb  # noqa: E402
import issue1917_gdp_flag_panel as fp  # noqa: E402

FLAG = "DISCOPT_HULL_CONIC_QUADRATIC"
METHOD = "hull"
SHIFT_WALL, SHIFT_NODES, WALL_RATIO_MAX = 10.0, 10.0, 0.9
_LOCK = threading.Lock()


def corpus(which: str) -> list[str]:
    from benchmarks import convex_gdp_generator as cg
    from benchmarks import gdplib_runner as gr
    from benchmarks import pyomo_gdp_examples as px

    out: list[str] = []
    if which in ("generated", "all"):
        out += list(cg.PANEL)
    if which in ("pyomo", "all"):
        out += px.names()
    if which in ("gdplib", "all"):
        out += [s.name for s in gr.discover_models()]
    return list(dict.fromkeys(out))


def _append(out: Path, rec: dict) -> None:
    with _LOCK:
        fp._append(out, rec)


def _worker_row(name: str, arm: str, on: bool | None, time_limit: float, grace: float) -> dict:
    env = fp._env(FLAG, bool(on)) if on is not None else dict(os.environ)
    cmd = [
        sys.executable,
        "-u",
        str(HERE.parent / "gdp_baseline.py"),
        "--worker",
        name,
        arm,
        str(time_limit),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + grace,
            env=env,
            cwd=str(REPO),
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT ")]
        if lines:
            rec = json.loads(lines[-1][len("@@RESULT ") :])
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            rec = {"error": " | ".join(tail)[:400], "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        rec = {"status": "killed", "objective": None, "bound": None, "wall": time_limit + grace}
    rec["elapsed"] = round(time.perf_counter() - t0, 2)
    return rec


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


def phase_fire(args) -> int:
    ns = SimpleNamespace(flag=FLAG, out=args.out, fire_timeout=args.fire_timeout)
    return fp.phase_fire(ns, corpus(args.corpus))


def _fired(out: Path) -> list[str]:
    return sorted(
        {
            r["instance"]
            for r in fp._rows(out)
            if r.get("phase") == "fire" and r.get("flag") == FLAG and r.get("fired")
        }
    )


def _pool(tasks: list, fn, workers: int) -> None:
    it = iter(tasks)

    def loop() -> None:
        while True:
            with _LOCK:
                task = next(it, None)
            if task is None:
                return
            fn(task)

    threads = [threading.Thread(target=loop) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


REF_CHILD = r"""
import json, math, sys, time
sys.path.insert(0, sys.argv[3])
import pyomo.environ as pyo
from benchmarks import gdplib_runner as gr
name, tl = sys.argv[1], float(sys.argv[2])
out = {}
try:
    m = gr.discover_models(include=[name])[0].builder()
    obj = next(m.component_data_objects(pyo.Objective, active=True))
    minimize = obj.sense == pyo.minimize
    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    t = time.perf_counter()
    res = pyo.SolverFactory("scip").solve(
        m, options={"limits/time": tl}, load_solutions=False
    )
    out["wall"] = round(time.perf_counter() - t, 3)
    out["status"] = str(res.solver.termination_condition)
    lb, ub = res.problem.lower_bound, res.problem.upper_bound
    bound = lb if minimize else ub
    out["bound"] = float(bound) if bound is not None and math.isfinite(float(bound)) else None
    out["minimize"] = minimize
    if len(res.solution) > 0:
        m.solutions.load_from(res)
        out["objective"] = float(pyo.value(obj))
    else:
        out["objective"] = None
except Exception as e:  # recorded, never silent
    out["error"] = f"{type(e).__name__}: {e}"[:300]
print("@@REF " + json.dumps(out), flush=True)
"""


def phase_reference(args) -> int:
    """SCIP (the ``scip`` executable through Pyomo's ASL interface) on Pyomo's own
    ``gdp.bigm`` of every fired instance; big-M is exact at every integer point,
    so its optimum is the GDP's.

    Not ``gdp_baseline``'s Pyomo arms: ``pyomo-hull`` passes
    ``well_defined_points_heuristic_solver``, which this Pyomo's ``gdp.hull`` does
    not define, and both arms call ``scip_direct``, which Pyomo 6.10 reports
    unavailable next to PySCIPOpt 6.2.1 (every instance errored, 2026-11-01)."""
    done = {r["instance"] for r in fp._rows(args.out) if r.get("phase") == "reference"}
    todo = [n for n in _fired(args.out) if n not in done]
    print(f"reference: {len(todo)} instances, pyomo gdp.bigm + SCIP, {args.ref_time_limit} s")

    def one(name: str) -> None:
        load = fp._wait_for_load(args.max_load)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", REF_CHILD, name, str(args.ref_time_limit), str(BENCH)],
                capture_output=True,
                text=True,
                timeout=args.ref_time_limit + args.grace,
                cwd=str(REPO),
            )
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@REF ")]
            if lines:
                rec = json.loads(lines[-1][len("@@REF ") :])
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                rec = {"error": " | ".join(tail)[:400], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            rec = {"status": "killed", "objective": None, "bound": None}
        rec.update(
            phase="reference",
            flag=FLAG,
            instance=name,
            solver="scip-pyomo-bigm",
            load1=round(load, 2),
            elapsed=round(time.perf_counter() - t0, 2),
        )
        _append(args.out, rec)
        print(
            f"ref {name:36s} status={rec.get('status')} obj={rec.get('objective')} "
            f"bound={rec.get('bound')} wall={rec.get('wall')} err={rec.get('error')}",
            flush=True,
        )

    _pool(todo, one, args.workers)
    return 0


def phase_solve(args) -> int:
    fired = _fired(args.out)
    if not fired:
        print("nothing to solve: no instance fired (run --phase fire first)")
        return 1
    done = {
        (r["instance"], r["arm"], r["rep"])
        for r in fp._rows(args.out)
        if r.get("phase") == "solve" and r.get("flag") == FLAG
    }
    # One task = both arms of one (instance, rep), back to back on one worker,
    # order alternated, so every A/B pair shares its machine state.
    tasks = [
        (name, rep, (False, True) if (rep + i) % 2 == 0 else (True, False))
        for rep in range(args.reps)
        for i, name in enumerate(fired)
    ]
    print(
        f"solve: {len(fired)} fired instances x 2 arms x {args.reps} reps, {args.workers} workers"
    )

    def one(task) -> None:
        name, rep, order = task
        for on in order:
            arm = "on" if on else "off"
            if (name, arm, rep) in done:
                continue
            load = fp._wait_for_load(args.max_load)
            rec = _worker_row(name, f"discopt-{METHOD}", on, args.time_limit, args.grace)
            rec.update(
                phase="solve",
                flag=FLAG,
                instance=name,
                method=METHOD,
                arm=arm,
                rep=rep,
                load1=round(load, 2),
                time_limit=args.time_limit,
            )
            _append(args.out, rec)
            print(
                f"solve rep{rep} {name:34s} {arm:3s} status={rec.get('status')} "
                f"obj={rec.get('objective')} bound={rec.get('bound')} "
                f"nodes={rec.get('node_count')} wall={rec.get('wall')} load={rec['load1']} "
                f"err={rec.get('error')}",
                flush=True,
            )

    _pool(tasks, one, args.workers)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _tol(v: float) -> float:
    return gb.ABS_TOL + gb.REL_TOL * abs(v)


def references(rows: list[dict]) -> dict[str, tuple[float | None, str, list[str]]]:
    """``name -> (value, source, conflicts)``; see the decision rule (i)."""
    from benchmarks import gdplib_runner as gr

    parity = gr.panel_oracle()
    solve = [r for r in rows if r.get("phase") == "solve" and r.get("flag") == FLAG]
    out: dict[str, tuple[float | None, str, list[str]]] = {}
    for name in {r["instance"] for r in solve}:
        claims = [
            (f"{r['arm']} rep{r['rep']}", float(r["objective"]))
            for r in solve
            if r["instance"] == name
            and str(r.get("status", "")).lower() == "optimal"
            and r.get("objective") is not None
        ]
        ref = [
            r
            for r in rows
            if r.get("phase") == "reference"
            and r["instance"] == name
            and str(r.get("status", "")).lower() == "optimal"
            and r.get("objective") is not None
            and r.get("bound") is not None
            and abs(float(r["bound"]) - float(r["objective"])) <= _tol(float(r["objective"]))
        ]
        if ref:
            value, source = float(ref[-1]["objective"]), "scip-pyomo-bigm"
        elif name in parity and parity[name].value is not None:
            value, source = float(parity[name].value), f"parity:{parity[name].source}"
        elif claims:
            value, source = claims[0][1], "arm-consensus"
        else:
            out[name] = (None, "none", [])
            continue
        conflicts = [who for who, v in claims if abs(v - value) > _tol(value)]
        out[name] = (value, source, conflicts)
    return out


def _sgm(vals: list[float], shift: float) -> float:
    return math.exp(statistics.fmean(math.log(v + shift) for v in vals)) - shift


def report(args) -> int:
    rows = fp._rows(args.out)
    fire = [r for r in rows if r.get("phase") == "fire" and r.get("flag") == FLAG]
    solve = [r for r in rows if r.get("phase") == "solve" and r.get("flag") == FLAG]
    fired = [r for r in fire if r["fired"]]
    print(f"== {FLAG}: {args.out}")
    print(
        f"fire: {len(fire)} lowerings checked; fired on {len(fired)}; identical on "
        f"{sum(r['fired'] is False for r in fire)}; lowering error (both arms or one) on "
        f"{sum(r['fired'] is None for r in fire)}"
    )
    for r in fire:
        if r["fired"] is None:
            print(
                f"   error {r['instance']}: OFF={r['off'].get('error')} ON={r['on'].get('error')}"
            )
    if not solve:
        print("no solve rows: nothing compared")
        return 1
    refs = references(rows)
    minimize = {r["instance"]: r.get("minimize", True) for r in solve if "minimize" in r}
    checks = 0
    incorrect, crossings, infeasible_pts, lost = [], [], [], []
    cells: dict[str, dict[str, list[dict]]] = {}
    for r in solve:
        name = r["instance"]
        cells.setdefault(name, {"off": [], "on": []})[r["arm"]].append(r)
        value, _src, _c = refs.get(name, (None, "none", []))
        oracle = SimpleNamespace(value=value) if value is not None else None
        r["verdict"] = gb.score(r, oracle)
        checks += 1
        if r["verdict"] == "incorrect":
            incorrect.append(r)
        b = r.get("bound")
        if value is not None and isinstance(b, (int, float)) and math.isfinite(b):
            past = b > value + _tol(value) if minimize.get(name, True) else b < value - _tol(value)
            if past:
                crossings.append(r)
        if r.get("objective") is not None and not gb.verified_feasible(r.get("in_pyomo")):
            infeasible_pts.append(r)
    conflicts = {n: c for n, (_v, _s, c) in refs.items() if c}

    print(
        f"\n{'instance':34s} {'ref':>12s} {'source':16s} {'OFF cert':>8s} {'ON cert':>7s} "
        f"{'OFF wall':>14s} {'ON wall':>14s} {'OFF nodes':>9s} {'ON nodes':>9s} {'load1':>9s}"
    )
    w_off, w_on, n_off, n_on = [], [], [], []
    cert_off = cert_on = 0
    for name in sorted(cells):
        arms = cells[name]
        s_off = sum(r["verdict"] == "solved" for r in arms["off"])
        s_on = sum(r["verdict"] == "solved" for r in arms["on"])
        cert_off += s_off >= 2
        cert_on += s_on >= 2
        if s_off > s_on:
            lost.append((name, s_off, s_on))

        def vals(key, rs):
            return [float(r[key]) for r in rs if isinstance(r.get(key), (int, float))]

        def ms(rs):
            ws = [min(w, args.time_limit) for w in vals("wall", rs)]
            if not ws:
                return None, None, "-"
            sd = statistics.stdev(ws) if len(ws) > 1 else 0.0
            return statistics.median(ws), sd, f"{statistics.median(ws):6.1f}±{sd:5.1f}"

        mo, _, so = ms(arms["off"])
        mn, _, sn = ms(arms["on"])
        no, nn = vals("node_count", arms["off"]), vals("node_count", arms["on"])
        if mo is not None and mn is not None:
            w_off.append(mo)
            w_on.append(mn)
        if no and nn:
            n_off.append(statistics.median(no))
            n_on.append(statistics.median(nn))
        loads = vals("load1", arms["off"] + arms["on"])
        value, src, _c = refs.get(name, (None, "none", []))
        print(
            f"{name:34s} {fp._f(value):>12s} {src:16s} {s_off:>4d}/{len(arms['off']):<3d} "
            f"{s_on:>3d}/{len(arms['on']):<3d} {so:>14s} {sn:>14s} "
            f"{fp._f(statistics.median(no) if no else None):>9s} "
            f"{fp._f(statistics.median(nn) if nn else None):>9s} "
            f"{min(loads) if loads else 0:4.1f}-{max(loads) if loads else 0:<4.1f}"
        )
    print(f"\nexecuted solves: {checks}")
    clean = not (incorrect or crossings or lost or infeasible_pts or conflicts)
    print(
        f"(i) cert-clean={clean}: incorrect={len(incorrect)} bound-crossings={len(crossings)} "
        f"certification-lost={len(lost)} unverified-incumbents={len(infeasible_pts)} "
        f"reference-conflicts={len(conflicts)}"
    )
    for r in incorrect + crossings + infeasible_pts:
        print(
            f"   {r['instance']} {r['arm']} rep{r['rep']}: verdict={r['verdict']} "
            f"status={r.get('status')} obj={r.get('objective')} bound={r.get('bound')} "
            f"in_pyomo={r.get('in_pyomo')}"
        )
    for name, a, b in lost:
        print(f"   lost: {name} certified {a} reps OFF -> {b} ON")
    for name, c in conflicts.items():
        print(f"   conflict: {name}: {c} disagree with the reference")
    if not w_off:
        print("no paired wall times")
        return 1
    sw_off, sw_on = _sgm(w_off, SHIFT_WALL), _sgm(w_on, SHIFT_WALL)
    sn_off, sn_on = _sgm(n_off, SHIFT_NODES), _sgm(n_on, SHIFT_NODES)
    ratio = sw_on / sw_off
    net = cert_on >= cert_off and ratio <= WALL_RATIO_MAX and sn_on <= sn_off
    print(
        f"(ii) net-positive={net}: certified instances OFF {cert_off} -> ON {cert_on} "
        f"(of {len(cells)}); SGM wall (shift {SHIFT_WALL:g} s, cap {args.time_limit:g} s) "
        f"OFF {sw_off:.2f} -> ON {sw_on:.2f} s, ratio {ratio:.3f} (need <= {WALL_RATIO_MAX}); "
        f"SGM nodes (shift {SHIFT_NODES:g}) OFF {sn_off:.1f} -> ON {sn_on:.1f}"
    )
    print(f"VERDICT: {'GRADUATE' if clean and net else 'KEEP OFF'}")
    return 0 if checks and clean else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=("fire", "reference", "solve"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--corpus", choices=("generated", "pyomo", "gdplib", "all"), default="all")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--ref-time-limit", type=float, default=300.0)
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=None)
    ap.add_argument("--fire-timeout", type=float, default=600.0)
    args = ap.parse_args(argv)
    if args.report:
        return report(args)
    if args.phase == "fire":
        return phase_fire(args)
    if args.phase == "reference":
        return phase_reference(args)
    if args.phase == "solve":
        return phase_solve(args)
    ap.error("one of --phase or --report is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
