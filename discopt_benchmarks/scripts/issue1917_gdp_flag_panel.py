#!/usr/bin/env python3
"""#1917: CLAUDE.md section 5 panels for the default-OFF GDP reformulation flags.

``DISCOPT_MBIGM_CONDITIONED`` (#1775 G2.3), ``DISCOPT_GDP_BOUND_PRETRANSFORM``
(G2.4), ``DISCOPT_HULL_CONIC_QUADRATIC``, ``DISCOPT_HULL_LIFTED_RATIO`` and
``DISCOPT_GDP_DISJUNCT_FBBT`` (G5.1) each change the lowering of a GDP model;
``DISCOPT_GDP_ENUMERATE_CUTOFF`` (G4.2) acts inside ``gdp_method="enumerate"``,
so its fire phase marks every instance (there is no lowering to compare). Their
panel is the ``gdp_parity`` panel
(24 GDPlib instances,
:data:`gdplib_runner.GDP_PARITY`) under the route each flag gates. A flag
graduates only on a run that is BOTH

* **cert-clean** -- 0 ``incorrect`` verdicts (``gdp_baseline.score``: a claimed
  optimum the oracle refutes, an incumbent better than the oracle, an
  infeasibility claim on a feasible instance, or an incumbent that violates the
  original Pyomo model), no dual bound past a proven oracle, and no instance
  ``solved`` OFF that is not ``solved`` ON; and
* **net-positive** -- more instances solved, or the same solved faster or in
  fewer nodes, broadly and not on one instance.

Two phases, both fresh-subprocess per (instance, arm), the flag set in the
child's environment (``=1`` ON, ``=0`` OFF; nothing else differs):

``--phase fire``
    Lower each instance with ``transformations.create_using("gdp.<method>")``
    OFF and ON and compare ``model_fingerprint``. An instance whose lowering is
    identical in both arms cannot be affected by the flag (CLAUDE.md section 6:
    the panel counts where the probe can fire). A lowering that raises is
    recorded with its error, never dropped.
``--phase solve``
    ``gdp_baseline.py --worker`` (the parity panel's own worker: build, convert,
    solve, re-check the incumbent in Pyomo) per (instance, arm, repetition),
    arms interleaved and their order alternated per repetition. By default only
    instances where the flag fired; ``--all`` adds the rest as a no-channel
    control.

``--report`` prints both bars and exits non-zero when no comparison ran, any
verdict is ``incorrect``, a bound crosses a proven oracle, or a ``solved``
instance is lost.

    python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \\
        --flag DISCOPT_HULL_CONIC_QUADRATIC --phase fire --out conic.jsonl
    python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \\
        --flag DISCOPT_HULL_CONIC_QUADRATIC --phase solve --reps 3 \\
        --time-limit 120 --out conic.jsonl
    python discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \\
        --flag DISCOPT_HULL_CONIC_QUADRATIC --report --out conic.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BENCH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE.parent))

#: flag -> the ``gdp_method`` values whose lowering it gates.
FLAG_METHODS: dict[str, tuple[str, ...]] = {
    "DISCOPT_MBIGM_CONDITIONED": ("mbigm",),
    "DISCOPT_GDP_BOUND_PRETRANSFORM": ("big-m", "mbigm"),
    "DISCOPT_HULL_CONIC_QUADRATIC": ("hull",),
    "DISCOPT_HULL_LIFTED_RATIO": ("hull",),
    "DISCOPT_GDP_DISJUNCT_FBBT": ("big-m",),
    "DISCOPT_GDP_ENUMERATE_CUTOFF": ("enumerate",),
}
#: Methods with no lowering to fingerprint: the flag acts inside the algorithm,
#: so the fire phase cannot rule an instance out and every instance is solved.
_SOLVE_TIME_METHODS = frozenset({"enumerate"})
_TRANSFORM = {"big-m": "gdp.bigm", "mbigm": "gdp.mbigm", "hull": "gdp.hull"}

FIRE_CHILD = r"""
import hashlib, json, sys, time, warnings
sys.path.insert(0, sys.argv[3])
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
import discopt.transformations as dt
name, transform = sys.argv[1], sys.argv[2]
out = {}
try:
    model = convert_pyomo_gdp(gr.discover_models(include=[name])[0].builder()).model
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low = dt.create_using(transform, model)
    out["lower_s"] = round(time.perf_counter() - t, 3)
    fp = repr(dt.model_fingerprint(low)).encode()
    out["fingerprint"] = hashlib.sha256(fp).hexdigest()
    out["n_cons"] = len(low._constraints)
    out["n_vars"] = len(low._variables)
except Exception as e:  # recorded, never silent
    out["error"] = f"{type(e).__name__}: {e}"[:300]
print("@@FIRE " + json.dumps(out), flush=True)
"""


def _env(flag: str, on: bool) -> dict:
    env = dict(os.environ)
    env[flag] = "1" if on else "0"
    return env


def _append(out: Path, rec: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _rows(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]


def phase_fire(args, names: list[str]) -> int:
    done = {(r["instance"], r["method"]) for r in _rows(args.out) if r.get("phase") == "fire"}
    for name in names:
        for method in FLAG_METHODS[args.flag]:
            if (name, method) in done:
                continue
            if method in _SOLVE_TIME_METHODS:
                rec = {
                    "phase": "fire",
                    "flag": args.flag,
                    "instance": name,
                    "method": method,
                    "fired": True,
                    "off": {"note": "solve-time flag: no lowering to compare"},
                    "on": {"note": "solve-time flag: no lowering to compare"},
                }
                _append(args.out, rec)
                print(f"fire {name:36s} {method:6s} fired=True (solve-time flag)", flush=True)
                continue
            arms = {}
            for on in (False, True):
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [sys.executable, "-c", FIRE_CHILD, name, _TRANSFORM[method], str(BENCH)],
                        capture_output=True,
                        text=True,
                        timeout=args.fire_timeout,
                        env=_env(args.flag, on),
                        cwd=str(REPO),
                    )
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@FIRE ")]
                    if lines:
                        arms[on] = json.loads(lines[-1][len("@@FIRE ") :])
                    else:
                        tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
                        arms[on] = {"error": f"exit {proc.returncode}: " + " | ".join(tail)[:300]}
                except subprocess.TimeoutExpired:
                    arms[on] = {"error": f"lowering exceeded {args.fire_timeout} s"}
                arms[on]["wall"] = round(time.perf_counter() - t0, 2)
            off, on_ = arms[False], arms[True]
            if "error" in off or "error" in on_:
                fired = None
            else:
                fired = off["fingerprint"] != on_["fingerprint"]
            rec = {
                "phase": "fire",
                "flag": args.flag,
                "instance": name,
                "method": method,
                "fired": fired,
                "off": off,
                "on": on_,
            }
            _append(args.out, rec)
            print(
                f"fire {name:36s} {method:6s} fired={fired} "
                f"off={off.get('error') or off.get('n_cons')} "
                f"on={on_.get('error') or on_.get('n_cons')} "
                f"lower_s={off.get('lower_s')}/{on_.get('lower_s')}",
                flush=True,
            )
    return 0


def _fired(args) -> set[tuple[str, str]]:
    return {
        (r["instance"], r["method"])
        for r in _rows(args.out)
        if r.get("phase") == "fire" and r.get("fired")
    }


def phase_solve(args, names: list[str]) -> int:
    import gdp_baseline as gb

    fired = _fired(args)
    done = {
        (r["instance"], r["method"], r["arm"], r["rep"])
        for r in _rows(args.out)
        if r.get("phase") == "solve"
    }
    plan = [(n, m) for n in names for m in FLAG_METHODS[args.flag] if args.all or (n, m) in fired]
    if not plan:
        print("nothing to solve: no instance fired (run --phase fire first)", flush=True)
        return 1
    print(f"solve: {len(plan)} (instance, method) pairs x 2 arms x {args.reps} reps", flush=True)
    for rep in range(args.reps):
        for i, (name, method) in enumerate(plan):
            order = (False, True) if (rep + i) % 2 == 0 else (True, False)
            for on in order:
                arm = "on" if on else "off"
                if (name, method, arm, rep) in done:
                    continue
                load = _wait_for_load(args.max_load)
                t0 = time.perf_counter()
                cmd = [
                    sys.executable,
                    "-u",
                    str(HERE.parent / "gdp_baseline.py"),
                    "--worker",
                    name,
                    f"discopt-{method}",
                    str(args.time_limit),
                ]
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=args.time_limit + args.grace,
                        env=_env(args.flag, on),
                        cwd=str(REPO),
                    )
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT ")]
                    if lines:
                        rec = json.loads(lines[-1][len("@@RESULT ") :])
                    else:
                        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                        rec = {"error": " | ".join(tail)[:400], "returncode": proc.returncode}
                except subprocess.TimeoutExpired:
                    rec = {
                        "status": "killed",
                        "objective": None,
                        "bound": None,
                        "wall": args.time_limit + args.grace,
                    }
                rec.update(
                    phase="solve",
                    flag=args.flag,
                    instance=name,
                    method=method,
                    arm=arm,
                    rep=rep,
                    load1=round(load, 2),
                    elapsed=round(time.perf_counter() - t0, 2),
                    time_limit=args.time_limit,
                )
                rec["verdict"] = gb.score(rec, gb_oracle().get(name))
                _append(args.out, rec)
                print(
                    f"solve rep{rep} {name:34s} {method:6s} {arm:3s} {rec['verdict']:18s} "
                    f"status={rec.get('status')} obj={rec.get('objective')} "
                    f"bound={rec.get('bound')} nodes={rec.get('node_count')} "
                    f"wall={rec.get('wall')} load={rec['load1']}",
                    flush=True,
                )
    return 0


def _wait_for_load(max_load: float | None) -> float:
    """Block until the 1-minute load average is at most ``max_load`` (CLAUDE.md
    section 9: a solve started under foreign load measures the load, not the flag;
    at load 55 on 14 cores a 120 s limit is a different experiment). Returns the
    load the solve started at, which every row records."""
    load = os.getloadavg()[0]
    waited = 0
    while max_load is not None and load > max_load:
        if waited % 300 == 0:
            print(f"   load gate: load1 {load:.1f} > {max_load:g}, waiting", flush=True)
        time.sleep(30)
        waited += 30
        load = os.getloadavg()[0]
    return load


_ORACLE: dict | None = None


def gb_oracle() -> dict:
    global _ORACLE
    if _ORACLE is None:
        from benchmarks import gdplib_runner as gr

        _ORACLE = gr.panel_oracle()
    return _ORACLE


def _bound_crosses(rec: dict, oracle) -> bool:
    """A dual bound strictly past a proven optimum (sense-aware, repo tolerance)."""
    import gdp_baseline as gb

    b = rec.get("bound")
    if oracle is None or oracle.value is None or b is None or not math.isfinite(b):
        return False
    tol = gb.ABS_TOL + gb.REL_TOL * abs(oracle.value)
    return b > oracle.value + tol if rec.get("minimize", True) else b < oracle.value - tol


def report(args) -> int:
    rows = _rows(args.out)
    fire = [r for r in rows if r.get("phase") == "fire" and r.get("flag") == args.flag]
    solve = [r for r in rows if r.get("phase") == "solve" and r.get("flag") == args.flag]
    print(f"== {args.flag}: {args.out}")
    n_err = sum(r["fired"] is None for r in fire)
    print(
        f"fire: {len(fire)} lowerings checked; fired on "
        f"{sum(bool(r['fired']) for r in fire)}; identical on "
        f"{sum(r['fired'] is False for r in fire)}; lowering error on {n_err}"
    )
    for r in fire:
        if r["fired"] is None:
            print(
                f"   error {r['instance']} {r['method']}: OFF={r['off'].get('error')} "
                f"ON={r['on'].get('error')}"
            )
    if not solve:
        print("no solve rows: nothing compared")
        return 1
    table = gb_oracle()
    checks = 0
    incorrect, crossings, lost, gained = [], [], [], []
    cells: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for r in solve:
        cells.setdefault((r["instance"], r["method"]), {"off": [], "on": []})[r["arm"]].append(r)
        checks += 1
        if r["verdict"] == "incorrect":
            incorrect.append(r)
        if _bound_crosses(r, table.get(r["instance"])):
            crossings.append(r)
    print(
        f"\n{'instance':34s} {'method':6s} {'OFF solved':>10s} {'ON solved':>9s} "
        f"{'OFF wall':>9s} {'ON wall':>9s} {'OFF nodes':>9s} {'ON nodes':>9s}  verdicts"
    )
    walls_off, walls_on = [], []
    for (name, method), arms in sorted(cells.items()):
        s_off = sum(r["verdict"] == "solved" for r in arms["off"])
        s_on = sum(r["verdict"] == "solved" for r in arms["on"])
        n = min(len(arms["off"]), len(arms["on"]))

        def med(key, rs):
            vals = [r.get(key) for r in rs if isinstance(r.get(key), (int, float))]
            return statistics.median(vals) if vals else None

        w_off, w_on = med("wall", arms["off"]), med("wall", arms["on"])
        if w_off is not None and w_on is not None:
            walls_off.append(min(w_off, args.time_limit))
            walls_on.append(min(w_on, args.time_limit))
        if s_off > s_on:
            lost.append((name, method, s_off, s_on))
        if s_on > s_off:
            gained.append((name, method, s_off, s_on))
        verdicts = ",".join(sorted({r["verdict"] for r in arms["off"] + arms["on"]}))
        print(
            f"{name:34s} {method:6s} {s_off:>6d}/{n:<3d} {s_on:>5d}/{n:<3d} "
            f"{_f(w_off):>9s} {_f(w_on):>9s} {_f(med('node_count', arms['off'])):>9s} "
            f"{_f(med('node_count', arms['on'])):>9s}  {verdicts}"
        )
    shift = 10.0
    sgm = lambda ws: math.exp(statistics.fmean(math.log(w + shift) for w in ws)) - shift  # noqa: E731
    print(f"\nexecuted solves: {checks}")
    print(
        f"cert-clean: incorrect={len(incorrect)} bound-crossings={len(crossings)} "
        f"solved-lost={len(lost)}"
    )
    for r in incorrect + crossings:
        print(
            f"   {r['instance']} {r['method']} {r['arm']} rep{r['rep']}: verdict={r['verdict']} "
            f"status={r.get('status')} obj={r.get('objective')} bound={r.get('bound')}"
        )
    for name, method, a, b in lost:
        print(f"   lost: {name} {method} solved {a} OFF -> {b} ON")
    print(f"net: solved-gained={len(gained)} {[(g[0], g[2], g[3]) for g in gained]}")
    if walls_off:
        print(
            f"net: SGM(wall, shift {shift:g} s, capped at {args.time_limit:g} s) over "
            f"{len(walls_off)} cells: OFF {sgm(walls_off):.2f} s -> ON {sgm(walls_on):.2f} s"
        )
    if checks == 0 or incorrect or crossings or lost:
        return 1
    return 0


def _f(v) -> str:
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--flag", required=True, choices=sorted(FLAG_METHODS))
    ap.add_argument("--phase", choices=("fire", "solve"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include", default="")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--max-load",
        type=float,
        default=None,
        help="start a solve only when the 1-minute load average is at most this",
    )
    ap.add_argument("--fire-timeout", type=float, default=900.0)
    ap.add_argument("--all", action="store_true", help="also solve instances it did not fire on")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report(args)
    from benchmarks import gdplib_runner as gr

    names = [n for n in (args.include.split(",") if args.include else gr.GDP_PARITY) if n]
    if args.phase == "fire":
        return phase_fire(args, names)
    if args.phase == "solve":
        return phase_solve(args, names)
    ap.error("one of --phase or --report is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
