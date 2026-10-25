#!/usr/bin/env python3
"""Screen small MINLPLib instances for "SCIP solves it, discopt does not".

Selects every MINLPLib instance with an ``.nl`` file, ``nvars <= --max-vars``,
``ncons <= --max-cons`` and a proven optimum (``gap == 0`` in
``instancedata.csv``; ``=inf=`` instances in ``minlplib.solu`` count as proven
infeasible), fetches the ``.nl`` files, and solves each with discopt and with
SCIP through **pyscipopt** (``pip install pyscipopt``; it bundles SCIP and its
``.nl`` reader, so no SCIP binary is needed -- unlike the runner's ``scip``
entry, which shells out to a binary at a fixed path).

Each solve runs in its own subprocess with a hard timeout of
``time_limit + 30`` s. Both solvers get ``gap = 1e-4``; SCIP runs single
threaded (its default). Results are appended to ``<out>/results_tl<T>.jsonl``
as they finish, so a rerun resumes. The verdict table checks every claimed
optimum and every dual bound against the MINLPLib oracle.

Usage::

    python -u discopt_benchmarks/scripts/scip_wins_screen.py --out <dir> \\
        --time-limit 30 --workers 3
    # confirm the candidates sequentially, at a longer cap:
    python -u discopt_benchmarks/scripts/scip_wins_screen.py --out <dir> \\
        --time-limit 120 --workers 1 --names-file <dir>/candidates_tl30.txt

Each run writes ``<out>/candidates_tl<T>.txt``: every instance SCIP solves within
``--max-scip-time`` that discopt does not, then every instance both solve where
discopt takes >= ``--slow-factor`` times SCIP's time and >= 5 s. Stage 2 reads the
stage-1 file, so the candidate list is never assembled by hand.

The run that produced ``config/suites/scip_wins.txt`` is written up in
``docs/dev/scip-wins-2026-10-25.md`` (the previous round:
``docs/dev/scip-wins-2026-10-21.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MINLPLIB = "https://www.minlplib.org"
GAP = 1e-4


def _solve_one(solver: str, nl: str, tl: float) -> None:
    """Subprocess entry: solve one instance, print one ``RESULT {json}`` line."""
    out: dict = {"solver": solver}
    if solver == "discopt":
        import discopt
        import discopt.modeling as dm

        out["discopt_file"] = discopt.__file__
        t0 = time.perf_counter()
        r = dm.from_nl(nl).solve(time_limit=tl, gap_tolerance=GAP)
        out.update(
            status=str(r.status),
            objective=r.objective,
            bound=r.bound,
            nodes=getattr(r, "node_count", None),
        )
    else:
        import pyscipopt

        t0 = time.perf_counter()
        m = pyscipopt.Model()
        m.hideOutput()
        m.readProblem(nl)
        m.setParam("limits/time", tl)
        m.setParam("limits/gap", GAP)
        m.setParam("limits/memory", 4000)
        m.optimize()
        out.update(
            status=m.getStatus(),
            objective=m.getObjVal() if m.getNSols() > 0 else None,
            bound=m.getDualbound(),
            nodes=m.getNNodes(),
        )
    out["wall"] = time.perf_counter() - t0
    print("RESULT " + json.dumps(out), flush=True)


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def build_panel(out: Path, max_vars: int, max_cons: int, names: set[str] | None) -> list[dict]:
    meta = out / "meta"
    data = _fetch(f"{MINLPLIB}/instancedata.csv", meta / "instancedata.csv")
    solu = _fetch(f"{MINLPLIB}/minlplib.solu", meta / "minlplib.solu")
    infeasible = {
        t[1]
        for t in (ln.split() for ln in solu.read_text().splitlines())
        if len(t) >= 2 and t[0] == "=inf="
    }
    panel, seen = [], set()
    for r in csv.DictReader(data.open(), delimiter=";"):
        name = r["name"]
        if name in seen or "'nl'" not in r["formats"]:
            continue
        if names is not None:
            if name not in names:
                continue
        elif int(r["nvars"]) > max_vars or int(r["ncons"]) > max_cons:
            continue
        try:
            proven = float(r["gap"]) == 0.0
        except ValueError:
            proven = False
        if not proven and name not in infeasible:
            continue
        seen.add(name)
        panel.append(
            {
                "name": name,
                "probtype": r["probtype"],
                "nvars": int(r["nvars"]),
                "ncons": int(r["ncons"]),
                "nint": int(r["nbinvars"]) + int(r["nintvars"]),
                "sense": r["objsense"],
                "convex": r["convex"],
                "opt": None if name in infeasible else float(r["primalbound"]),
            }
        )
    if names is not None and (missing := names - seen):
        sys.exit(f"ERROR: not in the index with a proven status: {sorted(missing)}")
    return panel


def run(out: Path, panel: list[dict], tl: float, workers: int) -> Path:
    results = out / f"results_tl{int(tl)}.jsonl"
    done = set()
    if results.exists():
        for ln in results.open():
            j = json.loads(ln)
            done.add((j["name"], j["solver"]))
    jobs = [
        (p["name"], s) for p in panel for s in ("discopt", "scip") if (p["name"], s) not in done
    ]
    print(f"{len(panel)} instances, {len(jobs)} jobs to run", flush=True)

    def one(job):
        name, solver = job
        nl = _fetch(f"{MINLPLIB}/nl/{name}.nl", out / "nl" / f"{name}.nl")
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--_solve", solver, str(nl), str(tl)],
                capture_output=True,
                text=True,
                timeout=tl + 30,
            )
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            res = (
                json.loads(lines[-1][7:])
                if lines
                else {"status": "CRASH", "rc": p.returncode, "err": p.stderr[-1500:]}
            )
        except subprocess.TimeoutExpired:
            res = {"status": "HARD_TIMEOUT"}
        res.update(name=name, solver=solver, proc_wall=time.perf_counter() - t0)
        return res

    executed = 0
    with ThreadPoolExecutor(workers) as ex, results.open("a") as fh:
        for res in ex.map(one, jobs):
            executed += 1
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(
                f"[{executed}/{len(jobs)}] {res['name']:28s} {res['solver']:8s} "
                f"{res['status']:14s} obj={res.get('objective')} t={res['proc_wall']:.1f}",
                flush=True,
            )
    print(f"EXECUTED {executed} of {len(jobs)}", flush=True)
    if executed != len(jobs):
        sys.exit(1)
    return results


def _close(a, b) -> bool:
    return a is not None and abs(a - b) <= GAP * max(1.0, abs(b)) + 1e-6


def verdict(res: dict, opt: float | None) -> str:
    st, obj = res["status"].lower(), res.get("objective")
    if st in ("crash", "hard_timeout"):
        return st.upper()
    if opt is None:  # proven infeasible
        return "OK" if st == "infeasible" else ("WRONG" if obj is not None else "FAIL")
    if st in ("optimal", "gaplimit"):  # SCIP stops on limits/gap with "gaplimit"
        return "OK" if _close(obj, opt) else "WRONG"
    if st == "infeasible":
        return "WRONG"
    return "FOUND_UNPROVEN" if _close(obj, opt) else "FAIL"


def analyze(
    panel: list[dict],
    results: Path,
    max_scip_time: float,
    slow_factor: float = 10.0,
    candidates: Path | None = None,
) -> int:
    res = {}
    for ln in results.open():
        j = json.loads(ln)
        res[(j["name"], j["solver"])] = j
    rows, bound_checks, violations = [], 0, []
    for p in panel:
        d, s = res.get((p["name"], "discopt")), res.get((p["name"], "scip"))
        if d is None or s is None:
            continue
        opt = p["opt"]
        b = d.get("bound")
        if opt is not None and b is not None:
            bound_checks += 1
            tol = GAP * max(1.0, abs(opt)) + 1e-6
            if (b > opt + tol) if p["sense"] == "min" else (b < opt - tol):
                violations.append((p["name"], b, opt))
        rows.append(dict(p, vd=verdict(d, opt), vs=verdict(s, opt), d=d, s=s))
    print(f"\ncompared {len(rows)} instances")
    for who in ("vd", "vs"):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[who]] = counts.get(r[who], 0) + 1
        print(f"  {'discopt' if who == 'vd' else 'scip':8s} {counts}")
    print(f"discopt dual-bound checks: {bound_checks}, oracle violations: {violations}")
    wrong = [r["name"] for r in rows if r["vd"].startswith("WRONG")]
    print(f"discopt WRONG verdicts: {wrong}")
    wins = [
        r for r in rows if r["vs"] == "OK" and r["vd"] != "OK" and r["s"]["wall"] <= max_scip_time
    ]
    print(f"\nSCIP OK in <= {max_scip_time:g} s, discopt not OK: {len(wins)}")
    for r in sorted(wins, key=lambda r: (r["vd"], r["s"]["wall"])):
        d = r["d"]
        print(
            f"  {r['name']:26s} {r['probtype']:7s} v={r['nvars']:>3} c={r['ncons']:>3} "
            f"i={r['nint']:>3} cvx={r['convex'] or '?':5s} scip {r['s']['wall']:6.2f}s | "
            f"discopt {r['vd']:14s} {d['status']:10s} {d.get('wall', d['proc_wall']):6.1f}s "
            f"obj={d.get('objective')} bound={d.get('bound')} opt={r['opt']}"
        )

    def wall(r: dict) -> float:
        return float(r.get("wall", r["proc_wall"]))

    slow = [
        r
        for r in rows
        if r["vs"] == "OK"
        and r["vd"] == "OK"
        and wall(r["d"]) >= 5.0
        and wall(r["d"]) >= slow_factor * wall(r["s"])
    ]
    print(f"\nboth OK, discopt >= {slow_factor:g}x SCIP and >= 5 s: {len(slow)}")
    for r in sorted(slow, key=lambda r: -wall(r["d"]) / max(wall(r["s"]), 1e-3)):
        print(
            f"  {r['name']:26s} discopt {wall(r['d']):6.1f}s ({r['d'].get('nodes')} nodes) | "
            f"scip {wall(r['s']):6.2f}s ({r['s'].get('nodes')} nodes)"
        )
    if candidates is not None:
        candidates.write_text("".join(f"{r['name']}\n" for r in wins + slow))
        print(f"wrote {len(wins) + len(slow)} candidates to {candidates}")
    if bound_checks == 0:
        print("ERROR: no bound was checked -- the probe measured nothing")
        return 1
    return 1 if violations or wrong else 0


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--_solve":
        _solve_one(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-vars", type=int, default=50)
    ap.add_argument("--max-cons", type=int, default=50)
    ap.add_argument(
        "--max-scip-time",
        type=float,
        default=5.0,
        help="only list instances SCIP solves within this many seconds",
    )
    ap.add_argument(
        "--slow-factor",
        type=float,
        default=10.0,
        help="also nominate instances both solve where discopt is this many times slower",
    )
    ap.add_argument(
        "--names-file",
        type=Path,
        help="restrict to these instances ('#' comments ignored); overrides the size filter",
    )
    a = ap.parse_args()
    names = None
    if a.names_file:
        names = {ln.split("#")[0].strip() for ln in a.names_file.read_text().splitlines()} - {""}
    a.out.mkdir(parents=True, exist_ok=True)
    panel = build_panel(a.out, a.max_vars, a.max_cons, names)
    results = run(a.out, panel, a.time_limit, a.workers)
    cand = a.out / f"candidates_tl{int(a.time_limit)}.txt"
    sys.exit(analyze(panel, results, a.max_scip_time, a.slow_factor, cand))


if __name__ == "__main__":
    main()
