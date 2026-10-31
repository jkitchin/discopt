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
optimum and every dual bound against the MINLPLib oracle, and (#2006) every
discopt incumbent, at any status: it is ``verify_point``-ed on a fresh parse, its
reported objective must be its own, and it must not beat the proven optimum. A
results file from before #2006 has no verification fields and fails ``analyze``;
rerun it rather than resume it.

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

**QPLIB arm** (``--corpus qplib``, #2006). Instances come from
``qplib_manifest.csv`` gated on ``usable_oracle`` (never by name) and are read
by discopt through ``discopt.interfaces.qplib``. SCIP reads QPLIB's own upstream
CPLEX ``.lp`` file (fetched from qplib.zib.de) with its native LP reader, so its
input shares no code with discopt's reader and needs no conversion step. Each
SCIP job first checks that input: every variable is fixed at the reference
``.sol`` point and SCIP must reproduce the published objective. An instance whose
``.lp`` fails that check is reported and dropped from the comparison, never
compared silently. QPLIB publishes best-known values (``=best=``), not proven
optima, so the oracle is one-sided: the reference point is feasible, hence a
dual bound past it, a claimed optimum worse than it, or an ``infeasible`` claim
is WRONG, while an incumbent *better* than it is not (its point is still
verified).

The run that produced ``config/suites/scip_wins.txt`` is written up in
``docs/dev/scip-wins-2026-10-25.md`` (the previous round:
``docs/dev/scip-wins-2026-10-21.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MINLPLIB = "https://www.minlplib.org"
QPLIB = "https://qplib.zib.de"
QPLIB_DIR = Path(
    os.environ.get(
        "DISCOPT_QPLIB_DIR",
        Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark" / "qplib",
    )
)
GAP = 1e-4
#: SCIP's QPLIB ``.lp`` must reproduce the published objective at the reference
#: point to this relative accuracy (the published values carry ~10 digits).
FIDELITY_RTOL = 1e-6


def _load(path: str):
    """Parse an instance into a discopt Model: ``.qplib`` natively, else ``.nl``."""
    if path.endswith(".qplib"):
        from discopt.interfaces import qplib

        return qplib.from_qplib(path)
    import discopt.modeling as dm

    return dm.from_nl(path)


def _solve_one(solver: str, nl: str, tl: float, ref_sol: str | None = None) -> None:
    """Subprocess entry: solve one instance, print one ``RESULT {json}`` line.

    ``ref_sol`` (QPLIB only) is the reference ``.sol`` the SCIP input is checked
    against before the solve; see :func:`_scip_fidelity`.
    """
    out: dict = {"solver": solver}
    if solver == "discopt":
        import discopt

        out["discopt_file"] = discopt.__file__
        t0 = time.perf_counter()
        r = _load(nl).solve(time_limit=tl, gap_tolerance=GAP)
        out.update(
            status=str(r.status),
            objective=r.objective,
            bound=r.bound,
            nodes=getattr(r, "node_count", None),
        )
        out["wall"] = time.perf_counter() - t0
        if r.x is not None and r.objective is not None:
            out.update(_verify_incumbent(nl, r.x))
        print("RESULT " + json.dumps(out), flush=True)
        return
    else:
        import pyscipopt

        if ref_sol is not None:
            out.update(_scip_fidelity(nl, ref_sol))
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


def _verify_incumbent(nl: str, x: dict) -> dict:
    """Check discopt's incumbent on a freshly parsed copy of the instance (#2006).

    A fresh parse, because a solve may rewrite its model in place. Returns
    ``verified`` (the point passes ``verify_point``) and ``verified_obj`` (its
    true objective), or ``verify_error`` when the point cannot be laid out over
    the declared variables -- a failure ``verdict`` reports, never a pass.
    """
    import numpy as np
    from discopt.validation.feasibility import verify_point

    fresh = _load(nl)
    try:
        flat = np.concatenate(
            [np.ravel(np.asarray(x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"verify_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    vr = verify_point(fresh, flat, with_objective=True)
    return {"verified": bool(vr.ok), "verified_obj": vr.objective, "verify_reason": vr.reason}


_SOL_RECORD = re.compile(r"^[A-Za-z_]+\d+$")


def _scip_fidelity(lp: str, sol: str) -> dict:
    """Check SCIP's reading of a QPLIB ``.lp`` against the reference point.

    Fixes every variable named in the ``.sol`` at its value and every other
    variable of the file at 0 (a ``.sol`` lists only nonzeros), then solves.
    SCIP's quadratic-objective auxiliary column, which the ``.lp`` does not
    declare, is left free so the objective can take its value. Returns
    ``fidelity_obj`` (SCIP's objective at the point, ``None`` if SCIP finds it
    infeasible) and ``fidelity_fixed`` (how many columns were fixed), which
    :func:`analyze` compares with the published value.
    """
    import pyscipopt

    point: dict[str, float] = {}
    for ln in Path(sol).read_text().splitlines():
        t = ln.split()
        if len(t) == 2 and t[0] != "objvar":
            if not _SOL_RECORD.match(t[0]):
                return {"fidelity_error": f"malformed .sol record {t[0]!r}"}
            point[t[0]] = float(t[1])
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(lp)
    declared = {v.name for v in m.getVars()}
    if missing := set(point) - declared:
        return {"fidelity_error": f".sol names absent from the .lp: {sorted(missing)[:5]}"}
    fixed = 0
    for v in m.getVars():
        if not _SOL_RECORD.match(v.name):
            continue  # SCIP's auxiliary objective column
        val = point.get(v.name, 0.0)
        if v.vtype() in ("BINARY", "INTEGER"):
            val = float(round(val))
        m.fixVar(v, val)
        fixed += 1
    m.setParam("limits/time", 60)
    m.optimize()
    obj = m.getObjVal() if m.getNSols() > 0 else None
    return {"fidelity_obj": obj, "fidelity_fixed": fixed, "fidelity_status": m.getStatus()}


def build_qplib_panel(
    out: Path, qdir: Path, max_vars: int, max_cons: int, names: set[str] | None
) -> list[dict]:
    """Select QPLIB instances from the manifest, gated on ``usable_oracle``.

    Never by name: ``names`` (a stage-2 list) only restricts a selection the
    manifest already allows. ``opt`` is the published best-known value and
    ``proven`` is False (QPLIB publishes no ``=opt=`` records), which makes
    :func:`verdict` one-sided. SCIP's input is QPLIB's upstream ``.lp``, fetched
    into ``out/lp``.
    """
    manifest = qdir / "qplib_manifest.csv"
    if not manifest.exists():
        sys.exit(f"ERROR: no QPLIB manifest at {manifest} (set DISCOPT_QPLIB_DIR)")
    panel, seen = [], set()
    for r in csv.DictReader(manifest.open()):
        name = r["name"]
        if r["usable_oracle"] != "True":
            continue
        if names is not None:
            if name not in names:
                continue
        elif int(r["nvars"]) > max_vars or int(r["ncons"]) > max_cons:
            continue
        seen.add(name)
        panel.append(
            {
                "name": name,
                "probtype": r["probtype"],
                "nvars": int(r["nvars"]),
                "ncons": int(r["ncons"]),
                "nint": int(r["n_integral"]),
                "sense": "max" if r["sense"] == "maximize" else "min",
                "convex": "False" if r["nonconvex"] == "True" else "True",
                "opt": float(r["reference"]),
                "proven": False,
                "files": {
                    "discopt": str(qdir / "qplib" / f"{name}.qplib"),
                    "scip": (f"{QPLIB}/lp/{name}.lp", str(out / "lp" / f"{name}.lp")),
                    "sol": str(qdir / "sol" / f"{name}.sol"),
                },
            }
        )
    if names is not None and (missing := names - seen):
        sys.exit(f"ERROR: not in the QPLIB manifest with a usable oracle: {sorted(missing)}")
    return panel


def _ssl_context():
    """certifi's CA bundle when installed: python.org builds on macOS ship no system
    CA store, and qplib.zib.de's chain does not verify without one."""
    try:
        import certifi
    except ImportError:
        return None
    import ssl

    return ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r:
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

    by_name = {p["name"]: p for p in panel}

    def one(job):
        name, solver = job
        files = by_name[name].get("files")
        extra: list[str] = []
        if files is None:  # MINLPLib: both solvers read the same .nl
            path = str(_fetch(f"{MINLPLIB}/nl/{name}.nl", out / "nl" / f"{name}.nl"))
        elif solver == "discopt":
            path = files["discopt"]
        else:
            url, dest = files["scip"]
            path = str(_fetch(url, Path(dest)))
            extra = [files["sol"]]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--_solve", solver, path, str(tl), *extra],
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


def _beats(obj: float, opt: float, sense: str) -> bool:
    """``obj`` is strictly better than the proven optimum, beyond tolerance."""
    tol = GAP * max(1.0, abs(opt)) + 1e-6
    return obj < opt - tol if sense == "min" else obj > opt + tol


def _worse(obj: float, ref: float, sense: str) -> bool:
    """``obj`` is strictly worse than ``ref``, beyond tolerance."""
    return _beats(ref, obj, sense) if obj is not None else True


def verdict(res: dict, opt: float | None, sense: str = "min", proven: bool = True) -> str:
    """Classify one solve against the oracle.

    ``proven=False`` (QPLIB) means ``opt`` is a best-known value at a verified
    feasible point, not a proven optimum. Then an incumbent better than ``opt``
    is not evidence of an infeasible point, so ``WRONG_BEATS_OPT`` does not
    apply, and a claimed optimum is OK when it is not worse than ``opt``; a
    claimed optimum worse than it, or ``infeasible``, is still WRONG.

    Every ``WRONG*`` verdict is a soundness failure, at any status (#2006): a
    claimed optimum away from the oracle, a claim on a proven-infeasible
    instance, an incumbent strictly better than the proven optimum (it must be
    infeasible), an incumbent that fails ``verify_point``, or a reported
    objective that is not the point's own. Only discopt's rows carry the
    verification fields; SCIP's are judged on status and objective alone.
    """
    st, obj = res["status"].lower(), res.get("objective")
    if st in ("crash", "hard_timeout"):
        return st.upper()
    if "verify_error" in res:
        return "WRONG_UNVERIFIABLE"
    if res.get("verified") is False:
        return "WRONG_INFEASIBLE_POINT"
    vobj = res.get("verified_obj")
    if obj is not None and vobj is not None and not _close(obj, vobj):
        return "WRONG_OBJECTIVE"
    if proven and opt is not None and obj is not None and _beats(obj, opt, sense):
        return "WRONG_BEATS_OPT"
    if opt is None:  # proven infeasible
        return "OK" if st == "infeasible" else ("WRONG" if obj is not None else "FAIL")
    if st in ("optimal", "gaplimit"):  # SCIP stops on limits/gap with "gaplimit"
        if not proven:
            return "WRONG" if _worse(obj, opt, sense) else "OK"
        return "OK" if _close(obj, opt) else "WRONG"
    if st == "infeasible":
        return "WRONG"
    if not proven:
        return "FAIL" if _worse(obj, opt, sense) else "FOUND_UNPROVEN"
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
    rows, bound_checks, verify_checks, violations, unverified = [], 0, 0, [], []
    fidelity_checks, fidelity_bad, fidelity_missing = 0, [], []
    for p in panel:
        d, s = res.get((p["name"], "discopt")), res.get((p["name"], "scip"))
        if d is None or s is None:
            continue
        opt, proven = p["opt"], p.get("proven", True)
        if "files" in p:  # QPLIB: SCIP read an upstream .lp; was it the same model?
            if "fidelity_obj" not in s and "fidelity_error" not in s:
                if s["status"] not in ("CRASH", "HARD_TIMEOUT"):
                    fidelity_missing.append(p["name"])
            else:
                fidelity_checks += 1
                fo = s.get("fidelity_obj")
                if fo is None or not abs(fo - opt) <= FIDELITY_RTOL * max(1.0, abs(opt)):
                    fidelity_bad.append((p["name"], fo, opt, s.get("fidelity_error")))
                    s = dict(s, status="INPUT_MISMATCH")
        b = d.get("bound")
        if opt is not None and b is not None:
            bound_checks += 1
            tol = GAP * max(1.0, abs(opt)) + 1e-6
            if (b > opt + tol) if p["sense"] == "min" else (b < opt - tol):
                violations.append((p["name"], b, opt))
        if "verified" in d or "verify_error" in d:
            verify_checks += 1
        elif d.get("objective") is not None:
            unverified.append(p["name"])
        vs = (
            "INPUT_MISMATCH"
            if s["status"] == "INPUT_MISMATCH"
            else verdict(s, opt, p["sense"], proven)
        )
        rows.append(dict(p, vd=verdict(d, opt, p["sense"], proven), vs=vs, d=d, s=s))
    print(f"\ncompared {len(rows)} instances")
    for who in ("vd", "vs"):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[who]] = counts.get(r[who], 0) + 1
        print(f"  {'discopt' if who == 'vd' else 'scip':8s} {counts}")
    print(f"discopt dual-bound checks: {bound_checks}, oracle violations: {violations}")
    print(f"discopt incumbents verified: {verify_checks}")
    if any("files" in p for p in panel):
        print(f"SCIP input fidelity checks: {fidelity_checks}, mismatches: {len(fidelity_bad)}")
        for name, fo, ref, err in fidelity_bad:
            print(f"  {name:26s} SCIP .lp objective at the reference {fo} vs {ref} {err or ''}")
        better = [
            (r["name"], r["d"].get("objective"), r["opt"])
            for r in rows
            if not r.get("proven", True)
            and r["d"].get("objective") is not None
            and r["d"].get("verified")
            and _beats(r["d"]["objective"], r["opt"], r["sense"])
        ]
        print(f"discopt verified incumbents better than the best-known value: {better}")
    wrong = [(r["name"], r["vd"]) for r in rows if r["vd"].startswith("WRONG")]
    print(f"discopt WRONG verdicts: {wrong}")
    for r in rows:
        if r["vd"].startswith("WRONG"):
            d = r["d"]
            print(
                f"  {r['name']:26s} {r['vd']:22s} status={d['status']} obj={d.get('objective')} "
                f"verified_obj={d.get('verified_obj')} opt={r['opt']} "
                f"{d.get('verify_reason') or d.get('verify_error') or ''}"
            )
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
    if fidelity_missing:
        print(f"ERROR: QPLIB SCIP rows with no input-fidelity check: {fidelity_missing}")
        return 1
    if unverified:
        # a resumed pre-#2006 results file, or an objective returned without a point
        print(f"ERROR: discopt incumbents with no verification: {unverified}")
        return 1
    return 1 if violations or wrong else 0


def main() -> None:
    if len(sys.argv) in (5, 6) and sys.argv[1] == "--_solve":
        _solve_one(sys.argv[2], sys.argv[3], float(sys.argv[4]), *sys.argv[5:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--corpus",
        choices=("minlplib", "qplib"),
        default="minlplib",
        help="MINLPLib .nl (both solvers) or QPLIB (discopt: .qplib, SCIP: upstream .lp)",
    )
    ap.add_argument("--qplib-dir", type=Path, default=QPLIB_DIR)
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
    if a.corpus == "qplib":
        panel = build_qplib_panel(a.out, a.qplib_dir, a.max_vars, a.max_cons, names)
    else:
        panel = build_panel(a.out, a.max_vars, a.max_cons, names)
    results = run(a.out, panel, a.time_limit, a.workers)
    cand = a.out / f"candidates_tl{int(a.time_limit)}.txt"
    sys.exit(analyze(panel, results, a.max_scip_time, a.slow_factor, cand))


if __name__ == "__main__":
    main()
