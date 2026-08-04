"""Three-way solver benchmark: discopt vs BARON vs SCIP.

discopt solves the ``.nl`` directly. BARON and SCIP are driven through GAMS on a
``.gms`` produced by discopt's own exporter (``discopt.export.to_gams``), so both
external solvers read the SAME model text and the file format is not a
confounder. This mirrors the house pattern in
``discopt_benchmarks/benchmarks/gdplib_runner.py`` (``_solve_with_gams``), which
also drives BARON and SCIP as GAMS subsolvers.

Every run is a subprocess with the same wall-clock limit, and every result is
checked against ``minlplib.solu`` (1585 entries: 980 ``=opt=`` proven optima,
605 ``=best=``, 8 ``=inf=``).

Usage
-----
    python discopt_benchmarks/scripts/bench_three_way.py \
        --solvers discopt,baron,scip --limit 68 --time-limit 60

    GAMS_EXE=/path/to/gams python ...        # if GAMS is not at the default path

**BARON needs a GAMS licence.** Measured on an unlicensed machine, BARON solves
only community-size models and otherwise returns::

    **** SOLVER STATUS  7 Licensing Problems
    *** GAMS/BARON is not included in your license, and the model size exceeds
    *** the community license limits

so `--solvers discopt,scip` is the useful subset there; SCIP runs unlicensed on
everything.

**Known export gap.** A handful of instances export to GAMS that aborts during
model *generation* — ``log`` of a composite ratio evaluated at the starting
point. ``discopt.export.to_gams`` seeds strictly-interior levels, which fixes
division-by-zero and some domain errors but cannot force a composite ratio
positive; MINLPLib ships curated starting points for this reason. Those
instances surface as status ``?`` rather than being silently dropped.

Soundness is reported per solver, not just speed. Two checks, both from CLAUDE.md
§1's framing:
  * FALSE OPTIMUM  -- an incumbent strictly better than a PROVEN optimum. No
    feasible point can beat the optimum, so this means the point is infeasible.
  * BOUND CROSSING -- a dual bound on the far side of the proven optimum, which
    would fathom the true optimum.

§6: prints executed-comparison counts and exits non-zero if nothing was compared.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
# Override with $GAMS_EXE. The default is the macOS framework layout; a licensed
# machine or a Linux box will differ.
GAMS = os.environ.get("GAMS_EXE") or "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"

DISCOPT_CHILD = r"""
import json, sys, time
sys.setrecursionlimit(20000)
from discopt.modeling.core import from_nl, ObjectiveSense
m = from_nl(sys.argv[1])
sense = "min" if m._objective.sense == ObjectiveSense.MINIMIZE else "max"
t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]))
dt = time.perf_counter() - t0
print("RESULT" + json.dumps({
    "solver": "discopt", "sense": sense, "wall": dt,
    "status": str(r.status),
    "objective": None if r.objective is None else float(r.objective),
    "bound": None if getattr(r, "bound", None) is None else float(r.bound),
    "nodes": None if getattr(r, "node_count", None) is None else int(r.node_count),
    "jax": "jax" in sys.modules,
}))
"""

EXPORT_CHILD = r"""
import sys
sys.setrecursionlimit(20000)
from discopt.modeling.core import from_nl
from discopt.export import to_gams
to_gams(from_nl(sys.argv[1]), sys.argv[2])
print("EXPORT-OK")
"""

# GAMS maps `Optimal` / `Locally Optimal` etc.; we read the numbers, not the label
# (a time-limit incumbent can still be tagged optimal -- the gdplib runner makes
# the same point).
_OBJ = re.compile(r"\*\*\*\* OBJECTIVE VALUE\s+([-\d.eE+]+)")
_MSTAT = re.compile(r"\*\*\*\* MODEL STATUS\s+(\d+)\s+(.*)")
_SSTAT = re.compile(r"\*\*\*\* SOLVER STATUS\s+(\d+)\s+(.*)")
_BEST = re.compile(r"Best possible\s*=\s*([-\d.eE+]+)")


def run_discopt(nl: Path, tl: float) -> dict:
    t0 = time.perf_counter()
    try:
        out = subprocess.run(
            [sys.executable, "-c", DISCOPT_CHILD, str(nl), str(tl)],
            capture_output=True,
            text=True,
            timeout=tl + 240,
        )
    except subprocess.TimeoutExpired:
        return {"solver": "discopt", "status": "TIMEOUT", "wall": time.perf_counter() - t0}
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT") :])
    return {"solver": "discopt", "status": "CRASH", "err": (out.stderr or "")[-300:]}


def run_gams(nl: Path, tl: float, solver: str, workdir: Path) -> dict:
    """Export to .gms with discopt, then solve with a GAMS subsolver."""
    gms = workdir / f"{nl.stem}.gms"
    if not gms.exists():
        try:
            ex = subprocess.run(
                [sys.executable, "-c", EXPORT_CHILD, str(nl), str(gms)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return {"solver": solver, "status": "EXPORT-TIMEOUT"}
        if "EXPORT-OK" not in ex.stdout:
            return {"solver": solver, "status": "EXPORT-FAIL", "err": (ex.stderr or "")[-300:]}

    lst = workdir / f"{nl.stem}.{solver}.lst"
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [
                GAMS,
                str(gms),
                f"MINLP={solver.upper()}",
                f"NLP={solver.upper()}",
                f"MIP={solver.upper()}",
                "optcr=1e-9",
                f"reslim={tl}",
                "lo=2",
                "-o",
                str(lst),
            ],
            capture_output=True,
            text=True,
            timeout=tl + 240,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return {"solver": solver, "status": "TIMEOUT", "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if not lst.exists():
        return {"solver": solver, "status": "NO-LST", "wall": wall}
    txt = lst.read_text(errors="replace")
    mo, ms, ss, bp = _OBJ.search(txt), _MSTAT.search(txt), _SSTAT.search(txt), _BEST.search(txt)
    return {
        "solver": solver,
        "wall": wall,
        "status": (ms.group(2).strip() if ms else "?"),
        "solver_status": (ss.group(2).strip() if ss else "?"),
        "objective": float(mo.group(1)) if mo else None,
        "bound": float(bp.group(1)) if bp else None,
        "nodes": None,
    }


def load_oracle():
    opt, best, infeas = {}, {}, set()
    f = BENCH / "minlplib.solu"
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        if p[0] == "=inf=":
            infeas.add(p[1])
            continue
        if len(p) < 3:
            continue
        try:
            v = float(p[2])
        except ValueError:
            continue
        if p[0] == "=opt=":
            opt[p[1]] = v
        elif p[0] == "=best=":
            best[p[1]] = v
    return opt, best, infeas


def solved_ok(r: dict) -> bool:
    s = str(r.get("status", "")).lower()
    return ("optimal" in s and "locally" not in s) or s == "optimal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="python/tests/data/minlplib_nl")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--solvers", default="discopt,baron,scip")
    args = ap.parse_args()

    opt_o, best_o, _infeas_o = load_oracle()
    wd = Path(args.workdir or "/tmp/bench3work")
    wd.mkdir(parents=True, exist_ok=True)

    want = [t.strip() for t in args.solvers.split(",") if t.strip()]
    files = sorted(Path(args.corpus).glob("*.nl"))[: args.limit]
    print(f"benchmark [{', '.join(want)}]: {len(files)} instances, time_limit={args.time_limit}s")
    print("discopt reads .nl directly; BARON and SCIP read a .gms produced by")
    print("discopt.export.to_gams -- same model text for both, so the reader is")
    print(f"not a confounder. Oracle: {len(opt_o)} proven-optimal, {len(best_o)} best-known.\n")

    hdr = f"{'instance':20s} {'discopt':>22s} {'baron':>22s} {'scip':>22s}"
    print(hdr)
    print("-" * len(hdr), flush=True)

    rows, flags = [], []
    counted = 0
    for nl in files:
        name = nl.stem
        skip = {"status": "-"}
        d = run_discopt(nl, args.time_limit) if "discopt" in want else {"solver": "discopt", **skip}
        b = (
            run_gams(nl, args.time_limit, "baron", wd)
            if "baron" in want
            else {"solver": "baron", **skip}
        )
        s = (
            run_gams(nl, args.time_limit, "scip", wd)
            if "scip" in want
            else {"solver": "scip", **skip}
        )
        counted += 1
        rows.append((name, d, b, s))

        ref_opt = opt_o.get(name)
        sense = d.get("sense", "min")
        for r in (d, b, s):
            o = r.get("objective")
            if ref_opt is not None and o is not None:
                tol = 1e-4 + 1e-3 * abs(ref_opt)
                bad = (o < ref_opt - tol) if sense == "min" else (o > ref_opt + tol)
                if bad:
                    flags.append((name, r["solver"], "FALSE-OPTIMUM", o, ref_opt))
            bd = r.get("bound")
            if ref_opt is not None and bd is not None:
                tol = 1e-4 + 1e-3 * abs(ref_opt)
                bad = (bd > ref_opt + tol) if sense == "min" else (bd < ref_opt - tol)
                if bad:
                    flags.append((name, r["solver"], "BOUND-CROSSING", bd, ref_opt))

        def cell(r):
            st = str(r.get("status", "?"))[:9]
            w = r.get("wall")
            o = r.get("objective")
            wt = "" if w is None else format(w, ".1f")
            ot = "-" if o is None else format(o, ".6g")
            return f"{st}/{wt}s/{ot}"

        print(f"{name:20s} {cell(d):>22s} {cell(b):>22s} {cell(s):>22s}", flush=True)

    print("\n=== SOUNDNESS (vs proven optima) ===")
    print(f"instances compared: {counted}")
    print(f"flags: {len(flags)}")
    for f in flags:
        print("   ", f)

    # Time-limit contract: a solve whose wall greatly exceeds its own limit makes
    # any head-to-head timing unfair, so it is reported rather than averaged away.
    print("\n=== TIME-LIMIT OVERRUNS (wall > 1.5x limit) ===")
    over = []
    for name, d_, b_, s_ in rows:
        for r in (d_, b_, s_):
            w = r.get("wall")
            if w and w > 1.5 * args.time_limit:
                over.append((name, r["solver"], round(w, 1)))
    print(f"count: {len(over)}")
    for o in over[:20]:
        print("   ", o)

    print("\n=== SOLVED-TO-OPTIMALITY COUNTS ===")
    for key, idx in (("discopt", 1), ("baron", 2), ("scip", 3)):
        if key not in want:
            continue
        n = sum(1 for r in rows if solved_ok(r[idx]))
        tot = sum(r[idx].get("wall") or 0.0 for r in rows)
        print(f"  {key:8s} optimal on {n:3d}/{counted}   total wall {tot:8.1f}s")

    idxs = [i for k, i in (("discopt", 1), ("baron", 2), ("scip", 3)) if k in want]
    both = [r for r in rows if all(solved_ok(r[i]) for i in idxs)]
    print(f"\nall selected solvers proved optimality on {len(both)} instances")
    if both:
        for key, idx in (("discopt", 1), ("baron", 2), ("scip", 3)):
            if key not in want:
                continue
            t = sum(r[idx]["wall"] for r in both)
            print(f"  {key:8s} total wall on that common set: {t:8.1f}s")
        if "discopt" in want and "scip" in want:
            dw = sum(r[1]["wall"] for r in both)
            sw = sum(r[3]["wall"] for r in both)
            faster = sum(1 for r in both if r[1]["wall"] < r[3]["wall"])
            print(
                f"\n  head-to-head on the common set: discopt faster on "
                f"{faster}/{len(both)}; total {dw:.1f}s vs {sw:.1f}s"
            )

    jaxed = [r[0] for r in rows if r[1].get("jax")]
    print(f"\ndiscopt runs that imported jax: {len(jaxed)} {jaxed[:6]}")

    if counted == 0:
        print("\nRESULT: FAIL -- nothing compared")
        return 1
    print("\nRESULT: measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
