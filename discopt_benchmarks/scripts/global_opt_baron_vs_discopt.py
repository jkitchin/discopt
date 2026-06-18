#!/usr/bin/env python3
"""Global-optimization head-to-head: discopt vs full-license BARON (via GAMS).

Runs every vendored MINLPLib ``.nl`` instance
(``python/tests/data/minlplib_nl/*.nl``) through **both** solvers under an
identical per-problem time budget, checks each result against the MINLPLib
``primalbound`` oracle, and writes a markdown + JSON report.

Why this script exists (and is not a Makefile ``*-compare`` target): the
benchmark runner's BARON adapter feeds the ``.nl`` straight to the solver
binary, which only works with the *demo-limited* AMPL-ASL BARON (<=10 vars).
The full CMU-license BARON ships inside GAMS and reads ``.gms``/``.bar``, not
``.nl``. So BARON here is driven the way prior survey iterations established:
fetch ``minlplib.org/gms/<name>.gms``, run
``gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=<T>``, and parse the
``.lst``.

discopt is run in an isolated subprocess per instance so a single crash or
hang never poisons the rest of the sweep, and timing is clean.

Usage:
    python -m discopt_benchmarks.scripts.global_opt_baron_vs_discopt \
        [--time-limit 60] [--instances a,b,c] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NL_DIR = REPO / "python" / "tests" / "data" / "minlplib_nl"
CACHE = Path(os.path.expanduser("~/.cache/discopt/minlplib/current"))
INSTANCEDATA = CACHE / "instancedata.csv"
GMS_CACHE = REPO / "reports" / "gms_cache"
GAMS = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"
GMS_URL = "https://www.minlplib.org/gms/{name}.gms"
UA = "discopt-bench (jkitchin@andrew.cmu.edu)"

# correctness tolerance (matches conftest abs=1e-6, rel=1e-4)
ATOL, RTOL = 1e-6, 1e-4


def matches(obj: float | None, known: float | None) -> bool:
    if obj is None or known is None or math.isnan(known):
        return False
    return abs(obj - known) <= RTOL * max(1.0, abs(known)) + ATOL


def is_correct(obj: float | None, known: float | None) -> bool | None:
    """True/False vs the known optimum; None when no oracle or no objective."""
    if obj is None or known is None or math.isnan(known):
        return None
    return matches(obj, known)


def nl_is_maximize(name: str) -> bool:
    """Objective sense from the .nl ``O`` segment (``O<k> 1`` == maximize)."""
    try:
        with open(NL_DIR / f"{name}.nl", errors="replace") as fh:
            for line in fh:
                if line.startswith("O"):
                    parts = line.split()
                    return len(parts) >= 2 and parts[1].strip() == "1"
    except OSError:
        pass
    return False


def _claims_global(status: str) -> bool:
    """Did the solver assert a *certified* global optimum?

    discopt: status ``optimal``. BARON/GAMS: model status ``1 Optimal``
    (``2 Locally Optimal`` and ``8 Integer Solution`` are NOT certified global).
    """
    s = (status or "").strip().lower()
    return s == "optimal" or s.startswith("1 optimal")


# verdict vocabulary
OK, GAP, VIOLATION, NA = "ok", "GAP", "VIOLATION", "n/a"


def classify(status: str, obj: float | None, known: float | None, maximize: bool) -> str:
    """Honest correctness verdict against the proven global.

    - ``ok``        : incumbent matches the known global within tolerance.
    - ``VIOLATION`` : the non-negotiable red line — either the solver *claimed*
                      a certified global with the wrong value, or it returned an
                      incumbent strictly *better* than the proven global (an
                      impossible bound, i.e. a relaxation/incumbent bug).
    - ``GAP``       : an honest feasible/uncertified incumbent that is *worse*
                      than the global — a convergence gap, not a correctness bug.
    - ``n/a``       : no oracle, or no incumbent returned.
    """
    if obj is None or known is None or math.isnan(known):
        return NA
    if matches(obj, known):
        return OK
    tol = RTOL * max(1.0, abs(known)) + ATOL
    strictly_better = (obj > known + tol) if maximize else (obj < known - tol)
    if strictly_better or _claims_global(status):
        return VIOLATION
    return GAP


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_known_optima() -> dict[str, float]:
    pb: dict[str, float] = {}
    with open(INSTANCEDATA) as f:
        r = csv.reader(f, delimiter=";")
        h = next(r)
        i_name, i_pb = h.index("name"), h.index("primalbound")
        for row in r:
            if not row:
                continue
            with contextlib.suppress(ValueError, IndexError):
                pb[row[i_name]] = float(row[i_pb])
    return pb


def all_instances() -> list[str]:
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(str(NL_DIR / "*.nl")))


# --------------------------------------------------------------------------- #
# discopt (isolated subprocess per instance)
# --------------------------------------------------------------------------- #
DISCOPT_WORKER = r"""
import json, sys, time
from discopt.modeling.core import from_nl
nl, tl = sys.argv[1], float(sys.argv[2])
t0 = time.perf_counter()
try:
    model = from_nl(nl)
    res = model.solve(time_limit=tl, gap_tolerance=1e-4)
    dt = time.perf_counter() - t0
    lb = getattr(res, "lower_bound", None)
    print(json.dumps({
        "ok": True,
        "status": str(getattr(res, "status", "")),
        "objective": (None if res.objective is None else float(res.objective)),
        "lower_bound": (None if lb is None else float(lb)),
        "gap": (None if res.gap is None else float(res.gap)),
        "node_count": int(getattr(res, "node_count", 0) or 0),
        "wall_time": dt,
    }))
except Exception as e:
    dt = time.perf_counter() - t0
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "wall_time": dt}))
"""


@dataclass
class SolverRun:
    status: str = "ERROR"
    objective: float | None = None
    lower_bound: float | None = None
    gap: float | None = None
    node_count: int = 0
    wall_time: float = 0.0
    error: str | None = None


def run_discopt(name: str, tl: float) -> SolverRun:
    nl = str(NL_DIR / f"{name}.nl")
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", DISCOPT_WORKER, nl, str(tl)],
            capture_output=True,
            text=True,
            env=env,
            timeout=tl + 60,
        )
    except subprocess.TimeoutExpired:
        return SolverRun(
            status="TIME_LIMIT", wall_time=tl + 60, error="outer-timeout (solver hung past budget)"
        )
    dt = time.perf_counter() - t0
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return SolverRun(
            status="ERROR", wall_time=dt, error=(proc.stderr.strip()[-200:] or "no-json-output")
        )
    if not d.get("ok"):
        return SolverRun(status="ERROR", wall_time=d.get("wall_time", dt), error=d.get("error"))
    return SolverRun(
        status=d["status"],
        objective=d["objective"],
        lower_bound=d["lower_bound"],
        gap=d["gap"],
        node_count=d["node_count"],
        wall_time=d["wall_time"],
    )


# --------------------------------------------------------------------------- #
# BARON via GAMS
# --------------------------------------------------------------------------- #
def fetch_gms(name: str) -> Path | None:
    GMS_CACHE.mkdir(parents=True, exist_ok=True)
    dst = GMS_CACHE / f"{name}.gms"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        req = urllib.request.Request(GMS_URL.format(name=name), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if not data:
            return None
        dst.write_bytes(data)
        return dst
    except Exception:
        return None


_MS = re.compile(r"\*\*\*\* MODEL STATUS\s+(\d+)\s+(.+)")
_OV = re.compile(r"\*\*\*\* OBJECTIVE VALUE\s+([-\d.eE+]+)")
_RU = re.compile(r"RESOURCE USAGE, LIMIT\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def parse_lst(lst: str) -> SolverRun:
    run = SolverRun(status="ERROR")
    if m := _MS.search(lst):
        run.status = f"{m.group(1)} {m.group(2).strip()}"
    if m := _OV.search(lst):
        with contextlib.suppress(ValueError):
            run.objective = float(m.group(1))
    if m := _RU.search(lst):
        with contextlib.suppress(ValueError):
            run.wall_time = float(m.group(1))
    return run


def run_baron(name: str, tl: float) -> SolverRun:
    gms = fetch_gms(name)
    if gms is None:
        return SolverRun(status="NO_GMS", error="could not fetch .gms from minlplib.org")
    work = Path(tempfile.mkdtemp(prefix=f"baron_{name}_"))
    try:
        local_gms = work / f"{name}.gms"
        shutil.copy(gms, local_gms)
        t0 = time.perf_counter()
        try:
            subprocess.run(
                [
                    GAMS,
                    f"{name}.gms",
                    "minlp=baron",
                    "optcr=0",
                    "optca=1e-9",
                    f"reslim={int(tl)}",
                    "lo=2",
                    "-o",
                    f"{name}.lst",
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=tl + 90,
            )
        except subprocess.TimeoutExpired:
            return SolverRun(status="TIME_LIMIT", wall_time=tl + 90, error="gams outer-timeout")
        wall = time.perf_counter() - t0
        lst_path = work / f"{name}.lst"
        if not lst_path.exists():
            return SolverRun(status="ERROR", wall_time=wall, error="no .lst produced")
        run = parse_lst(lst_path.read_text(errors="replace"))
        if run.wall_time == 0.0:
            run.wall_time = wall
        return run
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# orchestration + report
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    instance: str
    known: float | None
    discopt: SolverRun
    baron: SolverRun
    maximize: bool = False
    d_verdict: str = NA
    b_verdict: str = NA


def fmt(x: float | None, w: int = 12, p: int = 5) -> str:
    return f"{x:{w}.{p}g}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def tri(v: bool | None) -> str:
    """Back-compat shim used by the sibling .nl harness (bool -> label)."""
    return {True: OK, False: VIOLATION, None: NA}[v]


def finalize(row: Row) -> Row:
    row.d_verdict = classify(row.discopt.status, row.discopt.objective, row.known, row.maximize)
    row.b_verdict = classify(row.baron.status, row.baron.objective, row.known, row.maximize)
    return row


def write_report(rows: list[Row], tl: float, out_dir: Path, ts: str) -> Path:
    md = out_dir / f"global_opt_baron_vs_discopt_{ts}.md"
    js = out_dir / f"global_opt_baron_vs_discopt_{ts}.json"

    def tally(get):
        c = {OK: 0, GAP: 0, VIOLATION: 0, NA: 0}
        for r in rows:
            c[get(r)] += 1
        return c

    d = tally(lambda r: r.d_verdict)
    b = tally(lambda r: r.b_verdict)
    n_oracle = sum(r.known is not None for r in rows)

    lines = [
        "# Global Optimization Benchmark — discopt vs BARON (GAMS, full license)",
        "",
        f"- Instances: **{len(rows)}** vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`)",
        f"- Per-problem time limit: **{int(tl)} s**; gap tolerance 1e-4; "
        "correctness tol abs=1e-6 rel=1e-4 vs MINLPLib `primalbound`",
        f"- discopt: isolated subprocess `Model.solve(time_limit={int(tl)}, gap_tolerance=1e-4)`",
        "- BARON: `gams <name>.gms minlp=baron optcr=0 optca=1e-9 "
        f"reslim={int(tl)}` (CMU full license)",
        f"- Generated: {ts}",
        "",
        "## Verdict vocabulary",
        "",
        "| verdict | meaning |",
        "|---|---|",
        "| `ok` | incumbent matches the proven global within tolerance |",
        "| `GAP` | honest feasible/uncertified incumbent **worse** than the "
        "global — a convergence gap in the time budget, *not* a correctness bug |",
        "| `VIOLATION` | **the non-negotiable red line**: solver claimed a "
        "certified global with the wrong value, or returned an incumbent "
        "strictly *better* than the proven global (impossible → bug) |",
        "| `n/a` | no oracle, or no incumbent returned (e.g. parser error) |",
        "",
        "## Correctness summary",
        "",
        f"- Instances with a known optimum (oracle): **{n_oracle}/{len(rows)}**",
        f"- **discopt — VIOLATIONS: {d[VIOLATION]}**  ·  ok {d[OK]}  ·  "
        f"gap {d[GAP]}  ·  n/a {d[NA]}",
        f"- **BARON   — VIOLATIONS: {b[VIOLATION]}**  ·  ok {b[OK]}  ·  "
        f"gap {b[GAP]}  ·  n/a {b[NA]}",
        "",
        (
            "> ✅ **Zero discopt correctness violations.**"
            if d[VIOLATION] == 0
            else f"> ❌ **{d[VIOLATION]} discopt VIOLATION(S) — investigate.**"
        ),
        "",
        "## Per-instance results",
        "",
        "| instance | known | discopt obj | d | d status | d time | "
        "BARON obj | b | b status | b time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x.instance):
        lines.append(
            f"| {r.instance} | {fmt(r.known)} | {fmt(r.discopt.objective)} | "
            f"{r.d_verdict} | {r.discopt.status} | {r.discopt.wall_time:.2f} | "
            f"{fmt(r.baron.objective)} | {r.b_verdict} | {r.baron.status} | "
            f"{r.baron.wall_time:.2f} |"
        )

    viol = [r for r in rows if r.d_verdict == VIOLATION]
    if viol:
        lines += ["", "## ⚠️ discopt correctness VIOLATIONS", ""]
        for r in viol:
            lines.append(
                f"- **{r.instance}**: discopt {r.discopt.objective} "
                f"(status {r.discopt.status}) vs proven global {r.known}"
            )

    gaps = [r for r in rows if r.d_verdict == GAP]
    if gaps:
        lines += ["", "## discopt convergence gaps (honest, suboptimal in budget)", ""]
        for r in gaps:
            bw = "global" if r.b_verdict == OK else f"also {r.b_verdict}"
            lines.append(
                f"- **{r.instance}**: discopt {fmt(r.discopt.objective).strip()} "
                f"(`{r.discopt.status}`) vs global {fmt(r.known).strip()} "
                f"— BARON {fmt(r.baron.objective).strip()} ({bw})"
            )

    errs = [r for r in rows if r.discopt.error]
    if errs:
        lines += ["", "## discopt errors / load failures", ""]
        for r in errs:
            lines.append(f"- **{r.instance}**: {r.discopt.error}")

    md.write_text("\n".join(lines) + "\n")
    js.write_text(
        json.dumps(
            {
                "time_limit": tl,
                "timestamp": ts,
                "rows": [
                    {
                        "instance": r.instance,
                        "known": r.known,
                        "maximize": r.maximize,
                        "discopt": asdict(r.discopt),
                        "baron": asdict(r.baron),
                        "d_verdict": r.d_verdict,
                        "b_verdict": r.b_verdict,
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
    )
    return md


def rows_from_json(path: Path) -> tuple[list[Row], float]:
    data = json.loads(Path(path).read_text())
    rows = []
    for r in data["rows"]:
        row = Row(
            instance=r["instance"],
            known=r["known"],
            discopt=SolverRun(
                **{k: v for k, v in r["discopt"].items() if k in SolverRun.__annotations__}
            ),
            baron=SolverRun(
                **{k: v for k, v in r["baron"].items() if k in SolverRun.__annotations__}
            ),
            maximize=r.get("maximize", nl_is_maximize(r["instance"])),
        )
        rows.append(finalize(row))
    return rows, float(data.get("time_limit", 60.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument(
        "--instances", type=str, default="", help="comma-separated subset (default: all 62)"
    )
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports"))
    ap.add_argument("--skip-baron", action="store_true")
    ap.add_argument(
        "--from-json",
        type=str,
        default="",
        help="regenerate the report from a prior run's JSON (no solve)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")

    if args.from_json:
        rows, tl = rows_from_json(Path(args.from_json))
        md = write_report(rows, tl, out_dir, ts)
        nv = sum(r.d_verdict == VIOLATION for r in rows)
        print(
            f"# REGENERATED from {args.from_json}. discopt violations {nv}. Report: {md}",
            flush=True,
        )
        return 1 if nv else 0

    known = load_known_optima()
    insts = (
        [s.strip() for s in args.instances.split(",") if s.strip()]
        if args.instances
        else all_instances()
    )

    print(
        f"# global-opt head-to-head: {len(insts)} instances, time_limit={int(args.time_limit)}s",
        flush=True,
    )

    rows: list[Row] = []
    for i, name in enumerate(insts, 1):
        kn = known.get(name)
        d = run_discopt(name, args.time_limit)
        b = SolverRun(status="SKIPPED") if args.skip_baron else run_baron(name, args.time_limit)
        row = finalize(Row(name, kn, d, b, maximize=nl_is_maximize(name)))
        rows.append(row)
        ds = f"{d.status[:14]:14} {fmt(d.objective, 9)} {row.d_verdict:>9} {d.wall_time:6.1f}s"
        bs = f"{b.status[:14]:14} {fmt(b.objective, 9)} {row.b_verdict:>9} {b.wall_time:6.1f}s"
        print(f"[{i:2}/{len(insts)}] {name:20} {ds} | {bs}", flush=True)

    md = write_report(rows, args.time_limit, out_dir, ts)
    nv = sum(r.d_verdict == VIOLATION for r in rows)
    n_ok = sum(r.d_verdict == OK for r in rows)
    n_gap = sum(r.d_verdict == GAP for r in rows)
    print(f"\n# DONE. discopt: ok {n_ok}, gap {n_gap}, VIOLATIONS {nv}. Report: {md}", flush=True)
    return 1 if nv else 0


if __name__ == "__main__":
    raise SystemExit(main())
