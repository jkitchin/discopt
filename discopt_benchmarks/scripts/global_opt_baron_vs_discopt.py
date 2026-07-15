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


def root_gap_from(obj: float | None, root_bound: float | None) -> float | None:
    """Incumbent-relative root gap ``|obj - root_bound| / max(1, |obj|)``.

    The single definition used for BOTH solvers so ``root_gap_ratio_vs_baron``
    compares like with like; it mirrors ``SolveResult.root_gap`` (each solver
    normalizes by its own best incumbent). ``None`` when either input is missing
    or non-finite (an honest "not measurable", never a fabricated 0).
    """
    if obj is None or root_bound is None:
        return None
    if not (math.isfinite(obj) and math.isfinite(root_bound)):
        return None
    return abs(obj - root_bound) / max(1.0, abs(obj))


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


def bound_violates_oracle(bound: float | None, known: float | None, maximize: bool) -> bool:
    """Does the reported *dual bound* cross the known global optimum?

    The certificate invariant (CLAUDE.md): a valid dual bound never crosses the
    oracle — for a minimize problem the lower bound must satisfy ``bound <= opt``;
    for maximize the upper bound must satisfy ``bound >= opt``. A bound on the
    wrong side of the proven global (beyond tolerance) is an invalid bound — a
    false certificate seed — regardless of whether the incumbent is correct.
    None bound / no oracle → cannot judge → not a violation.
    """
    if bound is None or known is None or math.isnan(known):
        return False
    tol = RTOL * max(1.0, abs(known)) + ATOL
    return (bound < known - tol) if maximize else (bound > known + tol)


def classify(
    status: str,
    obj: float | None,
    known: float | None,
    maximize: bool,
    bound: float | None = None,
) -> str:
    """Honest correctness verdict against the proven global.

    - ``ok``        : incumbent matches the known global within tolerance.
    - ``VIOLATION`` : the non-negotiable red line — the solver *claimed* a
                      certified global with the wrong value, returned an incumbent
                      strictly *better* than the proven global, or reported a
                      *dual bound that crosses the oracle* (an impossible bound,
                      i.e. a relaxation/incumbent/bound bug).
    - ``GAP``       : an honest feasible/uncertified incumbent that is *worse*
                      than the global — a convergence gap, not a correctness bug.
    - ``n/a``       : no oracle, or no incumbent returned.
    """
    # An invalid dual bound is a VIOLATION even when the incumbent is fine or
    # absent — it is the core certificate failure.
    if bound_violates_oracle(bound, known, maximize):
        return VIOLATION
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
# G4 (baron-gap-plan.md): floor-separated timing. The fresh-process floor is the
# import tax (jax ~300 ms + pounce ~150 ms + discopt ~70 ms, Appendix B /
# baron-gap-plan §1.1); ``solve`` used to silently include jax's LAZY import on
# the first solve, so the phases were not separable. Pre-import the heavy deps
# explicitly and time them, so ``solve_time`` is pure engine time and
# ``import_time + parse_time + solve_time ~= wall_time`` (the G4 identity gate).
t_begin = time.perf_counter()
import jax  # noqa: F401  (heavy: ~300 ms; otherwise lazily imported inside solve)
import pounce  # noqa: F401  (heavy: ~150 ms; the NLP engine)
from discopt.modeling.core import from_nl
t_import = time.perf_counter() - t_begin
nl, tl = sys.argv[1], float(sys.argv[2])
try:
    t0 = time.perf_counter()
    model = from_nl(nl)
    t_parse = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = model.solve(time_limit=tl, gap_tolerance=1e-4)
    t_solve = time.perf_counter() - t0
    dt = time.perf_counter() - t_begin
    # A3: SolveResult exposes the certified dual bound as ``.bound`` (there is no
    # ``.lower_bound`` attribute — the old name silently read None on every run).
    # After A2, ``.bound`` is None on the no-relaxation class rather than a 1e30
    # sentinel, so a None here means "no dual bound", not "read failed".
    lb = getattr(res, "bound", None)
    # TAIL-1c: the root-node dual bound / relative root gap / root wall time are
    # produced by the solver (SolveResult.root_bound/root_gap/root_time) so the
    # root_gap_ratio_vs_baron gate is evaluable against BARON's own root bound.
    rb = getattr(res, "root_bound", None)
    rg = getattr(res, "root_gap", None)
    rt = getattr(res, "root_time", None)
    print(json.dumps({
        "ok": True,
        "status": str(getattr(res, "status", "")),
        "objective": (None if res.objective is None else float(res.objective)),
        "lower_bound": (None if lb is None else float(lb)),
        "gap": (None if res.gap is None else float(res.gap)),
        "root_bound": (None if rb is None else float(rb)),
        "root_gap": (None if rg is None else float(rg)),
        "root_time": (None if rt is None else float(rt)),
        "node_count": int(getattr(res, "node_count", 0) or 0),
        "wall_time": dt,
        "import_time": t_import,
        "parse_time": t_parse,
        "solve_time": t_solve,
    }))
except Exception as e:
    dt = time.perf_counter() - t_begin
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "wall_time": dt,
                      "import_time": t_import}))
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
    # TAIL-1c root-node instrumentation. ``root_bound`` is the strongest dual
    # bound available at the end of root processing (before the first branch);
    # ``root_gap`` is ``|obj - root_bound| / max(1, |obj|)`` (incumbent-relative,
    # matching SolveResult.root_gap so the discopt/BARON ratio is symmetric);
    # ``root_time`` is wall-clock seconds to reach that bound.
    root_bound: float | None = None
    root_gap: float | None = None
    root_time: float | None = None
    # G4 floor-separated timing (discopt side only; None for BARON rows and for
    # daemon-lane rows where the phases happen once in the warm process).
    # ``solve_time`` is pure engine time (imports pre-paid and measured in
    # ``import_time``); ``wall_time`` keeps its meaning as the row's total wall.
    import_time: float | None = None
    parse_time: float | None = None
    solve_time: float | None = None


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
        root_bound=d.get("root_bound"),
        root_gap=d.get("root_gap"),
        root_time=d.get("root_time"),
        import_time=d.get("import_time"),
        parse_time=d.get("parse_time"),
        solve_time=d.get("solve_time"),
    )


def run_discopt_daemon(name: str, tl: float) -> SolverRun:
    """G4 daemon lane: solve through the warm ``discopt`` daemon.

    Measures the deployment-realistic wall (a socket round-trip to a warm
    Python+JAX process, baron-gap-plan §1.1: easy-class geomean vs BARON drops
    7.5x -> 3.7x) instead of charging the ~0.5 s fresh-process import floor to
    every row. Falls back to the isolated-subprocess worker when the daemon is
    unreachable (mirroring the CLI contract), so a broken daemon can never
    invalidate a sweep — the fallback row still carries its floor-split fields.
    """
    from discopt.daemon import solve_via_daemon

    nl = str(NL_DIR / f"{name}.nl")
    t0 = time.perf_counter()
    reply = solve_via_daemon(nl, {"time_limit": tl, "gap_tolerance": 1e-4}, hard_deadline=tl + 60)
    dt = time.perf_counter() - t0
    if reply is None:
        return run_discopt(name, tl)  # daemon unreachable -> honest fallback
    if not reply.get("ok"):
        return SolverRun(status="ERROR", wall_time=dt, error=str(reply.get("error"))[:200])
    r = reply.get("result") or {}
    return SolverRun(
        status=str(r.get("status", "")),
        objective=r.get("objective"),
        lower_bound=r.get("bound"),
        gap=r.get("gap"),
        node_count=int(r.get("node_count") or 0),
        wall_time=dt,
        root_bound=r.get("root_bound"),
        root_gap=r.get("root_gap"),
        root_time=r.get("root_time"),
        solve_time=r.get("wall_time"),  # the daemon's in-process solve wall
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

# BARON's per-iteration progress log (GAMS ``lo=3`` routes it to stdout). The
# header is ``Iteration  Time (s)  Mem  Lower bound  Upper bound  Progress`` and
# each data row is ``<iter> <time> <mem>MB <lower> <upper> <progress>%``. The
# FIRST data row (iteration 1) is BARON's root-node relaxation, so its lower
# bound is BARON's root dual bound and its time is the root wall-clock. All
# bounds are printed in *internal-minimization* sense (BARON minimizes ``objvar``
# after GAMS's max→min reformulation), matching how ``parse_lst`` reads the
# .lst; the caller un-negates for maximize models.
_BAR_ITER = re.compile(
    r"^\s*(\d+)\s+([-\d.eE+]+)\s+[-\d.eE+]+MB\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+",
    re.MULTILINE,
)


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


_BAR_TOTAL = re.compile(r"Total no\. of BaR iterations:\s+(\d+)")


def parse_baron_nodes(log: str) -> int:
    """BARON node count from the ``lo=3`` stdout (G4, baron-gap-plan §6).

    Primary: the ``Total no. of BaR iterations: N`` summary line (BaR
    iterations == nodes processed — the count BARON reports for itself and the
    semantics of the 2026-06-18 baseline's node_count column). Fallback: the
    iteration number of the LAST progress-table row. 0 when neither exists.
    """
    if m := _BAR_TOTAL.search(log):
        with contextlib.suppress(ValueError):
            return int(m.group(1))
    matches = list(_BAR_ITER.finditer(log))
    if matches:
        with contextlib.suppress(ValueError):
            return int(matches[-1].group(1))
    return 0


def parse_baron_root(log: str, maximize: bool) -> tuple[float | None, float | None]:
    """Extract BARON's root-node dual bound and the time to reach it.

    Returns ``(root_bound, root_time)`` in *original-objective* sense (un-negated
    for maximize models, mirroring ``run_baron``'s objective handling), or
    ``(None, None)`` when the iteration log is absent (e.g. BARON solved in
    preprocessing and emitted no progress table). The root bound is the ``Lower
    bound`` column of iteration 1 — the strongest bound BARON has before it
    branches, the exact analogue of ``SolveResult.root_bound``.
    """
    m = _BAR_ITER.search(log)
    if m is None:
        return None, None
    with contextlib.suppress(ValueError):
        root_time = float(m.group(2))
        lb_internal = float(m.group(3))
        root_bound = -lb_internal if maximize else lb_internal
        return root_bound, root_time
    return None, None


def run_baron(name: str, tl: float, maximize: bool = False) -> SolverRun:
    gms = fetch_gms(name)
    if gms is None:
        return SolverRun(status="NO_GMS", error="could not fetch .gms from minlplib.org")
    work = Path(tempfile.mkdtemp(prefix=f"baron_{name}_"))
    try:
        local_gms = work / f"{name}.gms"
        shutil.copy(gms, local_gms)
        t0 = time.perf_counter()
        try:
            # lo=3 routes BARON's per-iteration progress log to stdout so the
            # root-node (iteration 1) dual bound is recoverable (TAIL-1c); lo=2
            # suppressed it. The .lst still carries model status / objective /
            # resource usage, so parse_lst is unaffected.
            proc = subprocess.run(
                [
                    GAMS,
                    f"{name}.gms",
                    "minlp=baron",
                    "optcr=0",
                    "optca=1e-9",
                    f"reslim={int(tl)}",
                    "lo=3",
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
        # TAIL-1c: recover BARON's root-node bound/time from the iteration log and
        # form the incumbent-relative root gap (same definition as discopt's).
        run.root_bound, run.root_time = parse_baron_root(proc.stdout or "", maximize)
        run.root_gap = root_gap_from(run.objective, run.root_bound)
        # G4: BARON's node count. Every prior sweep recorded node_count=0 (never
        # parsed), which manufactured the "BARON prunes to 1 node" misreading
        # (baron-gap-plan §1.2 — the '1' in the table is its MODEL STATUS code).
        # BARON's ``lo=3`` stdout ends with ``Total no. of BaR iterations: N``
        # (BaR iterations == nodes processed, the metric BARON itself reports;
        # validated: alan=3 matching the 2026-06-18 baseline). Fallback: the last
        # progress-table row's iteration number. Never fabricate — leave 0 when
        # neither is present (e.g. solved in preprocessing with no table).
        run.node_count = parse_baron_nodes(proc.stdout or "")
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
    row.d_verdict = classify(
        row.discopt.status,
        row.discopt.objective,
        row.known,
        row.maximize,
        row.discopt.lower_bound,
    )
    row.b_verdict = classify(
        row.baron.status,
        row.baron.objective,
        row.known,
        row.maximize,
        row.baron.lower_bound,
    )
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

    # G4 (baron-gap-plan §6): the three wall ratios. ``wall/baron`` is the
    # historical number (charges discopt the fresh-process import floor);
    # ``solve/baron`` excludes the floor (pure engine time); a --via-daemon run's
    # wall IS deployment-realistic, so its wall/baron plays the third role.
    def _geomean(xs: list[float]) -> float | None:
        xs = [x for x in xs if x and x > 0]
        if not xs:
            return None
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    _wall_r = [
        r.discopt.wall_time / r.baron.wall_time
        for r in rows
        if r.baron.wall_time and r.baron.wall_time > 0.03 and r.discopt.wall_time
    ]
    _solve_r = [
        r.discopt.solve_time / r.baron.wall_time
        for r in rows
        if r.baron.wall_time and r.baron.wall_time > 0.03 and r.discopt.solve_time
    ]
    ratio_lines = []
    if (g := _geomean(_wall_r)) is not None:
        ratio_lines.append(f"- geomean wall/BARON: **{g:.1f}x** (n={len(_wall_r)})")
    if (g := _geomean(_solve_r)) is not None:
        ratio_lines.append(
            f"- geomean solve/BARON (import floor excluded): **{g:.1f}x** (n={len(_solve_r)})"
        )

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
        *ratio_lines,
        "",
        "## Verdict vocabulary",
        "",
        "| verdict | meaning |",
        "|---|---|",
        "| `ok` | incumbent matches the proven global within tolerance |",
        "| `GAP` | honest feasible/uncertified incumbent **worse** than the "
        "global — a convergence gap in the time budget, *not* a correctness bug |",
        "| `VIOLATION` | **the non-negotiable red line**: solver claimed a "
        "certified global with the wrong value, returned an incumbent "
        "strictly *better* than the proven global, or reported a **dual bound "
        "crossing the oracle** (all impossible → bug) |",
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
        "## Root-gap instrumentation (TAIL-1c)",
        "",
        "Root gap = `|obj − root_bound| / max(1, |obj|)` at the end of root "
        "processing (before the first branch), for each solver against its own "
        "incumbent. `ratio` = discopt / BARON (the `root_gap_ratio_vs_baron` "
        "gate quantity; lower is tighter, gate target ≤ 1.3). Rows with no "
        "root bound on either side are excluded from the ratio.",
        "",
        "| instance | d root_gap | d root_t | BARON root_gap | b root_t | ratio |",
        "|---|---|---|---|---|---|",
    ]
    _ratios: list[float] = []
    for r in sorted(rows, key=lambda x: x.instance):
        drg, brg = r.discopt.root_gap, r.baron.root_gap
        ratio = None
        if drg is not None and brg is not None and brg > 1e-10 and math.isfinite(drg):
            ratio = drg / brg
            _ratios.append(ratio)
        lines.append(
            f"| {r.instance} | {fmt(drg, 10)} | {fmt(r.discopt.root_time, 8)} | "
            f"{fmt(brg, 10)} | {fmt(r.baron.root_time, 8)} | {fmt(ratio, 8)} |"
        )
    d_pop = sum(r.discopt.root_gap is not None for r in rows)
    b_pop = sum(r.baron.root_gap is not None for r in rows)
    mean_ratio = (sum(_ratios) / len(_ratios)) if _ratios else None
    lines += [
        "",
        f"- discopt root_gap populated: **{d_pop}/{len(rows)}**  ·  "
        f"BARON root_gap populated: **{b_pop}/{len(rows)}**",
        f"- Mean root_gap ratio (discopt/BARON) over **{len(_ratios)}** "
        f"co-populated instances: **{fmt(mean_ratio).strip()}** "
        f"(gate `root_gap_ratio_vs_baron` target ≤ 1.3)",
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
            bad_bound = bound_violates_oracle(r.discopt.lower_bound, r.known, r.maximize)
            reason = (
                f"dual bound {fmt(r.discopt.lower_bound).strip()} crosses the proven "
                f"global {fmt(r.known).strip()} (invalid bound)"
                if bad_bound
                else f"incumbent {r.discopt.objective} (status {r.discopt.status}) "
                f"vs proven global {r.known}"
            )
            lines.append(f"- **{r.instance}**: {reason}")

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
        "--via-daemon",
        action="store_true",
        help="G4: solve through the warm discopt daemon (deployment-realistic "
        "wall, no per-row import floor; one excluded warm-up solve first)",
    )
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

    solve_one = run_discopt_daemon if args.via_daemon else run_discopt
    if args.via_daemon:
        # One excluded warm-up solve: the first daemon request pays the spawn +
        # import cost that the lane exists to amortize (baron-gap-plan §6).
        print("# warming daemon (excluded solve) ...", flush=True)
        run_discopt_daemon(insts[0], min(args.time_limit, 30.0))

    rows: list[Row] = []
    for i, name in enumerate(insts, 1):
        kn = known.get(name)
        mx = nl_is_maximize(name)
        d = solve_one(name, args.time_limit)
        b = (
            SolverRun(status="SKIPPED")
            if args.skip_baron
            else run_baron(name, args.time_limit, maximize=mx)
        )
        row = finalize(Row(name, kn, d, b, maximize=mx))
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
