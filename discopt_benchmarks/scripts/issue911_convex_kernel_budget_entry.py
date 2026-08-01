#!/usr/bin/env python
"""Issue #911 entry experiment — does a DECLINED convex-kernel attempt add wall?

**Hypothesis under test.** ``Model.solve`` hands the convex kernel
``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET=120)`` seconds
(``_convex_kernel.try_convex_solve``), adopts the result **only** when it certifies
optimality, and then calls ``solve_model`` with the caller's **full** ``time_limit``
again (``modeling/core.py``: ``_primary_tl = time_limit - _fb_reserve``). The elapsed
kernel time is never deducted, so an eligible-but-uncertifiable model pays its whole
default-path budget **plus** the attempt, and ``solve(time_limit=T)`` runs for ~2T.

**Falsifying experiment (this script).** Run every convex-kernel-eligible in-repo
instance with ``DISCOPT_CONVEX_KERNEL`` OFF and ON, arms interleaved, N replicates,
each solve in its own subprocess, and compare *total* wall against the requested
budget. Two budgets are required: at a budget the kernel can meet, a declined attempt
never happens and the hazard is invisible (issue #911: "at a 45 s budget the hazard is
invisible; it only appears at small budgets").

**Kill criterion.** If no eligible instance's ON-arm wall materially exceeds the OFF
arm's (excess over OFF > 5 s AND ON wall > budget + 5 s) at ANY budget, the additive
hazard is not reproducible on this corpus and the fix must be re-scoped rather than
built.

Eligibility is MEASURED, not assumed: ``build_convex_spec`` is run over the corpus and
only the instances it accepts are panelled (CLAUDE.md §6 — the executed comparison
count is printed and a run that measured nothing exits non-zero).

Usage::

    python -u discopt_benchmarks/scripts/issue911_convex_kernel_budget_entry.py --census
    python -u discopt_benchmarks/scripts/issue911_convex_kernel_budget_entry.py \
        --budgets 10,45 --reps 2

Internal child mode: ``--solve <nl-path> <0|1> <budget>``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS = _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl"
# The full MINLPLib snapshot (CLAUDE.md "Benchmark instance corpus"). Only 3 in-repo
# instances are convex-kernel eligible, and all three are small enough to certify
# quickly, so the DECLINE half of the class has to be drawn from here.
_SNAPSHOT = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
_SOLU = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu"))
_CHILD_SLACK = 300.0
_CORPUS_DIR = _CORPUS


def _corpus_files() -> list[Path]:
    return sorted(_CORPUS_DIR.glob("*.nl"))


def _oracle() -> dict[str, float]:
    """MINLPLib reference optima, the correctness oracle for this panel."""
    out: dict[str, float] = {}
    if not _SOLU.exists():
        return out
    for line in _SOLU.read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            # A non-numeric third field is a `.solu` marker line, not an optimum.
            with contextlib.suppress(ValueError):
                out[p[1]] = float(p[2])
    return out


# --------------------------------------------------------------------------- child


def _run_child(nl_path: str, flag: str, budget: float) -> int:
    """Solve ONE instance in this (fresh) process and print one JSON line."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Both arms are set EXPLICITLY. Inferring an arm from a default the harness does
    # not control is how a panel silently compares OFF against OFF (#902).
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1" if flag == "1" else "0"

    import discopt  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415
    from discopt.solvers import _convex_kernel  # noqa: PLC0415

    out: dict = {
        "instance": Path(nl_path).stem,
        "flag": flag,
        "budget": float(budget),
        # CLAUDE.md §8: record which module was actually loaded, and the arm the
        # loaded module itself reports -- not the arm we believe we set.
        "discopt_file": discopt.__file__,
        "kernel_enabled": bool(_convex_kernel.convex_kernel_enabled()),
        # Marker for the version under test: absent pre-fix, present post-fix.
        "has_last_attempt_seconds": hasattr(_convex_kernel, "last_attempt_seconds"),
    }

    m = from_nl(nl_path)
    t0 = time.perf_counter()
    r = m.solve(time_limit=float(budget))
    out["wall"] = time.perf_counter() - t0
    out["status"] = getattr(r, "status", None)
    out["objective"] = getattr(r, "objective", None)
    out["bound"] = getattr(r, "bound", None)
    out["gap_certified"] = getattr(r, "gap_certified", None)
    out["node_count"] = getattr(r, "node_count", None)
    # Post-fix only: what the attempt actually cost, as the solver itself accounts it.
    if out["has_last_attempt_seconds"]:
        out["attempt_seconds"] = float(_convex_kernel.last_attempt_seconds())
    print("RESULT " + json.dumps(out), flush=True)
    return 0


def _census(files: list[Path] | None = None) -> int:
    """Print, for every corpus instance, whether build_convex_spec accepts it."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1"
    from discopt.modeling.core import from_nl  # noqa: PLC0415
    from discopt.solvers._convex_kernel import build_convex_spec  # noqa: PLC0415

    eligible: list[str] = []
    examined = 0
    for f in files if files is not None else _corpus_files():
        examined += 1
        t0 = time.perf_counter()
        try:
            spec = build_convex_spec(from_nl(str(f)))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            print(f"{f.stem:24s} ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        dt = time.perf_counter() - t0
        if spec is not None:
            eligible.append(f.stem)
            print(f"{f.stem:24s} ELIGIBLE   spec_build={dt:.2f}s", flush=True)
        else:
            print(f"{f.stem:24s} declined   ({dt:.2f}s)", flush=True)
    print(f"\nCENSUS examined={examined} eligible={len(eligible)}", flush=True)
    print("ELIGIBLE " + json.dumps(eligible), flush=True)
    if examined == 0:
        print("CENSUS measured NOTHING", flush=True)
        return 1
    return 0


# --------------------------------------------------------------------------- parent


def _spawn(nl: Path, flag: str, budget: float) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve",
        str(nl),
        flag,
        str(budget),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=budget + _CHILD_SLACK, check=False
    )
    outer = time.perf_counter() - t0
    rec: dict = {
        "instance": nl.stem,
        "flag": flag,
        "budget": budget,
        "outer_wall": outer,
        "rc": proc.returncode,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[len("RESULT ") :]))
    if "wall" not in rec:
        rec["stderr_tail"] = proc.stderr[-1500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--budgets", default="10,45")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--instances", default="", help="comma-separated stems (default: census)")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--snapshot", action="store_true", help="draw instances from the MINLPLib snapshot"
    )
    ap.add_argument("--glob", default="", help="census only snapshot stems matching these globs")
    ap.add_argument("--solve", nargs=3, metavar=("NL", "FLAG", "BUDGET"))
    args = ap.parse_args()

    global _CORPUS_DIR
    if args.snapshot:
        _CORPUS_DIR = _SNAPSHOT

    if args.solve:
        return _run_child(args.solve[0], args.solve[1], float(args.solve[2]))
    if args.census:
        files = None
        if args.glob:
            files = sorted(
                {f for pat in args.glob.split(",") for f in _CORPUS_DIR.glob(f"{pat.strip()}.nl")}
            )
        return _census(files)

    if args.instances:
        stems = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:
        print("--instances is required (run --census first)", flush=True)
        return 2
    budgets = [float(b) for b in args.budgets.split(",")]

    try:
        load1 = os.getloadavg()[0]
    except OSError:  # pragma: no cover - platform without getloadavg
        load1 = float("nan")
    print(f"LOAD at start: {load1:.2f} (1-min)", flush=True)

    records: list[dict] = []
    # Interleaved: OFF/ON adjacent within a rep, so a drift in machine load hits both
    # arms alike (CLAUDE.md §9).
    for rep in range(args.reps):
        for budget in budgets:
            for stem in stems:
                nl = _CORPUS_DIR / f"{stem}.nl"
                if not nl.exists():
                    print(f"{stem}: MISSING {nl}", flush=True)
                    continue
                for flag in ("0", "1"):
                    rec = _spawn(nl, flag, budget)
                    rec["rep"] = rep
                    records.append(rec)
                    print(
                        f"rep{rep} b={budget:g} {stem:22s} flag={flag} "
                        f"wall={rec.get('wall', float('nan')):.2f} "
                        f"status={rec.get('status')} "
                        f"kernel_enabled={rec.get('kernel_enabled')}",
                        flush=True,
                    )

    # ---- correctness ----------------------------------------------------------
    # This fix only ever SHORTENS a budget, so it cannot make a result unsound -- but
    # "cannot" is a claim, and CLAUDE.md §1 wants it checked rather than asserted.
    oracle = _oracle()
    checked = incorrect = 0
    for r in records:
        opt = oracle.get(r["instance"])
        if opt is None or r.get("objective") is None:
            continue
        checked += 1
        obj, bound = r["objective"], r.get("bound")
        tol = 1e-2 * max(1.0, abs(opt))
        # A CERTIFIED optimum must match the oracle; any dual bound must not sit on
        # the wrong side of it. Sense is read off the reported bound/incumbent pair.
        bad_obj = bool(r.get("gap_certified")) and abs(obj - opt) > tol
        bad_bound = False
        if bound is not None:
            btol = 1e-4 * max(1.0, abs(opt))
            bad_bound = (bound > opt + btol) if bound <= obj + tol else (bound < opt - btol)
        if bad_obj or bad_bound:
            incorrect += 1
            print(
                f"INCORRECT {r['instance']} flag={r['flag']} b={r['budget']:g} "
                f"obj={obj} bound={bound} opt={opt} certified={r.get('gap_certified')}",
                flush=True,
            )
    print(f"\nORACLE CHECK: checked={checked} incorrect={incorrect}", flush=True)

    # ---- analysis -------------------------------------------------------------
    comparisons = 0
    hazards = 0
    print("\n" + "=" * 92, flush=True)
    print(
        f"{'instance':22s} {'budget':>7s} {'OFF wall':>10s} {'ON wall':>10s} "
        f"{'excess':>9s} {'over-budget':>12s}",
        flush=True,
    )
    for budget in budgets:
        for stem in stems:
            off = [
                r["wall"]
                for r in records
                if r["instance"] == stem
                and r["flag"] == "0"
                and r["budget"] == budget
                and "wall" in r
            ]
            on = [
                r["wall"]
                for r in records
                if r["instance"] == stem
                and r["flag"] == "1"
                and r["budget"] == budget
                and "wall" in r
            ]
            if not off or not on:
                continue
            comparisons += 1
            mo, mn = statistics.median(off), statistics.median(on)
            sdo = statistics.stdev(off) if len(off) > 1 else float("nan")
            sdn = statistics.stdev(on) if len(on) > 1 else float("nan")
            excess = mn - mo
            over = mn - budget
            hazard = excess > 5.0 and over > 5.0
            hazards += int(hazard)
            print(
                f"{stem:22s} {budget:7g} {mo:7.2f}±{sdo:4.2f} {mn:7.2f}±{sdn:4.2f} "
                f"{excess:+9.2f} {over:+12.2f} {'HAZARD' if hazard else ''}",
                flush=True,
            )

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.out}", flush=True)

    # CLAUDE.md §6: a probe that compared nothing is a failure, not a pass.
    print(f"\nEXECUTED COMPARISONS: {comparisons}", flush=True)
    if comparisons == 0:
        print("VERDICT: measured NOTHING -- probe did not fire", flush=True)
        return 1
    if incorrect:
        print(f"VERDICT: {incorrect} INCORRECT results -- hard gate failed", flush=True)
        return 1
    print(
        f"VERDICT: {'HAZARD CONFIRMED' if hazards else 'hazard NOT reproduced'} "
        f"({hazards}/{comparisons} instance-budget cells)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
