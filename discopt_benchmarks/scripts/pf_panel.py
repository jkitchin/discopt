#!/usr/bin/env python
"""PF0 panel outcome + differential harness (issue #632).

The shared, measurement-only gate for the PF series
(``docs/dev/sota-proof-plan.md`` §2). It makes **no library change and adds no
flag** — every mode here only *observes* the solver, so any PF item can produce a
comparable before/after column and a soundness certificate.

Three modes:

1. **Panel** (default) — run the 62 vendored ``python/tests/data/minlplib_nl``
   instances end-to-end (``model.solve(time_limit=BUDGET)``, default 30 s) and
   emit per-instance ``{status, objective, bound, nodes, wall, sense}`` plus a
   summary ``{proved, feasible, timeout, total_wall}``. Each solve runs in an
   **isolated subprocess** (``--jobs N`` subprocess-parallel): one hang cannot
   kill the panel (it is killed at ``budget + grace``), and every worker reads
   ``DISCOPT_*``/``JAX_*`` env flags fresh in its own process.

2. **Diff** (``--vs REF.json``) — run the panel (or reuse ``--current CUR.json``)
   and report per-instance deltas: proved gained/lost, node ratio, bound
   direction. **Exits non-zero and prints loudly** if any instance's dual bound
   got looser than the reference, or crossed the reference objective — the
   EP3-catcher (a "cheaper node" that skips per-node separation shows up here as
   a looser bound).

3. **Differential** (``--differential --env-a "..." --env-b "..."``) — for a
   sample of instances, compare root + a few child-box relaxation bounds between
   two env configurations. Asserts ``bound_b`` is at-least-as-tight as
   ``bound_a`` (sense-aware, per box) — env-b is the candidate/claimed-tighter
   config; a looser box exits non-zero. Then runs the feasible-point soundness
   sampler (reused from ``uniform_engine_validation``) on the env-b relaxations:
   any cut point (0 cuts required) exits non-zero.

Usage (repo root, venv active, extension built)::

    cd discopt && source .venv/bin/activate
    export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

    # panel -> reference JSON
    python discopt_benchmarks/scripts/pf_panel.py --jobs 4 --out ref.json

    # regression check of the current tree against the standing baseline
    python discopt_benchmarks/scripts/pf_panel.py --vs docs/dev/data/pf-baseline.jsonl

    # differential bound + soundness gate between two configs
    python discopt_benchmarks/scripts/pf_panel.py --differential \
        --env-a "" --env-b "DISCOPT_NODE_PROBING=1" \
        --instances nvs09,tspn05,st_e38

Determinism: instances sorted by name; child boxes a fixed bisection schedule.
Wall/node numbers are wall-clock and vary run-to-run — read ``proved`` as the
signal, wall as an order of magnitude.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_SCRIPTS = _REPO / "discopt_benchmarks" / "scripts"
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_SCRIPTS))

_DEFAULT_BUDGET = 30.0
# Grace beyond the solve budget before a worker is treated as hung and killed.
_GRACE_S = 60.0
# Numerical tolerances (mirror conftest.py: abs=1e-6, rel=1e-4).
_ABS_TOL = 1e-6
_REL_TOL = 1e-4


def _all_instances() -> list[str]:
    return sorted(p.stem for p in _NL_DIR.glob("*.nl"))


def _bound_tol(ref: float) -> float:
    return _ABS_TOL + _REL_TOL * abs(ref)


# ---------------------------------------------------------------------------
# In-process single-instance solve (runs INSIDE a worker subprocess)
# ---------------------------------------------------------------------------
def _solve_one(name: str, budget: float) -> dict:
    """End-to-end solve of one instance; returns the panel record dict."""
    import time

    from discopt.modeling.core import ObjectiveSense, from_nl

    nl_path = _NL_DIR / f"{name}.nl"
    if not nl_path.exists():
        return {"instance": name, "status": "missing", "error": f"no {nl_path.name}"}

    model = from_nl(str(nl_path))
    sense = "min"
    if model._objective is not None and model._objective.sense is ObjectiveSense.MAXIMIZE:
        sense = "max"

    # Harness-only knob (PF1 / #632): override the library default
    # in_tree_presolve_stride so an OFF-vs-ON panel can be produced on the
    # *identical* binary (controls for run-to-run variance when reading node
    # ratios). This forwards the existing solve_model kwarg — no library flag is
    # added. Unset => library default (currently 1, FBBT-only in-tree presolve).
    solve_kwargs: dict = {"time_limit": budget}
    _itp = os.environ.get("DISCOPT_ITP_STRIDE")
    if _itp is not None and _itp != "":
        solve_kwargs["in_tree_presolve_stride"] = int(_itp)

    t0 = time.perf_counter()
    res = model.solve(**solve_kwargs)
    wall = time.perf_counter() - t0

    def _f(x: object) -> float | None:
        if x is None:
            return None
        xf = float(x)
        return xf if np.isfinite(xf) else None

    return {
        "instance": name,
        "status": getattr(res, "status", None),
        "objective": _f(getattr(res, "objective", None)),
        "bound": _f(getattr(res, "bound", None)),
        "gap": _f(getattr(res, "gap", None)),
        "nodes": int(getattr(res, "node_count", 0) or 0),
        "wall": wall,
        "sense": sense,
    }


# ---------------------------------------------------------------------------
# Differential worker: root + child-box relaxation bounds for one instance
# ---------------------------------------------------------------------------
def _diff_bounds_one(name: str, children: int, do_sample: bool) -> dict:
    """Root + child-box relaxation bounds via the in-house Rust simplex, plus an
    optional feasible-point soundness sample. Runs INSIDE a worker subprocess so
    the caller's env config is read fresh."""
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.modeling.core import ObjectiveSense, from_nl
    from engine_perf_probe import _child_boxes  # reuse the fixed bisection schedule

    nl_path = _NL_DIR / f"{name}.nl"
    if not nl_path.exists():
        return {"instance": name, "error": f"no {nl_path.name}"}

    model = from_nl(str(nl_path))
    sense = "min"
    if model._objective is not None and model._objective.sense is ObjectiveSense.MAXIMIZE:
        sense = "max"
    lb, ub = flat_variable_bounds(model)

    relaxer = MccormickLPRelaxer(model)

    def _bound(clb: np.ndarray, cub: np.ndarray) -> float | None:
        r = relaxer.solve_at_node(clb, cub)
        b = getattr(r, "lower_bound", None)
        if b is None or not np.isfinite(float(b)) or r.status != "optimal":
            return None
        return float(b)

    boxes: list[dict] = [{"box": "root", "bound": _bound(lb.copy(), ub.copy())}]
    for i, (_col, clb, cub) in enumerate(_child_boxes(lb, ub, children)):
        boxes.append({"box": f"child{i}", "bound": _bound(clb, cub)})

    rec: dict = {"instance": name, "sense": sense, "boxes": boxes}

    if do_sample:
        try:
            from uniform_engine_validation import _sample_soundness

            rel = None  # _sample_soundness rebuilds its own uniform relaxation
            viol, checked = _sample_soundness(model, rel, n=1000)
            rec["sample_max_violation"] = float(viol)
            rec["sample_checked"] = int(checked)
        except Exception as exc:  # noqa: BLE001
            rec["sample_error"] = repr(exc)[:160]
    return rec


# ---------------------------------------------------------------------------
# Subprocess orchestration (isolation + parallelism + fresh env)
# ---------------------------------------------------------------------------
def _worker_main(args: argparse.Namespace) -> int:
    """Hidden entrypoint: solve/measure ONE instance, print one JSON line."""
    if args.worker_mode == "panel":
        rec = _solve_one(args.worker_instance, args.budget)
    else:
        rec = _diff_bounds_one(args.worker_instance, args.children, args.sample)
    sys.stdout.write(json.dumps(rec, sort_keys=True) + "\n")
    return 0


def _spawn(
    name: str, *, mode: str, budget: float, children: int, sample: bool, env: dict[str, str]
) -> dict:
    """Run one worker subprocess; return its record (never raises)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-instance",
        name,
        "--worker-mode",
        mode,
        "--budget",
        str(budget),
        "--children",
        str(children),
    ]
    if sample:
        cmd.append("--sample")
    child_env = dict(os.environ)
    child_env.update(env)
    timeout = budget + _GRACE_S if mode == "panel" else _GRACE_S + 120.0
    try:
        proc = subprocess.run(cmd, env=child_env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"instance": name, "status": "hang", "error": f"killed after {timeout:.0f}s"}
    if proc.returncode != 0:
        return {
            "instance": name,
            "status": "error",
            "error": (proc.stderr or "").strip()[-300:],
        }
    line = ""
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            line = ln  # last JSON line wins (ignore any stray prints)
    if not line:
        return {"instance": name, "status": "error", "error": "no JSON from worker"}
    try:
        return json.loads(line)
    except Exception as exc:  # noqa: BLE001
        return {"instance": name, "status": "error", "error": f"bad JSON: {exc!r}"}


def _run_parallel(
    names: list[str],
    *,
    mode: str,
    budget: float,
    children: int,
    sample: bool,
    jobs: int,
    env: dict[str, str],
) -> list[dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {
            ex.submit(
                _spawn,
                n,
                mode=mode,
                budget=budget,
                children=children,
                sample=sample,
                env=env,
            ): n
            for n in names
        }
        from concurrent.futures import as_completed

        for done, fut in enumerate(as_completed(futs), start=1):
            n = futs[fut]
            rec = fut.result()
            results[n] = rec
            if rec.get("status") is not None:
                status = rec["status"]
            elif rec.get("error"):
                status = "error"
            elif "boxes" in rec:
                status = f"{len(rec['boxes'])} boxes"
            else:
                status = "?"
            print(f"  [{done}/{len(names)}] {n:<16} {status}", file=sys.stderr)
    return [results[n] for n in names]


# ---------------------------------------------------------------------------
# Panel mode
# ---------------------------------------------------------------------------
def _classify(rec: dict) -> str:
    """proved | feasible | timeout | infeasible | error."""
    status = rec.get("status")
    if status in ("optimal", "infeasible"):
        return "proved" if status == "optimal" else "infeasible"
    if rec.get("objective") is not None:
        return "feasible"
    if status in ("missing", "hang", "error"):
        return "error"
    return "timeout"


def _summarize(rows: list[dict]) -> dict:
    proved = feasible = timeout = infeasible = error = 0
    total_wall = 0.0
    for r in rows:
        c = _classify(r)
        proved += c == "proved"
        feasible += c == "feasible"
        timeout += c == "timeout"
        infeasible += c == "infeasible"
        error += c == "error"
        total_wall += float(r.get("wall") or 0.0)
    return {
        "proved": proved,
        "feasible": feasible,
        "timeout": timeout,
        "infeasible": infeasible,
        "error": error,
        "total_wall": total_wall,
        "n": len(rows),
    }


def _run_panel(names: list[str], budget: float, jobs: int) -> dict:
    print(
        f"=== PF0 panel: {len(names)} instances, budget={budget:g}s, jobs={jobs} ===",
        file=sys.stderr,
    )
    rows = _run_parallel(
        names, mode="panel", budget=budget, children=0, sample=False, jobs=jobs, env={}
    )
    return {"budget": budget, "summary": _summarize(rows), "rows": rows}


def _print_panel(panel: dict) -> None:
    s = panel["summary"]
    print("\n=== PF0 panel outcome ===")
    hdr = f"{'instance':<18}{'status':<12}{'nodes':>8}{'bound':>16}{'objective':>16}{'wall':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in panel["rows"]:
        b = r.get("bound")
        o = r.get("objective")
        print(
            f"{r['instance']:<18}{str(r.get('status')):<12}{int(r.get('nodes') or 0):>8}"
            f"{(f'{b:.5g}' if b is not None else '-'):>16}"
            f"{(f'{o:.5g}' if o is not None else '-'):>16}"
            f"{float(r.get('wall') or 0.0):>9.1f}"
        )
    print("-" * len(hdr))
    print(
        f"proved={s['proved']} feasible={s['feasible']} timeout={s['timeout']} "
        f"infeasible={s['infeasible']} error={s['error']}  "
        f"total_wall={s['total_wall']:.0f}s  (n={s['n']})"
    )


# ---------------------------------------------------------------------------
# Diff mode (--vs REF)
# ---------------------------------------------------------------------------
def _load_rows(path: Path) -> dict[str, dict]:
    """Load a panel JSON (either {'rows': [...]} or a JSONL of records)."""
    text = path.read_text()
    rows: list[dict] = []
    try:  # whole-file JSON: a {"rows": [...]} panel dump or a bare list
        obj = json.loads(text)
        if isinstance(obj, dict) and "rows" in obj:
            rows = obj["rows"]
        elif isinstance(obj, list):
            rows = obj
    except json.JSONDecodeError:  # JSONL: one record per line
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                rows.append(json.loads(ln))
    return {r["instance"]: r for r in rows if "instance" in r}


def _tighter(sense: str, a: float, b: float) -> bool:
    """Is bound ``b`` at least as tight as ``a`` (within tol)?"""
    tol = _bound_tol(a)
    if sense == "max":  # dual bound is an upper bound: tighter = smaller
        return b <= a + tol
    return b >= a - tol  # min: tighter = larger


def _diff(cur: dict[str, dict], ref: dict[str, dict]) -> int:
    print("\n=== PF0 diff vs reference ===")
    hdr = f"{'instance':<18}{'proved Δ':<12}{'node ratio':>12}{'bound dir':>12}"
    print(hdr)
    print("-" * len(hdr))
    regressions: list[str] = []
    gained = lost = 0
    for name in sorted(set(cur) | set(ref)):
        c = cur.get(name)
        r = ref.get(name)
        if c is None:
            print(f"{name:<18}{'(new-only in ref)':<12}")
            continue
        if r is None:
            print(f"{name:<18}{'(new)':<12}")
            continue
        cp = _classify(c) == "proved"
        rp = _classify(r) == "proved"
        proved_delta = "same"
        if cp and not rp:
            proved_delta = "GAINED"
            gained += 1
        elif rp and not cp:
            proved_delta = "LOST"
            lost += 1

        cn = int(c.get("nodes") or 0)
        rn = int(r.get("nodes") or 0)
        ratio = f"{cn / rn:.2f}" if rn > 0 else ("-" if cn == 0 else "inf")

        sense = c.get("sense") or r.get("sense") or "min"
        cb = c.get("bound")
        rb = r.get("bound")
        ro = r.get("objective")
        bdir = "-"
        if cb is not None and rb is not None:
            if _tighter(sense, rb, cb):
                bdir = "ok"
            else:
                bdir = "LOOSER"
                regressions.append(f"{name}: bound {cb:.6g} looser than ref {rb:.6g} ({sense})")
            # crossed the reference objective?
            if ro is not None:
                tol = _bound_tol(ro)
                crossed = cb > ro + tol if sense == "min" else cb < ro - tol
                if crossed:
                    bdir = "CROSSED"
                    regressions.append(
                        f"{name}: bound {cb:.6g} CROSSED ref objective {ro:.6g} ({sense})"
                    )
        print(f"{name:<18}{proved_delta:<12}{ratio:>12}{bdir:>12}")
    print("-" * len(hdr))
    print(f"proofs gained={gained}  lost={lost}")
    if regressions:
        print("\n*** BOUND REGRESSION(S) DETECTED — PF0 GATE FAILED ***")
        for msg in regressions:
            print(f"  !! {msg}")
        return 1
    if lost:
        print("\n*** PROOFS LOST vs reference — PF0 GATE FAILED ***")
        return 1
    print("\nPF0 diff gate: GREEN (no looser/crossed bound, no lost proof)")
    return 0


# ---------------------------------------------------------------------------
# Differential mode (--differential --env-a --env-b)
# ---------------------------------------------------------------------------
def _parse_env(spec: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for tok in spec.split():
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(f"bad --env token (want KEY=VALUE): {tok!r}")
        k, v = tok.split("=", 1)
        env[k] = v
    return env


def _differential(
    names: list[str], env_a: dict[str, str], env_b: dict[str, str], children: int, jobs: int
) -> int:
    print("\n=== PF0 differential (env-b claimed at-least-as-tight as env-a) ===")
    print(f"  env-a: {env_a or '(baseline env)'}")
    print(f"  env-b: {env_b or '(baseline env)'}")
    rows_a = {
        r["instance"]: r
        for r in _run_parallel(
            names,
            mode="bounds",
            budget=0.0,
            children=children,
            sample=False,
            jobs=jobs,
            env=env_a,
        )
    }
    rows_b = {
        r["instance"]: r
        for r in _run_parallel(
            names,
            mode="bounds",
            budget=0.0,
            children=children,
            sample=True,
            jobs=jobs,
            env=env_b,
        )
    }

    looser: list[str] = []
    print("\n--- per-box bound comparison (b >= a - tol, sense-aware) ---")
    for name in names:
        ra, rb = rows_a.get(name), rows_b.get(name)
        if not ra or not rb or "boxes" not in ra or "boxes" not in rb:
            print(f"  {name:<16} (no bounds)")
            continue
        sense = rb.get("sense", "min")
        ba = {x["box"]: x["bound"] for x in ra["boxes"]}
        for x in rb["boxes"]:
            box, b = x["box"], x["bound"]
            a = ba.get(box)
            if a is None or b is None:
                continue
            if not _tighter(sense, a, b):
                looser.append(f"{name}/{box}: b={b:.6g} looser than a={a:.6g} ({sense})")
        root_a = ba.get("root")
        root_b = next((x["bound"] for x in rb["boxes"] if x["box"] == "root"), None)
        print(
            f"  {name:<16} root_a={_g(root_a)} root_b={_g(root_b)}  "
            f"boxes={len(rb['boxes'])} sense={sense}"
        )

    print("\n--- feasible-point soundness on env-b relaxations (0 cuts required) ---")
    worst = 0.0
    unsound: list[str] = []
    for name in names:
        rb = rows_b.get(name, {})
        if "sample_max_violation" in rb:
            v = rb["sample_max_violation"]
            worst = max(worst, v)
            flag = "  <-- CUT!" if v > 1e-5 else ""
            if v > 1e-6:
                chk = rb.get("sample_checked")
                print(f"  {name:<16} max_violation={v:.2e} checked={chk}{flag}")
            if v > 1e-5:
                unsound.append(f"{name}: violation {v:.2e}")
        elif "sample_error" in rb:
            print(f"  {name:<16} sample_error: {rb['sample_error']}")
    print(
        f"  worst violation across sampled instances: {worst:.2e} "
        f"({'SOUND' if worst <= 1e-5 else 'UNSOUND'})"
    )

    rc = 0
    if looser:
        print("\n*** ENV-B PRODUCED A LOOSER BOX — DIFFERENTIAL GATE FAILED ***")
        for m in looser:
            print(f"  !! {m}")
        rc = 1
    if unsound:
        print("\n*** ENV-B RELAXATION CUT A FEASIBLE POINT — UNSOUND ***")
        for m in unsound:
            print(f"  !! {m}")
        rc = 1
    if rc == 0:
        print("\nPF0 differential gate: GREEN (env-b at-least-as-tight, 0 cuts)")
    return rc


def _g(x: float | None) -> str:
    return f"{x:.6g}" if x is not None else "-"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--instances",
        type=str,
        default=None,
        help="comma-separated instance stems (default: all 62 vendored)",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=_DEFAULT_BUDGET,
        help="per-instance solve time limit, seconds (default 30)",
    )
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker subprocesses")
    ap.add_argument("--out", type=str, default=None, help="write panel JSON here")
    ap.add_argument(
        "--vs",
        type=str,
        default=None,
        help="diff mode: reference panel JSON/JSONL to compare against",
    )
    ap.add_argument(
        "--current",
        type=str,
        default=None,
        help="diff mode: reuse this current panel JSON instead of re-running",
    )
    ap.add_argument(
        "--differential", action="store_true", help="differential bound + soundness mode"
    )
    ap.add_argument("--env-a", type=str, default="", help="differential: env config A (KEY=V ...)")
    ap.add_argument("--env-b", type=str, default="", help="differential: env config B (KEY=V ...)")
    ap.add_argument(
        "--children", type=int, default=4, help="differential: child boxes per instance (default 4)"
    )
    # hidden worker flags
    ap.add_argument("--worker-instance", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-mode", type=str, default="panel", help=argparse.SUPPRESS)
    ap.add_argument("--sample", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_instance is not None:
        return _worker_main(args)

    if args.instances:
        names = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:
        names = _all_instances()

    if args.differential:
        return _differential(
            names, _parse_env(args.env_a), _parse_env(args.env_b), args.children, args.jobs
        )

    if args.vs:
        if args.current:
            cur = _load_rows(Path(args.current))
        else:
            panel = _run_panel(names, args.budget, args.jobs)
            _print_panel(panel)
            cur = {r["instance"]: r for r in panel["rows"]}
        ref = _load_rows(Path(args.vs))
        return _diff(cur, ref)

    # panel mode
    panel = _run_panel(names, args.budget, args.jobs)
    _print_panel(panel)
    if args.out:
        Path(args.out).write_text(json.dumps(panel, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
