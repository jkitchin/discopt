"""Card 2a — differential panel for the #208 aux cascade wired at all six sites.

Consolidation plan (``docs/dev/consolidation-plan-2026-07-28.md``) Card 2a. The
``DISCOPT_OBBT_CASCADE_AUX`` flag graduated default-ON in #208, but it was resolved
and passed at exactly **one** of the six ``obbt_tighten_root`` call sites; the other
five took the function default ``False``. Card 2a resolves it once (in
``SolverTuning``) and passes it everywhere, which *changes bounds* at five sites —
Regime C (``§0.1``), so it needs a differential ON-vs-OFF panel.

The two arms
------------

* **OFF arm** — the frozen Phase 0 baseline ``reports/panel_baseline_f154dcff.json``:
  defaults, 45 s, 119 instances, i.e. the cascade live at the single ``root_reduce``
  site. This is the *status quo*, and it is not a stale artifact: Phase 1's
  ``panel_baseline.py --check`` reproduced it on this tree (255 comparisons, PASS),
  so its node counts and certified objectives are the current behaviour.
* **ON arm** — this script, run on the Card 2a tree at the baseline's own budget and
  corpus, defaults (cascade at all six sites).

Running the OFF arm from the frozen artifact rather than re-solving it is deliberate:
node counts and certified objectives are deterministic and were just re-verified,
while re-running 119 instances twice buys nothing but an hour of machine time. Wall
deltas *are* temporally separated and are reported as such — they are context, never
the verdict (CLAUDE.md §9).

Proving the wiring fired
------------------------

A wiring panel that shows "no change" is ambiguous: it can mean the change is benign
or that it never ran. So the child **wraps** ``obbt_tighten_root`` and records, per
call, the caller's ``file:line`` and the ``cascade_aux`` value it received. The
summary prints the per-site call counts; **zero cascade-ON calls at the five new
sites over the whole corpus is a FAILURE**, not a pass (CLAUDE.md §6). The wrapper
lives here, in the harness, never in the shipped hot path.

Gates (CLAUDE.md §5 / plan §0.1)
--------------------------------

* **cert-clean** — no certified objective disagrees with OFF, no OFF-optimal
  instance regresses to non-optimal, no dual bound passes the reference optimum
  (sense-aware, ``=best=`` oracles excluded from the soundness arm since an unproven
  value is not ground truth), no ``gap_certified`` regression.
* **quality-clean** (#902) — no instance's incumbent degrades vs OFF.
* **net effect** — total/instance node counts, certifications gained and lost, wall.

Usage::

    python -u discopt_benchmarks/scripts/card2a_cascade_aux_panel.py
    python -u discopt_benchmarks/scripts/card2a_cascade_aux_panel.py --subset 10
    python -u discopt_benchmarks/scripts/card2a_cascade_aux_panel.py --on reports/x.json

Internal child mode: ``--solve <instance> <budget>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

_DEFAULT_BASELINE = _REPO_ROOT / "reports" / "panel_baseline_f154dcff.json"
_REPORTS_DIR = _REPO_ROOT / "reports"

_ABS_TOL = 1e-6
_REL_TOL = 1e-4
_CHILD_TIMEOUT_SLACK = 120.0

#: The five sites Card 2a newly wires (the sixth, ``root_reduce``, was already ON).
_NEW_SITE_FILES = (
    "solver.py",
    "lp_spatial_bb.py",
    "disjunctive_config_bound.py",
)


# --------------------------------------------------------------------------- #
# Child: solve ONE instance on defaults, with an obbt_tighten_root call tap.   #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    import discopt._jax.obbt as _obbt_mod  # noqa: PLC0415
    from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: PLC0415
    from scripts.panel_baseline import instance_path  # noqa: PLC0415

    # --- the call tap ------------------------------------------------------- #
    # Every production call site does a *local* ``from discopt._jax.obbt import
    # obbt_tighten_root`` inside the calling function, so the name is resolved at
    # call time and patching the module attribute here catches all six.
    sites: dict[str, dict] = {}
    _orig = _obbt_mod.obbt_tighten_root

    def _tap(*a, **kw):
        frame = sys._getframe(1)
        key = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
        rec = sites.setdefault(key, {"calls": 0, "cascade_on": 0, "cascade_off": 0})
        rec["calls"] += 1
        rec["cascade_on" if kw.get("cascade_aux") else "cascade_off"] += 1
        # No try/except: a failure inside the tap must crash the child loudly
        # (CLAUDE.md §7) rather than degrade the probe into a no-op.
        return _orig(*a, **kw)

    _obbt_mod.obbt_tighten_root = _tap

    nl = str(instance_path(instance))
    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "budget": float(budget),
    }
    try:
        model = from_nl(nl)
        out["sense"] = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        t0 = time.perf_counter()
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["gap"] = None if r.gap is None else float(r.gap)
        out["gap_certified"] = bool(r.gap_certified)
        out["node_count"] = int(r.node_count)
        out["root_bound"] = None if r.root_bound is None else float(r.root_bound)
        out["root_gap"] = None if r.root_gap is None else float(r.root_gap)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    out["obbt_sites"] = sites

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _solve_one(instance: str, budget: float) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance, str(budget)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget + _CHILD_TIMEOUT_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "status": "child_timeout", "budget": float(budget)}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "status": "child_crashed",
        "budget": float(budget),
        "stderr_tail": proc.stderr[-800:],
    }


def _load1() -> float:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # pragma: no cover
        return float("nan")


def _obj_match(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


def _run_on_arm(instances: list[str], budget: float) -> tuple[list[dict], dict]:
    load_start = _load1()
    load_peak = 0.0 if math.isnan(load_start) else load_start
    t0 = time.perf_counter()
    print(
        f"Card 2a ON arm: {len(instances)} instance(s), {budget:.0f}s budget, DEFAULT "
        f"settings (cascade wired at all six sites), one subprocess per instance.",
        flush=True,
    )
    print(f"1-min load at start: {load_start:.2f}\n", flush=True)
    rows: list[dict] = []
    for i, inst in enumerate(instances, 1):
        row = _solve_one(inst, budget)
        rows.append(row)
        lv = _load1()
        if not math.isnan(lv):
            load_peak = max(load_peak, lv)
        n_new = sum(
            r["cascade_on"]
            for k, r in (row.get("obbt_sites") or {}).items()
            if k.split(":")[0] in _NEW_SITE_FILES
        )
        print(
            f"  [{i:3d}/{len(instances)}] {inst:24s} {str(row.get('status')):11s} "
            f"nodes={str(row.get('node_count', '-')):>8s} "
            f"obj={str(row.get('objective'))[:14]:>14s} "
            f"bound={str(row.get('bound'))[:14]:>14s} "
            f"cert={'Y' if row.get('gap_certified') else '.'} "
            f"w={row.get('wall', float('nan')):6.1f} "
            f"newsite_cascade_calls={n_new}",
            flush=True,
        )
    return rows, {
        "budget_seconds": budget,
        "total_wall_seconds": time.perf_counter() - t0,
        "load_start": load_start,
        "load_peak": load_peak,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _oracles() -> dict:
    from utils.reference_optima import reference_oracle  # noqa: PLC0415

    return reference_oracle


def evaluate(off_rows: dict, on_rows: dict, oracle) -> dict:
    """Differential ON-vs-OFF verdict. Every check counts what it executed."""
    cert_violations: list[str] = []
    quality_violations: list[str] = []
    cert_gained: list[str] = []
    cert_lost: list[str] = []
    node_changed: list[tuple[str, int, int]] = []
    errored: list[str] = []
    counts = {
        "obj_agreement": 0,
        "optimality_regression": 0,
        "bound_vs_oracle": 0,
        "cert_status": 0,
        "quality": 0,
        "node": 0,
    }
    bad = ("errored", "child_crashed", "child_timeout")
    tot_off = tot_on = 0
    wall_off = wall_on = 0.0

    for inst in sorted(set(off_rows) & set(on_rows)):
        off, on = off_rows[inst], on_rows[inst]
        off_status, on_status = str(off.get("status")), str(on.get("status"))
        if on_status in bad:
            errored.append(f"{inst}: ON {on_status} {str(on.get('error', ''))[:120]}")
        if off_status in bad or on_status in bad:
            continue
        sense = on.get("sense") or off.get("sense") or "min"

        # (1) both optimal -> objectives must agree.
        if off_status == "optimal" and on_status == "optimal":
            counts["obj_agreement"] += 1
            if not _obj_match(off.get("objective"), on.get("objective")):
                cert_violations.append(
                    f"{inst}: optimal objective mismatch OFF={off.get('objective')} "
                    f"ON={on.get('objective')}"
                )

        # (2) OFF-optimal must not regress.
        if off_status == "optimal":
            counts["optimality_regression"] += 1
            if on_status != "optimal":
                cert_violations.append(f"{inst}: OFF optimal but ON {on_status}")

        # (3) ON dual bound must not pass a PROVEN reference optimum.
        ref = oracle(inst)
        if ref is not None and ref.proven and isinstance(on.get("bound"), (int, float)):
            counts["bound_vs_oracle"] += 1
            b, opt = float(on["bound"]), float(ref.value)
            tol = _ABS_TOL + _REL_TOL * max(abs(opt), abs(b))
            if sense == "min" and b > opt + tol:
                cert_violations.append(
                    f"{inst}: ON lower bound {b} ABOVE reference optimum {opt} ({ref.source})"
                )
            if sense == "max" and b < opt - tol:
                cert_violations.append(
                    f"{inst}: ON upper bound {b} BELOW reference optimum {opt} ({ref.source})"
                )

        # (4) certification must not regress.
        counts["cert_status"] += 1
        c_off, c_on = bool(off.get("gap_certified")), bool(on.get("gap_certified"))
        if c_off and not c_on:
            cert_lost.append(inst)
            cert_violations.append(f"{inst}: CERTIFICATION LOST (gap_certified True -> False)")
        elif c_on and not c_off:
            cert_gained.append(inst)

        # (5) incumbent quality (#902).
        o_obj, n_obj = off.get("objective"), on.get("objective")
        if o_obj is not None:
            counts["quality"] += 1
            if n_obj is None:
                quality_violations.append(
                    f"{inst}: PRIMAL LOST — OFF {o_obj} ({off_status}), ON none ({on_status})"
                )
            else:
                qtol = _ABS_TOL + _REL_TOL * max(abs(o_obj), abs(n_obj))
                worse = (n_obj > o_obj + qtol) if sense == "min" else (n_obj < o_obj - qtol)
                if worse:
                    quality_violations.append(
                        f"{inst}: INCUMBENT WORSE OFF={o_obj} ({off_status}) "
                        f"ON={n_obj} ({on_status})"
                    )

        # (6) net effect.
        no, nn = off.get("node_count"), on.get("node_count")
        if isinstance(no, int) and isinstance(nn, int):
            counts["node"] += 1
            tot_off += no
            tot_on += nn
            if no != nn:
                node_changed.append((inst, no, nn))
        if isinstance(off.get("wall"), (int, float)) and isinstance(on.get("wall"), (int, float)):
            wall_off += float(off["wall"])
            wall_on += float(on["wall"])

    return {
        "counts": counts,
        "executed_comparisons": sum(counts.values()),
        "cert_violations": cert_violations,
        "cert_clean": not cert_violations,
        "quality_violations": quality_violations,
        "quality_clean": not quality_violations,
        "cert_gained": cert_gained,
        "cert_lost": cert_lost,
        "node_changed": node_changed,
        "nodes_off": tot_off,
        "nodes_on": tot_on,
        "wall_off": wall_off,
        "wall_on": wall_on,
        "errored": errored,
    }


def _site_totals(rows: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for r in rows:
        for key, rec in (r.get("obbt_sites") or {}).items():
            a = agg.setdefault(key, {"calls": 0, "cascade_on": 0, "cascade_off": 0, "instances": 0})
            a["calls"] += rec["calls"]
            a["cascade_on"] += rec["cascade_on"]
            a["cascade_off"] += rec["cascade_off"]
            a["instances"] += 1
    return dict(sorted(agg.items()))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == "--solve":
        return _run_child(argv[1], float(argv[2]))

    p = argparse.ArgumentParser(description="Card 2a cascade_aux differential panel.")
    p.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    p.add_argument("--subset", help="integer count or comma-separated instance names")
    p.add_argument("--on", help="reuse a previously written ON-arm artifact instead of re-solving")
    p.add_argument("--out", help="ON-arm artifact path")
    args = p.parse_args(argv)

    base = json.loads(Path(args.baseline).read_text())
    if base.get("schema") != "panel_baseline/1":
        raise SystemExit(f"ERROR: {args.baseline} is not a panel_baseline/1 artifact.")
    off_rows = {r["instance"]: r for r in base["rows"]}
    budget = float(base["budget_seconds"])

    from scripts.panel_baseline import _resolve_subset  # noqa: PLC0415

    instances = _resolve_subset(sorted(off_rows), args.subset)

    if args.on:
        on_art = json.loads(Path(args.on).read_text())
        on_list, meta = on_art["rows"], on_art.get("meta", {})
        on_list = [r for r in on_list if r["instance"] in set(instances)]
    else:
        on_list, meta = _run_on_arm(instances, budget)
        out = Path(args.out) if args.out else _REPORTS_DIR / "card2a_cascade_aux_on.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"schema": "card2a/1", "meta": meta, "rows": on_list}, indent=1) + "\n"
        )
        print(f"\nON-arm artifact written: {out}", flush=True)

    on_rows = {r["instance"]: r for r in on_list}
    v = evaluate(off_rows, on_rows, _oracles())
    sites = _site_totals(on_list)

    print("\n" + "=" * 78, flush=True)
    print("obbt_tighten_root call sites observed in the ON arm:", flush=True)
    for key, rec in sites.items():
        print(
            f"  {key:44s} calls={rec['calls']:6d}  cascade_on={rec['cascade_on']:6d}  "
            f"cascade_off={rec['cascade_off']:6d}  instances={rec['instances']:4d}",
            flush=True,
        )
    new_on = sum(r["cascade_on"] for k, r in sites.items() if k.split(":")[0] in _NEW_SITE_FILES)
    print(f"  -> cascade-ON calls at the FIVE NEW sites: {new_on}", flush=True)

    print("\n" + "-" * 78, flush=True)
    print(
        f"executed comparisons: {v['executed_comparisons']} {v['counts']}",
        flush=True,
    )
    print(
        f"nodes  OFF={v['nodes_off']}  ON={v['nodes_on']}  "
        f"({v['nodes_on'] - v['nodes_off']:+d}, "
        f"{100 * (v['nodes_on'] - v['nodes_off']) / max(1, v['nodes_off']):+.1f}%) over "
        f"{v['counts']['node']} paired rows",
        flush=True,
    )
    print(
        f"wall   OFF={v['wall_off']:.1f}s  ON={v['wall_on']:.1f}s "
        f"({v['wall_on'] - v['wall_off']:+.1f}s) "
        "[temporally separated arms — context, not a timing claim]",
        flush=True,
    )
    print(f"certifications gained: {len(v['cert_gained'])} {v['cert_gained']}", flush=True)
    print(f"certifications lost  : {len(v['cert_lost'])} {v['cert_lost']}", flush=True)
    print(f"node counts changed  : {len(v['node_changed'])}", flush=True)
    for inst, a, b in v["node_changed"]:
        print(f"    {inst:24s} {a} -> {b} ({b - a:+d})", flush=True)
    if v["errored"]:
        print(f"\nERRORED ({len(v['errored'])}):", flush=True)
        for e in v["errored"]:
            print(f"  - {e}", flush=True)
    print(
        f"\ncert-clean   : {'PASS' if v['cert_clean'] else 'FAIL'} "
        f"({len(v['cert_violations'])} violation(s))",
        flush=True,
    )
    for s in v["cert_violations"]:
        print(f"  - {s}", flush=True)
    print(
        f"quality-clean: {'PASS' if v['quality_clean'] else 'FAIL'} "
        f"({len(v['quality_violations'])} violation(s))",
        flush=True,
    )
    for s in v["quality_violations"]:
        print(f"  - {s}", flush=True)
    print("=" * 78, flush=True)

    if v["executed_comparisons"] == 0:
        print("FAIL: zero comparisons executed — this panel proved nothing.", flush=True)
        return 3
    if new_on == 0 and not args.subset:
        print(
            "FAIL: the five newly-wired sites never ran with the cascade ON over the "
            "whole corpus — the panel measured nothing about the change (CLAUDE.md §6).",
            flush=True,
        )
        return 4
    if not v["cert_clean"]:
        return 1
    if not v["quality_clean"]:
        return 2
    print("PASS: cert-clean and quality-clean.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
