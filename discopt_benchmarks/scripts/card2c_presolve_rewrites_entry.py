"""Card 2c.2 **entry experiment** — how much of the Rust presolve is thrown away?

Consolidation plan Card 2c.2. ``run_root_presolve`` runs the Rust orchestrator and
keeps the **presolved repr** (so the repr-side rewrites do reach in-tree FBBT and
the root FBBT), but the only thing pushed back into the Python ``Model`` DAG is the
*bounds* vector (``propagate_bounds_to_model``). Every rewrite a pass makes to the
model *structure* — ``simplify``'s big-M work, ``coefficient_strengthening``'s row
rewrites, ``redundancy``'s row removals — is invisible to the Python relaxation
compiler, which is compiled from the Python DAG.

The card's decision rule:

* **a nontrivial set of instances has non-bound rewrites** ⇒ repr-level adoption is
  the fix, and it is far too large for this card — record the measured counts and
  hand it to a named follow-up;
* **near-zero instances affected** ⇒ drop the three passes from the default list
  (a Regime-C differential panel) and record the measurement.

What is counted, per pass, per instance: rows removed, rows rewritten, variables
fixed, aux vars/constraints introduced, and (separately, because it *is* adopted)
half-bounds tightened. The distinction that decides the card is
**non-bound changes vs bound-only changes**.

This probe does no solve: it builds the repr exactly as ``solve_model`` does and
runs the orchestrator with ``solve_model``'s own arguments, so it is cheap and
measures the real default pass list. Per CLAUDE.md §6 it prints the executed
per-pass delta count and exits non-zero when nothing was examined; per §7 nothing
around the measurement swallows an exception.

Usage::

    python -u discopt_benchmarks/scripts/card2c_presolve_rewrites_entry.py
    python -u discopt_benchmarks/scripts/card2c_presolve_rewrites_entry.py --subset 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

#: The three passes the card is about.
_TARGET_PASSES = ("simplify", "coefficient_strengthening", "redundancy")

#: Non-bound delta fields — the ones ``propagate_bounds_to_model`` cannot carry.
_LIST_FIELDS = ("constraints_removed", "constraints_rewritten", "vars_fixed")
_INT_FIELDS = ("aux_vars_introduced", "aux_constraints_introduced")

_CHILD_TIMEOUT = 300.0


def _run_child(instance: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    from discopt._jax.presolve_pipeline import run_root_presolve  # noqa: PLC0415
    from discopt._rust import model_to_repr  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415
    from scripts.panel_baseline import instance_path  # noqa: PLC0415

    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "rust_file": sys.modules["discopt._rust"].__file__,
    }
    try:
        model = from_nl(str(instance_path(instance)))
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        t0 = time.perf_counter()
        # Exactly solve_model's call (solver.py root-presolve block): every other
        # argument takes run_root_presolve's own default, which IS the default
        # pass list. Measuring a different list would measure a different solver.
        _new_repr, stats = run_root_presolve(
            repr_,
            eliminate=True,
            polynomial=False,
            fbbt=True,
            time_limit_ms=30_000,
        )
        out["presolve_wall"] = time.perf_counter() - t0
        per_pass: dict[str, dict] = {}
        for d in stats.get("deltas", []):
            name = str(d["pass_name"])
            agg = per_pass.setdefault(
                name,
                {f: 0 for f in (*_LIST_FIELDS, *_INT_FIELDS)} | {"bounds_tightened": 0, "n": 0},
            )
            agg["n"] += 1
            agg["bounds_tightened"] += int(d.get("bounds_tightened", 0))
            for f in _LIST_FIELDS:
                agg[f] += len(d.get(f, []) or [])
            for f in _INT_FIELDS:
                agg[f] += int(d.get(f, 0))
        out["per_pass"] = per_pass
        out["iterations"] = stats.get("iterations")
        out["terminated_by"] = stats.get("terminated_by")
        out["n_deltas"] = len(stats.get("deltas", []))
    except Exception as exc:
        out["error"] = repr(exc)
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _solve_one(instance: str) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_CHILD_TIMEOUT, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "error": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {"instance": instance, "error": "child_crashed", "stderr_tail": proc.stderr[-800:]}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--solve":
        return _run_child(argv[1])

    p = argparse.ArgumentParser(description="Card 2c.2 entry experiment.")
    p.add_argument("--subset", help="integer count or comma-separated instance names")
    p.add_argument(
        "--out", default=str(_REPO_ROOT / "reports" / "card2c_presolve_rewrites.json")
    )
    args = p.parse_args(argv)

    from scripts.panel_baseline import _resolve_subset, corpus_instances  # noqa: PLC0415

    instances = _resolve_subset(corpus_instances(), args.subset)
    print(f"Card 2c.2 entry experiment over {len(instances)} corpus instance(s).", flush=True)

    rows = []
    n_deltas = 0
    affected: dict[str, list[str]] = {p_: [] for p_ in _TARGET_PASSES}
    bound_only: dict[str, list[str]] = {p_: [] for p_ in _TARGET_PASSES}
    totals: dict[str, dict] = {}
    errors = []
    for i, inst in enumerate(instances, 1):
        row = _solve_one(inst)
        rows.append(row)
        if row.get("error"):
            errors.append(f"{inst}: {row['error']}")
            print(f"  [{i:3d}/{len(instances)}] {inst:24s} ERROR {row['error']}", flush=True)
            continue
        n_deltas += int(row.get("n_deltas", 0))
        per_pass = row.get("per_pass", {})
        marks = []
        for name in _TARGET_PASSES:
            agg = per_pass.get(name)
            if not agg:
                continue
            t = totals.setdefault(name, {k: 0 for k in agg})
            for k, val in agg.items():
                t[k] += int(val)
            nonbound = sum(int(agg[f]) for f in (*_LIST_FIELDS, *_INT_FIELDS))
            if nonbound:
                affected[name].append(inst)
                marks.append(f"{name}:{nonbound}")
            elif agg["bounds_tightened"]:
                bound_only[name].append(inst)
        print(
            f"  [{i:3d}/{len(instances)}] {inst:24s} deltas={row.get('n_deltas'):3d} "
            f"iters={row.get('iterations')} {row.get('terminated_by')} "
            f"nonbound=[{', '.join(marks) if marks else '-'}]",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "totals": totals,
                "affected": affected,
                "bound_only": bound_only,
                "errors": errors,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )

    n_ok = len(instances) - len(errors)
    print("\n" + "=" * 78, flush=True)
    print(
        f"executed: {n_ok} instances presolved, {n_deltas} per-pass deltas examined "
        f"({len(errors)} errored)",
        flush=True,
    )
    for name in _TARGET_PASSES:
        t = totals.get(name, {})
        print(
            f"{name:28s} fired on {t.get('n', 0):4d} pass-invocation(s); "
            f"rows_removed={t.get('constraints_removed', 0):6d} "
            f"rows_rewritten={t.get('constraints_rewritten', 0):6d} "
            f"vars_fixed={t.get('vars_fixed', 0):6d} "
            f"aux_vars={t.get('aux_vars_introduced', 0):5d} "
            f"bounds_tightened={t.get('bounds_tightened', 0):7d}",
            flush=True,
        )
        print(
            f"{'':28s} instances with NON-BOUND rewrites: {len(affected[name]):3d}"
            f"{' -> ' + ', '.join(affected[name][:12]) if affected[name] else ''}"
            f"{' …' if len(affected[name]) > 12 else ''}",
            flush=True,
        )
    total_affected = sorted({i for lst in affected.values() for i in lst})
    print(
        f"\nINSTANCES WITH ANY NON-BOUND REWRITE FROM THE THREE PASSES: "
        f"{len(total_affected)}/{n_ok}",
        flush=True,
    )
    print("=" * 78, flush=True)
    if n_deltas == 0:
        print(
            "FAIL: zero per-pass deltas examined — the probe measured nothing "
            "(CLAUDE.md §6).",
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
