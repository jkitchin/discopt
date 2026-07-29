#!/usr/bin/env python
"""Corpus sweep: OLD residual-scaled incumbent verifier vs the NEW row-scaled one.

The entry evidence for the consolidation plan's card "the incumbent verifier's
tolerance is scale-blind". Phase 5's differential panel scored ``cert-clean: FAIL(2)``
on ``nvs22`` in BOTH arms and root-caused it to the verifier, not the solver: the
tolerance ``abs_tol + rel_tol*|residual|`` is self-referential and solves to
``|r| <= abs_tol/(1 - rel_tol)`` — a pure **absolute** 1e-6 on every row scale.

This script answers the §0.4 question that a scale term always raises: *what does
the new form accept that the old one rejected, and did it accept anything it should
not have?* For every in-repo instance it solves once, takes the returned incumbent,
and scores it under BOTH tolerance forms — the old one re-implemented inline here so
the comparison is of *forms*, not of git revisions.

Reported per instance:

* ``old_ok`` / ``new_ok`` — the two verdicts;
* ``worst_abs`` / ``worst_rel`` — the largest row violation, absolute and relative
  to that row's own scale ``max(1, |b_i|, max_j |J_ij|*max(1,|x_j|))``;
* ``old_rows_examined`` vs ``rows`` — the alignment defect made visible: the old
  loop advanced one index per *constraint object* while the evaluator emits one row
  per *flat element*, so a model with vector constraints had rows it never looked at.

A ``True -> False`` flip is the alarming direction (the new form rejecting a point
the old one accepted is fine and expected — it is stricter — but it must be
attributable). A ``False -> True`` flip is only acceptable when ``worst_rel`` is
comfortably inside the relative tolerance; the summary prints the worst one so the
claim is bounded by a number rather than an adjective.

Usage::

    python -u discopt_benchmarks/scripts/verifier_scale_sweep.py --budget 20
    python -u discopt_benchmarks/scripts/verifier_scale_sweep.py --subset 12

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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "discopt_benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.panel_baseline import _short_sha, corpus_instances, instance_path  # noqa: E402

_REPORTS_DIR = _REPO_ROOT / "reports"
_ABS_TOL = 1e-6
_REL_TOL = 1e-4


def _old_form_verdict(model, x_flat):
    """The PRE-FIX ``_native_kernel_verify_point`` row loop, re-implemented verbatim.

    Kept here (rather than imported from git history) so the sweep compares tolerance
    *forms* on one tree. Returns ``(ok, rows_examined)``.
    """
    import numpy as np
    from discopt._jax.nlp_evaluator import cached_evaluator
    from discopt.modeling.core import Constraint

    evaluator = cached_evaluator(model)
    if evaluator.n_constraints == 0:
        return True, 0
    cons = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
    idx = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if idx >= cons.shape[0]:
            return False, idx
        val = float(cons[idx])
        if not math.isfinite(val):
            return False, idx
        tol = _ABS_TOL + _REL_TOL * abs(val)
        if c.sense == "<=":
            if val > tol:
                return False, idx + 1
        elif c.sense == ">=":
            if val < -tol:
                return False, idx + 1
        elif c.sense == "==":
            if abs(val) > tol:
                return False, idx + 1
        else:
            return False, idx
        idx += 1
    return True, idx


def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt
    import numpy as np
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import from_nl
    from discopt.validation.feasibility import row_scales, verify_point

    out: dict = {
        "instance": instance,
        "budget": float(budget),
        "discopt_file": discopt.__file__,
        # CLAUDE.md §8 marker: the module under test. Its absence means the child
        # imported a pre-fix discopt and every verdict below is from the wrong tree.
        "has_scale_verifier": True,
    }
    model = from_nl(str(instance_path(instance)))
    t0 = time.perf_counter()
    r = model.solve(time_limit=budget)
    out["wall"] = time.perf_counter() - t0
    out["status"] = str(r.status)
    out["objective"] = None if r.objective is None else float(r.objective)
    out["gap_certified"] = bool(r.gap_certified)
    if r.x is None:
        out["incumbent"] = False
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 0
    out["incumbent"] = True

    flat = np.concatenate([np.asarray(r.x[v.name], dtype=float).ravel() for v in model._variables])

    fresh_new = from_nl(str(instance_path(instance)))
    verdict = verify_point(fresh_new, flat)
    out["new_ok"] = bool(verdict.ok)
    out["new_refusal"] = verdict.refusal
    out["rows"] = int(verdict.n_rows_checked)
    out["new_worst_rel_reported"] = float(verdict.worst_relative)

    fresh_old = from_nl(str(instance_path(instance)))
    old_ok, old_rows = _old_form_verdict(fresh_old, flat)
    out["old_ok"] = bool(old_ok)
    out["old_rows_examined"] = int(old_rows)

    # Independent per-row residual/scale table (does not reuse either verifier).
    ev = NLPEvaluator(from_nl(str(instance_path(instance))))
    if ev.n_constraints > 0:
        body = np.asarray(ev.evaluate_constraints(flat), dtype=float)
        senses, rhss = [], []
        for c, sz in zip(
            ev._source_constraints,
            np.asarray(ev._constraint_flat_sizes).tolist(),
            strict=True,
        ):
            sz = int(sz)
            s = c.sense if isinstance(c.sense, str) else getattr(c.sense, "value", c.sense)
            senses.extend([s] * sz)
            rhss.extend([float(c.rhs)] * sz)
        senses = np.asarray(senses, dtype=object)
        rhss = np.asarray(rhss, dtype=float)
        signed = body - rhss
        viol = np.zeros_like(signed)
        viol[senses == "<="] = np.maximum(signed[senses == "<="], 0.0)
        viol[senses == ">="] = np.maximum(-signed[senses == ">="], 0.0)
        viol[senses == "=="] = np.abs(signed[senses == "=="])
        try:
            jac = np.asarray(ev.evaluate_jacobian(flat), dtype=float)
        except Exception as exc:  # recorded, never swallowed (CLAUDE.md §7)
            out["jacobian_error"] = repr(exc)[:200]
            jac = None
        scale = row_scales(jac, rhss, flat)
        rel = viol / scale
        out["worst_abs"] = float(np.max(viol))
        out["worst_rel"] = float(np.max(rel))
        out["worst_rel_row_scale"] = float(scale[int(np.argmax(rel))])
        out["n_rows_over_abs_tol"] = int(np.sum(viol > _ABS_TOL))
    else:
        out["worst_abs"] = 0.0
        out["worst_rel"] = 0.0
        out["n_rows_over_abs_tol"] = 0

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _solve(instance: str, budget: float) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve",
        instance,
        str(budget),
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=budget + 180.0, env=env)
    except subprocess.TimeoutExpired:
        return {"instance": instance, "status": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "status": "child_no_result",
        "stderr_tail": proc.stderr[-400:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", nargs=2, metavar=("INSTANCE", "BUDGET"))
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()

    if args.solve:
        return _run_child(args.solve[0], float(args.solve[1]))

    instances = sorted(corpus_instances())
    if args.subset:
        instances = instances[: args.subset]
    print(f"instances: {len(instances)}  budget: {args.budget}s", flush=True)
    print(f"load at start: {os.getloadavg()}", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances, 1):
        row = _solve(inst, args.budget)
        rows.append(row)
        print(
            f"[{i}/{len(instances)}] {inst:<24} status={row.get('status')} "
            f"inc={row.get('incumbent')} old={row.get('old_ok')} new={row.get('new_ok')} "
            f"worst_abs={row.get('worst_abs')} worst_rel={row.get('worst_rel')}",
            flush=True,
        )
    wall = time.perf_counter() - t0

    scored = [r for r in rows if r.get("incumbent") and "old_ok" in r and "new_ok" in r]
    executed = len(scored)
    flips_ft = [r for r in scored if not r["old_ok"] and r["new_ok"]]
    flips_tf = [r for r in scored if r["old_ok"] and not r["new_ok"]]
    agree = [r for r in scored if r["old_ok"] == r["new_ok"]]
    under_examined = [r for r in scored if r.get("old_rows_examined", 0) < r.get("rows", 0)]

    print("\n## VERDICT", flush=True)
    print(f"  EXECUTED COMPARISONS : {executed}", flush=True)
    print(f"  agree                : {len(agree)}", flush=True)
    print(f"  False -> True (old wrongly rejected) : {len(flips_ft)}", flush=True)
    for r in sorted(flips_ft, key=lambda r: -r.get("worst_rel", 0.0)):
        print(
            f"      {r['instance']:<24} worst_abs={r['worst_abs']:.4e} "
            f"worst_rel={r['worst_rel']:.4e} (row scale {r['worst_rel_row_scale']:.4e}) "
            f"status={r['status']} certified={r['gap_certified']}",
            flush=True,
        )
    print(f"  True -> False (new rejects)          : {len(flips_tf)}", flush=True)
    for r in flips_tf:
        print(
            f"      {r['instance']:<24} worst_abs={r['worst_abs']:.4e} "
            f"worst_rel={r['worst_rel']:.4e} refusal={r.get('new_refusal')}",
            flush=True,
        )
    print(
        f"  rows the OLD loop never examined     : {len(under_examined)} instances",
        flush=True,
    )
    for r in under_examined:
        print(
            f"      {r['instance']:<24} examined {r['old_rows_examined']} of {r['rows']}",
            flush=True,
        )
    if flips_ft:
        worst = max(r["worst_rel"] for r in flips_ft)
        print(f"  worst RELATIVE violation newly accepted: {worst:.4e}", flush=True)
    print(f"  wall {wall:.1f}s  load at end {os.getloadavg()}", flush=True)

    _REPORTS_DIR.mkdir(exist_ok=True)
    out_path = _REPORTS_DIR / f"verifier_scale_sweep_{_short_sha()}.json"
    out_path.write_text(
        json.dumps(
            {
                "budget": args.budget,
                "wall": wall,
                "executed_comparisons": executed,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"  artifact: {out_path}", flush=True)

    if executed == 0:
        print("FAIL: zero executed comparisons — the probe measured nothing", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
