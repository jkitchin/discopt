"""Phase 5.2 entry experiment — census rank #2, ``infinite_aux_bounds``.

The Phase 5.1 coverage census (``reports/phase5_kernel_coverage_census_c346fd73.json``)
ranks ``infinite_aux_bounds`` second: **9 instances, 379.6 s, 19.7 %** of baseline
wall.  Card 5.2-T then showed rank #1's headline was an *upper bound* — the producer's
decline ladder is ordered, so the first code masks whatever else the model would
decline on.  Two questions must be answered before any code is written, and this
script answers both on the **real** call site (the presolved root box, exactly as the
census taps it — a static pre-filter reading declared bounds gets this wrong on the
instances that matter, #902/tanksize).

E8.1 — FRAMING (is rank #2 masked the way rank #1 was?)
    Hypothesis: removing the ``infinite_aux_bounds`` test lets **>= 6 of 9** instances
    reach row-claiming (a spec is built, or the decline moves to the row-claiming
    stage), so 379.6 s / 19.7 % is close to recoverable rather than an upper bound.
    KILL: if <= 4 of 9 reach row-claiming, the rank-2 framing does not survive and the
    card is re-picked from the census ranking.

E8.2 — REPAIRABILITY (can the kernel represent these models at all?)
    Hypothesis: the infinite auxiliary bounds are an artifact of *forward* interval
    propagation, and are repairable by bound propagation on the relaxation's **own
    rows** (FBBT over ``A_ub x <= b_ub`` at the root box) — i.e. the producer declines
    conservatively rather than facing an inexpressible model.
    KILL: if on **>= 5 of 9** instances infinities survive FBBT to fixpoint, the repair
    is not "wire up propagation", and the card is not a producer change.

Method notes (CLAUDE.md §6-§10)
-------------------------------

* One subprocess per instance, ``DISCOPT_NATIVE_SPATIAL_KERNEL=1``, tapping
  ``build_spatial_kernel_spec`` at its live call site; the child exits as soon as the
  diagnosis is emitted, so a decline costs root-setup wall, not a budget.
* **No swallowed exceptions** (§7): the child does not wrap the diagnosis in
  ``try``.  A broken probe crashes and is reported as ``child_crashed``.
* **Executed counts** (§6): every arm carries its own counter, printed at the end,
  and the script exits non-zero when any load-bearing counter is zero.
* **Loaded-code assertion** (§8): the child asserts ``discopt.__file__`` and the
  presence of ``producer_stats`` before measuring, and asserts that the source of
  ``spatial_producer`` contains exactly ONE ``np.isinf`` call — the bypass shim used
  by arm E8.1 is only faithful if that call site is the sole one.

Usage
-----

::

    python -u discopt_benchmarks/scripts/phase52_infinite_aux_entry.py
    python -u discopt_benchmarks/scripts/phase52_infinite_aux_entry.py --subset gear4,hda

Internal child mode: ``--solve <instance> <budget>`` (one instance, one JSON line).
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

from scripts.panel_baseline import (  # noqa: E402
    _git_dirty,
    _load1,
    _short_sha,
    instance_path,
)

_INSTRUMENT_MARKER = "phase52-infinite-aux-entry-v1"
_BASELINE = _REPO_ROOT / "reports" / "panel_baseline_f154dcff.json"
_CENSUS = _REPO_ROOT / "reports" / "phase5_kernel_coverage_census_c346fd73.json"
_REPORTS_DIR = _REPO_ROOT / "reports"
_DEFAULT_BUDGET = 45.0
_CHILD_TIMEOUT_SLACK = 180.0

_INF = float("inf")
_FBBT_PASSES = 30


# --------------------------------------------------------------------------- #
# FBBT over the relaxation's own rows (the E8.2 instrument)                    #
# --------------------------------------------------------------------------- #
def fbbt_columns(a_ub, b, lo, hi, n_orig, passes=_FBBT_PASSES, tighten_original=False):
    """Interval bound propagation on ``A x <= b`` starting from ``[lo, hi]``.

    Returns ``(lo, hi, stats)``.  Pure measurement: the relaxation LP's own rows are
    valid at the root box, so any bound this derives is implied by the relaxation and
    cutting a column to it removes no LP-feasible point.  ``tighten_original=False``
    leaves the branchable original columns alone (they are the solver's box, not the
    kernel's to shrink here).

    ``stats`` counts rounds, tightenings applied and any emptiness detected, so the
    caller can prove the probe fired (§6).
    """
    import numpy as np

    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    indptr, indices, data = a_ub.indptr, a_ub.indices, a_ub.data
    nrows = a_ub.shape[0]
    st = {"rounds": 0, "tightenings": 0, "row_scans": 0, "empty": False}
    for _rnd in range(passes):
        st["rounds"] += 1
        changed = False
        for r in range(nrows):
            p0, p1 = int(indptr[r]), int(indptr[r + 1])
            if p1 <= p0:
                continue
            st["row_scans"] += 1
            cols = indices[p0:p1]
            coef = data[p0:p1]
            # min contribution of each term over the box
            contrib = np.where(coef > 0, coef * lo[cols], coef * hi[cols])
            finite = np.isfinite(contrib)
            n_inf = int((~finite).sum())
            s_fin = float(contrib[finite].sum()) if finite.any() else 0.0
            rhs = float(b[r])
            for t in range(p1 - p0):
                a_k = float(coef[t])
                if a_k == 0.0:
                    continue
                k = int(cols[t])
                if k < n_orig and not tighten_original:
                    continue
                # residual = min over the box of the OTHER terms
                if finite[t]:
                    rest_inf = n_inf
                    rest = s_fin - float(contrib[t])
                else:
                    rest_inf = n_inf - 1
                    rest = s_fin
                if rest_inf > 0:
                    continue
                slack = rhs - rest
                if not math.isfinite(slack):
                    continue
                if a_k > 0:
                    new_hi = slack / a_k
                    if new_hi < hi[k] - 1e-9 * (1.0 + abs(float(hi[k]))):
                        hi[k] = new_hi
                        st["tightenings"] += 1
                        changed = True
                else:
                    new_lo = slack / a_k
                    if new_lo > lo[k] + 1e-9 * (1.0 + abs(float(lo[k]))):
                        lo[k] = new_lo
                        st["tightenings"] += 1
                        changed = True
                if lo[k] > hi[k] + 1e-7 * (1.0 + abs(float(hi[k]))):
                    st["empty"] = True
                    return lo, hi, st
        if not changed:
            break
    return lo, hi, st


# --------------------------------------------------------------------------- #
# Child                                                                       #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"

    import inspect  # noqa: PLC0415

    import discopt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import scipy.sparse as sp  # noqa: PLC0415
    from discopt._jax import spatial_producer  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415

    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "budget": float(budget),
        "instrument_marker": None,
        "counts": {},
    }
    src = inspect.getsource(spatial_producer)
    # §8: the bypass shim below replaces ``np.isinf`` in the producer's namespace.
    # It is faithful ONLY if the producer calls ``np.isinf`` exactly once (the
    # ``infinite_aux_bounds`` gate).  Assert it, do not assume it.
    n_isinf = src.count("np.isinf(")
    out["producer_np_isinf_call_sites"] = n_isinf
    if not hasattr(spatial_producer, "producer_stats") or n_isinf != 1:
        out["status"] = "instrument_missing"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 3
    out["instrument_marker"] = _INSTRUMENT_MARKER

    _orig = spatial_producer.build_spatial_kernel_spec

    class _NoIsinf:
        """``numpy`` proxy whose ``isinf`` is identically False.

        Scoped to the single producer call site asserted above; every other numpy
        attribute forwards untouched.
        """

        def __getattr__(self, name):
            if name == "isinf":
                return lambda a: np.zeros(np.shape(a), dtype=bool)
            return getattr(np, name)

    def _diagnose(model, bounds) -> dict:
        """The whole experiment, run once on the live (model, presolved box)."""
        from discopt._jax.uniform_relax import build_uniform_relaxation  # noqa: PLC0415

        counts = out["counts"]
        d: dict = {}
        lb = np.asarray(bounds[0], dtype=np.float64)
        ub = np.asarray(bounds[1], dtype=np.float64)
        n_orig = len(model._variables)
        d["n_orig"] = n_orig
        d["presolved_box_infinite_cols"] = int((~np.isfinite(lb)).sum() + (~np.isfinite(ub)).sum())

        # --- arm 0: reproduce the census decline -------------------------------
        spatial_producer.reset_producer_stats()
        spec0 = _orig(model, bounds=bounds)
        st0 = spatial_producer.producer_stats()
        counts["reproduce_calls"] = counts.get("reproduce_calls", 0) + 1
        d["decline_reproduced"] = st0["last"]
        d["spec_without_bypass"] = spec0 is not None

        # --- the real-box relaxation, exactly as the producer builds it ---------
        rel = build_uniform_relaxation(
            model, box=(lb, ub), skip_separable_floor=True, skip_convex_lift=True
        )
        milp = rel.model
        bnds = np.asarray(milp._bounds, dtype=np.float64)
        ncol = int(bnds.shape[0])
        c = np.asarray(milp._c, dtype=np.float64).ravel()
        d["n_cols"] = ncol
        d["n_aux"] = ncol - n_orig
        counts["relaxations_built"] = counts.get("relaxations_built", 0) + 1

        col_lo, col_hi = bnds[:, 0].copy(), bnds[:, 1].copy()
        inf_lo = ~np.isfinite(col_lo)
        inf_hi = ~np.isfinite(col_hi)
        inf_cols = np.flatnonzero(inf_lo | inf_hi)
        d["infinite_entries"] = int(inf_lo.sum() + inf_hi.sum())
        d["infinite_cols"] = int(inf_cols.size)
        d["infinite_cols_original"] = int((inf_cols < n_orig).sum())
        d["infinite_cols_aux"] = int((inf_cols >= n_orig).sum())
        counts["columns_examined"] = counts.get("columns_examined", 0) + ncol
        counts["infinite_columns_examined"] = counts.get("infinite_columns_examined", 0) + int(
            inf_cols.size
        )

        # which lifted family owns each infinite aux column
        owner: dict[int, str] = {}
        for w, *_r in rel.bilinear_linform_specs:
            owner.setdefault(int(w), "blf")
        for _k, w in rel.monomial_map.items():
            owner.setdefault(int(w), "monomial")
        for _k, w in rel.univariate_square_map.items():
            owner.setdefault(int(w), "univariate_square")
        for (_j, w), _v in rel.affine_square_map.items():
            owner.setdefault(int(w), "affine_square")
        for fname, w, *_r in rel.univariate_atom_specs:
            owner.setdefault(int(w), f"univariate:{fname}")
        for _k, w in rel.trilinear_map.items():
            owner.setdefault(int(w), "trilinear")
        for _k, w in rel.multilinear_map.items():
            owner.setdefault(int(w), "multilinear")
        for _k, w in rel.ratio_map.items():
            owner.setdefault(int(w), "ratio")

        a_ub = sp.csr_matrix(milp._A_ub, dtype=np.float64) if milp._A_ub is not None else None
        b = np.asarray(milp._b_ub, dtype=np.float64).ravel() if a_ub is not None else None
        d["n_rows"] = 0 if a_ub is None else int(a_ub.shape[0])
        colnnz = np.diff(a_ub.tocsc().indptr) if a_ub is not None else np.zeros(ncol, dtype=int)

        detail = []
        for k in inf_cols[:40]:
            k = int(k)
            detail.append(
                {
                    "col": k,
                    "kind": "original" if k < n_orig else "aux",
                    "owner": owner.get(k, "unclaimed_intermediate"),
                    "lo": None if not np.isfinite(col_lo[k]) else float(col_lo[k]),
                    "hi": None if not np.isfinite(col_hi[k]) else float(col_hi[k]),
                    "row_nnz": int(colnnz[k]),
                    "obj_coeff": float(c[k]),
                }
            )
        d["infinite_col_detail"] = detail
        d["infinite_cols_with_obj_cost"] = int(
            sum(1 for k in inf_cols if abs(float(c[int(k)])) > 0.0)
        )
        d["infinite_cols_in_no_row"] = int(sum(1 for k in inf_cols if int(colnnz[int(k)]) == 0))
        by_owner: dict[str, int] = {}
        for k in inf_cols:
            key = "original" if int(k) < n_orig else owner.get(int(k), "unclaimed_intermediate")
            by_owner[key] = by_owner.get(key, 0) + 1
        d["infinite_cols_by_owner"] = by_owner

        # --- arm E8.2: does FBBT on the relaxation's own rows finitize them? ----
        if a_ub is not None:
            f_lo, f_hi, fst = fbbt_columns(a_ub, b, col_lo, col_hi, n_orig)
            counts["fbbt_row_scans"] = counts.get("fbbt_row_scans", 0) + int(fst["row_scans"])
            counts["fbbt_tightenings"] = counts.get("fbbt_tightenings", 0) + int(fst["tightenings"])
            rem_lo = ~np.isfinite(f_lo)
            rem_hi = ~np.isfinite(f_hi)
            d["fbbt"] = {
                **fst,
                "infinite_entries_after": int(rem_lo.sum() + rem_hi.sum()),
                "infinite_cols_after": int(np.flatnonzero(rem_lo | rem_hi).size),
                "aux_infinite_cols_after": int(
                    sum(1 for k in np.flatnonzero(rem_lo | rem_hi) if int(k) >= n_orig)
                ),
                "all_finite": bool(not (rem_lo.any() or rem_hi.any())),
                "aux_all_finite": bool(
                    not any(int(k) >= n_orig for k in np.flatnonzero(rem_lo | rem_hi))
                ),
                # The producer's gate tests EVERY column, so this — not
                # ``aux_all_finite`` — is what decides whether a propagation repair
                # would let the instance past it.
                "gate_would_pass": bool(not (rem_lo.any() or rem_hi.any())),
            }
            surv = [
                {
                    "col": int(k),
                    "kind": "original" if int(k) < n_orig else "aux",
                    "owner": owner.get(int(k), "unclaimed_intermediate")
                    if int(k) >= n_orig
                    else "original",
                    "row_nnz": int(colnnz[int(k)]),
                    "obj_coeff": float(c[int(k)]),
                }
                for k in np.flatnonzero(rem_lo | rem_hi)[:40]
            ]
            d["fbbt_survivor_detail"] = surv
        else:
            d["fbbt"] = None

        # --- arm E8.1: bypass the gate, what does the ladder say next? ----------
        spatial_producer.reset_producer_stats()
        _saved_np = spatial_producer.np
        spatial_producer.np = _NoIsinf()
        try:
            spec1 = _orig(model, bounds=bounds)
        finally:
            spatial_producer.np = _saved_np
        st1 = spatial_producer.producer_stats()
        counts["bypass_calls"] = counts.get("bypass_calls", 0) + 1
        d["bypass_spec_built"] = spec1 is not None
        d["bypass_next_decline"] = st1["last"]
        d["bypass_next_detail"] = st1["last_detail"]
        d["bypass_reasons"] = st1["reasons"]
        if spec1 is not None:
            d["bypass_spec_terms"] = int(np.size(spec1["term_kind"]) + np.size(spec1["blf_w"]))
            gl = np.asarray(spec1["global_lo"], dtype=np.float64)
            gh = np.asarray(spec1["global_hi"], dtype=np.float64)
            d["bypass_spec_infinite_entries"] = int(
                (~np.isfinite(gl)).sum() + (~np.isfinite(gh)).sum()
            )
        return d

    def _tapped(model, bounds=None):
        # Deliberately NOT wrapped in try/except (§7): a broken probe must crash.
        out["wall_to_gate"] = time.perf_counter() - t0
        out["diagnosis"] = _diagnose(model, bounds)
        out["phase"] = "diagnosed"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        sys.stdout.flush()
        os._exit(0)

    spatial_producer.build_spatial_kernel_spec = _tapped

    nl = str(instance_path(instance))
    t0 = time.perf_counter()
    model = from_nl(nl)
    out["n_vars"] = len(model._variables)
    out["n_cons"] = len(model._constraints)
    r = model.solve(time_limit=budget)
    # Reaching here means the producer was never called — the kernel gate did not
    # fire, which is itself a finding (and never silently a pass).
    out["phase"] = "producer_never_called"
    out["status"] = str(r.status)
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _census_infinite_aux_instances() -> list[str]:
    data = json.loads(_CENSUS.read_text())
    got = []
    for r in data["rows"]:
        p = r.get("producer") or {}
        if p.get("last") == "infinite_aux_bounds":
            got.append(r["instance"])
    return sorted(got)


def _baseline_walls() -> dict:
    if not _BASELINE.exists():
        return {}
    return {r["instance"]: r for r in json.loads(_BASELINE.read_text()).get("rows", [])}


def _run_one(instance: str, budget: float) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance, str(budget)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget + _CHILD_TIMEOUT_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "phase": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "phase": "child_crashed",
        "stderr_tail": proc.stderr[-2000:],
    }


def _e81_verdict(n: int) -> str:
    if n >= 6:
        return "SUPPORTED"
    return "KILLED" if n <= 4 else "INCONCLUSIVE (5)"


def _render(report: dict) -> str:
    rows = report["rows"]
    base = _baseline_walls()
    out_lines: list[str] = ["", "=" * 100]
    out_lines.append("PHASE 5.2 ENTRY EXPERIMENT — census rank #2 `infinite_aux_bounds`")
    out_lines.append("=" * 100)
    out_lines.append(
        f"tree {report['git_sha']}{'-dirty' if report['git_dirty'] else ''}  "
        f"budget {report['budget']}s  instances {len(rows)}  "
        f"wall {report['total_wall_seconds']:.1f}s  "
        f"load {report['load_start']:.2f}->{report['load_peak']:.2f}"
    )
    out_lines.append("")
    out_lines.append(
        "--- E8.1 FRAMING: bypass the gate, where does the ladder stop next? " + "-" * 30
    )
    out_lines.append(
        f"{'instance':<18}{'reproduced':<22}{'next decline':<32}{'spec?':<7}{'wall(s)':>9}"
    )
    reach = []
    for r in rows:
        d = r.get("diagnosis") or {}
        nxt = d.get("bypass_next_decline") or "-"
        det = d.get("bypass_next_detail") or ""
        if det:
            nxt = f"{nxt}:{det}"
        built = bool(d.get("bypass_spec_built"))
        if built:
            nxt = "NONE (spec built)"
            reach.append(r["instance"])
        w = base.get(r["instance"], {}).get("wall")
        out_lines.append(
            f"{r['instance']:<18}{str(d.get('decline_reproduced')):<22}{nxt:<32}"
            f"{('yes' if built else 'no'):<7}{(float(w) if w else 0.0):>9.1f}"
        )
    tot_base = sum(float(v.get("wall") or 0.0) for v in base.values()) or 1.0
    reach_wall = sum(float(base.get(i, {}).get("wall") or 0.0) for i in reach)
    grp_wall = sum(float(base.get(r["instance"], {}).get("wall") or 0.0) for r in rows)
    out_lines.append("")
    out_lines.append(
        f"  reach row-claiming with the gate bypassed: {len(reach)} / {len(rows)}  "
        f"({', '.join(reach) if reach else 'none'})"
    )
    out_lines.append(
        f"  wall of the reachable set: {reach_wall:.1f}s of {tot_base:.1f}s corpus "
        f"= {100 * reach_wall / tot_base:.1f}%   (whole group {grp_wall:.1f}s = "
        f"{100 * grp_wall / tot_base:.1f}%)"
    )
    out_lines.append("")
    out_lines.append("--- E8.2 REPAIRABILITY: FBBT on the relaxation's own rows " + "-" * 40)
    out_lines.append(
        f"{'instance':<18}{'cols':>6}{'inf cols':>10}{'orig/aux':>10}{'no-row':>8}"
        f"{'cost':>6}{'box inf':>9}{'after FBBT':>12}{'gate ok':>9}"
    )
    repaired = []
    for r in rows:
        d = r.get("diagnosis") or {}
        f = d.get("fbbt") or {}
        if f.get("gate_would_pass"):
            repaired.append(r["instance"])
        oa = f"{d.get('infinite_cols_original', 0)}/{d.get('infinite_cols_aux', 0)}"
        out_lines.append(
            f"{r['instance']:<18}{d.get('n_cols', 0):>6}{d.get('infinite_cols', 0):>10}"
            f"{oa:>10}"
            f"{d.get('infinite_cols_in_no_row', 0):>8}{d.get('infinite_cols_with_obj_cost', 0):>6}"
            f"{d.get('presolved_box_infinite_cols', 0):>9}"
            f"{f.get('infinite_cols_after', '-'):>12}{str(f.get('gate_would_pass')):>9}"
        )
    out_lines.append("")
    out_lines.append(
        f"  gate would pass after FBBT repair: {len(repaired)} / {len(rows)}  "
        f"({', '.join(repaired) if repaired else 'none'})"
    )
    out_lines.append(
        "  ('box inf' = infinite entries in the PRESOLVED ROOT BOX handed to the producer — "
        "an original variable the solver could not bound; no relaxation-side repair reaches it)"
    )
    out_lines.append("")
    out_lines.append("--- OWNERS OF THE INFINITE COLUMNS " + "-" * 62)
    for r in rows:
        d = r.get("diagnosis") or {}
        out_lines.append(f"  {r['instance']:<18}{json.dumps(d.get('infinite_cols_by_owner', {}))}")
    out_lines.append("")
    out_lines.append("--- EXECUTED PROBES " + "-" * 76)
    ex = report["executed"]
    for k in sorted(ex):
        out_lines.append(f"  {k:<32}{ex[k]}")
    out_lines.append("")
    out_lines.append("--- VERDICT vs the pre-stated kill criteria " + "-" * 54)
    out_lines.append(
        f"  E8.1 hypothesis: >=6 of {len(rows)} reach row-claiming.  measured "
        f"{len(reach)}  -> {_e81_verdict(len(reach))}"
    )
    n_surv = len(rows) - len(repaired)
    out_lines.append(
        f"  E8.2 hypothesis: infinities repairable by FBBT on the relaxation's own rows; "
        f"KILL if >=5 of {len(rows)} survive.  survived {n_surv}  -> "
        f"{'KILLED' if n_surv >= 5 else 'SUPPORTED'}"
    )
    both = [i for i in reach if i in repaired]
    out_lines.append(
        f"  JOINT (the only set a producer-side card could actually serve): "
        f"{len(both)}  ({', '.join(both) if both else 'none'})"
    )
    return "\n".join(out_lines)


def cmd_run(args: argparse.Namespace) -> int:
    instances = _census_infinite_aux_instances()
    if args.subset:
        want = {s.strip() for s in args.subset.split(",") if s.strip()}
        instances = [i for i in instances if i in want]
    if not instances:
        print("FAIL: no instances resolved", file=sys.stderr)
        return 2
    load_start = _load1()
    load_peak = load_start
    t0 = time.perf_counter()
    rows = []
    for i, inst in enumerate(instances, 1):
        row = _run_one(inst, args.budget)
        rows.append(row)
        load_peak = max(load_peak, _load1())
        d = row.get("diagnosis") or {}
        print(
            f"[{i:2d}/{len(instances)}] {inst:<20} phase={row.get('phase', '-'):<22} "
            f"next={d.get('bypass_next_decline')} spec={d.get('bypass_spec_built')} "
            f"fbbt_aux_finite={(d.get('fbbt') or {}).get('aux_all_finite')}",
            flush=True,
        )
        if row.get("phase") == "child_crashed":
            print(row.get("stderr_tail", ""), file=sys.stderr, flush=True)

    ex: dict = {"instances": len(rows)}
    for r in rows:
        for k, v in (r.get("counts") or {}).items():
            ex[k] = ex.get(k, 0) + int(v)
    ex["diagnosed"] = sum(1 for r in rows if r.get("phase") == "diagnosed")
    ex["crashed"] = sum(1 for r in rows if r.get("phase") in ("child_crashed", "child_timeout"))
    report = {
        "schema": "phase52_infinite_aux_entry/1",
        "git_sha": _short_sha(),
        "git_dirty": _git_dirty(),
        "budget": float(args.budget),
        "instrument_marker": _INSTRUMENT_MARKER,
        "census": str(_CENSUS.relative_to(_REPO_ROOT)),
        "baseline": str(_BASELINE.relative_to(_REPO_ROOT)),
        "total_wall_seconds": time.perf_counter() - t0,
        "load_start": load_start,
        "load_peak": load_peak,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "executed": ex,
        "rows": rows,
    }
    _REPORTS_DIR.mkdir(exist_ok=True)
    out_path = _REPORTS_DIR / f"phase52_infinite_aux_entry_{report['git_sha']}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(_render(report))
    print(f"\nartifact: {out_path.relative_to(_REPO_ROOT)}")
    # §6: a probe that examined nothing must FAIL, not read as a pass.
    if ex["diagnosed"] == 0 or ex.get("bypass_calls", 0) == 0 or ex.get("fbbt_row_scans", 0) == 0:
        print("\nFAIL: zero executed probes — the experiment measured nothing", file=sys.stderr)
        return 2
    if ex["crashed"]:
        print(f"\nFAIL: {ex['crashed']} child(ren) crashed", file=sys.stderr)
        return 2
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print(_render(json.loads(Path(args.report).read_text())))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", nargs=2, metavar=("INSTANCE", "BUDGET"))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--budget", type=float, default=_DEFAULT_BUDGET)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    if args.solve:
        return _run_child(args.solve[0], float(args.solve[1]))
    if args.report:
        return cmd_report(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
