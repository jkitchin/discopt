"""Phase 5.2 entry experiment — the UNMASKABLE tail: census ranks #3 and #5.

The item-8 entry experiment (``phase52_infinite_aux_entry.py``) killed census rank #2
and, filing its result, re-derived the census ranking by **ladder position**: a
decline code's attributed wall is an upper bound exactly to the extent the code is
tested early.  Ranks #3 (``probe_real_shape_mismatch``) and #5
(``fixed_row_box_dependent``) are the producer's **last two tests**, so every instance
in them has already passed every other decline test and their attributed wall is
recoverable wall by construction — 12 instances, 268.6 s, 13.9 % of corpus wall.

Both are the *same shaped failure*.  The producer identifies structure on a probe box
and validates it against the real box: every row it did not claim as a term envelope
must be **box-independent**, else freezing it as a fixed row is unsound.  Rank #3 is
"the two builds do not even have the same shape"; rank #5 is "same shape, but an
unclaimed row's coefficients move with the box".

E8.3 — WHERE DO THE BOX-DEPENDENT ROWS COME FROM?
    Hypothesis: the offending rows are envelope rows of terms the relaxer **does**
    register (in ``bilinear_linform_specs`` / ``monomial_map`` /
    ``univariate_square_map`` / ``affine_square_map`` / ``univariate_atom_specs``),
    which the producer failed to *claim* because of its support/row-count predicate —
    so the fix is a claiming-predicate change inside the producer.
    KILL: if on **>= 6 of the 12** instances the offending rows touch aux columns
    registered in **no** map (genuinely unregistered factorable intermediates), the
    fix is a new registration/envelope family in the relaxer — a different and much
    larger card — and item 8 stops with the measurement rather than starting it.

Method notes (CLAUDE.md §6-§10)
-------------------------------

* Producer tapped at its **real call site** (the presolved root box), one subprocess
  per instance, ``DISCOPT_NATIVE_SPATIAL_KERNEL=1`` — same as the 5.1 census.
* The producer's own state is read, never re-implemented: ``_decline`` is wrapped so
  that on the decline of interest it captures the **caller frame's locals**
  (``env_rows``, the probe/real matrices, the claim bookkeeping).  Re-deriving row
  ownership in the probe would measure the probe, not the producer.
* No swallowed exceptions (§7); executed counts printed and a zero count exits
  non-zero (§6); the loaded module and an instrument marker are asserted (§8).
"""

from __future__ import annotations

import argparse
import json
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

_INSTRUMENT_MARKER = "phase52-boxdep-rows-entry-v1"
_BASELINE = _REPO_ROOT / "reports" / "panel_baseline_f154dcff.json"
_CENSUS = _REPO_ROOT / "reports" / "phase5_kernel_coverage_census_c346fd73.json"
_REPORTS_DIR = _REPO_ROOT / "reports"
_DEFAULT_BUDGET = 45.0
_CHILD_TIMEOUT_SLACK = 180.0

_CODES = ("probe_real_shape_mismatch", "fixed_row_box_dependent")


# --------------------------------------------------------------------------- #
# Child                                                                       #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"

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
    if not hasattr(spatial_producer, "producer_stats"):
        out["status"] = "instrument_missing"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 3
    out["instrument_marker"] = _INSTRUMENT_MARKER

    _orig_producer = spatial_producer.build_spatial_kernel_spec
    _orig_decline = spatial_producer._decline
    captured: dict = {}

    def _capturing_decline(code: str, detail: str = ""):
        # Read the PRODUCER's own state at the moment it gives up — its claim
        # bookkeeping, its two builds, its env_rows.  Re-deriving these in the probe
        # would measure the probe (CLAUDE.md §6: the instrument must observe the
        # thing, not a look-alike).
        if code in _CODES and "frame" not in captured:
            captured["code"] = code
            captured["detail"] = detail
            captured["frame"] = dict(sys._getframe(1).f_locals)
        return _orig_decline(code, detail)

    def _classify_cols(cols, n_orig, owner):
        seen = {"original": 0, "unregistered_aux": 0}
        for c in cols:
            c = int(c)
            if c < n_orig:
                seen["original"] = seen.get("original", 0) + 1
            else:
                k = owner.get(c, "unregistered_aux")
                seen[k] = seen.get(k, 0) + 1
        return seen

    def _diagnose(model, bounds) -> dict:
        counts = out["counts"]
        d: dict = {}
        spatial_producer.reset_producer_stats()
        captured.clear()
        spec = _orig_producer(model, bounds=bounds)
        st = spatial_producer.producer_stats()
        counts["producer_calls"] = counts.get("producer_calls", 0) + 1
        d["decline"] = st["last"]
        d["decline_detail"] = st["last_detail"]
        d["spec_built"] = spec is not None
        if "frame" not in captured:
            d["captured"] = False
            return d
        d["captured"] = True
        f = captured["frame"]
        rel = f["rel"]
        n_orig = int(f["n_orig"])
        d["code"] = captured["code"]
        d["detail"] = captured["detail"]

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
        for _spec in rel.composite_multivar_specs:
            w = _spec[0] if isinstance(_spec, (list, tuple)) else None
            if w is not None:
                owner.setdefault(int(w), "composite_multivar")
        d["registered_aux_columns"] = len(owner)
        # ``coverage`` names the relaxation KIND the relaxer applied to each node it
        # enveloped, independently of whether that lift was registered in a
        # structural map.  Comparing this histogram against ``term_family_sizes``
        # names the families whose rows exist but whose columns the producer cannot
        # see — which is the actionable output of this experiment.
        cov: dict[str, int] = {}
        for _kind, _tight in (getattr(rel, "coverage", {}) or {}).values():
            cov[str(_kind)] = cov.get(str(_kind), 0) + 1
        d["coverage_kinds"] = cov
        d["term_family_sizes"] = {
            "blf": len(rel.bilinear_linform_specs),
            "monomial": len(rel.monomial_map),
            "univariate_square": len(rel.univariate_square_map),
            "affine_square": len(rel.affine_square_map),
            "univariate_atom": len(rel.univariate_atom_specs),
            "trilinear": len(rel.trilinear_map),
            "multilinear": len(rel.multilinear_map),
            "ratio": len(rel.ratio_map),
            "composite_multivar": len(rel.composite_multivar_specs),
        }

        a_probe = sp.csr_matrix(f["milp"]._A_ub, dtype=np.float64)
        a_probe.sort_indices()
        b_probe = np.asarray(f["milp"]._b_ub, dtype=np.float64).ravel()
        rel_real = f["rel_real"]
        a_real = sp.csr_matrix(rel_real.model._A_ub, dtype=np.float64)
        a_real.sort_indices()
        b_real = np.asarray(rel_real.model._b_ub, dtype=np.float64).ravel()
        d["probe_shape"] = list(a_probe.shape)
        d["real_shape"] = list(a_real.shape)
        d["probe_nnz"] = int(a_probe.nnz)
        d["real_nnz"] = int(a_real.nnz)
        d["n_orig"] = n_orig

        if captured["code"] == "probe_real_shape_mismatch":
            # Which family emitted a different number of rows?  The two builds run the
            # same code on different boxes, so a row-count delta localizes to whichever
            # envelope's row count is box-conditional.
            d["real_term_family_sizes"] = {
                "blf": len(rel_real.bilinear_linform_specs),
                "monomial": len(rel_real.monomial_map),
                "univariate_square": len(rel_real.univariate_square_map),
                "affine_square": len(rel_real.affine_square_map),
                "univariate_atom": len(rel_real.univariate_atom_specs),
                "trilinear": len(rel_real.trilinear_map),
                "multilinear": len(rel_real.multilinear_map),
                "ratio": len(rel_real.ratio_map),
                "composite_multivar": len(rel_real.composite_multivar_specs),
            }
            d["row_delta"] = int(a_real.shape[0]) - int(a_probe.shape[0])
            d["col_delta"] = int(a_real.shape[1]) - int(a_probe.shape[1])
            d["coverage_probe"] = len(getattr(rel, "coverage", {}) or {})
            d["coverage_real"] = len(getattr(rel_real, "coverage", {}) or {})
            counts["shape_rows_compared"] = counts.get("shape_rows_compared", 0) + 1
            return d

        # fixed_row_box_dependent — the producer returns on the FIRST offender; walk
        # every unclaimed row so the class, not the instance, is characterized.
        env_rows = {int(x) for x in f["env_rows"]}
        d["env_rows"] = len(env_rows)
        d["claimed_aux"] = len(f.get("claimed_aux") or ())
        pi, pj, pv = a_probe.indptr, a_probe.indices, a_probe.data
        ri, rj, rv = a_real.indptr, a_real.indices, a_real.data
        offenders = []
        n_checked = 0
        kinds: dict[str, int] = {}
        col_class: dict[str, int] = {}
        for r in range(a_probe.shape[0]):
            if r in env_rows:
                continue
            n_checked += 1
            p0, p1 = int(pi[r]), int(pi[r + 1])
            q0, q1 = int(ri[r]), int(ri[r + 1])
            kind = None
            if (p1 - p0) != (q1 - q0):
                kind = "nnz"
            elif not np.array_equal(pj[p0:p1], rj[q0:q1]):
                kind = "pattern"
            elif not np.allclose(pv[p0:p1], rv[q0:q1], rtol=1e-9, atol=1e-9):
                kind = "coeffs"
            elif abs(float(b_probe[r]) - float(b_real[r])) > 1e-6 * (1.0 + abs(float(b_real[r]))):
                kind = "rhs"
            if kind is None:
                continue
            kinds[kind] = kinds.get(kind, 0) + 1
            cols = [int(x) for x in pj[p0:p1]]
            cls = _classify_cols(cols, n_orig, owner)
            for k, v in cls.items():
                col_class[k] = col_class.get(k, 0) + v
            aux_cols = [c for c in cols if c >= n_orig]
            offenders.append(
                {
                    "row": r,
                    "kind": kind,
                    "support": cols[:12],
                    "aux_owners": sorted({owner.get(c, "unregistered_aux") for c in aux_cols}),
                    "n_aux": len(aux_cols),
                    "probe_coeffs": [round(float(x), 6) for x in pv[p0:p1]][:12],
                    "real_coeffs": [round(float(x), 6) for x in rv[q0:q1]][:12],
                    "probe_rhs": float(b_probe[r]),
                    "real_rhs": float(b_real[r]),
                }
            )
        counts["unclaimed_rows_checked"] = counts.get("unclaimed_rows_checked", 0) + n_checked
        counts["offending_rows_found"] = counts.get("offending_rows_found", 0) + len(offenders)
        d["unclaimed_rows_checked"] = n_checked
        d["offending_rows"] = len(offenders)
        d["offender_kinds"] = kinds
        d["offender_column_classes"] = col_class
        owners_hist: dict[str, int] = {}
        for o in offenders:
            key = ",".join(o["aux_owners"]) if o["aux_owners"] else "NO_AUX"
            owners_hist[key] = owners_hist.get(key, 0) + 1
        d["offender_aux_owner_hist"] = owners_hist
        # The kill criterion reads this: are the offending rows over aux columns the
        # relaxer registered (claiming-predicate fix) or over unregistered ones
        # (a new envelope family — a different card)?
        d["all_offender_aux_registered"] = bool(
            offenders and all("unregistered_aux" not in o["aux_owners"] for o in offenders)
        )
        d["any_offender_aux_unregistered"] = bool(
            any("unregistered_aux" in o["aux_owners"] for o in offenders)
        )
        d["offender_sample"] = offenders[:8]
        return d

    def _tapped(model, bounds=None):
        out["wall_to_gate"] = time.perf_counter() - t0
        out["diagnosis"] = _diagnose(model, bounds)
        out["phase"] = "diagnosed"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        sys.stdout.flush()
        os._exit(0)

    spatial_producer._decline = _capturing_decline
    spatial_producer.build_spatial_kernel_spec = _tapped

    nl = str(instance_path(instance))
    t0 = time.perf_counter()
    model = from_nl(nl)
    out["n_vars"] = len(model._variables)
    out["n_cons"] = len(model._constraints)
    r = model.solve(time_limit=budget)
    out["phase"] = "producer_never_called"
    out["status"] = str(r.status)
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _census_instances() -> list[str]:
    data = json.loads(_CENSUS.read_text())
    got = []
    for r in data["rows"]:
        last = (r.get("producer") or {}).get("last")
        if last in _CODES:
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
    return {"instance": instance, "phase": "child_crashed", "stderr_tail": proc.stderr[-2000:]}


def _render(report: dict) -> str:
    rows = report["rows"]
    base = _baseline_walls()
    ln: list[str] = ["", "=" * 100]
    ln.append("PHASE 5.2 ENTRY EXPERIMENT — the unmaskable tail (census ranks #3 and #5)")
    ln.append("=" * 100)
    ln.append(
        f"tree {report['git_sha']}{'-dirty' if report['git_dirty'] else ''}  "
        f"budget {report['budget']}s  instances {len(rows)}  "
        f"wall {report['total_wall_seconds']:.1f}s  "
        f"load {report['load_start']:.2f}->{report['load_peak']:.2f}"
    )
    ln.append("")
    ln.append("--- RANK #5 `fixed_row_box_dependent` — what are the offending rows? " + "-" * 30)
    ln.append(
        f"{'instance':<20}{'kind':<10}{'unclaimed':>10}{'offend':>8}{'env rows':>10}"
        f"{'aux registered?':>18}{'wall(s)':>9}"
    )
    registered, unregistered = [], []
    for r in rows:
        d = r.get("diagnosis") or {}
        if d.get("code") != "fixed_row_box_dependent":
            continue
        allreg = d.get("all_offender_aux_registered")
        anyun = d.get("any_offender_aux_unregistered")
        (unregistered if anyun else registered).append(r["instance"])
        w = base.get(r["instance"], {}).get("wall") or 0.0
        ln.append(
            f"{r['instance']:<20}{str(d.get('detail')):<10}{d.get('unclaimed_rows_checked', 0):>10}"
            f"{d.get('offending_rows', 0):>8}{d.get('env_rows', 0):>10}"
            f"{('ALL' if allreg else ('some UNREG' if anyun else '-')):>18}{float(w):>9.1f}"
        )
    ln.append("")
    ln.append("  offender aux-owner histogram per instance:")
    for r in rows:
        d = r.get("diagnosis") or {}
        if d.get("code") != "fixed_row_box_dependent":
            continue
        ln.append(f"    {r['instance']:<20}{json.dumps(d.get('offender_aux_owner_hist', {}))}")
    ln.append("")
    ln.append("--- RANK #3 `probe_real_shape_mismatch` — where do the shapes diverge? " + "-" * 27)
    ln.append(f"{'instance':<20}{'probe A':>16}{'real A':>16}{'drow':>7}{'dcol':>7}{'wall(s)':>9}")
    for r in rows:
        d = r.get("diagnosis") or {}
        if d.get("code") != "probe_real_shape_mismatch":
            continue
        w = base.get(r["instance"], {}).get("wall") or 0.0
        ln.append(
            f"{r['instance']:<20}{str(d.get('probe_shape')):>16}{str(d.get('real_shape')):>16}"
            f"{d.get('row_delta', 0):>7}{d.get('col_delta', 0):>7}{float(w):>9.1f}"
        )
    ln.append("")
    for r in rows:
        d = r.get("diagnosis") or {}
        if d.get("code") != "probe_real_shape_mismatch":
            continue
        ln.append(f"    {r['instance']} probe families: {json.dumps(d.get('term_family_sizes'))}")
        ln.append(
            f"    {r['instance']} real  families: {json.dumps(d.get('real_term_family_sizes'))}"
        )
    ln.append("")
    ln.append("--- RELAXATION KINDS APPLIED vs STRUCTURAL FAMILIES REGISTERED " + "-" * 34)
    ln.append("  (a kind present here with no matching registered family is a lift whose")
    ln.append("   box-dependent rows exist but whose column the producer cannot see)")
    for r in rows:
        d = r.get("diagnosis") or {}
        if not d.get("captured"):
            continue
        fam = {k: v for k, v in (d.get("term_family_sizes") or {}).items() if v}
        ln.append(f"    {r['instance']:<20}kinds={json.dumps(d.get('coverage_kinds', {}))}")
        ln.append(f"    {'':<20}registered={json.dumps(fam)}")
    ln.append("")
    ln.append("--- EXECUTED PROBES " + "-" * 76)
    ex = report["executed"]
    for k in sorted(ex):
        ln.append(f"  {k:<32}{ex[k]}")
    ln.append("")
    ln.append("--- VERDICT vs the pre-stated kill criterion " + "-" * 53)
    n_un = len(unregistered)
    ln.append(
        f"  E8.3: KILL if >= 6 of {len(rows)} have offending rows over UNREGISTERED aux "
        f"columns.  measured {n_un}  -> {'KILLED' if n_un >= 6 else 'SUPPORTED'}"
    )
    ln.append(f"     all-registered: {', '.join(registered) if registered else 'none'}")
    ln.append(f"     some-unregistered: {', '.join(unregistered) if unregistered else 'none'}")
    return "\n".join(ln)


def cmd_run(args: argparse.Namespace) -> int:
    instances = _census_instances()
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
            f"code={d.get('code')} offenders={d.get('offending_rows')} "
            f"unreg={d.get('any_offender_aux_unregistered')}",
            flush=True,
        )
        if row.get("phase") == "child_crashed":
            print(row.get("stderr_tail", ""), file=sys.stderr, flush=True)

    ex: dict = {"instances": len(rows)}
    for r in rows:
        for k, v in (r.get("counts") or {}).items():
            ex[k] = ex.get(k, 0) + int(v)
    ex["diagnosed"] = sum(1 for r in rows if r.get("phase") == "diagnosed")
    ex["captured"] = sum(1 for r in rows if (r.get("diagnosis") or {}).get("captured"))
    ex["crashed"] = sum(1 for r in rows if r.get("phase") in ("child_crashed", "child_timeout"))
    report = {
        "schema": "phase52_boxdep_rows_entry/1",
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
    out_path = _REPORTS_DIR / f"phase52_boxdep_rows_entry_{report['git_sha']}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(_render(report))
    print(f"\nartifact: {out_path.relative_to(_REPO_ROOT)}")
    if ex["diagnosed"] == 0 or ex["captured"] == 0 or ex.get("producer_calls", 0) == 0:
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
