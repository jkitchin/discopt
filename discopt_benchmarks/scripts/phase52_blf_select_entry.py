#!/usr/bin/env python
"""Phase 5.2 entry experiment: does `IncrementalMcCormickLP._select` already solve
`blf_row_count`?

**The claim under test** (consolidation plan, Phase 5 "What 5.2 should take next"):

> the producer declines models whose lifted model constraints share a term's
> support, while `IncrementalMcCormickLP._select` identifies exactly the same
> envelope rows *numerically* on the probe box and has done since #861. Porting that
> matcher is a Python change worth 7 instances.

If true, `blf_row_count` is a producer-side unblock — port an existing matcher, no
new math. If false, the card must not start an envelope build on a false premise.

**Kill criterion.** The claim dies if either:

1. `_select`'s term classes do not include the class `blf_row_count` declines on
   (`rel.bilinear_linform_specs` — a product of two *affine forms*), or
2. the data `spatial_producer` has in hand is insufficient to compute the expected
   closed-form envelope rows that a `_select`-style match needs.

Both are checked by construction, on the real corpus instances, with counts.

**What is measured, per instance** (the 7 `blf_row_count` instances plus a served
control): the producer's own probe build is reproduced, and for every BLF term the
candidate row set (rows touching `w` with support ⊆ operands∪{w}) is dumped with each
row's coefficients and rhs. That says what the extra rows actually *are* — the
premise of any matcher is that they are box-INDEPENDENT lifted model rows.

Prints an executed-assertion count and exits non-zero when it is zero (CLAUDE.md §6).
Exceptions are never swallowed (§7). The loaded module and markers unique to the
version under test are asserted before any measurement (§8).

Usage::

    python -u discopt_benchmarks/scripts/phase52_blf_select_entry.py
"""

from __future__ import annotations

import argparse
import inspect
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

from scripts.panel_baseline import _short_sha, instance_path  # noqa: E402

_REPORTS_DIR = _REPO_ROOT / "reports"
_TOL = 1e-12

#: The 7 `blf_row_count` instances, read off `reports/phase5_kernel_coverage_census_
#: c346fd73.json` (census rank #6). `tanksize` is the served control: it is the
#: instance #764 validated the BLF path on, so its BLF terms must claim exactly 4.
_BLF_INSTANCES = ["st_e01", "st_e05", "st_e08", "st_e09", "st_e11", "st_e40", "syn05hfsg"]
_CONTROL_INSTANCES = ["tanksize"]

_CHILD_TIMEOUT_S = 300


def _check_select_classes() -> dict:
    """Assertion 1 — which term classes does `_select` actually cover?

    Read off the source of `IncrementalMcCormickLP.__init__`, which is where
    `_select` is defined and called. Structural, not a guess: the call sites name
    their dicts.
    """
    from discopt._jax import incremental_mccormick as im

    src = inspect.getsource(im)
    covered = {
        "bilinear": "self.bilin_rows[(i, j, a)] = _select(" in src,
        "monomial": "self.mono_rows[(i, a, p)] = _select(" in src,
        "affine_square": "self.affsq_rows[(j, a, coeff, const)] = _select(" in src,
    }
    return {
        "has_select": "def _select(" in src,
        "covered_classes": covered,
        # The class `blf_row_count` declines on. If this module never references the
        # field, `_select` has never seen the class and there is nothing to port.
        "mentions_bilinear_linform_specs": "bilinear_linform_specs" in src,
        # RETRACTED PREDICATE (CLAUDE.md §11). The first version of this probe also
        # accepted a case-insensitive "linform" substring anywhere in the module and
        # OR-ed it into the verdict. The module's only hit is the word "LinForm"
        # inside a docstring about `_emit_1d` on an affine base — unrelated to
        # `bilinear_linform_specs` — so the probe printed "CLAIM HOLDS" for a claim
        # its own data falsified. Kept as a recorded field, never as a verdict input.
        "loose_linform_substring_hits": src.lower().count("linform"),
        "n_select_call_sites": src.count("= _select("),
    }


def _producer_blf_census(instance: str) -> dict:
    """Reproduce the producer's probe build and dump every BLF term's candidates."""
    import numpy as np
    import scipy.sparse as sp
    from discopt._jax.uniform_relax import build_uniform_relaxation
    from discopt.modeling.core import from_nl

    model = from_nl(str(instance_path(instance)))
    if any(int(getattr(v, "size", 1)) != 1 for v in model._variables):
        return {"skipped": "vector_variables"}
    n_orig = len(model._variables)
    lb = np.array([float(np.min(v.lb)) for v in model._variables], dtype=np.float64)
    ub = np.array([float(np.max(v.ub)) for v in model._variables], dtype=np.float64)

    # The producer's probe box, verbatim (spatial_producer.build_spatial_kernel_spec).
    root_sign = np.where(lb >= 0.0, 1, np.where(ub <= 0.0, -1, 0))
    lb_p = np.empty(n_orig)
    ub_p = np.empty(n_orig)
    for k in range(n_orig):
        if root_sign[k] < 0:
            lb_p[k], ub_p[k] = -(7.0 + k), -1.0
        else:
            lb_p[k], ub_p[k] = 1.0, 7.0 + k

    rel = build_uniform_relaxation(
        model, box=(lb_p, ub_p), skip_separable_floor=True, skip_convex_lift=True
    )
    milp = rel.model
    if milp._A_ub is None:
        return {"skipped": "probe_relaxation_no_rows"}
    A = sp.csr_matrix(milp._A_ub, dtype=np.float64)
    A.sort_indices()
    b = np.asarray(milp._b_ub, dtype=np.float64).ravel()
    csc = A.tocsc()
    indptr, indices, data = A.indptr, A.indices, A.data

    def support(r):
        return {int(indices[t]) for t in range(indptr[r], indptr[r + 1]) if abs(data[t]) > _TOL}

    def entries(r):
        return {int(indices[t]): float(data[t]) for t in range(indptr[r], indptr[r + 1])}

    def rows_with_col(c):
        return csc.indices[csc.indptr[c] : csc.indptr[c + 1]]

    terms = []
    for w, a_dict, _a_const, b_dict, _b_const in rel.bilinear_linform_specs:
        a_cols = {int(k) for k in a_dict}
        b_cols = {int(k) for k in b_dict}
        allowed = a_cols | b_cols | {int(w)}
        rows = [int(r) for r in rows_with_col(int(w)) if support(r) <= allowed]
        rec = {
            "w": int(w),
            "n_a_cols": len(a_dict),
            "n_b_cols": len(b_dict),
            "n_rows": len(rows),
        }
        if len(rows) != 4:
            rec["candidates"] = [
                {
                    "row": r,
                    "coeffs": entries(r),
                    "rhs": float(b[r]),
                    "support": sorted(support(r)),
                }
                for r in rows
            ]
        terms.append(rec)

    counts: dict[str, int] = {}
    for t in terms:
        counts[str(t["n_rows"])] = counts.get(str(t["n_rows"]), 0) + 1
    return {
        "n_blf_terms": len(terms),
        "row_count_histogram": counts,
        "n_terms_not_4": sum(1 for t in terms if t["n_rows"] != 4),
        "n_cols": int(A.shape[1]),
        "n_rows_total": int(A.shape[0]),
        "terms": terms,
    }


def _spec_sufficiency() -> dict:
    """Assertion 2 — is the recorded BLF spec enough to build the expected rows?

    `_select` needs, per envelope row, the closed-form `(coeffs, rhs)`. For a BLF
    term `w = A·B` those are McCormick rows in `(aL, aH, bL, bH)` — the interval
    enclosures of the two affine forms. `_emit_mccormick` receives them as `ba`/`bb`
    from `ctx.bounds(node)` (an `evaluate_interval` on the ORIGINAL DAG node) but
    records only `(w, a.coeffs, a.const, b.coeffs, b.const)`. Verified structurally
    against the source so the answer is not an opinion.
    """
    from discopt._jax import uniform_relax as ur

    src = inspect.getsource(ur._emit_mccormick)
    appended = "ctx.bilinear_linform_specs.append(" in src
    # The tuple the producer unpacks, verbatim from the source.
    records_bounds = "ba" in src.split("append(")[1].split(")")[0] if appended else None
    return {
        "emit_mccormick_appends_spec": appended,
        "spec_tuple_records_form_bounds": bool(records_bounds),
        "spec_arity_in_producer": 5,  # `for w, a_dict, a_const, b_dict, b_const in ...`
        "ba_bb_come_from": "ctx.bounds(node) -> evaluate_interval on the original DAG",
    }


def _run_child(instance: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import discopt

    out = {"instance": instance, "discopt_file": discopt.__file__}
    # §8 marker: the decline code under test must exist in the loaded producer.
    from discopt._jax import spatial_producer

    psrc = inspect.getsource(spatial_producer)
    if "blf_row_count" not in psrc:
        out["error"] = "MARKER ABSENT: producer has no blf_row_count decline code"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 2
    out["marker_ok"] = True
    t0 = time.perf_counter()
    out["census"] = _producer_blf_census(instance)
    out["wall"] = time.perf_counter() - t0
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _parent(args) -> int:
    import discopt

    print(f"# discopt: {discopt.__file__}", flush=True)
    print(f"# load(1m) at start: {os.getloadavg()[0]:.2f}", flush=True)

    executed = 0

    print("\n## Assertion 1 — which term classes does `_select` cover?", flush=True)
    sel = _check_select_classes()
    executed += 1 + len(sel["covered_classes"]) + 2
    for k, v in sel.items():
        print(f"  {k}: {v}", flush=True)
    # The verdict rests on ONE predicate: does the module that owns `_select`
    # reference the field the producer declines on? Nothing looser.
    blf_covered = sel["mentions_bilinear_linform_specs"]
    print(f"  => `_select` covers the BLF (affine-form product) class: {blf_covered}", flush=True)

    print("\n## Assertion 2 — is the recorded BLF spec sufficient for a matcher?", flush=True)
    suf = _spec_sufficiency()
    executed += len(suf)
    for k, v in suf.items():
        print(f"  {k}: {v}", flush=True)

    print("\n## Measurement — BLF candidate rows on the real declining instances", flush=True)
    rows = []
    for inst in _BLF_INSTANCES + _CONTROL_INSTANCES:
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--child", inst],
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            cwd=str(_REPO_ROOT),
        )
        row = {"instance": inst, "rc": proc.returncode}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                row.update(json.loads(line[len("RESULT_JSON ") :]))
        if proc.returncode != 0 and "error" not in row:
            row["error"] = (proc.stderr or "")[-800:]
        rows.append(row)
        cen = row.get("census", {})
        executed += int(cen.get("n_blf_terms", 0))
        tag = "CONTROL" if inst in _CONTROL_INSTANCES else "declines"
        print(
            f"  {inst:12s} [{tag:8s}] rc={row['rc']} blf_terms={cen.get('n_blf_terms')} "
            f"hist={cen.get('row_count_histogram')} not4={cen.get('n_terms_not_4')}",
            flush=True,
        )
        for t in cen.get("terms", []):
            if t["n_rows"] != 4:
                print(
                    f"      term w={t['w']} |A|={t['n_a_cols']} |B|={t['n_b_cols']} "
                    f"rows={t['n_rows']}",
                    flush=True,
                )
                for c in t.get("candidates", []):
                    co = {k: round(v, 6) for k, v in c["coeffs"].items()}
                    print(
                        f"        row {c['row']:4d} rhs={c['rhs']:+.6g} "
                        f"supp={c['support']} coeffs={co}",
                        flush=True,
                    )

    print("\n## VERDICT", flush=True)
    print(f"  EXECUTED ASSERTIONS/COMPARISONS : {executed}", flush=True)
    total_terms = sum(int(r.get("census", {}).get("n_blf_terms", 0)) for r in rows)
    not4 = sum(int(r.get("census", {}).get("n_terms_not_4", 0)) for r in rows)
    print(f"  BLF terms examined              : {total_terms}", flush=True)
    print(f"  BLF terms claiming != 4 rows    : {not4}", flush=True)
    classes = sorted(k for k, v in sel["covered_classes"].items() if v)
    records_bounds = suf["spec_tuple_records_form_bounds"]
    print(f"  `_select` call sites            : {sel['n_select_call_sites']}", flush=True)
    print(f"  `_select` classes               : {classes}", flush=True)
    print(f"  BLF class covered by `_select`  : {blf_covered}", flush=True)
    print(f"  spec records form bounds ba/bb  : {records_bounds}", flush=True)
    if blf_covered:
        print("  CLAIM HOLDS: `_select` already covers the declining class.", flush=True)
    else:
        print(
            "  CLAIM FALSIFIED: `_select` covers bilinear / monomial / affine_square\n"
            "  ONLY. The declining class is `bilinear_linform_specs` (product of two\n"
            "  AFFINE FORMS), which `incremental_mccormick` never sees — the field is\n"
            "  consumed by `spatial_producer` alone. There is no matcher to port.",
            flush=True,
        )

    artifact = _REPORTS_DIR / f"phase52_blf_select_entry_{_short_sha()}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "schema": "phase52_blf_select_entry/1",
                "git_sha": _short_sha(),
                "executed": executed,
                "select_classes": sel,
                "spec_sufficiency": suf,
                "blf_class_covered_by_select": blf_covered,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nartifact: {artifact}", flush=True)

    if executed == 0:
        print("FAIL: zero executed assertions — the probe measured nothing.", flush=True)
        return 1
    bad = [r for r in rows if r.get("rc", 1) != 0]
    if bad:
        print(f"FAIL: {len(bad)} child(ren) exited non-zero: {[r['instance'] for r in bad]}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", metavar="INSTANCE", default=None)
    args = ap.parse_args()
    if args.child:
        return _run_child(args.child)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
