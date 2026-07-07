"""T1.1 entry experiment: incremental McCormick engine coverage on out-of-scope shapes.

For each of three out-of-scope instances — (a) maximize, (b) mixed int+continuous
bilinear, (c) general-NL — construct IncrementalMcCormickLP directly (bypassing
_is_in_scope), record whether it validates (.ok), enumerate the lifted-LP term
families present (non-empty varmap entries), and time the cold build path.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
from discopt._jax.discretization import DiscretizationState
from discopt._jax.incremental_mccormick import IncrementalMcCormickLP
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import ObjectiveSense, VarType

# Families the incremental engine patches in closed form today.
_COVERED = {"bilinear", "monomial_p2"}

# varmap keys that denote actual lifted nonlinear term families (skip bookkeeping
# keys like "original", "*_stages", "*_signatures", encoding flags).
_FAMILY_KEYS = [
    "bilinear", "trilinear", "multilinear", "monomial", "monomial_pw",
    "univariate", "univariate_relaxations", "composite_relaxations",
    "composite_multivar_relaxations", "univariate_piecewise_relaxations",
    "univariate_square", "univariate_square_relaxations", "fractional_power",
    "bilinear_pw", "bilinear_lambda",
]

INSTANCES = {
    "syn05m (maximize)": "syn05m",
    "ex1263 (mixed int+cont bilinear)": "ex1263",
    "st_e38 (general-NL)": "st_e38",
}


def _families_present(varmap):
    fams = {}
    for k in _FAMILY_KEYS:
        v = varmap.get(k)
        if v:
            fams[k] = len(v) if hasattr(v, "__len__") else 1
    # split monomial by power
    mono = varmap.get("monomial", {})
    if mono:
        by_pow = {}
        for key in mono:
            p = key[1] if isinstance(key, tuple) and len(key) > 1 else "?"
            by_pow[p] = by_pow.get(p, 0) + 1
        fams["monomial"] = by_pow  # {power: count}
    return fams


def run(label, name):
    path = f"python/tests/data/minlplib/{name}.nl"
    print(f"\n=== {label}  [{name}] ===")
    if not os.path.exists(path):
        print("  NOT VENDORED — skip")
        return None
    model = dm.from_nl(path)
    sense = "MAX" if model._objective.sense == ObjectiveSense.MAXIMIZE else "MIN"
    n_int = sum(v.size for v in model._variables if v.var_type in (VarType.INTEGER, VarType.BINARY))
    n_cont = sum(v.size for v in model._variables if v.var_type == VarType.CONTINUOUS)
    print(f"  sense={sense}  vars: {n_int} int/bin + {n_cont} continuous")

    terms = classify_nonlinear_terms(model)
    n = len(model._variables)
    lb = np.array([1.0] * n)
    ub = np.array([7.0 + k for k in range(n)])

    # Cold build + timing (median of a few).
    families = {}
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            relax, varmap = build_milp_relaxation(
                model, terms, DiscretizationState(), bound_override=(lb, ub)
            )
            times.append(time.perf_counter() - t0)
            families = _families_present(varmap)
        except Exception as e:
            print(f"  cold build_milp_relaxation FAILED: {type(e).__name__}: {e}")
            break
    cold_ms = 1e3 * float(np.median(times)) if times else float("nan")
    print(f"  cold build_milp_relaxation: {cold_ms:.2f} ms/call")
    print(f"  lifted term families present: {families}")

    inc = IncrementalMcCormickLP(model, terms)
    print(f"  IncrementalMcCormickLP.ok = {inc.ok}")

    # Which present families are NOT covered by the incremental patch table?
    present = set(families)
    # normalize monomial: covered only for power 2
    uncovered = []
    for fam in present:
        if fam == "monomial":
            powers = families["monomial"]
            for p in powers:
                if p != 2:
                    uncovered.append(f"monomial^{p}")
        elif fam not in _COVERED and fam != "bilinear":
            uncovered.append(fam)
    covered = [
        f
        for f in present
        if f == "bilinear" or (f == "monomial" and 2 in families.get("monomial", {}))
    ]
    print(f"  covered families: {covered}")
    print(f"  UNCOVERED families (would fall back): {uncovered}")
    return {"name": name, "sense": sense, "ok": inc.ok, "families": families,
            "uncovered": uncovered, "cold_ms": cold_ms}


if __name__ == "__main__":
    results = []
    for label, name in INSTANCES.items():
        try:
            r = run(label, name)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  EXPERIMENT ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n\n===== SUMMARY =====")
    for r in results:
        print(f"{r['name']:10s} sense={r['sense']} ok={r['ok']} "
              f"cold={r['cold_ms']:.1f}ms uncovered={r['uncovered']}")
