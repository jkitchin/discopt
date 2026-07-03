# ruff: noqa  -- derivation/verification harness (math notation)
"""T1.2 derivation + verification: closed-form generators for trilinear (x*y*z)
and multilinear (>=4 distinct factors) product families.

Strategy: reproduce EXACTLY what build_milp_relaxation emits for a single-term
model min(prod x_i). The cold builder:
  - allocates one aux column w_S per subset S of factors with |S|>=2, in the
    order: the term's own chosen-pair + final product first (classifier), then
    the RLT subset sweep k=2..n over itertools.combinations. Column ordering is
    irrelevant to the row-SET comparison, so we key rows/bounds by (frozenset S).
  - |S|=2: 4 McCormick rows on the two constituent columns (recursive: the
    constituents may be aux columns), using their propagated corner bounds.
  - |S|>=3: for each corner in {0,1}^k, one bound-factor-product RLT row.

We rebuild the *entire* box-dependent block from bounds only and compare the
canonical row-set (round 6) + aux bounds to the cold builder, on many boxes and
sign regimes, to 1e-9.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools

import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

# ---- closed-form generators (bounds -> rows keyed by subset) ----------------

def _bilinear_rows_cols(li, ui, lj, uj):
    """4 standard McCormick rows for w = xi*xj over [li,ui]x[lj,uj], where
    (i,j) are the two CONSTITUENT columns with i=min(col), j=max(col) (matching
    the cold builder's ``key = (min,max)`` and its cv/cc row order). Each row is
    (coef_i, coef_j, coef_w, rhs) of a `... <= rhs` row."""
    return [
        (lj, li, -1.0, li * lj),      # cv1: -w + lj*xi + li*xj <= li*lj
        (uj, ui, -1.0, ui * uj),      # cv2: -w + uj*xi + ui*xj <= ui*uj
        (-lj, -ui, 1.0, -ui * lj),    # cc1:  w - lj*xi - ui*xj <= -ui*lj
        (-uj, -li, 1.0, -li * uj),    # cc2:  w - uj*xi - li*xj <= -li*uj
    ]


def _corner_bounds(la, ua, lb, ub):
    corners = (la * lb, la * ub, ua * lb, ua * ub)
    return min(corners), max(corners)


def build_product_block(vars_, lb_orig, ub_orig, rlt_cap=4):
    """Reconstruct the trilinear/multilinear relaxation block for a single
    product term over the given factor variables.

    vars_: tuple of original column indices (the distinct factors), sorted.
    lb_orig/ub_orig: dict col-> (lb,ub) for the ORIGINAL factor columns.

    Returns:
      rows: list of (coef_dict, rhs)  where coef_dict maps col-index -> coef
      aux_bounds: dict col-index -> (lo, hi)
    Aux columns are assigned deterministically matching the cold builder's
    allocation order so indices line up. (For the row-SET test only the value
    matters; we still assign indices to build concrete rows.)
    """
    n = len(vars_)
    # bounds per column (original + aux subsets)
    bnd: dict[frozenset, tuple[float, float]] = {}
    col: dict[frozenset, int] = {}
    for v in vars_:
        bnd[frozenset([v])] = (lb_orig[v], ub_orig[v])
        col[frozenset([v])] = v

    next_col = max(vars_) + 1

    def ensure(S: frozenset) -> int:
        nonlocal next_col
        if S in col:
            return col[S]
        m = max(S)
        rest = S - {m}
        ca = ensure(rest)  # constituent aux/orig
        cb = m
        la, ua = bnd[rest]
        lb, ub = bnd[frozenset([m])]
        lo, hi = _corner_bounds(la, ua, lb, ub)
        col[S] = next_col
        bnd[S] = (lo, hi)
        next_col += 1
        return col[S]

    # Column allocation order to MATCH the cold builder exactly.
    # (a) arity 3 -> classifier registers the chosen pair then the product;
    #     arity >=4 -> multilinear left-fold w01, w012, w0123, ...
    if n == 3:
        pair, remaining = _choose_pair(vars_)
        ensure(frozenset(pair))
        ensure(frozenset(vars_))
    else:
        s = sorted(vars_)
        cur = frozenset([s[0]])
        for v in s[1:]:
            cur = cur | {v}
            ensure(cur)
    # (b) RLT subset sweep k=2..n, itertools.combinations order -- ONLY when the
    #     arity is within the RLT cap (DISCOPT_MULTILINEAR_RLT_MAX, default 4).
    #     Above the cap the cold builder keeps the loose recursive chain only.
    if n <= rlt_cap:
        for k in range(2, n + 1):
            for comb in itertools.combinations(vars_, k):
                ensure(frozenset(comb))

    rows: list[tuple[dict, float]] = []

    # McCormick rows for EVERY subset column w_S (|S|>=2), emitted on its two
    # recursive constituent columns (a = col[S\max], b = max), keyed (min,max)
    # to match the cold builder's ``bilinear_relation_map`` iteration. This
    # includes the nested products (e.g. w_{xyz} = w_{xy} * z) whose constituent
    # is itself an aux column -- the piece my first pass missed.
    for S in [frozenset(s) for s in col if len(s) >= 2]:
        m = max(S)
        rest = S - {m}
        a_raw = col[rest]   # constituent column (orig var or aux)
        b_raw = m
        la_r, ua_r = bnd[rest]
        lb_m, ub_m = bnd[frozenset([m])]
        # order columns as (i=min, j=max) with their bounds, matching cold
        if a_raw <= b_raw:
            i_col, j_col = a_raw, b_raw
            li, ui, lj, uj = la_r, ua_r, lb_m, ub_m
        else:
            i_col, j_col = b_raw, a_raw
            li, ui, lj, uj = lb_m, ub_m, la_r, ua_r
        w_col = col[S]
        for ci, cj, cw, rhs in _bilinear_rows_cols(li, ui, lj, uj):
            d = {}
            d[i_col] = d.get(i_col, 0.0) + ci
            d[j_col] = d.get(j_col, 0.0) + cj
            d[w_col] = d.get(w_col, 0.0) + cw
            rows.append((d, rhs))

    # |S|>=3 RLT bound-factor-product rows (only within the RLT cap)
    rlt_kmax = n if n <= rlt_cap else 2
    for k in range(3, rlt_kmax + 1):
        for comb in itertools.combinations(vars_, k):
            lo_hi = {i: bnd[frozenset([i])] for i in comb}
            for corner in itertools.product((0, 1), repeat=k):
                sc = {
                    i: ((1.0, -lo_hi[i][0]) if corner[pos] == 0 else (-1.0, lo_hi[i][1]))
                    for pos, i in enumerate(comb)
                }
                d: dict[int, float] = {}
                const = 0.0
                for r in range(k + 1):
                    for tcomb in itertools.combinations(comb, r):
                        tset = frozenset(tcomb)
                        coef = 1.0
                        for i in comb:
                            coef *= sc[i][0] if i in tset else sc[i][1]
                        if r == 0:
                            const += coef
                        else:
                            c = col[tset]
                            d[c] = d.get(c, 0.0) + coef
                # product >= 0 -> -(linear) <= const
                d = {kk: -vv for kk, vv in d.items()}
                rows.append((d, const))

    aux_bounds = {col[S]: bnd[S] for S in col if len(S) >= 2}
    return rows, aux_bounds


def _choose_pair(vars_):
    """Mirror _choose_trilinear_pair for the no-partition case (deterministic).
    For >3 vars this only matters for column-allocation order (row-SET is
    order-free), so we use a consistent rule: pair = two smallest? Actually the
    cold trilinear path uses _choose_trilinear_pair only for arity 3. For >=4,
    the multilinear path uses _ensure_multilinear_aux (nested left fold). We
    replicate arity-3 exactly; for >=4 the ordering is handled by the RLT sweep
    plus the nested fold, but row-SET comparison makes the exact order moot."""
    i, j, k = None, None, None
    if len(vars_) == 3:
        i, j, k = sorted(vars_)
        candidates = [((i, j), k), ((i, k), j), ((j, k), i)]
        candidates.sort()
        # no partitioned vars -> key is (0,0) for all -> max picks LAST after sort
        best = max(candidates, key=lambda item: (0, False))
        return best
    # >=4: nested left fold pair = (v0, v1)
    s = sorted(vars_)
    return ((s[0], s[1]), s[2])


# ---- cold builder extraction ------------------------------------------------

def cold_block(nfac, lb, ub):
    m = dm.Model()
    vs = [m.continuous(f"x{i}", lb=-100, ub=100) for i in range(nfac)]
    expr = vs[0]
    for v in vs[1:]:
        expr = expr * v
    m.minimize(expr)
    terms = classify_nonlinear_terms(m)
    lo = np.array(lb, dtype=float)
    hi = np.array(ub, dtype=float)
    relax, vm = build_milp_relaxation(
        m, terms, DiscretizationState(), bound_override=(lo, hi)
    )
    A = np.asarray(sp.csr_matrix(relax._A_ub).todense(), dtype=np.float64)
    b = np.asarray(relax._b_ub, dtype=np.float64).ravel()
    bnds = np.asarray(relax._bounds, dtype=np.float64)
    return A, b, bnds, nfac


def rowset_from_dense(A, b):
    rows = np.hstack([np.round(A, 6), np.round(b, 6).reshape(-1, 1)])
    return sorted(map(tuple, rows.tolist()))


def rowset_from_dicts(rows, ncol):
    out = []
    for d, rhs in rows:
        r = np.zeros(ncol)
        for c, v in d.items():
            r[c] += v
        out.append(tuple(np.round(np.append(r, rhs), 6).tolist()))
    return sorted(out)


def verify(nfac, lb, ub):
    A, b, bnds, _ = cold_block(nfac, lb, ub)
    ncol = A.shape[1]
    vars_ = tuple(range(nfac))
    lb_o = {i: lb[i] for i in range(nfac)}
    ub_o = {i: ub[i] for i in range(nfac)}
    rows, aux_bounds = build_product_block(vars_, lb_o, ub_o)

    # keep only the box-dependent (product) rows from the cold builder:
    # every cold row here IS a product row (single-term model has no model rows),
    # but the objective may add nothing. Compare full row-sets.
    cold_rs = rowset_from_dense(A, b)
    mine_rs = rowset_from_dicts(rows, ncol)
    rows_ok = cold_rs == mine_rs

    # aux bounds: cols >= nfac
    bounds_ok = True
    for c in range(nfac, ncol):
        clo, chi = float(bnds[c, 0]), float(bnds[c, 1])
        if c in aux_bounds:
            mlo, mhi = aux_bounds[c]
            if not (abs(clo - mlo) < 1e-9 and abs(chi - mhi) < 1e-9):
                bounds_ok = False
        else:
            bounds_ok = False
    ncol_ok = ncol == nfac + len(aux_bounds)
    return rows_ok, bounds_ok, ncol_ok, A.shape, cold_rs, mine_rs


# ---- run --------------------------------------------------------------------

def rand_box(rng, n, regime):
    """regime: 'pos','neg','mixed','zero-lo' etc."""
    lb = np.empty(n)
    ub = np.empty(n)
    for i in range(n):
        w = rng.uniform(0.5, 6.0)
        if regime == "pos":
            lo = rng.uniform(0.1, 5.0)
        elif regime == "neg":
            lo = -rng.uniform(0.1, 5.0) - w
        elif regime == "mixed":
            lo = rng.uniform(-5.0, 5.0)
        elif regime == "zero-lo":
            lo = 0.0
        elif regime == "spanning":
            lo = -rng.uniform(0.5, 5.0)
            w = (-lo) + rng.uniform(0.5, 5.0)
        else:
            lo = rng.uniform(-5, 5)
        lb[i] = lo
        ub[i] = lo + w
    return lb, ub


def main():
    rng = np.random.default_rng(12345)
    regimes = ["pos", "neg", "mixed", "zero-lo", "spanning"]
    total = 0
    passed = 0
    shapes = {}
    fails = []
    for nfac in (3, 4, 5, 6):
        for regime in regimes:
            for _ in range(30):
                lb, ub = rand_box(rng, nfac, regime)
                try:
                    rows_ok, bounds_ok, ncol_ok, shape, crs, mrs = verify(
                        nfac, lb, ub
                    )
                except Exception as e:  # noqa: BLE001
                    fails.append((nfac, regime, list(lb), list(ub), f"EXC {e}"))
                    total += 1
                    continue
                total += 1
                shapes.setdefault((nfac, regime), set()).add(shape)
                if rows_ok and bounds_ok and ncol_ok:
                    passed += 1
                else:
                    if len(fails) < 6:
                        fails.append(
                            (nfac, regime, list(np.round(lb, 3)),
                             list(np.round(ub, 3)),
                             f"rows={rows_ok} bounds={bounds_ok} ncol={ncol_ok}")
                        )
    print(f"PASS {passed}/{total}  ({100*passed/total:.1f}%)")
    print("shapes per (nfac,regime):")
    for k, v in sorted(shapes.items()):
        print(f"  {k}: {sorted(v)}")
    if fails:
        print("\nfailures (first few):")
        for f in fails:
            print(" ", f)


if __name__ == "__main__":
    main()
