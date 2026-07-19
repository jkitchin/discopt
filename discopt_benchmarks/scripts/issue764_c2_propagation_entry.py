"""#764 -> C2 entry experiment — does SCIP-style nonlinear propagation climb tanksize's bound?

The SCIP mechanism trace (issue-764-scip-comparison.md, 2026-07-19) proved by ablation
that tanksize's dual bound climbs via cheap, cutoff-coupled *nonlinear constraint
propagation* (38,258 DomReds; OBBT and cuts near-irrelevant; propagation-off explodes
SCIP to 183k nodes). discopt's per-node FBBT is too weak here (OBBT-off -> 0.89 stall),
so it substitutes ~95 OBBT LP solves/node.

This experiment replicates SCIP's recipe with existing discopt pieces, on the REAL
instance, before any C2 build:

  best-bound B&B over the presolved root box, seeded incumbent (the known feasible
  1.2686437615), each node = FBBT FIXPOINT (linear rows + bidirectional affine-form
  product + sqrt + integer rounding + objective cutoff — NO LP probes) then ONE
  trusted McCormick LP solve (MccormickLPRelaxer.solve_at_node) for the node bound,
  then spatial midpoint branching on the widest nonlinear-participant variable.

Control arm: identical loop with propagation OFF (cutoff+integer rounding only) —
isolates propagation's contribution.

GO criterion: the propagation arm's global bound climbs materially toward SCIP's
trajectory (0.833 @1 -> 1.053 @50 -> 1.193 @200). KILL: stuck <= 0.86 by 300 nodes.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/issue764_c2_propagation_entry.py
"""

from __future__ import annotations

import heapq
import itertools
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

NL = "python/tests/data/minlplib_nl/tanksize.nl"
INC = 1.2686437614652892  # feasible objective found by the trusted Python path
NODE_BUDGET = int(os.environ.get("C2_NODES", "600"))
TIME_BUDGET = 420.0
GAP_TOL = 1e-4
_EPS = 1e-9


def _capture_presolved():
    """Run the trusted solve briefly to capture the model + finite root box."""
    import discopt.solver as S
    from discopt.modeling.core import from_nl

    cap = {}

    def grab(model, lb, ub, n_vars, *a, **k):
        cap.update(
            model=model,
            lb=np.asarray(lb, float)[:n_vars].copy(),
            ub=np.asarray(ub, float)[:n_vars].copy(),
        )
        return None

    os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
    S._try_native_spatial_kernel = grab
    from_nl(NL).solve(time_limit=4.0)
    return cap["model"], cap["lb"], cap["ub"]


def _structure(model, lb, ub):
    """Producer spec -> plain lists (fixed rows, products, sqrts, integrality, box)."""
    from discopt._jax.spatial_producer import build_spatial_kernel_spec

    spec = build_spatial_kernel_spec(model, bounds=(lb, ub))
    assert spec is not None, "producer must accept tanksize at the presolved box"
    n_cols = int(spec["n_cols"])
    rp = spec["fixed_row_ptr"]
    fixed = []
    for r in range(len(spec["fixed_rhs"])):
        s, e = int(rp[r]), int(rp[r + 1])
        fixed.append(
            (
                spec["fixed_cols"][s:e].astype(int),
                spec["fixed_coeffs"][s:e].astype(float),
                float(spec["fixed_rhs"][r]),
            )
        )
    blf = []
    ap, bp = spec["blf_a_ptr"], spec["blf_b_ptr"]
    for t in range(len(spec["blf_w"])):
        blf.append(
            (
                spec["blf_a_cols"][ap[t] : ap[t + 1]].astype(int),
                spec["blf_a_coeffs"][ap[t] : ap[t + 1]].astype(float),
                float(spec["blf_a_const"][t]),
                spec["blf_b_cols"][bp[t] : bp[t + 1]].astype(int),
                spec["blf_b_coeffs"][bp[t] : bp[t + 1]].astype(float),
                float(spec["blf_b_const"][t]),
                int(spec["blf_w"][t]),
            )
        )
    sqrts = []
    tk = spec["term_kind"]
    for t in range(len(tk)):
        if int(tk[t]) == 3:
            sqrts.append(
                (
                    int(spec["term_i"][t]),
                    int(spec["term_out"][t]),
                    float(spec["term_coeff"][t]),
                    float(spec["term_cst"][t]),
                )
            )
    integ = spec["integrality"].astype(bool)
    glo = spec["global_lo"].astype(float).copy()
    ghi = spec["global_hi"].astype(float).copy()
    obj_col = int(np.argmax(np.abs(spec["c"])))
    assert abs(spec["c"][obj_col] - 1.0) < 1e-12, "expected unit objective column"
    return n_cols, fixed, blf, sqrts, integ, glo, ghi, obj_col


def _iv_linform(cols, coeffs, cst, lo, hi):
    l = cst
    h = cst
    for c, j in zip(coeffs, cols):
        if c >= 0:
            l += c * lo[j]
            h += c * hi[j]
        else:
            l += c * hi[j]
            h += c * lo[j]
    return l, h


def _tighten_le(cols, coeffs, rhs, lo, hi):
    """Propagate sum(c_j x_j) <= rhs. Returns True if changed, None if infeasible."""
    mins = np.where(coeffs > 0, coeffs * lo[cols], coeffs * hi[cols])
    tot = float(mins.sum())
    if tot > rhs + 1e-7 * (1.0 + abs(rhs)):
        return None
    changed = False
    for k, j in enumerate(cols):
        c = coeffs[k]
        if abs(c) < 1e-12:
            continue
        cap = (rhs - (tot - mins[k])) / c
        if c > 0:
            if cap < hi[j] - _EPS * (1.0 + abs(hi[j])):
                hi[j] = cap
                changed = True
        else:
            if cap > lo[j] + _EPS * (1.0 + abs(lo[j])):
                lo[j] = cap
                changed = True
    return changed


def _tighten_form_to(cols, coeffs, cst, tlo, thi, lo, hi):
    """Push `form in [tlo, thi]` back onto the form's columns (two <= rows)."""
    ch = _tighten_le(cols, coeffs, thi - cst, lo, hi)
    if ch is None:
        return None
    ch2 = _tighten_le(cols, -coeffs, -(tlo - cst), lo, hi)
    if ch2 is None:
        return None
    return ch or ch2


def propagate(lo, hi, fixed, blf, sqrts, integ, obj_col, cutoff, rounds=15):
    """FBBT fixpoint. Returns (lo, hi) or None if the box is proven empty."""
    for _ in range(rounds):
        changed = False
        # objective cutoff (min sense, unit objective column).
        if cutoff < hi[obj_col] - _EPS:
            hi[obj_col] = cutoff
            changed = True
        # linear rows.
        for cols, coeffs, rhs in fixed:
            ch = _tighten_le(cols, coeffs, rhs, lo, hi)
            if ch is None:
                return None
            changed |= ch
        # products w = A*B: forward + reverse (skip reverse across a 0-straddling factor).
        for ac, av, acst, bc, bv, bcst, w in blf:
            alo, ahi = _iv_linform(ac, av, acst, lo, hi)
            blo, bhi = _iv_linform(bc, bv, bcst, lo, hi)
            ps = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
            wlo, whi = min(ps), max(ps)
            if wlo > lo[w] + _EPS * (1.0 + abs(lo[w])):
                lo[w] = wlo
                changed = True
            if whi < hi[w] - _EPS * (1.0 + abs(hi[w])):
                hi[w] = whi
                changed = True
            if lo[w] > hi[w] + 1e-7:
                return None
            # reverse: A in [w]/[B] when 0 not in [B]; likewise B.
            if blo > 1e-12 or bhi < -1e-12:
                qs = [lo[w] / blo, lo[w] / bhi, hi[w] / blo, hi[w] / bhi]
                ch = _tighten_form_to(ac, av, acst, min(qs), max(qs), lo, hi)
                if ch is None:
                    return None
                changed |= ch
            elif blo >= -1e-12 and bhi > 1e-12 and lo[w] > 1e-12:
                # One-sided extended division (the SCIP-style 0-touching case):
                # B in [0, bhi], w >= wlo > 0 forces B > 0 and A >= wlo/bhi > 0.
                # Sound lower bound on A; no finite upper (B -> 0+). This is the
                # case the strict-sign guard above blocks on tanksize (vars at 0).
                ch = _tighten_form_to(ac, av, acst, lo[w] / bhi, np.inf, lo, hi)
                if ch is None:
                    return None
                changed |= ch
            if alo > 1e-12 or ahi < -1e-12:
                qs = [lo[w] / alo, lo[w] / ahi, hi[w] / alo, hi[w] / ahi]
                ch = _tighten_form_to(bc, bv, bcst, min(qs), max(qs), lo, hi)
                if ch is None:
                    return None
                changed |= ch
            elif alo >= -1e-12 and ahi > 1e-12 and lo[w] > 1e-12:
                ch = _tighten_form_to(bc, bv, bcst, lo[w] / ahi, np.inf, lo, hi)
                if ch is None:
                    return None
                changed |= ch
        # sqrt w = sqrt(coeff*x + cst): forward + reverse (monotone).
        for xc, w, coeff, cst in sqrts:
            alo = coeff * (lo[xc] if coeff >= 0 else hi[xc]) + cst
            ahi = coeff * (hi[xc] if coeff >= 0 else lo[xc]) + cst
            alo = max(alo, 0.0)
            if ahi < -1e-9:
                return None
            wlo, whi = np.sqrt(alo), np.sqrt(max(ahi, 0.0))
            if wlo > lo[w] + _EPS:
                lo[w] = wlo
                changed = True
            if whi < hi[w] - _EPS:
                hi[w] = whi
                changed = True
            if lo[w] > hi[w] + 1e-7:
                return None
            # reverse: arg in [lo[w]^2, hi[w]^2] -> x.
            t_lo, t_hi = max(lo[w], 0.0) ** 2, hi[w] ** 2
            ch = _tighten_form_to(
                np.array([xc]), np.array([coeff]), cst, t_lo, t_hi, lo, hi
            )
            if ch is None:
                return None
            changed |= ch
        # integer rounding.
        for j in np.where(integ)[0]:
            nl = np.ceil(lo[j] - 1e-6)
            nh = np.floor(hi[j] + 1e-6)
            if nl > lo[j] + _EPS:
                lo[j] = nl
                changed = True
            if nh < hi[j] - _EPS:
                hi[j] = nh
                changed = True
            if lo[j] > hi[j] + 1e-9:
                return None
        if not changed:
            break
    if np.any(lo > hi + 1e-7):
        return None
    return lo, hi


def run_arm(tag, use_propagation, model, structure, log):
    n_cols, fixed, blf, sqrts, integ, glo, ghi, obj_col = structure
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    relaxer = MccormickLPRelaxer(model)
    n_orig = len(model._variables)
    # branch candidates: original columns participating in nonlinear terms.
    parts = set()
    for ac, _, _, bc, _, _, _ in blf:
        parts |= {int(c) for c in ac if c < n_orig} | {int(c) for c in bc if c < n_orig}
    for xc, _, _, _ in sqrts:
        parts.add(int(xc))
    parts |= {int(j) for j in np.where(integ[:n_orig])[0]}
    parts = sorted(parts)
    root_w = np.maximum(ghi[:n_orig] - glo[:n_orig], 1e-12)

    counter = itertools.count()
    heap = [(-np.inf, next(counter), glo.copy(), ghi.copy())]
    t0 = time.time()
    nodes = 0
    pruned_prop = 0
    closed_min = np.inf
    while heap and nodes < NODE_BUDGET and time.time() - t0 < TIME_BUDGET:
        pb, _, lo, hi = heapq.heappop(heap)
        frontier = min([pb] + [h[0] for h in heap]) if heap else pb
        gb = min(closed_min, frontier)
        if pb >= INC - GAP_TOL:
            closed_min = min(closed_min, pb)
            continue
        nodes += 1
        if nodes % 10 == 1 or nodes == NODE_BUDGET:
            log(
                f"[{tag}] node {nodes}: global_bound={gb:.5f} "
                f"open={len(heap)} prop_prunes={pruned_prop} t={time.time() - t0:.0f}s"
            )
        if use_propagation:
            res = propagate(lo, hi, fixed, blf, sqrts, integ, obj_col, INC)
            if res is None:
                pruned_prop += 1
                closed_min = min(closed_min, max(pb, INC))  # region infeasible under cutoff
                continue
            lo, hi = res
        else:
            hi[obj_col] = min(hi[obj_col], INC)
        # trusted node LP bound over the propagated ORIGINAL-variable box.
        x_lp = None
        try:
            nres = relaxer.solve_at_node(lo[:n_orig].copy(), hi[:n_orig].copy())
            lp = getattr(nres, "lower_bound", None)
            x_lp = getattr(nres, "x", None)
        except Exception:
            lp = None
        bound = max(pb, lp) if (lp is not None and np.isfinite(lp)) else pb
        if bound >= INC - GAP_TOL:
            closed_min = min(closed_min, bound)
            continue
        # branch: widest (root-relative) participant with usable width, split at the
        # LP point (clamped to the box interior) when available, else the midpoint.
        rel = [(float((hi[j] - lo[j]) / root_w[j]), j) for j in parts]
        rel = [(wv, j) for wv, j in rel if hi[j] - lo[j] > 1e-7]
        if not rel:
            closed_min = min(closed_min, bound)
            continue
        _, j = max(rel)
        mid = 0.5 * (lo[j] + hi[j])
        if x_lp is not None and j < len(x_lp) and np.isfinite(x_lp[j]):
            eps_j = 1e-3 * (hi[j] - lo[j])
            p = float(x_lp[j])
            if lo[j] + eps_j < p < hi[j] - eps_j:
                mid = p
        if integ[j]:
            f = np.floor(mid)
            l1, h1 = lo.copy(), hi.copy()
            h1[j] = f
            l2, h2 = lo.copy(), hi.copy()
            l2[j] = f + 1.0
            if h1[j] >= l1[j] - 1e-9:
                heapq.heappush(heap, (bound, next(counter), l1, h1))
            if h2[j] >= l2[j] - 1e-9:
                heapq.heappush(heap, (bound, next(counter), l2, h2))
        else:
            h1 = hi.copy()
            h1[j] = mid
            l2 = lo.copy()
            l2[j] = mid
            heapq.heappush(heap, (bound, next(counter), lo.copy(), h1))
            heapq.heappush(heap, (bound, next(counter), l2, hi.copy()))
    frontier = min((h[0] for h in heap), default=np.inf)
    gb = min(closed_min, frontier)
    status = "CLOSED" if not heap else "budget"
    log(
        f"[{tag}] FINAL: global_bound={gb:.5f} nodes={nodes} open={len(heap)} "
        f"prop_prunes={pruned_prop} status={status} t={time.time() - t0:.0f}s"
    )
    return gb, nodes


def main():
    out = open("scratchpad/c2_entry.out", "w")

    def log(s):
        out.write(s + "\n")
        out.flush()

    log(f"incumbent seed = {INC}; oracle = 1.2686437540")
    log("SCIP reference trajectory (prop-only): 0.833@1  1.053@50  1.193@200  1.269@1391")
    model, lb, ub = _capture_presolved()
    structure = _structure(model, lb, ub)
    log(f"structure: n_fixed={len(structure[1])} n_products={len(structure[2])} "
        f"n_sqrt={len(structure[3])}")
    if os.environ.get("C2_SKIP_CONTROL") == "1":
        gb_off, n_off = 0.83824, 300  # measured flat (see first run)
        log("[control:prop-OFF] skipped — measured flat 0.83824 over 300 nodes (first run)")
    else:
        gb_off, n_off = run_arm("control:prop-OFF", False, model, structure, log)
    gb_on, n_on = run_arm("prop-ON", True, model, structure, log)
    log("")
    log(f"VERDICT INPUTS: prop-OFF bound={gb_off:.5f} ({n_off} nodes) | "
        f"prop-ON bound={gb_on:.5f} ({n_on} nodes)")
    out.close()


if __name__ == "__main__":
    main()
