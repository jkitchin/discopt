#!/usr/bin/env python
"""Issue #786 / #790-P1 — SCIP-aligned LP-OA branch-and-cut PROTOTYPE (convex family).

The #786 entry experiment proved in-tree cutting is a real lever. This prototype
tests the SOTA-aligned ARCHITECTURE that delivers it — the one SCIP/BARON use and
the one the NLP-per-node path cannot: an **LP relaxation at every node, cut into
natively**, not an NLP per node with cuts bolted on.

Per node (all reusing the validated #781/#786 machinery):
  * relaxation = linear rows + OA tangents of the convex nonlinear rows over the
    node box, refined to OA convergence;
  * separate GMI (tableau) + c-MIR + cover into the node LP under pooled top-K
    selection — cheap because the relaxation is an LP;
  * the LP optimum is the node dual bound (sound: LP ⊇ the integer-feasible set);
  * an integer-integral, OA-tight LP vertex is a genuine feasible point -> an
    incumbent (the LP-NLP-BB primal, minimal form: no separate NLP solve needed
    when the vertex is already tight);
  * branch on the most-fractional integer into two covering child boxes.

Best-bound worklist; fathom by bound. Reports nodes-to-certify and wall, next to
the current NLP-BB path (`model.solve`) on the same instances — the decision on
whether to build the Rust LP-OA kernel (#790 P1) rests on this.

Soundness: every node bound is an LP relaxation optimum over an outer
approximation of the node's integer-feasible set (OA tangents + integrality-valid
cuts), hence a valid dual bound; the accepted incumbent is integer-integral AND
OA-tight (nonlinear residual <= tol), i.e. genuinely feasible, so its objective
is a valid primal bound. Cuts are re-separated fresh per node over that node's
box — never shared across siblings (the C-43 lesson) — so no node-scoping bug is
possible.

THROWAWAY prototype — no shipped solver code.
"""

from __future__ import annotations

import heapq
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discopt._jax.cmir_cuts import separate_cmir  # noqa: E402
from discopt._jax.cover_cuts import separate_cover_cuts  # noqa: E402
from issue781_cutmgmt_probe import (  # noqa: E402
    PANEL,
    CutPool,
    RootModel,
    select_cuts,
    separate_gmi,
    solve_lp_highs,
)
from issue786_intree_value_probe import child_fbbt  # noqa: E402

OA_TOL = 1e-6
CUT_VIOL_TOL = 1e-6
INT_TOL = 1e-6
SEP_ROUNDS = 12
GAP_TOL = 1e-4
MAX_NODES = 20000
NODE_TIME = 60.0


def node_relax(rm, lo, hi, lb_s, ub_s, separate=True):
    """LP-OA relaxation over [lo,hi] with in-tree separation. Returns (bound, x)
    in the LP's maximize-c'x internal sense; (None, None) if infeasible."""
    saved = (rm.lb, rm.ub)
    rm.lb, rm.ub = lo, hi
    try:
        ca, cb = [], []
        pool = CutPool()

        def oa():
            obj, x, duals, h, _n = solve_lp_highs(rm, ca, cb)
            for _ in range(60):
                if x is None:
                    break
                added = 0
                for i, v in rm.nonlinear_violation(x).items():
                    if v > OA_TOL:
                        t = rm.oa_tangent(i, x)
                        if t is not None:
                            ca.append(t[0])
                            cb.append(t[1])
                            added += 1
                if not added:
                    break
                obj, x, duals, h, _n = solve_lp_highs(rm, ca, cb)
            return obj, x, duals, h

        obj, x, duals, h = oa()
        if x is None:
            return None, None
        if separate:
            for _ in range(SEP_ROUNDS):
                a_all = np.vstack([rm.A_le] + ([np.array(ca)] if ca else []))
                b_all = np.concatenate([rm.b_le] + ([np.array(cb)] if cb else []))
                cands = list(
                    separate_cmir(a_all, b_all, x, lb_s, ub_s, rm.is_int, max_cuts=24, duals=duals)
                )
                for cover, rhs in separate_cover_cuts(a_all, b_all, x, rm.is_bin, max_cuts=32):
                    arr = np.zeros(rm.n)
                    arr[list(cover)] = 1.0
                    cands.append((arr, float(rhs)))
                if h is not None:
                    cands += separate_gmi(rm, h, x, a_all, b_all, rm.A_eq.shape[0])
                for arr, r in cands:
                    arr = np.asarray(arr, float)
                    if arr @ x - float(r) > CUT_VIOL_TOL:
                        pool.offer(arr, float(r))
                chosen = select_cuts(pool.violated(x), x)
                if not chosen:
                    break
                for arr, r in chosen:
                    ca.append(arr)
                    cb.append(r)
                obj, x, duals, h = oa()
                if x is None:
                    return None, None
        return obj, x
    finally:
        rm.lb, rm.ub = saved


def integer_and_tight(rm, x):
    """Is x integer-integral on all int vars AND OA-tight (feasible)?"""
    for j in range(rm.n):
        if rm.is_int[j] and abs(x[j] - round(x[j])) > INT_TOL:
            return False
    return all(v <= 1e-5 for v in rm.nonlinear_violation(x).values())


def most_fractional(rm, x):
    best, bj = INT_TOL, None
    for j in range(rm.n):
        if rm.is_int[j]:
            f = abs(x[j] - round(x[j]))
            if min(f, 1 - f) > best:
                best, bj = min(f, 1 - f), j
    return bj


def solve_lpoa_bandc(rm, opt, seed_incumbent=None, separate=True):
    """Best-bound LP-OA branch-and-cut. Returns dict with status/bound/incumbent/nodes.

    ``seed_incumbent``: start the search with a known feasible objective (e.g. the
    oracle optimum) so the measurement isolates the DUAL side — nodes-to-certify,
    which in-tree cutting drives — from the primal (a separate, solvable concern:
    SCIP uses rounding/diving/RENS to find incumbents, out of scope for this
    architecture probe). With the optimum seeded, nodes-to-certify = how many
    nodes the LP-OA B&C tree needs to PROVE optimality. ``separate=False`` runs
    the same tree WITHOUT in-tree cutting (OA-only node bounds) — the ablation
    that isolates the cutting's contribution to the tree size."""
    lb_s = np.where(np.isfinite(rm.lb_sep), rm.lb_sep, 0.0)
    ub_s = np.where(np.isfinite(np.minimum(rm.ub, rm.ub_sep)), np.minimum(rm.ub, rm.ub_sep), 1e5)
    # maximize: LP bound is an UPPER bound; incumbent is a LOWER bound.
    incumbent = -np.inf if seed_incumbent is None else float(seed_incumbent)
    first_inc_node = None
    # heap of (-upper_bound, tie, lo, hi): pop the node with the LARGEST upper
    # bound first (best-bound for maximize).
    tie = 0
    root_b, root_x = node_relax(rm, rm.lb.copy(), rm.ub.copy(), lb_s, ub_s, separate)
    if root_x is None:
        return dict(status="infeasible", nodes=0)
    heap = [(-root_b, tie, rm.lb.copy(), rm.ub.copy(), root_b, root_x)]
    nodes = 0
    t0 = time.time()
    global_ub = root_b
    while heap:
        if nodes >= MAX_NODES or time.time() - t0 > NODE_TIME:
            break
        neg_pb, _, lo, hi, nb, nx = heapq.heappop(heap)
        pb = -neg_pb
        # global upper bound = max pb over frontier (heap top) — but we popped it;
        # recompute from remaining frontier + this node.
        global_ub = max([pb] + [-h[0] for h in heap]) if heap else pb
        if pb <= incumbent + GAP_TOL * max(1.0, abs(incumbent)):
            continue  # fathom: cannot beat incumbent
        nodes += 1
        # (nb, nx) are this node's relaxation from when it was created; re-use.
        x = nx
        if integer_and_tight(rm, x):
            if nb > incumbent:
                incumbent = nb
                if first_inc_node is None:
                    first_inc_node = nodes
            continue
        j = most_fractional(rm, x)
        if j is None:
            continue
        for lo2, hi2 in (
            (lo.copy(), _seti(hi, j, np.floor(x[j]))),
            (_seti(lo, j, np.ceil(x[j])), hi.copy()),
        ):
            lo2, hi2 = child_fbbt(rm, lo2, hi2)
            if np.any(lo2 > hi2 + 1e-9):
                continue
            cb2, cx2 = node_relax(rm, lo2, hi2, lb_s, ub_s, separate)
            if cx2 is None:
                continue
            cb2 = min(cb2, pb)  # child bound inherits parent (tighter of the two)
            if cb2 <= incumbent + GAP_TOL * max(1.0, abs(incumbent)):
                continue
            tie += 1
            heapq.heappush(heap, (-cb2, tie, lo2, hi2, cb2, cx2))
    wall = time.time() - t0
    # final global upper bound
    global_ub = max([incumbent] + [-h[0] for h in heap]) if heap else incumbent
    gap = abs(global_ub - incumbent) / max(1.0, abs(incumbent)) if incumbent > -np.inf else None
    certified = incumbent > -np.inf and gap is not None and gap <= GAP_TOL
    return dict(
        status="optimal" if certified else ("feasible" if incumbent > -np.inf else "no_inc"),
        bound=global_ub,
        incumbent=None if incumbent == -np.inf else incumbent,
        nodes=nodes,
        wall=wall,
        first_inc_node=first_inc_node,
        gap=gap,
    )


def _seti(arr, j, v):
    a = arr.copy()
    a[j] = v
    return a


def main():
    import discopt.modeling as dm

    snap = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
    report = {}
    for name, info in PANEL.items():
        opt = info["opt"]
        print(f"\n===== {name} opt={opt} =====", flush=True)
        # LP-OA branch-and-cut prototype
        rm = RootModel(name)
        r = solve_lpoa_bandc(rm, opt)
        print(
            f"  LP-OA B&C: {r['status']} inc={r.get('incumbent')} bound={r.get('bound'):.3f} "
            f"nodes={r['nodes']} wall={r['wall']:.1f}s gap={r.get('gap')}",
            flush=True,
        )
        # NLP-BB baseline (current path) for comparison, same wall budget
        os.environ["DISCOPT_NLPBB_ROOT_CUTS"] = "0"
        t0 = time.time()
        rb = dm.from_nl(snap + name + ".nl").solve(time_limit=NODE_TIME)
        nlp = dict(
            status=str(rb.status),
            obj=rb.objective,
            bound=rb.bound,
            nodes=rb.node_count,
            wall=time.time() - t0,
            cert=bool(rb.gap_certified),
        )
        print(
            f"  NLP-BB    : {nlp['status']} obj={nlp['obj']} bound={nlp['bound']} "
            f"nodes={nlp['nodes']} wall={nlp['wall']:.1f}s cert={nlp['cert']}",
            flush=True,
        )
        report[name] = dict(opt=opt, lpoa=r, nlpbb=nlp)
    print("\n===== VERDICT (#786 / #790-P1: does LP-OA B&C beat NLP-BB?) =====")
    lpoa_cert = sum(1 for r in report.values() if r["lpoa"].get("status") == "optimal")
    nlp_cert = sum(1 for r in report.values() if r["nlpbb"].get("cert"))
    print(f"  certified: LP-OA B&C {lpoa_cert}/{len(report)}  vs  NLP-BB {nlp_cert}/{len(report)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "results", "issue786"))
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/lpoa_bandc_prototype_{stamp}.json"
    with open(outp, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
