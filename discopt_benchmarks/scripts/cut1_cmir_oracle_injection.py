"""CUT-1 entry experiment — inject SCIP's aggregation/complemented-MIR cuts into
discopt's ROOT McCormick LP and measure the root-bound closure.

This is the decisive spike the Phase-3 1c NO-GO left open. 1c measured discopt's
OWN separators firing on the integer-product class (~0% close) and concluded the
residual was "separator DEPTH". The zerohalf build then found the graphpart
non-movement is LP-vertex geometry, not depth. CUT-1 tests the remaining question
head-on for the nvs17/19/24 integer-product family (scip-gap-closing-plan §1.3's
c-MIR workhorse class):

    Would a CORRECT aggregation / complemented-MIR cut actually close discopt's
    root gap if injected, or does discopt's relaxation not absorb it (a
    relaxation-mismatch that would make a native c-MIR build inert)?

Method (the oracle test):
  1. Build discopt's root lifted McCormick LP (``build_milp_relaxation``) at a
     bound box tightened to SCIP's presolved integer bounds (so the two relaxations
     share the same variable frame — the cut is generated in that frame). Solve it
     -> ``root_off`` (discopt's own root LP floor).
  2. Run SCIP root-only (node limit 1), read its LP rows, keep the ``cmir``/``agg``
     separator cut rows. Resolve each SCIP auxiliary variable to its monomial
     (product x_i*x_j or square x_i^2) from the McCormick envelope row names +
     underestimate/overestimate rows. Map each cut into discopt's lifted column
     space via the relaxation ``varmap``.
  3. Append the mapped cut rows to discopt's lifted LP and re-solve -> ``root_on``.
  4. gap_closed = (root_on - root_off) / (opt - root_off).

KILL CRITERION (per the task): if injecting SCIP's own aggregation/c-MIR cuts
closes < ~15% of the root gap on >=2 of the 3 instances -> relaxation-mismatch,
recommend NO-GO. If >=15-30%+ closes -> GO.

Usage:
    PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/cut1_cmir_oracle_injection.py
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import contextlib

import numpy as np

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_RESULTS_DIR = _BENCH_ROOT / "results"

PANEL = ["nvs17", "nvs19", "nvs24"]
OPT = {"nvs17": -1100.4, "nvs19": -1098.4, "nvs24": -1033.2}
CAP_SECONDS = 120.0


# --------------------------------------------------------------------------- #
# SCIP side: extract cut rows + resolve auxvar -> monomial identity
# --------------------------------------------------------------------------- #
def _scip_root(name: str):
    """Return (scip_bounds, cut_rows, aux_identity, scip_root_bound, scip_trivial).

    scip_bounds: dict t_iK-name -> (lo, hi) presolved integer bounds (1-indexed K).
    cut_rows:    list of (name, {varname: coef}, rhs)  for cmir/agg cuts (<= form).
    aux_identity: dict auxvar-name -> ('prod', i, j) | ('pow', i)  (0-indexed orig).
    """
    from pyscipopt import Model

    nl = str(_NL_DIR / f"{name}.nl")

    # trivial floor: separation + propagation + presolve off, one node.
    m0 = Model()
    m0.hideOutput()
    m0.readProblem(nl)
    m0.setParam("limits/nodes", 1)
    m0.setParam("limits/time", CAP_SECONDS)
    m0.setParam("separating/maxrounds", 0)
    m0.setParam("separating/maxroundsroot", 0)
    m0.setParam("presolving/maxrounds", 0)
    m0.setParam("propagating/maxrounds", 0)
    m0.setParam("propagating/maxroundsroot", 0)
    m0.optimize()
    try:
        trivial = float(m0.getDualbound())
    except Exception:
        trivial = None

    # root with default cut loop, one node, presolve OFF so var names stay t_iK and
    # the cut/var frame is the un-presolved integer box (matches discopt's box).
    m = Model()
    m.hideOutput()
    m.readProblem(nl)
    m.setParam("limits/nodes", 1)
    m.setParam("limits/time", CAP_SECONDS)
    m.setParam("presolving/maxrounds", 0)
    m.optimize()
    scip_root = None
    with contextlib.suppress(Exception):
        scip_root = float(m.getDualbound())

    rows = m.getLPRowsData()

    # Presolved integer bounds on t_iK.
    scip_bounds: dict[str, tuple[float, float]] = {}
    for v in m.getVars(transformed=True):
        if v.name.startswith("t_i"):
            scip_bounds[v.name] = (v.getLbLocal(), v.getUbLocal())

    def _orig_index(tname: str) -> int:
        # t_iK -> K-1 (0-indexed original variable).
        mo = re.match(r"t_i(\d+)", tname)
        return int(mo.group(1)) - 1 if mo else -1

    # Resolve auxvar -> monomial.
    # (a) pow auxvars from 'underestimate_pow'/'overestimate_pow' rows: single orig factor.
    # (b) prod auxvars from 'underestimate_prod'/'overestimate_prod' rows: two orig factors.
    # (c) 'minor_auxvar_pow_A_auxvar_pow_B_auxvar_prod_C_*' names:
    #     prod_C = (var of pow_A) * (var of pow_B).
    aux_identity: dict[str, tuple] = {}
    pow_of: dict[str, int] = {}  # auxvar_pow_N -> orig index i (means i^2)

    for r in rows:
        nm = r.name
        cols, vals = r.getCols(), r.getVals()
        vs = {c.getVar().name: v for c, v in zip(cols, vals, strict=False)}
        auxs = [n for n in vs if n.startswith("auxvar")]
        origs = [n for n in vs if n.startswith("t_i")]
        if (
            (nm.startswith("underestimate_pow") or nm.startswith("overestimate_pow"))
            and len(auxs) == 1
            and len(origs) == 1
        ):
            i = _orig_index(origs[0])
            aux_identity[auxs[0]] = ("pow", i)
            pow_of[auxs[0]] = i
        elif (
            (nm.startswith("underestimate_prod") or nm.startswith("overestimate_prod"))
            and len(auxs) == 1
            and len(origs) == 2
        ):
            i, j = sorted(_orig_index(o) for o in origs)
            aux_identity[auxs[0]] = ("prod", i, j)

    # (c) decode via SCIP's ``minor_<pow_A>_<pow_B>_<prod_C>_*`` rows. These are the
    # 2x2 principal-minor PSD cuts on the moment matrix of a variable pair (a, b):
    # by construction pow_A = a^2, pow_B = b^2, prod_C = a*b. So each such row is a
    # constraint linking the three, and any one identity determines the pair for the
    # other two. Iterate to a fixpoint over the collected triples.
    minor_re = re.compile(r"minor_(auxvar_pow_\d+)_(auxvar_pow_\d+)_(auxvar_prod_\d+)_")
    triples = []
    for r in rows:
        mo = minor_re.match(r.name)
        if mo:
            triples.append((mo.group(1), mo.group(2), mo.group(3)))

    def _propagate_triples():
        changed = True
        while changed:
            changed = False
            for pa, pb, prod in triples:
                ia = pow_of.get(pa)
                ib = pow_of.get(pb)
                pident = aux_identity.get(prod)
                # forward: both pows known -> prod = a*b
                if ia is not None and ib is not None and prod not in aux_identity:
                    aux_identity[prod] = ("prod", *sorted((ia, ib)))
                    changed = True
                # backward: prod known as (i,j) -> pows are i^2 and j^2.
                if pident is not None and pident[0] == "prod":
                    _, i, j = pident
                    pair = {i, j}
                    known = {p: pow_of[p] for p in (pa, pb) if p in pow_of}
                    unknown = [p for p in (pa, pb) if p not in pow_of]
                    remaining = sorted(pair - set(known.values()))
                    if len(unknown) == 1 and len(remaining) == 1:
                        pow_of[unknown[0]] = remaining[0]
                        aux_identity[unknown[0]] = ("pow", remaining[0])
                        changed = True

    _propagate_triples()

    # (d) Numeric anchor: solve the SCIP root LP once, read auxvar + original LP
    # values, and for every still-unresolved prod/pow auxvar find the monomial whose
    # value at the LP point matches the auxvar value. At the relaxation optimum SCIP
    # pins many aux columns to their defining product (the minor/envelope rows are
    # tight), so this resolves the auxvars the structural decoders (a)-(c) missed
    # without depending on SCIP-internal expression introspection.
    try:
        sol = m.getBestSol() if m.getNSols() > 0 else None
        tvals = {}
        avals = {}
        for v in m.getVars(transformed=True):
            try:
                val = m.getSolVal(sol, v) if sol is not None else v.getLPSol()
            except Exception:
                continue
            if v.name.startswith("t_i"):
                tvals[_orig_index(v.name)] = val
            elif v.name.startswith("auxvar_prod_") or v.name.startswith("auxvar_pow_"):
                avals[v.name] = val
        idxs = sorted(tvals)
        for a, val in avals.items():
            if a in aux_identity:
                continue
            best = None
            # squares
            for i in idxs:
                if abs(tvals[i] * tvals[i] - val) < 1e-4 * (1 + abs(val)):
                    best = ("pow", i)
            # products (prefer prod for auxvar_prod_*, pow for auxvar_pow_*)
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    i, j = idxs[ii], idxs[jj]
                    if abs(tvals[i] * tvals[j] - val) < 1e-4 * (1 + abs(val)) and a.startswith(
                        "auxvar_prod_"
                    ):
                        best = ("prod", i, j)
            if best is not None:
                aux_identity[a] = best
                if best[0] == "pow":
                    pow_of[a] = best[1]
        _propagate_triples()  # let numeric anchors chain through the minor triples
    except Exception:
        pass

    # Extract cmir/agg cut rows in <= form (rhs finite side).
    cut_rows = []
    for r in rows:
        base = "".join(ch for ch in r.name if not ch.isdigit())
        if not (base.startswith("cmir") or base.startswith("agg")):
            continue
        cols, vals = r.getCols(), r.getVals()
        coef = {c.getVar().name: float(v) for c, v in zip(cols, vals, strict=False)}
        const = float(r.getConstant())
        lhs, rhs = float(r.getLhs()), float(r.getRhs())
        # normalize to  a.x <= b   (move constant to rhs; flip >= rows).
        if rhs < 1e19:  # a.x + const <= rhs
            cut_rows.append((r.name, coef, rhs - const))
        elif lhs > -1e19:  # a.x + const >= lhs  ->  -a.x <= -(lhs-const)
            cut_rows.append((r.name, {k: -v for k, v in coef.items()}, -(lhs - const)))

    return scip_bounds, cut_rows, aux_identity, scip_root, trivial


# --------------------------------------------------------------------------- #
# discopt side: build lifted LP at a box, map+append cuts, solve
# --------------------------------------------------------------------------- #
def _map_cut_to_discopt(coef, rhs, aux_identity, varmap, n_total):
    """Map a SCIP cut (over t_i + auxvars) to a length-n_total discopt row.

    Returns (row, rhs, unmapped) where unmapped counts terms we could not place.
    """
    orig = varmap["original"]  # {orig_idx: col}
    bil = varmap["bilinear"]  # {(i,j): col}
    mon = varmap["monomial"]  # {(i,2): col}
    row = np.zeros(n_total, dtype=np.float64)
    unmapped = 0
    for vname, c in coef.items():
        if vname.startswith("t_i"):
            mo = re.match(r"t_i(\d+)", vname)
            i = int(mo.group(1)) - 1
            col = orig.get(i)
            if col is None:
                unmapped += 1
                continue
            row[col] += c
        elif vname.startswith("auxvar"):
            ident = aux_identity.get(vname)
            if ident is None:
                unmapped += 1
                continue
            if ident[0] == "prod":
                _, i, j = ident
                col = bil.get((i, j)) or bil.get((j, i))
                if col is None:
                    unmapped += 1
                    continue
                row[col] += c
            elif ident[0] == "pow":
                _, i = ident
                col = mon.get((i, 2))
                if col is None:
                    unmapped += 1
                    continue
                row[col] += c
            else:
                unmapped += 1
        else:  # slack / objective aux we cannot place
            unmapped += 1
    return row, rhs, unmapped


def _solve_lifted(milp):
    """Solve a MilpRelaxationModel as a pure LP; return lower bound or None."""
    milp._integrality = None
    res = milp.solve(time_limit=CAP_SECONDS, backend="simplex")
    # MilpRelaxationResult: 'bound' is the rigorous dual LB; for a pure LP it equals
    # 'objective'. Prefer 'bound', fall back to 'objective'.
    lb = getattr(res, "bound", None)
    if lb is None:
        lb = getattr(res, "objective", None)
    st = getattr(res, "status", "?")
    return (float(lb) if lb is not None else None), st


def run_instance(name: str) -> dict:
    import discopt.modeling as dm
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    row: dict = {"instance": name, "opt": OPT.get(name)}
    t0 = time.monotonic()

    scip_bounds, cut_rows, aux_identity, scip_root, trivial = _scip_root(name)
    row["scip_root_bound"] = scip_root
    row["scip_trivial"] = trivial
    row["n_scip_cmir_cuts"] = len(cut_rows)

    m = dm.from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(m)
    r = MccormickLPRelaxer(m, backend="simplex")

    # Bound box = SCIP's presolved integer bounds where available, else declared.
    nvar = len(m._variables)
    nlb = np.array([v.lb for v in m._variables], dtype=np.float64)
    nub = np.array([v.ub for v in m._variables], dtype=np.float64)
    for k, (lo, hi) in scip_bounds.items():
        mo = re.match(r"t_i(\d+)", k)
        idx = int(mo.group(1)) - 1
        if 0 <= idx < nvar:
            nlb[idx] = max(nlb[idx], lo)
            nub[idx] = min(nub[idx], hi)
    row["box_lb"] = nlb.tolist()
    row["box_ub"] = nub.tolist()

    milp, varmap = build_milp_relaxation(m, terms, r._disc, bound_override=(nlb, nub))
    n_total = len(milp._c)
    row["n_lifted_cols"] = n_total

    root_off, st_off = _solve_lifted(milp)
    row["root_off"] = root_off
    row["root_off_status"] = st_off

    # Map cuts.
    mapped_rows = []
    mapped_rhs = []
    total_unmapped = 0
    n_fully_mapped = 0
    for _cname, coef, rhs in cut_rows:
        rr, b, unmapped = _map_cut_to_discopt(coef, rhs, aux_identity, varmap, n_total)
        total_unmapped += unmapped
        if unmapped == 0 and np.any(np.abs(rr) > 1e-12):
            mapped_rows.append(rr)
            mapped_rhs.append(b)
            n_fully_mapped += 1
    row["n_cuts_fully_mapped"] = n_fully_mapped
    row["n_unmapped_terms"] = total_unmapped

    # Rebuild a fresh relaxation (solve() may mutate) and append mapped cut rows.
    milp2, varmap2 = build_milp_relaxation(m, terms, r._disc, bound_override=(nlb, nub))
    if mapped_rows:
        A_new = np.vstack(mapped_rows)
        b_new = np.array(mapped_rhs, dtype=np.float64)
        _append_rows(milp2, A_new, b_new)
    root_on, st_on = _solve_lifted(milp2)
    row["root_on"] = root_on
    row["root_on_status"] = st_on

    opt = OPT.get(name)
    if root_off is not None and root_on is not None and opt is not None:
        denom = opt - root_off
        row["gap_closed"] = (root_on - root_off) / denom if abs(denom) > 1e-9 else 1.0
        row["delta_bound"] = root_on - root_off
    else:
        row["gap_closed"] = None
        row["delta_bound"] = None

    # --- SEPARATED-relaxation measurement (the DECISIVE one) ---
    # The bare build_milp_relaxation LP above lacks discopt's default per-node
    # separation chain (multilinear-hull / edge-concave / RLT / PSD). The real
    # question is whether SCIP's c-MIR cuts help discopt's *actual* relaxation, so
    # inject the same mapped cut rows through solve_at_node's ``inherited_cuts`` seam
    # (which runs the full separation chain, then appends the cut pool) on the same
    # SCIP-tightened box.
    try:
        sep_off = r.solve_at_node(nlb, nub, time_limit=CAP_SECONDS, separate=True)
        row["sep_root_off"] = (
            float(sep_off.lower_bound) if sep_off.lower_bound is not None else None
        )
        row["sep_root_off_status"] = sep_off.status
        inj = None
        if mapped_rows:
            A_new = np.vstack(mapped_rows)
            b_new = np.array(mapped_rhs, dtype=np.float64)
            inj = (A_new, b_new)
        sep_on = r.solve_at_node(
            nlb, nub, time_limit=CAP_SECONDS, separate=True, inherited_cuts=inj
        )
        row["sep_root_on"] = float(sep_on.lower_bound) if sep_on.lower_bound is not None else None
        row["sep_root_on_status"] = sep_on.status
        if row["sep_root_off"] is not None and row["sep_root_on"] is not None and opt is not None:
            d2 = opt - row["sep_root_off"]
            row["sep_delta_bound"] = row["sep_root_on"] - row["sep_root_off"]
            row["sep_gap_closed"] = (
                (row["sep_root_on"] - row["sep_root_off"]) / d2 if abs(d2) > 1e-9 else 1.0
            )
    except Exception as exc:  # noqa: BLE001
        row["sep_note"] = f"{type(exc).__name__}: {exc}"

    row["wall"] = time.monotonic() - t0
    return row


def _append_rows(milp, A_new, b_new):
    """Append a.x <= b rows to a MilpRelaxationModel's inequality system."""
    import scipy.sparse as sp

    A_ub = milp._A_ub
    b_ub = milp._b_ub
    A_new = np.asarray(A_new, dtype=np.float64)
    b_new = np.asarray(b_new, dtype=np.float64).ravel()
    if A_ub is None:
        milp._A_ub = A_new
        milp._b_ub = b_new
        return
    if sp.issparse(A_ub):
        milp._A_ub = sp.vstack([A_ub, sp.csr_matrix(A_new)]).tocsr()
        milp._b_ub = np.concatenate([np.asarray(b_ub).ravel(), b_new])
    else:
        milp._A_ub = np.vstack([np.asarray(A_ub), A_new])
        milp._b_ub = np.concatenate([np.asarray(b_ub).ravel(), b_new])


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"CUT-1 — SCIP c-MIR oracle injection over {PANEL}\n")
    hdr = (
        f"{'instance':<8} {'opt':>10} {'root_off':>12} {'root_on':>12} "
        f"{'Δbound':>10} {'gap_cl':>8} {'#cuts':>6} {'#mapped':>8} {'#unmap':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in PANEL:
        try:
            row = run_instance(name)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            row = {"instance": name, "status": "error", "note": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

        def _f(x, w=12):
            return f"{x:>{w}.4f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"

        print(
            f"{row.get('instance', ''):<8} {_f(row.get('opt'), 10)} "
            f"{_f(row.get('root_off'))} {_f(row.get('root_on'))} "
            f"{_f(row.get('delta_bound'), 10)} {_f(row.get('gap_closed'), 8)} "
            f"{str(row.get('n_scip_cmir_cuts', '-')):>6} "
            f"{str(row.get('n_cuts_fully_mapped', '-')):>8} "
            f"{str(row.get('n_unmapped_terms', '-')):>7}"
        )
        print(
            f"         [separated relaxation] sep_off={_f(row.get('sep_root_off'), 12)} "
            f"sep_on={_f(row.get('sep_root_on'), 12)} "
            f"sepΔ={_f(row.get('sep_delta_bound'), 8)} "
            f"sep_gap_cl={_f(row.get('sep_gap_closed'), 8)}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"cut1_cmir_oracle_injection_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "task": "CUT-1",
                "generated": stamp,
                "panel": PANEL,
                "opt": OPT,
                "cap_seconds": CAP_SECONDS,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nPersisted -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
