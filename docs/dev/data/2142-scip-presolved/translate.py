"""#2142: translate SCIP's transformed (presolved) problem, written as .cip, into a discopt
SpatialKernelSpec over the ORIGINAL spec's column indices.

  python -u translate.py            # fidelity gates F1/F2 (PREREG.md), executed-assertion count

Active SCIP vars <t_xj> keep column j. Fixed / aggregated / negated columns are fixed to [0,0] in
the kernel and recovered by `expand` from the cip's affine maps. Each product in a [nonlinear]
constraint becomes a term (square, kind 1 p=2) or a BLF form; when the constraint is exactly
`coef_w * x_w + coef_p * prod == 0` the existing column w is the output, otherwise a fresh aux
column is appended and the constraint becomes a linear equality."""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIP = os.path.join(HERE, "..", "2142-scip-hull-ablation", "presolved_homog.cip")
SPEC = os.path.join(HERE, "..", "2142-exact-hull", "scip", "cstr_homog_spec.npz")
NUM = r"[0-9]+(?:\.[0-9]*)?(?:e[+-]?[0-9]+)?"
VAR = r"<(~?)t_x(\d+)>(?:\[[CBI]\])?"


def _lin_terms(s):
    """Parse '+0.95<t_x75>[C] -<~t_x6>[B] -0.99' etc. -> (dict col->coef, const). A negated
    var ~x contributes coef*(1 - x)."""
    row, const = {}, 0.0
    pos = 0
    s = s.strip()
    tok = re.compile(rf"\s*([+-]?)\s*({NUM})?\s*\*?\s*(?:{VAR})?")
    while pos < len(s):
        m = tok.match(s, pos)
        assert m and m.end() > pos, (s, pos)
        sign = -1.0 if m.group(1) == "-" else 1.0
        coef = float(m.group(2)) if m.group(2) else None
        if m.group(4) is None:
            assert coef is not None, (s, pos)
            const += sign * coef
        else:
            a = sign * (1.0 if coef is None else coef)
            j = int(m.group(4))
            if m.group(3) == "~":
                const += a
                row[j] = row.get(j, 0.0) - a
            else:
                row[j] = row.get(j, 0.0) + a
        pos = m.end()
    return row, const


def _split_cmp(body):
    m = re.match(r"^(.*?)\s*(<=|==|>=)\s*([+-]?" + NUM + r")\s*;\s*$", body)
    assert m, body
    return m.group(1), m.group(2), float(m.group(3))


def _nl_terms(s):
    """Nonlinear lhs: products '-0.95*<t_x75>*<t_x77>', squares '-(<t_x77>)^2', linear and
    constant terms. Returns (lin dict, const, list of (coef, i, j)) with i == j for squares."""
    prods, lin, const = [], {}, 0.0
    s = s.replace(" ", "")
    tok = re.compile(
        rf"([+-]?)({NUM})?\*?(?:\(<t_x(\d+)>\)\^2|<t_x(\d+)>\*<t_x(\d+)>|<t_x(\d+)>)?"
    )
    pos = 0
    while pos < len(s):
        m = tok.match(s, pos)
        assert m and m.end() > pos, (s, pos)
        sign = -1.0 if m.group(1) == "-" else 1.0
        coef = float(m.group(2)) if m.group(2) else None
        a = sign * (1.0 if coef is None else coef)
        if m.group(3):
            i = int(m.group(3)); prods.append((a, i, i))
        elif m.group(4):
            prods.append((a, int(m.group(4)), int(m.group(5))))
        elif m.group(6):
            j = int(m.group(6)); lin[j] = lin.get(j, 0.0) + a
        else:
            assert coef is not None, (s, pos)
            const += a
        pos = m.end()
    return lin, const, prods


def parse_cip(path=CIP):
    txt = open(path).read()
    vars_sec = txt.split("\nVARIABLES\n")[1].split("\nFIXED\n")[0]
    fixed_sec = txt.split("\nFIXED\n")[1].split("\nCONSTRAINTS\n")[0]
    cons_sec = txt.split("\nCONSTRAINTS\n")[1].split("\nEND")[0]
    vre = re.compile(
        rf"\[(\w+)\] <t_x(\d+)>: obj=([+-]?{NUM}), global bounds=\[([+-]?{NUM}),([+-]?{NUM})\]"
    )
    active = {}
    for line in vars_sec.strip().splitlines():
        m = vre.search(line)
        assert m, line
        active[int(m.group(2))] = dict(kind=m.group(1), obj=float(m.group(3)),
                                       lo=float(m.group(4)), hi=float(m.group(5)))
    # affine maps of eliminated columns: j -> (dict k->a, b) in terms of (possibly eliminated) cols
    aff = {}
    for line in fixed_sec.strip().splitlines():
        m = re.search(r"<t_x(\d+)>: obj=([+-]?" + NUM + r").*?, (fixed|aggregated|negated):\s*(.*)$", line)
        assert m, line
        j, how, rest = int(m.group(1)), m.group(3), m.group(4)
        assert float(m.group(2)) == 0.0, ("eliminated column with objective", line)
        if how == "fixed":
            aff[j] = ({}, float(rest))
        elif how == "negated":
            mm = re.match(rf"({NUM}) - <t_x(\d+)>", rest.strip())
            assert mm, rest
            aff[j] = ({int(mm.group(2)): -1.0}, float(mm.group(1)))
        else:
            row, const = _lin_terms(rest)
            aff[j] = (row, const)
    # resolve chains to active columns
    def resolve(j, depth=0):
        assert depth < 50, j
        if j in active:
            return {j: 1.0}, 0.0
        row, b = aff[j]
        out, cst = {}, b
        for k, a in row.items():
            r2, b2 = resolve(k, depth + 1)
            cst += a * b2
            for kk, aa in r2.items():
                out[kk] = out.get(kk, 0.0) + a * aa
        return out, cst
    amap = {j: resolve(j) for j in aff}
    cons = []
    for line in cons_sec.strip().splitlines():
        m = re.match(r"\s*\[(\w+)\] <([^>]+)>:\s*(.*)$", line)
        assert m, line
        kind, name, body = m.group(1), m.group(2), m.group(3)
        lhs, op, rhs = _split_cmp(body)
        if kind == "nonlinear":
            lin, const, prods = _nl_terms(lhs)
        else:
            assert kind in ("linear", "varbound", "setppc"), kind
            lin, const = _lin_terms(lhs)
            prods = []
        cons.append(dict(kind=kind, name=name, lin=lin, const=const, prods=prods, op=op, rhs=rhs))
    return active, amap, cons


def build_spec(spec0):
    """spec0: dict of the ORIGINAL kernel spec arrays. Returns (spec, expand, compress, info)."""
    active, amap, cons = parse_cip()
    n0 = int(spec0["n_cols"])
    assert all(j < n0 for j in active) and all(j < n0 for j in amap)
    assert set(active) | set(amap) == set(range(n0)), "cip does not cover every original column"
    lo = np.zeros(n0); hi = np.zeros(n0); integ = np.zeros(n0, dtype=np.int64); c = np.zeros(n0)
    for j, v in active.items():
        lo[j], hi[j], c[j] = v["lo"], v["hi"], v["obj"]
        integ[j] = 1 if v["kind"] in ("binary", "integer") else 0
    lo, hi, integ, c = list(lo), list(hi), list(integ), list(c)
    rows = []  # (dict, rhs) meaning row . x <= rhs
    terms = []  # (i, w) w = x_i^2
    blfs = []  # (w, s, i, j)  w = (s x_i) * x_j
    prod_col = {}
    used_out = set(int(w) for w in [])
    n_new = 0

    def add(lin, op, rhs):
        lin = {k: a for k, a in lin.items() if a != 0.0}
        if op in ("<=", "=="):
            rows.append((dict(lin), rhs))
        if op in (">=", "=="):
            rows.append(({k: -a for k, a in lin.items()}, -rhs))

    for cs in cons:
        lin, rhs = dict(cs["lin"]), cs["rhs"] - cs["const"]
        if not cs["prods"]:
            add(lin, cs["op"], rhs)
            continue
        # exact definition  a_w x_w + a_p prod == 0  -> w = (-a_p/a_w) prod, w not yet an output
        if (cs["op"] == "==" and len(cs["prods"]) == 1 and len(lin) == 1 and rhs == 0.0):
            (w, aw), = lin.items()
            ap, i, j = cs["prods"][0]
            s = -ap / aw
            if w not in used_out and w not in (i, j) and integ[w] == 0:
                used_out.add(w)
                if i == j and s == 1.0:
                    terms.append((i, w))
                else:
                    blfs.append((w, s, i, j))
                continue
        for (ap, i, j) in cs["prods"]:
            key = (min(i, j), max(i, j))
            if key not in prod_col:
                w = n0 + n_new; n_new += 1
                prod_col[key] = w
                cand = [lo[i] * lo[j], lo[i] * hi[j], hi[i] * lo[j], hi[i] * hi[j]]
                l_, h_ = min(cand), max(cand)
                if i == j:
                    l_ = 0.0 if lo[i] <= 0.0 <= hi[i] else min(lo[i] ** 2, hi[i] ** 2)
                    h_ = max(lo[i] ** 2, hi[i] ** 2)
                lo.append(l_); hi.append(h_); integ.append(0); c.append(0.0)
                if i == j:
                    terms.append((i, w))
                else:
                    blfs.append((w, 1.0, i, j))
            w = prod_col[key]
            lin[w] = lin.get(w, 0.0) + ap
        add(lin, cs["op"], rhs)
    n = n0 + n_new
    ptr, cols, coefs, rhs_ = [0], [], [], []
    for r, b in rows:
        cols += list(r); coefs += list(r.values()); rhs_.append(b); ptr.append(len(cols))
    idt = np.asarray(spec0["fixed_cols"]).dtype
    spec = dict(spec0)
    spec.update(
        n_cols=n, n_orig=int(spec0["n_orig"]),
        c=np.array(c, float), integrality=np.array(integ, dtype=np.asarray(spec0["integrality"]).dtype),
        global_lo=np.array(lo, float), global_hi=np.array(hi, float),
        fixed_row_ptr=np.array(ptr, dtype=idt), fixed_cols=np.array(cols, dtype=idt),
        fixed_coeffs=np.array(coefs, float), fixed_rhs=np.array(rhs_, float),
        term_kind=np.array([1] * len(terms), dtype=idt), term_i=np.array([t[0] for t in terms], dtype=idt),
        term_j=np.array([0] * len(terms), dtype=idt), term_out=np.array([t[1] for t in terms], dtype=idt),
        term_p=np.array([2] * len(terms), dtype=idt), term_coeff=np.ones(len(terms)),
        term_cst=np.zeros(len(terms)),
        blf_w=np.array([b[0] for b in blfs], dtype=idt),
        blf_a_ptr=np.arange(len(blfs) + 1, dtype=idt), blf_a_cols=np.array([b[2] for b in blfs], dtype=idt),
        blf_a_coeffs=np.array([b[1] for b in blfs], float), blf_a_const=np.zeros(len(blfs)),
        blf_b_ptr=np.arange(len(blfs) + 1, dtype=idt), blf_b_cols=np.array([b[3] for b in blfs], dtype=idt),
        blf_b_coeffs=np.ones(len(blfs)), blf_b_const=np.zeros(len(blfs)),
    )
    ob0 = set(int(j) for j in np.asarray(spec0["obbt_candidates"]))
    spec["obbt_candidates"] = np.array(
        sorted([j for j in ob0 if j in active] + list(range(n0, n))),
        dtype=np.asarray(spec0["obbt_candidates"]).dtype)

    def expand(x):
        x = np.array(x, dtype=float)
        assert x.size >= n, (x.size, n)  # LP points carry the envelope columns after n
        y = x[:n0].copy()
        for j, (row, b) in amap.items():
            y[j] = b + sum(a * x[k] for k, a in row.items())
        return y

    def compress(y):
        y = np.asarray(y, float)
        x = np.zeros(n)
        x[:n0] = y
        for j in amap:
            x[j] = 0.0
        for (i, j), w in prod_col.items():
            x[w] = y[i] * y[j]
        return x

    info = dict(n_active=len(active), n_elim=len(amap), n_cons=len(cons), n_rows=len(rows),
                n_terms=len(terms), n_blf=len(blfs), n_new_aux=n_new)
    return spec, expand, compress, info, amap


def residual(spec, x):
    sys.path.insert(0, os.path.join(HERE, "..", "2142-hull-empty"))
    import specio
    return specio.residuals(spec, x)["max"]


def scip_solve(spec, tl=600):
    import pyscipopt as scip
    n = int(spec["n_cols"]); lo, hi, integ, c = spec["global_lo"], spec["global_hi"], spec["integrality"], spec["c"]
    m = scip.Model(); m.hideOutput()
    x = [m.addVar(name=f"x{j}", vtype="I" if integ[j] else "C",
                  lb=None if lo[j] <= -1e19 else float(lo[j]), ub=None if hi[j] >= 1e19 else float(hi[j]))
         for j in range(n)]
    m.setObjective(scip.quicksum(float(c[j]) * x[j] for j in range(n) if c[j] != 0.0), "minimize")
    ptr, cols, co, rhs = spec["fixed_row_ptr"], spec["fixed_cols"], spec["fixed_coeffs"], spec["fixed_rhs"]
    for r in range(len(rhs)):
        m.addCons(scip.quicksum(float(co[k]) * x[int(cols[k])] for k in range(int(ptr[r]), int(ptr[r + 1]))) <= float(rhs[r]))
    for k in range(len(spec["term_kind"])):
        kind, i, w, p = (int(spec[q][k]) for q in ("term_kind", "term_i", "term_out", "term_p"))
        assert kind == 1
        m.addCons(x[w] == x[i] ** p)
    for k in range(len(spec["blf_w"])):
        A = float(spec["blf_a_const"][k]) + scip.quicksum(float(spec["blf_a_coeffs"][q]) * x[int(spec["blf_a_cols"][q])] for q in range(int(spec["blf_a_ptr"][k]), int(spec["blf_a_ptr"][k + 1])))
        B = float(spec["blf_b_const"][k]) + scip.quicksum(float(spec["blf_b_coeffs"][q]) * x[int(spec["blf_b_cols"][q])] for q in range(int(spec["blf_b_ptr"][k]), int(spec["blf_b_ptr"][k + 1])))
        m.addCons(x[int(spec["blf_w"][k])] == A * B)
    m.setParam("limits/time", tl); m.setParam("parallel/maxnthreads", 1)
    m.optimize()
    sol = m.getBestSol()
    return m.getStatus(), m.getObjVal(), m.getDualbound(), m.getNNodes(), np.array([m.getSolVal(sol, v) for v in x])


if __name__ == "__main__":
    n_assert = 0
    d = dict(np.load(SPEC))
    spec0 = {k: (int(v) if v.ndim == 0 and k in ("n_cols", "n_orig") else v) for k, v in d.items()}
    spec, expand, compress, info, amap = build_spec(spec0)
    print("translation:", info, flush=True)
    st0, f0, db0, nn0, x0 = scip_solve(spec0)
    print(f"SCIP original: {st0} obj={f0} nodes={nn0} resid_orig={residual(spec0, x0):.2e}", flush=True)
    st1, f1, db1, nn1, x1 = scip_solve(spec)
    print(f"SCIP translated: {st1} obj={f1} dual={db1} nodes={nn1}", flush=True)
    assert st1 == "optimal" and abs(f1 - 3.0620073) <= 1e-5, (st1, f1); n_assert += 1  # F1
    r_exp = residual(spec0, expand(x1))
    print(f"F2a expanded translated optimum on ORIGINAL spec: {r_exp:.2e}")
    assert r_exp <= 1e-5, r_exp; n_assert += 1  # SCIP feastol 1e-6 per row over chained maps (PREREG amended)
    # the aggregation maps hold at the original optimum (they are implied equalities)
    agg_v = max(abs(x0[j] - (b + sum(a * x0[k] for k, a in row.items()))) for j, (row, b) in amap.items())
    print(f"F2b aggregation maps at the original optimum: max |viol| {agg_v:.2e}")
    r_cmp = residual(spec, compress(x0))
    print(f"F2c compressed original optimum on TRANSLATED spec: {r_cmp:.2e}")
    assert r_cmp <= 1e-5, r_cmp; n_assert += 1
    np.savez(os.path.join(HERE, "cstr_homog_scippre_spec.npz"), **{k: np.asarray(v) for k, v in spec.items()})
    print(f"executed assertions: {n_assert}")
    sys.exit(0 if n_assert else 1)
