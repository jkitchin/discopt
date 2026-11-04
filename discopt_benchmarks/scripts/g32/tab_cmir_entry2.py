"""G3.2 entry experiment #2: c-MIR on TABLEAU aggregations (lambda = B^-1 row) with simple+variable bound
substitution, vs plain GMI. Validity witness: each cut must keep the HiGHS-optimal point.
usage: tab_cmir_entry.py <pkl> <mode: cmir|cmir_novub> <rounds>"""
import pickle, sys, math, numpy as np, scipy.sparse as sp, highspy
d = pickle.load(open(sys.argv[1], "rb")); MODE = sys.argv[2]; ROUNDS = int(sys.argv[3])
c = np.asarray(d["c"], float); I = np.asarray(d["integrality"]) > 0; n = len(c)
lb = np.array([-np.inf if b[0] is None else b[0] for b in d["bounds"]], float)
ub = np.array([np.inf if b[1] is None else b[1] for b in d["bounds"]], float); ub[ub >= 1e19] = np.inf
A0 = sp.vstack([d["A_ub"], d["A_eq"]]).tocsr().astype(float)
rlo0 = np.r_[np.full(d["A_ub"].shape[0], -np.inf), d["b_eq"]]; rhi0 = np.r_[d["b_ub"], d["b_eq"]]
from discopt.solvers.milp_highs import solve_milp as hsolve
ref = hsolve(c=c, A_ub=d["A_ub"], b_ub=d["b_ub"], A_eq=d["A_eq"], b_eq=d["b_eq"], bounds=d["bounds"], integrality=d["integrality"], time_limit=60)
xref = np.asarray(ref.x); print("ref", ref.status, ref.objective, flush=True)
# variable upper bounds x_k <= sum U_v y_v (binaries) from original <= rows with rhs 0
vubs = {}
Aub = d["A_ub"].tocsr()
for i in range(Aub.shape[0]):
    cols = Aub.indices[Aub.indptr[i]:Aub.indptr[i+1]]; v = Aub.data[Aub.indptr[i]:Aub.indptr[i+1]]
    r = dict(zip(cols.tolist(), v.tolist())); cont = [j for j in r if not I[j]]
    if d["b_ub"][i] == 0 and len(cont) == 1 and len(r) >= 2 and r[cont[0]] > 0 and lb[cont[0]] >= 0 and all(r[j] < 0 for j in r if j != cont[0]):
        vubs.setdefault(cont[0], []).append({j: -r[j] / r[cont[0]] for j in r if j != cont[0]})
print("VUB rows", sum(len(v) for v in vubs.values()), flush=True)
def cmir(row, beta, x, xl, xu, xint, use_vub):
    """sum row_j x_j <= beta over extended columns; returns best (cut dict, rhs, eff) as  cut.x <= rhs."""
    acc = {}; terms = []  # integer coef accum, continuous (coef_on_s, s-map, const)
    for j, a in row.items():
        if abs(a) < 1e-12: continue
        if xint[j]: acc[j] = acc.get(j, 0.0) + a; continue
        opts = []
        if np.isfinite(xl[j]): opts.append((x[j] - xl[j], 2, "lb", None))
        if np.isfinite(xu[j]): opts.append((xu[j] - x[j], 1, "ub", None))
        if use_vub and j < n:
            for V in vubs.get(j, []): opts.append((sum(U * x[y] for y, U in V.items()) - x[j], 0, "vub", V))
        if not opts: return None
        # HiGHS-style: positive coef wants the upper-type bound (slack gets coef -a <0... ) choose closest
        sl, _, kd, V = min(opts, key=lambda t: (round(t[0], 9), t[1]))
        if kd == "lb": beta -= a * xl[j]; terms.append((a, {j: 1.0}, -xl[j]))
        elif kd == "ub": beta -= a * xu[j]; terms.append((-a, {j: -1.0}, xu[j]))
        else:
            for y, U in V.items(): acc[y] = acc.get(y, 0.0) + a * U
            mm = dict(V); mm[j] = mm.get(j, 0.0) - 1.0; terms.append((-a, mm, 0.0))
    if not acc: return None
    comp = {j: x[j] > 0.5 * (xl[j] + xu[j]) for j in acc}  # binaries/bounded ints
    beta_c = beta - sum(acc[j] * xu[j] for j in acc if comp[j]) - sum(acc[j] * xl[j] for j in acc if not comp[j])
    a_t = {j: (-acc[j] if comp[j] else acc[j]) for j in acc}
    sneg = [t for t in terms if t[0] < 0]
    deltas = set(abs(v) for j, v in a_t.items() if 1e-6 < x[j] - xl[j] and 1e-6 < xu[j] - x[j] and abs(v) > 1e-9)
    deltas |= {1.0}; deltas |= {dd / k for dd in list(deltas) for k in (2, 4, 8)}
    if MODE == "gmi": deltas = {1.0}
    best = None
    for dl in deltas:
        bt = beta_c / dl; f = bt - math.floor(bt)
        if f < 0.01 or f > 0.99: continue
        cut = {}; rhs = math.floor(bt)
        for j, v in a_t.items():
            vv = v / dl; fj = vv - math.floor(vv); g = math.floor(vv) + max(0.0, fj - f) / (1 - f)
            if comp[j]: cut[j] = cut.get(j, 0.0) - g; rhs -= g * xu[j]
            else: cut[j] = cut.get(j, 0.0) + g; rhs += g * xl[j]
        for ck, mm, m0 in sneg:
            w = ck / dl / (1 - f)
            for jj, q in mm.items(): cut[jj] = cut.get(jj, 0.0) + w * q
            rhs -= w * m0
        cut = {j: v * dl for j, v in cut.items() if abs(v) > 1e-12}; rhs *= dl
        act = sum(v * x[j] for j, v in cut.items()); nrm = math.sqrt(sum(v * v for v in cut.values()))
        if nrm == 0 or max(abs(v) for v in cut.values()) > 1e9: continue
        eff = (act - rhs) / nrm
        if eff > 1e-5 and (best is None or eff > best[2]): best = (cut, rhs, eff)
    return best
CA, CB = [], []; nchk = 0
for rnd in range(ROUNDS):
    A = sp.vstack([A0] + ([sp.csr_matrix((np.concatenate([list(cu.values()) for cu in CA]), (np.repeat(np.arange(len(CA)), [len(cu) for cu in CA]), np.concatenate([list(cu.keys()) for cu in CA]).astype(int))), shape=(len(CA), n))] if CA else [])).tocsr()
    rlo = np.r_[rlo0, np.full(len(CB), -np.inf)]; rhi = np.r_[rhi0, CB]; m = A.shape[0]
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf; Ac = A.tocsc()
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = m; lp.col_cost_ = c; lp.col_lower_ = lb; lp.col_upper_ = np.where(np.isinf(ub), inf, ub)
    lp.row_lower_ = np.where(np.isinf(rlo), -inf, rlo); lp.row_upper_ = rhi; lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ac.indptr; lp.a_matrix_.index_ = Ac.indices; lp.a_matrix_.value_ = Ac.data
    st = h.passModel(lp); assert st == highspy.HighsStatus.kOk, st; h.run(); assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal, (h.modelStatusToString(h.getModelStatus()), m, len(rhi), lp.row_upper_[:3])
    x = np.asarray(h.getSolution().col_value); obj = h.getInfo().objective_function_value
    print(f"round {rnd:2d} root LP {obj:.4f} cuts {len(CA)}", flush=True)
    rowact = A @ x
    # extended columns: x (n) then row variables r_i = A_i x with bounds [rlo, rhi]
    xe = np.r_[x, rowact]; xl = np.r_[lb, rlo]; xu = np.r_[ub, rhi]; xint = np.r_[I, np.zeros(m, bool)]
    basics = h.getBasicVariables()[1]
    new = []
    for pos, bv in enumerate(basics):
        if bv < 0 or not I[bv] or min(x[bv] - math.floor(x[bv]), math.ceil(x[bv]) - x[bv]) < 1e-6: continue
        lam = np.asarray(h.getBasisInverseRow(pos)[1])  # lam^T (A x - r) = 0
        lam[np.abs(lam) < 1e-11] = 0.0
        agg = A.T @ lam
        row = {j: float(v) for j, v in enumerate(agg) if abs(v) > 1e-11}
        for i in np.nonzero(lam)[0]: row[n + i] = -float(lam[i])
        for sgn in (1.0, -1.0):
            cu = cmir({j: sgn * v for j, v in row.items()}, 0.0, xe, xl, xu, xint, MODE == "cmir")
            if cu is None: continue
            cut, rhs, eff = cu
            # expand row variables r_i = A_i x
            cx = {}
            for j, v in cut.items():
                if j < n: cx[j] = cx.get(j, 0.0) + v
                else:
                    Ai = A.getrow(j - n)
                    for jj, q in zip(Ai.indices, Ai.data): cx[int(jj)] = cx.get(int(jj), 0.0) + v * q
            ok = True
            for jj in [k for k, v in cx.items() if abs(v) < 1e-9]:  # safe cleanup: relax rhs by the term's min
                v = cx.pop(jj); lo_t = min(v * lb[jj], v * ub[jj])
                if not np.isfinite(lo_t): ok = False; break
                rhs -= lo_t
            if not ok or not cx: continue
            act = sum(v * x[j] for j, v in cx.items()); eff = (act - rhs) / math.sqrt(sum(v * v for v in cx.values()))
            if eff > 1e-5: new.append((cx, rhs, eff))
    print(f'   cand {len(new)} fracbasics {sum(1 for bv in basics if bv>=0 and bv<n and I[bv] and min(x[bv]-math.floor(x[bv]), math.ceil(x[bv])-x[bv])>=1e-6)}', flush=True)
    if not new: break
    new.sort(key=lambda t: -t[2])
    for cx, rhs, eff in new[:int(__import__('os').environ.get('CAP','200'))]:
        viol = sum(v * xref[j] for j, v in cx.items()) - rhs
        assert viol <= 1e-6 * (1 + abs(rhs)) + 1e-7 * sum(abs(v) for v in cx.values()), ("cut removes the MILP optimum", viol); nchk += 1
        CA.append(cx); CB.append(rhs)
print(f"validity checks {nchk}"); assert nchk > 0
