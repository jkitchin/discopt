"""#2142 §114 step 2: ground truth on the captured hull boxes WITHOUT the 1e24 magnitudes.
Per box (all binaries fixed by the box): fix every integer column and every zero-width column,
then forward-evaluate every lifted column whose operands are all fixed (in float, topological order)
and check it against the box (1e-9 abs + 1e-9 rel) -- a violation there is an exact, checkable
reason the box is empty. The eps-perspective chains of a binary at 0 (w = 1e8, products 1e16/1e24)
are all fixed this way, so the REDUCED model handed to SCIP (default numerics, infinity 1e20) has
only the remaining columns with their box bounds and constants substituted. SCIP (objective 0,
120 s) then searches for a feasible point; a returned point is assembled into the full 320-column
vector and verified by the independent numpy evaluator (specio.residuals <= 1e-6) and box
membership (1e-6 abs + 1e-9 rel). Optionally c^T x <= cutoff (arg 'cut').
  python -u c10_reduced.py BOXES.npz [cut] [b0 b1 ...]"""
import math, os, sys
import numpy as np
import pyscipopt as scip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio  # noqa: E402

d, kw = specio.load(sys.argv[1])
args = sys.argv[2:]
use_cut = bool(args) and args[0] == "cut"
if use_cut:
    args = args[1:]
n = kw["n_cols"]
integ = np.asarray(kw["integrality"]).astype(bool)
ptr, cols, co, rhs = kw["fixed_row_ptr"], kw["fixed_cols"], kw["fixed_coeffs"], kw["fixed_rhs"]
ops = []  # (out, deps, kind, data)
for k in range(len(kw["term_kind"])):
    kind, i, j, w, p = (int(kw[q][k]) for q in ("term_kind", "term_i", "term_j", "term_out", "term_p"))
    ops.append((w, [i, j] if kind == 0 else [i], ("T", kind, i, j, p, float(kw["term_coeff"][k]), float(kw["term_cst"][k]))))
for k in range(len(kw["blf_w"])):
    sa = slice(int(kw["blf_a_ptr"][k]), int(kw["blf_a_ptr"][k + 1]))
    sb = slice(int(kw["blf_b_ptr"][k]), int(kw["blf_b_ptr"][k + 1]))
    A = (float(kw["blf_a_const"][k]), kw["blf_a_cols"][sa].tolist(), kw["blf_a_coeffs"][sa].tolist())
    B = (float(kw["blf_b_const"][k]), kw["blf_b_cols"][sb].tolist(), kw["blf_b_coeffs"][sb].tolist())
    ops.append((int(kw["blf_w"][k]), A[1] + B[1], ("B", A, B)))
outs = {o for o, _, _ in ops}
order, done = [], set(range(n)) - outs
while len(order) < len(ops):
    prog = False
    for op in ops:
        if op[0] not in done and set(op[1]) <= done:
            order.append(op); done.add(op[0]); prog = True
    assert prog


def ev(data, val):
    """val(j) -> number or SCIP expr."""
    if data[0] == "T":
        _, kind, i, j, p, a, b = data
        if kind == 0:
            return val(i) * val(j)
        if kind == 1:
            return val(i) ** p
        if kind == 2:
            return (a * val(i) + b) ** 2
        if kind == 3:
            v = a * val(i) + b
            return math.sqrt(v) if isinstance(v, float) else scip.sqrt(v)
        return (a * val(i) + b) ** p
    _, A, B = data
    fa = A[0] + sum(c_ * val(j) for j, c_ in zip(A[1], A[2]))
    fb = B[0] + sum(c_ * val(j) for j, c_ in zip(B[1], B[2]))
    return fa * fb


nb = len(d["rec_id"])
boxes = [int(b) for b in args] or list(range(nb))
n_exec = 0
tally = {}
for b in boxes:
    lo0, hi0 = d["rec_pre_lo"][b].astype(float), d["rec_pre_hi"][b].astype(float)
    lo, hi = lo0, hi0
    ilo, ihi = np.ceil(lo[integ] - 1e-6), np.floor(hi[integ] + 1e-6)
    if not np.all(ilo == ihi):
        print(f"box {b}: free binaries, skipped", flush=True); tally["skipped"] = tally.get("skipped", 0) + 1
        continue
    fixed = {}
    for j in np.where(integ)[0]:
        fixed[int(j)] = float(np.ceil(lo[j] - 1e-6))
    for j in range(n):
        if not integ[j] and lo[j] == hi[j] and j not in outs:
            fixed[j] = float(lo[j])
    viol = []
    lo, hi = lo.copy(), hi.copy()
    for _ in range(50):  # singleton-row bound tightening + forward evaluation, to a fixpoint
        nfix = len(fixed)
        for o, deps, data in order:
            if o not in fixed and all(q in fixed for q in deps):
                v = float(ev(data, lambda q: fixed[q]))
                tol = 1e-9 + 1e-9 * abs(v)
                if v < lo[o] - tol or v > hi[o] + tol:
                    viol.append((o, v, lo[o], hi[o]))
                fixed[o] = v
        for r in range(len(rhs)):
            s = slice(int(ptr[r]), int(ptr[r + 1]))
            free_ = [(int(j), float(c_)) for j, c_ in zip(cols[s], co[s]) if int(j) not in fixed]
            if len(free_) != 1:
                continue
            const = sum(float(c_) * fixed[int(j)] for j, c_ in zip(cols[s], co[s]) if int(j) in fixed)
            j, c_ = free_[0]
            bnd = (float(rhs[r]) - const) / c_
            if c_ > 0:
                hi[j] = min(hi[j], bnd)
            else:
                lo[j] = max(lo[j], bnd)
            if j not in outs and hi[j] - lo[j] <= 1e-12 * max(1.0, abs(lo[j])):
                if hi[j] < lo[j] - 1e-9 - 1e-9 * abs(lo[j]):
                    viol.append(("bounds", j, lo[j], hi[j]))
                fixed[j] = 0.5 * (lo[j] + hi[j])
        if len(fixed) == nfix:
            break
    m = scip.Model(); m.hideOutput()
    m.setParam("limits/time", 120.0); m.setParam("parallel/maxnthreads", 1)
    m.setParam("limits/solutions", 1)
    X = {}
    for j in range(n):
        if j in fixed:
            continue
        X[j] = m.addVar(name=f"x{j}", lb=None if lo[j] <= -1e19 else float(lo[j]), ub=None if hi[j] >= 1e19 else float(hi[j]))
    val = lambda q: fixed[q] if q in fixed else X[q]  # noqa: E731
    big = 0.0
    for r in range(len(rhs)):
        s = slice(int(ptr[r]), int(ptr[r + 1]))
        const = sum(float(c_) * fixed[int(j)] for j, c_ in zip(cols[s], co[s]) if int(j) in fixed)
        varpart = [(int(j), float(c_)) for j, c_ in zip(cols[s], co[s]) if int(j) not in fixed]
        big = max(big, abs(const))
        if not varpart:
            if const > rhs[r] + 1e-6 + 1e-9 * abs(const):
                viol.append(("row", r, const, rhs[r]))
            continue
        m.addCons(scip.quicksum(c_ * X[j] for j, c_ in varpart) <= float(rhs[r]) - const)
    for o, deps, data in order:
        if o in fixed:
            continue
        m.addCons(X[o] == ev(data, val))
    if use_cut and math.isfinite(float(d["rec_cutoff"][b])):
        cexpr = sum(float(kw["c"][j]) * val(j) for j in np.where(kw["c"] != 0)[0])
        m.addCons(cexpr <= float(d["rec_cutoff"][b]))
    if viol:
        st, wit = "empty_by_forward_eval", None
    else:
        m.optimize(); st = m.getStatus(); wit = None
        if m.getNSols():
            sol = m.getBestSol()
            x = np.array([fixed[j] if j in fixed else m.getSolVal(sol, X[j]) for j in range(n)])
            r = specio.residuals(kw, x)
            t = 1e-6 + 1e-9 * np.maximum(np.abs(lo), np.abs(hi))
            t = 1e-6 + 1e-9 * np.maximum(np.abs(lo0), np.abs(hi0))
            inbox = bool(np.all(x >= lo0 - t) and np.all(x <= hi0 + t))
            wit = (float(r["max"]), inbox, float(kw["c"] @ x))
            if r["max"] <= 1e-6 and inbox:
                st = "WITNESS"
            else:
                st = st + "_unverified"
    n_exec += 1
    tally[st] = tally.get(st, 0) + 1
    print(f"box {b} id {int(d['rec_id'][b])}: {st} free_cols {len(X)} max|const| {big:.3g} "
          f"forward_viol {viol[:2]} witness {wit}", flush=True)
print("tally", tally, "(cutoff applied)" if use_cut else "(no cutoff)")
print(f"executed box checks: {n_exec}")
sys.exit(0 if n_exec else 1)
