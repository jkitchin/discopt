"""Issue #661 entry experiment: Shor SDP dual bound on qap.

Measures whether the global semidefinite (Shor) relaxation produces a useful dual
bound on qap, compared to McCormick (~0), RLT-1 (352891, LP), oracle dual (149106),
and optimum (388214). LOCAL scratch script -- does not touch production solver code.
"""

import itertools
import json
import time
import sys

import numpy as np
import cvxpy as cp

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/qap.nl"

OPT = 388214.0
BEST_DUAL = 149106.0
MCCORMICK = 0.0
RLT1_LP = 352890.9

results = {"instance": "qap", "opt": OPT, "best_dual": BEST_DUAL,
           "mccormick": MCCORMICK, "rlt1_lp_gauge": RLT1_LP, "runs": {}}


def extract_qap():
    """Load qap.nl and extract (Q, c_lin, offset, A_eq, b_eq, binary_vars)."""
    import discopt.modeling as dm
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.rlt import _reconstruct_quadratic_objective
    from discopt._jax.obbt import _extract_linear_constraints
    from discopt._jax.model_utils import binary_flat_cols

    model = dm.from_nl(NL)
    terms = classify_nonlinear_terms(model)
    relax, info = build_milp_relaxation(model, terms, DiscretizationState())
    n = len(model._variables)
    recon = _reconstruct_quadratic_objective(relax, info, n)
    assert recon is not None, "objective not a pure quadratic"
    Q, c_lin, offset = recon
    A_ub, b_ub, A_eq, b_eq, n2 = _extract_linear_constraints(model)
    assert n2 == n
    binary_vars = frozenset(binary_flat_cols(model))
    A_eq = np.asarray(A_eq.todense()) if hasattr(A_eq, "todense") else np.asarray(A_eq)
    b_eq = np.asarray(b_eq, dtype=float)
    return Q, c_lin, offset, A_eq, b_eq, binary_vars, n


def solve_shor(Q, c_lin, offset, A_eq, b_eq, n, *, add_rlt1=False, add_gangster=False,
               solver="SCS", time_limit=600):
    """Solve the Shor SDP relaxation of a binary QP.

    min <Q,X> + c'x + offset
    s.t.  [[1, x'],[x, X]] >= 0 (PSD),  diag(X)=x,  0<=x<=1,  A_eq x = b_eq
    optional RLT-1:  A_eq_row . X[:,p] = b x_p    for each equality & var p
    optional gangster: X_ij = 0 for mutually exclusive assignment pairs
    """
    x = cp.Variable(n)
    X = cp.Variable((n, n), symmetric=True)
    M = cp.bmat([[np.array([[1.0]]), cp.reshape(x, (1, n))],
                 [cp.reshape(x, (n, 1)), X]])
    cons = [M >> 0, cp.diag(X) == x, x >= 0, x <= 1, A_eq @ x == b_eq]

    if add_rlt1:
        # lifted equality: for each equality row a, (a . X[:,p]) = b_r * x_p
        for r in range(A_eq.shape[0]):
            a = A_eq[r]
            cons.append(A_eq[r] @ X == b_eq[r] * x)
        # McCormick / RLT bound-factor rows on X (redundant with PSD+diag partly,
        # but cheap and tightening): 0 <= X_ij <= min(x_i,x_j), X_ij>=x_i+x_j-1
        cons.append(X >= 0)
        cons.append(X <= cp.reshape(x, (n, 1)) @ np.ones((1, n)))
        cons.append(X <= np.ones((n, 1)) @ cp.reshape(x, (1, n)))

    if add_gangster:
        # exclusive pairs: two binaries both in an equality sum_k x_k = 1 -> X_ij=0
        zero_pairs = set()
        for r in range(A_eq.shape[0]):
            supp = [k for k in range(n) if abs(A_eq[r, k] - 1.0) < 1e-9]
            if abs(b_eq[r] - 1.0) < 1e-9 and all(abs(A_eq[r, k]) < 1e-9 or abs(A_eq[r, k] - 1.0) < 1e-9 for k in range(n)):
                for i, j in itertools.combinations(supp, 2):
                    zero_pairs.add((min(i, j), max(i, j)))
        for (i, j) in zero_pairs:
            cons.append(X[i, j] == 0)

    obj = cp.Minimize(cp.trace(Q @ X) + c_lin @ x + offset)
    prob = cp.Problem(obj, cons)
    t0 = time.time()
    kw = {"verbose": False}
    if solver == "SCS":
        kw.update(max_iters=20000, eps=1e-5, time_limit_secs=time_limit)
    prob.solve(solver=getattr(cp, solver), **kw)
    dt = time.time() - t0
    return prob.value, prob.status, dt


def brute_qap(F, D):
    """Brute force Koopmans-Beckmann QAP min over permutations. Returns (opt, Q, n2)."""
    n = F.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(n)):
        c = 0.0
        for i in range(n):
            for j in range(n):
                c += F[i, j] * D[perm[i], perm[j]]
        best = min(best, c)
    return best


def synthetic_experiment():
    """Small n=4 Koopmans-Beckmann QAP: brute opt vs McCormick(0) vs Shor vs Shor+RLT1."""
    rng = np.random.default_rng(0)
    out = []
    for n in (4, 5):
        F = rng.integers(0, 10, size=(n, n)).astype(float)
        np.fill_diagonal(F, 0)
        D = rng.integers(0, 10, size=(n, n)).astype(float)
        D = (D + D.T)
        np.fill_diagonal(D, 0)
        opt = brute_qap(F, D)
        # variables x_{i,k} = facility i at location k, flat index i*n+k
        N = n * n
        Q = np.zeros((N, N))
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    for l in range(n):
                        Q[i * n + k, j * n + l] += F[i, j] * D[k, l]
        Q = 0.5 * (Q + Q.T)
        c_lin = np.zeros(N)
        # assignment: each facility one loc, each loc one facility
        rows = []
        b = []
        for i in range(n):
            r = np.zeros(N); r[i * n:(i + 1) * n] = 1.0; rows.append(r); b.append(1.0)
        for k in range(n):
            r = np.zeros(N)
            for i in range(n):
                r[i * n + k] = 1.0
            rows.append(r); b.append(1.0)
        A_eq = np.array(rows); b_eq = np.array(b)
        shor, st1, dt1 = solve_shor(Q, c_lin, 0.0, A_eq, b_eq, N, solver="CLARABEL")
        shor_rlt, st2, dt2 = solve_shor(Q, c_lin, 0.0, A_eq, b_eq, N,
                                        add_rlt1=True, add_gangster=True, solver="CLARABEL")
        out.append({"n": n, "opt": float(opt), "mccormick": 0.0,
                    "shor": float(shor), "shor_status": st1,
                    "shor_rlt1_gangster": float(shor_rlt), "shor_rlt1_status": st2})
        print(f"synthetic n={n}: opt={opt:.1f} McC=0 Shor={shor:.2f}({st1}) "
              f"Shor+RLT1+gang={shor_rlt:.2f}({st2})")
    return out


if __name__ == "__main__":
    print("=== synthetic small-QAP validation ===")
    results["synthetic"] = synthetic_experiment()

    print("\n=== real qap (n=15, 225 binaries) ===")
    Q, c_lin, offset, A_eq, b_eq, binary_vars, n = extract_qap()
    print(f"n={n}, A_eq {A_eq.shape}, offset={offset}, "
          f"eig(Q) in [{np.linalg.eigvalsh(Q).min():.0f}, {np.linalg.eigvalsh(Q).max():.0f}]")
    results["qap_meta"] = {"n": n, "n_eq": int(A_eq.shape[0]),
                           "offset": float(offset),
                           "eig_min": float(np.linalg.eigvalsh(Q).min()),
                           "eig_max": float(np.linalg.eigvalsh(Q).max())}

    for tag, kw in [("shor_plain", {}),
                    ("shor_rlt1_gangster", {"add_rlt1": True, "add_gangster": True})]:
        for solver in ("CLARABEL", "SCS"):
            try:
                print(f"solving qap {tag} with {solver} ...", flush=True)
                val, st, dt = solve_shor(Q, c_lin, offset, A_eq, b_eq, n, solver=solver, **kw)
                print(f"  {tag}/{solver}: bound={val} status={st} time={dt:.1f}s")
                results["runs"][f"{tag}_{solver}"] = {
                    "bound": (float(val) if val is not None else None),
                    "status": st, "time_s": dt}
                if st in ("optimal", "optimal_inaccurate") and val is not None:
                    break  # got a usable value, don't need the other solver
            except Exception as e:
                print(f"  {tag}/{solver} FAILED: {e}")
                results["runs"][f"{tag}_{solver}"] = {"error": str(e)}

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "qap_shor_results.json", "w") as f:
        json.dump(results, f, indent=2)
