"""Brute-force soundness check for `DISCOPT_OA_INFEASIBLE_NOGOOD` (#1141 items 2/4).

A no-good cut deletes an integer assignment from the master permanently. If the
proof behind it is wrong, the deleted assignment may have held the optimum and the
run returns a false certificate. This enumerates EVERY assignment, solves each
fixed continuous problem independently, takes the true optimum from that
enumeration, and checks the flag ON/OFF answers against it.

Prints an executed-comparison count and exits non-zero on any mismatch (§6).
"""
import argparse, itertools, os, sys
import numpy as np
import discopt.modeling as dm


def build(n, K, seed, cap_scale):
    """Cardinality-constrained min-variance portfolio: convex, and MOST cardinality
    assignments are genuinely infeasible at a tight variance cap."""
    rng = np.random.default_rng(seed)
    F = rng.normal(scale=0.1, size=(n, 3))
    Sigma = F @ F.T / 3 + np.diag(0.01 + 0.02 * rng.random(n))
    mu = 0.10 + 0.02 * rng.random(n)
    cap = cap_scale * float(np.mean(np.diag(Sigma)) / K)
    return Sigma, mu, cap


def model_of(n, K, Sigma, mu, cap, fixed=None):
    m = dm.Model("nogood")
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    if fixed is None:
        b = [m.binary(f"b{i}") for i in range(n)]
    else:
        b = [float(v) for v in fixed]
    quad = 0
    for i in range(n):
        for j in range(n):
            quad = quad + float(Sigma[i, j]) * x[i] * x[j]
    m.subject_to(quad <= cap)
    m.subject_to(sum(x) == 1.0)
    for i in range(n):
        m.subject_to(x[i] - b[i] <= 0.0)
    if fixed is None:
        m.subject_to(sum(b) <= K)
    m.minimize(-sum(float(mu[i]) * x[i] for i in range(n)))
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--K", type=int, default=3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--cap-scale", type=float, default=0.9)
a = ap.parse_args()

compared = 0
bad = 0
for seed in range(a.seeds):
    Sigma, mu, cap = build(a.n, a.K, seed, a.cap_scale)

    # --- ground truth: enumerate every assignment, solve each fixed problem ----
    best = None
    feas_assign = 0
    for bits in itertools.product((0.0, 1.0), repeat=a.n):
        if sum(bits) > a.K:
            continue
        sub = model_of(a.n, a.K, Sigma, mu, cap, fixed=bits)
        r = sub.solve(time_limit=20, gap_tolerance=1e-9)
        if str(r.status) in ("optimal",) and r.objective is not None:
            feas_assign += 1
            if best is None or r.objective < best:
                best = r.objective
    print(f"seed {seed}: {feas_assign} feasible assignments, brute-force optimum "
          f"{best!r}", flush=True)
    assert best is not None, "no feasible assignment; the draw tests nothing"

    for arm, flag in (("off", "0"), ("on", "1")):
        os.environ["DISCOPT_OA_INFEASIBLE_NOGOOD"] = flag
        os.environ["DISCOPT_OA_NODE_CUTS"] = "0"
        m = model_of(a.n, a.K, Sigma, mu, cap)
        r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                    time_limit=60, gap_tolerance=1e-6)
        summary = ((r.mip_nlp_trace or {}).get("summary") or {})
        proven = summary.get("proven_infeasible_assignments")
        compared += 1
        tol = 1e-5 * max(1.0, abs(best))
        ok_obj = r.objective is not None and abs(r.objective - best) <= tol
        ok_bound = r.bound is None or r.bound <= best + tol
        flagged = "" if (ok_obj and ok_bound) else "   <<< WRONG"
        if not (ok_obj and ok_bound):
            bad += 1
        print(f"  {arm:3s} status={r.status} obj={r.objective!r} bound={r.bound!r} "
              f"proven_infeasible={proven}{flagged}", flush=True)

print(f"\nEXECUTED COMPARISONS: {compared}   WRONG: {bad}")
sys.exit(1 if (bad or compared == 0) else 0)
