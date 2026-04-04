"""Adversary test: Hock-Schittkowski Problem 35
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming Codes",
        LNEMS 187, 1981, Problem 35.
Known optimal: 1/9 = 0.111111...
Problem class: convex NLP

min  9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1*x2 + 2x1*x3
s.t. x1 + x2 + 2*x3 <= 3
     0 <= x1, x2, x3
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm

m = dm.Model("hs035")
x1 = m.continuous("x1", lb=0, ub=10)
x2 = m.continuous("x2", lb=0, ub=10)
x3 = m.continuous("x3", lb=0, ub=10)

m.minimize(
    9
    - 8 * x1
    - 6 * x2
    - 4 * x3
    + 2 * x1**2
    + 2 * x2**2
    + x3**2
    + 2 * x1 * x2
    + 2 * x1 * x3
)

m.subject_to(x1 + x2 + 2 * x3 <= 3)

KNOWN_OPTIMAL = 1.0 / 9.0  # 0.111111...

# --- Solve with all three backends ---
backends = ["ipm", "ipopt", "ripopt"]
results = {}

for backend in backends:
    try:
        r = m.solve(time_limit=120, nlp_solver=backend)
        results[backend] = r
        rel_err = (
            abs(r.objective - KNOWN_OPTIMAL) / max(1, abs(KNOWN_OPTIMAL))
            if r.objective is not None
            else float("inf")
        )
        print(
            f"{backend:>6}: obj={r.objective}  "
            f"time={r.wall_time:.3f}s  "
            f"nodes={r.node_count}  "
            f"status={r.status}  "
            f"rel_err={rel_err:.2e}"
        )
    except Exception as e:
        print(f"{backend:>6}: FAILED - {e}")
        results[backend] = None

print(f"\nKnown optimal: {KNOWN_OPTIMAL:.10f}")
for backend, r in results.items():
    if r is not None and r.objective is not None:
        rel_err = abs(r.objective - KNOWN_OPTIMAL) / max(
            1, abs(KNOWN_OPTIMAL)
        )
        print(f"{backend:>6}: {'PASS' if rel_err < 1e-3 else 'FAIL'}")
    else:
        print(f"{backend:>6}: NO SOLUTION")

if results.get("ipm") and results["ipm"].x:
    r = results["ipm"]
    print(f"\nSolution: x1={r.x['x1']:.6f} x2={r.x['x2']:.6f} x3={r.x['x3']:.6f}")
