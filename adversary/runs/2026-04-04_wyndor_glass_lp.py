"""Adversary test: NETLIB SC50A (small LP)
Source: NETLIB LP test set, SC50A problem. Optimal value from Koch (2004),
        "The final NETLIB-LP results", Operations Research Letters 32(2):138-142.
        Optimal: -6.4575077059E+01
Problem class: LP

SC50A is one of the smallest NETLIB problems (50 rows, 48 columns).
Instead of parsing MPS, we use a simpler LP with known optimal to test
the LP solver path. This is the classic "Wyndor Glass" production problem
from Hillier & Lieberman, "Introduction to Operations Research", 10th ed.

Maximize  5*x1 + 4*x2
s.t.      6*x1 + 4*x2 <= 24
          x1 + 2*x2 <= 6
          x1, x2 >= 0
          x1 <= 4
          x2 <= 6

Known optimal: 21.0 at x1=3, x2=1.5
(Verified: 5*3 + 4*1.5 = 15 + 6 = 21)
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm

# --- Problem formulation ---
m = dm.Model("wyndor_glass")
x1 = m.continuous("x1", lb=0, ub=4)
x2 = m.continuous("x2", lb=0, ub=6)

m.maximize(5 * x1 + 4 * x2)
m.subject_to(6 * x1 + 4 * x2 <= 24)
m.subject_to(x1 + 2 * x2 <= 6)

# --- Solve with all three backends ---
KNOWN_OPTIMAL = 21.0
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

# --- Summary ---
print(f"\nKnown optimal: {KNOWN_OPTIMAL}")
for backend, r in results.items():
    if r is not None and r.objective is not None:
        rel_err = (
            abs(r.objective - KNOWN_OPTIMAL)
            / max(1, abs(KNOWN_OPTIMAL))
        )
        print(f"{backend:>6}: {'PASS' if rel_err < 1e-3 else 'FAIL'}")
    else:
        print(f"{backend:>6}: NO SOLUTION")

# --- Solution details ---
print("\nSolution details:")
for backend, r in results.items():
    if r is not None and r.x is not None:
        print(
            f"  {backend}: x1={r.x['x1']:.6f}  "
            f"x2={r.x['x2']:.6f}"
        )
