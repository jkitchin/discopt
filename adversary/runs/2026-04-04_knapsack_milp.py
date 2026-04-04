"""Adversary test: 0-1 Knapsack (MILP)
Source: Martello & Toth, "Knapsack Problems", Example 2.1 (adapted).
        5 items, analytically verifiable by enumeration.
Known optimal: 13 (items 1,2,4 selected)
Problem class: MILP
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm

# 5-item 0-1 knapsack
values = [4, 5, 3, 4, 2]
weights = [3, 4, 2, 3, 1]
capacity = 10

m = dm.Model("knapsack")
x = [m.binary(f"x{i}") for i in range(5)]

# Maximize total value
m.maximize(sum(values[i] * x[i] for i in range(5)))

# Weight constraint
m.subject_to(sum(weights[i] * x[i] for i in range(5)) <= capacity)

# Brute force verification
best_val = 0
best_sel = None
for mask in range(2**5):
    sel = [(mask >> i) & 1 for i in range(5)]
    w = sum(weights[i] * sel[i] for i in range(5))
    v = sum(values[i] * sel[i] for i in range(5))
    if w <= capacity and v > best_val:
        best_val = v
        best_sel = sel
print(f"Brute force: value={best_val} selection={best_sel}")
KNOWN_OPTIMAL = float(best_val)

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

print(f"\nKnown optimal: {KNOWN_OPTIMAL}")
for backend, r in results.items():
    if r is not None and r.objective is not None:
        rel_err = abs(r.objective - KNOWN_OPTIMAL) / max(
            1, abs(KNOWN_OPTIMAL)
        )
        print(f"{backend:>6}: {'PASS' if rel_err < 1e-3 else 'FAIL'}")
    else:
        print(f"{backend:>6}: NO SOLUTION")
