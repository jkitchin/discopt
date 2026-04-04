"""Adversary test: Hock-Schittkowski Problem 71
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming Codes",
        Lecture Notes in Economics and Mathematical Systems 187, 1981, Problem 71.
Known optimal: 17.0140173
Problem class: nonconvex NLP
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import discopt.modeling as dm

# --- Problem formulation ---
# min  x1*x4*(x1 + x2 + x3) + x3
# s.t. x1*x2*x3*x4 >= 25
#      x1^2 + x2^2 + x3^2 + x4^2 = 40
#      1 <= x1, x2, x3, x4 <= 5

m = dm.Model("hs071")
x1 = m.continuous("x1", lb=1, ub=5)
x2 = m.continuous("x2", lb=1, ub=5)
x3 = m.continuous("x3", lb=1, ub=5)
x4 = m.continuous("x4", lb=1, ub=5)

m.minimize(x1 * x4 * (x1 + x2 + x3) + x3)

# Inequality constraint: x1*x2*x3*x4 >= 25
m.subject_to(x1 * x2 * x3 * x4 >= 25)

# Equality constraint: x1^2 + x2^2 + x3^2 + x4^2 = 40
m.subject_to(x1**2 + x2**2 + x3**2 + x4**2 == 40)

# --- Solve with default settings ---
result = m.solve(time_limit=120)

print(f"Status:    {result.status}")
print(f"Objective: {result.objective}")
print(f"Gap:       {result.gap}")
print(f"Nodes:     {result.node_count}")
print(f"Time:      {result.wall_time:.2f}s")

if result.x is not None:
    print(f"\nSolution:")
    print(f"  x1 = {result.x['x1']:.6f}")
    print(f"  x2 = {result.x['x2']:.6f}")
    print(f"  x3 = {result.x['x3']:.6f}")
    print(f"  x4 = {result.x['x4']:.6f}")

    # Check constraints
    xv = [float(result.x["x1"]), float(result.x["x2"]),
           float(result.x["x3"]), float(result.x["x4"])]
    prod = xv[0] * xv[1] * xv[2] * xv[3]
    sumsq = sum(xi**2 for xi in xv)
    print(f"\nConstraint check:")
    print(f"  x1*x2*x3*x4 = {prod:.6f} (>= 25)")
    print(f"  sum(xi^2)    = {sumsq:.6f} (== 40)")

# --- Compare to known optimal ---
KNOWN_OPTIMAL = 17.0140173
if result.objective is not None:
    rel_err = abs(result.objective - KNOWN_OPTIMAL) / max(1, abs(KNOWN_OPTIMAL))
    print(f"\nKnown optimal: {KNOWN_OPTIMAL}")
    print(f"Relative error: {rel_err:.2e}")
    print("PASS" if rel_err < 1e-3 else "FAIL")
else:
    print("FAIL: no solution found")
