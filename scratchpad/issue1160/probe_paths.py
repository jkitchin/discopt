"""Which extractor claims the axis-sum model, and how many rows does it emit?

Prints an executed-check count and exits non-zero if nothing was checked.
"""
import numpy as np
import discopt.modeling.core as dm
from discopt._relax import problem_classifier as pc

checks = 0

def build_linear():
    m = dm.Model("sum_axis")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(-dm.sum(A))
    return m

m = build_linear()
print("class:", pc.classify_problem(m))
checks += 1

try:
    terms, const = pc._extract_linear_coefficients_sparse(m._constraints[0].body, m, 6)
    print("algebraic row terms:", terms, "const:", const)
    checks += 1
except Exception as exc:
    print("algebraic refused:", type(exc).__name__, exc)
    checks += 1

try:
    lp = pc.extract_lp_data_algebraic(m)
    print("algebraic LP A_eq shape:", np.asarray(pc.dense_A(lp.A_eq)).shape)
except Exception as exc:
    print("algebraic LP refused:", type(exc).__name__, exc)
checks += 1

lp2 = pc.extract_lp_data(m)
print("extract_lp_data A_eq shape:", np.asarray(pc.dense_A(lp2.A_eq)).shape, "b:", lp2.b_eq)
checks += 1

tape = pc._extract_lp_data_tape(m)
print("tape LP:", None if tape is None else (np.asarray(pc.dense_A(tape.A_eq)).shape, tape.b_eq))
checks += 1

print("CHECKS:", checks)
assert checks > 0
