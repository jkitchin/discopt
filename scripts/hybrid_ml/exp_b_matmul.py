"""Exp B: can the DAG express matrix-shaped NN layers with trainable weights?

Probe 1: Variable @ Variable (bilinear matmul) in a constraint.
Probe 2: the actual NN-layer pattern: tanh(expr) @ Variable where expr has a
         broadcast trailing axis — i.e. a batched dense layer over (nfe, ncp) points.
"""

import discopt.modeling as dm
from discopt.modeling import Model
from discopt.solvers.nlp_pounce import solve_nlp_from_model

# ---- probe 1: plain Variable @ Variable ----------------------------------
try:
    m = Model()
    A = m.continuous("A", shape=(2, 3), lb=-2, ub=2)
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", shape=(2,), lb=-5, ub=5)
    m.subject_to(A @ x == y)
    m.minimize(dm.sum((y - 1.0) ** 2) + dm.sum(A**2) + dm.sum(x**2))
    r = solve_nlp_from_model(m, options={"print_level": 0})
    print(f"probe1 Variable@Variable : {r.status.name}  obj={r.objective:.4e}")
except Exception as e:  # noqa: BLE001
    print(f"probe1 Variable@Variable : FAILED — {type(e).__name__}: {e}")

# ---- probe 2: batched dense layer over collocation-shaped expression ------
try:
    m = Model()
    nfe, ncp, nh = 4, 2, 5
    c = m.continuous("c", shape=(nfe, ncp), lb=0.0, ub=1.5)
    W1 = m.continuous("W1", shape=(1, nh), lb=-8, ub=8)
    b1 = m.continuous("b1", shape=(nh,), lb=-8, ub=8)
    W2 = m.continuous("W2", shape=(nh, 1), lb=-8, ub=8)

    h = dm.tanh(c[:, :, None] @ W1 + b1)  # (nfe, ncp, 1) @ (1, nh) -> (nfe, ncp, nh)
    out = h @ W2  # (nfe, ncp, nh) @ (nh, 1) -> (nfe, ncp, 1)
    m.subject_to(out[:, :, 0] == 0.5 * c)
    m.minimize(dm.sum(W1**2) + dm.sum(b1**2) + dm.sum(W2**2) + dm.sum((c - 0.7) ** 2))
    r = solve_nlp_from_model(m, options={"print_level": 0})
    print(f"probe2 batched NN layer  : {r.status.name}  obj={r.objective:.4e}")
except Exception as e:  # noqa: BLE001
    print(f"probe2 batched NN layer  : FAILED — {type(e).__name__}: {e}")
