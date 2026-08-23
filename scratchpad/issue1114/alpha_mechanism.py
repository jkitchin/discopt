"""#1114 mechanism check: WHY does alphaBB abstain on the reduced-space class?

The entry experiment measured 0/52 finite alphaBB bounds on 8 CustomCall models.
This turns that sample into a class statement by calling ``rigorous_alpha``
directly on a CustomCall objective and reporting what it does (raise / +inf /
finite). No exception is swallowed (CLAUDE.md §7): a raise is reported as the
answer, not hidden.
"""

import sys

import numpy as np

import discopt
import discopt.modeling as dm
from discopt._alphabb_rigorous import rigorous_alpha
from discopt._relax.mcbox import MCBox

import jax.numpy as jnp

print(f"discopt.__file__={discopt.__file__}", flush=True)


def _exp(x):
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


checks = 0

# (a) a CustomCall objective -- the class #1114 is about.
m = dm.Model("custom")
x = m.continuous("x", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
f = dm.custom(lambda a, b: a * _exp(-b) + b * _exp(-a), name="f")
m.minimize(f(x[0], x[1]))

# (b) the same function written in NATIVE expression ops -- the control. If (b)
# yields a finite alpha and (a) does not, the abstention is caused by the opaque
# CustomCall node, not by the box or the algebra.
m2 = dm.Model("native")
y = m2.continuous("y", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
m2.minimize(y[0] * dm.exp(-y[1]) + y[1] * dm.exp(-y[0]))

from discopt.solver import _alphabb_node_box  # the real box builder

for label, mm in (("customcall", m), ("native", m2)):
    n = sum(v.size for v in mm._variables)
    box = _alphabb_node_box(mm, np.full(n, 0.1), np.full(n, 2.0))
    expr = mm._objective.expression
    checks += 1
    try:
        alpha = np.asarray(rigorous_alpha(expr, mm, box), dtype=np.float64)
    except Exception as e:  # reported, never swallowed
        print(f"{label:12s} RAISED {type(e).__name__}: {e}", flush=True)
        continue
    finite = bool(np.all(np.isfinite(alpha)))
    print(f"{label:12s} alpha={alpha} all_finite={finite}", flush=True)

print(f"checks={checks}", flush=True)
if checks == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
