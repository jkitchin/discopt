"""PR #1159 review, finding 2: does the NaN test exercise the branch it guards?

``test_bound_expression_does_not_produce_nan_on_mixed_infinities`` claims plain
float addition "would give -inf + inf = nan". This probe computes what the
UNGUARDED left fold would produce on the test's own input, and on the case the
review proposes instead, so the claim is measured rather than assumed (§6).
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import numpy as np
from discopt._relax.gdp_reformulate import _bound_expression
from discopt.modeling.core import SumOverExpression

checks = 0


def unguarded_fold(terms, model):
    """What the plain ``lo += t_lo`` / ``hi += t_hi`` fold would return."""
    lo = hi = 0.0
    for t in terms:
        t_lo, t_hi = _bound_expression(t, model)
        lo = lo + t_lo
        hi = hi + t_hi
    return lo, hi


# --- the case the test actually uses -----------------------------------------
m = dm.Model("as_written")
lo_free = m.continuous("lo_free", lb=-np.inf, ub=0.0)
hi_free = m.continuous("hi_free", lb=0.0, ub=np.inf)
terms = [lo_free, hi_free]
print("per-term intervals:", [_bound_expression(t, m) for t in terms])
guarded = _bound_expression(SumOverExpression(list(terms)), m)
plain = unguarded_fold(terms, m)
print(f"as written  -> guarded {guarded}   unguarded {plain}")
checks += 1
same = guarded == plain
print(f"  branch exercised? {not same}  (identical results means the test cannot fail)")

# --- the case the review proposes --------------------------------------------
m2 = dm.Model("diverging")
x = m2.continuous("x", lb=-5.0, ub=0.0)      # log(x): (-inf, -inf)
y = m2.continuous("y", lb=0.0, ub=np.inf)    # exp(y): (1, +inf)
terms2 = [dm.log(x), dm.exp(y)]
print("per-term intervals:", [_bound_expression(t, m2) for t in terms2])
guarded2 = _bound_expression(SumOverExpression(list(terms2)), m2)
plain2 = unguarded_fold(terms2, m2)
print(f"proposed    -> guarded {guarded2}   unguarded {plain2}")
checks += 1
print(f"  unguarded produces NaN? {any(np.isnan(v) for v in plain2)}")
print(f"  guarded produces NaN?   {any(np.isnan(v) for v in guarded2)}")

print(f"executed_assertions={checks}")
if checks == 0:
    sys.exit(1)
