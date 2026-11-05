"""#2142 exact-hull, probe 0: what GDPlib cstr's disjunct rows look like, and what the
current hull lowering makes of them (bound magnitudes).

  PYTHONPATH=/home/user/discopt/python python -u p0_structure.py
Prints an executed-assertion count (CLAUDE.md §6)."""
import sys

import numpy as np

import discopt
import discopt._rust as R
import discopt._relax.gdp_reformulate as G
from discopt.modeling.core import (
    BinaryOp, Constant, Constraint, FunctionCall, IndexExpression, UnaryOp, Variable,
    _DisjunctiveConstraint,
)
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert b"DISCOPT_INT_ROUND_EXACT" not in open(R.__file__, "rb").read(), "stale _rust (§114 flag)"
n_assert = 2


def dump(e, depth=0):
    pad = "  " * depth
    if isinstance(e, Variable):
        return f"{pad}Var {e.name} shape={e.shape} lb={np.min(e.lb):g} ub={np.max(e.ub):g}\n"
    if isinstance(e, Constant):
        return f"{pad}Const {np.asarray(e.value).ravel()[:4]}\n"
    if isinstance(e, IndexExpression):
        return f"{pad}Index {e.index}\n" + dump(e.base, depth + 1)
    if isinstance(e, BinaryOp):
        return f"{pad}BinaryOp {e.op!r}\n" + dump(e.left, depth + 1) + dump(e.right, depth + 1)
    if isinstance(e, UnaryOp):
        return f"{pad}UnaryOp {e.op!r}\n" + dump(e.operand, depth + 1)
    if isinstance(e, FunctionCall):
        return f"{pad}Call {e.func_name}\n" + "".join(dump(a, depth + 1) for a in e.args)
    terms = G._sumover_terms(e)
    if terms is not None:
        return f"{pad}SumOver[{len(terms)}]\n" + "".join(dump(t, depth + 1) for t in terms)
    return f"{pad}{type(e).__name__}\n"


m = convert_pyomo_gdp(cstr.build_model()).model
print("objective:", m._objective.sense if hasattr(m._objective, "sense") else "", str(m._objective.expr)[:200] if hasattr(m._objective, "expr") else m._objective)
print("variables:", len(m._variables))
for v in m._variables:
    print(f"  {v.name:12s} {v.var_type.name:10s} shape={v.shape} lb={np.min(v.lb):g} ub={np.max(v.ub):g}")
plan = G._collect_indicator_hull_groups(m)
print("hull groups:", len(plan.groups))
nl_rows = []
for key, (dc, sels) in plan.groups.items():
    print(f"group {key!r}: {len(dc.disjuncts)} disjuncts, selectors {[s.name for s in sels]}")
    for k, d in enumerate(dc.disjuncts):
        for it in d:
            if isinstance(it, Constraint):
                lin = G._is_linear(it.body)
                print(f"   d{k} {'lin' if lin else 'NL '} {it.sense} {str(it.body)[:90]} rhs={it.rhs}")
                if not lin:
                    nl_rows.append((key, k, it))
for c in m._constraints:
    assert not isinstance(c, _DisjunctiveConstraint), "unexpected explicit disjunction"
n_assert += 1
print(f"\nnonlinear disjunct rows: {len(nl_rows)}")
assert len(nl_rows) == 5, len(nl_rows)
n_assert += 1
for key, k, it in nl_rows[:2]:
    print(f"--- group {key!r} disjunct {k} sense {it.sense} rhs {it.rhs}")
    print(dump(it.body), end="")

for method in ("hull", "big-m"):
    low = G.reformulate_gdp(m, method=method)
    mags = [max(abs(float(np.min(v.lb))), abs(float(np.max(v.ub)))) for v in low._variables]
    print(f"\nlowered [{method}]: vars {len(low._variables)} rows {len(low._constraints)} "
          f"max |bound| {max(mags):.3e}")
    nl = [c for c in low._constraints if not G._is_linear(c.body)]
    hull_nl = [c for c in nl if (c.name or "").startswith("_hull_")]
    print(f"  nonlinear rows: {len(nl)} (model's own {len(nl) - len(hull_nl)}, hull {len(hull_nl)})")
    if method == "hull":
        for c in hull_nl[:2]:
            print(f"  {c.name}:\n{dump(c.body, 2)}", end="")
        assert len(hull_nl) == 10, len(hull_nl)  # 5 equalities, each as <= and >=
        n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
