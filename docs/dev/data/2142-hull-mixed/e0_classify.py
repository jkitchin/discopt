"""#2142 hull-mixed entry, part 0: how many of GDPlib cstr's nonlinear disjunct rows
does the convexity detector prove convex, and does the mixed lowering fire?

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u e0_classify.py
Prints an executed-assertion count (CLAUDE.md §6)."""
import os
WT = os.environ["DISCOPT_TREE"]
import discopt
import discopt._relax.gdp_reformulate as G
assert discopt.__file__.startswith(WT), discopt.__file__
assert hasattr(G, "_hull_mixed_big_m_rows"), "marker absent"
n_assert = 2
from discopt.modeling.core import _DisjunctiveConstraint, _IndicatorConstraint, Constraint
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt._relax.convexity import classify_constraint
import gdplib.cstr as cstr

m = convert_pyomo_gdp(cstr.build_model()).model
kinds = {}
for c in m._constraints:
    kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
print("constraint kinds:", kinds)
rows = []
plan = G._collect_indicator_hull_groups(m)
for key, (dc, sels) in plan.groups.items():
    for k, d in enumerate(dc.disjuncts):
        for it in d:
            if isinstance(it, Constraint):
                rows.append(it)
for c in m._constraints:
    if isinstance(c, _DisjunctiveConstraint):
        for d in c.disjuncts:
            rows.extend(it for it in d if isinstance(it, Constraint))
lin = [r for r in rows if G._is_linear(r.body)]
nl = [r for r in rows if not G._is_linear(r.body)]
conv = [r for r in nl if classify_constraint(r, m)]
print(f"hull groups: {len(plan.groups)}; disjunct rows: {len(rows)}; linear {len(lin)}; "
      f"nonlinear {len(nl)}; nonlinear proven convex {len(conv)}")
for r in nl:
    print(f"  {'CONVEX ' if r in conv else 'unknown'} {r.sense} {str(r.body)[:110]}")
assert rows; n_assert += 1
for arm in ("0", "1"):
    os.environ["DISCOPT_HULL_MIXED_BIGM"] = arm
    G._HULL_MIXED_STATS.update(big_m=0, perspective=0)
    lowered = G.reformulate_gdp(m, method="hull")
    print(f"arm {arm}: lowered rows {len(lowered._constraints)} vars {len(lowered._variables)} "
          f"stats {G._HULL_MIXED_STATS}")
    if arm == "1":
        assert G._HULL_MIXED_STATS["big_m"] == len(nl) - len(conv), G._HULL_MIXED_STATS
        n_assert += 1
    else:
        assert G._HULL_MIXED_STATS["big_m"] == 0
        n_assert += 1
print(f"executed assertions: {n_assert}")
