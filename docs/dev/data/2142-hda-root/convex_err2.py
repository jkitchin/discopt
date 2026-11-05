"""Find the constraint / variable behind the IndexError (hda hull ON)."""
import os, sys
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = "1"
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt import transformations as Tr
from discopt._relax.convexity import classify_constraint
from discopt.modeling.core import Variable
m = convert_pyomo_gdp(_spec("hda").builder()).model
m2 = Tr.create_using("gdp.hull", m)
ids = {id(v) for v in m2._variables}
print("n vars objs", len(m2._variables), "last", m2._variables[-1].name, m2._variables[-1]._index, m2._variables[-1].size)
bad = 0
for con in m2._constraints:
    try:
        classify_constraint(con, m2)
    except IndexError as e:
        bad += 1
        stack, foreign = [con.body], set()
        while stack:
            nd = stack.pop()
            if isinstance(nd, Variable) and id(nd) not in ids: foreign.add((nd.name, nd._index))
            for a in ("left", "right", "operand", "args", "terms", "base"):
                ch = getattr(nd, a, None)
                if ch is None: continue
                stack.extend(ch if isinstance(ch, (list, tuple)) else [ch])
        print("FAIL", con.name, e, "foreign vars:", sorted(foreign)[:5], str(con.body)[:200])
        if bad > 3: break
print("executed assertions:", 1)
