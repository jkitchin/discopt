"""Independently reproduce the review findings on PR #1149 before fixing."""
import copy, pickle
import discopt.mpec as mpec
import discopt.modeling.core as dm
from discopt.solver import solve_model

print("marker:", hasattr(mpec, "carry_complementarities"), mpec.__file__)
checks = 0

# HIGH 1 — solve_model bypasses the refusal
def build_h1():
    m = dm.Model("h1")
    z = m.continuous("z", lb=-1, ub=1)
    m.maximize(z)
    m.mcp(z + 1.0, z, name="p0")
    return m, z

m, z = build_h1()
try:
    m.solve(time_limit=5)
    print("H1 Model.solve: NOT refused  <-- unexpected")
except NotImplementedError:
    print("H1 Model.solve: refused (correct)")
checks += 1

m, z = build_h1()
try:
    r = solve_model(m, time_limit=5)
    zi = r.x if r.x is not None else None
    print(f"H1 solve_model: status={r.status} obj={r.objective} z={zi}  <-- BYPASS")
except NotImplementedError:
    print("H1 solve_model: refused")
checks += 1

# HIGH 2 — role, not bounds, gates the lowering refusal
m2 = dm.Model("h2")
z2 = m2.continuous("z", lb=-1, ub=1)
w2 = m2.continuous("w", lb=0, ub=1)
m2.maximize(z2 + w2)
boxed = mpec.Complementarity(z2 + w2, z2, "p0", g_bounds=(-1.0, 1.0))
try:
    mpec.reformulate_gdp(m2, [boxed])
    print("H2: box-bounded relation LOWERED without refusal  <-- role gated it")
except NotImplementedError:
    print("H2: refused (correct)")
checks += 1

# HIGH 3 — weakref breaks deepcopy / pickle
m3 = dm.Model("h3")
a = m3.continuous("a", lb=0, ub=10); b = m3.continuous("b", lb=0, ub=10)
m3.minimize((a - 1) ** 2 + (b - 1) ** 2)
m3.complementarity(a, b, name="p0")
try:
    clone = copy.deepcopy(m3)
    print("H3 deepcopy: ok; clone unlowered =", [p.describe() for p in mpec.unlowered_relations(clone)])
except Exception as e:
    print("H3 deepcopy raised:", type(e).__name__, e)
try:
    pickle.dumps(m3)
    print("H3 pickle: ok")
except Exception as e:
    print("H3 pickle raised:", type(e).__name__, str(e)[:80])
checks += 2

# MEDIUM 4 — two unnamed relations collide
m4 = dm.Model("m4")
p = m4.continuous("p", lb=0, ub=5); q = m4.continuous("q", lb=0, ub=5)
r_ = m4.continuous("r", lb=0, ub=5); s_ = m4.continuous("s", lb=0, ub=5)
m4.minimize(p + q + r_ + s_)
m4.complementarity(p, q)
m4.complementarity(r_, s_)
names = [c.name for c in m4._constraints if getattr(c, "name", None)]
dupes = {n: names.count(n) for n in set(names) if names.count(n) > 1}
print("M4 constraint names:", names, "COLLISIONS:", dupes)
checks += 1

# MEDIUM 6 — flat indices over-report for an indexed operand
m6 = dm.Model("m6")
xv = m6.continuous("x", shape=3, lb=0, ub=5)
yv = m6.continuous("y", lb=0, ub=5)
m6.minimize(dm.sum(xv) + yv)
pr = m6.complementarity(xv[2], yv, name="ix")
print("M6 flat_source_indices(x[2] _|_ y):", mpec.flat_source_indices(m6, pr))
checks += 1

# LOW 7 — elements() identity contract
m7 = dm.Model("m7")
u = m7.continuous("u", lb=0, ub=5); v = m7.continuous("v", lb=0, ub=5)
m7.minimize(u + v)
pr7 = m7.complementarity(u, v, name="e")
e1, e2 = pr7.elements(m7), pr7.elements(m7)
print("L7 elements()[0] is self:", e1[0] is pr7, "| stable across calls:", e1[0] is e2[0])
checks += 1

# LOW 8 — sentinel reinterprets a genuine finite bound
m8 = dm.Model("m8")
z8 = m8.continuous("z", lb=0, ub=2e19)
rel8 = mpec.box_mcp(z8, z8, lb=0.0, ub=2e19, name="s")
print("L8 g_bounds for ub=2e19:", rel8.g_bounds, "role:", rel8.role.value)
checks += 1

print("EXECUTED_CHECKS:", checks)
assert checks > 0
