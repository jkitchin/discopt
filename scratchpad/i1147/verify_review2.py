"""Reproduce the two blocking findings of review 5115268644 on PR head 1bd4641."""
import discopt.mpec as mpec
import discopt.modeling.core as dm

checks = 0

# B1 — flat_source_indices uses Variable._index (home-model position), not the
# TARGET model's ordering.
home = dm.Model("home")
x = home.continuous("x", shape=2, lb=0, ub=5)   # _index 0, size 2
y = home.continuous("y", lb=0, ub=5)            # _index 1, size 1
home.minimize(dm.sum(x) + y)
pair = home.complementarity(x, y, name="p")

target = dm.Model("target")
target._variables = [y, x]                       # deliberately the OTHER order
target._rebuild_name_index()
got = mpec.flat_source_indices(target, pair)
print(f"B1 target order [y, x] -> {got}   (correct: x at 1,2 and y at 0 => [1, 2, 0])")
checks += 1

# B2a — the lowering method is global state on the shared relation.
m1 = dm.Model("m1"); m2 = dm.Model("m2")
a = m1.continuous("a", lb=0, ub=5); b = m1.continuous("b", lb=0, ub=5)
m1.minimize(a + b)
m2._variables = [a, b]; m2._rebuild_name_index(); m2.minimize(a + b)
shared = mpec.complementarity(a, b, "shared")
mpec.reformulate_gdp(m1, [shared])
mpec.reformulate_sos1(m2, [shared])
print(f"B2a after gdp in m1 then sos1 in m2: lowering_in(m1)={shared.lowering_in(m1)!r} "
      f"lowering_in(m2)={shared.lowering_in(m2)!r} "
      f"| legacy global field present: {hasattr(shared, 'lowering')}")
checks += 1

# B2b — carry_complementarities treats a non-None pair.lowering as proof that
# SRC was lowered, so an UNLOWERED source hands the destination a lowered mark.
src = dm.Model("src")
src._variables = [a, b]; src._rebuild_name_index(); src.minimize(a + b)
src._complementarities.append(shared)            # recorded, never lowered into src
print(f"B2b src carries rows? {shared.is_lowered_into(src)}")
dst = dm.Model("dst")
dst._variables = [a, b]; dst._rebuild_name_index(); dst.minimize(a + b)
mpec.carry_complementarities(src, dst, pass_name="probe")
print(f"B2b dst marked lowered after carry from an UNLOWERED src? "
      f"{shared.is_lowered_into(dst)}  | unlowered_relations(dst)={mpec.unlowered_relations(dst)}")
checks += 1

print("EXECUTED_CHECKS:", checks)
assert checks > 0
