"""Entry experiment for #1147: measure complementarity-pair loss per pass."""
from discopt import Model
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt._relax.integer_product_reform import expand_integer_products
from discopt._relax.factorable_reform import factorable_reformulate
from discopt._relax.binary_multilinear_reform import reformulate_binary_multilinear

checks = 0

def build(extra=None):
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    if extra:
        extra(m)
    m.complementarity(x, y, name="pair0")
    return m

def report(label, out, before):
    global checks
    after = len(getattr(out, "_complementarities", []))
    checks += 1
    print(f"{label:34s} before={before} after={after} new_model={out is not None}")
    return after

m = build()
before = len(m._complementarities)
for meth in ("big-m", "hull", "mbigm"):
    m = build()
    report(f"gdp {meth}", reformulate_gdp(m, meth), len(m._complementarities))

def add_intprod(m):
    k = m.integer("k", lb=0, ub=3)
    z = m.continuous("z", lb=0, ub=5)
    m.subject_to(k * z <= 4)

m = build(add_intprod)
report("expand_integer_products", expand_integer_products(m), len(m._complementarities))

def add_div(m):
    w = m.continuous("w", lb=1, ub=5)
    v = m.continuous("v", lb=1, ub=5)
    m.subject_to(v / w <= 4)

m = build(add_div)
report("factorable_reformulate", factorable_reformulate(m), len(m._complementarities))

m = build()
out = reformulate_binary_multilinear(m)
print("binary_multilinear abstains:", out is m)
checks += 1

assert checks > 0, "probe executed no comparisons"
print("EXECUTED_COMPARISONS:", checks)
