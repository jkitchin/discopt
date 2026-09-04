"""#1039: localize the super-optimal reported objective.  Hypothesis: the
incumbent's objective is read off the relaxation's auxiliary/epigraph variables
instead of being recomputed at the incumbent point, so a factorable model with
auxiliaries reports the RELAXED value.  Kill criterion: if a model whose
objective needs no auxiliary (a plain posynomial sum) also reports an objective
that disagrees with the oracle, the auxiliary hypothesis is falsified."""
import sys, warnings
import discopt
from discopt import Model
import discopt.modeling as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

POS = dict(lb=1e-3, ub=1e3)


def check(label, build, f_oracle, tl=5.0):
    m, names = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(solver="bb", time_limit=tl)
    vals = [float(r.value(v)) for v in names]
    orc = f_oracle(*vals)
    d = r.objective - orc
    flag = "  <-- MISMATCH" if abs(d) > 1e-9 else ""
    print(f"{label:26s} status={r.status:8s} objective={r.objective!r}")
    print(f"{'':26s} point={vals} oracle={orc!r} delta={d:+.6e}{flag}", flush=True)
    return abs(d) > 1e-9


def b_div():
    m = Model("div"); x = m.continuous("x", **POS); y = m.continuous("y", **POS)
    m.minimize(x / y + y / x); return m, (x, y)


def b_sum():
    # No division: x + y is affine, needs no auxiliary at all.
    m = Model("sum"); x = m.continuous("x", **POS); y = m.continuous("y", **POS)
    m.minimize(x + y); return m, (x, y)


def b_bilin():
    # Bilinear -> needs a McCormick auxiliary, but no division.
    m = Model("bilin"); x = m.continuous("x", **POS); y = m.continuous("y", **POS)
    m.minimize(x * y + 1.0 / (x * y)); return m, (x, y)


def b_sq():
    m = Model("sq"); x = m.continuous("x", **POS); y = m.continuous("y", **POS)
    m.minimize((x - 1.0) ** 2 + (y - 2.0) ** 2); return m, (x, y)


n = 0
mismatched = []
for lbl, bld, orc in (
    ("division (x/y + y/x)", b_div, lambda x, y: x / y + y / x),
    ("affine (x + y)", b_sum, lambda x, y: x + y),
    ("bilinear (xy + 1/xy)", b_bilin, lambda x, y: x * y + 1.0 / (x * y)),
    ("separable sq", b_sq, lambda x, y: (x - 1.0) ** 2 + (y - 2.0) ** 2),
):
    if check(lbl, bld, orc):
        mismatched.append(lbl)
    n += 1

print(f"\nEXECUTED CHECKS: {n}")
print(f"MISMATCHED: {mismatched or 'none'}")
sys.exit(0 if n else 1)
