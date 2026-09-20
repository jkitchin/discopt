"""Is ``_alphabb_rigorous.rigorous_alpha`` actually rigorous? Exact-arithmetic answer.

    verify_rigorous_alpha_exact.py <tree_root> <yes|no: expect the #1397 marker>

``rigorous_alpha`` feeds ``solver._compute_alphabb_bound``, i.e. a per-node DUAL
BOUND -- a certificate. It claims

    lambda_min >= H[i,i].lo - sum_{j != i} max(|H[i,j].lo|, |H[i,j].hi|)

and sets ``alpha_i = max(0, -0.5 * gershgorin_lo_i)``. The inequality is exact in
the reals, but evaluating it in round-to-nearest binary64 may round the
off-diagonal radius DOWN, which raises the bound above its true value and leaves
``alpha`` BELOW ``-lambda_min/2``. The alphaBB body is then nonconvex, so its box
minimum is not a lower bound on ``q`` and the node bound can exceed the true one:
a false dual bound.

This probe CALLS ``rigorous_alpha`` and grades its output against the same
formula evaluated in EXACT rational arithmetic over the very same interval-Hessian
float entries (every binary64 is a rational, so ``Fraction`` is exact and the
comparison contributes no error of its own). A violation is a proof.

Note on an earlier version of this probe: it reimplemented the shipped arithmetic
locally and so reported identical numbers before and after the fix even with the
load gate passing -- it was grading a copy of the old code. It now calls the real
function, which is the only thing that makes the load gate mean anything (CLAUDE.md
measurement discipline SS6/SS8).
"""

import os
import sys
from fractions import Fraction

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
from discopt import _alphabb_rigorous as RA  # noqa: E402
from discopt._relax.convexity.interval_ad import interval_hessian  # noqa: E402

assert RA.__file__.startswith(TREE), f"loaded {RA.__file__}, expected under {TREE}"
HAS = "#1397" in inspect.getsource(RA)
assert HAS == EXPECT, (
    f"#1397 marker {'present' if HAS else 'absent'} in _alphabb_rigorous but expected "
    f"{'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: {RA.__file__} marker={HAS}", flush=True)


def exact_row_lower_bounds(h_lo, h_hi):
    """The Gershgorin row bounds in exact rational arithmetic over the same floats."""
    n = h_lo.shape[0]
    out = []
    for i in range(n):
        radius = Fraction(0)
        for j in range(n):
            if j == i:
                continue
            radius += max(abs(Fraction(float(h_lo[i, j]))), abs(Fraction(float(h_hi[i, j]))))
        out.append(Fraction(float(h_lo[i, i])) - radius)
    return out


def quadratic_model(n, scale, rng):
    """A dense quadratic whose Hessian entries sit at ``scale``."""
    m = dm.Model("q")
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(n)]
    C = rng.uniform(-1.0, 1.0, size=(n, n)) * scale
    C = 0.5 * (C + C.T)
    expr = 0.0
    for i in range(n):
        expr = expr + float(C[i, i]) * 0.5 * xs[i] ** 2
        for j in range(i + 1, n):
            expr = expr + float(C[i, j]) * xs[i] * xs[j]
    m.minimize(expr)
    return m, expr


CHECKS = 0
N_VIOLATIONS = 0
VIOLATIONS = []
WORST = Fraction(0)
BY_SCALE: dict[float, list[int]] = {}
rng = np.random.default_rng(139742)

for n in (4, 8, 16):
    for scale in (1e0, 1e3, 1e6, 1e9, 1e12):
        BY_SCALE.setdefault(scale, [0, 0])
        for _ in range(8):
            m, expr = quadratic_model(n, scale, rng)
            alpha = np.asarray(RA.rigorous_alpha(expr, m), dtype=np.float64)
            iad = interval_hessian(expr, m)
            h_lo = np.asarray(iad.hess.lo, dtype=np.float64)
            h_hi = np.asarray(iad.hess.hi, dtype=np.float64)
            exact = exact_row_lower_bounds(h_lo, h_hi)
            for i in range(n):
                CHECKS += 1
                BY_SCALE[scale][1] += 1
                # The requirement: alpha_i must dominate -b_i/2 for the EXACT b_i.
                need = -exact[i] / 2 if exact[i] < 0 else Fraction(0)
                if Fraction(float(alpha[i])) < need:
                    N_VIOLATIONS += 1
                    BY_SCALE[scale][0] += 1
                    short = need - Fraction(float(alpha[i]))
                    WORST = max(WORST, short)
                    if len(VIOLATIONS) < 12:
                        VIOLATIONS.append(
                            f"n={n:3d} scale={scale:.0e} row {i:3d}: alpha={alpha[i]:.17e} "
                            f"< required {float(need):.17e} (short by {float(short):.3e})"
                        )

print(f"\nEXECUTED ASSERTIONS: {CHECKS}")
print(f"rows where alpha is PROVABLY below -lambda_min/2: {N_VIOLATIONS} of {CHECKS}")
for sc in sorted(BY_SCALE):
    bad, tot = BY_SCALE[sc]
    print(f"   Hessian scale {sc:8.0e}: {bad:4d} / {tot:4d} rows unsound")
for f in VIOLATIONS:
    print("   UNSOUND:", f)
print(f"worst proven alpha shortfall: {float(WORST):.6e}")

if CHECKS == 0:
    print("PROBE MEASURED NOTHING")
    sys.exit(2)
sys.exit(1 if N_VIOLATIONS else 0)
