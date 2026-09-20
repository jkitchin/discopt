"""#1397 entry/exit experiment for ``certify_convex`` (the interval/Gershgorin path).

    verify_certify_convex.py <tree_root> <yes|no: expect the #1397 marker>

``certify_convex`` compared Gershgorin's rigorous eigenvalue bounds -- and a
rank-1 coefficient interval -- against an absolute ``_PSD_TOL = 1e-10``. Both
quantities carry the units of the Hessian, so the constant means nothing away
from ``||H|| ~ 1``.

Two arms over a scale sweep, oracle by construction (algebra, not the code under
test):

  A. CAPABILITY -- ``scale*(x - y)**2`` is convex for every positive scale (its
     Hessian ``2*scale*[[1,-1],[-1,1]]`` has eigenvalues ``0`` and ``4*scale``).
     Outward rounding renders the exact zero as ``~ -u*||H||``, so an absolute
     1e-10 loses the certificate once ``||H|| >~ 1e5``.
  B. SOUNDNESS -- ``scale*(x*y)`` has Hessian ``scale*[[0,1],[1,0]]``, eigenvalues
     ``+-scale``: nonconvex by 100% of its own magnitude at every scale. A
     certificate here licenses treating it as its own convex underestimator.

Also sweeps a diagonal form that is convex in one variable and linear in the
other -- the "quadratic in a subset of the variables" case -- in both curvature
directions.
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import discopt.modeling as dm  # noqa: E402
from discopt._relax.convexity import certificate as CERT  # noqa: E402
from discopt._relax.convexity.certificate import certify_convex  # noqa: E402
from discopt._relax.convexity.lattice import Curvature  # noqa: E402

assert CERT.__file__.startswith(TREE), f"loaded {CERT.__file__}, expected under {TREE}"
HAS = "psd_decision_slack" in inspect.getsource(CERT)
assert HAS == EXPECT, (
    f"#1397 scale-aware slack {'present' if HAS else 'absent'} in certificate.py but "
    f"expected {'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: {CERT.__file__} scale_aware={HAS}", flush=True)

SCALES = (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)

CHECKS = 0
LOST = []  # convex, but not certified -- silent capability loss
UNSOUND = []  # indefinite, but certified -- false convexity certificate


def model():
    m = dm.Model("scaled")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x)
    return m, x, y


for scale in SCALES:
    m, x, y = model()

    # A. convex at every scale, in both curvature directions.
    for label, expr, want in (
        ("scale*(x-y)^2", scale * (x - y) ** 2, Curvature.CONVEX),
        ("-scale*(x-y)^2", -(scale * (x - y) ** 2), Curvature.CONCAVE),
        ("scale*x^2 + y", scale * x**2 + y, Curvature.CONVEX),
        ("-scale*x^2 + y", -(scale * x**2) + y, Curvature.CONCAVE),
    ):
        CHECKS += 1
        got = certify_convex(expr, m)
        if got is not want:
            LOST.append(f"scale={scale:.0e} {label:16s} want {want!s:18s} got {got!s}")

    # B. indefinite at every scale -- must never be certified.
    for label, expr in (("scale*x*y", scale * (x * y)), ("-scale*x*y", -(scale * (x * y)))):
        CHECKS += 1
        got = certify_convex(expr, m)
        if got is not None:
            UNSOUND.append(f"scale={scale:.0e} {label:16s} CERTIFIED {got!s}")

print(f"\nEXECUTED ASSERTIONS: {CHECKS}")
print(f"A. convex body whose certificate was LOST: {len(LOST)}")
for f in LOST:
    print("   LOST:  ", f)
print(f"B. indefinite body CERTIFIED convex/concave (unsound): {len(UNSOUND)}")
for f in UNSOUND:
    print("   UNSOUND:", f)

if CHECKS == 0:
    print("PROBE MEASURED NOTHING")
    sys.exit(2)
if UNSOUND:
    print("SOUNDNESS VIOLATION")
    sys.exit(1)
sys.exit(1 if LOST else 0)
