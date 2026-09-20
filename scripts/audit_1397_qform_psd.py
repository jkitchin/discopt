"""#1397 entry/exit experiment for ``quadratic_is_psd`` (the convexity certificate).

    verify_qform_psd.py <tree_root> <yes|no: expect the #1397 marker>

``quadratic_is_psd`` accepts a matrix as PSD when ``eigvalsh(Q).min() >= -tol``
with ``tol = _PSD_TOL = 1e-10`` at both call sites in ``convexity/certificate.py``.
``tol`` is an absolute constant; ``eigvalsh``'s error on a symmetric ``Q`` is
O(eps*||Q||), so the constant carries no information about the sign of the true
minimum eigenvalue once ``||Q|| >~ 1e6``. The verdict is a *convexity
certificate*: ``Curvature.CONVEX`` licenses treating the body as its own convex
underestimator, so a false positive is a relaxation that does not relax.

Three arms, each with a construction-time oracle (``Q = V diag(ev) V^T``), so
"is it PSD" never comes from the code under test:

  A. FALSE POSITIVE -- an indefinite ``Q`` (exactly one negative eigenvalue,
     swept from -1e-12 down to -1.0) accepted as PSD. Unsound: a false
     convexity certificate.
  B. FALSE NEGATIVE -- a genuinely PSD ``Q`` carrying an EXACT zero eigenvalue
     (the common case: a quadratic in a subset of the model's variables) refused.
     Not unsound -- the caller falls back to the rigorous interval-Hessian path --
     but it is a silent capability loss, and it is the reason a scale-aware slack
     cannot simply demand strict positivity here.
  C. PARITY -- well-scaled mixed-sign matrices must get the identical verdict
     before and after, or the change is a tolerance change rather than a
     yardstick change (#1397's stated non-goal).
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
from discopt._relax import quadratic_form as QF  # noqa: E402
from discopt._relax.convexity import certificate as CERT  # noqa: E402

for mod in (QF, CERT):
    assert mod.__file__.startswith(TREE), f"loaded {mod.__file__}, expected under {TREE}"
HAS = "#1397" in inspect.getsource(QF.quadratic_is_psd)
assert HAS == EXPECT, (
    f"#1397 marker {'present' if HAS else 'absent'} in quadratic_is_psd but expected "
    f"{'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: {QF.__file__} marker={HAS}", flush=True)
print(f"            certificate module: {CERT.__file__}", flush=True)

EPS = float(np.finfo(np.float64).eps)
TOL = 1e-10  # the constant this fix removed, kept as the baseline yardstick
SCALES = (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)


def sym(ev, rng):
    """A symmetric matrix with *exactly* the eigenvalues ``ev``, random basis."""
    n = len(ev)
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Q = V @ np.diag(np.asarray(ev, dtype=np.float64)) @ V.T
    return 0.5 * (Q + Q.T)


def verdict(Q):
    """What the certificate's call sites ask. ``None``/``False`` both mean abstain."""
    return QF.quadratic_is_psd(Q) if HAS else QF.quadratic_is_psd(Q, tol=TOL)


CHECKS = 0
FALSE_POS = []
FALSE_NEG = []
PARITY_DIFF = []
#: For every admitted indefinite Q, |true negative eigenvalue| / ||Q||_F. This is
#: the quantity that decides whether admitting it is a soundness loss: the bound
#: error from treating a form with min eigenvalue -d as convex is at most
#: d*diam(box)^2/2, and d/||Q|| is that error measured against the magnitude of
#: the form itself over the same box. A ratio of order eps is the same order as
#: evaluating the form in floating point; a ratio of order 1 is a false bound.
REL_NONCONVEXITY = []

rng = np.random.default_rng(13970)

# ---- A. indefinite must never be certified PSD --------------------------------
for n in (2, 4, 8, 16, 32):
    for scale in SCALES:
        for true_neg in (-1e-12, -1e-9, -1e-6, -1e-3, -1e-1, -1.0):
            for _ in range(5):
                ev = np.concatenate([[true_neg], rng.uniform(0.5, 1.0, size=n - 1) * scale])
                Q = sym(ev, rng)
                CHECKS += 1
                if verdict(Q) is True:
                    REL_NONCONVEXITY.append(abs(true_neg) / max(np.linalg.norm(Q, "fro"), 1e-300))
                    FALSE_POS.append(
                        f"n={n:2d} ||Q||_F={np.linalg.norm(Q, 'fro'):.3e} "
                        f"true_min={true_neg:.2e} computed_min="
                        f"{float(np.linalg.eigvalsh(Q).min()):+.3e} CERTIFIED CONVEX"
                    )

# ---- B. genuine PSD with a zero eigenvalue must stay certified ----------------
for n in (2, 4, 8, 16, 32):
    for scale in SCALES:
        for nzero in (1, max(1, n // 2)):
            for _ in range(5):
                ev = np.concatenate(
                    [np.zeros(nzero), rng.uniform(0.5, 1.0, size=n - nzero) * scale]
                )
                Q = sym(ev, rng)
                CHECKS += 1
                if verdict(Q) is not True:
                    FALSE_NEG.append(
                        f"n={n:2d} ||Q||_F={np.linalg.norm(Q, 'fro'):.3e} "
                        f"{nzero} exact zero eigenvalue(s), computed_min="
                        f"{float(np.linalg.eigvalsh(Q).min()):+.3e} REFUSED"
                    )

# ---- C. parity on well-scaled matrices ---------------------------------------
for _ in range(400):
    n = int(rng.integers(2, 9))
    Q = sym(rng.uniform(-1.0, 1.0, size=n), rng)
    absolute = float(np.linalg.eigvalsh(0.5 * (Q + Q.T)).min()) >= -TOL
    CHECKS += 1
    got = verdict(Q)
    if bool(got is True) != absolute:
        PARITY_DIFF.append(
            f"||Q||_F={np.linalg.norm(Q, 'fro'):.3e} absolute={absolute} now={got!r} "
            f"computed_min={float(np.linalg.eigvalsh(Q).min()):+.3e}"
        )

print(f"\nEXECUTED ASSERTIONS: {CHECKS}")
print(f"A. indefinite Q certified PSD (false convexity certificates): {len(FALSE_POS)}")
for f in FALSE_POS[:10]:
    print("   FALSE_POS:", f)
if len(FALSE_POS) > 10:
    print(f"   ... and {len(FALSE_POS) - 10} more")
print(f"B. genuinely PSD Q (exact zero eigenvalue) refused: {len(FALSE_NEG)}")
for f in FALSE_NEG[:10]:
    print("   FALSE_NEG:", f)
if len(FALSE_NEG) > 10:
    print(f"   ... and {len(FALSE_NEG) - 10} more")
if REL_NONCONVEXITY:
    r = np.asarray(REL_NONCONVEXITY)
    print(
        f"   worst RELATIVE nonconvexity admitted: {r.max():.3e} = {r.max() / EPS:.1f}*u"
        f" (median {np.median(r):.3e})"
    )
    # The raw admission COUNT is not comparable between the two gates -- they admit
    # different sets -- so count only the admissions that are materially nonconvex,
    # i.e. whose nonconvexity exceeds the arithmetic's own resolution by 100x.
    for thr, label in ((100.0 * EPS, "100*u"), (1e-12, "1e-12")):
        print(
            f"   MATERIAL admissions (relative nonconvexity > {label} = {thr:.2e}): "
            f"{int((r > thr).sum())} of {r.size}"
        )
print(f"C. verdict differs from the absolute gate on a well-scaled Q: {len(PARITY_DIFF)}")
for f in PARITY_DIFF[:10]:
    print("   PARITY_DIFF:", f)

if CHECKS == 0:
    print("PROBE MEASURED NOTHING")
    sys.exit(2)
sys.exit(0)
