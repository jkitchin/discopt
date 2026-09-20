"""#1397 entry/exit experiment for ``certify_g_convex``'s PSD acceptance slack.

    audit_1397_g_convexity_slack.py <tree_root> <yes|no: expect the #1397 marker>

``certify_g_convex`` decided ``lambda_min(H + rho*Outer) >= 0`` and
``lambda_max(H - rho*Outer) <= 0`` against an absolute ``_PSD_TOL = 1e-10``,
whose comment claimed it was "matching the ordinary convexity certificate" -- it
stopped matching once ``certificate.py`` became scale-aware earlier in this same
audit. This is the *sound* certificate in the module (its two siblings,
``is_g_convex_pointwise`` and ``least_convexifying_rho``, are documented
floating-point diagnostics and already scale by ``max(1, max|H|)``), and it is
reached on the default path from ``mccormick_lp.py``, ``g_convex_cut.py``,
``g_products_ratios.py`` and ``g_convex_inject.py`` -- none of which pass ``tol``.

The tolerance value is READ FROM THE MODULE under test, never reimplemented here,
so both arms grade the shipped comparison rather than a paraphrase of it.

Two arms over a magnitude sweep, oracle by construction:

  A. SOUNDNESS -- a *diagonal* interval matrix ``diag(-delta, m, m, m)``. Diagonal
     means the Gershgorin row bound is exact (no off-diagonal radius to round
     outward), so ``lambda_min = -delta`` with no enclosure slop and the measured
     quantity is the gate alone. The graded number is the admitted *relative*
     nonconvexity ``delta / ||aug||_F``: a scale-free figure that bounds the
     relative error of treating the form as convex, independently of the box.
     An absolute licence makes it ``1e-10 / ||aug||``, unbounded as ``||aug||``
     shrinks; the scaled slack caps it at ``K*u``.
  B. CAPABILITY -- the same matrix with an exact zero eigenvalue displaced by the
     ``O(u*||aug||)`` excursion an interval enclosure's outward rounding produces.
     Genuinely PSD, so it must stay certified; an absolute ``1e-10`` refuses it
     once ``u*||aug|| > 1e-10``, i.e. from ``||aug|| ~ 1e6`` up.

Exits non-zero when either arm finds a violation, and also when either arm graded
zero verdicts (CLAUDE.md section 6: a probe that measured nothing must not read as
a pass). Expected: FAILS arm A (and arm B) on the pre-fix tree, passes on the fix.
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
from discopt._relax.convexity import g_convexity as G  # noqa: E402
from discopt._relax.convexity.eigenvalue import gershgorin_lambda_min  # noqa: E402
from discopt._relax.convexity.interval import Interval  # noqa: E402

assert G.__file__.startswith(TREE), f"loaded {G.__file__}, expected under {TREE}"
SRC = inspect.getsource(G)
HAS = "def _psd_slack" in SRC
assert HAS == EXPECT, (
    f"#1397 scale-aware slack {'present' if HAS else 'absent'} in g_convexity.py but "
    f"expected {'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: {G.__file__} scale_aware={HAS}", flush=True)

U = float(np.finfo(np.float64).eps)
MAGNITUDES = (1e-8, 1e-6, 1e-3, 1e0, 1e3, 1e6, 1e9, 1e12)

#: Ceiling on admitted relative nonconvexity. ``psd_decision_slack`` licenses
#: ``K*u`` with ``K = 32``, i.e. ~7.1e-15; this leaves two decades of headroom
#: and is a *constant*, which is the whole claim.
RELATIVE_CEILING = 1e-12


def gate(aug) -> float:
    """The shipped acceptance slack for ``lambda_min(aug) >= -slack``.

    Reads the tolerance from the module under test: ``_psd_slack`` after the fix,
    the absolute ``_PSD_TOL`` default before it. No value is hardcoded here.
    """
    if HAS:
        return float(G._psd_slack(aug, None))
    return float(G._PSD_TOL)


def diagonal(entries):
    m = np.diag(np.asarray(entries, dtype=np.float64))
    return Interval(m.copy(), m.copy())


def fro(entries) -> float:
    return float(np.linalg.norm(np.asarray(entries, dtype=np.float64)))


CHECKS = 0
UNSOUND = []  # materially nonconvex, but admitted as PSD
LOST = []  # genuinely PSD within enclosure roundoff, but refused

# ---- arm A: soundness -------------------------------------------------------
for mag in MAGNITUDES:
    for exponent in range(-22, -3):
        delta = 10.0**exponent
        entries = [-delta] + [mag] * 3
        aug = diagonal(entries)
        slack = gate(aug)
        CHECKS += 1
        if gershgorin_lambda_min(aug) >= -slack:
            rel = delta / fro(entries)
            if rel > RELATIVE_CEILING:
                UNSOUND.append((mag, delta, slack, rel))

# ---- arm B: capability ------------------------------------------------------
for mag in MAGNITUDES:
    for factor in (0.1, 0.5, 1.0):
        displacement = 5.0 * U * mag * factor
        entries = [-displacement] + [mag] * 3
        aug = diagonal(entries)
        slack = gate(aug)
        CHECKS += 1
        if gershgorin_lambda_min(aug) < -slack:
            LOST.append((mag, displacement, slack))

print(f"executed comparisons: {CHECKS}", flush=True)
if CHECKS == 0:
    print("FAIL: probe graded nothing", flush=True)
    sys.exit(2)

print(f"\narm A (soundness): {len(UNSOUND)} admissions above the "
      f"{RELATIVE_CEILING:.0e} relative-nonconvexity ceiling")
for mag, delta, slack, rel in UNSOUND[:12]:
    print(f"  |aug|~{mag:8.0e}  lambda_min=-{delta:8.0e}  slack={slack:9.3e}  "
          f"relative nonconvexity={rel:9.3e}  ({rel / U:.3g}*u)")
if len(UNSOUND) > 12:
    print(f"  ... and {len(UNSOUND) - 12} more")
if UNSOUND:
    worst = max(UNSOUND, key=lambda r: r[3])
    print(f"  WORST: relative nonconvexity {worst[3]:.3e} at |aug|~{worst[0]:.0e}")

print(f"\narm B (capability): {len(LOST)} genuinely-PSD matrices refused")
for mag, disp, slack in LOST[:12]:
    print(f"  |aug|~{mag:8.0e}  displacement={disp:9.3e}  slack={slack:9.3e}  REFUSED")
if len(LOST) > 12:
    print(f"  ... and {len(LOST) - 12} more")

status = 0 if not UNSOUND and not LOST else 1
print(f"\n{'PASS' if status == 0 else 'FAIL'}: unsound={len(UNSOUND)} lost={len(LOST)}")
sys.exit(status)
