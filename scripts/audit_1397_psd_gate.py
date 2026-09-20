"""#1397 entry/exit experiment for the convex-objective PSD gate.

Run against BOTH trees with the marker asserted present/absent (CLAUDE.md §8):

    verify_psd_gate.py <tree_root> <yes|no: expect the #1397 marker>

Measures, over a parameter sweep in ||H|| (the scale the absolute gate ignores):

  A. FALSE POSITIVES -- indefinite Hessians declared PSD. Each matrix is built as
     ``Q diag(ev) Q^T`` with exactly one NEGATIVE eigenvalue, so "is it PSD" has a
     known answer that does not come from the code under test. Every admission is
     a licence to emit an invalid dual bound.
  B. FALSE NEGATIVES -- genuinely PD Hessians (all eigenvalues positive and well
     above the eigensolver's roundoff) that the gate declines. A few are expected
     and harmless (the gate abstains, it does not err), but a yardstick that
     declines everything large would have bought soundness with uselessness, so
     this arm is measured rather than assumed.
  C. NO-OP ARM -- well-scaled Hessians must get the IDENTICAL verdict from the
     absolute gate and the scale-aware one.
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import discopt.solver as S  # noqa: E402
import numpy as np  # noqa: E402

assert S.__file__.startswith(TREE), f"loaded {S.__file__}, expected under {TREE}"
HAS = hasattr(S, "_hessian_is_psd_with_margin")
if HAS:
    _src = inspect.getsource(S._hessian_is_psd_with_margin)
    HAS = "#1397" in _src
assert HAS == EXPECT, (
    f"#1397 marker {'present' if HAS else 'absent'} but expected "
    f"{'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: {S.__file__} marker={HAS}", flush=True)

EPS = float(np.finfo(np.float64).eps)


def verdict(H):
    """The gate under test, in whichever tree was loaded."""
    if HAS:
        return bool(S._hessian_is_psd_with_margin(H))
    # Baseline: the absolute gate, spelled exactly as the old code spelled it.
    return float(np.linalg.eigvalsh(0.5 * (H + H.T)).min()) >= S._CONVEX_OBJ_PSD_TOL


CHECKS = 0
FALSE_POS = []
FALSE_NEG = []
NOOP_DIFF = []

rng = np.random.default_rng(1397)
SCALES = (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16)

# ---- A. indefinite must be refused -------------------------------------------
for n in (2, 4, 8, 16, 32):
    for scale in SCALES:
        for true_neg in (-1e-12, -1e-9, -1e-6, -1e-3, -1e-1, -1.0):
            for _ in range(6):
                Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
                ev = np.concatenate([[true_neg], rng.uniform(0.5, 1.0, size=n - 1) * scale])
                H = Q @ np.diag(ev) @ Q.T
                H = 0.5 * (H + H.T)
                CHECKS += 1
                if verdict(H):
                    FALSE_POS.append(
                        f"n={n} ||H||_F={np.linalg.norm(H, 'fro'):.3e} true_min={true_neg:.3e} "
                        f"computed_min={float(np.linalg.eigvalsh(H).min()):+.3e} DECLARED PSD"
                    )

# ---- B. genuinely PD, eigenvalues far above roundoff, should be accepted ------
for n in (2, 4, 8, 16, 32):
    for scale in SCALES:
        for _ in range(6):
            Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            # smallest eigenvalue held at 1e-3 * scale: a condition number of 1e3,
            # far above the O(eps*||H||) roundoff at every scale in the sweep.
            ev = np.concatenate([[1e-3 * scale], rng.uniform(0.5, 1.0, size=n - 1) * scale])
            ev = np.maximum(ev, 1e-5)
            H = Q @ np.diag(ev) @ Q.T
            H = 0.5 * (H + H.T)
            CHECKS += 1
            if not verdict(H):
                FALSE_NEG.append(
                    f"n={n} ||H||_F={np.linalg.norm(H, 'fro'):.3e} true_min={ev.min():.3e} DECLINED"
                )

# ---- C. no-op on well-scaled Hessians ----------------------------------------
for _ in range(300):
    n = int(rng.integers(2, 9))
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ev = rng.uniform(-1.0, 1.0, size=n)  # mixed signs: both verdicts exercised
    H = Q @ np.diag(ev) @ Q.T
    H = 0.5 * (H + H.T)
    absolute = float(np.linalg.eigvalsh(H).min()) >= S._CONVEX_OBJ_PSD_TOL
    CHECKS += 1
    if verdict(H) != absolute:
        NOOP_DIFF.append(
            f"||H||_F={np.linalg.norm(H, 'fro'):.3e} absolute={absolute} scale_aware={verdict(H)}"
        )

print(f"EXECUTED ASSERTIONS: {CHECKS}")
print(f"A. indefinite declared PSD (false bounds licensed): {len(FALSE_POS)}")
for f in FALSE_POS[:12]:
    print("   FALSE_POS:", f)
if len(FALSE_POS) > 12:
    print(f"   ... and {len(FALSE_POS) - 12} more")
print(f"B. genuinely PD declined (abstention, not error): {len(FALSE_NEG)}")
for f in FALSE_NEG[:6]:
    print("   FALSE_NEG:", f)
print(f"C. verdict differs from the absolute gate on a well-scaled H: {len(NOOP_DIFF)}")
for f in NOOP_DIFF[:6]:
    print("   NOOP_DIFF:", f)

if CHECKS == 0:
    print("PROBE MEASURED NOTHING")
    sys.exit(2)
# On the FIXED tree all three must be zero. On the baseline, A is expected to be
# non-zero -- that is the defect -- so the script reports and the caller compares.
sys.exit(0)
