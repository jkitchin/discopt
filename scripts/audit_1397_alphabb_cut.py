"""#1397: is the alphaBB CUT's safety margin scale-exposed? Graded on the real cut.

    verify_alphabb_cut.py <tree_root> <yes|no: expect the #1397 marker>

``generate_alphabb_quadratic_oa_cuts_from_evaluator`` builds the separable alphaBB
underestimator of a nonconvex quadratic row and cuts with its TANGENT at x*:

    q_under(x) = q(x) - sum_i alpha_i (x_i - lb_i)(ub_i - x_i)

``q_under <= q`` holds for any ``alpha >= 0`` because each bracket is nonnegative
in the box -- that part is unconditional. The soundness condition is that
``q_under`` be CONVEX, since only a convex function has its tangent as a global
underestimator; otherwise the cut can exclude points satisfying ``q(x) <= 0``.
Convexity requires

    H + 2*diag(alpha) >= 0   i.e.   alpha >= -lambda_min(H)/2

against the TRUE minimum eigenvalue. The generator computes alpha from
``eigvalsh``, which can report a minimum ABOVE the true one by ~5.42*u*||H||, and
covered that with an absolute ``ALPHABB_SAFETY = 1e-6``.

How alpha is obtained here matters (CLAUDE.md SS6/SS8): an earlier version of this
probe reimplemented the generator's arithmetic and therefore reported identical
numbers before and after the fix, load gate and all -- it was grading a copy of the
old code. This version calls the generator and RECOVERS the alpha it actually used
from the returned cut. The generator sets

    under_grad[curved] = jac[k, curved] - alpha * (lb + ub - 2*x*)[curved]

and ``generate_oa_cut`` copies ``under_grad`` into ``cut.coeffs`` unchanged, so
with ``x*`` chosen off-centre the division is exact and alpha is recoverable. The
oracle for ``lambda_min`` is the construction ``C = V diag(ev) V^T``, never a
second call to the code under test.

HYPOTHESIS: for ||H|| >~ 1e10 the recovered alpha falls below ``-lambda_min/2``.
KILL CRITERION: zero shortfalls over the sweep to ||H||_F = 1e14.
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1])
EXPECT = (sys.argv[2] if len(sys.argv) > 2 else "yes").lower() == "yes"
sys.path.insert(0, os.path.join(TREE, "python"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
from discopt._relax.cutting_planes import (  # noqa: E402
    generate_alphabb_quadratic_oa_cuts_from_evaluator,
)
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._relax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.modeling import Model  # noqa: E402

SRC = inspect.getsource(generate_alphabb_quadratic_oa_cuts_from_evaluator)
HAS = "#1397" in SRC
assert HAS == EXPECT, (
    f"#1397 marker {'present' if HAS else 'absent'} in the alphaBB cut generator but "
    f"expected {'present' if EXPECT else 'absent'} -- wrong tree loaded"
)
print(f"LOAD GATE ok: marker={HAS}", flush=True)

SCALES = (1e0, 1e3, 1e6, 1e9, 1e10, 1e12, 1e14)


def build(n, scale, neg_frac, rng):
    """A nonconvex quadratic row ``0.5 x^T C x <= 1`` with C's spectrum known exactly."""
    lam_true = -neg_frac * scale
    ev = np.concatenate([[lam_true], rng.uniform(0.5, 1.0, size=n - 1) * scale])
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    C = V @ np.diag(ev) @ V.T
    C = 0.5 * (C + C.T)

    m = Model("alphabb_row")
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(n)]
    m.minimize(sum(xs[1:], xs[0]))
    body = 0.0
    for i in range(n):
        body = body + float(C[i, i]) * 0.5 * xs[i] ** 2
        for j in range(i + 1, n):
            body = body + float(C[i, j]) * xs[i] * xs[j]
    m.subject_to(body <= 1.0)
    return m, C, lam_true


CHECKS = 0
SKIPPED = 0
N_SHORT = 0
SHORT = []
WORST_REL = 0.0
BY_SCALE: dict[float, list[int]] = {}
rng = np.random.default_rng(139743)

for n in (2, 3, 5, 8):
    for scale in SCALES:
        BY_SCALE.setdefault(scale, [0, 0])
        for neg_frac in (1e-3, 1e-1, 1.0):
            for _ in range(6):
                m, C, lam_true = build(n, scale, neg_frac, rng)
                evaluator = NLPEvaluator(m)
                lb, ub = flat_variable_bounds(m)
                # Off-centre so ``lb + ub - 2*x*`` is nonzero and alpha is recoverable.
                x_star = np.full(n, 0.3) + rng.uniform(-0.1, 0.1, size=n)
                cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
                    evaluator,
                    x_star,
                    lb,
                    ub,
                    constraint_senses=["<="],
                    convex_mask=[False],
                )
                if len(cuts) != 1:
                    SKIPPED += 1
                    continue
                jac = np.asarray(evaluator.evaluate_jacobian(x_star), dtype=np.float64)[0]
                denom = lb + ub - 2.0 * x_star
                assert np.all(np.abs(denom) > 1e-6), "x* landed on a box centre"
                recovered = (jac - np.asarray(cuts[0].coeffs, dtype=np.float64)) / denom
                alpha = float(np.median(recovered))
                spread = float(np.max(np.abs(recovered - alpha)))
                assert spread <= 1e-6 * max(1.0, abs(alpha)), (
                    f"recovered alpha is not uniform across the curved block "
                    f"(spread {spread:.3e}, alpha {alpha:.3e}) -- the recovery is wrong, "
                    f"not the code under test"
                )
                need = -0.5 * lam_true
                CHECKS += 1
                BY_SCALE[scale][1] += 1
                if alpha < need:
                    N_SHORT += 1
                    BY_SCALE[scale][0] += 1
                    rel = (need - alpha) / max(abs(lam_true), 1e-300)
                    WORST_REL = max(WORST_REL, rel)
                    if len(SHORT) < 12:
                        SHORT.append(
                            f"n={n:2d} ||H||_F={np.linalg.norm(C, 'fro'):.3e} "
                            f"lam_true={lam_true:+.6e} alpha={alpha:.17e} "
                            f"need>={need:.17e} short={need - alpha:.3e} ({rel:.2e} rel)"
                        )

print(f"\nEXECUTED ASSERTIONS: {CHECKS}   (rows the generator declined to cut: {SKIPPED})")
print(f"cuts whose alpha is below -lambda_min/2 (invalid relaxation cut): {N_SHORT} of {CHECKS}")
for sc in sorted(BY_SCALE):
    bad, tot = BY_SCALE[sc]
    print(f"   Hessian scale {sc:8.0e}: {bad:4d} / {tot:4d} cuts invalid")
for f in SHORT:
    print("   INVALID:", f)
print(f"worst relative alpha shortfall: {WORST_REL:.3e}")

if CHECKS == 0:
    print("PROBE MEASURED NOTHING")
    sys.exit(2)
sys.exit(1 if N_SHORT else 0)
