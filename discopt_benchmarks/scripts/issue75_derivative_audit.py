"""AUTHORITATIVE derivative audit: orders 0, 1 and 2, scored against TRUTH.

Why this exists alongside the two older probes. `issue75_rewrite_hardening_probe`
(values/gradients) and `issue75_second_order_probe` (Hessians) both score the tape
against JAX. That was the right criterion while JAX was strictly the more accurate
arm -- the tape's job was to reproduce `_relax/dag_compiler.py`. It stopped being the
right criterion once the tape overtook JAX in the exponential tails: JAX's
`sigmoid'(40)` returns 0.0 against a true 4.248e-18, and a JAX-relative probe
scores the tape's CORRECT 4.248e-18 as a defect. Those two probes now report
inverted verdicts at exactly the points this branch improved.

So the oracle here is mpmath at 600 digits with ANALYTIC closed forms -- not
numerical differentiation, and not either backend. Both arms are scored on equal
terms, so "JAX does it too" cannot excuse a wrong answer and neither can "the tape
matches JAX".

Two instrument bugs found while building this, both worth remembering:

  * At `mp.dps = 60` the oracle itself underflowed: `1 + exp(-300)` rounds to
    exactly 1.0, so it claimed `sigmoid'(300) = 0` and scored the tape's correct
    5.148e-131 as a 5.148e169 defect. Precision must exceed the smallest scale
    probed -- `exp(-745)` is 5e-324, hence 600 digits.
  * Relative error must ask whether the TRUE value is representable as a double
    before scoring a non-finite result. `log''(1e-300)` is -1e600; -inf is the
    correct binary64 answer, not a defect. Checking `got` first scored those
    points inf for BOTH arms, which is the tell that it was the instrument.

No try/except (§7). Executed-comparison counter, non-zero exit at zero (§6).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import Constant, FunctionCall

if "mpmath" not in sys.modules:
    import importlib.util

    if importlib.util.find_spec("mpmath") is None:
        print(
            "mpmath is required: this probe's whole point is an oracle independent "
            "of both backends. Install it (pip install mpmath) rather than falling "
            "back to a float reference -- a float oracle underflows exactly where "
            "the interesting points are.",
            file=sys.stderr,
        )
        sys.exit(1)

from mpmath import mp, mpf  # noqa: E402

mp.dps = 600

FLOOR = mpf("1e-300")
DBL_MAX = mpf("1.7976931348623157e308")
TOL = 1e-9


def _sig(a):
    """1/(1+e^-a), written so neither arm forms ``1 + (something negligible)``."""
    if a >= 0:
        t = mp.e ** (-a)
        return mp.mpf(1) / (1 + t)
    t = mp.e**a
    return t / (1 + t)


# Each oracle returns (value, gradient list, Hessian as nested list).
def o_log1p(a):
    u = 1 + a
    return mp.log(u), [1 / u], [[-1 / u**2]]


def o_log2(a):
    l2 = mp.log(2)
    return mp.log(a) / l2, [1 / (a * l2)], [[-1 / (a**2 * l2)]]


def o_sigmoid(a):
    s = _sig(a)
    return s, [s * (1 - s)], [[s * (1 - s) * (1 - 2 * s)]]


def o_softplus(a):
    s = _sig(a)
    v = mp.log1p(mp.e**a) if a <= 0 else a + mp.log1p(mp.e ** (-a))
    return v, [s], [[s * (1 - s)]]


def o_entropy(a):
    """``x*log(max(x, 1e-300))`` -- the floor is part of the semantics, not a nicety.

    Scoring against the unclamped ``x*log(x)`` would manufacture false defects at
    every point below the floor, where both backends deliberately return the
    clamped value.
    """
    if a > FLOOR:
        return a * mp.log(a), [mp.log(a) + 1], [[1 / a]]
    return a * mp.log(FLOOR), [mp.log(FLOOR)], [[mp.mpf(0)]]


def o_centropy(a, b):
    """``x*log(max(x, 1e-300)/y)`` -- floor on the NUMERATOR only."""
    if a > FLOOR:
        v = a * (mp.log(a) - mp.log(b))
        g = [mp.log(a) - mp.log(b) + 1, -a / b]
        h = [[1 / a, -1 / b], [-1 / b, a / b**2]]
    else:
        v = a * (mp.log(FLOOR) - mp.log(b))
        g = [mp.log(FLOOR) - mp.log(b), -a / b]
        h = [[mp.mpf(0), -1 / b], [-1 / b, a / b**2]]
    return v, g, h


def o_signpower3(a):
    p, s, ab = mp.mpf(3), mp.sign(a), abs(a)
    return s * ab**p, [p * ab ** (p - 1)], [[p * (p - 1) * s * ab ** (p - 2)]]


def o_abs(a):
    return abs(a), [mp.sign(a)], [[mp.mpf(0)]]


def o_sign(a):
    return mp.sign(a), [mp.mpf(0)], [[mp.mpf(0)]]


def o_exp(a):
    e = mp.e**a
    return e, [e], [[e]]


def o_log(a):
    return mp.log(a), [1 / a], [[-1 / a**2]]


def o_sqrt(a):
    r = mp.sqrt(a)
    return r, [1 / (2 * r)], [[-1 / (4 * r**3)]]


TAILS = [-745.0, -700.0, -300.0, -40.0, -1.0, 0.0, 1.0, 40.0, 300.0, 700.0, 745.0]

# (name, n_vars, builder, oracle, points, kink points)
# Kink points are reported but NOT scored: subgradient ties and clamp crossovers
# where more than one answer is defensible.
CASES = [
    (
        "log1p",
        1,
        lambda v: dm.log1p(v[0]),
        o_log1p,
        [
            [-0.99],
            [-0.5],
            [-1e-3],
            [-1e-8],
            [-1e-13],
            [-1e-17],
            [0.0],
            [1e-17],
            [1e-13],
            [1e-8],
            [1e-5],
            [4.9e-4],
            [5e-4],
            [5.1e-4],
            [1.778e-3],
            [0.5],
            [1.0],
            [1e8],
            [1e100],
            [1e300],
        ],
        set(),
    ),
    (
        "log2",
        1,
        lambda v: dm.log2(v[0]),
        o_log2,
        [[1e-300], [1e-30], [1e-8], [0.5], [1.0], [2.0], [1e8], [1e300]],
        set(),
    ),
    ("sigmoid", 1, lambda v: dm.sigmoid(v[0]), o_sigmoid, [[a] for a in TAILS], set()),
    (
        "softplus",
        1,
        lambda v: dm.softplus(v[0]),
        o_softplus,
        [[a] for a in TAILS],
        {(0.0,)},  # softplus contains max(a, 0); the kink is exactly at 0
    ),
    (
        "entropy",
        1,
        lambda v: FunctionCall("entropy", v[0]),
        o_entropy,
        [[1e-320], [1e-300], [1e-299], [1e-30], [1e-8], [0.5], [1.0], [2.718281828], [100.0]],
        {(1e-300,)},  # exactly the clamp crossover: a max() subgradient tie
    ),
    (
        "centropy",
        2,
        lambda v: FunctionCall("centropy", v[0], v[1]),
        o_centropy,
        [
            [1e-320, 1.0],
            [1e-300, 1.0],
            [1e-30, 1.0],
            [1e-8, 1e-8],
            [0.5, 0.5],
            [1.0, 2.0],
            [2.0, 1e-30],
            [100.0, 3.0],
            [1e300, 1e300],
        ],
        {(1e-300, 1.0)},
    ),
    (
        "signpower",
        1,
        lambda v: FunctionCall("signpower", v[0], Constant(3.0)),
        o_signpower3,
        [[-100.0], [-2.0], [-1e-8], [1e-8], [2.0], [100.0]],
        set(),
    ),
    (
        "abs",
        1,
        lambda v: abs(v[0]),
        o_abs,
        [[-1e300], [-1.0], [-1e-300], [0.0], [1e-300], [1.0], [1e300]],
        {(0.0,)},
    ),
    ("sign", 1, lambda v: dm.sign(v[0]), o_sign, [[-1.0], [-1e-300], [1e-300], [1.0]], set()),
    # Controls: native tape opcodes, no rewrite. Must stay clean in every column.
    ("exp", 1, lambda v: dm.exp(v[0]), o_exp, [[-700.0], [-1.0], [0.0], [1.0], [700.0]], set()),
    ("log", 1, lambda v: dm.log(v[0]), o_log, [[1e-300], [1e-8], [1.0], [1e8], [1e300]], set()),
    ("sqrt", 1, lambda v: dm.sqrt(v[0]), o_sqrt, [[1e-300], [1e-8], [1.0], [1e8], [1e300]], set()),
]

ORDERS = ["value", "grad", "hess"]

# Known residuals, each with the evidence that it is not this branch's doing.
# This is an allowlist, not a tolerance relaxation: anything NOT listed fails the
# run, and a listed entry that has silently started passing is also reported so
# the list cannot go stale.
#
# EMPTY, and that is the point: every entry this list once held has been fixed at
# the root rather than tolerated. All four were one defect class -- a derivative
# rule that materializes a SQUARED quantity, so an intermediate leaves binary64
# range while the answer it is computing sits comfortably inside it:
#
#   entropy  hess @ 1e-299        log'' = -1/m**2 = -1e598, while (x log x)'' =
#                                 1/x = 1e299 is an ordinary number
#   centropy grad/hess @ 1e300    quotient rule forms y**2 = 1e600, while
#                                 -x/y**2 = -1e-300 is representable
#   centropy hess @ 1e-320        clamped branch logged floor/y, so log'' formed
#                                 -1/q**2 with q = 1e-300
#
# Fixed by pounce #489 (Kahan quotient rule; fused xlogx/centropy opcodes whose
# rules never square anything) and by folding log(floor) to a constant in
# `_nl_expr_compiler`. Keep this dict empty: a new entry needs the evidence that
# the residual is not the branch's own doing, and "hard to fix" is not that.
KNOWN_TAPE_RESIDUALS: dict = {}


def make_evaluator(backend, n, build):
    os.environ["DISCOPT_NLP_EVAL"] = backend
    for mod in [k for k in list(sys.modules) if "tape_nlp" in k]:
        del sys.modules[mod]
    from discopt._tape_nlp_evaluator import build_evaluator

    m = dm.Model("audit")
    vs = [m.continuous(f"x{i}", lb=-1e309, ub=1e309) for i in range(n)]
    m.minimize(build(vs))
    m.subject_to(sum(vs) <= 1e309)  # linear: contributes no curvature

    def _jax_factory():
        from discopt._relax.nlp_evaluator import cached_evaluator

        return cached_evaluator(m)

    ev = build_evaluator(m, _jax_factory)
    assert ("Tape" in type(ev).__name__) == (backend == "tape"), type(ev).__name__
    return ev


def relerr(got, truth):
    """Relative error against high-precision truth.

    Order matters: ask whether TRUTH is representable as a double first. When the
    true magnitude overflows binary64, a correctly-signed inf is the best a double
    can do and scores clean; a finite answer there is the defect.
    """
    if abs(truth) >= DBL_MAX:
        if np.isfinite(got):
            return float("inf")
        return 0.0 if (got > 0) == (truth > 0) else float("inf")
    if not np.isfinite(got):
        return float("inf")
    return float(abs(mpf(float(got)) - truth) / max(abs(truth), mpf("1e-300")))


def main():
    rows, kinks = [], []
    compared = 0
    bad = {"tape": [], "jax": []}

    for name, n, build, oracle, pts, kinkset in CASES:
        evs = {"tape": make_evaluator("tape", n, build), "jax": make_evaluator("jax", n, build)}
        worst = {"tape": [0.0] * 3, "jax": [0.0] * 3}

        for p in pts:
            x = np.array(p, dtype=float)
            lam = np.zeros(1)
            ov, og, oh = oracle(*[mpf(float(c)) for c in p])
            errs = {}
            for who, e in evs.items():
                v = float(e.evaluate_objective(x))
                g = np.asarray(e.evaluate_gradient(x), dtype=float).ravel()
                h = np.asarray(e.evaluate_lagrangian_hessian(x, 1.0, lam), dtype=float)
                errs[who] = [
                    relerr(v, ov),
                    max(relerr(g[i], og[i]) for i in range(n)),
                    max(relerr(h[i, j], oh[i][j]) for i in range(n) for j in range(n)),
                ]
            compared += 3

            if tuple(p) in kinkset:
                kinks.append((name, p, errs["tape"], errs["jax"]))
                continue
            for who in ("tape", "jax"):
                for k in range(3):
                    worst[who][k] = max(worst[who][k], errs[who][k])
                    if errs[who][k] > TOL:
                        bad[who].append((name, tuple(p), ORDERS[k], errs[who][k]))
        rows.append((name, len(pts), worst))

    hdr = (
        f"{'operator':10s} {'pts':>4s} | {'TAPE val':>10s} {'TAPE grad':>10s} {'TAPE hess':>10s}"
        f" | {'JAX val':>10s} {'JAX grad':>10s} {'JAX hess':>10s}"
    )
    print("\nworst relative error vs 600-digit analytic truth (kink points excluded)")
    print(hdr)
    print("-" * len(hdr))
    for name, npts, w in rows:
        print(
            f"{name:10s} {npts:4d} | "
            f"{w['tape'][0]:10.3e} {w['tape'][1]:10.3e} {w['tape'][2]:10.3e} | "
            f"{w['jax'][0]:10.3e} {w['jax'][1]:10.3e} {w['jax'][2]:10.3e}"
        )

    print(f"\nkink / clamp-crossover points (reported, not scored): {len(kinks)}")
    for name, p, te, je in kinks:
        t = [f"{v:.2e}" for v in te]
        j = [f"{v:.2e}" for v in je]
        print(f"   {name:10s} @ {p}  tape {t}  jax {j}")

    print(f"\nJAX points over tol {TOL:g}: {len(bad['jax'])}")
    for name, p, o, e in bad["jax"]:
        print(f"   {name:10s} {o:5s} @ {p}  relerr {e:.3e}")
    print("   (informational: JAX is the historical authority, not the shipped default.")
    print("    Where the tape is clean and JAX is not, the tape is the ACCURATE arm.)")

    unexpected = [b for b in bad["tape"] if (b[0], b[1], b[2]) not in KNOWN_TAPE_RESIDUALS]
    expected = {(b[0], b[1], b[2]) for b in bad["tape"]}
    stale = [k for k in KNOWN_TAPE_RESIDUALS if k not in expected]

    print(
        f"\nTAPE points over tol {TOL:g}: {len(bad['tape'])} "
        f"({len(bad['tape']) - len(unexpected)} known, {len(unexpected)} NEW)"
    )
    for name, p, o, e in bad["tape"]:
        tag = "KNOWN" if (name, p, o) in KNOWN_TAPE_RESIDUALS else "*** NEW ***"
        print(f"   [{tag}] {name:10s} {o:5s} @ {p}  relerr {e:.3e}")
        if (name, p, o) in KNOWN_TAPE_RESIDUALS:
            print(f"        {KNOWN_TAPE_RESIDUALS[(name, p, o)]}")

    if stale:
        print(f"\nSTALE allowlist entries -- these now PASS and should be removed: {len(stale)}")
        for k in stale:
            print(f"   {k}")

    print(f"\nderivative comparisons executed: {compared}")
    if compared == 0:
        print("PROBE ASSERTED NOTHING", file=sys.stderr)
        return 1
    if unexpected:
        print(
            f"FAIL: {len(unexpected)} tape defect(s) outside the known-residual set",
            file=sys.stderr,
        )
        return 2
    if stale:
        print("FAIL: known-residual allowlist is stale", file=sys.stderr)
        return 3
    print("PASS: no tape defect outside the documented residuals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
