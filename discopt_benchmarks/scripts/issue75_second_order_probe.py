"""ADVERSARIAL: second derivatives through the rewrites, at hostile points.

The hardening (1c54b726) verified VALUES and GRADIENTS. The NLP subsolve consumes
the Lagrangian HESSIAN, which was never compared through the rewrite layer. Second
derivatives are where the rewrites are most fragile:

  * `_log1p` is built from `select`, and BOTH arms of a select evaluate and both
    contribute partials. An arm that is merely unused, not unreachable, can still
    inject inf/nan into a second-order sweep (the classic `jnp.where` trap).
  * the Kahan arm divides by `u - 1`; the quotient rule squares that denominator
    once for the gradient and AGAIN for the Hessian.
  * `softplus` now contains `max(a,0)`, which has no second derivative at 0.

Criterion: a TAPE defect is a point where JAX is finite and the tape is not, or
where both are finite and they disagree beyond FP noise. Where JAX is also
non-finite that is reported separately -- not scored against the tape.

No try/except (§7). Executed-comparison counter, non-zero exit at zero (§6).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import Constant, FunctionCall

# (name, n_vars, builder, hostile points)
CASES = [
    (
        "log1p",
        1,
        lambda v: dm.log1p(v[0]),
        [[-0.9999999], [-0.5], [-1e-17], [0.0], [1e-17], [0.5], [1.0], [1e8], [1e300]],
    ),
    ("log2", 1, lambda v: dm.log2(v[0]), [[1e-300], [1e-8], [0.5], [1.0], [1e8], [1e300]]),
    (
        "sigmoid",
        1,
        lambda v: dm.sigmoid(v[0]),
        [[-745.0], [-300.0], [-40.0], [0.0], [40.0], [300.0], [745.0]],
    ),
    (
        "softplus",
        1,
        lambda v: dm.softplus(v[0]),
        [[-745.0], [-300.0], [-40.0], [0.0], [40.0], [300.0], [700.0], [745.0]],
    ),
    (
        "entropy",
        1,
        lambda v: FunctionCall("entropy", v[0]),
        [[0.0], [1e-320], [1e-300], [1e-30], [1e-5], [0.5], [1.0], [100.0]],
    ),
    (
        "centropy",
        2,
        lambda v: FunctionCall("centropy", v[0], v[1]),
        [[0.0, 1.0], [1e-300, 1.0], [1e-8, 1e-8], [0.5, 0.5], [2.0, 1e-300], [1e300, 1e300]],
    ),
    (
        "signpower",
        2,
        lambda v: FunctionCall("signpower", v[0], Constant(3.0)),
        [[-2.0, 1.0], [-1e-8, 1.0], [0.0, 1.0], [1e-8, 1.0], [2.0, 1.0]],
    ),
    ("abs", 1, lambda v: abs(v[0]), [[-1.0], [-1e-300], [0.0], [1e-300], [1.0]]),
    # controls: native opcodes, no rewrite. Must stay clean in every column.
    ("exp", 1, lambda v: dm.exp(v[0]), [[-700.0], [0.0], [1.0], [700.0]]),
    ("log", 1, lambda v: dm.log(v[0]), [[1e-300], [1.0], [1e300]]),
    ("sqrt", 1, lambda v: dm.sqrt(v[0]), [[1e-300], [1.0], [1e300]]),
]


def make_evaluator(backend, n, build):
    os.environ["DISCOPT_NLP_EVAL"] = backend
    for mod in [k for k in list(sys.modules) if "tape_nlp" in k]:
        del sys.modules[mod]
    from discopt._tape_nlp_evaluator import build_evaluator

    m = dm.Model("adv")
    vs = [m.continuous(f"x{i}", lb=-1e309, ub=1e309) for i in range(n)]
    m.minimize(build(vs))
    # A real constraint so lambda_ is non-empty and the LAGRANGIAN (not just the
    # objective Hessian) is exercised.
    m.subject_to(sum(vs) <= 1e309)

    def _jax_factory():
        from discopt._jax.nlp_evaluator import cached_evaluator

        return cached_evaluator(m)

    ev = build_evaluator(m, _jax_factory)
    return ev, type(ev).__name__


rows = []
compared = 0

for name, n, build, pts in CASES:
    tape_ev, tape_kind = make_evaluator("tape", n, build)
    jax_ev, jax_kind = make_evaluator("jax", n, build)
    assert "Tape" in tape_kind, f"{name}: expected a tape evaluator, got {tape_kind}"
    assert "Tape" not in jax_kind, f"{name}: expected the JAX evaluator, got {jax_kind}"

    tape_only_bad = 0
    both_bad = 0
    worst = 0.0
    worst_at = None
    for p in pts:
        x = np.array(p, dtype=float)
        lam = np.array([1.0])
        th = np.asarray(tape_ev.evaluate_lagrangian_hessian(x, 1.0, lam), dtype=float)
        jh = np.asarray(jax_ev.evaluate_lagrangian_hessian(x, 1.0, lam), dtype=float)
        compared += 1

        tok, jok = bool(np.all(np.isfinite(th))), bool(np.all(np.isfinite(jh)))
        if not jok:
            both_bad += 1
            continue
        if not tok:
            tape_only_bad += 1
            print(f"  !! {name} @ {p}: jax finite {jh.ravel()} | TAPE {th.ravel()}")
            continue
        denom = max(float(np.max(np.abs(th))), float(np.max(np.abs(jh))), 1e-300)
        d = float(np.max(np.abs(th - jh))) / denom
        if d > worst:
            worst, worst_at = d, p
    rows.append((name, len(pts), tape_only_bad, both_bad, worst, worst_at))

hdr = (
    f"{'operator':10s} {'pts':>4s} {'TAPE-ONLY nonfin':>17s} "
    f"{'jax nonfin too':>15s} {'worst rel':>11s}  at"
)
print("\n" + hdr)
print("-" * len(hdr))
defects = 0
for name, n, tob, bb, w, wa in rows:
    flag = "  <== DEFECT" if (tob or w > 1e-10) else ""
    defects += bool(tob or w > 1e-10)
    print(f"{name:10s} {n:4d} {tob:17d} {bb:15d} {w:11.3e}  {wa}{flag}")

print(f"\nhessian comparisons executed: {compared}")
print(f"operators with a hessian defect: {defects}")
if compared == 0:
    print("PROBE ASSERTED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(2 if defects else 0)
