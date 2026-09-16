"""#1276 (C) entry experiment: would an N-ARY registered atom buy anything?

What C proposes is not a new envelope -- it is a new NODE. ``canonical_expr._sum``
flattens nested sums, so a multivariate composite written in primitives never
exists as a node of its own: its terms become siblings of every other objective
term, ``uniform_relax._try_convex_lift`` is offered the whole (mixed-curvature)
sum, abstains, and descends to the individual terms. Registering the composite
makes it one node and routes it, with a derived Hessian verdict, into that same
lift. So every question C raises is answerable before writing the registry, by
comparing three spellings of identical mathematics:

    alone      the composite IS the objective        -> can the lift certify it at all?
    flattened  composite + an indefinite blocker     -> is the certificate lost?
    grouped    the same, behind ``g == composite``   -> a zero-implementation
                                                        stand-in for the registration,
                                                        and the ceiling on C's gain

#1276's KILL CRITERION: if the lift already fires on the primitive spelling and
the bounds match, the registration buys nothing and the issue should be closed
rather than built.

RESULT -- the criterion fires, for two independent reasons.

**1. On the CALPHAD class C was motivated by, the lift never certifies the
composite at any box width, so registration has nothing to hand it.** Probed over
12 random sub-boxes at each of six widths from 1.0 down to 0.005 (288 probes):

    composite               w=1     w=0.5   w=0.25  w=0.1   w=0.02  w=0.005
    cef cross product        0/12    0/12    0/12    0/12    0/12    0/12
    cef excess (RK x z)      0/12    0/12    0/12    0/12    0/12    0/12
    cef full objective       0/12    0/12    0/12    0/12    0/12    0/12
    CONTROL convex sum-exp  12/12   12/12   12/12   12/12   12/12   12/12

This is structural, not a budget artefact: a CEF objective's cross terms are
bilinear, whose Hessian ``[[0, c], [c, 0]]`` has eigenvalues ``+-|c|`` on every
box, so the interval-Gershgorin certificate can never be sign-definite. The
control fires at every width, so the probe is known to be able to say yes.

**2. Where a composite DOES certify, recovering it is worth nothing, because its
terms already carry tight envelopes.** Flattening genuinely loses the certificate
(a real effect: ``sum-exp`` and ``entropy`` each certify alone, score 0 flattened
and 1 again grouped) -- but measured against a dense-sampled truth, both arms sit
at a 0.00% root gap either way.

**3. Even on the composite best suited to C, the lift is not the answer.**
``log(exp x + exp y + exp z)`` is convex as a whole while the factorable path
relaxes the outer ``log`` as CONCAVE over a separately relaxed exp-sum -- the
worst case term-wise relaxation has. Here the lift DOES fire in both spellings,
and still:

    arm                    root bound    truth     root gap
    flattened, atom off      -0.10865   1.55196     107.00%
    grouped,   atom off      -0.09679   1.55196     106.24%
    flattened, atom ON        1.55195   1.55196       0.00%
    grouped,   atom ON        1.55195   1.55196       0.00%

Grouping is worth 0.76 points of a 107-point gap. What closes it is the
purpose-built log-sum-exp treatment (``DISCOPT_LOGSUMEXP_ATOM``, default OFF),
which is a hand-written envelope, not something derivable from a lowering -- and a
user-supplied envelope is exactly what component A deliberately excluded.

So C is declined. (Note for #632's ATOM-REDUNDANCY-REVIEW cluster, out of scope
here: the last two rows are a default-OFF flag taking a 107% root gap to 0.00%.)
"""

from __future__ import annotations

import itertools
import os

import discopt.modeling as dm
import numpy as np
from discopt._relax.uniform_relax import build_uniform_relaxation

R_T = 8.314 * 1000.0
L0, L1 = 20000.0, 5000.0
G0 = np.random.default_rng(0).normal(0.0, 8000.0, size=(2, 2))
BLOCK_C = 3.0


def _lifts(model) -> tuple[int, list[str]]:
    rel = build_uniform_relaxation(model)
    kinds = [k for k, _ in rel.coverage.values()]
    return kinds.count("composite_convex"), sorted(set(kinds))


# --------------------------------------------------------------------------- #
# 1. Does the lift certify a CEF composite on ANY box?
# --------------------------------------------------------------------------- #
def _standalone(nv, los, his, make) -> bool:
    m = dm.Model("probe")
    xs = [m.continuous(f"x{i}", lb=float(los[i]), ub=float(his[i])) for i in range(nv)]
    m.minimize(make(xs))
    return _lifts(m)[0] > 0


CEF_CASES = {
    "cef cross product": (2, lambda x: float(G0[0, 0]) * x[0] * x[1]),
    "cef excess (RK x z)": (
        3,
        lambda x: x[0] * (1 - x[0]) * (L0 + L1 * (2 * x[0] - 1)) * x[1]
        + R_T * (dm.xlogx(x[0]) + dm.xlogx(x[2])),
    ),
    "cef full objective": (
        4,
        lambda x: sum(float(G0[i, j]) * x[i] * x[2 + j] for i in range(2) for j in range(2))
        + R_T * sum(dm.xlogx(v) for v in x),
    ),
    "CONTROL convex (sum exp)": (3, lambda x: dm.exp(x[0] + x[1]) + dm.exp(x[1] + x[2])),
}
WIDTHS = [1.0, 0.5, 0.25, 0.1, 0.02, 0.005]
SAMPLES = 12


# --------------------------------------------------------------------------- #
# 2/3. flattened vs grouped, against a dense-sampled truth
# --------------------------------------------------------------------------- #
def _three_arm_model(kind, make, lo, hi, extra_row=None):
    m = dm.Model(kind)
    x = [m.continuous(f"x{i}", lb=lo, ub=hi) for i in range(3)]
    u = m.continuous("u", lb=0.2, ub=2.0)
    v = m.continuous("v", lb=0.2, ub=2.0)
    if extra_row is not None:
        m.subject_to(x[0] + x[1] + x[2] == extra_row)
    blocker = BLOCK_C * u * v
    if kind == "alone":
        m.minimize(make(x))
    elif kind == "flattened":
        m.minimize(make(x) + blocker)
    elif kind == "grouped":
        g = m.continuous("g", lb=-1e5, ub=1e5)
        m.subject_to(g == make(x))
        m.minimize(g + blocker)
    else:
        raise AssertionError(kind)
    return m


def _truth_free(np_make, lo, hi, n=161):
    grid = np.linspace(lo, hi, n)
    best = min(np_make(a, b, c) for a, b, c in itertools.product(grid, repeat=3))
    return float(best) + BLOCK_C * 0.2 * 0.2


def _truth_simplex(np_make, lo, hi, total, n=401):
    grid = np.linspace(lo, hi, n)
    best = np.inf
    for a in grid:
        for b in grid:
            c = total - a - b
            if lo <= c <= hi:
                best = min(best, float(np_make(a, b, c)))
    assert np.isfinite(best), "the truth grid enclosed no feasible point"
    return best + BLOCK_C * 0.2 * 0.2


ARM_CASES = [
    (
        "sum-exp",
        lambda x: dm.exp(x[0] + x[1]) + dm.exp(x[1] + x[2]) + (x[0] - x[2]) * (x[0] - x[2]),
        lambda a, b, c: np.exp(a + b) + np.exp(b + c) + (a - c) ** 2,
        -1.0,
        1.0,
        None,
    ),
    (
        "entropy",
        lambda x: R_T * (dm.xlogx(x[0]) + dm.xlogx(x[1]) + dm.xlogx(x[2])),
        lambda a, b, c: R_T * (a * np.log(a) + b * np.log(b) + c * np.log(c)),
        1e-6,
        1.0,
        None,
    ),
    (
        "log-sum-exp",
        lambda x: dm.log(dm.exp(x[0]) + dm.exp(x[1]) + dm.exp(x[2])),
        lambda a, b, c: np.log(np.exp(a) + np.exp(b) + np.exp(c)),
        -2.0,
        2.0,
        1.0,
    ),
]


def main() -> int:
    probes = 0
    rng = np.random.default_rng(5)

    print("1. does the lift certify the composite STANDING ALONE, at any width?")
    print(f"{'composite':>26} | " + "  ".join(f"w={w:<6g}" for w in WIDTHS))
    cef_hits = 0
    control_hits = 0
    for label, (nv, make) in CEF_CASES.items():
        cells = []
        for w in WIDTHS:
            hits = 0
            for _ in range(SAMPLES):
                lo = rng.uniform(1e-6, max(1e-6 + 1e-9, 1.0 - w), size=nv)
                hi = np.minimum(lo + w, 1.0)
                hits += bool(_standalone(nv, lo, hi, make))
                probes += 1
            cells.append(f"{hits:2d}/{SAMPLES}  ")
            if label.startswith("CONTROL"):
                control_hits += hits
            else:
                cef_hits += hits
        print(f"{label:>26} | " + "".join(cells), flush=True)
    assert control_hits == len(WIDTHS) * SAMPLES, (
        f"the control certified only {control_hits} times — the probe cannot say yes, "
        "so its zeros mean nothing"
    )

    print("\n2/3. flattened vs grouped, against a dense-sampled truth")
    print(
        f"{'composite':>12} {'arm':>10} {'atom':>5} | {'root bound':>12} "
        f"{'truth':>12} {'gap':>9} {'#lift':>5}"
    )
    grouped_gain = {}
    for label, make, np_make, lo, hi, total in ARM_CASES:
        truth = (
            _truth_free(np_make, lo, hi)
            if total is None
            else _truth_simplex(np_make, lo, hi, total)
        )
        flags = ("0", "1") if label == "log-sum-exp" else ("0",)
        for flag in flags:
            os.environ["DISCOPT_LOGSUMEXP_ATOM"] = flag
            gaps = {}
            for kind in ("flattened", "grouped"):
                nlift = _lifts(_three_arm_model(kind, make, lo, hi, total))[0]
                r = _three_arm_model(kind, make, lo, hi, total).solve(time_limit=120, max_nodes=1)
                rb = r.root_bound if r.root_bound is not None else r.bound
                scale = max(1.0, abs(truth))
                assert rb <= truth + 1e-4 * scale, f"{label}/{kind}: FALSE BOUND {rb} > {truth}"
                gaps[kind] = (truth - rb) / scale
                print(
                    f"{label:>12} {kind:>10} {flag:>5} | {rb:12.5f} {truth:12.5f} "
                    f"{gaps[kind]:9.2%} {nlift:5d}",
                    flush=True,
                )
                probes += 1
            if flag == "0":
                grouped_gain[label] = gaps["flattened"] - gaps["grouped"]
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)

    print()
    print("grouping the composite (the ceiling on what C could deliver) is worth:")
    for label, gain in grouped_gain.items():
        print(f"    {label:>12}: {gain:+.2%} of root gap")
    worst = max(grouped_gain.values())
    print()
    print(f"best grouping gain across the panel: {worst:.2%}")
    print("PROCEED" if worst > 0.05 else "KILL", "(criterion: >5 points of root gap)")
    print(f"EXECUTED_PROBES={probes}")
    return 0 if probes else 1


if __name__ == "__main__":
    raise SystemExit(main())
