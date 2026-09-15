"""Entry experiment for issue #1238 (CLAUDE.md §4) — run BEFORE implementing.

HYPOTHESIS
    An n-ary ``Min``/``Max`` node with the envelope the relaxation layer already
    carries is sound and *strictly tighter* than the binary fold it replaces, so
    lifting the binary-only consumers to n-ary buys bound.

WHY THIS IS THE RIGHT PROBE
    ``uniform_relax._build_multivar`` — reached by ``build_uniform_relaxation``,
    the default per-node engine since #632 — already handles ``len(args) >= 2``
    and emits the convex-hull facets ``w >= a_i`` (``max``) / ``w <= a_i``
    (``min``). So the n-ary arm can be measured TODAY, by constructing the node
    directly, without touching a line of the IR. If it is not tighter there, it
    is not tighter anywhere the default solve path looks.

ARMS (three mathematically identical spellings of the same function)
    left : ``min(min(min(a1, a2), a3), a4)``  -- the only spelling before #1238
    bal  : ``min(min(a1, a2), min(a3, a4))``  -- what #1238 ships
    nary : ``min(a1, a2, a3, a4)``            -- the unshipped IR change

KILL CRITERION (the issue's own)
    If the n-ary envelope is not tighter than the balanced binary fold, ship only
    the balanced fold — it needs no flag, no IR change and no corpus panel, and
    it still removes the ``O(n)`` depth.

RESULT (2026-09-15, recorded in ``docs/dev/performance-plan.md`` §64)
    300 comparisons, 900 soundness checks: n-ary tighter 0, equal 300, looser 0,
    unsound 0. The kill criterion fires.

USAGE
    python -u scripts/entry_1238_minmax_nary_vs_fold.py

    Prints per-point progress and, at the end, an executed-comparison count; it
    exits non-zero when that count is zero, so a probe that traversed nothing
    cannot read as a pass (CLAUDE.md §6).
"""

from __future__ import annotations

import sys

import discopt
import discopt.modeling as dm
import numpy as np
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.modeling.core import FunctionCall

#: Unique to this probe; asserted before anything is measured so a run against
#: the wrong tree cannot be read as a result (CLAUDE.md §8).
MARKER = "entry_1238_minmax_nary_vs_fold"

LB, UB = -1.0, 1.5


# --------------------------------------------------------------------------- #
# the three spellings
# --------------------------------------------------------------------------- #
def left_deep(fname: str, args: list) -> object:
    acc = args[0]
    for a in args[1:]:
        acc = FunctionCall(fname, acc, a)
    return acc


def balanced(fname: str, args: list) -> object:
    cur = list(args)
    while len(cur) > 1:
        nxt = [FunctionCall(fname, cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def nary(fname: str, args: list) -> object:
    return FunctionCall(fname, *args)


SPELLINGS = (("left", left_deep), ("bal", balanced), ("nary", nary))


# --------------------------------------------------------------------------- #
# argument families: each is (symbolic builder, numpy evaluator) so the same
# arguments are relaxed and sampled, not two hand-kept copies.
# --------------------------------------------------------------------------- #
ARG_KINDS = {
    "affine": (
        lambda x, n, c: [c[i] * x[i] + 0.5 * (i - n / 2) for i in range(n)],
        lambda P, n, c: np.stack([c[i] * P[:, i] + 0.5 * (i - n / 2) for i in range(n)], 1),
    ),
    "square": (
        lambda x, n, c: [c[i] * x[i] * x[i] + x[(i + 1) % n] for i in range(n)],
        lambda P, n, c: np.stack([c[i] * P[:, i] ** 2 + P[:, (i + 1) % n] for i in range(n)], 1),
    ),
    "bilinear": (
        lambda x, n, c: [c[i] * x[i] * x[(i + 1) % n] for i in range(n)],
        lambda P, n, c: np.stack([c[i] * P[:, i] * P[:, (i + 1) % n] for i in range(n)], 1),
    ),
    "exp": (
        lambda x, n, c: [c[i] * dm.exp(x[i]) for i in range(n)],
        lambda P, n, c: np.stack([c[i] * np.exp(P[:, i]) for i in range(n)], 1),
    ),
    # every argument over the SAME two variables: if sharing is what an n-ary
    # envelope could exploit, this is where it would show.
    "shared": (
        lambda x, n, c: [c[i] * x[0] + (1.0 - c[i]) * x[1] for i in range(n)],
        lambda P, n, c: np.stack([c[i] * P[:, 0] + (1.0 - c[i]) * P[:, 1] for i in range(n)], 1),
    ),
}


def lp_bound(model) -> float | None:
    """The relaxation's bound, mapped back into the MODEL's objective space.

    ``build_uniform_relaxation`` always minimises internally and negates a
    ``maximize`` model, so the producer's documented ``model = sign * (internal +
    offset)`` (``solver.py``) has to be applied; comparing raw internal objectives
    across senses would read every maximize point as unsound.
    """
    flb, fub = flat_variable_bounds(model)
    rel = build_uniform_relaxation(model, box=(flb, fub))
    res = rel.model.solve(backend="simplex")
    if res.objective is None:
        return None
    return rel.obj_sense_sign * (float(res.objective) + rel.obj_offset)


def main() -> int:
    assert MARKER == "entry_1238_minmax_nary_vs_fold"
    print(f"# discopt from {discopt.__file__}", flush=True)
    print(f"# marker={MARKER}", flush=True)

    rng = np.random.default_rng(1238)
    comparisons = 0
    soundness_checks = 0
    tighter = looser = equal = 0
    unsound: list[tuple] = []
    no_bound: list[tuple] = []

    for kind, (build_args, np_eval) in ARG_KINDS.items():
        for fname in ("min", "max"):
            for sense in ("min", "max"):
                for n in (3, 4, 5, 6, 8):
                    for trial in range(3):
                        c = rng.uniform(-1.5, 1.5, size=n)
                        vals: dict[str, float | None] = {}
                        for label, fold in SPELLINGS:
                            m = dm.Model(f"{kind}_{fname}_{sense}_{n}_{trial}_{label}")
                            x = m.continuous("x", shape=(n,), lb=LB, ub=UB)
                            w = fold(fname, build_args(x, n, c))
                            (m.minimize if sense == "min" else m.maximize)(w)
                            vals[label] = lp_bound(m)

                        comparisons += 1
                        if any(v is None for v in vals.values()):
                            no_bound.append((kind, fname, sense, n, trial))
                            print(f"{kind} {fname} {sense} n={n} t={trial} NO BOUND", flush=True)
                            continue

                        # Sampling gives an INNER estimate of the true box
                        # optimum, so a sound outer bound must lie outside it.
                        pts = rng.uniform(LB, UB, size=(40000, n))
                        arr = np_eval(pts, n, c)
                        red = arr.min(axis=1) if fname == "min" else arr.max(axis=1)
                        truth = float(red.min() if sense == "min" else red.max())

                        for label, v in vals.items():
                            soundness_checks += 1
                            if sense == "min" and v > truth + 1e-6:
                                unsound.append((label, kind, fname, sense, n, trial, v, truth))
                            if sense == "max" and v < truth - 1e-6:
                                unsound.append((label, kind, fname, sense, n, trial, v, truth))

                        # Tightness: a LARGER bound is tighter for a min sense, a
                        # smaller one for a max sense.
                        d = vals["nary"] - vals["bal"]
                        if sense == "max":
                            d = -d
                        if d > 1e-9 * max(1.0, abs(vals["bal"])):
                            tighter += 1
                        elif d < -1e-9 * max(1.0, abs(vals["bal"])):
                            looser += 1
                        else:
                            equal += 1

                        print(
                            f"{kind:9s} {fname:3s} {sense:3s} n={n} t={trial} "
                            f"left={vals['left']:+.10f} bal={vals['bal']:+.10f} "
                            f"nary={vals['nary']:+.10f} truth~{truth:+.6f}",
                            flush=True,
                        )

    print()
    print(f"COMPARISONS_EXECUTED {comparisons}")
    print(f"SOUNDNESS_CHECKS_EXECUTED {soundness_checks}")
    print(f"nary tighter than balanced fold : {tighter}")
    print(f"nary equal   to  balanced fold  : {equal}")
    print(f"nary looser  than balanced fold : {looser}")
    print(f"points with no bound            : {len(no_bound)}")
    print(f"UNSOUND (bound past the truth)  : {len(unsound)}")
    for u in unsound[:10]:
        print("   ", u)

    if comparisons == 0 or soundness_checks == 0:
        print("PROBE EXECUTED ZERO COMPARISONS — the measurement did not happen", file=sys.stderr)
        return 2
    print()
    print(
        "VERDICT: kill criterion FIRES — ship the balanced fold only"
        if tighter == 0
        else "VERDICT: the n-ary envelope is tighter somewhere; re-read §64"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
