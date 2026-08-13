"""Entry experiment for the DIRECT backend (CLAUDE.md §4) — run BEFORE implementing.

HYPOTHESIS
    On bound-constrained models whose objective is a non-MCBox ``dm.custom``
    (``CustomCall``) body, a DIRECT global search followed by discopt's local NLP
    finds a strictly better objective than discopt's *current* path for such a
    model, which is a single local NLP from discopt's default start.

WHY THIS IS THE RIGHT PROBE
    ``docs/global_optimization.md`` documents that an opaque ``dm.custom`` body has
    no algebraic relaxation, so the model degrades to the local NLP path with
    ``gap_certified=False`` (``solver.py``, the ``_model_contains_custom_call``
    branch). The claim motivating a DIRECT backend is that this leaves real
    objective value on the table on multimodal problems. That is a measurable
    claim about a *class*, not an instance, so it is measured on a panel.

ARMS (both use discopt for the local step, so the comparison isolates what
DIRECT adds — a better starting point — rather than comparing two local solvers)
    A: ``m.solve()``                              -- today's path, discopt's default start
    B: prototype DIRECT for ``--evals`` evaluations -> best point x*,
       then ``m.solve(initial_solution={x: x*})`` -- same local solver, DIRECT's start

SECOND MEASURED CLAIM (survey conclusion, endorsements 1 and 2; Fig. 15)
    One-side trisection + selecting one rectangle among ties reduces
    evaluations-to-accuracy versus the original "all long sides + accept ties".
    Measured here on the same panel plus the survey's own linear drag function.

KILL CRITERION
    Arm B must strictly improve on >= 2/3 of the panel AND reach 1e-2 relative
    accuracy on >= 3/4. If not, the premise is wrong: record the falsification in
    docs/dev/ and re-scope before writing the production implementation.

MEASUREMENT DISCIPLINE (CLAUDE.md §6-§10)
    * Executed-comparison counter is printed and the script exits non-zero if it
      is zero -- a probe that silently compares nothing must not read as a pass.
    * No bare ``except`` around anything being measured; failures are recorded as
      failures and re-raised at the end if they invalidate the panel.
    * ``discopt.__file__`` is asserted and printed before any measurement.
    * Per-instance progress is flushed as it happens; a long run is not silent.

Usage::

    python -u scripts/direct_entry_experiment.py --evals 2000
    python -u scripts/direct_entry_experiment.py --evals 2000 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Prototype DIRECT  (throwaway; the production version lives in
# python/discopt/solvers/direct.py once this experiment passes)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class _Partition:
    """Rectangles of the unit hypercube, each with a center and a split-count vector.

    Side length of rectangle ``i`` in dimension ``k`` is ``3 ** -t[i, k]``, so the
    center-to-vertex distance is exact and rectangles bucket into discrete size
    levels by ``t`` — no floating-point comparison of sizes anywhere.
    """

    n: int
    centers: list[np.ndarray] = field(default_factory=list)
    t: list[np.ndarray] = field(default_factory=list)
    f: list[float] = field(default_factory=list)

    def d(self, i: int) -> float:
        """Center-to-vertex distance of rectangle ``i`` in the unit cube."""
        sides = 3.0 ** (-self.t[i].astype(np.float64))
        return 0.5 * float(np.sqrt(np.sum(sides**2)))

    def add(self, c: np.ndarray, t: np.ndarray, fv: float) -> None:
        self.centers.append(c)
        self.t.append(t)
        self.f.append(fv)


def _lower_right_hull(pts: list[tuple[float, float, int]]) -> list[int]:
    """Rectangles that minimize ``f - K*d`` for SOME ``K > 0`` — DIRECT's Eq. (3).

    ``pts`` is [(d, f, index)], one entry per size level, sorted by ``d`` ascending.

    Two steps, and the second is easy to get wrong:

    1. Monotone-chain **lower** convex hull. Its vertices, left to right, have
       monotonically increasing supporting slopes.
    2. Restrict to supporting slopes that are **positive**, i.e. keep only the
       suffix beginning at the minimum-``f`` hull vertex. Without this, the
       leftmost (smallest, worst-valued) rectangle is always returned even though
       no ``K > 0`` ever makes it the minimizer — it is optimal only for ``K < 0``,
       which corresponds to preferring *smaller* rectangles with *worse* values.

    Verified against a brute-force sweep over ``K`` in
    ``scripts/direct_entry_experiment`` self-checks.
    """
    hull: list[tuple[float, float, int]] = []
    for d, fv, idx in pts:
        while len(hull) >= 2:
            (d1, f1, _), (d2, f2, _) = hull[-2], hull[-1]
            # drop hull[-1] if it lies on/above the segment hull[-2] -> current
            if (f2 - f1) * (d - d1) >= (fv - f1) * (d2 - d1):
                hull.pop()
            else:
                break
        hull.append((d, fv, idx))
    if not hull:
        return []
    # Step 2: the supporting slope turns positive exactly at the min-f vertex.
    start = min(range(len(hull)), key=lambda i: (hull[i][1], -hull[i][0]))
    return [idx for _, _, idx in hull[start:]]


def direct_minimize(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int = 2000,
    epsilon: float = 1e-4,
    divide: str = "one",
    break_ties: bool = True,
) -> tuple[float, np.ndarray, int, list[tuple[int, float]]]:
    """Prototype DIRECT on a finite box.

    Parameters
    ----------
    divide : {"one", "all"}
        ``"all"`` is the original 1993 rule (sample and trisect every long side).
        ``"one"`` is the Jones 2001 revision: trisect a single long side, chosen
        as the long dimension trisected the fewest times over the whole search,
        ties broken by lower index (survey p. 17 fn. 3).
    break_ties : bool
        ``True`` selects one rectangle per size level among those tied for
        potentially optimal; ``False`` is the original "accept all ties".

    Returns
    -------
    (best_f, best_x, n_evals, history)
        ``history`` is [(eval_count, best_f_so_far)] for convergence plots.
    """
    if divide not in ("one", "all"):
        raise ValueError(f"divide must be 'one' or 'all', got {divide!r}")
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError("DIRECT requires a finite box on every variable")
    n = lb.size
    width = ub - lb

    n_evals = 0
    history: list[tuple[int, float]] = []

    def evaluate(u: np.ndarray) -> float:
        """f at unit-cube point u (mapped to the real box)."""
        nonlocal n_evals
        n_evals += 1
        return float(f(lb + u * width))

    part = _Partition(n)
    c0 = np.full(n, 0.5)
    part.add(c0, np.zeros(n, dtype=np.int64), evaluate(c0))
    best_i = 0
    history.append((n_evals, part.f[0]))

    # Per-dimension trisection counter for the "one long side" tie-break rule.
    split_counts = np.zeros(n, dtype=np.int64)

    while n_evals < max_evals:
        f_min = part.f[best_i]

        # --- select potentially optimal rectangles -------------------------
        # Reduce to the best rectangle per size level (level key = the sorted t
        # vector's induced d; equal t multisets share a d exactly).
        by_level: dict[tuple[int, ...], int] = {}
        for i in range(len(part.f)):
            key = tuple(sorted(part.t[i].tolist()))
            cur = by_level.get(key)
            if cur is None or part.f[i] < part.f[cur]:
                by_level[key] = i
        pts = sorted(((part.d(i), part.f[i], i) for i in by_level.values()), key=lambda p: p[0])
        hull = _lower_right_hull(pts)

        # Condition (4): the rectangle's best achievable Lipschitz bound must beat
        # f_min by epsilon*|f_min|. For a hull point, the smallest admissible K is
        # the slope to the next hull point to its right.
        # Condition (4): f_j - K*d_j <= f_min - eps*|f_min| must hold for some K in
        # the rectangle's admissible range. f - K*d decreases in K, so the best
        # chance is the LARGEST admissible K — the slope to the next hull point.
        # For the rightmost (largest) rectangle K is unbounded above, so f - K*d
        # -> -inf and the condition always holds: the biggest rectangle is always
        # potentially optimal. (Using K = 0 there instead, as a naive reading
        # suggests, all but excludes the largest rectangle and starves the global
        # search.)
        selected: list[int] = []
        dmap = {i: (part.d(i), part.f[i]) for i in hull}
        for pos, i in enumerate(hull):
            di, fi = dmap[i]
            if pos + 1 == len(hull):
                if di > 0.0:
                    selected.append(i)
                continue
            dj, fj = dmap[hull[pos + 1]]
            if dj <= di:
                continue
            # Largest K for which i still beats j:
            #   f_i - K d_i <= f_j - K d_j  <=>  K <= (f_j - f_i) / (d_j - d_i).
            k = (fj - fi) / (dj - di)
            if fi - k * di <= f_min - epsilon * abs(f_min) + 1e-12:
                selected.append(i)
        if not selected:
            selected = [hull[-1]] if hull else [best_i]

        if not break_ties:
            # Original behaviour: every rectangle tied with a selected one at the
            # same size level is also selected.
            expanded: list[int] = []
            for i in selected:
                key = tuple(sorted(part.t[i].tolist()))
                for j in range(len(part.f)):
                    if tuple(sorted(part.t[j].tolist())) == key and part.f[j] <= part.f[i] + 1e-12:
                        expanded.append(j)
            selected = sorted(set(expanded))

        # --- divide the selected rectangles --------------------------------
        progressed = False
        for i in list(selected):
            if n_evals >= max_evals:
                break
            t_i = part.t[i]
            c_i = part.centers[i]
            t_min = int(t_i.min())
            M = np.flatnonzero(t_i == t_min)
            delta = 3.0 ** (-(t_min + 1))

            if divide == "one":
                # Long side trisected the fewest times so far; lower index wins.
                order = sorted(M.tolist(), key=lambda k: (int(split_counts[k]), k))
                dims = [order[0]]
            else:
                dims = M.tolist()

            # Sample c +- delta*e_k for the dimensions we will split.
            w: dict[int, float] = {}
            samples: dict[int, tuple[float, float]] = {}
            for k in dims:
                if n_evals + 2 > max_evals:
                    break
                cp = c_i.copy()
                cp[k] = min(1.0, c_i[k] + delta)
                fp = evaluate(cp)
                cm = c_i.copy()
                cm[k] = max(0.0, c_i[k] - delta)
                fm = evaluate(cm)
                samples[k] = (fm, fp)
                w[k] = min(fm, fp)
            if not samples:
                break
            progressed = True

            # Split in increasing order of w so the best values land in the
            # biggest subrectangles (survey Algorithm 1).
            n_before = len(part.f)
            split_order = sorted(samples.keys(), key=lambda k: (w[k], k))
            t_child = t_i.copy()
            for k in split_order:
                t_child[k] += 1
                split_counts[k] += 1
                fm, fp = samples[k]
                cm = c_i.copy()
                cm[k] = max(0.0, c_i[k] - delta)
                cp = c_i.copy()
                cp[k] = min(1.0, c_i[k] + delta)
                part.add(cm, t_child.copy(), fm)
                part.add(cp, t_child.copy(), fp)
            # The center rectangle shrinks in every split dimension.
            part.t[i] = t_child

            # Track the incumbent over every rectangle added by this division —
            # ``divide="all"`` adds 2 per split dimension, so checking only the
            # last pair silently misses improvements found on earlier dimensions.
            for j in range(n_before, len(part.f)):
                if part.f[j] < part.f[best_i]:
                    best_i = j
            history.append((n_evals, part.f[best_i]))

        if not progressed:
            break

    best_u = part.centers[best_i]
    return part.f[best_i], lb + best_u * width, n_evals, history


# ══════════════════════════════════════════════════════════════════════════════
# Panel: standard multimodal DFO test functions with published optima
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TestFunc:
    name: str
    n: int
    lb: np.ndarray
    ub: np.ndarray
    fstar: float
    jnp_body: Callable  # written with jax.numpy -> opaque to the relaxation layer
    np_body: Callable  # numpy twin, for the DIRECT prototype


def _panel() -> list[TestFunc]:
    import jax.numpy as jnp

    P: list[TestFunc] = []

    def mk(name, n, lo, hi, fstar, body):
        """Register a test function from a single ``body(v, xp)`` written against
        an array module, so the jnp twin (opaque to discopt) and the numpy twin
        (driven by the DIRECT prototype) are the *same* algebra by construction."""
        P.append(
            TestFunc(
                name=name,
                n=n,
                lb=np.broadcast_to(np.asarray(lo, dtype=np.float64), (n,)).copy(),
                ub=np.broadcast_to(np.asarray(hi, dtype=np.float64), (n,)).copy(),
                fstar=fstar,
                jnp_body=lambda v, _b=body: _b(v, jnp),
                np_body=lambda v, _b=body: _b(v, np),
            )
        )

    # --- n = 2 --------------------------------------------------------------
    def branin(v, xp):
        x1, x2 = v[0], v[1]
        return (
            (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * math.pi)) * xp.cos(x1)
            + 10
        )

    mk("branin", 2, [-5.0, 0.0], [10.0, 15.0], 0.397887, branin)

    def camel(v, xp):
        x1, x2 = v[0], v[1]
        return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

    mk("six_hump_camel", 2, [-3.0, -2.0], [3.0, 2.0], -1.0316285, camel)

    def goldstein(v, xp):
        x1, x2 = v[0], v[1]
        a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        b = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return a * b

    mk("goldstein_price", 2, -2.0, 2.0, 3.0, goldstein)

    def shubert(v, xp):
        s1 = sum((i + 1) * xp.cos((i + 2) * v[0] + (i + 1)) for i in range(5))
        s2 = sum((i + 1) * xp.cos((i + 2) * v[1] + (i + 1)) for i in range(5))
        return s1 * s2

    mk("shubert", 2, -10.0, 10.0, -186.7309, shubert)

    def rastrigin(v, xp):
        return 10 * v.shape[0] + xp.sum(v**2 - 10 * xp.cos(2 * math.pi * v))

    # NOTE — the boxes for the origin-centred functions (rastrigin, ackley,
    # griewank) are deliberately ASYMMETRIC. DIRECT's very first evaluation is the
    # centre of the box, so a symmetric box around an optimum at the origin is
    # solved exactly at evaluation 1: relative error 0 for every variant, which
    # makes both the arm-A/arm-B comparison and the evaluations-to-accuracy
    # comparison vacuous. Shifting the box moves the optimum off the centre so the
    # panel measures search rather than a lucky first sample.
    mk("rastrigin_2", 2, -4.12, 6.12, 0.0, rastrigin)
    mk("rastrigin_5", 5, -4.12, 6.12, 0.0, rastrigin)

    def ackley(v, xp):
        nn = v.shape[0]
        return (
            -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(v**2) / nn))
            - xp.exp(xp.sum(xp.cos(2 * math.pi * v)) / nn)
            + 20
            + math.e
        )

    mk("ackley_2", 2, -25.768, 39.768, 0.0, ackley)
    mk("ackley_5", 5, -25.768, 39.768, 0.0, ackley)

    def levy(v, xp):
        w = 1 + (v - 1) / 4
        term1 = xp.sin(math.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + xp.sin(2 * math.pi * w[-1]) ** 2)
        mid = xp.sum((w[:-1] - 1) ** 2 * (1 + 10 * xp.sin(math.pi * w[:-1] + 1) ** 2))
        return term1 + mid + term3

    mk("levy_4", 4, -10.0, 10.0, 0.0, levy)

    def griewank(v, xp):
        idx = np.arange(1.0, v.shape[0] + 1.0)
        return 1 + xp.sum(v**2) / 4000 - xp.prod(xp.cos(v / np.sqrt(idx)))

    mk("griewank_3", 3, -45.0, 75.0, 0.0, griewank)

    # --- Hartman 3 / 6 ------------------------------------------------------
    H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    H3_P = np.array(
        [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0381, 0.5743, 0.8828],
        ]
    )
    H_c = np.array([1.0, 1.2, 3.0, 3.2])

    def hartman3(v, xp):
        return -xp.sum(H_c * xp.exp(-xp.sum(H3_A * (v - H3_P) ** 2, axis=1)))

    mk("hartman_3", 3, 0.0, 1.0, -3.86278, hartman3)

    H6_A = np.array(
        [
            [10.0, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3.0, 3.5, 1.7, 10, 17, 8],
            [17.0, 8, 0.05, 10, 0.1, 14],
        ]
    )
    H6_P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    def hartman6(v, xp):
        return -xp.sum(H_c * xp.exp(-xp.sum(H6_A * (v - H6_P) ** 2, axis=1)))

    mk("hartman_6", 6, 0.0, 1.0, -3.32237, hartman6)

    def michalewicz(v, xp):
        idx = np.arange(1.0, v.shape[0] + 1.0)
        return -xp.sum(xp.sin(v) * xp.sin(idx * v**2 / math.pi) ** 20)

    mk("michalewicz_2", 2, 0.0, math.pi, -1.8013, michalewicz)

    return P


# ══════════════════════════════════════════════════════════════════════════════
# Arms
# ══════════════════════════════════════════════════════════════════════════════


def self_check() -> int:
    """Validate the prototype before believing any verdict it produces.

    Two independent checks, both of which caught real bugs in this file:

    1. The potentially-optimal selection is cross-checked against a brute-force
       sweep over ``K`` — the definition of Eq. (3) rather than a reimplementation
       of it. This caught a sign error in the Eq. (4) slope (``K_max`` was
       negated), which rejected every rectangle except the largest and made the
       search stall, and a missing restriction of the hull to positive supporting
       slopes.
    2. The evaluation counts are compared against the survey's own published
       figures (§2.3.1 and Fig. 15). A prototype that does not reproduce the
       literature it is derived from cannot be used to kill or confirm anything.

    Returns a process exit code; prints an executed-assertion count.
    """
    checks = 0

    def chk(cond: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(msg)

    print("[self-check] selection vs brute-force sweep over K", flush=True)
    rng = np.random.default_rng(0)
    Ks = np.logspace(-6, 6, 20000)
    for trial in range(200):
        d = np.unique(np.sort(rng.uniform(0.01, 1.0, int(rng.integers(1, 9)))))
        fv = rng.uniform(-5.0, 5.0, d.size)
        pts = [(float(d[i]), float(fv[i]), i) for i in range(d.size)]
        want = sorted({int(np.argmin(fv - K * d)) for K in Ks})
        chk(_lower_right_hull(pts) == want, f"hull mismatch on trial {trial}: {pts}")
    print("[self-check]   200 randomized cases matched the definition", flush=True)

    print("[self-check] evaluation counts vs the survey's published figures", flush=True)

    def first(hist, tol):
        return next((ev for ev, v in hist if abs(v - 1.0) <= tol), None)

    lin = lambda v: 1.0 + float(np.sum(v))  # noqa: E731

    # Survey §2.3.1, 1 + x1 + x2 on the unit square, epsilon = 1e-4.
    _, _, _, h = direct_minimize(
        lin, np.zeros(2), np.ones(2), max_evals=4000, divide="all", break_ties=False
    )
    got_1pct, got_001 = first(h, 1e-2), first(h, 1e-4)
    print(
        f"[self-check]   1+x1+x2: 1% at {got_1pct} (survey 90), 0.01% at {got_001} (survey 616)",
        flush=True,
    )
    chk(got_1pct is not None and got_1pct <= 300, "1+x1+x2 must reach 1% well under 300 evals")
    chk(got_001 is not None and got_001 <= 2000, "1+x1+x2 must reach 0.01% under 2000 evals")

    # Survey Fig. 15, 1 + x1 + ... + x5. The two improved variants are the ones
    # with a crisp published number, and they match to within a few evaluations.
    results = {}
    for label, divide, ties in (
        ("all+ties", "all", False),
        ("all+notie", "all", True),
        ("one+notie", "one", True),
    ):
        _, _, _, h5 = direct_minimize(
            lin, np.zeros(5), np.ones(5), max_evals=40000, divide=divide, break_ties=ties
        )
        results[label] = first(h5, 1e-2)
    print(
        f"[self-check]   1+..+x5 to 1%: all+ties={results['all+ties']} (survey 14492), "
        f"all+notie={results['all+notie']} (survey 470), "
        f"one+notie={results['one+notie']} (survey 192)",
        flush=True,
    )
    chk(
        results["all+notie"] is not None and abs(results["all+notie"] - 470) <= 60,
        f"all+notie should land near the published 470, got {results['all+notie']}",
    )
    chk(
        results["one+notie"] is not None and abs(results["one+notie"] - 192) <= 40,
        f"one+notie should land near the published 192, got {results['one+notie']}",
    )
    chk(
        results["all+ties"] is not None and results["all+ties"] > 5000,
        "the original variant should be dramatically worse, per Fig. 15",
    )
    chk(
        results["one+notie"] < results["all+notie"] < results["all+ties"],
        "the survey's ordering of the three variants must be reproduced",
    )

    # Determinism and the finite-box precondition.
    a = direct_minimize(lin, np.zeros(3), np.ones(3), max_evals=400)
    b = direct_minimize(lin, np.zeros(3), np.ones(3), max_evals=400)
    chk(a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2], "must be deterministic")
    try:
        direct_minimize(lin, np.array([-np.inf, 0.0]), np.ones(2))
        chk(False, "an infinite bound must raise")
    except ValueError as exc:
        chk("finite box" in str(exc), f"wrong error for an infinite bound: {exc}")

    print(f"\n[self-check] executed assertions: {checks}", flush=True)
    if checks == 0:
        print("[self-check] ERROR: nothing was asserted", flush=True)
        return 2
    print("[self-check] PASS", flush=True)
    return 0


def _build_model(tf: TestFunc):
    """A discopt model whose objective is an opaque dm.custom body over a finite box."""
    import discopt.modeling as dm

    m = dm.Model(f"direct_entry_{tf.name}")
    x = m.continuous("x", shape=tf.n, lb=tf.lb, ub=tf.ub)
    body = dm.custom(tf.jnp_body, name=tf.name)
    m.minimize(body(x))
    return m, x


def _rel_err(value: float, fstar: float) -> float:
    return abs(value - fstar) / max(1.0, abs(fstar))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=2000, help="DIRECT evaluation budget")
    ap.add_argument("--time-limit", type=float, default=120.0, help="per-solve time limit")
    ap.add_argument("--json", type=str, default=None, help="write results as JSON here")
    ap.add_argument("--only", type=str, default=None, help="comma-separated subset of names")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="validate the prototype against brute force and the survey figures, then exit",
    )
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    # CLAUDE.md §8 — verify which code we actually loaded, before measuring.
    import discopt
    import discopt.modeling as dm

    print(f"discopt.__file__ = {discopt.__file__}", flush=True)
    print(f"discopt.__version__ = {getattr(discopt, '__version__', '?')}", flush=True)
    assert hasattr(dm, "custom"), "dm.custom missing — wrong discopt loaded"
    print(f"numpy {np.__version__}", flush=True)

    panel = _panel()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        panel = [t for t in panel if t.name in keep]
        if not panel:
            raise SystemExit(f"--only matched nothing; names: {[t.name for t in _panel()]}")

    comparisons = 0  # CLAUDE.md §6 — the executed-comparison counter
    rows: list[dict] = []
    failures: list[str] = []

    print(
        f"\n{'instance':<18} {'n':>2} {'A local-only':>14} {'B DIRECT+local':>15} "
        f"{'f*':>12} {'B better?':>10} {'B relerr':>10}",
        flush=True,
    )
    print("-" * 92, flush=True)

    for tf in panel:
        row: dict = {"name": tf.name, "n": tf.n, "fstar": tf.fstar}

        # --- Arm A: today's path (single local NLP from discopt's default start)
        t0 = time.perf_counter()
        m_a, _ = _build_model(tf)
        res_a = m_a.solve(time_limit=args.time_limit)
        row["A_status"] = res_a.status
        row["A_objective"] = None if res_a.objective is None else float(res_a.objective)
        row["A_gap_certified"] = bool(res_a.gap_certified)
        row["A_wall"] = time.perf_counter() - t0

        # --- Arm B: prototype DIRECT, then the SAME local solver from its best point
        t0 = time.perf_counter()
        f_direct, x_direct, n_ev, _hist = direct_minimize(
            tf.np_body, tf.lb, tf.ub, max_evals=args.evals
        )
        row["B_direct_only"] = float(f_direct)
        row["B_evals"] = int(n_ev)

        m_b, xb = _build_model(tf)
        res_b = m_b.solve(time_limit=args.time_limit, initial_solution={xb: x_direct})
        cand = [f_direct]
        if res_b.objective is not None:
            cand.append(float(res_b.objective))
        row["B_refined"] = None if res_b.objective is None else float(res_b.objective)
        row["B_objective"] = float(min(cand))
        row["B_status"] = res_b.status
        row["B_gap_certified"] = bool(res_b.gap_certified)
        row["B_wall"] = time.perf_counter() - t0

        # --- the comparison ------------------------------------------------
        a = row["A_objective"]
        b = row["B_objective"]
        if a is None:
            failures.append(f"{tf.name}: arm A returned no objective (status={res_a.status})")
            row["B_better"] = None
            row["B_relerr"] = _rel_err(b, tf.fstar)
        else:
            comparisons += 1
            row["B_better"] = bool(b < a - 1e-9)
            row["B_relerr"] = _rel_err(b, tf.fstar)
            row["A_relerr"] = _rel_err(a, tf.fstar)

        # Soundness spot-check: neither arm may claim a certificate here.
        if res_a.gap_certified or res_b.gap_certified:
            failures.append(
                f"{tf.name}: a non-MCBox CustomCall model reported gap_certified=True "
                f"(A={res_a.gap_certified}, B={res_b.gap_certified}) — "
                "investigate before trusting this panel"
            )

        print(
            f"{tf.name:<18} {tf.n:>2} "
            f"{'n/a' if a is None else f'{a:14.6g}':>14} "
            f"{b:15.6g} {tf.fstar:12.6g} "
            f"{str(row.get('B_better')):>10} {row['B_relerr']:10.3g}",
            flush=True,
        )
        rows.append(row)

    # ── second measured claim: division / tie-breaking, on the panel + the
    #    survey's own linear drag function ──────────────────────────────────
    print("\nSecond claim — evaluations to 1e-2 relative accuracy by division rule:", flush=True)
    print(
        f"{'instance':<18} {'all+ties':>10} {'all+notie':>10} {'one+notie':>10}",
        flush=True,
    )
    print("-" * 52, flush=True)

    variant_rows: list[dict] = []
    drag = [
        TestFunc(
            f"linear_drag_{k}",
            k,
            np.zeros(k),
            np.ones(k),
            1.0,
            None,
            (lambda v: 1.0 + float(np.sum(v))),
        )
        for k in (2, 5)
    ]
    for tf in drag + panel:
        counts: dict[str, int | None] = {}
        for label, divide, ties in (
            ("all+ties", "all", False),
            ("all+notie", "all", True),
            ("one+notie", "one", True),
        ):
            _, _, _, hist = direct_minimize(
                tf.np_body,
                tf.lb,
                tf.ub,
                max_evals=args.evals,
                divide=divide,
                break_ties=ties,
            )
            hit = next(
                (ev for ev, fv in hist if _rel_err(fv, tf.fstar) <= 1e-2),
                None,
            )
            counts[label] = hit
            comparisons += 1
        variant_rows.append({"name": tf.name, **counts})
        print(
            f"{tf.name:<18} "
            f"{str(counts['all+ties']):>10} {str(counts['all+notie']):>10} "
            f"{str(counts['one+notie']):>10}",
            flush=True,
        )

    # ── verdict ────────────────────────────────────────────────────────────
    scored = [r for r in rows if r.get("B_better") is not None]
    n_better = sum(1 for r in scored if r["B_better"])
    n_accurate = sum(1 for r in rows if r["B_relerr"] <= 1e-2)
    n_total = len(rows)

    print(f"\n{'=' * 92}", flush=True)
    print(f"executed comparisons: {comparisons}", flush=True)
    print(
        f"Arm B strictly better on {n_better}/{len(scored)} scored "
        f"(kill criterion: >= {math.ceil(2 * len(scored) / 3)} needed)",
        flush=True,
    )
    print(
        f"Arm B within 1e-2 relative on {n_accurate}/{n_total} "
        f"(kill criterion: >= {math.ceil(3 * n_total / 4)} needed)",
        flush=True,
    )

    solved = sum(
        1
        for r in variant_rows
        if r["one+notie"] is not None
        and r["all+ties"] is not None
        and r["one+notie"] <= r["all+ties"]
    )
    comparable = sum(
        1 for r in variant_rows if r["one+notie"] is not None and r["all+ties"] is not None
    )
    print(
        f"one-side+no-ties needed <= evaluations of all+ties on {solved}/{comparable} instances",
        flush=True,
    )

    if failures:
        print("\nFAILURES / anomalies:", flush=True)
        for msg in failures:
            print(f"  - {msg}", flush=True)

    verdict_better = n_better >= math.ceil(2 * len(scored) / 3) if scored else False
    verdict_accurate = n_accurate >= math.ceil(3 * n_total / 4) if n_total else False
    passed = verdict_better and verdict_accurate
    print(
        f"\nVERDICT: {'PASS — proceed to implementation' if passed else 'KILL — re-scope'}",
        flush=True,
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "evals": args.evals,
                    "rows": rows,
                    "variants": variant_rows,
                    "comparisons": comparisons,
                    "n_better": n_better,
                    "n_scored": len(scored),
                    "n_accurate": n_accurate,
                    "n_total": n_total,
                    "failures": failures,
                    "passed": passed,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.json}", flush=True)

    # CLAUDE.md §6 — a probe that compared nothing must never read as a pass.
    if comparisons == 0:
        print("ERROR: zero executed comparisons — this probe measured nothing.", flush=True)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
