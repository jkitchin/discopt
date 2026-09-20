"""Entry experiment: is `milp_highs.py`'s in-tree interrupt poll a "dead clock"?

`docs/dev/performance-plan.md` recorded, as a still-open defect:

    `terminate_polls == restarts` on this master, so the 1 s in-tree interrupt
    poll never fires: `milp_highs.py` resets `last_poll` at every restart and
    restarts are ~49 ms apart. That is a dead clock on any frequently-restarting
    master.

The observation is correct; the inference is what this script tests. The
invariant `solve_milp_with_lazy_cuts` owes its caller is not "the interrupt arm
fires" but **"the hook is consulted at least every `terminate_poll_s`"** -- and
a restart *is* a consultation, so resetting the interval from it is what a rate
limiter should do. If that reading is right, the max gap between consultations
of ANY context tracks `poll_interval` in every regime, and the interrupt arm
fires precisely where restarts go quiet.

KILL CRITERION (pre-registered, and revised once -- see below). The claim names
a *mechanism*: the restart reset. So the test has to be differential, not a bare
threshold. If the reset starves the hook, the arms with MANY restarts must show
LARGER gaps than the arms with none. If restart-heavy arms show gaps no worse
than restart-free ones, the claimed mechanism is falsified.

The first version of this probe used a bare "max gap > 1.5 x poll_interval on
any arm" instead. That criterion tripped -- at 1.80 s, on the arm with ZERO
restarts, where the claimed mechanism cannot be operating at all. It was
measuring the wrong thing: HiGHS offers `kCallbackMipInterrupt` when it reaches
one, not on a timer, so every gap is `poll_interval` + time-to-next-callback and
a bare threshold scores that inherent sampling lag as if it were the defect. The
threshold is kept below as a SECOND, separate reading -- it is a real fact about
the hook's worst-case resolution, just not evidence for this claim.

CLAUDE.md §6: prints EXECUTED_CONSULTATIONS and exits non-zero if it is zero, so
a run that measured nothing cannot read as a pass. CLAUDE.md §7: the callbacks
do not catch anything -- a raising hook must crash this probe, not be scored as
"continue".

Result (2026-09-20, 118 consultations over five arms): the claimed mechanism is
FALSIFIED -- the 66-restart arm has the *smallest* max gap of the long arms
(1.00 s) and the 0-restart arm the largest (1.80 s), which is the opposite of
what a restart-driven starvation would produce. See the retraction note in
`docs/dev/performance-plan.md` for the full table.
"""

import sys
import time

import numpy as np

import discopt.solvers.milp_highs as _m
from discopt.solvers.milp_highs import solve_milp_with_lazy_cuts

POLL = 1.0


def dense_knapsack(n, seed):
    """Dense random knapsack-ish MILP: many improving incumbents to cut."""
    rng = np.random.default_rng(seed)
    c = -rng.integers(10, 100, size=n).astype(float)
    a = rng.integers(10, 100, size=(3, n)).astype(float)
    return c, a, a.sum(axis=1) * 0.5, [(0.0, 1.0)] * n, np.ones(n, dtype=int)


def market_split(m_rows, n, seed):
    """Cornuejols-Dawande market split: hard for B&B, so ONE tree runs for seconds.

    This is the regime the in-tree poll exists for -- the `rsyn0820m02m` case in
    `solve_milp_with_lazy_cuts`'s docstring, where the master separates rarely
    enough that a restart-only hook has nothing to judge.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 100, size=(m_rows, n)).astype(float)
    b = np.floor(a.sum(axis=1) / 2.0)
    c = np.concatenate([np.zeros(n), np.ones(2 * m_rows)])
    A_eq = np.hstack([a, -np.eye(m_rows), np.eye(m_rows)])
    bounds = [(0.0, 1.0)] * n + [(0.0, 1e4)] * (2 * m_rows)
    integrality = np.concatenate([np.ones(n, dtype=int), np.zeros(2 * m_rows, dtype=int)])
    return c, A_eq, b, bounds, integrality


def run(name, build, n_int, cut_budget, tl, eq):
    c, A, b, bounds, integrality = build
    cuts = [0]
    log = []

    def lazy_cb(x):
        # A no-good cut on the binaries: valid, and it forces a restart, which is
        # all this probe needs from the separator.
        if cuts[0] >= cut_budget:
            return None
        cuts[0] += 1
        ones = x[:n_int] > 0.5
        coeffs = np.zeros(len(c))
        coeffs[:n_int] = np.where(ones, 1.0, -1.0)
        return [(coeffs, float(ones.sum()) - 1.0)]

    def term_cb(snap):
        log.append((float(snap["elapsed"]), str(snap["context"])))
        return False  # never stop: we are measuring cadence, not termination

    kw = {"A_eq": A, "b_eq": b} if eq else {"A_ub": A, "b_ub": b}
    t0 = time.time()
    r = solve_milp_with_lazy_cuts(
        c,
        bounds=bounds,
        integrality=integrality,
        time_limit=tl,
        gap_tolerance=1e-9,
        lazy_callback=lazy_cb,
        terminate_callback=term_cb,
        terminate_poll_s=POLL,
        **kw,
    )
    wall = time.time() - t0

    ctx = {}
    for _, k in log:
        ctx[k] = ctx.get(k, 0) + 1
    ts = [e for e, _ in log]
    # Include start->first and last->exit: a hook starved at either end is just
    # as starved as one starved in the middle.
    gaps = [ts[0]] + [y - x for x, y in zip(ts, ts[1:])] + [wall - ts[-1]] if ts else []

    print(
        f"\n[{name}] wall={wall:.2f}s status={r.status} nodes={r.node_count} "
        f"restarts={ctx.get('restart', 0)} interrupt={ctx.get('interrupt', 0)}",
        flush=True,
    )
    print(f"  consultations={len(log)} by_context={ctx}", flush=True)
    if gaps:
        print(f"  gap: max={max(gaps):.3f}s mean={np.mean(gaps):.3f}s (poll={POLL}s)", flush=True)
    return len(log), gaps, ctx.get("restart", 0)


def main():
    print("FILE:", _m.__file__, flush=True)
    arms = [
        ("knapsack/cut-every-incumbent", dense_knapsack(45, 7), 45, 10_000, 30.0, False),
        ("knapsack/no-cuts", dense_knapsack(55, 11), 55, 0, 30.0, False),
        ("knapsack/3-cuts", dense_knapsack(55, 11), 55, 3, 30.0, False),
        ("marketsplit-6x50/one-long-tree", market_split(6, 50, 3), 50, 0, 25.0, True),
        ("marketsplit-4x30/cut-every-incumbent", market_split(4, 30, 5), 30, 10_000, 25.0, True),
    ]
    total, worst = 0, 0.0
    # Only arms that ran long enough for the interval to matter can say anything
    # about cadence; a solve shorter than one interval polls zero times by design.
    heavy, free = [], []
    for name, build, n_int, budget, tl, eq in arms:
        k, gaps, restarts = run(name, build, n_int, budget, tl, eq)
        total += k
        if not gaps:
            continue
        worst = max(worst, max(gaps))
        if sum(gaps) < 2.0 * POLL:
            continue
        (heavy if restarts > 1 else free).append((name, max(gaps), restarts))

    print(f"\nEXECUTED_CONSULTATIONS={total}", flush=True)
    if total == 0:
        print("PROBE MEASURED NOTHING -- do not read this run as a pass", flush=True)
        return 1
    if not heavy or not free:
        print(
            f"INCONCLUSIVE: need a long arm on BOTH sides of the mechanism "
            f"(restart-heavy={len(heavy)}, restart-free={len(free)})",
            flush=True,
        )
        return 1

    h = max(g for _, g, _ in heavy)
    f = max(g for _, g, _ in free)
    print("  restart-heavy arms: " + ", ".join(f"{n} r={r} maxgap={g:.3f}s" for n, g, r in heavy))
    print("  restart-free  arms: " + ", ".join(f"{n} r={r} maxgap={g:.3f}s" for n, g, r in free))
    print(f"\nMAXGAP_RESTART_HEAVY={h:.3f}s  MAXGAP_RESTART_FREE={f:.3f}s", flush=True)
    print(f"WORST_GAP_ANY_ARM={worst:.3f}s  POLL_INTERVAL={POLL}s", flush=True)

    # (1) The claim under test: does the restart reset starve the hook?
    if h > f:
        print("VERDICT-1 (reset starves the hook): CONFIRMED -- real defect", flush=True)
        rc = 1
    else:
        print(
            f"VERDICT-1 (reset starves the hook): FALSIFIED -- restart-heavy "
            f"max gap {h:.3f}s <= restart-free {f:.3f}s",
            flush=True,
        )
        rc = 0
    # (2) Separate reading: the hook's worst-case resolution vs its nominal one.
    print(
        f"VERDICT-2 (worst-case resolution): {worst / POLL:.2f}x the nominal "
        f"interval -- a caller budgeting by terminate_poll_s should size for this",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
