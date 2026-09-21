"""#1397 entry experiment: is `_EMPTY_INTERVAL_FEAS_TOL` compared against a
quantity that carries the problem's scale?

The constant is 1e-6 and it guards "is this crossover / residual real, or is it
the rounding error of the arithmetic that produced it?" That second question is
NOT scale-free. Summing or differencing floats of magnitude M carries an error of
order n*u*M; at M = 1e10 one ulp is already 1.9e-6, past the whole tolerance. So
on a large box an ordinary rounding crossover is read as a PROVED empty interval
and the node is pruned -- discarding a box that may hold the optimum.

Graded against an ORACLE, not against the solver: each crossover below is
manufactured by walking a known number of ulps up from the upper bound, so the
exact-arithmetic answer ("these bounds describe a nonempty, degenerate box") is
known by construction and `eigvalsh`-style self-grading is impossible.

Arm A (soundness): a pure round-off crossover must NOT be pruned.
Arm B (no weakening): a crossover far beyond both the tolerance and the round-off
bound must STILL be pruned -- a fix that simply inflates the tolerance fails here.

Prints an executed-assertion count and exits non-zero when it is zero (§6).
Exits 1 when any arm fails.
"""

import sys
from pathlib import Path

import numpy as np

MAGNITUDES = (1e0, 1e3, 1e6, 1e8, 1e9, 1e10, 1e11, 1e12, 1e14)
ULPS = (1, 2, 8)
U = float(np.finfo(np.float64).eps)


def _up(value: float, k: int) -> float:
    """*value* advanced by exactly *k* ulps -- a crossover of known, exact size."""
    out = value
    for _ in range(k):
        out = float(np.nextafter(out, np.inf))
    return out


def main() -> int:
    import discopt._relax.nonlinear_bound_tightening as nbt

    marker = hasattr(nbt, "_roundoff_slack")
    print(f"LOAD GATE  module : {nbt.__file__}")
    print(f"LOAD GATE  marker : {'PRESENT' if marker else 'ABSENT'} (_roundoff_slack)")
    print(f"           TOL    : {nbt._EMPTY_INTERVAL_FEAS_TOL}")
    if len(sys.argv) > 1:
        want = sys.argv[1] == "1"
        if marker != want:
            print(f"FAIL: expected marker {'PRESENT' if want else 'ABSENT'}")
            return 2

    checked = 0
    false_infeasible: list[str] = []
    missed_infeasible: list[str] = []

    # ---- Arm A: a pure round-off crossover must survive -------------------------
    for m in MAGNITUDES:
        for k in ULPS:
            ub = m
            lb = _up(ub, k)            # lb > ub by exactly k ulps of m
            crossover = lb - ub
            arr_lb = np.array([lb], dtype=np.float64)
            arr_ub = np.array([ub], dtype=np.float64)
            nbt._snap_tolerant_crossovers(arr_lb, arr_ub)
            snapped = arr_lb[0] <= arr_ub[0]
            checked += 1
            if not snapped:
                false_infeasible.append(
                    f"M={m:.0e} k={k}ulp crossover={crossover:.3e} "
                    f"(= {crossover / (U * max(1.0, m)):.2f} u*M) left crossed -> pruned"
                )

    # ---- Arm B: a genuine crossover must still be pruned ------------------------
    for m in MAGNITUDES:
        ub = m
        # 1e6 x the largest round-off bound any sane slack could claim, and at
        # least 1.0, so this is unambiguously a real empty interval at every scale.
        genuine = max(1.0, 1e6 * 64.0 * U * max(1.0, m))
        lb = ub + genuine
        arr_lb = np.array([lb], dtype=np.float64)
        arr_ub = np.array([ub], dtype=np.float64)
        nbt._snap_tolerant_crossovers(arr_lb, arr_ub)
        checked += 1
        if arr_lb[0] <= arr_ub[0]:
            missed_infeasible.append(f"M={m:.0e} crossover={genuine:.3e} was snapped away")

    print(f"\nEXECUTED ASSERTIONS: {checked}")
    if checked == 0:
        print("FAIL: the probe compared nothing (§6)")
        return 1

    print(f"\nArm A -- round-off crossovers wrongly pruned: {len(false_infeasible)}")
    for s in false_infeasible:
        print(f"  {s}")
    print(f"Arm B -- genuine crossovers wrongly snapped: {len(missed_infeasible)}")
    for s in missed_infeasible:
        print(f"  {s}")

    if false_infeasible or missed_infeasible:
        print("\nFAIL")
        return 1
    print("\nPASS: no false infeasibility from round-off, no genuine one lost")
    return 0


if __name__ == "__main__":
    sys.exit(main())
