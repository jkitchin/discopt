"""Reusable soundness harness for bound-changing changes (cert:T0.4).

Phases 2–4 of the certification-gap plan alter the relaxation math (new
reductions, new cuts). Every such change ships behind the *differential bound
test* of §3: on a fixed set of boxes a valid dual bound must

  * never cross the true box optimum   (``bound <= oracle + tol`` for min), and
  * never regress against a baseline    (``new >= old - tol``),

and every emitted cut must be *valid* — it removes no feasible point. This
module provides the two reusable assertions those phases consume, plus small
oracle helpers. It has no dependency on solver internals: callers pass in the
relaxer / oracle / cut as plain callables and arrays, so the harness is cheap to
unit-test and safe to import anywhere.

Sense convention: all bounds are in the *minimization* sense (a dual bound is a
lower bound). For a maximization model, negate before calling (mirroring how the
solver reports internally-minimized bounds).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Box = tuple[np.ndarray, np.ndarray]  # (lower, upper) vectors
BoundFn = Callable[[Box], float]
OracleFn = Callable[[Box], float]


class SoundnessError(AssertionError):
    """Raised when a relaxation bound or a cut violates a soundness property."""


def _as_box(box: Box) -> Box:
    lo, hi = box
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"box lower/upper shape mismatch: {lo.shape} vs {hi.shape}")
    return lo, hi


def assert_bound_sound(
    relaxer_fn: BoundFn,
    boxes: Sequence[Box],
    oracle_fn: OracleFn,
    tol: float = 1e-6,
    *,
    baseline_fn: BoundFn | None = None,
    sense: str = "min",
) -> list[dict]:
    """Assert a relaxation's dual bound is sound (and optionally non-regressing).

    For every box, computes ``new = relaxer_fn(box)`` and ``oracle_fn(box)`` (the
    true box optimum from a trusted dense solve or a stored known optimum) and
    requires, in the minimization sense:

      * **validity** — ``new <= oracle + tol``. A lower bound that exceeds the
        true optimum is a false certificate; this is the catastrophic failure
        mode Phases 2–3 guard against.
      * **non-regression** — when ``baseline_fn`` is given, ``new >= old - tol``
        (the new relaxer is at least as tight as the old one on this box).

    For ``sense="max"`` the inequalities flip (an upper bound must be
    ``>= oracle - tol`` and non-regression means ``new <= old + tol``).

    Returns a per-box report list; raises :class:`SoundnessError` on the first
    violation (so a failing box is easy to reproduce).
    """
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    report: list[dict] = []
    for i, box in enumerate(boxes):
        lo, hi = _as_box(box)
        new = float(relaxer_fn((lo, hi)))
        oracle = float(oracle_fn((lo, hi)))
        old = None if baseline_fn is None else float(baseline_fn((lo, hi)))

        if sense == "min":
            if new > oracle + tol:
                raise SoundnessError(
                    f"box {i}: dual bound {new:.12g} exceeds box optimum "
                    f"{oracle:.12g} by {new - oracle:.3g} > tol {tol:.3g} "
                    f"(false certificate)"
                )
            if old is not None and new < old - tol:
                raise SoundnessError(
                    f"box {i}: new bound {new:.12g} regressed below baseline "
                    f"{old:.12g} by {old - new:.3g} > tol {tol:.3g}"
                )
        else:  # max
            if new < oracle - tol:
                raise SoundnessError(
                    f"box {i}: upper bound {new:.12g} below box optimum "
                    f"{oracle:.12g} by {oracle - new:.3g} > tol {tol:.3g} "
                    f"(false certificate)"
                )
            if old is not None and new > old + tol:
                raise SoundnessError(
                    f"box {i}: new bound {new:.12g} regressed above baseline "
                    f"{old:.12g} by {new - old:.3g} > tol {tol:.3g}"
                )
        report.append({"box": i, "bound": new, "oracle": oracle, "baseline": old})
    return report


def assert_cut_valid(
    cut: tuple[np.ndarray, float],
    feasible_points: Sequence[np.ndarray],
    tol: float = 1e-6,
) -> list[dict]:
    """Assert an inequality cut ``a·x <= b`` removes no feasible point.

    ``cut`` is ``(a, b)``. A cut is *valid* iff every known feasible point
    satisfies it: ``a·x <= b + tol``. A violation means the separator cut away a
    genuinely feasible (and potentially optimal) point — an unsound cut.

    Returns a per-point report; raises :class:`SoundnessError` on the first
    violated point.
    """
    a, b = cut
    a = np.asarray(a, dtype=np.float64)
    b = float(b)
    report: list[dict] = []
    for i, x in enumerate(feasible_points):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != a.shape:
            raise ValueError(
                f"point {i} shape {x.shape} != cut normal shape {a.shape}"
            )
        lhs = float(a @ x)
        slack = b - lhs  # >= -tol required
        if slack < -tol:
            raise SoundnessError(
                f"feasible point {i} violates cut a·x <= b: "
                f"a·x = {lhs:.12g} > b = {b:.12g} (violation {-slack:.3g} > tol {tol:.3g})"
            )
        report.append({"point": i, "lhs": lhs, "rhs": b, "slack": slack})
    return report


def known_optimum_oracle(optima: dict[str, float], instance: str) -> OracleFn:
    """A constant box-independent oracle from a stored known-optimum table.

    Useful when the differential test's boxes are all sub-boxes of the root box
    (so the whole-model optimum is a valid oracle for the box optimum only when
    the box contains the global solution — callers restrict accordingly). Mainly
    a convenience for wiring the global50 known-optima file into the harness.
    """
    value = float(optima[instance])

    def _oracle(_box: Box) -> float:
        return value

    return _oracle
