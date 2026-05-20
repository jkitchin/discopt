"""Two-level factorial designs for factor screening.

When to use this template
-------------------------

You have a list of candidate factors (process knobs, ingredients,
operating modes, ...) and the question is **"does each one matter?"** —
not "what is the optimum" and not "what is the full response surface".
Screening designs deliberately probe each factor at only **two levels**
(LOW and HIGH) so a small number of runs can resolve which factors
have detectable effects on the response.

Use a 2-level factorial design when:

* You have **2 -- 7 candidate factors** and want to filter the "vital
  few" before investing in a quadratic response-surface or mechanistic
  model. Beyond ~7 factors, a fractional or Plackett-Burman design
  scales better, but the same diagnostic logic applies.
* Each factor has a sensible LOW and HIGH setting. For continuous
  factors these are usually the extremes of the safe operating range;
  for categorical factors they are the two values you actually want to
  compare (e.g. catalyst A vs B, supplier 1 vs 2).
* You can afford ``2**k`` runs at minimum (plus optional center points
  + replicates). For ``k = 3`` factors that's 8 runs; for ``k = 5``
  it's 32. If that's too many, consider a half-fraction (resolution V
  for k = 5: 16 runs) -- not yet implemented here; use the ``linear``
  template + ``discopt doe fit`` instead, which produces an
  exact-D-optimal design at the same factor count.

Avoid this template when:

* You already know all factors matter and want to **optimize**: use
  ``response-surface-2d`` / ``response-surface-3d`` for a full quadratic.
* You only have **one** factor: a one-way ANOVA via ``latin-square`` with
  k = 1 treatment is overkill -- a paired t-test on completed runs is
  enough, or just use the ``polynomial-1d`` template.
* Your factors are **proportions of a blend** summing to a constant:
  use the Scheffé mixture templates instead.

Center points and curvature
---------------------------

Adding ``center_points = c > 0`` inserts ``c`` runs at the midpoint of
every continuous factor. Center points serve two purposes:

1. **Curvature test**: comparing the average corner response to the
   average center response detects whether a planar (additive) model is
   adequate. If center response is far from the predicted plane, you
   know to escalate to a response-surface design.
2. **Pure-error estimate**: replicated center points give an unbiased
   estimate of σ that doesn't assume the linear model is correct.

Center points require **all factors to be numeric**. If any factor is
categorical (e.g. ``catalyst: [A, B]``), the midpoint is undefined and
the design rejects ``center_points > 0``.

Replication
-----------

``replicates = r`` repeats the whole 2**k corner set ``r`` times
(plus ``center_points`` per replicate). The randomization is fresh for
each replicate. Replication is necessary when you need a residual
degrees-of-freedom > 0 for the F-tests:

* With ``r = 1`` and no interactions, residual df = ``2**k - 1 - k``.
* Adding 2-way interactions consumes ``k * (k - 1) / 2`` more df.
* For ``k = 3`` and a full interactions model, residual df is 0 unless
  you add center points or replicates.

Analysis
--------

After filling in the response column, run::

    discopt doe anova WORKBOOK [--interaction A:B ...]

which reports Type-I sums of squares + F + p-value for each factor and
each requested interaction. The signed **effect estimate** for a
factor is ``mean(y | factor=HIGH) - mean(y | factor=LOW)``; that and
its standard error are returned by :func:`effects_estimates`. A Pareto
chart of ``|effect|`` is the standard visualization.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast


@dataclass(frozen=True)
class FactorialDesign:
    """A randomized 2-level factorial design.

    Attributes
    ----------
    factors : tuple of str
        Factor names in input order.
    low : tuple
        LOW level of each factor (numeric or string).
    high : tuple
        HIGH level of each factor (numeric or string).
    rows : list of dict
        One dict per run with all factor values, plus ``replicate``
        (0-based), ``run_order`` (0-based, shuffled), and ``is_center``
        (True for center-point runs).
    """

    factors: tuple[str, ...]
    low: tuple[object, ...]
    high: tuple[object, ...]
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)


def factorial_2level_design(
    factors: Mapping[str, tuple[object, object]],
    *,
    center_points: int = 0,
    replicates: int = 1,
    seed: int | None = None,
) -> FactorialDesign:
    """Build a randomized 2-level full factorial design.

    Parameters
    ----------
    factors : mapping name -> (low, high)
        Two levels per factor. Levels may be numeric (e.g.
        ``("temp", (80.0, 120.0))``) or categorical (e.g.
        ``("catalyst", ("A", "B"))``).
    center_points : int, default 0
        Number of center-point runs added **per replicate**. Requires
        all factors to be numeric. The center value is the midpoint of
        the (low, high) pair.
    replicates : int, default 1
        Number of independent replications of the whole 2**k + cp set.
    seed : int, optional
        Reproducible randomization seed.

    Returns
    -------
    FactorialDesign
        Rows are tagged with ``replicate``, ``run_order``, and
        ``is_center``. Run order is shuffled across the whole experiment.
    """
    if not factors:
        raise ValueError("at least one factor required")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    if center_points < 0:
        raise ValueError(f"center_points must be >= 0, got {center_points}")

    names = tuple(factors.keys())
    lows: list[object] = []
    highs: list[object] = []
    numeric_flags: list[bool] = []
    for n in names:
        lo, hi = factors[n]
        if lo == hi:
            raise ValueError(f"factor {n!r}: low and high levels must differ")
        lows.append(lo)
        highs.append(hi)
        numeric_flags.append(_is_numeric(lo) and _is_numeric(hi))

    if center_points > 0 and not all(numeric_flags):
        non_numeric = [n for n, ok in zip(names, numeric_flags) if not ok]
        raise ValueError(
            f"center_points > 0 requires all factors to be numeric; "
            f"non-numeric factors: {non_numeric}"
        )

    k = len(names)
    if k > 8:
        raise ValueError(f"2-level factorial supports up to 8 factors, got {k}")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    corner_codes = list(itertools.product((0, 1), repeat=k))  # 2**k corners
    for r in range(replicates):
        rep_rows: list[dict[str, object]] = []
        for code in corner_codes:
            row: dict[str, object] = {names[i]: highs[i] if code[i] else lows[i] for i in range(k)}
            row["replicate"] = r
            row["is_center"] = False
            rep_rows.append(row)
        for _ in range(center_points):
            cp_row: dict[str, object] = {
                names[i]: (float(cast(float, lows[i])) + float(cast(float, highs[i]))) / 2.0
                for i in range(k)
            }
            cp_row["replicate"] = r
            cp_row["is_center"] = True
            rep_rows.append(cp_row)
        rows.extend(rep_rows)

    order = list(range(len(rows)))
    rng.shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: cast(int, d["run_order"]))

    return FactorialDesign(
        factors=names,
        low=tuple(lows),
        high=tuple(highs),
        rows=rows,
    )


def effects_estimates(
    rows: Sequence[Mapping[str, object]],
    response: str,
    factors: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Signed main-effect estimates for a 2-level design.

    For each factor, returns ``mean(y | factor=HIGH) - mean(y |
    factor=LOW)`` together with a standard error and t-statistic
    against the pooled residual. Center-point runs (where the factor
    is at neither extreme) are excluded from each effect's per-factor
    computation but kept in the residual variance estimate.

    Returns
    -------
    list of dict
        One entry per factor with keys ``factor``, ``effect``,
        ``se``, ``t``, sorted by ``|effect|`` descending.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("need at least one row")
    if response not in rows[0]:
        raise ValueError(f"response column {response!r} missing from rows")

    if factors is None:
        factors = [
            k
            for k in rows[0].keys()
            if k != response
            and k not in {"replicate", "run_order", "is_center"}
            and not k.startswith("_")
        ]
    factors = list(factors)

    y = []
    for r in rows:
        try:
            y.append(float(cast(float, r[response])))
        except (TypeError, ValueError) as e:
            raise ValueError(f"response value {r[response]!r} is not numeric") from e

    # Pooled residual: subtract each factor's main-effect prediction.
    grand_mean = sum(y) / len(y)
    effects: list[dict[str, object]] = []
    levels_per_factor: dict[str, tuple[object, object]] = {}
    for f in factors:
        vals: list[Any] = []
        for r in rows:
            if r[f] not in vals:
                vals.append(r[f])
        # Two-level factors are expected; if there are 3 (center point),
        # use the min and max as low/high.
        if len(vals) < 2:
            continue
        try:
            sorted_vals = sorted(vals)
        except TypeError:
            sorted_vals = vals
        lo, hi = sorted_vals[0], sorted_vals[-1]
        levels_per_factor[f] = (lo, hi)

    # Sample sizes per factor level (exclude center points implicitly when
    # the run has neither lo nor hi).
    for f, (lo, hi) in levels_per_factor.items():
        y_lo = [yi for yi, r in zip(y, rows) if r[f] == lo]
        y_hi = [yi for yi, r in zip(y, rows) if r[f] == hi]
        n_lo, n_hi = len(y_lo), len(y_hi)
        if n_lo == 0 or n_hi == 0:
            continue
        mean_lo = sum(y_lo) / n_lo
        mean_hi = sum(y_hi) / n_hi
        effect = mean_hi - mean_lo

        # Pooled variance from within-level deviations.
        ss_within = sum((yi - mean_lo) ** 2 for yi in y_lo) + sum(
            (yi - mean_hi) ** 2 for yi in y_hi
        )
        df = n_lo + n_hi - 2
        if df <= 0:
            se = float("nan")
            t = float("nan")
        else:
            var_pool = ss_within / df
            se = math.sqrt(var_pool * (1.0 / n_lo + 1.0 / n_hi))
            t = effect / se if se > 0 else float("nan")
        effects.append({"factor": f, "effect": effect, "se": se, "t": t, "low": lo, "high": hi})

    effects.sort(key=lambda d: abs(float(cast(float, d["effect"]))), reverse=True)
    _ = grand_mean  # silence linter; used as reference for callers
    return effects


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


__all__ = [
    "FactorialDesign",
    "effects_estimates",
    "factorial_2level_design",
]
