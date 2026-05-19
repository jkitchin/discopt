"""Fractional factorial designs built by MILP row selection.

A full :math:`2^k` factorial doubles in size with every new factor and
becomes unaffordable past ``k = 7`` or so. A **fractional factorial**
picks :math:`2^{k-p}` of those :math:`2^k` corners so that:

* the chosen subset still spans all :math:`k` factors (every main effect
  is estimable), and
* the columns of the model matrix that we still want to estimate are
  **pairwise orthogonal**, so OLS coefficients have no within-set
  confounding.

The classical generator algebra (defining contrasts, words like
``I = ABCD``) is one way to construct such a fraction. This module takes
a different route: it formulates the row-selection problem as a small
MILP and lets discopt's solver pick the runs.

Formulation
-----------

Let :math:`c_{r,i} \\in \\{-1, +1\\}` be the coded level of factor
:math:`i` in row :math:`r` of the full :math:`2^k` factorial, and let
:math:`x_r \\in \\{0, 1\\}` indicate whether row :math:`r` is kept. For
each pair of effect columns :math:`(u, v)` that we want mutually
orthogonal,

.. math::

    \\sum_{r=1}^{2^k} u_r \\, v_r \\, x_r = 0,

together with :math:`\\sum_r x_r = n_{\\text{runs}}`. Which pairs we
enforce depends on the requested **resolution**:

* **Resolution III** (default): all pairs among
  :math:`\\{I, m_1, \\dots, m_k\\}`. Main effects are clear of one
  another but may be aliased with two-factor interactions.
* **Resolution IV**: resolution-III pairs, plus
  :math:`m_i \\perp m_j m_l` for all distinct triples :math:`(i, j, l)`.
  Equivalently, every three-factor-interaction column is balanced over
  the kept rows. Main effects are clear of any 2FI.
* **Resolution V**: all pairs among
  :math:`\\{I, m_1, \\dots, m_k\\} \\cup \\{m_i m_j : i < j\\}`. Mains
  and two-factor interactions are all mutually orthogonal.

When to use this
----------------

* You have ``k = 4 ... 12`` factors and want a **screening** design but
  cannot afford the full :math:`2^k` corners.
* You want main effects unconfounded with each other (R = III), or with
  two-factor interactions as well (R = IV), or you want to estimate the
  full main + 2FI model (R = V).

If you do not need the structured orthogonality and just want a small
parameter-estimation-friendly design, an exact D-optimal design via
:func:`discopt.doe.optimal_experiment` on a linear template will
typically be the better choice -- it does not constrain the runs to lie
on the :math:`\\{-1, +1\\}` lattice and adapts to your prior model.

Notes on infeasibility
----------------------

Not every (``k``, ``n_runs``, ``resolution``) combination is feasible.
A resolution-V design with ``k = 5`` needs at least 16 runs; a
resolution-IV design with ``k = 4`` needs at least 8. When the MILP
returns infeasible the function raises ``ValueError`` with the
parameter combination and a suggested minimum size.

References
----------
Box, Hunter and Hunter, *Statistics for Experimenters*, 2nd ed., Ch. 6.
Wu and Hamada, *Experiments: Planning, Analysis, and Optimization*, Ch. 5.
"""

from __future__ import annotations

import itertools
import random
from typing import Mapping

import numpy as np

from discopt.doe.screening import FactorialDesign


def fractional_factorial_design(
    factors: Mapping[str, tuple[object, object]],
    *,
    n_runs: int | None = None,
    resolution: int = 3,
    extra_pairs: list[tuple[str, str]] | None = None,
    center_points: int = 0,
    replicates: int = 1,
    seed: int | None = None,
    time_limit: float = 60.0,
) -> FactorialDesign:
    """Build a fractional factorial design by row-selection MILP.

    Parameters
    ----------
    factors : mapping name -> (low, high)
        Two-level factors. Numeric or categorical levels are both
        accepted; categorical levels are coded :math:`-1, +1` for the
        MILP and substituted back into output rows.
    n_runs : int, optional
        Number of corner runs to keep (per replicate, excluding center
        points). Must be a power of 2 in :math:`[k+1,\\, 2^k]`. Defaults
        to the smallest power of 2 that is feasible for the requested
        resolution (``max(k+1, 8)`` rounded up).
    resolution : int, default 3
        Required resolution (3, 4, or 5). See module docstring.
    extra_pairs : list of (str, str), optional
        Additional **factor-name pairs** whose two-factor interaction
        column should be orthogonal to every main effect (i.e. clear of
        aliasing with mains). Useful for R=III designs where the user
        knows a particular 2FI matters and wants it estimable
        separately from the mains.
    center_points : int, default 0
        Center-point runs added per replicate (numeric factors only).
    replicates : int, default 1
        Number of times to replicate the selected fraction.
    seed : int, optional
        Seed for run-order randomization (the MILP itself is
        deterministic).
    time_limit : float, default 60.0
        Solver wall-clock limit in seconds.

    Returns
    -------
    FactorialDesign
        Rows are tagged with ``replicate``, ``run_order``, and
        ``is_center`` exactly like :func:`factorial_2level_design`,
        so :func:`effects_estimates` and :func:`anova_report` apply
        unchanged.
    """
    if not factors:
        raise ValueError("at least one factor required")
    if resolution not in (3, 4, 5):
        raise ValueError(f"resolution must be 3, 4, or 5; got {resolution}")
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
    if k < 2:
        raise ValueError(f"fractional factorial needs k >= 2 factors; got {k}")
    if k > 12:
        raise ValueError(f"fractional factorial supports up to 12 factors via MILP, got {k}")

    N = 2**k
    # Coded {-1, +1} full factorial.
    codes = np.array(list(itertools.product((-1, 1), repeat=k)), dtype=np.int64)  # (N, k)

    # Default n_runs: smallest power of 2 admitting the requested resolution.
    min_runs = _min_runs(k, resolution)
    if n_runs is None:
        n_runs = min_runs
    else:
        if n_runs <= 0 or (n_runs & (n_runs - 1)) != 0:
            raise ValueError(f"n_runs must be a positive power of 2; got {n_runs}")
        if n_runs >= N:
            raise ValueError(
                f"n_runs ({n_runs}) >= full factorial size ({N}); "
                f"use factorial_2level_design instead"
            )
        if n_runs < min_runs:
            raise ValueError(
                f"n_runs={n_runs} is too small for resolution {resolution} "
                f"with k={k} factors; need at least {min_runs}"
            )

    # Build the list of pairs (u_col, v_col) of N-vectors whose dot
    # product over selected rows must vanish.
    intercept = np.ones(N, dtype=np.int64)
    main_cols = [codes[:, i] for i in range(k)]
    twofi_cols: list[tuple[tuple[int, int], np.ndarray]] = []
    for i in range(k):
        for j in range(i + 1, k):
            twofi_cols.append(((i, j), codes[:, i] * codes[:, j]))

    pair_products: list[np.ndarray] = []

    # I ⊥ m_i for every i.
    for i in range(k):
        pair_products.append(intercept * main_cols[i])
    # m_i ⊥ m_j for i < j.
    for i in range(k):
        for j in range(i + 1, k):
            pair_products.append(main_cols[i] * main_cols[j])

    if resolution >= 4:
        # m_l ⊥ m_i m_j for all distinct triples. Equivalent to: every
        # 3FI column has balanced sign over the chosen rows.
        for a, i, j in itertools.combinations(range(k), 3):
            pair_products.append(codes[:, a] * codes[:, i] * codes[:, j])
        # I ⊥ m_i m_j: every 2FI column balanced.
        for (_, _), col in twofi_cols:
            pair_products.append(intercept * col)

    if resolution >= 5:
        # m_i m_j ⊥ m_l m_p for distinct pairs.
        for a in range(len(twofi_cols)):
            for b in range(a + 1, len(twofi_cols)):
                pair_products.append(twofi_cols[a][1] * twofi_cols[b][1])
        # m_l ⊥ m_i m_j (any l, including l in {i,j}) is implied by
        # the m_a*m_b ⊥ m_c relations: when l == i, c_l*c_i*c_j = c_j,
        # which is already enforced via intercept ⊥ m_j.

    if extra_pairs:
        name_to_idx = {n: i for i, n in enumerate(names)}
        for fa, fb in extra_pairs:
            if fa not in name_to_idx or fb not in name_to_idx:
                raise ValueError(f"extra_pairs references unknown factor: {(fa, fb)}")
            ia, ib = name_to_idx[fa], name_to_idx[fb]
            if ia == ib:
                continue
            twofi = codes[:, ia] * codes[:, ib]
            for li in range(k):
                if li in (ia, ib):
                    continue
                pair_products.append(codes[:, li] * twofi)
            # Also intercept ⊥ this 2FI.
            pair_products.append(intercept * twofi)

    selected_rows = _solve_row_milp(N, n_runs, pair_products, time_limit)

    # Map coded rows back to original factor levels.
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for r in range(replicates):
        rep_rows: list[dict[str, object]] = []
        for idx in selected_rows:
            row: dict[str, object] = {
                names[i]: highs[i] if codes[idx, i] == 1 else lows[i] for i in range(k)
            }
            row["replicate"] = r
            row["is_center"] = False
            rep_rows.append(row)
        for _ in range(center_points):
            cp_row: dict[str, object] = {
                names[i]: (float(lows[i]) + float(highs[i])) / 2.0 for i in range(k)
            }
            cp_row["replicate"] = r
            cp_row["is_center"] = True
            rep_rows.append(cp_row)
        rows.extend(rep_rows)

    order = list(range(len(rows)))
    rng.shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: d["run_order"])

    return FactorialDesign(
        factors=names,
        low=tuple(lows),
        high=tuple(highs),
        rows=rows,
    )


def _min_runs(k: int, resolution: int) -> int:
    """Smallest power of 2 admitting an OA of strength (resolution - 1) on k factors.

    Necessary conditions (Rao bounds):

    * R=3: ``n >= k + 1`` and ``n`` a power of 2 with ``n >= 4``.
    * R=4: ``n >= 2k`` (each main and its complement balanced).
    * R=5: ``n >= 1 + k + k(k-1)/2``.
    """
    if resolution == 3:
        lower = max(k + 1, 4)
    elif resolution == 4:
        lower = max(2 * k, 8)
    else:  # 5
        lower = max(1 + k + k * (k - 1) // 2, 16)
    n = 1
    while n < lower:
        n <<= 1
    return n


def _solve_row_milp(
    N: int,
    n_runs: int,
    pair_products: list[np.ndarray],
    time_limit: float,
) -> list[int]:
    """Pick ``n_runs`` of ``N`` rows so each product column sums to 0.

    Returns indices into the full 2^k factorial. Raises ValueError if
    the MILP is infeasible.
    """
    from discopt import Model

    m = Model("fractional_factorial")
    x = m.binary("x", shape=(N,))

    m.subject_to(sum(x[r] for r in range(N)) == n_runs)

    seen: set[tuple[int, ...]] = set()
    for prod in pair_products:
        key = tuple(int(v) for v in prod)
        if key in seen:
            continue
        seen.add(key)
        # Avoid constants (column already balanced or all-same): a column
        # of all-+1 with n_runs odd would be infeasible, but here columns
        # are products of ±1 so sums of constants are caught naturally.
        m.subject_to(sum(int(prod[r]) * x[r] for r in range(N)) == 0)

    # Feasibility objective: minimize 0. The solver may need a real
    # objective; use a deterministic small linear function in r so the
    # solution is unique-up-to-ties.
    m.minimize(sum(r * x[r] for r in range(N)) * 0.0)

    result = m.solve(time_limit=time_limit, gap_tolerance=1e-9)
    if result.x is None or result.status not in {"optimal", "feasible"}:
        raise ValueError(
            f"fractional factorial MILP infeasible or unsolved "
            f"(status={result.status}); try a larger n_runs or lower resolution"
        )

    xv = np.asarray(result.x["x"], dtype=float).ravel()
    selected = [r for r in range(N) if xv[r] > 0.5]
    if len(selected) != n_runs:
        raise ValueError(
            f"MILP returned {len(selected)} runs, expected {n_runs} (status={result.status})"
        )
    return selected


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


__all__ = [
    "fractional_factorial_design",
]
