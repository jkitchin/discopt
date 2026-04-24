"""
Variable Selection for AMP (Adaptive Multivariate Partitioning).

Implements two strategies from JOGO 2018 for choosing which variables to partition:

1. max_cover: Partition ALL variables appearing in any nonlinear term.
   Gives the tightest relaxation but introduces more binary variables in the MILP.

2. min_vertex_cover: Find the MINIMUM set of variables such that every nonlinear
   term contains at least one selected variable.  Solved as a small MILP.
   Reduces the number of binary variables introduced, keeping the relaxation MILP
   tractable for large problems.

3. auto: Use max_cover when ≤15 candidates, otherwise min_vertex_cover.

Theory:
  Nagarajan et al., JOGO 2018 (Section 3.2):
  "We use a greedy max-cover heuristic or a minimum vertex cover on the interaction
   graph to decide which variables to discretize."
"""

from __future__ import annotations

from discopt._jax.term_classifier import NonlinearTerms


def _all_terms(terms: NonlinearTerms) -> list[tuple[int, ...]]:
    """Flatten all nonlinear terms (bilinear + trilinear) into a single list."""
    all_t: list[tuple[int, ...]] = []
    all_t.extend(terms.bilinear)
    all_t.extend(terms.trilinear)
    return all_t


def _max_cover(terms: NonlinearTerms) -> list[int]:
    """Return all partition candidates (max-cover strategy)."""
    return list(terms.partition_candidates)


def _min_vertex_cover(terms: NonlinearTerms) -> list[int]:
    """Minimum vertex cover of the term-variable interaction graph.

    Formulation (integer program):
        min  sum(y_v for v in candidates)
        s.t. sum(y_v for v in term) >= 1   for each nonlinear term
             y_v in {0, 1}

    Solved using HiGHS (via highspy or discopt's MILP wrapper).
    Falls back to max_cover if HiGHS is not available.
    """
    all_t = _all_terms(terms)
    candidates = list(terms.partition_candidates)

    # Trivial cases
    if not candidates:
        return []
    if not all_t:
        return []
    if len(candidates) <= 2:
        # With ≤2 variables, max_cover and min_vertex_cover are the same
        return candidates

    try:
        return _solve_vertex_cover_milp(candidates, all_t)
    except Exception:
        greedy = _greedy_vertex_cover(candidates, all_t)
        return greedy if greedy else candidates


def _weighted_min_vertex_cover(
    terms: NonlinearTerms,
    distance: dict[int, float],
) -> list[int]:
    """Weighted minimum vertex cover using Alpine-style incumbent distance scores."""
    all_t = _all_terms(terms)
    candidates = list(terms.partition_candidates)

    if not candidates or not all_t:
        return []
    if len(candidates) <= 2:
        return candidates

    try:
        return _solve_vertex_cover_milp(candidates, all_t, weights=distance)
    except Exception:
        greedy = _greedy_vertex_cover(candidates, all_t, weights=distance)
        return greedy if greedy else candidates


def _solve_vertex_cover_milp(
    candidates: list[int],
    terms: list[tuple[int, ...]],
    weights: dict[int, float] | None = None,
) -> list[int]:
    """Solve the minimum vertex cover as a MILP using HiGHS.

    Variables: y_v ∈ {0,1} for each candidate v
    Objective: min sum(y_v)
    Constraints: for each term t: sum(y_v for v in t ∩ candidates) >= 1
    """
    import numpy as np

    n = len(candidates)
    var_to_col = {v: i for i, v in enumerate(candidates)}

    # Build constraint matrix: one row per term
    valid_terms = [t for t in terms if any(v in var_to_col for v in t)]
    m_rows = len(valid_terms)

    if m_rows == 0:
        return []

    # Coefficients for the covering constraints: A @ y >= 1
    A = np.zeros((m_rows, n), dtype=np.float64)
    for row_idx, term in enumerate(valid_terms):
        for v in term:
            if v in var_to_col:
                A[row_idx, var_to_col[v]] = 1.0

    # Objective: minimize sum(y) or a weighted cover objective.
    c: np.ndarray
    if weights is None:
        c = np.ones(n, dtype=np.float64)
    else:
        positive = [
            abs(float(weights.get(v, 0.0)))
            for v in candidates
            if abs(float(weights.get(v, 0.0))) > 0.0
        ]
        heavy = 1.0 if not positive else 1.0 / min(positive)
        c = np.array(
            [
                (heavy if abs(float(weights.get(v, 0.0))) <= 1e-6 else 1.0 / abs(float(weights[v])))
                for v in candidates
            ],
            dtype=np.float64,
        )

    # Convert A >= 1 to standard form: -A @ y <= -1
    A_le = -A
    b_le = -np.ones(m_rows, dtype=np.float64)

    try:
        import highspy

        h = highspy.Highs()
        h.silent()
        h.setOptionValue("time_limit", 30.0)

        # Add binary variables (y_v ∈ {0,1} for each candidate)
        for i in range(n):
            h.addBinary(obj=c[i])  # binary + set objective coefficient

        # Add covering constraints: -A @ y <= -1  (i.e., A @ y >= 1)
        for row_idx in range(m_rows):
            nonzero_cols = np.array(
                [i for i in range(n) if A_le[row_idx, i] != 0.0], dtype=np.int32
            )
            vals = np.array([A_le[row_idx, i] for i in nonzero_cols], dtype=np.float64)
            h.addRow(-np.inf, float(b_le[row_idx]), len(nonzero_cols), nonzero_cols, vals)

        h.run()
        model_status = h.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            greedy = _greedy_vertex_cover(candidates, valid_terms, weights=weights)
            return greedy if greedy else list(dict.fromkeys(candidates))

        sol = h.getSolution()
        y = np.array(sol.col_value[:n])
        selected = [candidates[i] for i in range(n) if y[i] > 0.5]
        selected_set = set(selected)

        # Verify coverage (fallback to max_cover if something went wrong)
        for term in valid_terms:
            if not any(v in selected_set for v in term):
                greedy = _greedy_vertex_cover(candidates, valid_terms, weights=weights)
                return greedy if greedy else list(dict.fromkeys(candidates))

        return selected

    except ImportError:
        greedy = _greedy_vertex_cover(candidates, valid_terms, weights=weights)
        return greedy if greedy else list(dict.fromkeys(candidates))


def _greedy_vertex_cover(
    candidates: list[int],
    terms: list[tuple[int, ...]],
    weights: dict[int, float] | None = None,
) -> list[int]:
    """Greedy vertex cover: repeatedly pick the variable covering the most uncovered terms.

    O(n * m) but good approximation for small instances.
    """
    selected = []
    term_set = [set(t) for t in terms]
    n_terms = len(term_set)
    uncovered = set(range(n_terms))

    while uncovered:
        # Find variable with most coverage
        best_var = None
        best_score = -1.0
        for v in candidates:
            if v in selected:
                continue
            count = sum(1 for idx in uncovered if v in term_set[idx])
            if count <= 0:
                continue

            if weights is None:
                score = float(count)
            else:
                dist = abs(float(weights.get(v, 0.0)))
                weight = max(dist, 1e-6)
                score = float(count) * weight

            if score > best_score:
                best_score = score
                best_var = v
        if best_var is None or best_score <= 0.0:
            break
        selected.append(best_var)
        newly_covered = {idx for idx in uncovered if best_var in term_set[idx]}
        uncovered -= newly_covered

    return selected


def pick_partition_vars(
    terms: NonlinearTerms,
    method: str = "auto",
    distance: dict[int, float] | None = None,
) -> list[int]:
    """Select variables to partition for the AMP relaxation.

    Parameters
    ----------
    terms : NonlinearTerms
        Output of classify_nonlinear_terms(model).
    method : str, default "auto"
        Variable selection strategy:
        - ``"max_cover"``: all variables appearing in any nonlinear term.
        - ``"min_vertex_cover"``: minimum set covering all terms (MILP-based).
        - ``"auto"``: max_cover if ≤15 candidates, else min_vertex_cover.
        - ``"adaptive_vertex_cover"``: Alpine-style weighted cover using incumbent
          distances when available, else falls back to ``"auto"``.

    Returns
    -------
    list[int]
        Flat variable indices selected for partitioning. Guaranteed to cover
        every bilinear and trilinear term (i.e., each term has ≥1 selected var).
    """
    if not terms.partition_candidates:
        return []

    if method == "max_cover":
        return _max_cover(terms)
    elif method == "min_vertex_cover":
        return _min_vertex_cover(terms)
    elif method == "auto":
        if len(terms.partition_candidates) <= 15:
            return _max_cover(terms)
        else:
            return _min_vertex_cover(terms)
    elif method == "adaptive_vertex_cover":
        if len(terms.partition_candidates) <= 15 or not distance:
            return _max_cover(terms)
        return _weighted_min_vertex_cover(terms, distance)
    else:
        raise ValueError(
            f"Unknown variable selection method: {method!r}. "
            "Choose from 'max_cover', 'min_vertex_cover', 'auto', "
            "'adaptive_vertex_cover'."
        )
