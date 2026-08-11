"""Provably-sound implied-integer detection.

A variable declared continuous may be *forced* to integer values by the model.
Marking such a variable integer is value-preserving (it cannot cut off any
feasible — hence any optimal — point) and lets the integer-product reformulation
(``integer_product_reform``) tighten bilinear terms that involve it. Marking a
*non*-implied variable integer would cut off feasible points and is the cardinal
correctness violation, so this detector uses only a **rigorously sound**
sufficient condition and is conservative everywhere else.

**Sound condition (integer-defining equality).** A variable ``x`` is integer at
every feasible point if there is a *linear* equality constraint

    Σ_j a_j x_j + c = 0      (sense "==")

with all ``a_j`` and ``c`` integer, ``x`` appearing with coefficient ``±1``, and
**every other** variable with a nonzero coefficient already known integer
(declared integer/binary, or itself proven implied-integer in an earlier round).
Then ``x = ∓(Σ_{j≠x} a_j x_j + c)`` is an integer combination of integers ⇒ ``x``
is integer.

This is exactly the structure the ``ex126x`` trim-loss models carry: e.g.
``x5 - x35 - 2·x36 - 4·x37 = 0`` with ``x35,x36,x37`` binary ⇒ ``x5`` integer.

Detection iterates to a fixpoint so chains (``x`` integer because ``y`` was just
proven integer) are caught. Range links like ``b ≤ x ≤ b+4`` (inequalities) are
**never** sufficient and are correctly ignored.
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import Constraint, Model, VarType

from .gdp_reformulate import _is_linear
from .problem_classifier import _extract_linear_coefficients_sparse, _NotLinearError

_INT_TOL = 1e-9


def _is_int_value(x: float) -> bool:
    return abs(x - round(x)) <= _INT_TOL


def detect_implied_integers(model: Model) -> set[tuple[int, int]]:
    """Return ``{(variable._index, flat_element)}`` for every continuous variable
    the model *provably* forces to integer values (see module docstring).

    Conservative: under-detection is safe (a missed tightening); the returned set
    is sound — constraining any of these variables integer leaves the feasible
    region's relevant projection, and hence the optimum, unchanged.

    **Row representation (#863).** Rows are kept as their nonzeros only. The dense
    predecessor built an ``np.zeros(n)`` per equality *body* and RETAINED one per
    integer-data row, then re-derived each row's support with
    ``np.nonzero(np.abs(a) > tol)`` — an O(n) scan per row per fixpoint round.
    On ``watercontamination0202`` (106,711 variables / 107,209 constraints) this
    function cost **71.1 s and +31.2 GiB RSS**. The dense full-width row was never
    needed, only its support and the coefficients on it.

    The marking is **identical**, not merely similar, and that matters because this
    function marks variables INTEGER: a wrongly-marked variable cuts off feasible
    points, the cardinal correctness violation. Identity holds term by term:

    * ``_extract_linear_coefficients_sparse`` is the accumulator the dense wrapper
      is already built on, so the coefficient values are bit-identical.
    * The integer-coefficient test ran over all ``n`` slots; entries absent from the
      row are exactly ``0.0``, which passes ``|a - round(a)| <= tol``, so testing
      only the stored entries accepts and rejects exactly the same rows. It uses
      ``np.round`` on the stored values for the same half-to-even tie behaviour.
    * The support is sorted ASCENDING, which is the order ``np.nonzero`` produced,
      and rows stay in ``model._constraints`` order. The fixpoint loop below is
      order-sensitive in principle, so preserving both orders — rather than merely
      preserving the set of rows — is what makes the marked set identical.
    * The support is now computed once per row instead of once per row per round;
      ``a`` never changes between rounds, so the recomputation was pure waste.

    ``test_863_sparse_implied_integer.py`` checks that identity against an
    independent dense reimplementation of the predecessor over the in-repo ``.nl``
    corpus, not just that it got faster.
    """
    n = sum(v.size for v in model._variables)
    flat = [(v, e) for v in model._variables for e in range(v.size)]
    # Known-integer mask: declared integer/binary to start; grows as we prove more.
    is_int = np.array(
        [flat[i][0].var_type in (VarType.INTEGER, VarType.BINARY) for i in range(n)],
        dtype=bool,
    )

    # Pre-extract integer-data linear equality rows once, as (support, coefficients)
    # with the support ascending.
    eq_rows: list[tuple[list[int], dict[int, float]]] = []
    for c in model._constraints:
        if not isinstance(c, Constraint) or c.sense != "==":
            continue
        if not _is_linear(c.body):
            continue
        try:
            terms, const = _extract_linear_coefficients_sparse(c.body, model, n)
        except _NotLinearError:
            continue
        if terms:
            vals = np.fromiter(terms.values(), dtype=np.float64, count=len(terms))
            if not np.all(np.abs(vals - np.round(vals)) <= _INT_TOL):
                continue
        if not _is_int_value(float(const)):
            continue
        nz = sorted(i for i, v in terms.items() if abs(v) > _INT_TOL)
        eq_rows.append((nz, terms))

    marked: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        for nz, terms in eq_rows:
            for idx in nz:
                if is_int[idx]:
                    continue
                if abs(abs(terms[idx]) - 1.0) > _INT_TOL:
                    continue  # coefficient must be ±1 for the integer-quotient proof
                if all(is_int[j] for j in nz if j != idx):
                    var, elem = flat[idx]
                    marked.add((var._index, elem))
                    is_int[idx] = True
                    changed = True
    return marked


def mark_implied_integers(model: Model) -> int:
    """Mark every detected implied-integer variable's ``var_type`` as INTEGER,
    in place. Returns the number of (scalar-element) markings applied.

    A variable is promoted to INTEGER only when **all** of its scalar elements are
    implied-integer (the per-element granularity of detection is preserved for the
    common scalar case; array variables are promoted only when fully covered, to
    avoid changing the type of a partially-continuous block)."""
    detected = detect_implied_integers(model)
    if not detected:
        return 0
    by_var: dict[int, set[int]] = {}
    for vidx, elem in detected:
        by_var.setdefault(vidx, set()).add(elem)
    count = 0
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            continue
        elems = by_var.get(v._index)
        if elems is not None and len(elems) == v.size:
            v.var_type = VarType.INTEGER
            count += v.size
    return count
