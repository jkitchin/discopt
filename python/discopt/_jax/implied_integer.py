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
from .problem_classifier import _extract_linear_coefficients, _NotLinearError

_INT_TOL = 1e-9


def _is_int_value(x: float) -> bool:
    return abs(x - round(x)) <= _INT_TOL


def detect_implied_integers(model: Model) -> set[tuple[int, int]]:
    """Return ``{(variable._index, flat_element)}`` for every continuous variable
    the model *provably* forces to integer values (see module docstring).

    Conservative: under-detection is safe (a missed tightening); the returned set
    is sound — constraining any of these variables integer leaves the feasible
    region's relevant projection, and hence the optimum, unchanged.
    """
    n = sum(v.size for v in model._variables)
    flat = [(v, e) for v in model._variables for e in range(v.size)]
    # Known-integer mask: declared integer/binary to start; grows as we prove more.
    is_int = np.array(
        [flat[i][0].var_type in (VarType.INTEGER, VarType.BINARY) for i in range(n)],
        dtype=bool,
    )

    # Pre-extract integer-data linear equality rows once.
    eq_rows: list[tuple[np.ndarray, float]] = []
    for c in model._constraints:
        if not isinstance(c, Constraint) or c.sense != "==":
            continue
        if not _is_linear(c.body):
            continue
        try:
            a, const = _extract_linear_coefficients(c.body, model, n)
        except _NotLinearError:
            continue
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.abs(a - np.round(a)) <= _INT_TOL) or not _is_int_value(float(const)):
            continue
        eq_rows.append((a, float(const)))

    marked: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        for a, _const in eq_rows:
            nz = np.nonzero(np.abs(a) > _INT_TOL)[0]
            for idx in nz:
                if is_int[idx]:
                    continue
                if abs(abs(a[idx]) - 1.0) > _INT_TOL:
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
