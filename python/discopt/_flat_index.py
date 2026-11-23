"""Single source of truth for resolving a scalar variable reference to a flat slot.

Every layer that reasons about *structure* rather than *values* — term
classification, McCormick relaxation construction, OBBT, cutting planes,
sparsity — needs to turn an expression like ``v[2]`` or ``w[1, 0]`` into the
integer position that entry occupies in the stacked ``x`` vector the solver
actually works with. Before issue #941 each of those layers open-coded that
arithmetic, and most of them got negative indices wrong: they took ``-1``
literally and returned ``base_offset - 1``, which is a perfectly valid slot
belonging to a *different variable*. Nothing downstream errors on a wrong-but-
in-range slot, so the relaxation was simply built for a bilinear pair that does
not exist in the model — an invalid relaxation that cut off the true optimum and
was then reported as ``optimal`` with ``gap_certified=True``.

The layout this module inverts is fixed by ``_relax/dag_compiler.py``, which
materializes a shaped variable as ``x_flat[off : off + size].reshape(shape)`` in
C (row-major) order and then applies ``a[index]``. For a full-rank all-integer
index that composition is exactly ``x_flat[off + ravel_multi_index(idx, shape)]``,
which is what :func:`resolve_scalar_slot` computes.

**Returning ``None`` is always sound; returning a wrong slot never is.** Callers
treat ``None`` as "not a scalar variable reference" and fall back to a more
general (more conservative) path. So every case this module cannot resolve
exactly — a slice, a partial index, a symbolic index, an out-of-range index — is
refused rather than guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from discopt.modeling.core import IndexExpression, Variable

if TYPE_CHECKING:
    from discopt.modeling.core import Expression, Model


def normalize_axis_index(i: Any, dim: int) -> int | None:
    """Return ``i`` normalized into ``[0, dim)``, or ``None`` if it is not a valid index.

    ``bool`` is rejected explicitly. It is a subclass of :class:`int` in Python,
    so ``isinstance(True, int)`` is ``True``, but numpy reads ``x[True]`` as a
    *mask* — it adds an axis — not as ``x[1]``. Accepting it would resolve a
    mask to a scalar slot.

    Out-of-range indices are refused rather than clamped. numpy raises where
    ``jnp`` silently clamps, so the two evaluation paths do not agree on what
    ``v[7]`` means for a shape-(4,) variable; inventing a slot here would bake in
    one of those answers. Refusal leaves the caller on its general path.
    """
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
        return None
    norm = int(i)
    if norm < 0:
        norm += dim
    if not 0 <= norm < dim:
        return None
    return norm


def flat_index_in_shape(index: Any, shape: tuple[int, ...]) -> int | None:
    """C-order flat position of ``index`` within ``shape``, or ``None``.

    ``index`` may be a bare integer (only for a 1-D ``shape``) or a tuple of
    integers whose length matches ``shape``. A shorter tuple is a *partial*
    index, which leaves an array rather than a scalar, and a longer one is
    invalid; both return ``None``.
    """
    idx = index if isinstance(index, tuple) else (index,)
    if len(idx) != len(shape):
        return None

    flat = 0
    for i, dim in zip(idx, shape):
        norm = normalize_axis_index(i, dim)
        if norm is None:
            return None
        flat = flat * dim + norm
    return flat


def resolve_scalar_slot(expr: Expression, model: Model) -> int | None:
    """Flat ``x`` slot named by ``expr``, or ``None`` if it does not name exactly one.

    Resolves three forms:

    * a scalar :class:`~discopt.modeling.core.Variable` (``size == 1``),
    * an :class:`~discopt.modeling.core.IndexExpression` whose base is a
      ``Variable`` and whose index selects a single element, and
    * a **chain** of all-integer ``IndexExpression`` nodes ending at a ``Variable``
      that together select a single element: ``x[k][s]``, which is what
      ``y = x[k]; ... y[s]`` builds in every time-stepped transcription (#2293).

    Everything else — a multi-element ``Variable`` with no index, a slice, a
    partial or out-of-range index, a non-``Variable`` base, a chain with a
    non-integer part anywhere — returns ``None``.

    **Why a chain is exact.** An all-integer index of length ``r`` on an
    ``n``-d array selects the leading ``r`` axes and leaves the trailing
    ``n - r``, so the next all-integer index continues on axis ``r``: the parts
    concatenate to one full index. Each element is still normalized and
    range-checked against *its own* axis. A slice, ``None``/newaxis, ``Ellipsis``
    or mask in any part breaks the concatenation, so it is refused, not guessed.
    The tape lowering applies the same rule (``_nl_expr_compiler.
    _static_scalar_slot``, #2283); before #2293 this resolver refused the chain,
    so every structure layer saw ``y[s]`` as an opaque subexpression -- no
    envelope in the relaxation, and a full-box interval-cache key per node,
    measured at 2.9 GB on a 14,336-row model.

    Deliberately arithmetic rather than probe-based: indexing an
    ``np.arange(size).reshape(shape)`` probe would match numpy semantics by
    construction, but allocating one per index node is O(size) per leaf and
    O(size·leaves) over a model build — the same quadratic root-setup cost issue
    #654 removed from the per-variable offset scan.
    """
    if isinstance(expr, Variable):
        # ``size == 1`` rather than ``not shape``: a shape-(1,) or (1, 1)
        # variable occupies exactly one slot, and the callers this replaces
        # already resolved it that way. Widening the refusal here would drop
        # structure those layers currently detect — sound, but a capability
        # regression, and #941 is about the *wrong* slot, not a missing one.
        if expr.size == 1:
            return model._flat_var_offset(expr)
        return None

    if not isinstance(expr, IndexExpression):
        return None
    base: Expression = expr
    parts: list[tuple] = []
    while isinstance(base, IndexExpression):
        index = base.index
        parts.append(index if isinstance(index, tuple) else (index,))
        base = base.base
    if not isinstance(base, Variable):
        return None
    shape = tuple(base.shape or ())
    if not shape:
        return None

    if len(parts) == 1:
        flat = flat_index_in_shape(expr.index, shape)
    else:
        # Every part of a chain must be all-integer for the parts to
        # concatenate; `flat_index_in_shape` then range-checks each element
        # against its own axis and refuses a partial or over-long result.
        joined = tuple(i for part in reversed(parts) for i in part)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in joined):
            return None
        flat = flat_index_in_shape(joined, shape)
    if flat is None:
        return None
    return model._flat_var_offset(base) + flat
