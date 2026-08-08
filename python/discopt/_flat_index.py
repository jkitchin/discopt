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

The layout this module inverts is fixed by ``_jax/dag_compiler.py``, which
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

    Resolves two forms:

    * a scalar :class:`~discopt.modeling.core.Variable` (``size == 1``), and
    * an :class:`~discopt.modeling.core.IndexExpression` whose base is a
      ``Variable`` and whose index selects a single element.

    Everything else — a multi-element ``Variable`` with no index, a slice, a
    partial or out-of-range index, a non-``Variable`` base — returns ``None``.

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
    base = expr.base
    if not isinstance(base, Variable):
        return None
    shape = tuple(base.shape or ())
    if not shape:
        return None

    flat = flat_index_in_shape(expr.index, shape)
    if flat is None:
        return None
    return model._flat_var_offset(base) + flat
