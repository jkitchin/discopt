"""Disjunct-local bounds for the big-M lowering (#1775 G5.1, spec §7; #1917).

Big-M sizes the relaxation of a row of disjunct ``j`` by the interval bound of its
body over the *declared* box. But the row is relaxed only while ``j`` is inactive,
and under ``SELECT_ONE``/``OR`` semantics some sibling ``k != j`` is then active.
So the body need only be bounded over the union of the siblings' boxes, where
each sibling's box is FBBT over the model's global rows together with that
sibling's own rows. That box is often far smaller than the declared one.

:func:`disjunct_local_big_m_caps` returns, per disjunct row (keyed by
``id(row)``), that bound with ``_compute_big_m``'s own safety factor. The lowering
uses ``min(M_today, cap)``, gated by ``DISCOPT_GDP_DISJUNCT_FBBT``, default OFF.

Soundness, rule by rule:

* **The sibling box is an outer box.** It is FBBT over a *subset* of the rows
  that hold whenever the sibling is active: the global plain rows and the
  sibling's own plain rows. Other disjunctions, logic and fast-API rows are
  dropped, which only enlarges the box. ``tightening.fbbt_box`` intersects with
  the declared bounds, so it never loosens them.
* **A sibling FBBT calls infeasible is not dropped from the union.** That verdict
  rests on a tolerance, so the sibling keeps the global-rows box. A cap is never
  derived from an unproven empty set.
* **Only root disjunctions with one-way ``EXACTLY_ONE`` / ``AT_LEAST_ONE``
  semantics** take a cap. That means a ``SELECT_ONE`` either_or or block group,
  or an ``OR``. For a reified disjunction an inactive disjunct is not merely
  relaxed. A nested disjunction's rows are relaxed relative to its parent too,
  and both are left alone.
* **Alignment is checked, not assumed.** The flat FBBT bounds are mapped to
  variables only when the repr's variable blocks are the model's variables, by
  name and size, in order. Otherwise no cap is produced at all.
* **The body bound is outward-rounded** (:func:`evaluate_interval`) and gets
  ``_compute_big_m``'s ``max(|M|, 1e-8) * 1.01``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import Constraint, Model


def _with_rows(model: Model, rows: list[Constraint]) -> Model:
    out = Model(f"{model.name}_disjunct_fbbt")
    out._variables = list(model._variables)
    out._parameters = list(model._parameters)
    out._rebuild_name_index()
    out._constraints = list(rows)
    out._objective = model._objective
    return out


def _fbbt(model: Model) -> Optional[tuple[np.ndarray, np.ndarray, bool]]:
    """``(lb, ub, infeasible)`` per scalar in ``model._variables`` order, or ``None``
    when the repr's blocks do not line up with the model's variables."""
    from discopt._rust import model_to_repr
    from discopt.tightening import fbbt_box

    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    names = list(repr_.var_names())
    sizes = [int(np.prod(s)) if len(s) else 1 for s in repr_.var_shapes()]
    if names != [v.name for v in model._variables] or sizes != [
        int(v.size) for v in model._variables
    ]:
        return None
    r = fbbt_box(model, repr_=repr_)
    return np.asarray(r.lb, dtype=np.float64), np.asarray(r.ub, dtype=np.float64), r.infeasible


def _box(model: Model, lb: np.ndarray, ub: np.ndarray) -> dict:
    from discopt._relax.convexity.interval import Interval

    box: dict = {}
    off = 0
    for v in model._variables:
        n = int(v.size)
        box[v] = Interval(lb[off : off + n].reshape(v.shape), ub[off : off + n].reshape(v.shape))
        off += n
    return box


def _cap(row: Constraint, box: dict) -> Optional[float]:
    """The big-M of ``row`` over ``box`` (``_compute_big_m``'s formula), or ``None``."""
    from discopt._relax.convexity.interval_eval import evaluate_interval

    try:
        iv = evaluate_interval(row.body - row.rhs, None, box)
    except Exception:  # noqa: BLE001 - no enclosure means no cap; today's M stands
        return None
    lo, hi = float(np.min(iv.lo)), float(np.max(iv.hi))
    if row.sense == "<=":
        m = hi
    elif row.sense == ">=":
        m = -lo
    elif row.sense == "==":
        m = max(hi, -lo)
    else:
        return None
    if not np.isfinite(m):
        return None
    return max(abs(m), 1e-8) * 1.01


def disjunct_local_big_m_caps(model: Model) -> dict[int, float]:
    """``id(row) -> cap`` for the rows of root one-way disjunctions; see the module
    docstring. Empty when nothing qualifies or the FBBT cannot be aligned."""
    from discopt.gdp.tree import gdp_tree
    from discopt.modeling.core import SelectorActivation, SelectorCardinality

    tree = gdp_tree(model)
    global_rows = [c for c in model._constraints if type(c) is Constraint]
    base = _fbbt(_with_rows(model, global_rows))
    if base is None:
        return {}
    g_lb, g_ub, _ = base
    caps: dict[int, float] = {}
    for node in tree.disjunctions:
        activation, cardinality = node.semantics.value
        if activation is not SelectorActivation.ONE_WAY or cardinality not in (
            SelectorCardinality.EXACTLY_ONE,
            SelectorCardinality.AT_LEAST_ONE,
        ):
            continue
        if len(node.disjuncts) < 2:
            continue
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for d in node.disjuncts:
            own = [c for c in d.rows if type(c) is Constraint]
            r = _fbbt(_with_rows(model, global_rows + own))
            if r is None:
                return {}
            lb, ub, infeasible = r
            # A tolerance-based "infeasible" is not a proof: keep the global box.
            if infeasible:
                boxes.append((g_lb, g_ub))
            else:
                boxes.append((np.maximum(lb, g_lb), np.minimum(ub, g_ub)))
        for j, d in enumerate(node.disjuncts):
            sib = [b for k, b in enumerate(boxes) if k != j]
            u_lb = np.min([b[0] for b in sib], axis=0)
            u_ub = np.max([b[1] for b in sib], axis=0)
            if np.any(u_lb > u_ub):
                continue
            box = _box(model, u_lb, u_ub)
            for row in d.rows:
                if type(row) is not Constraint:
                    continue
                cap = _cap(row, box)
                if cap is not None:
                    caps[id(row)] = cap
    return caps


def _enabled() -> bool:
    from discopt.solver_tuning import current as _tuning

    return bool(_tuning().gdp_disjunct_fbbt)


__all__ = ["disjunct_local_big_m_caps"]
