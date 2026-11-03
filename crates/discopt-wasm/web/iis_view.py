"""Turn an :class:`~discopt.infeasibility.IISResult` into data the page can draw.

The browser page (``index.html``) shows an IIS as a table and as a
constraint--variable graph. Both are drawn in JavaScript from the plain,
JSON-serializable dict built here, so all knowledge of discopt's expression
tree stays on the Python side.

How a script's IIS reaches the page: after a script finishes, ``worker.js``
calls :func:`collect` on the script's namespace. Every ``IISResult`` the script
left in a top-level variable is drawn. The script itself stays plain discopt,
so it runs unchanged outside the browser, and ``python/tests/test_wasm_examples.py``
runs it that way.

This file is served next to the page and written into Pyodide's file system by
``worker.js``. It is not part of the discopt wheel, so it uses only what the
wheel the page ships already has. ``python/tests/test_wasm_iis_view.py``
imports it from this path and checks the dict it builds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from discopt.infeasibility import IISResult
from discopt.modeling.core import IndexExpression, Model, Variable
from discopt.serialize import _children

#: Bumped when the dict's shape changes, so the page can refuse a dict it does
#: not understand instead of drawing it wrongly.
SCHEMA = 1

#: The graph shows at most this many constraints that are *not* in the IIS but
#: share a variable with it. The rest are counted, not drawn.
MAX_CONTEXT = 12


# --- variable references ----------------------------------------------------


def _element_index(var: Variable, index: Any) -> tuple[int, ...] | None:
    """The element ``var[index]`` names, as a full integer tuple, or ``None``.

    ``None`` means the index selects more than one element (a slice, an array,
    fewer indices than dimensions), so the reference is to the whole variable.
    """
    shape = tuple(var.shape)
    idx = index if isinstance(index, tuple) else (index,)
    if len(idx) != len(shape):
        return None
    out = []
    for i, n in zip(idx, shape):
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)):
            return None
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            return None
        out.append(i)
    return tuple(out)


def element_id(var: Variable, index: tuple[int, ...] = ()) -> str:
    """Display name of one element: ``x``, ``a[1]`` or ``x[0,2]``."""
    if not index:
        return var.name
    return f"{var.name}[{','.join(str(i) for i in index)}]"


def variable_refs(expr: Any) -> list[tuple[Variable, tuple[int, ...] | None]]:
    """Every variable reference in ``expr``, in first-seen order, without repeats.

    A reference is ``(variable, index)``: ``index`` is the element tuple for
    ``a[1]`` (``()`` for a scalar variable) and ``None`` when a vector variable
    is used whole, as in ``dm.sum(a)``.
    """
    seen: dict[tuple[int, tuple[int, ...] | None], tuple[Variable, tuple[int, ...] | None]] = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
            idx = _element_index(node.base, node.index)
            seen.setdefault((id(node.base), idx), (node.base, idx))
            continue
        if isinstance(node, Variable):
            # A one-element variable used bare *is* its element: (0,) for shape
            # (1,), so it meets the id its bound gets, not a second node.
            idx = (0,) * len(node.shape) if node.size == 1 else None
            seen.setdefault((id(node), idx), (node, idx))
            continue
        # Reversed so the walk visits children left to right.
        stack.extend(reversed(_children(node)))
    return list(seen.values())


# --- the payload ------------------------------------------------------------


def _find_model(iis: IISResult, namespace: dict[str, Any]) -> Model | None:
    """The model in ``namespace`` this IIS was computed from.

    Matched by identity: ``compute_iis`` returns the model's own constraint and
    variable objects, so a model holding them is the one it came from.
    """
    con_ids = {id(c) for c in iis.constraints}
    var_ids = {id(v) for v, _ in iis.variable_bounds}
    for value in namespace.values():
        if not isinstance(value, Model):
            continue
        cons = {id(c) for c in value._constraints}
        vars_ = {id(v) for v in value._variables}
        if con_ids <= cons and var_ids <= vars_:
            return value
    return None


def _bound_value(var: Variable, side: str, index: tuple[int, ...]) -> float:
    bound = np.asarray(var.lb if side == "lower" else var.ub, dtype=np.float64)
    return float(np.broadcast_to(bound, var.shape)[index] if index else bound.reshape(-1)[0])


def _constraint_text(con: Any) -> str:
    """discopt's own normalized form of a constraint, ``body sense rhs``."""
    return f"{con.body} {con.sense} {float(con.rhs):g}"


def iis_payload(
    iis: IISResult,
    model: Model | None = None,
    *,
    label: str = "",
    max_context: int = MAX_CONTEXT,
) -> dict[str, Any]:
    """The JSON-serializable description of ``iis`` the page draws.

    Members are the IIS's constraints, then its bounds. Each lists the
    variable elements it touches (``vars``, ids into ``variables``).

    ``variables`` holds every element some member touches. A constraint that
    uses a vector variable whole is linked to that variable's elements already
    in the graph; if there are none, the whole variable is one node.

    With ``model`` given, ``context`` lists up to ``max_context`` constraints
    that are not in the IIS but share a variable with it, linked only to the
    variables already in the graph, and ``context_total`` counts all of them.
    They are not part of the conflict; the page draws them to show where the
    conflict sits in the model.
    """
    var_ids: dict[str, dict[str, Any]] = {}  # element id -> node, in insertion order

    def add_var(var: Variable, idx: tuple[int, ...] | None) -> str:
        key = element_id(var, idx or ())
        var_ids.setdefault(
            key,
            {
                "id": key,
                "base": var.name,
                "index": list(idx) if idx else [],
                "whole": idx is None and var.size > 1,
            },
        )
        return key

    # Bounds first: they name exact elements, which whole-variable uses below
    # then link to.
    bound_members = []
    elems = iis.bound_elements or [(v, s, ()) for v, s in iis.variable_bounds]
    for var, side, idx in elems:
        idx = tuple(int(i) for i in idx)
        if not idx and var.size == 1:
            # compute_iis only splits bounds of vectors with more than one
            # element, so a one-element variable's bound arrives as ``()``.
            idx = (0,) * len(var.shape)
        key = add_var(var, idx if idx or var.size == 1 else None)
        value = _bound_value(var, side, idx)
        op = ">=" if side == "lower" else "<="
        bound_members.append(
            {
                "kind": "bound",
                "name": f"{key} {op} {value:g}",
                "text": f"{key} {op} {value:g}",
                "side": side,
                "value": value,
                "variable": var.name,
                "vars": [key],
            }
        )

    def link(refs: list[tuple[Variable, tuple[int, ...] | None]], create: bool) -> list[str]:
        out: list[str] = []
        for var, idx in refs:
            if idx is None and var.size > 1:
                present = [k for k, n in var_ids.items() if n["base"] == var.name]
                if present:
                    out.extend(k for k in present if k not in out)
                    continue
            key = element_id(var, idx or ())
            if key not in var_ids:
                if not create:
                    continue
                add_var(var, idx)
            if key not in out:
                out.append(key)
        return out

    constraint_members = []
    for i, con in enumerate(iis.constraints):
        constraint_members.append(
            {
                "kind": "constraint",
                "name": con.name or f"constraint {i}",
                "named": bool(con.name),
                "text": _constraint_text(con),
                "sense": str(con.sense),
                "vars": link(variable_refs(con.body), create=True),
            }
        )

    members = constraint_members + bound_members
    for i, member in enumerate(members):
        member["id"] = f"m{i}"

    context: list[dict[str, Any]] = []
    context_total = 0
    model_info = None
    if model is not None:
        in_iis = {id(c) for c in iis.constraints}
        for con in model._constraints:
            if id(con) in in_iis or not hasattr(con, "body"):
                continue
            shared = link(variable_refs(con.body), create=False)
            if not shared:
                continue
            context_total += 1
            if len(context) < max_context:
                context.append(
                    {
                        "id": f"k{len(context)}",
                        "kind": "context",
                        "name": con.name or f"constraint {len(context)}",
                        "named": bool(con.name),
                        "text": _constraint_text(con),
                        "vars": shared,
                    }
                )
        model_info = {
            "name": getattr(model, "name", "") or "",
            "n_constraints": len(model._constraints),
            "n_variables": int(sum(v.size for v in model._variables)),
        }

    return {
        "schema": SCHEMA,
        "label": label,
        "proven_irreducible": bool(iis.proven_irreducible),
        "n_solves": int(iis.n_solves),
        "model": model_info,
        "members": members,
        # Sorted, so the graph lists a[0], a[1], a[2] in order however the
        # members happened to mention them.
        "variables": sorted(var_ids.values(), key=lambda n: (n["base"], n["index"])),
        "context": context,
        "context_total": context_total,
    }


def collect(namespace: dict[str, Any]) -> list[dict[str, Any]]:
    """A payload for every ``IISResult`` bound to a top-level name in ``namespace``.

    Names starting with ``_`` are skipped. One result bound to two names is
    drawn once, under the first.
    """
    out = []
    seen: set[int] = set()
    for name, value in list(namespace.items()):
        if name.startswith("_") or not isinstance(value, IISResult) or id(value) in seen:
            continue
        seen.add(id(value))
        out.append(iis_payload(value, _find_model(value, namespace), label=name))
    return out


def collect_json(namespace: Any) -> str:
    """:func:`collect` as a JSON string, the form ``worker.js`` posts to the page.

    ``namespace`` may be a Pyodide proxy of a dict; ``dict()`` copies it.
    """
    return json.dumps(collect(dict(namespace)))
