"""Render the neutral corpus AST (``corpus.py``) into discopt models.

Runs in the main discopt environment (the parity test imports this). It must
not import pyomo or SUSPECT. Mirror image of ``render_pyomo.py`` -- the two
renderers walk the *same* neutral AST, which is what makes the comparison a
head-to-head on identical mathematics.
"""

from __future__ import annotations

import discopt.modeling as dm
from discopt.modeling.core import FunctionCall, _wrap

# Inverse-trig atoms discopt's backend supports (FunctionCall -> jnp.arcsin etc.)
# but for which discopt.modeling ships no convenience wrapper. Built directly via
# FunctionCall so the head-to-head can still exercise them.
_FUNCTIONCALL_ATOMS = ("asin", "acos", "atan")


def _eval(node, vars_):
    """Recursively turn a neutral AST node into a discopt Expression."""
    if isinstance(node, (int, float)):
        return float(node)
    head = node[0]
    if head == "var":
        return vars_[node[1]]
    args = [_eval(a, vars_) for a in node[1:]]
    if head == "+":
        acc = args[0]
        for a in args[1:]:
            acc = acc + a
        return acc
    if head == "-":
        return args[0] - args[1]
    if head == "*":
        acc = args[0]
        for a in args[1:]:
            acc = acc * a
        return acc
    if head == "/":
        return args[0] / args[1]
    if head == "neg":
        return -args[0]
    if head == "pow":
        return args[0] ** node[2]
    if head == "exp":
        return dm.exp(args[0])
    if head == "log":
        return dm.log(args[0])
    if head == "sqrt":
        return dm.sqrt(args[0])
    if head == "sin":
        return dm.sin(args[0])
    if head == "cos":
        return dm.cos(args[0])
    if head == "tan":
        return dm.tan(args[0])
    if head == "log2":
        return dm.log2(args[0])
    if head == "abs":
        return abs(args[0])
    if head in _FUNCTIONCALL_ATOMS:
        return FunctionCall(head, _wrap(args[0]))
    raise ValueError(f"unknown AST head: {head!r}")


def _build_model_and_vars(instance: dict):
    """Build the discopt ``Model`` and return it with its variable map and
    constraint-name order. Shared by :func:`build_discopt` and
    :func:`build_discopt_items`."""
    m = dm.Model(instance["name"])
    vars_ = {name: m.continuous(name, lb=lb, ub=ub) for name, (lb, ub) in instance["vars"].items()}

    obj = instance.get("objective")
    if obj is not None:
        expr = _eval(obj["expr"], vars_)
        if obj["sense"] == "min":
            m.minimize(expr)
        else:
            m.maximize(expr)

    constraint_names: list[str] = []
    for con in instance["constraints"]:
        body = _eval(con["expr"], vars_)
        rhs = con["rhs"]
        if con["op"] == "<=":
            m.subject_to(body <= rhs, name=con["name"])
        elif con["op"] == ">=":
            m.subject_to(body >= rhs, name=con["name"])
        else:
            m.subject_to(body == rhs, name=con["name"])
        constraint_names.append(con["name"])

    return m, vars_, constraint_names


def build_discopt(instance: dict):
    """Build a discopt ``Model`` from a neutral corpus instance.

    Returns ``(model, constraint_names)`` where ``constraint_names`` lists the
    corpus constraint names in the same order they were added, so the parity
    test can align ``model._constraints`` with SUSPECT's named verdicts.
    """
    m, _vars, constraint_names = _build_model_and_vars(instance)
    return m, constraint_names


def build_discopt_items(instance: dict):
    """Build the model and the *raw* (pre-canonicalisation) body of every
    comparable item.

    Returns ``(model, items)`` where each item is
    ``{"key": str, "body": Expression}`` and ``body`` is the corpus AST
    rendered directly into a discopt :class:`Expression`, untouched by the
    model's ``<=``-canonicalisation of constraints. This is exactly the body
    SUSPECT reports monotonicity / interval bounds on (its ``obj.expr`` /
    ``con.body``), so the monotonicity and FBBT cross-checks compare identical
    raw mathematics on both sides with **zero** normalisation. The bodies share
    the model's :class:`Variable` objects, so ``classify_monotonicity`` and
    ``evaluate_interval`` read their declared bounds.
    """
    m, vars_, _cnames = _build_model_and_vars(instance)
    items: list[dict] = []
    obj = instance.get("objective")
    if obj is not None:
        items.append({"key": "objective", "body": _eval(obj["expr"], vars_)})
    for con in instance["constraints"]:
        items.append({"key": con["name"], "body": _eval(con["expr"], vars_)})
    return m, items
