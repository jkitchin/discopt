"""Render the neutral corpus AST (``corpus.py``) into Pyomo models for SUSPECT.

Runs only inside the isolated SUSPECT environment (py3.10 + pyomo 6.1.2). It
must not import discopt or modern numpy. See ``README.md`` for the env recipe.
"""

from __future__ import annotations

import math

import pyomo.environ as pe


def _eval(node, m):
    """Recursively turn a neutral AST node into a Pyomo expression."""
    if isinstance(node, (int, float)):
        return float(node)
    head = node[0]
    if head == "var":
        return getattr(m, node[1])
    args = [_eval(a, m) for a in node[1:]]
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
        return pe.exp(args[0])
    if head == "log":
        return pe.log(args[0])
    if head == "sqrt":
        return pe.sqrt(args[0])
    if head == "sin":
        return pe.sin(args[0])
    if head == "cos":
        return pe.cos(args[0])
    if head == "tan":
        return pe.tan(args[0])
    if head == "asin":
        return pe.asin(args[0])
    if head == "acos":
        return pe.acos(args[0])
    if head == "atan":
        return pe.atan(args[0])
    if head == "log2":
        # Pyomo has no native log2; SUSPECT sees a positively-scaled log,
        # whose curvature matches log (concave on x>0).
        return pe.log(args[0]) / math.log(2.0)
    if head == "abs":
        return abs(args[0])
    raise ValueError(f"unknown AST head: {head!r}")


_OP = {
    "<=": lambda body, rhs: body <= rhs,
    ">=": lambda body, rhs: body >= rhs,
    "==": lambda body, rhs: body == rhs,
}


def build_pyomo(instance: dict):
    """Build a Pyomo ``ConcreteModel`` from a neutral corpus instance."""
    m = pe.ConcreteModel(name=instance["name"])
    for vname, (lb, ub) in instance["vars"].items():
        setattr(m, vname, pe.Var(bounds=(lb, ub)))

    obj = instance.get("objective")
    if obj is not None:
        sense = pe.minimize if obj["sense"] == "min" else pe.maximize
        m.objective = pe.Objective(expr=_eval(obj["expr"], m), sense=sense)

    for con in instance["constraints"]:
        body = _eval(con["expr"], m)
        setattr(m, con["name"], pe.Constraint(expr=_OP[con["op"]](body, con["rhs"])))

    return m
