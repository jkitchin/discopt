"""Minimal, validated .nl prefix-expression parser -> a small typed DAG.

Only the opcodes appearing in the LR-0 targets are handled; any unknown opcode
raises loudly (no silent fallback). Produces (node) trees whose leaves are
('var', i), ('const', c) and internal nodes ('op', name, children...). Also
returns the J-linear terms per constraint and the objective linear terms.

This is a PROBE parser. It is cross-checked against the Rust .nl oracle
(evaluate_objective / evaluate_constraint) to 1e-8 before any relaxation is
trusted (see validate()).
"""

from __future__ import annotations

import re

import numpy as np

UNARY = {"o16": "neg", "o39": "sqrt", "o43": "ln", "o41": "exp", "o42": "log10"}
BINARY = {"o0": "+", "o1": "-", "o2": "*", "o3": "/", "o5": "^"}
NARY = {"o54": "sum", "o53": "prod", "o2 ": None}


class Node:
    __slots__ = ("kind", "a", "b", "children")

    def __init__(self, kind, a=None, b=None, children=None):
        self.kind = kind      # 'var','const','neg','sqrt','ln','exp','+','-','*','/','^','sum'
        self.a = a
        self.b = b
        self.children = children


def parse_tokens(toks):
    it = iter(toks)

    def rec():
        t = next(it)
        if t in UNARY:
            return Node(UNARY[t], a=rec())
        if t in BINARY:
            a = rec(); b = rec()
            return Node(BINARY[t], a=a, b=b)
        if t == "o54" or t == "o53":
            n = int(next(it))
            ch = [rec() for _ in range(n)]
            return Node("sum" if t == "o54" else "prod", children=ch)
        if t.startswith("n"):
            return Node("const", a=float(t[1:]))
        if t.startswith("v"):
            return Node("var", a=int(t[1:]))
        raise ValueError(f"unknown opcode/token {t!r}")

    return rec()


def evaluate(node, x):
    k = node.kind
    if k == "var":
        return x[node.a]
    if k == "const":
        return node.a
    if k == "neg":
        return -evaluate(node.a, x)
    if k == "sqrt":
        return np.sqrt(evaluate(node.a, x))
    if k == "ln":
        return np.log(evaluate(node.a, x))
    if k == "exp":
        return np.exp(evaluate(node.a, x))
    if k == "+":
        return evaluate(node.a, x) + evaluate(node.b, x)
    if k == "-":
        return evaluate(node.a, x) - evaluate(node.b, x)
    if k == "*":
        return evaluate(node.a, x) * evaluate(node.b, x)
    if k == "/":
        return evaluate(node.a, x) / evaluate(node.b, x)
    if k == "^":
        return evaluate(node.a, x) ** evaluate(node.b, x)
    if k == "sum":
        return sum(evaluate(c, x) for c in node.children)
    if k == "prod":
        r = 1.0
        for c in node.children:
            r = r * evaluate(c, x)
        return r
    raise ValueError(k)


def load_nl_expressions(path):
    """Return dict with 'nvars', 'obj' node, 'obj_lin' {i:coeff}, 'cons' list of
    {'body':node, 'lin':{i:coeff}, 'sense':int, 'rhs':float}. Parses O/C/J/r/b
    segments. sense codes (r-section): 1 <=, 2 >=, 3 free, 4 ==."""
    lines = [ln.rstrip("\n") for ln in open(path)]
    n_vars = int(lines[1].split()[0])
    n_cons = int(lines[1].split()[1])

    # a line is a NEW segment header iff it starts with an uppercase segment
    # letter or one of the lowercase segment letters (b, r, k) followed by a
    # digit/space. Expression tokens are o.. / v.. / n.. — never segment heads.
    seg_re = re.compile(r"^[COJGVSLbrk]\d|^[CO]\d+\s|^[brk]$|^S\d")

    def is_header(s):
        return bool(seg_re.match(s.strip()))
    # locate segments
    cons_body = {}
    obj_body = None
    cons_lin = {j: {} for j in range(n_cons)}
    obj_lin = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        mC = re.match(r"^C(\d+)$", ln)
        mO = re.match(r"^O(\d+)\s", ln)
        mJ = re.match(r"^J(\d+)\s+(\d+)$", ln)
        mG = re.match(r"^G(\d+)\s+(\d+)$", ln)
        if mC:
            j = int(mC.group(1))
            blk = []
            i += 1
            while i < len(lines) and not is_header(lines[i]):
                blk.append(lines[i].strip()); i += 1
            # blk are the expr tokens (each line one token)
            toks = [t for t in blk if t != ""]
            cons_body[j] = parse_tokens(toks)
            continue
        if mO:
            j = int(mO.group(1))
            blk = []
            i += 1
            while i < len(lines) and not is_header(lines[i]):
                blk.append(lines[i].strip()); i += 1
            toks = [t for t in blk if t != ""]
            obj_body = parse_tokens(toks) if toks else None
            continue
        if mJ:
            j = int(mJ.group(1)); k = int(mJ.group(2))
            for r in range(k):
                i += 1
                vi, co = lines[i].split()
                cons_lin[j][int(vi)] = float(co)
            i += 1
            continue
        if mG:
            j = int(mG.group(1)); k = int(mG.group(2))
            for r in range(k):
                i += 1
                vi, co = lines[i].split()
                obj_lin[int(vi)] = float(co)
            i += 1
            continue
        i += 1
    return {
        "nvars": n_vars, "ncons": n_cons,
        "obj": obj_body, "obj_lin": obj_lin,
        "cons": cons_body, "cons_lin": cons_lin,
    }


def cons_value(P, j, x):
    body = P["cons"].get(j)
    v = evaluate(body, x) if body is not None else 0.0
    for vi, co in P["cons_lin"][j].items():
        v = v + co * x[vi]
    return v


def obj_value(P, x):
    v = evaluate(P["obj"], x) if P["obj"] is not None else 0.0
    for vi, co in P["obj_lin"].items():
        v = v + co * x[vi]
    return v
