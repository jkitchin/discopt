#!/usr/bin/env python3
"""Generate one minimal text-.nl file per opcode so discopt's parser and
pounce.read_nl can be asked the same question: do you accept this operator?

One variable, no constraints, objective = <op>(v0) (or <op>(v0, 2)).
Exits non-zero if it writes nothing."""

import pathlib
import sys

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

HEAD = """g3 0 1 0
 1 0 1 0 0
 0 1
 0 0
 0 1
 0 0 0 1
 0 0
 0 1
 0 0
 0 0 0 0 0
"""
TAIL = """x1
0 0.5
b
0 0.1 2.0
k0
G0 1
0 0
"""

UNARY = {
    13: "floor",
    14: "ceil",
    15: "abs",
    16: "neg",
    37: "tanh",
    38: "tan",
    39: "sqrt",
    40: "sinh",
    41: "sin",
    42: "log10",
    43: "log",
    44: "exp",
    45: "cosh",
    46: "cos",
    47: "atanh",
    49: "atan",
    50: "asinh",
    51: "asin",
    52: "acosh",
    53: "acos",
    57: "round",
    58: "trunc",
    77: "pow2",
}
BINARY = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "div",
    5: "pow",
    48: "atan2",
    55: "intdiv",
    56: "precision",
    76: "pow_const",
    78: "cpow",
    59: "ATLEAST",
    60: "ATMOST",
    61: "PLTERM",
}
NARY = {11: "min", 12: "max", 54: "sum"}
TERNARY = {35: "if_then_else"}

n = 0


def write(op, name, body):
    global n
    (OUT / f"op{op:02d}_{name}.nl").write_text(HEAD + "O0 0\n" + body + TAIL)
    n += 1


for op, nm in UNARY.items():
    write(op, nm, f"o{op}\nv0\n")
for op, nm in BINARY.items():
    write(op, nm, f"o{op}\nv0\nn2\n")
for op, nm in NARY.items():
    write(op, nm, f"o{op}\n2\nv0\nn2\n")
for op, nm in TERNARY.items():
    write(op, nm, f"o{op}\nv0\nn1\nn2\n")

print(f"wrote {n} probe files to {OUT}")
if n == 0:
    sys.exit("PROBE DID NOT FIRE: no files written")
