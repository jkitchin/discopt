"""#1154 panel, arm 1 — INERTNESS on the in-repo .nl corpus.

The DISCOPT_GDP_SUMOVER flag can only change behaviour on a model that actually
contains a ``SumOverExpression``. That node is built exclusively by the Python
modeling API (``dm.sum(...)``); the .nl reader never emits one. This probe
*measures* that claim over all 66 vendored MINLPLib instances rather than
asserting it, and it counts both the instances loaded and the DAG nodes walked
so a silent no-op cannot read as a pass (CLAUDE.md §6).

If any instance did contain the node, the corpus differential would have to be
run as a full solve panel; the count below is what decides that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import discopt
from discopt.modeling.core import (
    BinaryOp,
    Constraint,
    CustomCall,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    from_nl,
)

CORPUS = Path("python/tests/data/minlplib_nl")

print("sources:", discopt.__file__)
import discopt._relax.gdp_reformulate as _g  # noqa: E402

src = Path(_g.__file__).read_text()
print("marker gdp_sumover:", src.count("_sumover_terms"))


def walk(node, seen):
    """Yield every node reachable from *node* (identity-deduplicated DAG walk)."""
    stack = [node]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        if isinstance(e, IndexExpression):
            stack.append(e.base)
        elif isinstance(e, BinaryOp):
            stack.extend((e.left, e.right))
        elif isinstance(e, MatMulExpression):
            stack.extend((e.left, e.right))
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, (FunctionCall, CustomCall)):
            stack.extend(e.args)
        elif isinstance(e, SumExpression):
            stack.append(e.operand)
        elif isinstance(e, SumOverExpression):
            stack.extend(e.terms)


instances = 0
nodes = 0
sumover_nodes = 0
offenders: list[str] = []
failed: list[str] = []

for path in sorted(CORPUS.glob("*.nl")):
    try:
        model = from_nl(str(path))
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed (§7)
        failed.append(f"{path.name}: {type(exc).__name__}: {exc}")
        print(f"  LOAD FAILED {path.name}: {type(exc).__name__}: {exc}", flush=True)
        continue
    instances += 1
    seen: set[int] = set()
    local = 0
    roots = [c.body for c in model._constraints if isinstance(c, Constraint)]
    if model._objective is not None:
        roots.append(model._objective.expression)
    for root in roots:
        for e in walk(root, seen):
            nodes += 1
            if isinstance(e, SumOverExpression):
                local += 1
    sumover_nodes += local
    if local:
        offenders.append(f"{path.name}: {local}")
    print(f"  {path.name}: nodes={len(seen)} sumover={local}", flush=True)

print()
print(f"instances_loaded={instances}")
print(f"nodes_walked={nodes}")
print(f"sumover_nodes={sumover_nodes}")
print(f"load_failures={len(failed)}")
if offenders:
    print("OFFENDERS:", offenders)
print(f"executed_assertions={instances}")

if instances == 0 or nodes == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
