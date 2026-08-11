#!/usr/bin/env python
"""Issue #865 entry experiment — how often does the convex kernel abstain on
`bilinear product`, and are the blocking products binary x (affine)?

For every .nl in the in-repo corpora: big-M-reformulate, walk each constraint
body for var*var products, and classify each one. Report the abstain reason from
`build_convex_spec` alongside the product census.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISCOPT_CONVEX_KERNEL", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
from discopt.modeling.core import (  # noqa: E402
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    UnaryOp,
    Variable,
    VarType,
)
from discopt.solvers import _convex_kernel as ck  # noqa: E402


def col_kind(m, offsets, col):
    """'B' binary, 'I' integer, 'C' continuous for a flat column."""
    k = 0
    for v in m._variables:
        for e in range(v.size):
            if k == col:
                if v.var_type == VarType.BINARY:
                    return "B"
                if v.var_type == VarType.INTEGER:
                    lo = float(np.asarray(v.lb).flat[e])
                    hi = float(np.asarray(v.ub).flat[e])
                    return "B" if lo >= -1e-9 and hi <= 1 + 1e-9 else "I"
                return "C"
            k += 1
    return "?"


def affine_cols(node, offsets, out):
    """Collect the columns of a (hopefully) affine subexpression; None if not affine."""
    if isinstance(node, Constant):
        return True
    if isinstance(node, (Variable, IndexExpression)):
        try:
            out.add(ck._col_of(node, offsets))
        except ck.NotConvexKernel:
            return False
        return True
    if isinstance(node, UnaryOp):
        return node.op == "neg" and affine_cols(node.operand, offsets, out)
    if isinstance(node, BinaryOp):
        if node.op in ("+", "-"):
            return affine_cols(node.left, offsets, out) and affine_cols(node.right, offsets, out)
        if node.op == "*":
            lc, rc = ck._as_const(node.left), ck._as_const(node.right)
            if lc is not None:
                return affine_cols(node.right, offsets, out)
            if rc is not None:
                return affine_cols(node.left, offsets, out)
            return False
        if node.op == "/":
            return ck._as_const(node.right) is not None and affine_cols(node.left, offsets, out)
        return False
    return False


def census(m):
    """Classify every var*var product in the model's constraint bodies."""
    offsets = ck._flat_offsets(m)
    tally = {}
    for con in m._constraints:
        stack = [con.body]
        while stack:
            nd = stack.pop()
            if isinstance(nd, BinaryOp):
                if (
                    nd.op == "*"
                    and ck._as_const(nd.left) is None
                    and ck._as_const(nd.right) is None
                ):
                    lk, rk = set(), set()
                    ok = affine_cols(nd.left, offsets, lk) and affine_cols(nd.right, offsets, rk)
                    if not ok:
                        tag = "nonaffine-factor"
                    else:
                        lkinds = {col_kind(m, offsets, c) for c in lk}
                        rkinds = {col_kind(m, offsets, c) for c in rk}
                        lb1 = len(lk) == 1 and lkinds == {"B"}
                        rb1 = len(rk) == 1 and rkinds == {"B"}
                        if lb1 or rb1:
                            tag = "binary x affine"
                        else:
                            tag = f"{sorted(lkinds)}x{sorted(rkinds)}"
                    tally[tag] = tally.get(tag, 0) + 1
                stack.extend([nd.left, nd.right])
            elif isinstance(nd, UnaryOp):
                stack.append(nd.operand)
            elif isinstance(nd, FunctionCall):
                stack.extend(nd.args)
    return tally


def reason(model):
    try:
        ck._build(model, None)
        return "BUILT"
    except ck.NotConvexKernel as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {e}"


def main(paths):
    from discopt._relax.gdp_reformulate import reformulate_gdp

    rows = []
    for p in paths:
        try:
            model = dm.from_nl(str(p))
            r = reason(model)
            t = census(reformulate_gdp(model, method="big-m")) if "bilinear" in r else {}
        except Exception as e:  # noqa: BLE001
            r, t = f"LOADFAIL {type(e).__name__}: {e}", {}
        rows.append((p.stem, r, t))
        print(f"{p.stem:28s} {r[:44]:46s} {t if t else ''}", flush=True)

    print("\n=== summary ===")
    blocked = [r for r in rows if r[1].startswith("bilinear")]
    print(
        f"{len(rows)} instances, {sum(1 for r in rows if r[1] == 'BUILT')} BUILT, "
        f"{len(blocked)} blocked on 'bilinear product'"
    )
    liftable = [r for r in blocked if r[2] and set(r[2]) == {"binary x affine"}]
    print(f"of the blocked: {len(liftable)} have ONLY binary x affine products (liftable):")
    for name, _r, t in liftable:
        print(f"   {name:28s} {t}")


if __name__ == "__main__":
    roots = [Path("python/tests/data/minlplib_nl"), Path("python/tests/data/minlplib")]
    files = sorted({f for root in roots for f in root.glob("*.nl")})
    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    main(files)
