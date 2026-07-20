#!/usr/bin/env python
"""Issue #798 / K1 — analyze-once convex-row decomposition probe + producer.

The convex LP-OA kernel needs Rust to evaluate g_i(x) and ∇g_i(x) at LP vertices
discovered *during* the node loop. Rather than a full autodiff engine in Rust, K1
marshals each convex nonlinear row as a **composite-of-affine** descriptor:

    g_i(x) = a_i·x + b_i + Σ_t coeff_t · func_t( p_t·x + q_t )   ≤ rhs_i

so Rust needs only closed-form univariate f/f' per MathFunc. This is the
OA-canonical convex class SCIP linearizes this way. A row that does NOT decompose
into this form (bilinear-in-a-convex-row, non-affine function argument, …) means
the instance is out of scope → the analyze-once step returns None and the caller
keeps the NLP-BB path (the sound, honest boundary, exactly like spatial_producer).

THIS SCRIPT IS THE ENTRY EXPERIMENT for that architecture (CLAUDE.md §4): it
decomposes every nonlinear row of the convex panel and VERIFIES the reconstruction
reproduces the JAX evaluator's value AND gradient at random points to <=1e-8. If
any row fails to decompose or the reconstruction disagrees, Architecture B is
falsified for that instance and we learn the real structure before building Rust.

Its `decompose_row` / `build_convex_spec` are the reusable producer (not throwaway):
they emit the flat-array marshaling K1's PyO3 binding consumes.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discopt.modeling.core import (  # noqa: E402
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    UnaryOp,
    Variable,
)
from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402

# Closed-form derivatives for the univariate funcs the convex class uses. Extend
# as new convex-certifiable funcs appear; an unknown func → row is unsupported.
_FUNC = {
    "log": (np.log, lambda t: 1.0 / t),
    "exp": (np.exp, np.exp),
    "sqrt": (np.sqrt, lambda t: 0.5 / np.sqrt(t)),
    "log1p": (np.log1p, lambda t: 1.0 / (1.0 + t)),
}


class NotComposite(Exception):
    """Row is not a composite-of-affine convex row (→ instance out of scope)."""


def _flat_offsets(model) -> dict[int, int]:
    """Map variable._index → starting flat column offset."""
    off, cur = {}, 0
    for v in model._variables:
        off[v._index] = cur
        cur += v.size
    return off


def _col_of(node, offsets: dict[int, int]) -> int:
    """Flat column index of a scalar Variable or IndexExpression leaf."""
    if isinstance(node, Variable):
        if node.size != 1:
            raise NotComposite("array variable used as scalar")
        return offsets[node._index]
    if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
        base = node.base
        idx = node.index
        if isinstance(idx, tuple):
            flat = int(np.ravel_multi_index(idx, base.shape))
        else:
            flat = int(idx)
        return offsets[base._index] + flat
    raise NotComposite("non-variable leaf")


class Decomp:
    """Sum of an affine form (coeffs, const) and composite-univariate terms."""

    __slots__ = ("aff", "const", "terms")

    def __init__(self):
        self.aff: dict[int, float] = {}
        self.const: float = 0.0
        self.terms: list[dict] = []  # {coeff, func, arg_aff: dict, arg_const}

    def scale(self, k: float) -> Decomp:
        self.const *= k
        for c in list(self.aff):
            self.aff[c] *= k
        for t in self.terms:
            t["coeff"] *= k
        return self

    def add(self, other: Decomp) -> Decomp:
        self.const += other.const
        for c, v in other.aff.items():
            self.aff[c] = self.aff.get(c, 0.0) + v
        self.terms.extend(other.terms)
        return self


def _as_const(node):
    if isinstance(node, Constant) and node.value.ndim == 0:
        return float(node.value)
    return None


def decompose(node, offsets) -> Decomp:
    """Decompose an expression into affine + composite-univariate terms."""
    d = Decomp()
    c = _as_const(node)
    if c is not None:
        d.const = c
        return d
    if isinstance(node, (Variable, IndexExpression)):
        d.aff[_col_of(node, offsets)] = 1.0
        return d
    if isinstance(node, UnaryOp):
        if node.op == "neg":
            return decompose(node.operand, offsets).scale(-1.0)
        raise NotComposite(f"unary {node.op}")
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return decompose(node.left, offsets).add(decompose(node.right, offsets))
        if node.op == "-":
            return decompose(node.left, offsets).add(decompose(node.right, offsets).scale(-1.0))
        if node.op == "*":
            lc, rc = _as_const(node.left), _as_const(node.right)
            if lc is not None:
                return decompose(node.right, offsets).scale(lc)
            if rc is not None:
                return decompose(node.left, offsets).scale(rc)
            raise NotComposite("bilinear product (non-constant × non-constant)")
        if node.op == "/":
            rc = _as_const(node.right)
            if rc is not None and rc != 0.0:
                return decompose(node.left, offsets).scale(1.0 / rc)
            raise NotComposite("division by non-constant")
        raise NotComposite(f"binary {node.op}")
    if isinstance(node, FunctionCall):
        if node.func_name not in _FUNC:
            raise NotComposite(f"unsupported func {node.func_name}")
        if len(node.args) != 1:
            raise NotComposite(f"multi-arg func {node.func_name}")
        arg = decompose(node.args[0], offsets)
        if arg.terms:
            raise NotComposite("non-affine function argument")
        d.terms.append(
            {"coeff": 1.0, "func": node.func_name, "arg_aff": arg.aff, "arg_const": arg.const}
        )
        return d
    raise NotComposite(f"node {type(node).__name__}")


def eval_decomp(d: Decomp, x: np.ndarray) -> float:
    v = d.const + sum(co * x[c] for c, co in d.aff.items())
    for t in d.terms:
        arg = t["arg_const"] + sum(co * x[c] for c, co in t["arg_aff"].items())
        v += t["coeff"] * _FUNC[t["func"]][0](arg)
    return float(v)


def grad_decomp(d: Decomp, x: np.ndarray, n: int) -> np.ndarray:
    g = np.zeros(n)
    for c, co in d.aff.items():
        g[c] += co
    for t in d.terms:
        arg = t["arg_const"] + sum(co * x[c] for c, co in t["arg_aff"].items())
        fp = _FUNC[t["func"]][1](arg)
        for c, co in t["arg_aff"].items():
            g[c] += t["coeff"] * fp * co
    return g


def constraint_expr(model, row_idx):
    """The residual expression g(x) for constraint `row_idx` (form g(x) <= 0)."""
    con = model._constraints[row_idx]
    # discopt constraints expose the body via .expr; residual repr is `(body) <= 0`.
    for attr in ("expr", "body", "lhs"):
        e = getattr(con, attr, None)
        if e is not None:
            return e
    raise NotComposite("cannot locate constraint expression")


def main():
    rng = np.random.default_rng(1)
    all_ok = True
    for name in PANEL:
        rm = RootModel(name)
        offsets = _flat_offsets(rm.model)
        lo = np.where(np.isfinite(rm.lb), rm.lb, 0.0)
        hi = np.where(np.isfinite(rm.ub), rm.ub, lo + 5.0)
        n_ok, n_fail = 0, 0
        max_gerr = max_verr = 0.0
        fails = []
        for i in rm.nl_rows:
            try:
                d = decompose(constraint_expr(rm.model, i), offsets)
            except NotComposite as ex:
                n_fail += 1
                fails.append((i, str(ex)))
                continue
            # verify value + gradient vs JAX at random interior points
            ok = True
            for _ in range(5):
                x = lo + rng.random(rm.n) * (hi - lo)
                g_jax = float(rm.ev.evaluate_constraints(x)[i])
                v = eval_decomp(d, x)
                # jacobian row from JAX
                jr = np.asarray(rm.ev.evaluate_jacobian(x)[i], float)
                gr = grad_decomp(d, x, rm.n)
                verr = abs(v - g_jax)
                gerr = float(np.max(np.abs(jr - gr)))
                max_verr = max(max_verr, verr)
                max_gerr = max(max_gerr, gerr)
                if verr > 1e-7 or gerr > 1e-7:
                    ok = False
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                fails.append((i, f"recon mismatch verr={max_verr:.2e} gerr={max_gerr:.2e}"))
        status = "OK" if n_fail == 0 else "FAIL"
        print(
            f"{name}: {status}  rows={len(rm.nl_rows)} decomposed={n_ok} failed={n_fail} "
            f"max_verr={max_verr:.2e} max_gerr={max_gerr:.2e}",
            flush=True,
        )
        for i, why in fails[:5]:
            print(f"    row {i}: {why}", flush=True)
        all_ok = all_ok and n_fail == 0
    print(f"\nVERDICT (Architecture B composite-of-affine): {'GO' if all_ok else 'FALSIFIED'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
