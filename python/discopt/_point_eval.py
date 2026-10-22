"""Evaluate an expression at a point without JAX (#1873).

Several helpers off the B&B path only need the *value* of an expression at a flat
point: the multi-objective sweeps' cross-evaluation, the GDP simplex lowering's
source residuals, the nonsmooth lift's warm-start completion. They used the JAX
DAG compiler for it, which put JAX on otherwise JAX-free solves and made those
features fail on an install without the ``discopt[jax]`` extra.

The POUNCE tape the NLP evaluator already uses lowers the same expressions to
``NlExpr`` nodes whose ``eval`` returns a float64 value. JAX remains the fallback
for an expression the tape has no opcode for (``dm.custom``, a matrix 2-norm, ...),
which is the same rule ``build_evaluator`` applies, and it refuses with the named
error when JAX is not installed.

Parameters are compile-time constants on the tape, so a compiled expression is
valid only for the parameter values it was built with. :class:`PointEvaluator`
keys its cache on those values, which is what lets the multi-objective sweeps
mutate ``Parameter.value`` between solves.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _tape_or_none(expr: Any, model: Any) -> np.ndarray | None:
    """``expr`` lowered to an object array of ``NlExpr`` nodes, or ``None``.

    ``None`` means the tape cannot represent it (or POUNCE is unusable), never a
    numerical failure, so a caller cannot confuse "no tape" with "bad point".
    """
    from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_array
    from discopt._tape_nlp_evaluator import pounce_usable

    if not pounce_usable():
        return None
    try:
        return compile_to_nl_array(expr, model)
    except UnsupportedForTape:
        return None


def _jax_fallback(expr: Any, model: Any, why: str) -> Callable[[np.ndarray], np.ndarray]:
    from discopt._optional_jax import require_jax

    require_jax(f"Evaluating this expression ({why})")
    from discopt._relax.dag_compiler import compile_expression

    fn = compile_expression(expr, model)
    return lambda x: np.asarray(fn(x), dtype=np.float64)


def compile_point_fn(expr: Any, model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """``fn(x_flat) -> value`` for ``expr``, with the same shape as the expression.

    Parameters are taken at their current values, the same snapshot
    ``dag_compiler.compile_expression`` takes.
    """
    nodes = _tape_or_none(expr, model)
    if nodes is None:
        return _jax_fallback(expr, model, "the POUNCE tape has no lowering for it")
    flat_nodes = nodes.reshape(-1).tolist()
    shape = nodes.shape

    def fn(x: np.ndarray) -> np.ndarray:
        xs = [float(t) for t in np.asarray(x, dtype=np.float64).reshape(-1)]
        out = np.fromiter((e.eval(xs) for e in flat_nodes), dtype=np.float64, count=len(flat_nodes))
        return out.reshape(shape)

    return fn


def evaluate_at(expr: Any, model: Any, x_flat: np.ndarray) -> np.ndarray:
    """The value of ``expr`` at ``x_flat`` (one-shot :func:`compile_point_fn`)."""
    return compile_point_fn(expr, model)(x_flat)


class PointEvaluator:
    """Evaluate fixed expressions at points, re-lowering only when parameters change."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._params: bytes | None = None
        # Keyed by ``id(expr)``: the caller holds the expressions, so ids are stable.
        self._cache: dict[int, Callable[[np.ndarray], np.ndarray]] = {}

    def _param_key(self) -> bytes:
        return b"|".join(
            np.ascontiguousarray(p.value, dtype=np.float64).tobytes()
            for p in self._model._parameters
        )

    def __call__(self, expr: Any, x_flat: np.ndarray) -> np.ndarray:
        params = self._param_key()
        if params != self._params:
            # Every lowering baked the old values in; none of them is valid now.
            self._cache.clear()
            self._params = params
        fn = self._cache.get(id(expr))
        if fn is None:
            fn = compile_point_fn(expr, self._model)
            self._cache[id(expr)] = fn
        return fn(x_flat)
