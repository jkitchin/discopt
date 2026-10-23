"""Numpy-evaluated "holes" in a POUNCE tape (#1903).

The POUNCE tape is built from a fixed set of ``NlExpr`` operations and cannot
call Python, so two kinds of node had no tape lowering and sent every model that
contained one to the JAX evaluator:

* ``dm.external`` blocks, whose value and derivatives come from user numpy
  callables. JAX was only the *plumbing* there: ``jax.pure_callback`` plus nested
  ``custom_jvp`` rules carried the caller's own Jacobian and Hessian.
* the spectral norm ``dm.norm(X, 2)`` of a 2-D ``X``, the largest singular value
  ``sigma_1``. It has no finite formula in the entries, but numpy's SVD gives its
  value, gradient and Hessian exactly.

The solver itself is not the obstacle: ``pounce.Problem`` drives a Python object
and calls its ``objective``/``gradient``/``jacobian``/``hessian`` methods every
iteration. So a hole is lowered as fresh tape variables ``t`` (indices ``n ..``),
its *argument* is taped separately, and the evaluator
(:class:`discopt._tape_nlp_evaluator.TapeNLPEvaluator`) evaluates ``t = f(a(x))``
with numpy and folds it back into ``x``-space by the chain rule. See that class
for the algebra.

Only lowerings that ask for it get holes: :func:`hole_sink` is opt-in, and with no
sink active the compiler refuses these nodes exactly as before, so every existing
caller is unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from typing import Any, Callable, Iterator, Optional

import numpy as np

__all__ = [
    "Hole",
    "HoleSink",
    "active_sink",
    "hole_sink",
    "spectral_norm",
    "spectral_norm_grad",
    "spectral_norm_hess",
]


@dataclasses.dataclass
class Hole:
    """One numpy-evaluated node: ``t = f(a)`` with ``a`` the lowered argument.

    ``out_shape``/``in_shape`` follow ``dm.external``'s convention: ``jac`` returns
    ``out_shape + in_shape`` and ``hess`` returns ``out_shape + in_shape + in_shape``.
    ``first`` is the tape index of the hole's first output variable.
    """

    name: str
    f: Callable[[np.ndarray], Any]
    jac: Callable[[np.ndarray], Any]
    hess: Callable[[np.ndarray], Any]
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    arg_nodes: np.ndarray  # object array of NlExpr, shape ``in_shape``
    first: int

    @property
    def n_out(self) -> int:
        return int(np.prod(self.out_shape, dtype=np.int64)) if self.out_shape else 1

    @property
    def n_in(self) -> int:
        return int(np.prod(self.in_shape, dtype=np.int64)) if self.in_shape else 1


class HoleSink:
    """Collects the holes of one lowering; ``n_base`` is the model's variable count."""

    def __init__(self, n_base: int) -> None:
        self.n_base = int(n_base)
        self.holes: list[Hole] = []

    @property
    def n_hole_vars(self) -> int:
        return sum(h.n_out for h in self.holes)

    def add(
        self,
        E: Any,
        *,
        name: str,
        f: Callable,
        jac: Callable,
        hess: Callable,
        arg_nodes: np.ndarray,
        out_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Register a hole and return its output as tape variables of ``out_shape``."""
        from discopt._nl_expr_compiler import UnsupportedForTape

        flat = np.asarray(arg_nodes, dtype=object).reshape(-1)
        # A hole's argument must be an ordinary x-space expression. A hole inside
        # another hole's argument would need the chain rule through two numpy
        # layers; refuse it rather than evaluate it wrongly.
        for node in flat:
            if any(int(j) >= self.n_base for j in node.variables()):
                raise UnsupportedForTape(f"{name} nested inside another numpy-evaluated node")
        first = self.n_base + self.n_hole_vars
        hole = Hole(
            name=name,
            f=f,
            jac=jac,
            hess=hess,
            in_shape=tuple(np.shape(arg_nodes)),
            out_shape=tuple(out_shape),
            arg_nodes=np.asarray(arg_nodes, dtype=object),
            first=first,
        )
        self.holes.append(hole)
        out = np.empty(hole.out_shape, dtype=object)
        flat_out = out.reshape(-1)
        for k in range(hole.n_out):
            flat_out[k] = E.var(first + k)
        return out


_SINK: contextvars.ContextVar[Optional[HoleSink]] = contextvars.ContextVar(
    "discopt_tape_hole_sink", default=None
)


def active_sink() -> Optional[HoleSink]:
    return _SINK.get()


@contextlib.contextmanager
def hole_sink(n_base: int) -> Iterator[HoleSink]:
    """Let the lowerings inside this block emit holes into the returned sink."""
    sink = HoleSink(n_base)
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


# --------------------------------------------------------------------------- #
# sigma_1, the spectral norm, with its exact derivatives
# --------------------------------------------------------------------------- #
def spectral_norm(x: np.ndarray) -> float:
    """``sigma_1(X)``, the largest singular value (``numpy.linalg.norm(X, 2)``)."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64), 2))


def spectral_norm_grad(x: np.ndarray) -> np.ndarray:
    """``d sigma_1 / dX = u_1 v_1^T`` for the top singular pair.

    Sign-invariant (flipping ``u`` and ``v`` together leaves the outer product
    unchanged). Where ``sigma_1`` is not simple it is a subgradient, not a
    gradient -- the same object the JAX SVD rule returns there.
    """
    a = np.asarray(x, dtype=np.float64)
    u, _, vt = np.linalg.svd(a)
    return np.outer(u[:, 0], vt[0])


def spectral_norm_hess(x: np.ndarray) -> np.ndarray:
    """``d^2 sigma_1 / dX^2`` as an ``(m, n, m, n)`` array.

    Through the symmetric Jordan-Wielandt matrix ``J = [[0, X], [X^T, 0]]``, whose
    largest eigenvalue is ``sigma_1`` with unit eigenvector ``w = (u, v)/sqrt(2)``.
    Second-order perturbation of a simple eigenvalue gives, for a direction ``E``
    (entering ``J`` as ``D = [[0, E], [E^T, 0]]``),

        d^2 lambda[E, E] = 2 * sum_{l != top} (w^T D w_l)^2 / (lambda_top - lambda_l),

    and ``w^T D w_l = <A_l, E>`` with ``A_l = w_u w_{l,v}^T + w_{l,u} w_v^T``. So
    ``H = sum_l c_l vec(A_l) vec(A_l)^T`` with ``c_l = 2 / (lambda_top - lambda_l)``.
    Checked against central differences of :func:`spectral_norm_grad` on random
    matrices (max error ~3e-9 at step 1e-5). Undefined where ``sigma_1`` is not
    simple (``lambda_top - lambda_l = 0``), exactly where the JAX SVD rule is.
    """
    a = np.asarray(x, dtype=np.float64)
    m, n = a.shape
    jw = np.zeros((m + n, m + n))
    jw[:m, m:] = a
    jw[m:, :m] = a.T
    lam, w = np.linalg.eigh(jw)
    top = int(np.argmax(lam))
    wt = w[:, top]
    h = np.zeros((m * n, m * n))
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in range(m + n):
            if col == top:
                continue
            wl = w[:, col]
            a_l = np.outer(wt[:m], wl[m:]) + np.outer(wl[:m], wt[m:])
            vec = a_l.reshape(-1)
            h += (2.0 / (lam[top] - lam[col])) * np.outer(vec, vec)
    return h.reshape(m, n, m, n)
