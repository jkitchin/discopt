"""The ``Surrogate`` protocol — the contract a trainable ML surrogate satisfies.

A *hybrid model* replaces one unknown term of a physics model (a rate law, a
transport coefficient, a closure) with a trainable ML surrogate, then trains the
surrogate's parameters *simultaneously* with the discretized states as one NLP
(see :mod:`discopt.nn.trainable` and :mod:`discopt.dae.fit`). This module declares
the small, duck-typed contract such a surrogate follows, so *any* object — not
just the built-in :class:`~discopt.nn.trainable.TrainableNetwork` /
:class:`~discopt.nn.trainable.TrainableKernelExpansion` — can plug into the
hybrid pipeline.

The one load-bearing requirement
--------------------------------
A surrogate participates in the dynamics **only** by being *called* inside your
ODE/DAE right-hand side and returning a discopt expression:

    def rhs(t, states, algebraics, controls):
        r = my_surrogate(states["cA"])      # expression in  ->  expression out
        return {"cA": -r, "cB": r}

So the single hard requirement is ``__call__(x) -> expression`` where the output
is an ordinary discopt expression built from (a) the native **smooth** intrinsics
(``exp``, ``log``, ``sqrt``, ``sin``/``cos``/``tan``, ``tanh``, ``sigmoid``,
``softplus``, ``abs``, ``min``/``max``, ``pow``, ``+ - * / @``) and (b) whatever
decision :class:`~discopt.modeling.core.Variable` objects you want trained. The
solver is entirely decoupled from the surrogate: :func:`discopt.nn.train` operates
on the assembled ``Model``, and :func:`discopt.dae.fit_trajectories` takes your
``rhs`` callable — neither calls the surrogate directly.

The conventional methods
------------------------
The remaining methods are *conveniences* the warm-start and objective helpers
rely on; every built-in surrogate provides them, and a custom one should too:

- ``parameters()`` — the trainable ``Variable`` objects (for introspection /
  structure-aware factorization / counting).
- ``n_parameters()`` — total scalar parameter count.
- ``l2_penalty()`` — ``sum(p**2)`` over the parameters, an L2 regularization
  term to add to the objective.
- ``initial_values()`` — a ``{Variable: array}`` warm start, merged with
  :meth:`TrajectoryFit.warm_start` via ``|`` to seed the solve.

What this constrains (and what it does not)
-------------------------------------------
The contract is about *shape*, not about *which ML model*. The real constraint is
mathematical: a surrogate is trainable in the NLP iff it reduces to a **smooth
expression in continuous decision variables**. That admits neural nets (smooth
activations), Gaussian-process / kernel means (linear in the coefficients),
polynomials, splines, soft/differentiable decision trees, and any fixed-structure
symbolic formula (e.g. a symbolic-regression result whose constants are trained
in the NLP). It excludes non-smooth or discrete structure — ReLU, hard-split
decision trees, or *searching* a symbolic structure — which belong on the frozen
path (embed via :func:`discopt.nn.add_predictor`, optimize *over* the model) or in
an MINLP, not the smooth simultaneous NLP.

Usage
-----
``Surrogate`` is ``runtime_checkable``, so ``isinstance(obj, Surrogate)`` reports
structural conformance (method presence). Because it is a
:class:`typing.Protocol`, a class need not inherit from it — implementing the
methods is enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from discopt.modeling.core import Variable


@runtime_checkable
class Surrogate(Protocol):
    """Structural contract for a trainable ML surrogate in a hybrid model.

    See the module docstring for the full rationale. In brief: ``__call__`` is
    the one load-bearing method (it produces the symbolic term embedded in the
    dynamics); the rest are the conventional methods the warm-start and objective
    helpers use. Conformance is structural — implement the methods, no
    inheritance required.
    """

    def __call__(self, *inputs: Any) -> Any:
        """Evaluate the surrogate symbolically: expression(s) in, expression out.

        The returned value must be an ordinary discopt expression built from the
        smooth intrinsics and the surrogate's trainable ``Variable`` objects.
        """
        ...

    def parameters(self) -> list[Variable]:
        """Return the trainable decision ``Variable`` objects."""
        ...

    def n_parameters(self) -> int:
        """Return the total number of scalar trainable parameters."""
        ...

    def l2_penalty(self) -> Any:
        """Return ``sum(p**2)`` over all parameters — an L2 regularization term."""
        ...

    def initial_values(self) -> dict[Variable, np.ndarray]:
        """Return a ``{Variable: array}`` warm start for the parameters."""
        ...
