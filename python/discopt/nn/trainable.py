"""Trainable ML surrogates for hybrid physics+ML models.

This module is the *training* counterpart to the frozen-network embedding in
:mod:`discopt.nn.formulations`. Where a frozen formulation bakes trained weights
into constraint coefficients (to *optimize over* a fixed surrogate), the classes
here create the weights as decision ``Variable`` objects and emit ordinary symbolic
expressions, so a surrogate can be *trained simultaneously* with the rest of a
model — e.g. a neural network standing in for an unknown rate law inside a
collocation-discretized DAE, trained as one sparse NLP (the simultaneous
neural-DAE approach of Lueg et al., arXiv:2504.04665).

Two regimes, one story:

- **Frozen** (``discopt.nn.formulations`` / :func:`discopt.nn.add_predictor`):
  weights are constants; you optimize inputs/outputs of a fixed net, with global
  optimality guarantees.
- **Trainable** (this module): weights are ``Variable`` objects; you fit them to data.
  Trained weights bridge back to the frozen path via
  :meth:`TrainableNetwork.freeze`.

Design invariants (see ``docs/dev/hybrid-ml-implementation-plan.md`` §0.3):

- Surrogates emit *symbolic* expressions (``dm.tanh`` etc.), never an opaque
  ``dm.custom`` callable — that would disable global certification, integer
  variables, and ``.nl`` export.
- Layers are emitted in *matrix* form (array-shaped weight ``Variable`` objects and
  ``@``), not per-neuron scalar loops.
- Only **smooth** activations are allowed on the trainable path (LINEAR, TANH,
  SIGMOID, SOFTPLUS); RELU is refused loudly because a gradient-based NLP solver
  cannot handle its kink.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Union

import numpy as np

import discopt.modeling as dm
from discopt.modeling.core import Model, Variable
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

if TYPE_CHECKING:
    from discopt.solvers import NLPResult


# ── activation dispatch ────────────────────────────────────────────────────

# Smooth activations only: each maps to a native, relaxable modeling intrinsic.
_SYMBOLIC_ACTIVATIONS: dict[Activation, Callable] = {
    Activation.LINEAR: lambda z: z,
    Activation.TANH: dm.tanh,
    Activation.SIGMOID: dm.sigmoid,
    Activation.SOFTPLUS: dm.softplus,
}


def _resolve_activation(activation: Union[str, Activation]) -> Activation:
    """Coerce a string/enum to an ``Activation``, refusing non-smooth choices."""
    if isinstance(activation, str):
        try:
            activation = Activation(activation.lower())
        except ValueError:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of "
                f"{[a.value for a in _SYMBOLIC_ACTIVATIONS]}."
            ) from None
    if activation == Activation.RELU:
        raise ValueError(
            "RELU is not supported on the trainable path: it is non-smooth, so a "
            "gradient-based NLP solver oscillates at the kink. Use 'softplus' (a "
            "smooth ReLU surrogate) instead. RELU remains available on the frozen "
            "path via the big-M formulation (discopt.nn.add_predictor)."
        )
    if activation not in _SYMBOLIC_ACTIVATIONS:
        raise ValueError(
            f"Activation {activation} is not supported on the trainable path; "
            f"expected one of {[a.value for a in _SYMBOLIC_ACTIVATIONS]}."
        )
    return activation


def _with_trailing_axis(expr):
    """Return ``expr`` with one extra trailing axis, i.e. ``expr[..., None]``.

    Works for array expressions ``(nfe, ncp) -> (nfe, ncp, 1)`` and scalar
    variables ``() -> (1,)`` alike (both verified against the modeling layer).
    """
    return expr[..., None]


# ── dense layer ────────────────────────────────────────────────────────────


class TrainableDense:
    """One dense layer with trainable weights: ``y = act(x @ W + b)``.

    The weight matrix ``W`` (shape ``(n_in, n_out)``) and bias ``b`` (shape
    ``(n_out,)``) are created as continuous decision ``Variable`` objects on ``model``.

    Parameters
    ----------
    model : Model
        Model the weight variables are added to.
    n_in, n_out : int
        Input and output widths.
    activation : str or Activation
        Smooth activation applied after the affine map (default ``"tanh"``).
        RELU is refused (see module docstring).
    weight_bounds : tuple of float
        ``(lb, ub)`` box on every weight and bias (default ``(-8.0, 8.0)``).
    name : str
        Unique prefix for the weight variable names (``f"{name}_W"``,
        ``f"{name}_b"``).
    """

    def __init__(
        self,
        model: Model,
        n_in: int,
        n_out: int,
        *,
        activation: Union[str, Activation] = "tanh",
        weight_bounds: tuple[float, float] = (-8.0, 8.0),
        name: str,
    ) -> None:
        if n_in < 1 or n_out < 1:
            raise ValueError(f"n_in and n_out must be >= 1, got ({n_in}, {n_out})")
        self.model = model
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.activation = _resolve_activation(activation)
        self.weight_bounds = (float(weight_bounds[0]), float(weight_bounds[1]))
        self.name = name

        lb, ub = self.weight_bounds
        self.W: Variable = model.continuous(
            f"{name}_W", shape=(self.n_in, self.n_out), lb=lb, ub=ub
        )
        self.b: Variable = model.continuous(f"{name}_b", shape=(self.n_out,), lb=lb, ub=ub)

    def parameters(self) -> list[Variable]:
        """Return the trainable weight variables ``[W, b]``."""
        return [self.W, self.b]

    def _affine(self, inputs: tuple):
        """Build the pre-activation ``x @ W + b`` from the input contract."""
        if len(inputs) == self.n_in:
            # Scalar-per-point inputs (each shape S): sum_i x_i ⊗ W[i, :].
            # Handles the n_in == 1 case (a single (nfe, ncp) expression) too.
            z = None
            for i, xi in enumerate(inputs):
                term = _with_trailing_axis(xi) * self.W[i : i + 1, :]  # (..., 1)*(1, n_out)
                z = term if z is None else z + term
            return z + self.b
        if len(inputs) == 1:
            # A single pre-stacked input with trailing feature axis n_in.
            return inputs[0] @ self.W + self.b
        raise ValueError(
            f"Layer '{self.name}' expects {self.n_in} scalar-per-point input(s) or "
            f"1 pre-stacked input with trailing axis {self.n_in}, got {len(inputs)} "
            "argument(s)."
        )

    def __call__(self, *inputs):
        """Apply the layer. See the module docstring for the input contract."""
        if not inputs:
            raise ValueError(f"Layer '{self.name}' called with no inputs.")
        z = self._affine(inputs)
        return _SYMBOLIC_ACTIVATIONS[self.activation](z)


# ── network ────────────────────────────────────────────────────────────────


class TrainableNetwork:
    """A feedforward network of :class:`TrainableDense` layers.

    Parameters
    ----------
    model : Model
        Model the weight variables are added to.
    sizes : sequence of int
        Layer widths ``[n_in, h1, ..., n_out]`` (at least two entries).
    activation : str, Activation, or sequence
        Activation for the hidden layers (default ``"tanh"``). A sequence gives
        per-layer activations for *all* layers (length ``len(sizes) - 1``) and
        overrides ``output_activation``.
    output_activation : str or Activation
        Activation for the final layer (default ``"linear"``).
    weight_bounds : tuple of float
        Box on every weight/bias (default ``(-8.0, 8.0)``).
    name : str
        Unique prefix for all weight variable names.

    Notes
    -----
    Call contract for ``__call__``: for a 1-feature input network (``sizes[0] ==
    1``) pass the single (possibly array-shaped) input expression; for an
    ``n``-feature input pass either ``n`` scalar-per-point expressions or one
    pre-stacked expression whose trailing axis is ``n``. The output trailing
    axis is squeezed when ``sizes[-1] == 1`` so a scalar-output net returns an
    expression of the same shape as a 1-feature input.
    """

    def __init__(
        self,
        model: Model,
        sizes: Sequence[int],
        *,
        activation: Union[str, Activation, Sequence[Union[str, Activation]]] = "tanh",
        output_activation: Union[str, Activation] = "linear",
        weight_bounds: tuple[float, float] = (-8.0, 8.0),
        name: str,
    ) -> None:
        sizes = [int(s) for s in sizes]
        if len(sizes) < 2:
            raise ValueError(f"sizes must have at least 2 entries (in, out), got {sizes}")
        if any(s < 1 for s in sizes):
            raise ValueError(f"all layer sizes must be >= 1, got {sizes}")

        n_layers = len(sizes) - 1
        if isinstance(activation, (str, Activation)):
            acts: list[Union[str, Activation]] = [activation] * (n_layers - 1) + [output_activation]
        else:
            acts = list(activation)
            if len(acts) != n_layers:
                raise ValueError(
                    f"per-layer activation sequence must have length {n_layers} "
                    f"(one per layer), got {len(acts)}"
                )

        self.model = model
        self.sizes = sizes
        self.name = name
        self.weight_bounds = (float(weight_bounds[0]), float(weight_bounds[1]))
        self.layers: list[TrainableDense] = [
            TrainableDense(
                model,
                sizes[i],
                sizes[i + 1],
                activation=acts[i],
                weight_bounds=weight_bounds,
                name=f"{name}_L{i}",
            )
            for i in range(n_layers)
        ]
        # Set by from_definition() to stash a pretrained seed for fine-tuning.
        self._seed_definition: NetworkDefinition | None = None

    @property
    def n_inputs(self) -> int:
        return self.sizes[0]

    @property
    def n_outputs(self) -> int:
        return self.sizes[-1]

    def parameters(self) -> list[Variable]:
        """Return all trainable weight variables, in layer order."""
        return [p for layer in self.layers for p in layer.parameters()]

    def n_parameters(self) -> int:
        """Total number of scalar trainable parameters."""
        return int(sum(p.size for p in self.parameters()))

    def l2_penalty(self):
        """Return ``sum(p**2)`` over every weight — an L2 regularization term."""
        terms = [dm.sum(p**2) for p in self.parameters()]
        out = terms[0]
        for t in terms[1:]:
            out = out + t
        return out

    def __call__(self, *inputs):
        """Evaluate the network symbolically on ``inputs``."""
        if not inputs:
            raise ValueError(f"Network '{self.name}' called with no inputs.")
        # First layer honors the scalar-per-point vs pre-stacked contract; every
        # hidden layer receives the single pre-stacked activation of the last.
        h = self.layers[0](*inputs)
        for layer in self.layers[1:]:
            h = layer(h)
        if self.n_outputs == 1:
            # Squeeze the trailing feature axis: (..., 1) -> (...).
            h = h[..., 0]
        return h

    # ── initialization ─────────────────────────────────────────────────────

    def initial_values(
        self,
        seed: Union[int, np.random.Generator] = 0,
        scale: float = 0.8,
    ) -> dict[Variable, np.ndarray]:
        """Glorot-style random initial weights, deterministic given ``seed``.

        ``W ~ N(0, (scale / sqrt(n_in))**2)`` and ``b ~ N(0, (0.3*scale)**2)`` —
        the recipe measured to converge in ``scripts/hybrid_ml/exp_c_paper_scale.py``.
        Values are clipped into ``weight_bounds``.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        lb, ub = self.weight_bounds
        out: dict[Variable, np.ndarray] = {}
        for layer in self.layers:
            w_std = scale / np.sqrt(layer.n_in)
            out[layer.W] = np.clip(rng.standard_normal((layer.n_in, layer.n_out)) * w_std, lb, ub)
            out[layer.b] = np.clip(rng.standard_normal(layer.n_out) * (0.3 * scale), lb, ub)
        return out

    # ── freeze / thaw bridge to the frozen path ────────────────────────────

    def _weight_arrays(self, values) -> list[tuple[np.ndarray, np.ndarray]]:
        """Resolve ``values`` into ``[(W_i, b_i), ...]`` numpy arrays per layer."""
        var_values = _as_variable_dict(self.model, values)
        arrays = []
        for layer in self.layers:
            try:
                W = np.asarray(var_values[layer.W], dtype=np.float64).reshape(
                    layer.n_in, layer.n_out
                )
                b = np.asarray(var_values[layer.b], dtype=np.float64).reshape(layer.n_out)
            except KeyError as e:
                raise KeyError(
                    f"values is missing weights for layer '{layer.name}' "
                    f"(variable {e}). Pass the solve result or a dict covering "
                    "every weight variable."
                ) from None
            arrays.append((W, b))
        return arrays

    def freeze(self, values, *, input_bounds=None) -> NetworkDefinition:
        """Extract trained weights into a frozen :class:`NetworkDefinition`.

        Parameters
        ----------
        values : dict, NLPResult, or SolveResult
            The trained weights: a ``dict[Variable, np.ndarray]``, an
            :class:`~discopt.solvers.NLPResult` (uses ``.x`` +
            :func:`~discopt.warm_start.unflatten_solution`), or a ``SolveResult``
            (uses ``.value``).
        input_bounds : tuple of (np.ndarray, np.ndarray), optional
            Forwarded to :class:`NetworkDefinition` (needed only for frozen
            formulations that require bound propagation).

        Returns
        -------
        NetworkDefinition
            A frozen network whose ``forward()`` matches this symbolic net.
        """
        arrays = self._weight_arrays(values)
        layers = [DenseLayer(W, b, layer.activation) for (W, b), layer in zip(arrays, self.layers)]
        return NetworkDefinition(layers, input_bounds=input_bounds)

    @classmethod
    def from_definition(
        cls,
        model: Model,
        definition: NetworkDefinition,
        *,
        weight_bounds: tuple[float, float] = (-8.0, 8.0),
        name: str,
    ) -> "TrainableNetwork":
        """Build a trainable network seeded from a (pretrained) frozen definition.

        The weights become fresh ``Variable`` objects bounded by ``weight_bounds``; the
        definition's values are returned by :meth:`initial_values_from_definition`
        so training can *fine-tune* from a pretrained start. This is how the
        existing readers (sklearn/torch/onnx) become *initializers* rather than
        just freezers — no reader change needed.
        """
        sizes = [definition.layers[0].n_inputs] + [ly.n_outputs for ly in definition.layers]
        acts = [ly.activation for ly in definition.layers]
        net = cls(
            model,
            sizes,
            activation=acts,
            weight_bounds=weight_bounds,
            name=name,
        )
        net._seed_definition = definition  # stash for initial_values_from_definition
        return net

    def initial_values_from_definition(
        self, definition: NetworkDefinition | None = None
    ) -> dict[Variable, np.ndarray]:
        """Initial weights taken verbatim from a frozen definition (fine-tuning start)."""
        definition = definition or self._seed_definition
        if definition is None:
            raise ValueError("no definition available; pass one or build via from_definition()")
        if len(definition.layers) != len(self.layers):
            raise ValueError(
                f"definition has {len(definition.layers)} layers, network has {len(self.layers)}"
            )
        out: dict[Variable, np.ndarray] = {}
        lb, ub = self.weight_bounds
        for layer, dl in zip(self.layers, definition.layers):
            out[layer.W] = np.clip(np.asarray(dl.weights, dtype=np.float64), lb, ub)
            out[layer.b] = np.clip(np.asarray(dl.biases, dtype=np.float64), lb, ub)
        return out


# ── kernel expansion (GP-mean / RBF) ───────────────────────────────────────


class TrainableKernelExpansion:
    """A kernel expansion ``r(x) = sum_j alpha_j * k(x, c_j)``, linear in ``alpha``.

    This is the trainable Gaussian-process-mean / radial-basis surrogate. The
    kernel centers ``c_j`` and lengthscale are **fixed by design** — only the
    coefficients ``alpha`` are decision variables. Being linear in its
    parameters, the ML block contributes no nonconvexity of its own (only the
    physics dynamics couple it), which gives the best conditioning of the
    trainable surrogates: zero initialization suffices and there are no
    weight-symmetry local minima (measured in
    ``scripts/hybrid_ml/gp_dae_prototype.py``).

    Hyperparameters (centers, lengthscale) stay *outside* the NLP on purpose: the
    marginal-likelihood objective that would tune them needs matrix
    inverses/determinants that do not fit the algebraic NLP cleanly. Tune them
    with a separate marginal-likelihood / cross-validation loop and pass the
    result here.

    Parameters
    ----------
    model : Model
        Model the ``alpha`` variable is added to.
    centers : np.ndarray
        Fixed kernel centers, shape ``(n_centers,)`` (1-D input only in v1).
    lengthscale : float
        Fixed RBF lengthscale ``l`` (> 0).
    kernel : str
        Kernel name; only ``"rbf"`` is supported in v1.
    alpha_bounds : tuple of float
        Box on each coefficient (default ``(-50.0, 50.0)``).
    name : str
        Unique prefix for the ``alpha`` variable name.
    """

    def __init__(
        self,
        model: Model,
        centers: np.ndarray,
        *,
        lengthscale: float,
        kernel: str = "rbf",
        alpha_bounds: tuple[float, float] = (-50.0, 50.0),
        name: str,
    ) -> None:
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 1:
            raise ValueError(
                f"centers must be 1-D (v1 supports 1-D kernel inputs only), got "
                f"shape {centers.shape}. Multi-dimensional kernels are out of scope "
                "for this version."
            )
        if kernel != "rbf":
            raise ValueError(f"unsupported kernel {kernel!r}; only 'rbf' is available in v1")
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}")

        self.model = model
        self.centers = centers
        self.lengthscale = float(lengthscale)
        self.kernel = kernel
        self.name = name
        self.alpha_bounds = (float(alpha_bounds[0]), float(alpha_bounds[1]))
        self.alpha: Variable = model.continuous(
            f"{name}_alpha", shape=(centers.shape[0],), lb=alpha_bounds[0], ub=alpha_bounds[1]
        )

    @property
    def n_centers(self) -> int:
        return int(self.centers.shape[0])

    def parameters(self) -> list[Variable]:
        return [self.alpha]

    def n_parameters(self) -> int:
        return self.n_centers

    def l2_penalty(self):
        """Return ``sum(alpha**2)`` — an L2 regularization term."""
        return dm.sum(self.alpha**2)

    def __call__(self, x):
        """Evaluate ``sum_j alpha_j * exp(-(x - c_j)^2 / (2 l^2))`` symbolically."""
        denom = 2.0 * self.lengthscale**2
        out = None
        for j in range(self.n_centers):
            term = self.alpha[j] * dm.exp(-((x - float(self.centers[j])) ** 2) / denom)
            out = term if out is None else out + term
        return out

    def initial_values(self) -> dict[Variable, np.ndarray]:
        """Zero initialization (measured sufficient for the linear-in-alpha fit)."""
        return {self.alpha: np.zeros(self.n_centers)}


# ── training entry point ───────────────────────────────────────────────────


def _as_variable_dict(model: Model, values) -> dict[Variable, np.ndarray]:
    """Coerce a dict / NLPResult / SolveResult into ``{Variable: array}``."""
    if isinstance(values, dict):
        return values
    # NLPResult: has a flat `.x`.
    x = getattr(values, "x", None)
    if x is not None and not callable(getattr(values, "value", None)):
        from discopt.warm_start import unflatten_solution

        return unflatten_solution(model, np.asarray(x))
    # SolveResult: has a `.value(var)` accessor.
    if callable(getattr(values, "value", None)):
        return {v: np.asarray(values.value(v)) for v in model._variables}
    raise TypeError(
        f"Cannot interpret {type(values).__name__} as trained weights; pass a "
        "dict[Variable, ndarray], an NLPResult, or a SolveResult."
    )


def train(
    model: Model,
    *,
    initial_solution: dict | None = None,
    gauss_newton: bool = True,
    options: dict | None = None,
    backend: str = "auto",
) -> "NLPResult":
    """Train a hybrid model with a single **local** NLP solve.

    This is a thin convenience wrapper: it validates the warm start, builds an
    :class:`~discopt._jax.nlp_evaluator.NLPEvaluator` (with the Gauss-Newton
    Hessian approximation when the objective is a sum of squares — the common
    case for data fitting), and calls the local NLP backend directly. It does
    **not** route through :meth:`Model.solve`, which would attempt global
    dispatch; the returned result therefore carries **no global optimality
    certificate**. For certified training see the (locked) HM4 track.

    Parameters
    ----------
    model : Model
        A hybrid model with objective (typically least-squares + regularization)
        and constraints (collocation, etc.) already set.
    initial_solution : dict, optional
        ``{Variable: value}`` warm start (e.g.
        ``fit.warm_start() | net.initial_values()``). Midpoints are used for any
        unspecified variable.
    gauss_newton : bool
        Use the Gauss-Newton Hessian for sum-of-squares objectives (default
        True). Falls back to the exact Hessian automatically if the objective is
        not recognized as a sum of squares.
    options : dict, optional
        Backend options (e.g. ``{"max_iter": 3000, "tol": 1e-8}``).
    backend : str
        ``"auto"`` (POUNCE, else cyipopt), ``"pounce"``, or ``"cyipopt"``.

    Returns
    -------
    NLPResult
        The local solve result; ``.x`` is the trained point.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.warm_start import validate_initial_solution

    if backend == "auto":
        from discopt.solvers.nlp_backend import available_backends

        avail = available_backends()
        backend = "pounce" if "pounce" in avail else "cyipopt"

    if backend not in ("pounce", "cyipopt"):
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'pounce', or 'cyipopt'")

    x0 = None
    if initial_solution is not None:
        x0 = validate_initial_solution(model, initial_solution)

    evaluator = NLPEvaluator(model, gauss_newton=gauss_newton)
    if x0 is None:
        lb, ub = evaluator.variable_bounds
        x0 = 0.5 * (np.clip(lb, -100.0, 100.0) + np.clip(ub, -100.0, 100.0))

    if backend == "pounce":
        from discopt.solvers.nlp_pounce import solve_nlp as _solve_pounce

        return _solve_pounce(evaluator, x0, options=options)
    from discopt.solvers.nlp_ipopt import solve_nlp as _solve_ipopt

    return _solve_ipopt(evaluator, x0, options=options)
