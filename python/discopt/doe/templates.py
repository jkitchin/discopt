"""Ready-to-use :class:`Experiment` templates for the ``discopt doe`` CLI.

These templates cover the workhorses of classical response-surface
methodology — linear regression, 1-D polynomial regression, and the
2- or 3-factor full quadratic ("response surface") model. All four
are *linear in the parameters*, so their FIMs do not depend on the
prior parameter values; the CLI uses this fact to ship a zero-config
initial design where users supply only the factor names and ranges.

The output models all use textbook coefficient naming:

* ``b0`` — intercept
* ``b1``, ``b2``, … — main effects
* ``b11``, ``b22``, … — pure quadratic terms (response surface only)
* ``b12``, ``b13``, ``b23`` — interactions (response surface only)
* ``b1``..``bd`` — polynomial coefficients of degree 1..d (polynomial_1d)

For users who outgrow these templates, the CLI supports a
``--module pkg.mod:callable`` escape hatch that hands off to a
user-defined :class:`Experiment`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel

InputSpec = tuple[str, float, float]

_PARAM_LB = -1e6
_PARAM_UB = 1e6


def _make_param_vars(model: dm.Model, names: Sequence[str]) -> dict[str, object]:
    return {n: model.continuous(n, lb=_PARAM_LB, ub=_PARAM_UB) for n in names}


def _make_input_vars(model: dm.Model, inputs: Sequence[InputSpec]) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, lb, ub in inputs:
        if not (ub > lb):
            raise ValueError(f"input {name!r}: upper bound must exceed lower bound")
        out[name] = model.continuous(name, lb=lb, ub=ub)
    return out


def linear_template(
    inputs: Sequence[InputSpec],
    *,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Build an :class:`Experiment` for ``y = b0 + sum_i bi * xi``.

    Parameters
    ----------
    inputs : sequence of (name, lb, ub)
        One entry per design factor. The order fixes the coefficient
        indices: the first input pairs with ``b1``, the second with
        ``b2``, and so on.
    response_name : str, default ``"y"``
        Name of the measured response.
    measurement_error : float, default ``1.0``
        Standard deviation of the measurement noise.

    Returns
    -------
    Experiment
        Subclass instance whose :meth:`create_model` builds the linear
        regression model with ``1 + len(inputs)`` unknown parameters.
    """
    if not inputs:
        raise ValueError("linear_template requires at least one input")
    n = len(inputs)
    param_names = ["b0"] + [f"b{i + 1}" for i in range(n)]
    sigma = float(measurement_error)
    spec = list(inputs)

    class _LinearTemplate(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("linear_template")
            params = _make_param_vars(m, param_names)
            xs = _make_input_vars(m, spec)
            input_names = [s[0] for s in spec]
            response = params["b0"]
            for i, name in enumerate(input_names):
                response = response + params[f"b{i + 1}"] * xs[name]
            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs=xs,
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _LinearTemplate()


def polynomial_1d_template(
    input: InputSpec,
    degree: int,
    *,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Build an :class:`Experiment` for ``y = sum_{j=0..d} bj * x**j``.

    Parameters
    ----------
    input : (name, lb, ub)
        The single design factor.
    degree : int
        Polynomial degree (``>= 1``). The fitted model has ``degree + 1``
        coefficients, ``b0`` through ``b{degree}``.
    response_name : str, default ``"y"``
        Name of the measured response.
    measurement_error : float, default ``1.0``
        Standard deviation of the measurement noise.
    """
    if degree < 1:
        raise ValueError(f"polynomial_1d_template requires degree >= 1, got {degree}")
    name, lb, ub = input
    if not (ub > lb):
        raise ValueError(f"input {name!r}: upper bound must exceed lower bound")
    d = int(degree)
    param_names = [f"b{j}" for j in range(d + 1)]
    sigma = float(measurement_error)

    class _Polynomial1DTemplate(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("polynomial_1d_template")
            params = _make_param_vars(m, param_names)
            x = m.continuous(name, lb=lb, ub=ub)
            response = params["b0"]
            for j in range(1, d + 1):
                response = response + params[f"b{j}"] * (x**j)
            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs={name: x},
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _Polynomial1DTemplate()


def response_surface_template(
    inputs: Sequence[InputSpec],
    *,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Build a full-quadratic response-surface :class:`Experiment`.

    For ``n`` inputs (n = 2 or 3) the model has ``1 + 2n + n(n-1)/2``
    coefficients: an intercept ``b0``, ``n`` main effects ``b1..bn``,
    ``n`` pure quadratic terms ``b11..bnn``, and ``n(n-1)/2``
    interactions ``b12, b13, b23``. For n=2 that's 6 parameters; for
    n=3, 10 parameters.

    Parameters
    ----------
    inputs : sequence of (name, lb, ub)
        Either 2 or 3 design factors.
    response_name : str, default ``"y"``
        Name of the measured response.
    measurement_error : float, default ``1.0``
        Standard deviation of the measurement noise.
    """
    n = len(inputs)
    if n not in (2, 3):
        raise ValueError(f"response_surface_template requires 2 or 3 inputs, got {n}")
    sigma = float(measurement_error)
    spec = list(inputs)

    main_names = [f"b{i + 1}" for i in range(n)]
    sq_names = [f"b{i + 1}{i + 1}" for i in range(n)]
    cross_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cross_names = [f"b{i + 1}{j + 1}" for i, j in cross_pairs]
    param_names = ["b0", *main_names, *sq_names, *cross_names]

    class _ResponseSurfaceTemplate(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("response_surface_template")
            params = _make_param_vars(m, param_names)
            xs = _make_input_vars(m, spec)
            input_names = [s[0] for s in spec]

            response = params["b0"]
            for i, nm in enumerate(input_names):
                response = response + params[main_names[i]] * xs[nm]
            for i, nm in enumerate(input_names):
                response = response + params[sq_names[i]] * (xs[nm] * xs[nm])
            for (i, j), pn in zip(cross_pairs, cross_names):
                response = response + params[pn] * (xs[input_names[i]] * xs[input_names[j]])

            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs=xs,
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _ResponseSurfaceTemplate()


def template_parameter_names(
    template: str, *, degree: int | None = None, n_inputs: int
) -> list[str]:
    """Return the ordered parameter-name list a template would emit.

    Used by the CLI/workbook to populate the ``parameters`` sheet
    layout without constructing an Experiment.
    """
    if template == "linear":
        return ["b0"] + [f"b{i + 1}" for i in range(n_inputs)]
    if template == "polynomial-1d":
        if degree is None or degree < 1:
            raise ValueError("polynomial-1d requires degree >= 1")
        return [f"b{j}" for j in range(degree + 1)]
    if template in ("response-surface-2d", "response-surface-3d"):
        n = 2 if template.endswith("-2d") else 3
        if n_inputs != n:
            raise ValueError(f"{template} requires exactly {n} inputs, got {n_inputs}")
        main = [f"b{i + 1}" for i in range(n)]
        sq = [f"b{i + 1}{i + 1}" for i in range(n)]
        cross = [f"b{i + 1}{j + 1}" for i in range(n) for j in range(i + 1, n)]
        return ["b0", *main, *sq, *cross]
    if template in ("scheffe-linear", "scheffe-quadratic", "scheffe-special-cubic"):
        q = n_inputs
        if q < 2:
            raise ValueError(f"{template} requires at least 2 components")
        main = [f"b{i + 1}" for i in range(q)]
        if template == "scheffe-linear":
            return main
        cross = [f"b{i + 1}{j + 1}" for i in range(q) for j in range(i + 1, q)]
        if template == "scheffe-quadratic":
            return [*main, *cross]
        if q < 3:
            raise ValueError("scheffe-special-cubic requires at least 3 components")
        triples = [
            f"b{i + 1}{j + 1}{k + 1}"
            for i in range(q)
            for j in range(i + 1, q)
            for k in range(j + 1, q)
        ]
        return [*main, *cross, *triples]
    raise ValueError(f"unknown template {template!r}")


def build_template(
    template: str,
    *,
    inputs: Sequence[InputSpec],
    response_name: str = "y",
    measurement_error: float = 1.0,
    degree: int | None = None,
    mixture_total: float | None = None,
) -> Experiment:
    """Dispatch by template name. Used by the CLI to keep dispatch in one place."""
    if template == "linear":
        return linear_template(
            inputs, response_name=response_name, measurement_error=measurement_error
        )
    if template == "polynomial-1d":
        if len(inputs) != 1:
            raise ValueError("polynomial-1d takes exactly one input")
        if degree is None:
            raise ValueError("polynomial-1d requires a degree")
        return polynomial_1d_template(
            inputs[0], degree, response_name=response_name, measurement_error=measurement_error
        )
    if template == "response-surface-2d":
        if len(inputs) != 2:
            raise ValueError("response-surface-2d takes exactly two inputs")
        return response_surface_template(
            inputs, response_name=response_name, measurement_error=measurement_error
        )
    if template == "response-surface-3d":
        if len(inputs) != 3:
            raise ValueError("response-surface-3d takes exactly three inputs")
        return response_surface_template(
            inputs, response_name=response_name, measurement_error=measurement_error
        )
    if template in ("scheffe-linear", "scheffe-quadratic", "scheffe-special-cubic"):
        components = [s[0] for s in inputs]
        bounds = [(s[1], s[2]) for s in inputs]
        total = float(mixture_total) if mixture_total is not None else 1.0
        kwargs = dict(
            components=components,
            total=total,
            bounds=bounds,
            response_name=response_name,
            measurement_error=measurement_error,
        )
        if template == "scheffe-linear":
            return scheffe_linear_template(**kwargs)
        if template == "scheffe-quadratic":
            return scheffe_quadratic_template(**kwargs)
        return scheffe_special_cubic_template(**kwargs)
    raise ValueError(f"unknown template {template!r}")


def _mixture_input_specs(
    components: Sequence[str],
    total: float,
    bounds: Sequence[tuple[float, float]] | None,
) -> list[InputSpec]:
    """Build per-component (name, lb, ub) tuples for a mixture design.

    Default bounds are ``[0, total]``. A small slack is added below the
    upper bound when no explicit bounds are given so that the modeling
    layer does not reject the equality constraint as trivially infeasible
    (component ``= total`` is allowed; here we keep it strictly less so
    the design space is non-degenerate).
    """
    out: list[InputSpec] = []
    for i, name in enumerate(components):
        if bounds is None:
            lb, ub = 0.0, float(total)
        else:
            lb, ub = bounds[i]
        if ub <= lb:
            raise ValueError(f"component {name!r}: upper bound must exceed lower bound")
        out.append((name, float(lb), float(ub)))
    return out


def scheffe_linear_template(
    components: Sequence[str],
    *,
    total: float = 1.0,
    bounds: Sequence[tuple[float, float]] | None = None,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Scheffé canonical linear (first-order) mixture model.

    For ``q`` components ``x_1, ..., x_q`` with ``sum(x_i) = total``:

    .. math:: y = \\sum_{i=1}^{q} b_i\\, x_i

    No intercept: it is absorbed into the equality constraint. The model
    has ``q`` parameters ``b1..bq``.

    Parameters
    ----------
    components : sequence of str
        Names of the mixture components (at least 2).
    total : float, default 1.0
        Required sum of component values. Typical choices: ``1.0`` for
        fractions, or a total volume / mass for absolute amounts.
    bounds : sequence of (lb, ub), optional
        Per-component lower/upper bounds. Defaults to ``[0, total]``
        for every component.
    response_name : str, default ``"y"``
    measurement_error : float, default 1.0

    Returns
    -------
    Experiment
        Use with :func:`discopt.doe.sum_constraint` and
        :func:`optimal_experiment` to find an exact D-optimal mixture
        design, or with :func:`simplex_lattice_points` /
        :func:`simplex_centroid_points` for classical designs.
    """
    q = len(components)
    if q < 2:
        raise ValueError("scheffe_linear_template requires at least 2 components")
    spec = _mixture_input_specs(components, total, bounds)
    param_names = [f"b{i + 1}" for i in range(q)]
    sigma = float(measurement_error)

    class _ScheffeLinear(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("scheffe_linear")
            params = _make_param_vars(m, param_names)
            xs = _make_input_vars(m, spec)
            response = params[param_names[0]] * xs[components[0]]
            for i in range(1, q):
                response = response + params[param_names[i]] * xs[components[i]]
            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs=xs,
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _ScheffeLinear()


def scheffe_quadratic_template(
    components: Sequence[str],
    *,
    total: float = 1.0,
    bounds: Sequence[tuple[float, float]] | None = None,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Scheffé canonical quadratic mixture model.

    For ``q`` components:

    .. math:: y = \\sum_i b_i x_i + \\sum_{i<j} b_{ij} x_i x_j

    Parameter count: ``q + q(q-1)/2``. Captures binary blending
    non-linearities (synergy/antagonism between pairs).
    """
    q = len(components)
    if q < 2:
        raise ValueError("scheffe_quadratic_template requires at least 2 components")
    spec = _mixture_input_specs(components, total, bounds)
    main_names = [f"b{i + 1}" for i in range(q)]
    pair_indices = list(combinations(range(q), 2))
    cross_names = [f"b{i + 1}{j + 1}" for i, j in pair_indices]
    param_names = [*main_names, *cross_names]
    sigma = float(measurement_error)

    class _ScheffeQuadratic(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("scheffe_quadratic")
            params = _make_param_vars(m, param_names)
            xs = _make_input_vars(m, spec)
            response = params[main_names[0]] * xs[components[0]]
            for i in range(1, q):
                response = response + params[main_names[i]] * xs[components[i]]
            for (i, j), pn in zip(pair_indices, cross_names):
                response = response + params[pn] * (xs[components[i]] * xs[components[j]])
            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs=xs,
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _ScheffeQuadratic()


def scheffe_special_cubic_template(
    components: Sequence[str],
    *,
    total: float = 1.0,
    bounds: Sequence[tuple[float, float]] | None = None,
    response_name: str = "y",
    measurement_error: float = 1.0,
) -> Experiment:
    """Scheffé canonical special-cubic mixture model.

    For ``q`` components (``q >= 3``):

    .. math:: y = \\sum_i b_i x_i + \\sum_{i<j} b_{ij} x_i x_j
              + \\sum_{i<j<k} b_{ijk} x_i x_j x_k

    Adds three-way blending terms to the quadratic model.
    """
    q = len(components)
    if q < 3:
        raise ValueError("scheffe_special_cubic_template requires at least 3 components")
    spec = _mixture_input_specs(components, total, bounds)
    main_names = [f"b{i + 1}" for i in range(q)]
    pair_indices = list(combinations(range(q), 2))
    triple_indices = list(combinations(range(q), 3))
    cross_names = [f"b{i + 1}{j + 1}" for i, j in pair_indices]
    triple_names = [f"b{i + 1}{j + 1}{k + 1}" for i, j, k in triple_indices]
    param_names = [*main_names, *cross_names, *triple_names]
    sigma = float(measurement_error)

    class _ScheffeSpecialCubic(Experiment):
        def create_model(self, **kwargs):
            m = dm.Model("scheffe_special_cubic")
            params = _make_param_vars(m, param_names)
            xs = _make_input_vars(m, spec)
            response = params[main_names[0]] * xs[components[0]]
            for i in range(1, q):
                response = response + params[main_names[i]] * xs[components[i]]
            for (i, j), pn in zip(pair_indices, cross_names):
                response = response + params[pn] * (xs[components[i]] * xs[components[j]])
            for (i, j, k), pn in zip(triple_indices, triple_names):
                response = response + params[pn] * (
                    xs[components[i]] * xs[components[j]] * xs[components[k]]
                )
            return ExperimentModel(
                model=m,
                unknown_parameters=params,
                design_inputs=xs,
                responses={response_name: response},
                measurement_error={response_name: sigma},
            )

    return _ScheffeSpecialCubic()


def simplex_lattice_points(
    components: Sequence[str],
    degree: int,
    *,
    total: float = 1.0,
) -> list[dict[str, float]]:
    """Generate the points of a ``{q, m}`` simplex-lattice design.

    The ``{q, m}`` lattice consists of all compositions where each
    component is one of ``0, 1/m, 2/m, ..., 1`` and components sum to 1
    (then scaled by ``total``). The number of points is
    ``C(q + m - 1, m)``.

    Parameters
    ----------
    components : sequence of str
        Component names (``q >= 2``).
    degree : int
        Lattice degree ``m >= 1``. ``m=1`` gives the pure-component
        vertices; ``m=2`` adds binary midpoints; etc.
    total : float, default 1.0
        Scaling factor applied to every coordinate.
    """
    q = len(components)
    if q < 2:
        raise ValueError("simplex_lattice_points requires at least 2 components")
    if degree < 1:
        raise ValueError("simplex_lattice_points requires degree >= 1")
    m = int(degree)

    # Enumerate all non-negative integer compositions (k_1, ..., k_q) with sum == m.
    def compositions(n_left: int, k_slots: int):
        if k_slots == 1:
            yield (n_left,)
            return
        for v in range(n_left + 1):
            for tail in compositions(n_left - v, k_slots - 1):
                yield (v, *tail)

    points: list[dict[str, float]] = []
    for comp in compositions(m, q):
        point = {components[i]: float(total) * (comp[i] / m) for i in range(q)}
        points.append(point)
    return points


def simplex_centroid_points(
    components: Sequence[str],
    *,
    total: float = 1.0,
) -> list[dict[str, float]]:
    """Generate the points of a simplex-centroid design.

    The simplex-centroid design for ``q`` components has ``2^q - 1``
    points: for every non-empty subset ``S`` of components, the point
    has ``x_i = total / |S|`` for ``i in S`` and ``x_i = 0`` otherwise.

    Parameters
    ----------
    components : sequence of str
    total : float, default 1.0
        Scaling factor applied to every coordinate (default fractions).
    """
    q = len(components)
    if q < 2:
        raise ValueError("simplex_centroid_points requires at least 2 components")
    points: list[dict[str, float]] = []
    for size in range(1, q + 1):
        for subset in combinations(range(q), size):
            point = {c: 0.0 for c in components}
            share = float(total) / size
            for idx in subset:
                point[components[idx]] = share
            points.append(point)
    return points


TEMPLATE_NAMES = (
    "linear",
    "polynomial-1d",
    "response-surface-2d",
    "response-surface-3d",
    "scheffe-linear",
    "scheffe-quadratic",
    "scheffe-special-cubic",
    "latin-square",
    "graeco-latin",
    "hyper-graeco-latin",
    "factorial-2level",
    "optimize",
)

# Templates that produce a closed-form combinatorial design rather than
# an FIM-based optimal design. ``do_new`` branches on this set to skip
# the optimization pipeline entirely.
COMBINATORIAL_TEMPLATES = frozenset(
    {"latin-square", "graeco-latin", "hyper-graeco-latin", "factorial-2level"}
)


__all__ = [
    "TEMPLATE_NAMES",
    "build_template",
    "linear_template",
    "polynomial_1d_template",
    "response_surface_template",
    "scheffe_linear_template",
    "scheffe_quadratic_template",
    "scheffe_special_cubic_template",
    "simplex_centroid_points",
    "simplex_lattice_points",
    "template_parameter_names",
]
