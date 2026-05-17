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


def template_parameter_names(template: str, *, degree: int | None = None, n_inputs: int) -> list[str]:
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
    raise ValueError(f"unknown template {template!r}")


def build_template(
    template: str,
    *,
    inputs: Sequence[InputSpec],
    response_name: str = "y",
    measurement_error: float = 1.0,
    degree: int | None = None,
) -> Experiment:
    """Dispatch by template name. Used by the CLI to keep dispatch in one place."""
    if template == "linear":
        return linear_template(inputs, response_name=response_name, measurement_error=measurement_error)
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
    raise ValueError(f"unknown template {template!r}")


TEMPLATE_NAMES = ("linear", "polynomial-1d", "response-surface-2d", "response-surface-3d")


__all__ = [
    "TEMPLATE_NAMES",
    "build_template",
    "linear_template",
    "polynomial_1d_template",
    "response_surface_template",
    "template_parameter_names",
]
