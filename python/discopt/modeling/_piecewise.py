"""Declared piecewise-linear functions of a model expression (issue #1482).

:meth:`Model.piecewise` lets a user state a *tabulated* univariate relationship
``y = f(x)`` -- a pump curve, a tariff schedule, a property fit -- as breakpoints
``b_0 < b_1 < ... < b_{n-1}`` and values ``v_0, ..., v_{n-1}``. ``f`` is the
continuous piecewise-linear interpolant of that table on ``[b_0, b_{n-1}]``.

Exact, not approximate
----------------------
The table *defines* ``f``. Every formulation here is an **exact** mixed-integer
linear representation of the graph ``{(x, f(x)) : b_0 <= x <= b_{n-1}}``: at any
feasible point of the lowered rows, ``y`` equals ``f(x)`` (up to the LP's own
feasibility tolerance), at the breakpoints and between them. So a model built
from these rows is the model the user declared, and a certificate on it is a
certificate on the user's problem. That is what separates this construct from
approximating a *nonlinear* term by a PWL surrogate, which is not a relaxation
and needs its own certificate story -- that is :func:`discopt.modeling.nonlinear_to_pwl`
(:mod:`discopt.modeling._pwl_transform`), whose ``"outer"`` mode builds a rigorous
outer approximation and whose ``"approximate"`` mode never claims a bound.

The encodings
-------------
All are standard, textbook-exact formulations of a univariate continuous PWL
function; see :cite:t:`Vielma2010` for the unifying treatment and
:cite:t:`Vielma2011` for the logarithmic one.

``"sos2"`` (alias ``"lambda"``)
    Convex-combination weights ``lambda`` with a declared SOS2 set
    (:meth:`Model.sos2`). The SOS2 row is lowered at solve time by the solver's
    existing SOS reformulation (one binary per breakpoint and O(n^2) pairwise
    exclusion rows), so it is the weakest and slowest route; it exists so the
    SOS2 relation itself is reachable. The ``.nl``/LP/MPS/GAMS writers refuse an
    SOS relation (no SOS section is emitted), so a model meant for export should
    use one of the other methods, whose rows are plain linear constraints.
``"log"`` (aliases ``"logarithmic"``, ``"ebd"``)
    The same weights, with the SOS2 adjacency enforced by ``ceil(log2(n - 1))``
    binaries through the Gray-code embedding in
    :mod:`discopt._relax.embedding` -- the construction AMP's ``convhull_ebd``
    option refers to. Fewest binaries.
``"disaggregated"`` (alias ``"dcc"``)
    One binary and one pair of weights per segment (disaggregated convex
    combination). Locally ideal, hence sharp: the LP relaxation's projection
    onto ``(x, y)`` is the convex hull of the graph.
``"incremental"`` (aliases ``"inc"``, ``"delta"``)
    Fill-the-segments ``delta`` variables with ``n - 2`` ordering binaries
    (Markowitz--Manne). Also locally ideal.

A two-breakpoint table is a single segment; every method then lowers to the two
linear equalities of that segment and adds no binary.

Conditioning
------------
Every convex-combination row is written *centred* on the first breakpoint,
``x - b_0 == sum_j (b_j - b_0) lambda_j`` (and ``y - c == sum_j (v_j - c)
lambda_j`` with ``c`` a central sample, see ``_value_reference``), never as
``x == sum_j b_j lambda_j``. The two are equal on the
exact feasible set, but a MILP solver satisfies ``sum(lambda) = 1`` and
``z in {0, 1}`` only to a tolerance ``eps``; the uncentred row then lets ``x``
drift by ``|b_0| * eps`` -- 0.5 at ``b_0 = 1e6`` with ``eps = 5e-7`` -- while
every row still reads as satisfied to a *relative* tolerance. That was a certified
wrong optimum and a certified "infeasible" (#1494). Centred, the drift is
``(b_{n-1} - b_0) * eps``, the table's own resolution scale. ``"incremental"``
has always been centred.

Domain agreement
----------------
Every formulation forces ``b_0 <= x <= b_{n-1}``. A declared domain wider than
that span would therefore be *silently clamped*, which changes the user's model.
It is refused instead: the bounds of ``x`` (declared bounds for a variable, a
rigorous interval enclosure for a composite expression) must lie inside the
breakpoint span. Bounds are mutable after construction, so the check is recorded
on the model and repeated by :meth:`Model.validate` before every solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type hints only
    from discopt.modeling.core import Expression, Model, Variable

__all__ = [
    "PIECEWISE_METHODS",
    "PiecewiseDomainError",
    "PiecewiseLinear",
    "normalize_piecewise_method",
]

#: Canonical method names accepted by :meth:`Model.piecewise`.
PIECEWISE_METHODS: tuple[str, ...] = ("sos2", "log", "disaggregated", "incremental")

_METHOD_ALIASES: dict[str, str] = {
    "sos2": "sos2",
    "lambda": "sos2",
    "log": "log",
    "logarithmic": "log",
    "ebd": "log",
    "disaggregated": "disaggregated",
    "dcc": "disaggregated",
    "incremental": "incremental",
    "inc": "incremental",
    "delta": "incremental",
}

#: Attribute under which a Model records the domain each ``piecewise`` call
#: relies on, re-checked by ``Model.validate``. A list of
#: ``(input_expr, span_lo, span_hi, label)`` tuples.
DOMAINS_ATTR = "_piecewise_domains"

#: Slack for a *composite* input's interval enclosure only. Interval arithmetic
#: rounds outward, so ``2*s`` with ``s in [0, 1]`` encloses as
#: ``[-1e-16, 2 + 4e-16]``: refusing that against a span ``[0, 2]`` would reject a
#: domain that agrees exactly. 1e-12 relative is ~4500 ulp -- far above rounding,
#: far below any feasibility tolerance, so it cannot hide a real clamp. A plain
#: variable (or element of one) is compared exactly against its declared bounds.
_ENCLOSURE_RTOL = 1e-12


class PiecewiseDomainError(ValueError):
    """The input's domain is not contained in the breakpoint span.

    Raised by :meth:`Model.piecewise` at declaration and by
    :meth:`Model.validate` if a bound is widened afterwards. The formulation would
    otherwise restrict the input to the span -- a silent clamp of the user's model.
    """


def normalize_piecewise_method(method: str) -> str:
    """Map a method name or alias to its canonical name, or raise ``ValueError``."""
    if not isinstance(method, str):
        raise TypeError(f"piecewise method must be a string, got {type(method).__name__}")
    try:
        return _METHOD_ALIASES[method.lower()]
    except KeyError as err:
        raise ValueError(
            f"Unknown piecewise method {method!r}. Choose from "
            f"{', '.join(repr(m) for m in PIECEWISE_METHODS)} "
            "(aliases: 'lambda', 'logarithmic', 'ebd', 'dcc', 'inc', 'delta')."
        ) from err


@dataclass(frozen=True)
class PiecewiseLinear:
    """A validated univariate piecewise-linear table.

    Attributes
    ----------
    breakpoints : tuple of float
        Strictly increasing, finite, at least two.
    values : tuple of float
        Finite, one per breakpoint.
    """

    breakpoints: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def from_table(
        cls,
        breakpoints: Any,
        values: Union[Any, Callable[[float], float]],
    ) -> "PiecewiseLinear":
        """Validate a table. *values* may be an array or a callable of one float.

        A callable is evaluated at each breakpoint; the function declared is then
        the interpolant of those samples, not the callable itself.
        """
        b = np.asarray(breakpoints, dtype=np.float64)
        if b.ndim != 1:
            raise ValueError(
                f"piecewise breakpoints must be one-dimensional, got shape {b.shape}. "
                "Multivariate piecewise-linear functions are not supported."
            )
        if b.size < 2:
            raise ValueError(f"piecewise needs at least 2 breakpoints, got {b.size}.")
        if not np.all(np.isfinite(b)):
            raise ValueError("piecewise breakpoints must all be finite.")
        steps = np.diff(b)
        if np.any(steps <= 0.0):
            k = int(np.argmax(steps <= 0.0))
            raise ValueError(
                "piecewise breakpoints must be strictly increasing; "
                f"breakpoints[{k}] = {b[k]!r} is followed by {b[k + 1]!r}. "
                "Repeated breakpoints (jump discontinuities) are not supported: "
                "the convex-combination rows would admit every value on the "
                "vertical segment at the jump. Model a discontinuous function "
                "with Model.either_or instead."
            )

        if callable(values):
            v = np.array([float(values(float(bk))) for bk in b], dtype=np.float64)
        else:
            v = np.asarray(values, dtype=np.float64)
        if v.shape != b.shape:
            raise ValueError(
                f"piecewise values have shape {v.shape} but breakpoints have shape "
                f"{b.shape}; supply one value per breakpoint."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("piecewise values must all be finite.")
        return cls(tuple(float(t) for t in b), tuple(float(t) for t in v))

    @property
    def n(self) -> int:
        """Number of breakpoints."""
        return len(self.breakpoints)

    @property
    def span(self) -> tuple[float, float]:
        """``(b_0, b_{n-1})``, the domain of the function."""
        return self.breakpoints[0], self.breakpoints[-1]

    def __call__(self, x):
        """Evaluate the interpolant. Raises outside the span rather than extrapolating."""
        xa = np.asarray(x, dtype=np.float64)
        lo, hi = self.span
        if np.any(xa < lo) or np.any(xa > hi):
            raise ValueError(f"piecewise function evaluated outside its domain [{lo}, {hi}].")
        out = np.interp(xa, self.breakpoints, self.values)
        return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Domain bookkeeping
# ---------------------------------------------------------------------------


def _input_enclosure(expr: "Expression", model: "Model") -> tuple[float, float, bool]:
    """Return ``(lo, hi, exact)`` for a scalar input expression.

    ``exact`` is True for a variable or an element of one: its declared bounds are
    the domain itself. Anything else is enclosed by interval arithmetic, which is
    rigorous (never narrower than the true range) but may be wider.
    """
    from discopt.modeling.core import IndexExpression, Variable

    if isinstance(expr, Variable):
        return float(np.min(expr.lb)), float(np.max(expr.ub)), True
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        v = expr.base
        return float(np.min(v.lb[expr.index])), float(np.max(v.ub[expr.index])), True

    from discopt._relax.convexity.interval_eval import evaluate_interval

    iv = evaluate_interval(expr, model)
    return float(np.min(iv.lo)), float(np.max(iv.hi)), False


def _domain_violation(expr: "Expression", model: "Model", lo: float, hi: float) -> Optional[str]:
    """Describe how the input's domain escapes ``[lo, hi]``, or ``None`` if it agrees."""
    e_lo, e_hi, exact = _input_enclosure(expr, model)
    slack = 0.0 if exact else _ENCLOSURE_RTOL * max(1.0, abs(lo), abs(hi))
    problems = []
    if not (e_lo >= lo - slack):
        problems.append(f"lower bound {e_lo:.12g} < first breakpoint {lo:.12g}")
    if not (e_hi <= hi + slack):
        problems.append(f"upper bound {e_hi:.12g} > last breakpoint {hi:.12g}")
    if not problems:
        return None
    kind = "declared bounds" if exact else "interval enclosure"
    return f"{kind} [{e_lo:.12g}, {e_hi:.12g}]: " + "; ".join(problems)


def _domain_error(label: str, detail: str) -> PiecewiseDomainError:
    return PiecewiseDomainError(
        f"piecewise {label}: the input's domain is not inside the breakpoint span "
        f"({detail}). The formulation would restrict the input to the span, "
        "silently changing the model. Tighten the input's bounds to lie inside "
        "the breakpoints, or extend the table to cover the whole domain. (For a "
        "composite input whose interval enclosure is looser than its true range, "
        "introduce a bounded auxiliary variable equal to it and pass that.)"
    )


def check_domains(model: "Model") -> None:
    """Re-verify every recorded ``piecewise`` domain; raise if one was widened.

    Called from :meth:`Model.validate`, i.e. once per solve. Cost is one bound
    lookup (or interval walk) per piecewise input, not a walk of the model.
    """
    for expr, lo, hi, label in getattr(model, DOMAINS_ATTR, ()):
        detail = _domain_violation(expr, model, lo, hi)
        if detail is not None:
            raise _domain_error(label, detail + "; a bound was widened after declaration")


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def _lin(coeffs, terms, constant: float = 0.0) -> "Expression":
    """``constant + sum(c * t)`` as a plain expression, skipping zero coefficients."""
    from discopt.modeling.core import _wrap

    expr: Optional[Expression] = None
    for c, t in zip(coeffs, terms):
        c = float(c)
        if c == 0.0:
            continue
        term = t if c == 1.0 else c * t
        expr = term if expr is None else expr + term
    if expr is None:
        return _wrap(float(constant))
    return expr + float(constant) if constant != 0.0 else expr


#: Smallest nonzero coefficient a centred value row may carry. The same floor
#: ``_pwl_transform._COEF_FLOOR`` keeps band coefficients above: an entry at or
#: below HiGHS's ``small_matrix_value`` (1e-12 at the most permissive) is dropped
#: on the way in, and the verified HiGHS route then refuses the whole model.
_TINY_COEF = 1e-11


def _value_reference(values) -> float:
    """The constant ``c`` a value row ``y == c + sum_j (v_j - c) w_j`` is centred on.

    Any ``c`` gives an exactly equivalent row (the weights sum to 1); ``c`` only
    decides conditioning. Centring on ``v_0`` alone is wrong when another sample
    equals ``v_0`` up to rounding -- ``cos`` sampled symmetrically gives
    ``v_j - v_0 ~ 1e-16``, a coefficient no LP backend keeps. So ``c`` is the
    sample minimising ``max_j |v_j - c|`` among those leaving every difference
    exactly zero or at least :data:`_TINY_COEF`. If no sample qualifies (every
    sample has a distinct near-duplicate) the row is left uncentred, ``c = 0``:
    that is the pre-#1494 row, exact, and no worse conditioned than before.
    """
    v = np.asarray(values, dtype=np.float64)
    spread = np.max(np.abs(v[:, None] - v[None, :]), axis=1)
    for k in np.argsort(spread, kind="stable"):
        d = v - v[k]
        if np.all((d == 0.0) | (np.abs(d) >= _TINY_COEF)):
            return float(v[k])
    return 0.0


def _lower_one(
    model: "Model",
    x: "Expression",
    y: "Expression",
    table: PiecewiseLinear,
    method: str,
    prefix: str,
) -> None:
    """Emit the rows tying scalar ``y`` to ``f(x)`` under *method*."""
    b = np.asarray(table.breakpoints)
    v = np.asarray(table.values)
    n = table.n
    add = model.subject_to

    if n == 2:
        # One segment: y is the affine interpolant and x is confined to [b0, b1].
        # Every encoding reduces to this; there is no discrete choice to make.
        t = model.continuous(f"{prefix}_t", lb=0.0, ub=1.0)
        add(x == _lin([b[1] - b[0]], [t], b[0]), name=f"{prefix}_x")
        add(y == _lin([v[1] - v[0]], [t], v[0]), name=f"{prefix}_y")
        return

    if method in ("sos2", "log"):
        lam = model.continuous(f"{prefix}_lam", shape=(n,), lb=0.0, ub=1.0)
        lams = [lam[j] for j in range(n)]
        add(_lin(np.ones(n), lams) == 1.0, name=f"{prefix}_convex")
        # Centred: see the ``Conditioning`` note in the module docstring (#1494).
        # Equivalent to ``x == sum(b_j lam_j)`` given the convexity row, but a
        # tolerance-level residual in ``sum(lam) = 1`` now moves x by
        # (span * residual), not (|b_0| * residual).
        cv = _value_reference(v)
        add(x == _lin(b - b[0], lams, b[0]), name=f"{prefix}_x")
        add(y == _lin(v - cv, lams, cv), name=f"{prefix}_y")
        if method == "sos2":
            model.sos2(lams, name=f"{prefix}_sos2")
            return
        from discopt._relax.embedding import build_embedding_map

        emb = build_embedding_map(n, encoding="gray")
        u = model.binary(f"{prefix}_u", shape=(emb.bit_count,))
        for bit in range(emb.bit_count):
            # A breakpoint whose every adjacent segment has this code bit = 1 may
            # carry weight only when u = 1, and symmetrically for bit = 0. With a
            # Gray code (adjacent segments differ in one bit) this leaves, for
            # every u in {0,1}^k, weight on at most one adjacent pair.
            pos = emb.positive_sets[bit]
            neg = emb.negative_sets[bit]
            if pos:
                add(
                    _lin(np.ones(len(pos)), [lams[j] for j in pos]) <= u[bit],
                    name=f"{prefix}_bit{bit}_on",
                )
            if neg:
                add(
                    _lin(np.ones(len(neg)), [lams[j] for j in neg]) <= 1 - u[bit],
                    name=f"{prefix}_bit{bit}_off",
                )
        return

    if method == "disaggregated":
        s = n - 1
        lo_w = model.continuous(f"{prefix}_wl", shape=(s,), lb=0.0, ub=1.0)
        hi_w = model.continuous(f"{prefix}_wr", shape=(s,), lb=0.0, ub=1.0)
        z = model.binary(f"{prefix}_z", shape=(s,))
        zs = [z[i] for i in range(s)]
        for i in range(s):
            add(lo_w[i] + hi_w[i] == zs[i], name=f"{prefix}_seg{i}")
        add(_lin(np.ones(s), zs) == 1.0, name=f"{prefix}_choose")
        w_terms = [lo_w[i] for i in range(s)] + [hi_w[i] for i in range(s)]
        # Centred for the same reason as the lambda rows (#1494): with
        # sum(wl + wr) = sum(z) = 1 this is exactly the uncentred row.
        cv = _value_reference(v)
        bx = np.concatenate([b[:-1], b[1:]]) - b[0]
        vy = np.concatenate([v[:-1], v[1:]]) - cv
        add(x == _lin(bx, w_terms, b[0]), name=f"{prefix}_x")
        add(y == _lin(vy, w_terms, cv), name=f"{prefix}_y")
        return

    if method == "incremental":
        s = n - 1
        d = model.continuous(f"{prefix}_delta", shape=(s,), lb=0.0, ub=1.0)
        ds = [d[i] for i in range(s)]
        add(x == _lin(np.diff(b), ds, b[0]), name=f"{prefix}_x")
        add(y == _lin(np.diff(v), ds, v[0]), name=f"{prefix}_y")
        z = model.binary(f"{prefix}_z", shape=(s - 1,))
        for i in range(s - 1):
            # delta_{i+1} <= z_i <= delta_i: segment i+1 starts filling only once
            # segment i is full.
            add(ds[i + 1] <= z[i], name=f"{prefix}_fill{i}_a")
            add(z[i] <= ds[i], name=f"{prefix}_fill{i}_b")
        return

    raise AssertionError(f"unnormalized piecewise method {method!r}")  # pragma: no cover


def build_piecewise(
    model: "Model",
    x: "Expression",
    table: PiecewiseLinear,
    method: str,
    name: Optional[str],
) -> "Variable":
    """Implementation of :meth:`Model.piecewise`; see there for the contract."""
    from discopt.modeling.core import Expression, _find_owning_model, _known_shape

    if not isinstance(x, Expression):
        raise TypeError(
            "piecewise() input must be a model expression (a variable, an element "
            f"of one, or an expression over them), got {type(x).__name__}."
        )
    owner = _find_owning_model(x)
    if owner is None:
        raise ValueError(
            "piecewise() input contains no variable; a constant input needs no "
            "piecewise function -- evaluate the table directly."
        )
    if owner is not model:
        raise ValueError("piecewise() input belongs to a different Model.")
    shape = _known_shape(x)
    if shape is None:
        raise ValueError(
            "piecewise() could not determine the shape of its input; pass a "
            "variable, an element of one, or a scalar expression."
        )

    model._aux_counter += 1
    tag = f"_pwl{model._aux_counter}"
    prefix = f"{tag}_{name}" if name else tag

    lo, hi = table.span
    elements = list(np.ndindex(*shape)) if shape else [()]
    inputs = [x[idx] if shape else x for idx in elements]

    # Domain agreement is checked for every element *before* anything is added,
    # so a refused call leaves the model untouched.
    problems = []
    for idx, xe in zip(elements, inputs):
        detail = _domain_violation(xe, model, lo, hi)
        if detail is not None:
            where = f"element {idx}" if shape else "input"
            problems.append(f"{where}: {detail}")
    if problems:
        label = repr(name) if name else "function"
        raise _domain_error(label, "; ".join(problems))

    v_lo, v_hi = float(min(table.values)), float(max(table.values))
    y = model.continuous(name or tag, shape=shape, lb=v_lo, ub=v_hi)

    domains = getattr(model, DOMAINS_ATTR, None)
    if domains is None:
        domains = []
        setattr(model, DOMAINS_ATTR, domains)
    for idx, xe in zip(elements, inputs):
        suffix = "_" + "_".join(str(i) for i in idx) if shape else ""
        ye = y[idx] if shape else y
        _lower_one(model, xe, ye, table, method, prefix + suffix)
        label = f"{name or tag}{list(idx) if shape else ''}"
        domains.append((xe, lo, hi, label))
    return y
