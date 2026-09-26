"""Named model transformations: ``apply_to`` / ``create_using`` (#1479).

discopt's reformulations are module functions the solver calls at fixed points
of :func:`~discopt.solver.solve_model` (the GDP lowering, the exact integer and
binary product linearizations, the MPEC complementarity lowerings). Each one
was reachable only through a whole ``Model.solve()`` selected by a keyword or a
``DISCOPT_*`` flag, so the only way to look at what a reformulation *does* was
a whole-solve A/B. This module gives each of them a name and a uniform entry
point, so a transformation can be run on its own, on a copy, and diffed::

    import discopt.transformations as dt

    dt.available()                                  # registered names
    hull = dt.create_using("gdp.hull", m)           # m is untouched
    dt.apply_to("mpec.sos1", m, pairs=m._complementarities)   # m is changed

    report = dt.check("integer.bilinear", m)        # copy + diff, no solve
    report.diff.added_variables

**What this module is not.** It adds no transformation and changes none: every
registered entry resolves, at call time, to the function it names on its module
(so a test's ``monkeypatch`` of that module attribute still takes effect). The
solver's call sites -- every GDP lowering including the AMP route's
``respect_disjunction_methods=False`` call, the binary/integer product passes,
the MPEC and bilevel complementarity lowerings -- dispatch through
:meth:`Transformation.apply`, which is the bare function call plus a check of
its return type. The solver still decides *when* each runs and whether to adopt
the result; the registry decides nothing. That dispatch was verified
bound-neutral per CLAUDE.md §5 (identical status, objective, bound and node
count against ``main`` on the #1479 panel).

**Two function contracts, one API.** The wrapped functions come in two shapes:

* *functional* -- ``f(model, **options) -> Model``: returns a **new** model (or
  ``model`` itself when there is nothing to do) and leaves the input alone. The
  new model shares the input's :class:`~discopt.modeling.core.Variable`
  objects, which still report ``var.model is input``.
* *in-place* -- ``f(model, **options) -> None``: appends rows and variables to
  ``model``.

``apply_to`` gives both an in-place meaning and ``create_using`` gives both a
copy meaning. For a functional transformation ``apply_to`` *adopts* the
returned model's state into the caller's object and re-points the new
auxiliaries' ``.model`` at it, so afterwards every variable of the model
reports that model as its owner -- which the raw functional result does not.
The adoption refuses (before touching anything) if the returned model is
referenced by anything it does not know how to re-point.

After a transformation that changed the model, two records that describe the
*pre-transformation* model are cleared: ``_source_nl_path`` (the solver would
otherwise hand POUNCE the original ``.nl`` for a model that no longer matches
it) and ``_last_solve_result``.
"""

from __future__ import annotations

import copy
import gc
import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Union

import numpy as np

from discopt.modeling.core import Model, Parameter, Variable

__all__ = [
    "ModelDiff",
    "Transformation",
    "TransformationFactory",
    "TransformationReport",
    "apply_to",
    "available",
    "check",
    "create_using",
    "diff_models",
    "get",
    "model_fingerprint",
    "register",
]

Style = Literal["functional", "in_place"]


@dataclass(frozen=True)
class Transformation:
    """One named transformation.

    Attributes
    ----------
    name : str
        Registry key, dotted by family (``"gdp.hull"``, ``"mpec.sos1"``).
    target : str or callable
        The implementing function, or ``"module:attr"`` resolved on first use
        (so importing this module imports none of the reformulation layers).
    style : {"functional", "in_place"}
        The implementing function's contract; see the module docstring.
    summary : str
        One line for :func:`available` listings.
    exact : bool
        ``True`` when the transformation preserves the feasible set and optimum
        (an exact reformulation, possibly with auxiliary variables); ``False``
        for a relaxation or regularization (Scholtes), whose optimum is only an
        approximation of the source model's.
    fixed_options : mapping
        Options this entry always passes (``"gdp.hull"`` is ``"gdp"`` with
        ``method="hull"``). Passing one of them again is refused.
    """

    name: str
    target: Union[str, Callable[..., Any]]
    style: Style
    summary: str
    exact: bool = True
    fixed_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def function(self) -> Callable[..., Any]:
        """The implementing function (the object the solver itself calls)."""
        if callable(self.target):
            return self.target
        module_name, _, attr = self.target.partition(":")
        fn: Callable[..., Any] = getattr(importlib.import_module(module_name), attr)
        return fn

    def _options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        clash = sorted(set(options) & set(self.fixed_options))
        if clash:
            raise TypeError(
                f"transformation {self.name!r} fixes {', '.join(clash)}; "
                f"use the base transformation to choose it"
            )
        return {**self.fixed_options, **options}

    def apply(self, model: Model, **options: Any) -> Model:
        """Run the function with its own contract; return the model it produced.

        Functional: the function's return value (possibly ``model`` itself).
        In-place: ``model``, after the function mutated it. This is exactly the
        call a solver call site makes -- no adoption, no invalidation.
        """
        out = self.function(model, **self._options(options))
        if self.style == "in_place":
            if out is not None:
                raise TypeError(
                    f"transformation {self.name!r} is registered in-place but its "
                    f"function returned {type(out).__name__}; fix the registration"
                )
            return model
        if not isinstance(out, Model):
            raise TypeError(
                f"transformation {self.name!r} is registered functional but its "
                f"function returned {type(out).__name__}, not a Model"
            )
        return out

    def apply_to(self, model: Model, **options: Any) -> Model:
        """Transform ``model`` in place and return it."""
        before = model_fingerprint(model)
        if self.style == "in_place":
            self.apply(model, **options)
        else:
            result = self.apply(model, **options)
            if result is not model:
                _adopt(model, result)
        if model_fingerprint(model) != before:
            _invalidate_source_records(model)
        return model

    def create_using(self, model: Model, **options: Any) -> Model:
        """Transform a copy of ``model`` and return the copy; ``model`` is untouched.

        The model and the options are deep-copied *together*, so an option that
        names parts of the model (``pairs=`` relations, a ``t=`` parameter) is
        mapped onto the copy's objects rather than left pointing at the original.
        """
        memo: dict[int, Any] = {}
        clone = copy.deepcopy(model, memo)
        cloned_options = copy.deepcopy(dict(options), memo)
        return self.apply_to(clone, **cloned_options)


# ── Adoption: give a functional result an in-place meaning ────────────────


def _adopt(model: Model, result: Model) -> None:
    """Make ``model`` become ``result``: take its state, re-point its owners.

    Every reference to ``result`` must be one we can re-point: a variable's or
    parameter's ``.model`` owner field. Anything else is refused *before*
    ``model`` is modified, because a model that silently kept a second owner
    would fail ``Model._resolve_variable``-style identity checks later, far from
    the cause.
    """
    holders: list[Any] = [*result._variables, *result._parameters]
    owned = {id(o): o for o in holders if o.model is result}
    owned_dicts = {id(o.__dict__) for o in owned.values()}

    # A module-level helper, not a closure: a closure over ``result`` would add
    # a ``cell`` referrer of its own.
    unknown = _unknown_referrers(result, owned, owned_dicts)
    if unknown:
        # A pass's nested helpers close over the model they build; once the pass
        # returns, those closures are unreachable cycles that only the cyclic
        # collector frees. Collect once and look again, so garbage is not
        # mistaken for a live second owner.
        gc.collect()
        unknown = _unknown_referrers(result, owned, owned_dicts)
    if unknown:
        raise TypeError(
            "apply_to cannot adopt the transformed model: it is referenced by "
            f"{sorted(set(unknown))} besides its own variables/parameters. Use "
            "create_using/apply, or teach discopt.transformations._adopt to "
            "re-point that reference."
        )
    for obj in owned.values():
        obj.model = model
    model.__dict__.clear()
    model.__dict__.update(result.__dict__)


def _unknown_referrers(result: Model, owned: dict, owned_dicts: set) -> list[str]:
    return [
        type(ref).__name__
        for ref in gc.get_referrers(result)
        if not isinstance(ref, types.FrameType)
        and id(ref) not in owned
        and id(ref) not in owned_dicts
    ]


def _invalidate_source_records(model: Model) -> None:
    model.__dict__.pop("_source_nl_path", None)
    if "_last_solve_result" in model.__dict__:
        model._last_solve_result = None


# ── Structural fingerprint and diff ───────────────────────────────────────


def _var_key(v: Variable) -> tuple:
    return (
        v.name,
        v.var_type.value,
        tuple(v.shape),
        np.asarray(v.lb, dtype=np.float64).tobytes(),
        np.asarray(v.ub, dtype=np.float64).tobytes(),
    )


def _param_key(p: Parameter) -> tuple:
    return (p.name, np.asarray(p.value, dtype=np.float64).tobytes())


def _con_key(c: Any) -> tuple:
    return (type(c).__name__, getattr(c, "name", None), repr(c))


def _block_key(block: tuple) -> tuple:
    A, x, sense, b, name = block
    return (
        x.name,
        sense,
        name,
        A.shape,
        A.indptr.tobytes(),
        A.indices.tobytes(),
        A.data.tobytes(),
        np.asarray(b).tobytes(),
    )


def model_fingerprint(model: Model) -> tuple:
    """A hashable structural snapshot of ``model``.

    Covers the variables (name, type, shape, bounds), parameters (name, value),
    every constraint (type, name, full expression text), the objective, the
    builder-resident linear rows and objective, and the complementarity
    records. Two models with equal fingerprints state the same problem in the
    same order; a fingerprint that changes across a call means the call
    mutated the model. Arrays are printed in full (no numpy summarization), so
    a change deep inside a large constant is not hidden behind ``...``.
    """
    with np.printoptions(threshold=sys.maxsize):
        obj = model._objective
        lin_obj = model._builder_linear_objective
        quad_obj = model._builder_quadratic_objective
        return (
            model.name,
            tuple(_var_key(v) for v in model._variables),
            tuple(_param_key(p) for p in model._parameters),
            tuple(_con_key(c) for c in model._constraints),
            None if obj is None else (repr(obj.expression), obj.sense.value),
            tuple(_block_key(b) for b in model._builder_linear_blocks),
            None if lin_obj is None else repr(lin_obj),
            None if quad_obj is None else repr(quad_obj),
            tuple(repr(c) for c in model._complementarities),
            tuple(sorted(str(v) for v in model._lowered_complementarities.values())),
        )


@dataclass(frozen=True)
class ModelDiff:
    """What a transformation changed, by name and expression text.

    Constraints are compared as multisets of their printed form, so a
    constraint that moved position is not reported, and one that was replaced
    by a textually identical copy (as every rebuilding pass does) is not either.
    """

    added_variables: tuple[str, ...]
    removed_variables: tuple[str, ...]
    rebounded_variables: tuple[str, ...]
    added_constraints: tuple[str, ...]
    removed_constraints: tuple[str, ...]
    objective_changed: bool

    @property
    def unchanged(self) -> bool:
        return not (
            self.added_variables
            or self.removed_variables
            or self.rebounded_variables
            or self.added_constraints
            or self.removed_constraints
            or self.objective_changed
        )


def _multiset_minus(a: list[str], b: list[str]) -> tuple[str, ...]:
    from collections import Counter

    return tuple(sorted((Counter(a) - Counter(b)).elements()))


def _row_texts(model: Model) -> list[str]:
    rows = [repr(c) for c in model._constraints]
    rows += [repr(_block_key(b)) for b in model._builder_linear_blocks]
    return rows


def diff_models(before: Model, after: Model) -> ModelDiff:
    """Structural difference between two models (typically a model and its transform)."""
    with np.printoptions(threshold=sys.maxsize):
        bv = {v.name: _var_key(v) for v in before._variables}
        av = {v.name: _var_key(v) for v in after._variables}
        bc = _row_texts(before)
        ac = _row_texts(after)
        # Fingerprint slots 4, 6, 7: the expression objective and the two
        # builder-resident objectives.
        fb, fa = model_fingerprint(before), model_fingerprint(after)
        bo = (fb[4], fb[6], fb[7])
        ao = (fa[4], fa[6], fa[7])
    return ModelDiff(
        added_variables=tuple(n for n in av if n not in bv),
        removed_variables=tuple(n for n in bv if n not in av),
        rebounded_variables=tuple(n for n in av if n in bv and av[n] != bv[n]),
        added_constraints=_multiset_minus(ac, bc),
        removed_constraints=_multiset_minus(bc, ac),
        objective_changed=bo != ao,
    )


@dataclass(frozen=True)
class TransformationReport:
    """The outcome of :func:`check`."""

    name: str
    result: Model
    diff: ModelDiff


def check(name: str, model: Model, **options: Any) -> TransformationReport:
    """Apply one transformation to a copy of ``model`` and diff it, without a solve.

    Raises ``AssertionError`` if the input model changed -- the one property
    ``create_using`` must never violate. The returned report carries the
    transformed copy (for a feasibility or bound experiment) and the
    :class:`ModelDiff` against the input. This is the unit of work CLAUDE.md §5
    verifies a reformulation with: apply it to a copy and compare, rather than
    A/B a whole solve behind a flag.
    """
    before = model_fingerprint(model)
    result = get(name).create_using(model, **options)
    if model_fingerprint(model) != before:
        raise AssertionError(f"create_using({name!r}) modified its input model")
    if result is model:
        raise AssertionError(f"create_using({name!r}) returned its input model")
    return TransformationReport(name=name, result=result, diff=diff_models(model, result))


# ── Registry ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Transformation] = {}


def register(
    name: str,
    target: Union[str, Callable[..., Any]],
    *,
    style: Style,
    summary: str,
    exact: bool = True,
    fixed_options: Optional[Mapping[str, Any]] = None,
) -> Transformation:
    """Register a transformation under ``name``; a duplicate name is refused."""
    if style not in ("functional", "in_place"):
        raise ValueError(f"style must be 'functional' or 'in_place', got {style!r}")
    if name in _REGISTRY:
        raise ValueError(f"transformation {name!r} is already registered")
    t = Transformation(
        name=name,
        target=target,
        style=style,
        summary=summary,
        exact=exact,
        fixed_options=dict(fixed_options or {}),
    )
    _REGISTRY[name] = t
    return t


def get(name: str) -> Transformation:
    """The registered transformation ``name``, or a ``KeyError`` listing the names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no transformation named {name!r}; registered: {', '.join(available())}"
        ) from None


#: Pyomo spelling, for readers coming from ``TransformationFactory("gdp.bigm")``.
TransformationFactory = get


def available() -> list[str]:
    """Sorted names of every registered transformation."""
    return sorted(_REGISTRY)


def apply_to(name: str, model: Model, **options: Any) -> Model:
    """``get(name).apply_to(model, **options)``: transform ``model`` in place."""
    return get(name).apply_to(model, **options)


def create_using(name: str, model: Model, **options: Any) -> Model:
    """``get(name).create_using(model, **options)``: transform a copy."""
    return get(name).create_using(model, **options)


# ── The built-in registrations: the existing functions, unchanged ─────────

_GDP = "discopt._relax.gdp_reformulate:reformulate_gdp"

register(
    "gdp",
    _GDP,
    style="functional",
    summary="Lower indicator/disjunctive/SOS/logical constraints (method=, default big-M)",
)
for _alias, _method in (
    ("gdp.bigm", "big-m"),
    ("gdp.hull", "hull"),
    ("gdp.mbigm", "mbigm"),
    ("gdp.auto", "auto"),
    ("gdp.simplex", "simplex"),
):
    register(
        _alias,
        _GDP,
        style="functional",
        summary=f"GDP lowering with method={_method!r}",
        fixed_options={"method": _method},
    )
register(
    "integer.bilinear",
    "discopt._relax.integer_product_reform:reformulate_integer_bilinear",
    style="functional",
    summary="Exactly linearize integer-factor bilinear products",
)
register(
    "integer.multilinear",
    "discopt._relax.integer_product_reform:reformulate_integer_multilinear",
    style="functional",
    summary="Exactly linearize integer-factor bilinear and multilinear products (#707)",
)
register(
    "binary.multilinear",
    "discopt._relax.binary_multilinear_reform:reformulate_binary_multilinear",
    style="functional",
    summary="Fortet/Glover-linearize binary multilinear monomials",
)
register(
    "mpec.gdp",
    "discopt.mpec:reformulate_gdp",
    style="in_place",
    summary="Complementarity pairs= as disjunctions (f==0) or (g==0)",
)
register(
    "mpec.sos1",
    "discopt.mpec:reformulate_sos1",
    style="in_place",
    summary="Complementarity pairs= as SOS1 sets",
)
register(
    "mpec.scholtes",
    "discopt.mpec:reformulate_scholtes",
    style="in_place",
    exact=False,
    summary="Scholtes regularization f*g <= t of complementarity pairs= (not exact)",
)
del _alias, _method
