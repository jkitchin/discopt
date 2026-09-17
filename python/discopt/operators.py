"""Domain operator registration — a named composite the relaxer treats as one atom.

Component **A** of #1248 (relaxation extension API, from the discopt-calphad
plugin needs in #1249).

Why
---
A plugin knows structure that generic factorable relaxation throws away. Written
in primitives, ``x(1-x)(L0 + L1(2x-1)) + RT[x ln x + (1-x) ln(1-x)]`` — the
Redlich-Kister binary the CALPHAD plugin prices phases with — is relaxed *term by
term*: each piece is enveloped over its own box and the pieces are summed, which
loses every cancellation between them.

What a registration buys
------------------------
``register_function(name, lower)`` names that composite. The model still contains
the **lowering** — the ordinary primitive expression — so evaluation, `.nl`
export, the Rust core, presolve and every other consumer are untouched and need
no new opcode. What changes is that the *relaxation* layer recognises the named
atom and envelopes it whole.

Measured payoff, and a retraction (CLAUDE.md §11). This module first reported a
root gap of "4% to 324%" and solves of "131 to 7559 nodes" for the primitive
spelling. Those numbers were real but they measured something else: ``entropy``
had no ``_UNIVARIATE_FN`` entry, so every ``dm.xlogx`` term in the primitive arm
reached the engine's interval floor (#1277). With that envelope in place the
primitive arm needs 23-51 nodes on the same family, and naming the composite
still helps — 15-47 nodes, a strict reduction in every row at the same optimum —
but by 1.1x-1.5x, not by two orders of magnitude. The per-row table is in
``python/tests/test_1248_register_function.py``.

Nothing is taken on trust. The lowering is the definition, and everything the
relaxer needs is DERIVED from it:

* ``f`` and ``f'`` come from evaluating the lowering and its symbolic derivative
  (:func:`discopt.bilevel.symbolic_diff.diff`) on the same tape the solver uses,
  so the envelope is built from the function the solver actually evaluates;
* the per-box curvature verdict comes from **interval arithmetic on ``f''``** —
  ``f'' >= 0`` on the box proves convex, ``f'' <= 0`` proves concave, anything
  else abstains and the engine falls back to its interval floor. ``diff`` takes
  subgradients at kinks, so that ``f''`` is the true one only where no kink
  (``abs``/``sign`` argument, pole, non-integer power's base at 0) can lie in
  the box; a box that may contain one abstains too (#1293).

So a registered operator cannot carry an unsound envelope: there is no
user-supplied envelope to be unsound. #1248's acceptance asks that a deliberately
unsound ``relax`` be rejected at registration; deriving it instead answers that
question by construction. (A user-supplied envelope, with the sampling gate
#1248 sketches, is deliberately NOT part of this increment — see the module
docstring of the test file for what that would take.)

Where the tightness comes from
------------------------------
The engine emits a secant/tangent envelope for a univariate node whose curvature
is known on the node's box (``uniform_relax._emit_1d``). A composite like the
Redlich-Kister binary is convex near the ends of ``[0, 1]`` and concave in the
middle, so on the root box the verdict abstains — exactly as it does for
``sin``/``cos`` — and branch-and-bound gets its gain as soon as the boxes become
single-curvature. That is the same mechanism the built-in atoms use, applied to a
composite the plugin names.

Example
-------
>>> import discopt.modeling as dm
>>> rk = dm.register_function(                                   # doctest: +SKIP
...     "rk_binary",
...     lambda x: x * (1 - x) * (3.0 + 1.5 * (2 * x - 1))
...     + dm.xlogx(x)
...     + dm.xlogx(1 - x),
... )
>>> m = dm.Model()                                               # doctest: +SKIP
>>> x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)                  # doctest: +SKIP
>>> m.minimize(rk(x))            # the model holds the lowering  # doctest: +SKIP
>>> r = m.solve()                # the relaxer sees one atom     # doctest: +SKIP
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Expression

logger = logging.getLogger(__name__)

__all__ = [
    "register_function",
    "registered_names",
    "get_registered",
    "clear_registered",
    "RegisteredFunction",
]

#: Attribute an expression carries to say "I am the lowering of this atom".
#: Read by :mod:`discopt._relax.canonical_expr`, which emits a named ``call``
#: node for it instead of canonicalizing the body term by term.
ATOM_ATTR = "_discopt_atom"

_LOCK = threading.RLock()
_REGISTRY: "dict[str, RegisteredFunction]" = {}


@dataclass
class RegisteredFunction:
    """A named univariate composite, plus everything derived from its lowering.

    The derived pieces are built lazily on first use and cached: a registration
    that is never solved with costs nothing beyond the name.
    """

    name: str
    lower: Callable[["Expression"], "Expression"]
    description: str = ""

    #: How many times the relaxation engine has taken this atom's envelope, in
    #: this process. A registration that never fires is indistinguishable from
    #: no registration at all by node count alone, so the count is the direct
    #: evidence that the mechanism ran (CLAUDE.md §6). Diagnostic only — nothing
    #: reads it back.
    use_count: int = 0

    # Derived lazily (see :meth:`_derived`).
    _probe: object = None

    def __call__(self, arg: "Expression") -> "Expression":
        """Build the lowering of this atom on ``arg``, tagged with its name."""
        from discopt.modeling.core import Expression

        body = self.lower(arg)
        if not isinstance(body, Expression):
            raise TypeError(
                f"register_function({self.name!r}): lower() must return a discopt "
                f"Expression built from dm.* primitives, got {type(body).__name__}. "
                "An opaque numeric callable cannot be relaxed; see dm.custom for that."
            )
        # Tagging the RESULT rather than wrapping it in a new node type is what
        # keeps every existing consumer working: to the evaluator, the Rust core,
        # the exporters and presolve this is an ordinary expression.
        try:
            # The tag carries the registration OBJECT, not just the name: the
            # model keeps this body, so the envelope must be derived from this
            # definition and no other (see :func:`atom_of`).
            setattr(body, ATOM_ATTR, (self.name, arg, self))
        except AttributeError:  # pragma: no cover - slotted node types
            logger.debug(
                "registered atom %r could not be tagged onto a %s; it will be relaxed term by term",
                self.name,
                type(body).__name__,
            )
        return body

    # -- derived numerics -------------------------------------------------- #
    def _derived(self):
        """``(f, f_prime, curvature(lo, hi))``, all derived from the lowering.

        Built once per registration. ``f``/``f'`` are evaluated on a one-variable
        probe model through the same tape the solver uses, so the envelope is
        built from the function the solver evaluates rather than from a second
        implementation of it.
        """
        if self._probe is not None:
            return self._probe

        import discopt.modeling as dm
        from discopt._tape_nlp_evaluator import make_evaluator
        from discopt.bilevel.symbolic_diff import diff

        probe = dm.Model(f"_atom_probe_{self.name}")
        t = probe.continuous("t", lb=-1e6, ub=1e6)
        body = self.lower(t)
        probe.minimize(body)
        d1 = diff(body, t)
        d2 = diff(d1, t)

        kinks = _kink_sites(body)
        value_ev = make_evaluator(probe)

        def f(x: float) -> float:
            return float(value_ev.evaluate_objective(np.asarray([float(x)], dtype=float)))

        def fp(x: float) -> float:
            g = value_ev.evaluate_gradient(np.asarray([float(x)], dtype=float))
            return float(np.asarray(g).ravel()[0])

        def curvature(lo: float, hi: float) -> Optional[str]:
            """Rigorous per-box curvature: the SIGN of an interval enclosure of f''.

            Interval arithmetic, not sampling — a verdict here is a proof on the
            whole box, and an enclosure that straddles zero (or is not finite)
            abstains, which costs tightness and never soundness.
            """
            from discopt._relax.convexity.interval import Interval
            from discopt._relax.convexity.interval_eval import evaluate_interval

            box = {
                t: Interval(
                    np.asarray(float(lo), dtype=np.float64),
                    np.asarray(float(hi), dtype=np.float64),
                )
            }
            try:
                # ``diff`` takes subgradients at kinks (abs' = sign, sign' = 0), so
                # f'' is blind to the kink's Dirac mass: -|t| read "convex" (#1293).
                # The symbolic f'' is the true one only where every kink argument
                # stays strictly on one side of its kink over the whole box.
                for arg, positive_only in kinks:
                    a = evaluate_interval(arg, probe, box)
                    a_lo, a_hi = float(np.asarray(a.lo)), float(np.asarray(a.hi))
                    if not (a_lo > 0.0 or (a_hi < 0.0 and not positive_only)):
                        return None
                enc = evaluate_interval(d2, probe, box)
            except Exception as exc:  # noqa: BLE001 - abstaining is always sound
                logger.debug("curvature of atom %r abstained: %s", self.name, exc)
                return None
            enc_lo = float(np.asarray(enc.lo))
            enc_hi = float(np.asarray(enc.hi))
            if not (math.isfinite(enc_lo) and math.isfinite(enc_hi)):
                return None
            if enc_lo >= 0.0:
                return "convex"
            if enc_hi <= 0.0:
                return "concave"
            return None

        self._probe = (f, fp, curvature)
        return self._probe

    def note_use(self) -> None:
        """Record that the engine took this atom's envelope (see :attr:`use_count`)."""
        with _LOCK:
            self.use_count += 1

    def envelope_entry(self):
        """``(f, f', curvature, domain_ok)`` in ``uniform_relax._UNIVARIATE_FN`` form.

        ``domain_ok`` is always True: a box outside the lowering's natural domain
        produces a non-finite ``f''`` enclosure, so ``curvature`` abstains there
        and the engine falls back to the interval floor — the same outcome the
        domain guard produces for the built-ins, reached by the same evidence.
        """
        f, fp, curvature = self._derived()
        return (f, fp, curvature, lambda lo: True)

    def interval_expr(self, arg: "Expression") -> "Expression":
        """The lowering on ``arg`` — what an interval enclosure of the atom means."""
        return self.lower(arg)


def _kink_sites(body: "Expression") -> list[tuple["Expression", bool]]:
    """Arguments at which *body* is not twice differentiable, as ``(arg, positive_only)``.

    The curvature verdict is a proof only where each ``arg`` excludes its kink on
    the box: ``arg > 0``, or ``arg < 0`` too unless ``positive_only``. Covers
    ``abs``/``sign`` (kink at 0), division and a negative integer power (pole at
    0), and a non-integer or variable exponent (defined, and smooth, for a
    positive base only). A node type this walk does not know raises, so the
    verdict abstains rather than assuming it smooth.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    sites: list[tuple["Expression", bool]] = []
    seen: set[int] = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (Constant, Parameter, Variable)):
            continue
        if isinstance(node, UnaryOp):
            if node.op == "abs":
                sites.append((node.operand, False))
            stack.append(node.operand)
        elif isinstance(node, FunctionCall):
            if node.func_name in ("abs", "sign"):
                sites.append((node.args[0], False))
            stack.extend(node.args)
        elif isinstance(node, BinaryOp):
            if node.op == "/":
                sites.append((node.right, False))
            elif node.op == "**":
                exp = node.right
                if isinstance(exp, Constant) and exp.value.size == 1:
                    c = float(exp.value.reshape(()))
                    if c != math.floor(c):
                        sites.append((node.left, True))
                    elif c < 0:
                        sites.append((node.left, False))
                else:
                    sites.append((node.left, True))
            stack.extend((node.left, node.right))
        elif isinstance(node, MatMulExpression):
            stack.extend((node.left, node.right))
        elif isinstance(node, SumExpression):
            stack.append(node.operand)
        elif isinstance(node, SumOverExpression):
            stack.extend(node.terms)
        elif isinstance(node, IndexExpression):
            stack.append(node.base)
        else:
            raise NotImplementedError(
                f"curvature: no kink rule for node type {type(node).__name__}"
            )
    return sites


def register_function(
    name: str,
    lower: Callable[["Expression"], "Expression"],
    *,
    description: str = "",
    replace: bool = False,
) -> RegisteredFunction:
    """Register a named univariate composite for the relaxation layer.

    Parameters
    ----------
    name : str
        The atom's name. It must not collide with a built-in operator: the
        relaxation engine keys envelopes by name, and shadowing ``exp`` would
        silently re-define it for every model in the process.
    lower : callable
        ``lower(x) -> Expression``, the **definition**, built from ``dm.*``
        primitives. Everything the relaxer uses is derived from it.
    description : str, optional
        Human-readable note, carried for diagnostics.
    replace : bool, default False
        Allow re-registering a name. Models built against the previous
        definition keep their body and stop being enveloped as an atom (they are
        relaxed term by term); only expressions built after the replacement use
        the new registration.

    Returns
    -------
    RegisteredFunction
        Call it with an expression to build the (tagged) lowering.

    Raises
    ------
    ValueError
        On a name collision with a built-in operator or an existing
        registration (without ``replace=True``), or on an empty name.
    """
    from discopt._relax.uniform_relax import _UNIVARIATE_FN

    if not isinstance(name, str) or not name.strip():
        raise ValueError("register_function(): name must be a non-empty string")
    if not callable(lower):
        raise TypeError("register_function(): lower must be callable")
    if name in _UNIVARIATE_FN or name in _BUILTIN_RESERVED:
        raise ValueError(
            f"register_function({name!r}): that name is a built-in discopt operator. "
            "Registering it would silently re-define the operator's envelope for every "
            "model in this process. Choose another name."
        )
    with _LOCK:
        if name in _REGISTRY and not replace:
            raise ValueError(
                f"register_function({name!r}): already registered. Pass replace=True to "
                "redefine it — models built against the previous definition then lose "
                "the atom envelope and are relaxed term by term."
            )
        fn = RegisteredFunction(name=name, lower=lower, description=description)
        _REGISTRY[name] = fn
    return fn


#: Names the engine handles outside ``_UNIVARIATE_FN`` that must not be shadowed.
_BUILTIN_RESERVED = frozenset(
    {"abs", "sign", "entropy", "centropy", "sigmoid", "softplus", "prod", "norm", "tan"}
)


def registered_names() -> list[str]:
    """Sorted names of the registered atoms."""
    with _LOCK:
        return sorted(_REGISTRY)


def get_registered(name: str) -> Optional[RegisteredFunction]:
    """The registration for ``name``, or ``None``."""
    with _LOCK:
        return _REGISTRY.get(name)


def clear_registered(name: Optional[str] = None) -> None:
    """Drop one registration, or all of them. Intended for tests."""
    with _LOCK:
        if name is None:
            _REGISTRY.clear()
        else:
            _REGISTRY.pop(name, None)


def atom_of(expr: object) -> Optional[tuple]:
    """``(name, arg)`` if ``expr`` is a tagged registered lowering, else ``None``.

    Also returns ``None`` for a tag whose registration has since been dropped OR
    REPLACED. The model holds the body built by the registration that tagged it,
    while the relaxer derives the envelope from whatever the registry holds under
    the name now; after ``register_function(name, ..., replace=True)`` those are
    different functions, and the new envelope would cut the old body's true
    points — a false bound. A stale tag therefore degrades to term-by-term
    relaxation of the body the model actually carries, which is sound.
    """
    tag = getattr(expr, ATOM_ATTR, None)
    if tag is None:
        return None
    name = str(tag[0])
    with _LOCK:
        if _REGISTRY.get(name) is not tag[2]:
            return None
    return (name, tag[1])
