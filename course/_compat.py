"""course._compat — convenience shim that maps the course's pedagogical API to
real ``discopt``.

The course lessons were written using a slightly-friendlier façade:

    m.add_variable(lb=..., ub=..., name=...)
    m.add_variables(n, lb=..., ub=..., vtype="binary", name=...)
    m.add_constraint(expr)
    m.add_constraints([expr1, expr2, ...])
    m.is_convex()
    m.solve(mode="local"|"global", verbose=..., x0=..., ...)

The real ``discopt.modeling.core.Model`` exposes:

    m.continuous(name, shape=..., lb=..., ub=...)
    m.binary(name, shape=...)
    m.integer(name, shape=..., lb=..., ub=...)
    m.subject_to(constraint_or_list)
    m.minimize(expr) / m.maximize(expr)
    m.solve(time_limit=..., gap_tolerance=..., branching_policy=..., ...)

This module monkey-patches the missing convenience methods onto ``Model`` and
provides a wrapper for ``solve`` that translates the lesson-friendly kwargs to
real ones (and warns about kwargs that have no real-API counterpart, rather
than silently accepting them).

Usage at the top of any lesson notebook::

    from course._compat import *   # noqa: F401,F403

This must be importable from the notebook's working directory; either run the
notebook from the repo root, or ``sys.path.insert(0, "<repo-root>")`` first.

The shim is **for the course only.** Production code should target the real
``discopt`` API directly.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import discopt as do
from discopt import Model
from discopt.modeling.core import Model as _RealModel

__all__ = ["do", "Model", "np"]


# ── name autogen ─────────────────────────────────────────────────────────────


def _autoname(self: _RealModel, kind: str) -> str:
    counter = getattr(self, "_compat_counter", 0)
    self._compat_counter = counter + 1
    return f"{kind}_{counter}"


# ── add_variable / add_variables ─────────────────────────────────────────────


def add_variable(self: _RealModel, lb: float = -1e15, ub: float = 1e15, name=None):
    """Course shim: scalar continuous variable. Routes to .continuous()."""
    name = name or _autoname(self, "x")
    return self.continuous(name, shape=(), lb=float(lb), ub=float(ub))


def add_variables(
    self: _RealModel,
    n,
    lb: Any = 0.0,
    ub: Any = 1e15,
    vtype: str = "continuous",
    name=None,
):
    """Course shim: vector/array variables.

    ``n`` may be ``int`` or ``tuple``. ``vtype`` ∈ {continuous, binary, integer}.
    """
    if isinstance(n, int):
        shape = (n,)
    else:
        shape = tuple(n)
    name = name or _autoname(self, vtype[0])
    if vtype == "binary":
        return self.binary(name, shape=shape)
    if vtype == "integer":
        return self.integer(name, shape=shape, lb=lb, ub=ub)
    if vtype == "continuous":
        return self.continuous(name, shape=shape, lb=lb, ub=ub)
    raise ValueError(f"unknown vtype: {vtype!r}")


# ── add_constraint / add_constraints ─────────────────────────────────────────


def add_constraint(self: _RealModel, constraint, name=None):
    """Course shim: route to subject_to() with a single constraint."""
    return self.subject_to(constraint, name=name)


def add_constraints(self: _RealModel, constraints, name=None):
    """Course shim: route to subject_to() with a list of constraints."""
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]
    return self.subject_to(list(constraints), name=name)


# ── is_convex stub ───────────────────────────────────────────────────────────


def is_convex(self: _RealModel) -> bool | None:
    """Course shim: convexity introspection.

    The real solver runs convexity classification internally during ``.solve``;
    a public introspection hook is not yet stable. This shim returns ``None``
    and warns once per model so lessons can demonstrate the *concept* without
    over-specifying the API.
    """
    if not getattr(self, "_compat_convex_warned", False):
        warnings.warn(
            "course._compat.is_convex() is a pedagogical stub; the real "
            "convexity classifier runs during solve. Returning None.",
            stacklevel=2,
        )
        self._compat_convex_warned = True
    return None


# ── solve wrapper ────────────────────────────────────────────────────────────


_real_solve = _RealModel.solve

# Lesson kwargs that map to real kwargs (or are deliberately swallowed).
_KW_PASSTHRU = {"time_limit", "gap_tolerance", "threads", "deterministic"}
_KW_KNOWN_LESSON = {
    "mode",                # "local" | "global" — informational
    "verbose",             # informational; real API is silent
    "x0",                  # initial point — maps to initial_solution
    "log_tree",            # informational
    "log_relaxation",      # informational
    "cuts",                # informational
    "branching",           # maps to branching_policy
    "symmetry_breaking",   # informational
    "feasibility_only",    # informational
    "return_basis",        # informational
    "method",              # "simplex"|"ipm" — maps to solver
    "relaxation",          # informational
}


def solve(self: _RealModel, **kwargs):
    """Course shim around Model.solve.

    Translates lesson-friendly kwargs to real ones and warns (rather than
    silently dropping) when a kwarg has no real-API counterpart.
    """
    real = {k: v for k, v in kwargs.items() if k in _KW_PASSTHRU}
    if "x0" in kwargs:
        real["initial_solution"] = kwargs["x0"]
    if "branching" in kwargs:
        real["branching_policy"] = kwargs["branching"]
    if "method" in kwargs:
        real["solver"] = kwargs["method"]
    unknown = set(kwargs) - _KW_PASSTHRU - _KW_KNOWN_LESSON - {"x0", "branching", "method"}
    if unknown:
        warnings.warn(
            f"course._compat.solve(): ignoring unrecognised kwargs {unknown}. "
            "These are course-pedagogical placeholders not in the real "
            "discopt.solve signature.",
            stacklevel=2,
        )
    return _real_solve(self, **real)


# ── attach to Model (idempotent) ─────────────────────────────────────────────


def _install():
    if getattr(_RealModel, "_compat_installed", False):
        return
    _RealModel.add_variable = add_variable
    _RealModel.add_variables = add_variables
    _RealModel.add_constraint = add_constraint
    _RealModel.add_constraints = add_constraints
    _RealModel.is_convex = is_convex
    _RealModel.solve = solve
    _RealModel._compat_installed = True


_install()
