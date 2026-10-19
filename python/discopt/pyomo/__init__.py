"""Optional Pyomo integration: use discopt as a Pyomo solver.

Install with ``pip install discopt[pyomo]``. Once Pyomo is present, the solver is
registered automatically (via the ``pyomo.solvers`` entry point, or on the first
``import discopt.pyomo`` / call to :func:`register`)::

    import pyomo.environ as pyo
    import discopt.pyomo  # registers 'discopt'

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.obj = pyo.Objective(expr=(m.x - 3) ** 2)
    res = pyo.SolverFactory("discopt").solve(m, tee=True)
    print(m.x.value)

That route lowers a GDP model with a Pyomo transformation before discopt sees it.
:func:`from_pyomo_gdp` (``discopt.pyomo.gdp``, #1775 G0.4) instead rebuilds the
Pyomo GDP model natively, disjunctions intact, so every ``gdp_method`` applies::

    from discopt.pyomo import from_pyomo_gdp

    result = from_pyomo_gdp(pyomo_model).solve(gdp_method="hull")

The bridge round-trips through a temporary AMPL ``.nl`` file in-process and maps
the solution back by column order. Duals/reduced costs are loaded best-effort into
``dual`` / ``rc`` Suffixes when the model declares them and discopt exposes them.
"""

from __future__ import annotations


def is_available() -> bool:
    """True if Pyomo is installed and the solver plugin can be used."""
    try:
        import pyomo  # noqa: F401

        return True
    except ImportError:
        return False


def register() -> None:
    """Idempotently register ``'discopt'`` with Pyomo's ``SolverFactory``.

    A no-op if Pyomo is not installed. Importing ``discopt.pyomo.solver`` applies
    the ``@SolverFactory.register`` decorator; this wrapper makes that explicit and
    safe to call repeatedly.
    """
    if not is_available():
        return
    from pyomo.opt import SolverFactory

    if "discopt" not in SolverFactory:
        from . import solver  # noqa: F401  (decorator registers on import)


# Best-effort auto-registration on import (the documented manual activation path).
register()


def __getattr__(name: str):
    # The Pyomo GDP converter (#1775 G0.4), imported on first use so that
    # ``import discopt.pyomo`` stays as light as the solver registration.
    if name in ("from_pyomo_gdp", "convert_pyomo_gdp"):
        from . import gdp

        return getattr(gdp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert_pyomo_gdp", "from_pyomo_gdp", "is_available", "register"]
