"""``relax_integrality``: the continuous relaxation of a model (#1620).

Registered as the ``"core.relax_integrality"`` transformation
(:mod:`discopt.transformations`), Pyomo's name for the same thing::

    import discopt.transformations as dt

    lp = dt.create_using("core.relax_integrality", m)   # m is untouched
    r = lp.solve()                                       # r.objective <= MILP optimum (min)

Every ``BINARY`` and ``INTEGER`` variable becomes ``CONTINUOUS`` over its own
declared box (a binary over ``[0, 1]``, an integer over its ``[lb, ub]``); rows,
objective and bounds are untouched. For a minimization the relaxation's optimum
is a valid lower bound on the original's -- it is a *relaxation*, so the
registry marks it ``exact=False``.

A model whose meaning depends on integrality beyond the variable types is
refused, loudly, rather than relaxed into something that is not a relaxation:

* a disjunction, indicator, SOS or logic relation (each is defined over binary
  indicators; lower it with the ``"gdp"`` transformation first, and the lowered
  rows then relax honestly), and
* a model whose lifted auxiliaries carry *implied* integrality
  (``_implied_integer_auxes``), which is cleared together with the types it
  rests on so no heuristic keeps treating a now-continuous column as integral.
"""

from __future__ import annotations

from discopt.modeling.core import Constraint, Model, VarType


def relax_integrality(model: Model) -> None:
    """Make every integer and binary variable of ``model`` continuous, in place.

    Raises
    ------
    ValueError
        If ``model`` carries a non-algebraic relation (disjunction, indicator,
        SOS, logic) whose semantics need integral indicators.
    """
    for con in model._constraints:
        if not isinstance(con, Constraint):
            name = getattr(con, "name", None)
            where = f" named {name!r}" if name else ""
            raise ValueError(
                f"relax_integrality: the model carries a {type(con).__name__}{where}, "
                "which is defined over binary indicators; relaxing the variable types "
                "would not relax it. Lower it first (transformation 'gdp'), then relax "
                "the lowered model."
            )
    changed = False
    for var in model._variables:
        if var.var_type in (VarType.BINARY, VarType.INTEGER):
            var.var_type = VarType.CONTINUOUS
            changed = True
    if not changed:
        return
    model._implied_integer_auxes = set()
    # Solve-time memoisation is keyed on the model's structure, which now differs
    # (convexity/classification, evaluators, FBBT structure); drop it the way
    # ``Model.__deepcopy__`` starts a copy cold.
    for key in [k for k in model.__dict__ if k.endswith("_cache")]:
        model.__dict__[key] = None
    # The fast-API Rust builder recorded each variable's type when it was created.
    if model._builder is not None:
        model._replay_builder()
