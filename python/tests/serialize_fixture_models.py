"""The fixed set of models the serialization golden fixtures are written from.

Plan: ``docs/dev/serialization-completion-plan.md`` §3.3 (WP-1). Every builder here
is run both by ``scripts/serialize_fixture_gen.py`` -- under a *historical* discopt
checked out at the last commit that wrote each schema version -- and by
``test_serialize_compat_matrix.py`` under the current code, so:

* only API that existed when schema 1.0 shipped (commit ``0a655f244``) is used,
  except where a case says otherwise; a builder that raises under an old version is
  recorded in the manifest as ``unbuildable_at``, never skipped silently;
* nothing here may depend on the current tree's helpers -- this file is imported by
  path into an old interpreter.

Each builder is zero-argument and returns ``(model, save_kwargs)``; ``save_kwargs``
is passed to ``dumps`` (only ``result`` is ever used).
"""

from __future__ import annotations

import os

import numpy as np

#: The corpus files the ``from_nl_*`` cases read. Resolved against this file, so the
#: generator reads the current tree's copy whichever discopt it imports; the
#: manifest records each file's SHA-256 so a changed input cannot pass unnoticed.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NL_FILES = {
    "from_nl_nvs01": os.path.join(DATA_DIR, "minlplib_nl", "nvs01.nl"),
    "from_nl_st_e13": os.path.join(DATA_DIR, "minlplib_nl", "st_e13.nl"),
    "from_nl_mpcc": os.path.join(DATA_DIR, "mpcc_complementarity.nl"),
}


def scalar_nlp():
    import discopt.modeling as dm

    m = dm.Model("scalar_nlp")
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(dm.exp(0.3 * x) + dm.log(x) * y + dm.sqrt(x + 1.0) - y**3)
    m.subject_to(x * y <= 3.0, name="xy")
    m.subject_to(x + y >= 1.0, name="cover")
    return m, {}


def array_bounds_inf():
    import discopt.modeling as dm

    m = dm.Model("array_bounds_inf")
    x = m.continuous(
        "x",
        shape=(3, 2),
        lb=np.array([[-np.inf, 0.0], [-1.0, -1e20], [0.0, 0.0]]),
        ub=np.array([[1.0, np.inf], [2.0, 1e20], [0.5, 3.0]]),
    )
    k = m.integer("k", shape=(2,), lb=0, ub=[5, 9])
    m.minimize(dm.sum(x**2) + k[0] - 2 * k[1])
    m.subject_to(dm.sum(x[0:2, 1]) + k[1] <= 7.0, name="mix")
    return m, {}


def mip_indicator_sos_disjunction():
    import discopt.modeling as dm

    m = dm.Model("mip_indicator_sos_disjunction")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    z = m.binary("z")
    m.minimize(dm.sum(x) - 2 * z)
    m.subject_to(x[0] + x[1] <= 8.0, name="cap")
    m.if_then(z, [x[0] >= 2.0], name="ind")
    m.sos1([x[0], x[1], x[2]], name="s1")
    m.either_or([[x[2] <= 1.0], [x[2] >= 5.0]], name="split")
    return m, {}


def shared_dag():
    import discopt.modeling as dm

    m = dm.Model("shared_dag")
    v = m.continuous("v", lb=0.0, ub=3.0)
    w = m.continuous("w", shape=(4,), lb=-1.0, ub=1.0)
    s = dm.exp(v) + dm.sin(w[0]) * w[1]  # one object, referenced 21 times
    m.minimize(s * s)
    for i in range(20):
        m.subject_to(s + 0.1 * i * w[i % 4] <= 50.0, name=f"r{i}")
    return m, {}


def fixed_scope():
    """Saved inside a scoped fix, so the document carries a non-trivial bound stack.

    ``Variable.fix()`` is not a context manager; ``Variable.fixed()`` is (both exist
    at 1.0). The generator writes the document while the ``with`` block is open.
    """
    import discopt.modeling as dm

    m = dm.Model("fixed_scope")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(dm.sum(x) + y**2)
    m.subject_to(x[0] - y >= 1.0, name="c")
    return m, {"_fixed": (x, [1.0, 2.0])}


def complementarity_lowered():
    import discopt.modeling as dm

    m = dm.Model("complementarity_lowered")
    p = m.continuous("p", lb=0.0, ub=10.0)
    q = m.continuous("q", lb=0.0, ub=10.0)
    m.minimize((p - 1) ** 2 + (q - 1) ** 2)
    m.complementarity(p, q, name="pq")
    return m, {}


def with_result():
    import discopt.modeling as dm

    m = dm.Model("with_result")
    a = m.continuous("a", lb=0.1, ub=10.0)
    b = m.continuous("b", lb=0.1, ub=5.0)
    m.minimize((a - 2.0) ** 2 + (b - 1.5) ** 2 + 0.1 * a * b)
    result = m.solve(time_limit=60)
    return m, {"result": result}


def _from_nl(case):
    import discopt.modeling as dm

    return dm.from_nl(NL_FILES[case]), {}


def from_nl_nvs01():
    return _from_nl("from_nl_nvs01")


def from_nl_st_e13():
    return _from_nl("from_nl_st_e13")


def from_nl_mpcc():
    return _from_nl("from_nl_mpcc")


def fast_api_blocks():
    import discopt.modeling as dm
    import scipy.sparse as sp

    m = dm.Model("fast_api_blocks")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=3.0)
    A = sp.csr_matrix(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]))
    m.add_linear_constraints(A, x, "<=", np.array([2.0, 4.0]), name="blk")
    Q = sp.csr_matrix(np.diag([1.0, 2.0, 0.5, 1.0]))
    m.add_quadratic_objective(Q, np.array([-1.0, 0.0, -2.0, 1.0]), x, constant=0.25)
    return m, {}


def parameters_and_start():
    import discopt.modeling as dm

    m = dm.Model("parameters_and_start")
    x = m.continuous("x", shape=(2,), lb=-3.0, ub=3.0)
    p = m.parameter("p", value=np.array([1.5, -0.5]))
    c = m.parameter("c", value=2.0)
    m.minimize(dm.sum((x - p) ** 2) + c * x[0])
    m.set_initial_point({x: [0.25, -0.75]})
    return m, {}


def decomposition_marks():
    import discopt.modeling as dm

    m = dm.Model("decomposition_marks")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=1.0)
    m.minimize(dm.sum(x))
    m.subject_to(x[0] + x[1] <= 1.0, name="b0")
    m.subject_to(x[2] + x[3] <= 1.0, name="b1")
    m.subject_to(x[0] + x[3] <= 1.5, name="link")
    m.mark_coupling("link")
    m.set_stage(x, 1)
    return m, {}


def registered_atom():
    """Needs ``register_function`` (1.1+); the tag is only written from 1.3."""
    import discopt.modeling as dm
    from discopt.operators import register_function

    fn = register_function("wp1_fixture_cubic", lambda t: t**3 - 2 * t, replace=True)
    m = dm.Model("registered_atom")
    x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
    m.minimize(fn(x[0]) + 2 * fn(x[1]) + dm.sum(x) ** 2)
    return m, {}


def logical_constraints():
    """Propositional logic: every connective, a shared subformula, a cardinality.

    Encoded from 1.5; every earlier writer refuses it (recorded as unbuildable).
    """
    import discopt.modeling as dm
    from discopt.modeling.core import atmost, exactly, land, lnot, lor

    m = dm.Model("logical_constraints")
    Y = m.boolean("Y", shape=(4,))
    z = m.boolean("z")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=10.0)
    m.minimize(dm.sum(x) + 3 * Y.variable[0] - 2 * Y.variable[3] + z.variable)
    for i in range(4):
        m.subject_to(x[i] >= 2 * Y.variable[i], name=f"link{i}")
    shared = Y[1] | ~Y[3]
    m.logical(Y[0].implies(shared), name="imp")
    m.logical(land(atmost(1, Y[0], Y[3]), lor(shared, Y[2])), name="card")
    m.logical(Y[1].equivalent_to(lnot(Y[2]) & z), name="iff")
    m.logical(exactly(1, [Y[1], Y[2]]))
    return m, {}


def complementarity_vector_lowered():
    """A vector complementarity, lowered (GDP) on declaration, element by element.

    The lowering records the rows it emitted; 1.5 writes their positions, and
    earlier writers drop them (a reload reads ``rows=None``, "untracked").
    """
    import discopt.modeling as dm

    m = dm.Model("complementarity_vector_lowered")
    p = m.continuous("p", shape=(2,), lb=0.0, ub=10.0)
    q = m.continuous("q", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(dm.sum((p - 1) ** 2) + dm.sum((q - 1) ** 2))
    m.subject_to(p[0] + q[1] >= 0.5, name="mix")
    m.complementarity(p, q, name="pq")
    return m, {}


def gdp_indicators():
    """Disjunctions whose indicators were read (1.5+, #1775 G1.1).

    ``Model.either_or`` returns nothing before 1.5, so every earlier version
    records this case as unbuildable (its own error, via ``--cases``). The
    indicators are tied together by an algebraic row on their binary columns and
    priced in the objective; ``gdp_indicators_logic`` ties them with
    ``Model.logical`` instead. A third disjunction leaves its indicators unread,
    so its row is written exactly as at 1.4.
    """
    import discopt.modeling as dm

    m = dm.Model("gdp_indicators")
    x = m.continuous("x", lb=0.0, ub=20.0)
    w = m.continuous("w", lb=0.0, ub=20.0)
    mode = m.either_or([[x <= 5.0], [x >= 15.0]], name="mode")
    pump = m.either_or([[w <= 3.0], [w >= 10.0], [w >= 6.0, w <= 7.0]], name="pump")
    m.either_or([[x + w <= 30.0], [x - w >= 0.0]], name="unread")
    hi_mode = mode.indicators[1].variable
    hi_pump = pump.indicators[1].variable
    # "high mode implies high pump", as algebra on the indicator columns.
    m.subject_to(hi_mode <= hi_pump, name="mode_implies_pump")
    m.minimize((x - 12.0) ** 2 + (w - 4.0) ** 2 + 3.0 * pump.indicators[2].variable)
    return m, {}


def gdp_indicators_logic():
    """Propositional logic over ``either_or`` indicators (1.5+, #1775 G1.1 + WP-3).

    The case G1.1 exists for: a ``logical`` row whose leaves are the indicator
    columns of two read disjunctions. "High mode implies high pump" moves the
    optimum from 10 (x=15, w=3) to 45 (x=15, w=10). Unbuildable before 1.5.
    """
    import discopt.modeling as dm

    m = dm.Model("gdp_indicators_logic")
    x = m.continuous("x", lb=0.0, ub=20.0)
    w = m.continuous("w", lb=0.0, ub=20.0)
    mode = m.either_or([[x <= 5.0], [x >= 15.0]], name="mode")
    pump = m.either_or([[w <= 3.0], [w >= 10.0]], name="pump")
    m.logical(mode.indicators[1].implies(pump.indicators[1]), name="mode_implies_pump")
    m.logical(mode.indicators[0] | pump.indicators[0] | mode.indicators[1], name="cover")
    m.minimize((x - 12.0) ** 2 + (w - 4.0) ** 2)
    return m, {}


def index_none_ellipsis():
    """``None`` (``numpy.newaxis``) and ``Ellipsis`` indices (1.6+, WP-3b).

    The spellings ``DAEBuilder.add_control`` and ``TrainableNetwork`` emit. Every
    earlier writer refuses the index, so it is recorded as unbuildable there.
    """
    import discopt.modeling as dm
    import numpy as np

    m = dm.Model("index_none_ellipsis")
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    u = m.continuous("u", shape=(2, 3), lb=-2.0, ub=2.0)
    target = np.array([0.5, -1.0, 1.5])
    # `x[None, :]` broadcasts x across u's rows; `u[..., 0]` is u's first column.
    m.minimize(dm.sum((u - x[None, :]) ** 2) + dm.sum((x - target) ** 2) + dm.sum(u[..., 0]))
    m.subject_to(dm.sum(u[:, None, 1]) >= -1.0, name="col1")
    return m, {}


def nested_disjunction():
    """A ``Model.disjunction()`` as a disjunct member of ``either_or`` (1.6+, WP-3b).

    The outer alternative is infeasible, so the optimum is the inner OR's best
    arm, x = 7; flattening the member into an AND of its rows would make the
    model infeasible. Earlier writers refuse (or crash on) the member.
    """
    import discopt.modeling as dm

    m = dm.Model("nested_disjunction")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.maximize(x)
    inner = m.disjunction([[x >= 1.0, x <= 3.0], [x >= 5.0, x <= 7.0]], name="inner")
    m.either_or([[inner], [x >= 30.0]], name="outer")
    return m, {}


#: name -> builder. Order is the manifest's order; never reorder or rename a case
#: once a shipped version's fixtures exist for it (fixtures are immutable, §0.3.11).
CASES = {
    f.__name__: f
    for f in (
        scalar_nlp,
        array_bounds_inf,
        mip_indicator_sos_disjunction,
        shared_dag,
        fixed_scope,
        complementarity_lowered,
        with_result,
        from_nl_nvs01,
        from_nl_st_e13,
        from_nl_mpcc,
        fast_api_blocks,
        parameters_and_start,
        decomposition_marks,
        registered_atom,
        # Added with schema 1.5 (WP-3 and G1.1). Each is recorded for every
        # earlier version too (``--cases``): what that version's writer did with
        # it, an unbuildable record carrying its own error where it could not.
        logical_constraints,
        complementarity_vector_lowered,
        gdp_indicators,
        gdp_indicators_logic,
        # Added with schema 1.6 (WP-3b), recorded for every earlier version too.
        index_none_ellipsis,
        nested_disjunction,
    )
}


def build(name: str):
    """Build case *name*; returns ``(model, dumps_kwargs, fixed)`` where *fixed* is a
    ``(variable, value)`` pair the caller must hold fixed while writing, or ``None``."""
    model, kwargs = CASES[name]()
    kwargs = dict(kwargs)
    fixed = kwargs.pop("_fixed", None)
    return model, kwargs, fixed
