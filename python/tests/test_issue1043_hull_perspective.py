"""#1043(a): the hull perspective must be exact at both integer faces.

The GDP ``hull`` reformulation writes each nonlinear disjunct row in perspective
form.  The pre-#1043 form was ``g(v_k / (y_k + eps)) * (y_k + eps) <= rhs * y_k``,
which is exact at *neither* integer face:

* ``y_k = 0`` -- the bound-linking rows pin every disaggregated variable of a
  de-selected disjunct to exactly 0, so the row body collapses to ``eps * g(0)``
  rather than 0.  For an **equality** row with ``g(0) != 0`` that is a hard,
  unsatisfiable violation: the reformulated model is empty in exact arithmetic,
  and the rigorous FBBT / McCormick-LP machinery correctly proves it empty.  The
  solver then returns ``infeasible`` for a model with a perfectly good optimum --
  a false infeasible, the worst class of error under CLAUDE.md 1.
* ``y_k = 1`` -- ``y_clamp`` was ``1 + eps``, so the selected disjunct's own row
  read ``g(v / (1 + eps)) * (1 + eps)``, an O(eps) distortion of the constraint
  the user actually wrote.

The fix is the Furman-Sawaya-Grossmann eps-perspective (``Furman2020`` in
``docs/references.bib``; the same form Pyomo.GDP's ``hull`` transformation uses)::

    yhat * g(v_k / yhat) - eps * g(0) * (1 - y_k) <= 0,   yhat = (1 - eps) * y_k + eps

which is exact at both faces: 0 at ``y_k = 0``, and ``g(v)`` at ``y_k = 1``.

Shrinking ``eps`` does not fix the old form -- the repro below is still exactly
infeasible at ``eps = 1e-14`` -- so these tests assert *cancellation*, not
smallness.
"""

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    HullPerspectiveOriginError,
    _body_at_zero,
    reformulate_gdp,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    UnaryOp,
    Variable,
    VarType,
)

# ---------------------------------------------------------------------------
# Shared model builders
# ---------------------------------------------------------------------------

#: Nonlinear bodies for the *de-selected* disjunct, each with ``g(0) != 0``.
#:
#: ``g(0) != 0`` is the whole trigger: it is what the ``y_k = 0`` face leaves
#: behind as ``eps * g(0)``.  The set spans the unary operators the perspective
#: has to survive (log, exp, a power, a trig function), because the fix computes
#: ``g(0)`` by walking the expression DAG and a missing operator there is a
#: refusal, not a wrong number.
OFF_BRANCHES = [
    # (id, f_b, f_b(0), max f_b over x in [-2, 2])
    ("log", lambda x: dm.log(x + 3.0), math.log(3.0), math.log(5.0)),
    ("exp", lambda x: dm.exp(x) - 3.0, -2.0, math.exp(2.0) - 3.0),
    ("cube", lambda x: x**3 + 1.0, 1.0, 9.0),
    ("cos", lambda x: dm.cos(x) + 1.0, 2.0, 2.0),
]

#: Bodies with ``g(0) == 0`` -- the control.  The ``y_k = 0`` residual is
#: ``eps * g(0) == 0`` even in the old formulation, so these must pass *before*
#: and after the fix.  Without them a green suite could just mean "something
#: changed", not "the defect is closed".
OFF_BRANCHES_ZERO_AT_ORIGIN = [
    ("cube0", lambda x: x**3, 0.0, 8.0),
    ("expm1", lambda x: dm.exp(x) - 1.0, 0.0, math.exp(2.0) - 1.0),
    ("sin", lambda x: dm.sin(x), 0.0, math.sin(1.0)),
]

#: The *selected* branch.  ``exp(x) + 2`` peaks at ``e^2 + 2 ~ 9.389`` on
#: ``[-2, 2]``, above every ``max f_b`` above, so disjunct 0 is the optimal
#: choice in every case and the expected objective is analytic.  Its own
#: ``g(0) = -2 != 0`` also exercises the ``y_k = 1`` face, where the old
#: ``yhat = 1 + eps`` distorted the row.
ON_BRANCH_MAX = math.exp(2.0) + 2.0


def _two_branch_model(name: str, f_b) -> dm.Model:
    """``max y`` subject to ``(y == exp(x) + 2) or (y == f_b(x))``, x in [-2, 2].

    Both disjuncts carry a nonlinear **equality**, which is what makes the
    de-selected one's eps residual a hard violation rather than something a
    feasibility tolerance can absorb.
    """
    m = dm.Model(name)
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-10.0, ub=20.0)
    m.maximize(y)
    m.either_or(
        [
            [Constraint(body=y - (dm.exp(x) + 2.0), sense="==", rhs=0.0)],
            [Constraint(body=y - f_b(x), sense="==", rhs=0.0)],
        ],
        name="br",
    )
    return m


# ---------------------------------------------------------------------------
# The end-to-end regression: no false infeasible
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(300)
class TestNoFalseInfeasible:
    def test_minimal_repro(self):
        """The issue's reproduction verbatim: ``infeasible`` before, ``log 2`` after.

        ``if_else`` is one producer of the trigger shape -- it emits the branch
        value as an equality on an auxiliary variable, and both branches here are
        nonlinear with ``g(0) != 0``.
        """
        m = dm.Model("issue1043")
        x = m.continuous("x", lb=-10.0, ub=10.0)
        m.maximize(x)
        m.subject_to(dm.if_else(x >= 0, dm.exp(x) - 1, dm.log(-x + 3)) <= 1.0)

        r = m.solve()
        assert r.status == "optimal", (
            f"#1043(a): hull perspective reported {r.status!r} for a model whose "
            f"optimum is log 2 = {math.log(2.0)!r}"
        )
        assert r.objective == pytest.approx(math.log(2.0), abs=1e-6)

    @pytest.mark.parametrize(
        ("case", "f_b", "g0", "max_b"),
        OFF_BRANCHES,
        ids=[c[0] for c in OFF_BRANCHES],
    )
    def test_deselected_nonlinear_equality(self, case, f_b, g0, max_b):
        """A de-selected disjunct with ``g(0) != 0`` must not empty the model."""
        assert g0 != 0.0, "this case is supposed to trigger the eps residual"
        assert max_b < ON_BRANCH_MAX, "disjunct 0 must be the optimal choice"

        r = _two_branch_model(f"off_{case}", f_b).solve(gdp_method="hull")
        assert r.status == "optimal", (
            f"#1043(a): {case} de-selected branch reported {r.status!r}; the "
            f"eps residual on the y_k = 0 face is eps * {g0!r}"
        )
        assert r.objective == pytest.approx(ON_BRANCH_MAX, rel=1e-6)

    @pytest.mark.parametrize(
        ("case", "f_b", "g0", "max_b"),
        OFF_BRANCHES_ZERO_AT_ORIGIN,
        ids=[c[0] for c in OFF_BRANCHES_ZERO_AT_ORIGIN],
    )
    def test_control_zero_at_origin(self, case, f_b, g0, max_b):
        """Control: ``g(0) == 0`` leaves no residual, so this passes either way."""
        assert g0 == 0.0
        assert max_b < ON_BRANCH_MAX

        r = _two_branch_model(f"ctl_{case}", f_b).solve(gdp_method="hull")
        assert r.status == "optimal"
        assert r.objective == pytest.approx(ON_BRANCH_MAX, rel=1e-6)


# ---------------------------------------------------------------------------
# The general contract: exactness at both integer faces
# ---------------------------------------------------------------------------

_UNARY = {
    "neg": lambda a: -a,
    "-": lambda a: -a,
    "abs": abs,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tanh": math.tanh,
}
_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "**": lambda a, b: a**b,
    "^": lambda a, b: a**b,
}


def _evaluate(expr, values: dict[str, float], counter: list[int]) -> float:
    """Evaluate an expression DAG at ``values``.

    Deliberately total: an unknown variable raises ``KeyError`` and an unknown
    operator raises ``KeyError`` too.  A probe that swallowed either would report
    "0 violations" for rows it never actually evaluated (CLAUDE.md 6, 7).
    """
    counter[0] += 1
    if isinstance(expr, Variable):
        return values[expr.name]
    if isinstance(expr, Constant):
        return float(np.asarray(expr.value).ravel()[0])
    if isinstance(expr, UnaryOp):
        return float(_UNARY[expr.op](_evaluate(expr.operand, values, counter)))
    if isinstance(expr, BinaryOp):
        return float(
            _BINARY[expr.op](
                _evaluate(expr.left, values, counter),
                _evaluate(expr.right, values, counter),
            )
        )
    if isinstance(expr, FunctionCall):
        args = [_evaluate(a, values, counter) for a in expr.args]
        return float(_UNARY[expr.func_name](*args))
    raise TypeError(f"unhandled node {type(expr).__name__}: {expr!r}")


def _hull_point(reformulated: dm.Model, origin: dict[str, float], active: int) -> dict:
    """Build the hull-space point that corresponds to ``origin`` with disjunct
    ``active`` selected.

    Names are *discovered* from the reformulated model rather than hardcoded per
    instance, so this works for any single-disjunction model (CLAUDE.md 2).
    """
    values = dict(origin)
    selectors, disaggregated = 0, 0
    for v in reformulated._variables:
        if v.name in values:
            continue
        stem, _, k_text = v.name.rpartition("_")
        k = int(k_text)
        if v.var_type == VarType.BINARY:
            values[v.name] = 1.0 if k == active else 0.0
            selectors += 1
            continue
        original = next((n for n in origin if stem.endswith(f"_v_{n}")), None)
        if original is None:
            raise AssertionError(f"unclassified hull variable {v.name!r}")
        values[v.name] = origin[original] if k == active else 0.0
        disaggregated += 1
    assert selectors >= 2, f"discovered only {selectors} selector binaries"
    assert disaggregated >= 2, f"discovered only {disaggregated} disaggregated vars"
    return values


def _max_violation(reformulated: dm.Model, values: dict[str, float]) -> tuple[float, str, int]:
    """Largest constraint violation of ``values``, with the row that attains it."""
    worst, worst_row, checked, nodes = 0.0, "", 0, [0]
    for con in reformulated._constraints:
        residual = _evaluate(con.body, values, nodes) - float(np.asarray(con.rhs).ravel()[0])
        if con.sense == "<=":
            violation = max(0.0, residual)
        elif con.sense == ">=":
            violation = max(0.0, -residual)
        else:
            violation = abs(residual)
        checked += 1
        if violation > worst:
            worst, worst_row = violation, con.name or "<unnamed>"
    assert checked > 0 and nodes[0] > 0, "PROBE FIRED NOTHING: no rows were evaluated"
    return worst, worst_row, checked


@pytest.mark.parametrize(
    ("case", "f_b", "g0", "max_b"),
    OFF_BRANCHES,
    ids=[c[0] for c in OFF_BRANCHES],
)
def test_hull_rows_exact_at_both_integer_faces(case, f_b, g0, max_b):
    """The reformulation must not cut off any point the disjunction admits.

    This is the contract the end-to-end tests observe indirectly, asserted
    directly on the rows: for each disjunct ``k``, the hull point built from a
    genuinely feasible ``(x, y)`` of disjunct ``k`` must satisfy *every*
    reformulated row.  Before the fix the de-selected disjunct's row was violated
    by ``eps * |g(0)|`` and the selected one by an O(eps) distortion.
    """
    reformulated = reformulate_gdp(_two_branch_model(f"faces_{case}", f_b), method="hull")
    assert reformulated is not None

    checks = 0
    for active, f in ((0, lambda t: math.exp(t) + 2.0), (1, None)):
        x_value = 0.5
        if active == 0:
            y_value = f(x_value)
        else:
            # Evaluate the parametrized branch numerically through the DAG.
            probe = dm.Model("probe")
            xv = probe.continuous("x", lb=-2.0, ub=2.0)
            y_value = _evaluate(f_b(xv), {"x": x_value}, [0])
        origin = {"x": x_value, "y": y_value}
        worst, row, n_rows = _max_violation(reformulated, _hull_point(reformulated, origin, active))
        assert worst <= 1e-9, (
            f"#1043(a): disjunct {active} face of {case} violates {row} by {worst:.6e} "
            f"({n_rows} rows checked); a feasible point of the original disjunction "
            f"must satisfy every hull row"
        )
        checks += 1
    assert checks == 2, "PROBE FIRED NOTHING: neither integer face was checked"


# ---------------------------------------------------------------------------
# g(0): the value, and the refusal
# ---------------------------------------------------------------------------


def _named(expr_fn):
    m = dm.Model("g0")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    return expr_fn(x, y), {"x": x, "y": y}


class TestBodyAtZero:
    @pytest.mark.parametrize(
        ("case", "build", "expected"),
        [
            ("affine", lambda x, y: 2.0 * x - 3.0 * y + 7.0, 7.0),
            ("exp", lambda x, y: dm.exp(x) - 1.0, 0.0),
            ("log", lambda x, y: dm.log(x + 3.0), math.log(3.0)),
            ("power", lambda x, y: x**3 + 1.0, 1.0),
            ("cos", lambda x, y: dm.cos(x) + 1.0, 2.0),
            ("product", lambda x, y: x * y + 4.0, 4.0),
            ("quotient", lambda x, y: (x + 2.0) / (y + 4.0), 0.5),
            ("nested", lambda x, y: dm.exp(dm.sin(x) + y) - 2.0, -1.0),
        ],
    )
    def test_value(self, case, build, expected):
        expr, variables = _named(build)
        assert _body_at_zero(expr, variables) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(
        ("case", "build"),
        [
            ("log_at_zero", lambda x, y: dm.log(x)),
            ("divide_by_zero", lambda x, y: 1.0 / x),
            ("sqrt_negative", lambda x, y: dm.sqrt(x - 1.0)),
        ],
        ids=["log_at_zero", "divide_by_zero", "sqrt_negative"],
    )
    def test_refuses_when_not_finite_at_origin(self, case, build):
        """No finite ``g(0)`` means no sound perspective -- refuse, do not guess.

        Fabricating a value here would leave an uncancelled eps residual on the
        de-selected disjunct, which is precisely the false-infeasible class this
        function exists to close (CLAUDE.md 3).
        """
        expr, variables = _named(build)
        with pytest.raises(HullPerspectiveOriginError):
            _body_at_zero(expr, variables)

    def test_refuses_unknown_variable(self):
        expr, variables = _named(lambda x, y: x + y)
        with pytest.raises(HullPerspectiveOriginError, match="not collected"):
            _body_at_zero(expr, {"x": variables["x"]})
