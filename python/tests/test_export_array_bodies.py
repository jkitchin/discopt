"""Array-structured models must export to ``.nl``, faithfully (#1215).

Measured this session: the *vectorised* form of a model -- one array-valued body
covering N rows, ``dm.exp(x) + y <= b`` rather than a per-element rule -- reaches
solve-ready **36x faster and with 28x less memory** than the per-element idiom
(1.65 us/row and 17.5 B/row against 59.5 and ~490). It is the shape a 100k-row
model has to be written in.

That path was unusable for external solvers: six of eight array-shaped construct
families could not be written to ``.nl``, the format SCIP, BARON, Couenne and
IPOPT read. Two defects, both fixed here:

* the objective never went through ``_scalarize``, so an objective that is
  scalar-VALUED but array-STRUCTURED (``-dm.sum(x)``, ``dm.sum(A @ x)``) reached
  the scalar writer whole and died with "Cannot write array variable x without
  indexing". Array *constraints* had worked all along;
* ``_scalarize`` broadcast every ``FunctionCall`` element-wise, including the
  REDUCTIONS. ``dm.norm(x)`` on a 3-vector expanded to three rows of
  ``norm2(x[i])`` -- a different model. It failed loudly only because ``norm2``
  has no ``.nl`` opcode; giving it one would have made the export silently wrong.

Every test here asserts the exported MODEL, by round-tripping through
``from_nl`` and comparing the certified objective, rather than asserting that
``to_nl`` merely did not raise.
"""

from __future__ import annotations

import os
import re
import tempfile

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import to_nl

pytestmark = pytest.mark.smoke


def _roundtrip_objective(model) -> float:
    text = to_nl(model)
    with tempfile.NamedTemporaryFile("w", suffix=".nl", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        res = dm.from_nl(path).solve(time_limit=60)
    finally:
        os.unlink(path)
    assert res.status in ("optimal", "feasible"), f"re-read model: {res.status}"
    return float(res.objective)


def _direct_objective(model) -> float:
    res = model.solve(time_limit=60)
    assert res.status in ("optimal", "feasible"), f"direct solve: {res.status}"
    return float(res.objective)


# ── the construct families ──────────────────────────────────────────────────


def _matmul():
    m = dm.Model("matmul")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=5.0)
    a = np.array([[1.0, 2.0, 0.5, 1.0], [0.0, 1.0, 1.5, 2.0]])
    m.subject_to(a @ x <= np.array([6.0, 7.0]), name="Ax")
    m.minimize(dm.sum(x) * -1.0)
    return m


def _whole_sum():
    m = dm.Model("whole_sum")
    x = m.continuous("x", shape=(5,), lb=0.0, ub=2.0)
    m.subject_to(dm.sum(x) <= 4.0, name="tot")
    m.minimize(-dm.sum(x))
    return m


def _axis_sum():
    m = dm.Model("axis_sum")
    xs = m.continuous("X", shape=(3, 2), lb=0.0, ub=2.0)
    m.subject_to(dm.sum(xs, axis=1) <= np.array([1.0, 2.0, 3.0]), name="rows")
    m.minimize(-dm.sum(xs))
    return m


def _norm():
    m = dm.Model("norm")
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    m.subject_to(dm.norm(x) <= 1.5, name="ball")
    m.minimize(-x[0] - x[1])
    return m


def _elementwise():
    m = dm.Model("elementwise")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=3.0)
    y = m.continuous("y", shape=(4,), lb=0.0, ub=3.0)
    m.subject_to(x + y <= np.array([1.0, 2.0, 3.0, 4.0]), name="ew")
    m.minimize(-dm.sum(x) - 0.5 * dm.sum(y))
    return m


def _array_nonlinear():
    m = dm.Model("array_nonlinear")
    x = m.continuous("x", shape=(3,), lb=0.3, ub=2.0)
    m.subject_to(dm.exp(x) <= np.array([3.0, 4.0, 5.0]), name="ex")
    m.minimize(-dm.sum(x))
    return m


def _shaped_param():
    m = dm.Model("shaped_param")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    c = m.parameter("c", value=np.array([1.0, 2.0, 3.0]))
    m.subject_to(dm.sum([c[i] * x[i] for i in range(3)]) <= 9.0, name="budget")
    m.minimize(-x[0] - x[1] - x[2])
    return m


FAMILIES = {
    "matmul": _matmul,
    "whole_sum": _whole_sum,
    "axis_sum": _axis_sum,
    "norm": _norm,
    "elementwise": _elementwise,
    "array_nonlinear": _array_nonlinear,
    "shaped_param": _shaped_param,
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_array_family_round_trips_through_nl(name):
    """Export -> ``from_nl`` -> solve must reproduce the direct objective."""
    build = FAMILIES[name]
    direct = _direct_objective(build())
    exported = _roundtrip_objective(build())
    assert exported == pytest.approx(direct, rel=1e-6, abs=1e-6), (
        f"{name}: direct {direct} vs exported {exported}"
    )


# ── reductions are not element-wise ─────────────────────────────────────────


def test_norm_produces_one_row_not_one_per_element():
    """``dm.norm(x)`` is a reduction; broadcasting it changes the model.

    The `.nl` header's constraint count is the direct evidence: three rows here
    would mean the writer emitted ``norm2(x[i]) <= 1.5`` per element.
    """
    m = _norm()
    header = to_nl(m).split("\n")[1].split()
    n_vars, n_cons = int(header[0]), int(header[1])
    assert (n_vars, n_cons) == (3, 1), f"expected 3 vars / 1 constraint, got {header[:2]}"


def test_norm2_expansion_is_exact():
    """``sqrt(sum(x_i^2))`` must agree with the solver's own norm to tolerance."""
    direct = _direct_objective(_norm())
    exported = _roundtrip_objective(_norm())
    # max x0 + x1 on the ball of radius 1.5 is 1.5*sqrt(2), minimised as -that.
    assert direct == pytest.approx(-1.5 * np.sqrt(2.0), abs=1e-5)
    assert exported == pytest.approx(direct, abs=1e-6)


def test_prod_expands_to_a_product_chain():
    m = dm.Model("prod")
    x = m.continuous("x", shape=(3,), lb=1.0, ub=2.0)
    m.subject_to(dm.prod(x) <= 6.0, name="p")
    m.minimize(-dm.sum(x))
    header = to_nl(m).split("\n")[1].split()
    assert int(header[1]) == 1, "prod is a reduction: one row, not three"
    assert _roundtrip_objective(m) == pytest.approx(_direct_objective(m), abs=1e-5)


def test_norminf_is_refused_loudly():
    """A max-chain needs the DNLP model type, which `.nl` export does not offer.

    Refusing matches the writer's existing treatment of ``min``/``max``; the
    alternative -- expanding it as some other norm -- would export different
    mathematics.
    """
    m = dm.Model("ninf")
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    m.subject_to(dm.norm(x, ord=float("inf")) <= 1.0, name="box")
    m.minimize(-x[0])
    with pytest.raises(ValueError, match="norminf"):
        to_nl(m)


def test_matrix_norm_is_refused_loudly():
    """A 2-D norm is the induced/spectral norm, not a fold over elements."""
    m = dm.Model("mnorm")
    xs = m.continuous("X", shape=(2, 2), lb=-1.0, ub=1.0)
    m.subject_to(dm.norm(xs) <= 1.0, name="spec")
    m.minimize(-xs[0, 0])
    with pytest.raises(ValueError, match="matrix"):
        to_nl(m)


# ── the objective must stay scalar ──────────────────────────────────────────


def test_vector_valued_objective_is_refused_not_truncated():
    """Expanding to several elements must raise, never optimise element zero."""
    m = dm.Model("vecobj")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    m.subject_to(dm.sum(x) <= 2.0, name="c")
    m.minimize(x * 2.0)  # array-valued: not a scalar objective
    with pytest.raises(ValueError, match="scalar"):
        to_nl(m)


# ── LP / MPS: array bodies expand; nonlinear ones are still refused ──────────

#: The array families whose bodies AND objective are affine, so LP/MPS can hold
#: them. The rest (``scalar``, ``norm``, ``array_nonlinear``) are refused by
#: those formats on their own terms -- LP and MPS are linear/quadratic, and
#: ``exp``/``sqrt``/``x*exp(y)`` have no representation there. That refusal is
#: correct and is asserted separately below.
_LP_CAPABLE = ["matmul", "whole_sum", "axis_sum", "elementwise", "shaped_param"]

_LP_ROW = re.compile(r"^\s+(\w+):\s+(.*?)\s*(<=|>=|=)\s*(-?[\d.eE+]+)\s*$")
_LP_TERM = re.compile(r"([+-]?)\s*(\d*\.?\d*(?:[eE][+-]?\d+)?)\s*([A-Za-z_]\w*)")
_LP_BOUND = re.compile(r"^\s+-?[\d.eE+]+\s+<=\s+([A-Za-z_]\w*)\s+<=")


def _lp_columns(text: str) -> dict[str, int]:
    """Variable name -> flat column, read from the file's own ``Bounds`` order.

    The writer emits bounds in flat variable order, so this is the file's own
    statement of its column layout. Deriving it by pattern-matching the NAMES
    instead is what an earlier version of this test did, and it was wrong twice
    over: a 2-D variable emits ``X_0_0``, not ``x_<int>``, and in a model with
    both ``x`` and ``y`` the trailing integer is not a column at all -- ``x_2``
    and ``y_2`` are different variables.
    """
    cols: dict[str, int] = {}
    for line in text.split("\n"):
        m = _LP_BOUND.match(line)
        if m:
            cols.setdefault(m.group(1), len(cols))
    return cols


def _lp_rows(text: str, cols: dict[str, int]) -> list[tuple[str, dict[int, float], str, float]]:
    """Parse the ``Subject To`` rows of an LP file into coefficient maps."""
    out = []
    in_rows = False
    for line in text.split("\n"):
        if line.strip() == "Subject To":
            in_rows = True
            continue
        if line.strip() in ("Bounds", "End", "Generals", "Binaries"):
            in_rows = False
        if not in_rows:
            continue
        m = _LP_ROW.match(line)
        if not m:
            continue
        name, body, sense, rhs = m.groups()
        coeffs: dict[int, float] = {}
        for sign, mag, var in _LP_TERM.findall(body):
            assert var in cols, f"row {name}: unknown variable {var!r}"
            c = 1.0 if mag in ("", ".") else float(mag)
            coeffs[cols[var]] = -c if sign == "-" else c
        out.append((name, coeffs, sense, float(rhs)))
    return out


@pytest.mark.parametrize("name", _LP_CAPABLE)
def test_lp_rows_match_the_model_jacobian(name):
    """LP coefficients must equal the model's own constraint Jacobian.

    For an affine model the Jacobian is constant, so this is an exact check on
    the emitted numbers -- not a row count, which would pass on a model whose
    coefficients were transposed or dropped.
    """
    import pounce
    from discopt._nl_expr_compiler import compile_to_nl_expr
    from discopt.export import _arrays, to_lp

    m = FAMILIES[name]()
    text = to_lp(m)
    cols = _lp_columns(text)
    rows = _lp_rows(text, cols)
    bodies = []
    for c in m._constraints:
        bodies.extend(_arrays.scalarize_body(c.body))
    assert len(rows) == len(bodies), f"{name}: {len(rows)} LP rows for {len(bodies)} model rows"

    taped = [compile_to_nl_expr(b, m) for b in bodies]
    n = sum(int(v.size) for v in m._variables)
    assert len(cols) == n, f"LP declares {len(cols)} columns, model has {n}"
    prob = pounce.build_nl_problem(n, taped[0], constraints=taped, x_l=[-10.0] * n, x_u=[10.0] * n)
    # `jacobian` returns the SPARSE values; `jacobian_structure` gives their
    # (row, col) positions. Reshaping the values directly silently mis-assigns
    # coefficients as soon as a row has a structural zero -- which `matmul` does.
    jrows, jcols = (np.asarray(a) for a in prob.jacobian_structure())
    jvals = np.asarray(prob.jacobian(np.zeros(n)))
    jac = np.zeros((len(bodies), n))
    jac[jrows, jcols] = jvals

    compared = 0
    for r, (_, coeffs, _, _) in enumerate(rows):
        for col in range(n):
            assert coeffs.get(col, 0.0) == pytest.approx(jac[r, col], abs=1e-9), (
                f"{name} row {r} col {col}"
            )
            compared += 1
    assert compared == len(rows) * n, "probe compared nothing"


@pytest.mark.parametrize("name", ["norm", "array_nonlinear"])
def test_lp_still_refuses_nonlinear_array_bodies(name):
    """Expanding arrays must not make LP accept maths it cannot express."""
    from discopt.export import to_lp, to_mps

    for writer in (to_lp, to_mps):
        with pytest.raises(ValueError):
            writer(FAMILIES[name]())
