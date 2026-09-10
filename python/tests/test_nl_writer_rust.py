"""The Rust ``.nl`` writer must be byte-identical to the Python one.

Writing ``.nl`` was the whole remaining external-solver performance gap: the
Python writer costs **16.05 µs/row of a 16.21 µs/row** model-to-file pipeline,
against oximo's 0.86 for the same work (``docs/dev/performance-plan.md`` §43).
``discopt_core::nl_writer`` replaces it and is 9.3x faster on a vectorised model,
putting discopt at 1.66 µs/row end to end -- 10.5x Pyomo and 1.57x oximo.

Because it is a *replacement* and not a new format, the bar is byte-identity
rather than "solves to the same answer". That is both the strongest check
available and the cheapest to act on: a divergence localises itself in the diff,
where a round-trip test would report the same optimum while hiding, say, a wrong
header census that only some solvers read.

The equality is asserted against the Python writer directly, by forcing it with
``DISCOPT_RUST_NL=0``, so these tests keep working as a differential even if the
Rust path is later made unconditional.
"""

from __future__ import annotations

import glob
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import to_nl
from discopt.export.nl import _rust_nl_text

pytestmark = pytest.mark.smoke

_CORPUS = sorted(glob.glob("python/tests/data/minlplib_nl/*.nl"))


def _python_nl(model) -> str:
    """The Python writer's output, forced."""
    prev = os.environ.get("DISCOPT_RUST_NL")
    os.environ["DISCOPT_RUST_NL"] = "0"
    try:
        return to_nl(model)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_RUST_NL", None)
        else:
            os.environ["DISCOPT_RUST_NL"] = prev


def _assert_identical(model, label: str) -> str:
    rust, python = to_nl(model), _python_nl(model)
    if rust != python:
        rl, pl = rust.split("\n"), python.split("\n")
        i = next(
            (
                j
                for j in range(max(len(rl), len(pl)))
                if (rl[j] if j < len(rl) else None) != (pl[j] if j < len(pl) else None)
            ),
            -1,
        )
        pytest.fail(
            f"{label}: writers diverge at line {i}\n"
            f"  python: {pl[i] if i < len(pl) else '<eof>'!r}\n"
            f"  rust:   {rl[i] if i < len(rl) else '<eof>'!r}"
        )
    return rust


# ── hand-written shapes ─────────────────────────────────────────────────────


def _minlp():
    """Integrality is the case POUNCE's `NlProblem` could not carry (§44).

    Exercises the canonical variable order and the ``nbv niv nlvbi nlvci nlvoi``
    header line: get that wrong and a discrete variable appearing nonlinearly is
    declared linear-discrete, which AMPL-compatible solvers read as *continuous*
    and silently relax (issue #210).
    """
    m = dm.Model("minlp")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    y = m.binary("y", shape=(2,))
    z = m.integer("z", lb=0, ub=5)
    m.subject_to(x[0] * x[1] + y[0] <= 4.0, name="bilin")
    m.subject_to(x[0] + y[1] + z <= 8.0, name="lin")
    m.minimize(-x[0] - 2.0 * y[0] - z)
    return m


def _vectorised():
    m = dm.Model("vec")
    x = m.continuous("x", shape=(5,), lb=0.1, ub=10.0)
    y = m.continuous("y", shape=(5,), lb=0.1, ub=10.0)
    m.subject_to(dm.exp(x) + y <= np.full(5, 9.0), name="c")
    m.minimize(dm.sum(x))
    return m


def _matmul():
    m = dm.Model("matmul")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=5.0)
    a = np.array([[1.0, 2.0, 0.5, 1.0], [0.0, 1.0, 1.5, 2.0]])
    m.subject_to(a @ x <= np.array([6.0, 7.0]), name="Ax")
    m.minimize(-dm.sum(x))
    return m


def _equality():
    m = dm.Model("eq")
    x = m.continuous("x", shape=(3,), lb=-5.0, ub=5.0)
    m.subject_to(x[0] + x[1] == 2.0, name="e")
    m.subject_to(dm.log(x[2] + 6.0) >= 1.0, name="g")
    m.minimize(x[0] * x[0] + x[1])
    return m


def _free_and_fixed():
    m = dm.Model("bounds")
    a = m.continuous("a", lb=-1e20, ub=1e20)
    b = m.continuous("b", lb=2.0, ub=2.0)
    c = m.continuous("c", lb=0.0, ub=1e20)
    m.subject_to(a + b + c <= 10.0, name="s")
    m.minimize(a + c)
    return m


SHAPES = {
    "minlp": _minlp,
    "vectorised": _vectorised,
    "matmul": _matmul,
    "equality": _equality,
    "free_and_fixed": _free_and_fixed,
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_writers_agree_byte_for_byte(name):
    _assert_identical(SHAPES[name](), name)


@pytest.mark.skipif(not _CORPUS, reason="MINLPLib corpus not present")
@pytest.mark.parametrize("path", _CORPUS, ids=lambda p: os.path.basename(p)[:-3])
def test_writers_agree_on_the_minlplib_corpus(path):
    """Real instances reach shapes no hand-written model does.

    Every operator in the IR, integrality mixes, free/fixed bounds, equality and
    range rows, and objectives that are pure constants.
    """
    _assert_identical(dm.from_nl(path), os.path.basename(path))


# ── the objective constant: a real bug this differential caught ─────────────


def test_objective_constant_is_written_not_dropped():
    """An objective's split-out constant must go back into its body.

    A constraint carries its constant in the ``r``-section bound; an objective
    has no bound to carry it. The first Rust draft split the constant out and
    never re-emitted it, which shifted the exported objective by that constant --
    ``nvs06``'s ``(0.1 * ...) + 1.2`` exported as if the ``+ 1.2`` were not
    there. Silent, and invisible to a row-count or status check.
    """
    m = dm.Model("objconst")
    x = m.continuous("x", shape=(2,), lb=0.5, ub=3.0)
    m.subject_to(x[0] + x[1] <= 4.0, name="c")
    m.minimize(x[0] * x[1] + 1.25)

    text = _assert_identical(m, "objconst")
    assert "n1.25\n" in text, "the objective constant is missing from the .nl body"
    # And it is counted: a body that is not `n0` is a nonlinear objective.
    assert text.split("\n")[2].startswith(" 0 1"), "nonlinear-objective count is wrong"


def test_constant_only_objective_still_round_trips():
    m = dm.Model("constobj")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.subject_to(x[0] + x[1] <= 1.0, name="c")
    m.minimize(x[0] * 0.0 + 7.5)
    text = _assert_identical(m, "constobj")
    assert "n7.5\n" in text


# ── refusals fall back rather than diverge ──────────────────────────────────


# ── builder-resident rows ───────────────────────────────────────────────────
#
# These used to be a refusal: builder rows lead the arena and trail every Python
# writer, so emitting the arena's order would have written the same model with
# its rows PERMUTED, and row order is how a solver's `.sol` duals map back to
# constraints. Refusing paired the FASTEST construction path with the SLOWEST
# writer -- a 20 000-row bulk model built in 0.31 us/row and exported at 22.83.
#
# `nl_writer::write_nl` now takes the builder-row boundary from the repr and
# reorders to the Python writers' order, so the fast path is available and the
# bytes are unchanged from what discopt has always written.


def _builder_model(order: str, n: int = 12):
    """A model mixing builder rows with an expression row, in either order."""
    m = dm.Model(f"blk_{order}")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=2.0)
    A = np.eye(n)
    b = 2.0 + np.arange(n) % 3
    if order == "expr_first":
        m.subject_to(x[0] * dm.exp(x[1]) <= 4.0, name="nl")
        m.add_linear_constraints(A, x, "<=", b, name="bulk")
    else:
        m.add_linear_constraints(A, x, "<=", b, name="bulk")
        m.subject_to(x[0] * dm.exp(x[1]) <= 4.0, name="nl")
    m.minimize(-x[0])
    return m


@pytest.mark.parametrize("order", ("expr_first", "bulk_first"))
def test_builder_rows_are_served_by_the_rust_writer(order):
    m = _builder_model(order)
    assert _rust_nl_text(m) is not None, "the Rust writer declined a builder model"
    _assert_identical(m, f"builder rows, {order}")


def test_builder_rows_with_an_array_valued_expression_row():
    """Fan-out and reordering at once: one array body is many rows."""
    n = 12
    m = dm.Model("blk_arr")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=2.0)
    y = m.continuous("y", shape=(n,), lb=0.1, ub=2.0)
    m.subject_to(dm.exp(x) + y <= 4.0, name="arr")
    m.add_linear_constraints(np.eye(n), x, "<=", np.full(n, 2.0), name="bulk")
    m.minimize(-x[0])
    text = _assert_identical(m, "builder + array body")
    assert int(text.split("\n")[1].split()[1]) == 2 * n


def test_the_fast_constraint_family_path_is_served_too():
    """`Model.constraint`'s linear fast path is builder-resident as well."""
    n = 12
    m = dm.Model("fam")
    idx = m.set("I", list(range(n)))
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    m.constraint(idx, lambda i: x[i] <= 2.0 + i % 3, name="fam")
    m.minimize(dm.sum(x))
    assert _rust_nl_text(m) is not None
    _assert_identical(m, "fast family")


@pytest.mark.parametrize("sense", ("<=", ">=", "=="))
def test_every_builder_sense_agrees(sense):
    n = 8
    m = dm.Model(f"sense_{sense}")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    m.add_linear_constraints(np.eye(n), x, sense, np.full(n, 2.0), name="s")
    m.subject_to(dm.log(x[0] + 1.0) <= 5.0, name="expr")
    m.minimize(dm.sum(x))
    _assert_identical(m, f"builder sense {sense}")


def test_the_reorder_boundary_is_the_builder_row_count():
    """The count travels with the repr, so it cannot disagree with the rows.

    Asserting it directly means a change to how `model_to_repr` assembles
    `constraints` shows up here rather than as a silent permutation.
    """
    from discopt._rust import model_to_repr

    n = 12
    m = _builder_model("expr_first", n)
    rep = model_to_repr(m, getattr(m, "_builder", None))
    assert rep.n_builder_constraints == n
    assert rep.n_constraints == n + 1
    # Builder rows really do lead the arena -- the premise the reorder undoes.
    names = [rep.constraint_name(i) for i in range(rep.n_constraints)]
    assert names[:n] == [f"bulk_{i}" for i in range(n)]
    assert names[n] == "nl"


def test_a_model_with_no_builder_rows_reports_a_zero_boundary():
    """Zero must mean "no reorder", which is what every non-builder path passes."""
    from discopt._rust import model_to_repr

    m = dm.Model("pure")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=2.0)
    m.subject_to(dm.exp(x[0]) <= 4.0, name="nl")
    m.minimize(-x[0])
    rep = model_to_repr(m, getattr(m, "_builder", None))
    assert rep.n_builder_constraints == 0


def test_custom_call_still_refuses_loudly():
    """``dm.custom`` has no arena form; the fallback must still raise, not export."""
    m = dm.Model("custom")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=2.0)
    f = dm.custom(lambda v: v * 2.0, name="double")
    m.subject_to(f(x[0]) + x[1] <= 5.0, name="c")
    m.minimize(-x[0])
    with pytest.raises(ValueError, match="dm.custom|double"):
        to_nl(m)


def test_opt_out_env_var_selects_the_python_writer(monkeypatch):
    """``DISCOPT_RUST_NL=0`` must isolate a suspected regression without a rebuild."""
    m = _vectorised()
    monkeypatch.setenv("DISCOPT_RUST_NL", "0")
    forced = to_nl(m)
    monkeypatch.delenv("DISCOPT_RUST_NL")
    assert to_nl(m) == forced


def test_matrix_norm_is_refused_by_the_rust_path_too():
    """A 2-D ``norm`` is the induced norm, not a fold -- neither writer may expand it.

    The Rust expansion originally applied its ``norm2`` reduction rule at any
    rank, so a matrix norm expanded ENTRYWISE and exported the Frobenius norm
    instead. It produced a valid file with one row and the wrong mathematics --
    exactly the silent-substitution failure the Python ``_scalarize`` had, and it
    was reproduced in Rust. Both must refuse.
    """
    m = dm.Model("mnorm")
    xs = m.continuous("X", shape=(2, 2), lb=-1.0, ub=1.0)
    m.subject_to(dm.norm(xs) <= 1.0, name="spec")
    m.minimize(-xs[0, 0])
    with pytest.raises(ValueError, match="matrix|induced"):
        to_nl(m)


def test_vector_norm_still_expands_and_agrees():
    """The 1-D case must keep working, and to one row, not one per element."""
    m = dm.Model("vnorm")
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    m.subject_to(dm.norm(x) <= 1.5, name="ball")
    m.minimize(-x[0] - x[1])
    text = _assert_identical(m, "vnorm")
    assert int(text.split("\n")[1].split()[1]) == 1, "a reduction is one row"
