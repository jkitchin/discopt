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


def test_builder_rows_fall_back_to_the_python_writer():
    """Builder rows are ordered differently in the arena, so the Rust path refuses.

    They sit AHEAD of the expression rows there and AFTER them in
    ``model._constraints``. Emitting the arena's order would write the same model
    with its rows PERMUTED -- and row order is how a solver's ``.sol`` duals map
    back to constraints, so it is not cosmetic.
    """
    m = dm.Model("blk")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=2.0)
    m.subject_to(x[0] * dm.exp(x[1]) <= 4.0, name="nl")
    m.add_linear_constraints(np.array([[1.0, 1.0, 1.0]]), x, "<=", np.array([2.0]))
    m.minimize(-x[0])
    _assert_identical(m, "builder rows")


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
