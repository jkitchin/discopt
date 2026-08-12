"""``m.implicit(formulation="full_space")`` -- lower a block instead of hiding it (#379, #75).

The ``custom_root`` node hides ``g(u, v) = 0`` behind a ``CustomCall``. Two
consequences follow from the equations being invisible, and these tests pin both
*and* their repair:

1. **JAX is pinned to the solve path.** ``_nl_expr_compiler`` raises
   ``UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")``, so a
   model containing the node falls back to the JAX evaluator. Lowered, the block
   is ordinary algebra and the tape takes it.
2. **No certificate is reachable.** A relaxation of an implicitly defined ``v``
   needs the defining equations; with only ``v = phi(u)`` there is nothing to
   relax, so ``_custom_call_reduced_admissible`` refuses the global path.

The sharpest statement of (2) is that the opaque node's *answer depends on the
Newton starting point* -- same box, same equations, different optimum -- because
which root Newton lands in is the only definition ``v`` has.
``test_opaque_node_answer_depends_on_x0_but_full_space_does_not`` pins that
falsification directly, so the motivation cannot silently rot.

Each subprocess driver prints a JAX module count; the paired opaque arm asserting
a NON-zero count is the vacuity control -- without it, "0 jax modules" could mean
the driver never solved anything.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import discopt.modeling as dm
import numpy as np
import pytest

# min -u  s.t.  v = sqrt(u), v <= 1.4, u in [1, 2]  ->  u = 1.96, obj = -1.96.
# The residual v**2 - u = 0 also admits v = -sqrt(u), for which v <= 1.4 is slack
# for every u, giving -2.0. Restricting v to a POSITIVE box picks the intended
# branch declaratively -- which is the whole point: the box is visible, an x0 is
# not.
_SQRT_BLOCK = """
import discopt.modeling as dm
m = dm.Model("sq")
u = m.continuous("u", lb=1.0, ub=2.0)
"""

_FULL_SPACE_TAIL = """
v = m.implicit(
    lambda U, V: [V[0] * V[0] - U[0]], [u], 1,
    formulation="full_space", bounds=(0.1, 3.0),
)
m.subject_to(v[0] <= 1.4)
m.minimize(-u)
"""

_OPAQUE_TAIL = """
v = m.implicit(lambda U, V: [V[0] * V[0] - U[0]], [u], 1, x0=[{x0}])
m.subject_to(v <= 1.4)
m.minimize(-u)
"""

_REPORT = """
r = m.solve(time_limit=120)
import sys
print("STATUS:" + str(r.status))
print("OBJ:" + repr(r.objective))
print("CERTIFIED:" + str(getattr(r, "gap_certified", None)))
print("JAXMODS:" + str(sum(1 for k in sys.modules if k == "jax" or k.startswith("jax."))))
"""


def _run_raw(src: str) -> dict:
    import os

    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(src)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=600,
    )
    assert out.returncode == 0, f"driver failed\nstdout={out.stdout}\nstderr={out.stderr[-3000:]}"
    parsed = {}
    for line in out.stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            parsed[k] = v
    return parsed


@pytest.mark.correctness
def test_full_space_lowering_solves_without_jax():
    """The lowered block solves on the tape: 0 jax modules, correct optimum."""
    res = _run_raw(_SQRT_BLOCK + _FULL_SPACE_TAIL + _REPORT)
    assert res["STATUS"] == "optimal", res
    assert res["JAXMODS"] == "0", f"full-space lowering still imported JAX: {res}"
    assert float(res["OBJ"]) == pytest.approx(-1.96, abs=1e-5), res

    # Vacuity control: the SAME problem through the opaque node must import JAX,
    # or "0" above is measuring an environment without JAX rather than the change.
    opaque = _run_raw(_SQRT_BLOCK + _OPAQUE_TAIL.format(x0=1.0) + _REPORT)
    # The opaque node runs the local-NLP path, which proves a feasible point and
    # not a global optimum, so it reports "feasible" (#998) — the status half of
    # the same forfeit that the next test pins as CERTIFIED == "False". The
    # lowered arm above keeps its genuine "optimal".
    assert opaque["STATUS"] == "feasible", opaque
    assert int(opaque["JAXMODS"]) > 0, (
        f"control did not import JAX, so the test is vacuous: {opaque}"
    )


@pytest.mark.correctness
def test_full_space_lowering_certifies_where_the_opaque_node_cannot():
    """Lowering restores the global certificate the CustomCall forfeits."""
    full = _run_raw(_SQRT_BLOCK + _FULL_SPACE_TAIL + _REPORT)
    opaque = _run_raw(_SQRT_BLOCK + _OPAQUE_TAIL.format(x0=1.0) + _REPORT)

    assert full["CERTIFIED"] == "True", f"lowered block did not certify: {full}"
    assert opaque["CERTIFIED"] == "False", (
        "the opaque node reported a certificate; if CustomCall models became "
        f"globally certifiable this test's premise needs revisiting: {opaque}"
    )


@pytest.mark.correctness
def test_opaque_node_answer_depends_on_x0_but_full_space_does_not():
    """The falsification that motivates the lowering, pinned.

    Same box, same equations: the opaque node returns a different optimum for a
    different Newton start, because ``x0`` silently selects the branch. The
    lowered model has both roots in the box and does not depend on ``x0`` at all.
    """
    pos = _run_raw(_SQRT_BLOCK + _OPAQUE_TAIL.format(x0=1.0) + _REPORT)
    neg = _run_raw(_SQRT_BLOCK + _OPAQUE_TAIL.format(x0=-1.0) + _REPORT)
    assert float(pos["OBJ"]) == pytest.approx(-1.96, abs=1e-5), pos
    assert float(neg["OBJ"]) == pytest.approx(-2.0, abs=1e-5), neg
    assert abs(float(pos["OBJ"]) - float(neg["OBJ"])) > 1e-3, (
        "the opaque node no longer depends on x0; this test's premise is stale"
    )

    # The lowered model has no x0 to depend on -- the branch is stated as a BOX,
    # which is visible to the relaxation where a starting point is not. Selecting
    # the OTHER root is then a declarative edit, and it moves the answer the way
    # the equations say it should.
    def _boxed(lo, hi):
        return (
            _SQRT_BLOCK
            + f"""
v = m.implicit(
    lambda U, V: [V[0] * V[0] - U[0]], [u], 1,
    formulation="full_space", bounds=({lo}, {hi}),
)
m.subject_to(v[0] <= 1.4)
m.minimize(-u)
"""
            + _REPORT
        )

    pos_box = _run_raw(_boxed(0.1, 3.0))  # v = +sqrt(u); v <= 1.4 binds at u = 1.96
    neg_box = _run_raw(_boxed(-3.0, -0.1))  # v = -sqrt(u); v <= 1.4 always slack
    assert float(pos_box["OBJ"]) == pytest.approx(-1.96, abs=1e-5), pos_box
    assert float(neg_box["OBJ"]) == pytest.approx(-2.0, abs=1e-5), neg_box
    assert pos_box["CERTIFIED"] == "True" and neg_box["CERTIFIED"] == "True", (pos_box, neg_box)


@pytest.mark.correctness
def test_nonlinear_multi_equation_block_solves_without_jax():
    """A 2x2 nonlinear block, not just the 1x1 square-root case."""
    src = """
import discopt.modeling as dm
m = dm.Model("cyc")
u = m.continuous("u", lb=0.5, ub=1.5)
v = m.implicit(
    lambda U, V: [
        V[0] - (0.3 * V[1] + 0.5 * U[0]),
        V[1] - (0.2 * V[0] * V[0] + 1.0),
    ],
    [u], 2, formulation="full_space", bounds=([-3.0, -3.0], [3.0, 3.0]),
)
m.minimize(v[0] + v[1] - u)
"""
    res = _run_raw(src + _REPORT)
    assert res["STATUS"] == "optimal", res
    assert res["JAXMODS"] == "0", f"2x2 nonlinear block still imported JAX: {res}"


@pytest.mark.smoke
def test_lowering_emits_one_variable_and_one_equation_per_unknown():
    m = dm.Model("shape")
    u = m.continuous("u", lb=0.0, ub=1.0)
    n_vars_before, n_cons_before = len(m._variables), len(m._constraints)
    v = m.implicit(
        lambda U, V: [V[0] - U[0], V[1] - V[0], V[2] - V[1] * V[1]],
        [u],
        3,
        formulation="full_space",
        bounds=(-5.0, 5.0),
    )
    assert len(m._variables) == n_vars_before + 1  # one VECTOR variable
    assert v.shape == (3,)
    assert len(m._constraints) == n_cons_before + 3  # one equation per unknown
    assert np.allclose(v.lb, -5.0) and np.allclose(v.ub, 5.0)


@pytest.mark.smoke
def test_residual_indexing_matches_the_custom_root_arm():
    """``u`` is flattened the same way in both arms, so one residual serves both.

    The ``custom_root`` arm hands ``residual`` a flat concatenation of the
    *raveled* inputs. A vector input plus a scalar input is the case that would
    catch a mismatch -- with only scalars every flattening looks alike.
    """
    m = dm.Model("flat")
    a = m.continuous("a", shape=(2,), lb=1.0, ub=2.0)
    b = m.continuous("b", lb=3.0, ub=4.0)
    seen = {}

    def residual(U, V):
        seen["n"] = len(U)
        # v0 = a0 + a1 + b : exercises every flattened slot, so a wrong order or
        # a dropped element changes the emitted equation.
        return [V[0] - (U[0] + U[1] + U[2])]

    v = m.implicit(residual, [a, b], 1, formulation="full_space", bounds=(0.0, 20.0))
    assert seen["n"] == 3, "u was not flattened to 3 scalar slots"

    m.minimize(v[0])
    r = m.solve(time_limit=60)
    # min over a in [1,2]^2, b in [3,4]  ->  1 + 1 + 3 = 5
    assert r.objective == pytest.approx(5.0, abs=1e-5), r.objective


@pytest.mark.smoke
def test_second_block_gets_a_distinct_name():
    """Two blocks in one model must not collide on the default variable name."""
    m = dm.Model("two")
    u = m.continuous("u", lb=0.0, ub=1.0)
    v1 = m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, formulation="full_space")
    v2 = m.implicit(lambda U, V: [V[0] - 2.0 * U[0]], [u], 1, formulation="full_space")
    assert v1.name != v2.name
    assert len({v.name for v in m._variables}) == len(m._variables)


@pytest.mark.smoke
def test_full_space_rejects_a_residual_of_the_wrong_length():
    m = dm.Model("bad")
    u = m.continuous("u", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="must return 2 entries, got 1"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 2, formulation="full_space")


@pytest.mark.smoke
def test_full_space_rejects_a_non_sequence_residual():
    m = dm.Model("bad2")
    u = m.continuous("u", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="sequence of 1 expressions"):
        m.implicit(lambda U, V: V[0] - U[0], [u], 1, formulation="full_space")


@pytest.mark.smoke
def test_inner_newton_knobs_are_refused_not_silently_dropped():
    """``x0`` / ``tol`` / ``max_iter`` steer an inner Newton solve this arm has not.

    Accepting them would be the worst outcome: the caller tunes a knob, nothing
    changes, and nothing says so. discopt has no per-variable start slot either,
    so a "recorded" ``x0`` would be read by nothing.
    """
    m = dm.Model("x0")
    u = m.continuous("u", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="x0= is not supported"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, x0=[0.5], formulation="full_space")
    with pytest.raises(ValueError, match="tol= does not apply"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, tol=1e-6, formulation="full_space")
    with pytest.raises(ValueError, match="max_iter= does not apply"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, max_iter=5, formulation="full_space")

    # The defaults must NOT trip the guard -- otherwise the arm is unusable.
    v = m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, formulation="full_space")
    assert v.shape == (1,)


@pytest.mark.smoke
def test_unknown_formulation_and_misplaced_bounds_are_refused():
    m = dm.Model("refuse")
    u = m.continuous("u", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="must be 'custom_root' or 'full_space'"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, formulation="reduced")
    # bounds= on the opaque arm has no variable to apply to. Refuse rather than
    # silently drop it (the node would then be unbounded and the caller would not
    # know).
    with pytest.raises(ValueError, match="only supported with formulation='full_space'"):
        m.implicit(lambda U, V: [V[0] - U[0]], [u], 1, bounds=(0.0, 1.0))
