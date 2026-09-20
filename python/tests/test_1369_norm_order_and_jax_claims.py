"""Norm-order validation, and the JAX claims the docs make, pinned as tests.

Two findings from a documentation review, fixed together because they are the same
mistake in two places — a statement that was true of the common case, asserted
unconditionally, with nothing executable holding it to account.

**1. `dm.norm` accepted orders it could not compile.** ``norm(ord=...)`` built the
node name as ``f"norm{ord}"`` with no validation, so ``dm.norm(X, "fro")`` produced
``"normfro"`` and then failed far from the call site, differently depending on which
consumer ran first::

    _relax/dag_compiler.py   ValueError: Unsupported norm order: 'normfro'
    export/_arrays.py        ValueError: could not convert string to float: 'fro'

Neither names what the caller passed. ``dm.norm`` is the only place that knows, so
it is the only place that can say so (CLAUDE.md §3 — refuse loudly rather than fail
cheaply later).

**2. "A default solve does not import JAX" was stated unconditionally** in README,
`docs/intro.md` and CLAUDE.md, two sentences after each of them documents the
fallback that makes it false. It holds for the tape-representable majority; it does
not hold for a model with no tape opcode, which loads the legacy JAX evaluator *on
the default path, by design*. The measurement behind the claim was real but taken
over eight tape-representable corpus instances, which cannot license the general
form. ``test_the_fallback_puts_jax_on_the_default_path`` is the counter-example, so
the corrected wording cannot silently revert.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. norm order validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ord_value", ["fro", "nuc", "Fro", "2", None, object()])
def test_a_non_numeric_norm_order_is_refused_at_the_call_site(ord_value):
    """The refusal must name the order the CALLER passed, not a synthesized node.

    ``"2"`` is in this list deliberately: a numeric *string* is accepted by
    ``float()`` and so remains supported; every other non-numeric order is not.
    """
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    if ord_value == "2":
        assert dm.norm(x, ord_value).func_name == "norm2"
        return
    with pytest.raises(ValueError, match="unsupported norm order"):
        dm.norm(x, ord_value)


def test_the_matrix_norm_refusal_names_a_way_forward():
    """A refusal that does not say what to do instead just moves the dead end."""
    m = dm.Model()
    X = m.continuous("X", shape=(3, 3), lb=-2.0, ub=2.0)
    with pytest.raises(ValueError) as exc:
        dm.norm(X, "fro")
    msg = str(exc.value)
    assert "'fro'" in msg, "the message must quote what the caller passed"
    assert "dm.sum" in msg, "the message must offer an explicit construction"


@pytest.mark.parametrize("ord_value", [0, 0.5, -1, -2.0, float("nan")])
def test_a_p_below_one_is_refused(ord_value):
    """The relaxation layer's envelope comes from ``||x||_inf <= ||x||_p <= ||x||_1``,
    which holds only for ``p >= 1``. Accepting ``p < 1`` would hand the compiler a
    node whose bounds are not valid — a soundness question, not a usability one.
    ``nan`` is covered by the same comparison and must not slip through."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    with pytest.raises(ValueError, match="unsupported norm order"):
        dm.norm(x, ord_value)


@pytest.mark.parametrize(
    "ord_value,expected",
    [
        (1, "norm1"),
        (2, "norm2"),
        (3, "norm3"),
        (1.5, "norm1.5"),
        (float("inf"), "norminf"),
        ("inf", "norminf"),
        ("Inf", "norminf"),
    ],
)
def test_supported_orders_are_unchanged(ord_value, expected):
    """The validation must not narrow what already worked."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    assert dm.norm(x, ord_value).func_name == expected


def test_a_supported_norm_still_solves():
    """End to end, so the refusal cannot be mistaken for a working gate on a path
    that no longer runs. Optimum of ``min ||x||_1`` s.t. ``sum(x) >= 1`` is 1.0."""
    m = dm.Model("norm_solves")
    x = m.continuous("x", shape=(3,), lb=-2.0, ub=2.0)
    m.minimize(dm.norm(x, 1))
    m.subject_to(dm.sum(x) >= 1.0)
    r = m.solve(time_limit=60)
    assert r.objective == pytest.approx(1.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent (min)"


# --------------------------------------------------------------------------- #
# 2. the JAX claim the docs make
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_a_tape_representable_solve_imports_no_jax():
    """The claim the docs make, in the form in which it is TRUE.

    Run in a subprocess: `sys.modules` is process-global and another test in the
    session may already have imported jax, which would make an in-process assertion
    pass or fail for reasons having nothing to do with this solve (CLAUDE.md §8 —
    know which code you actually loaded).
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import discopt.modeling as dm\n"
        "m = dm.Model('tape')\n"
        "x = m.continuous('x', lb=0.0, ub=4.0)\n"
        "y = m.integer('y', lb=0, ub=4)\n"
        "z = m.continuous('z', lb=0.0, ub=50.0)\n"
        "m.minimize(z + 2 * y)\n"
        "m.subject_to((x - 3.0) ** 2 <= z)\n"
        "m.subject_to(x + y >= 3)\n"
        "r = m.solve(time_limit=30)\n"
        "assert r.objective is not None, 'the probe never solved'\n"
        "n = len([k for k in sys.modules if k == 'jax' or k.startswith('jax.')])\n"
        "print('JAXCOUNT', n)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("JAXCOUNT")]
    assert line, f"probe printed no count; stdout={out.stdout[-500:]}"
    assert int(line[0].split()[1]) == 0, "a tape-representable solve must not import JAX"


@pytest.mark.slow
def test_the_fallback_puts_jax_on_the_default_path():
    """The counter-example, which is why the docs no longer say "a default solve
    never imports JAX".

    A matrix norm has no tape opcode, so ``build_evaluator`` calls the JAX factory
    and `_relax/nlp_evaluator` — which imports jax at module scope — is loaded. This
    is deliberate behaviour, not a leak; what was wrong was the documentation
    asserting the opposite two sentences after describing it.

    Asserting the *mechanism* (the evaluator module is loaded) rather than a module
    count, so the test says what happened rather than merely that a number moved.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import discopt.modeling as dm\n"
        "assert not [k for k in sys.modules if k == 'jax' or k.startswith('jax.')], (\n"
        "    'jax was already imported before the solve; the probe proves nothing')\n"
        "m = dm.Model('fallback')\n"
        "X = m.continuous('X', shape=(3, 3), lb=-2.0, ub=2.0)\n"
        "m.minimize(dm.norm(X, 2))\n"
        "m.subject_to(dm.sum(X) >= 1.0)\n"
        "m.solve(time_limit=30)\n"
        "n = len([k for k in sys.modules if k == 'jax' or k.startswith('jax.')])\n"
        "ev = 'discopt._relax.nlp_evaluator' in sys.modules\n"
        "print('JAXCOUNT', n, 'EVALUATOR', ev)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("JAXCOUNT")]
    assert line, f"probe printed no count; stdout={out.stdout[-500:]}"
    _, count, _, evaluator = line[0].split()
    assert evaluator == "True", (
        "the JAX evaluator was not loaded, so the documented fallback did not fire "
        "and this test is no longer the counter-example it claims to be"
    )
    assert int(count) > 0, "the fallback loaded the evaluator but no jax modules"
