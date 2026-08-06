"""The acceptance criterion for #75: solving must not import JAX.

Stage 4. The goal of the whole effort is **zero ``import jax`` reachable from
``Model.solve()`` on the equation-oriented path**. This makes that executable
rather than documented.

Two rules govern how it is checked, both learned the hard way:

* **Assert on ``sys.modules``, never on a source grep.** ``dag_compiler.py:225``
  reaches JAX via ``__import__("jax")``, a string that contains no ``import jax``
  substring. A grep-based check passes while JAX loads.
* **Run in a subprocess.** The test suite imports JAX for other reasons, so an
  in-process check can only ever report "already imported" and would be a
  guaranteed false negative.

Scoped to the equation-oriented path, per the owner's 2026-08-03 scope decision:
``dm.custom``/``CustomCall`` keeps JAX by contract (relaxing an opaque callable
needs AD *through* it), so a CustomCall model is excluded here by construction
and asserted separately to still work.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pounce")

# Both backends are DEFAULT ON since the §5 panel, so the tape arm sets nothing:
# these tests must exercise what a user actually gets. Stage 3 removed the
# NLP-derivative trigger (nlp_evaluator.py:22) and Stage 2 the separation-tangent
# one (uniform_relax.py:813); either alone leaves the other importing JAX.
TAPE_ENV: dict = {}
JAX_ENV = {"DISCOPT_NLP_EVAL": "jax", "DISCOPT_SEPGRAD": "jax"}

# The plan asks for "a set spanning all 10 ProblemClass values". The original
# four covered NLP/MINLP/QCQP only; a review sweep found the other six pass too,
# so the gap was coverage, not behavior. They are cheap (each is one subprocess
# solve) and they are what stops a leak reappearing on, say, the MILP-only path.
LINEAR_AND_QUADRATIC = {
    "lp": """
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.subject_to(x + 2 * y <= 8)
        m.subject_to(3 * x + y <= 9)
        m.minimize(-x - y)
    """,
    "qp": """
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.subject_to(x + y >= 1)
        m.minimize(x * x + 2 * y * y + x * y - x)
    """,
    "qcp": """
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.subject_to(x * x + y * y <= 4.0)
        m.minimize(-x - y)
    """,
    "milp": """
        a = m.integer("a", lb=0, ub=10)
        b = m.integer("b", lb=0, ub=10)
        m.subject_to(a + 2 * b <= 7)
        m.subject_to(3 * a + b <= 9)
        m.minimize(-5 * a - 4 * b)
    """,
    "miqp": """
        x = m.continuous("x", lb=-5, ub=5)
        k = m.integer("k", lb=-3, ub=3)
        m.subject_to(x + k >= 1)
        m.minimize(x * x + 2.0 * k * k - 3 * x)
    """,
    "miqcp": """
        x = m.continuous("x", lb=-5, ub=5)
        k = m.integer("k", lb=0, ub=4)
        m.subject_to(x * x + k <= 9.0)
        m.minimize(-x - 2.0 * k)
    """,
}

MODELS = {
    # (builder body, expected status) spanning the nonlinear ProblemClass values.
    "nlp_exp_log": """
        x = m.continuous("x", lb=0.2, ub=4.0)
        y = m.continuous("y", lb=0.2, ub=4.0)
        m.subject_to(dm.exp(x) + y * y <= 20.0)
        m.subject_to(x * y >= 1.0)
        m.minimize(x * x + y + dm.log(x))
    """,
    "minlp_binary": """
        x = m.continuous("x", lb=0.2, ub=4.0)
        y = m.continuous("y", lb=0.2, ub=4.0)
        b = m.binary("b")
        m.subject_to(dm.exp(x) + y * y <= 20.0)
        m.subject_to(x + y + b >= 2.0)
        m.minimize(x * x + y + dm.log(x) + 2.0 * b)
    """,
    "minlp_integer_trig": """
        x = m.continuous("x", lb=0.3, ub=3.0)
        k = m.integer("k", lb=0, ub=3)
        m.subject_to(dm.sin(x) + dm.sqrt(x) <= 2.0)
        m.subject_to(x + k >= 1.5)
        m.minimize(dm.cos(x) + x * x + 0.5 * k)
    """,
    "bilinear_qcqp": """
        x = m.continuous("x", lb=0.1, ub=5.0)
        y = m.continuous("y", lb=0.1, ub=5.0)
        z = m.continuous("z", lb=0.1, ub=5.0)
        m.subject_to(x * y + y * z <= 12.0)
        m.subject_to(x * z >= 0.6)
        m.minimize(x * x + y * y + z * z - x * y)
    """,
}

MODELS.update(LINEAR_AND_QUADRATIC)

SCRIPT = """\
import sys
from discopt import Model
import discopt.modeling as dm

m = Model()
{body}
r = m.solve()

leaked = sorted(k for k in sys.modules if k == "jax" or k.startswith("jax."))
print("STATUS:" + str(r.status))
print("OBJ:" + repr(None if r.objective is None else float(r.objective)))
print("JAXMODS:" + str(len(leaked)))
if leaked:
    print("LEAKED:" + ",".join(leaked[:8]))
"""


def _run(body: str, env_extra: dict) -> dict:
    import os

    env = dict(os.environ)
    env.update(env_extra)
    src = SCRIPT.format(body=textwrap.dedent(body).strip())
    out = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, env=env, timeout=600
    )
    assert out.returncode == 0, f"solve failed\nstdout={out.stdout}\nstderr={out.stderr[-3000:]}"
    parsed = {}
    for line in out.stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            parsed[k] = v
    return parsed


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODELS))
def test_solve_never_imports_jax(name):
    """A solve under both tape backends must leave ``sys.modules`` JAX-free."""
    res = _run(MODELS[name], TAPE_ENV)
    assert res.get("STATUS") is not None, f"no status parsed: {res}"
    assert res["JAXMODS"] == "0", (
        f"{name}: solve imported {res['JAXMODS']} jax modules "
        f"({res.get('LEAKED', '?')}) -- the #75 goal is not met"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODELS))
def test_tape_backend_agrees_with_jax_on_the_objective(name):
    """JAX-free must not mean different answers.

    Not a bound-neutrality proof — the tape is bound-CHANGING and the §5 panel is
    the real gate — but a wrong objective here would make that panel pointless.
    """
    jax_res = _run(MODELS[name], JAX_ENV)
    tape_res = _run(MODELS[name], TAPE_ENV)

    assert jax_res["STATUS"] == tape_res["STATUS"], (
        f"{name}: status {jax_res['STATUS']} (jax) vs {tape_res['STATUS']} (tape)"
    )
    jo, to = eval(jax_res["OBJ"]), eval(tape_res["OBJ"])  # noqa: S307 - repr of a float or None
    if jo is None or to is None:
        assert jo == to, f"{name}: one arm found an incumbent and the other did not"
        return
    assert abs(jo - to) / max(1.0, abs(jo)) <= 1e-6, f"{name}: objective {jo} (jax) vs {to} (tape)"


@pytest.mark.unit
@pytest.mark.parametrize(
    "module,symbol",
    [
        # Each of these is JAX-FREE code that used to live in a JAX-importing
        # module, so merely importing the symbol pulled the whole stack onto an
        # otherwise JAX-free solve. Each was found by a corpus sweep, never by
        # reading the code, and each is one `import` away from coming back.
        ("discopt._alphabb_rigorous", "rigorous_alpha"),
        ("discopt._hessian_cost_model", "estimate_dense_obj_hessian_compile_s"),
        ("discopt._evaluator_cache", "evaluator_fingerprint"),
        ("discopt._nl_expr_compiler", "compile_to_nl_expr"),
        ("discopt._tape_nlp_evaluator", "TapeNLPEvaluator"),
    ],
)
def test_jax_free_helper_modules_do_not_import_jax(module, symbol):
    """These modules must stay importable without loading JAX.

    Subprocess, because the suite has already imported JAX by this point and an
    in-process check could only ever report "already loaded".
    """
    import subprocess
    import sys

    src = (
        f"import sys; from {module} import {symbol}; "
        f"assert '{symbol}' in dir(); "
        "leaked = sorted(k for k in sys.modules if k == 'jax' or k.startswith('jax.')); "
        "print('LEAKED' + str(len(leaked))); "
        "assert not leaked, leaked[:6]"
    )
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (
        f"{module}.{symbol} imports JAX\nstdout={out.stdout}\nstderr={out.stderr[-1500:]}"
    )
    assert "LEAKED0" in out.stdout


@pytest.mark.slow
@pytest.mark.parametrize(
    "env,label",
    [
        (JAX_ENV, "both opt-outs"),
        ({"DISCOPT_NLP_EVAL": "jax"}, "NLP opt-out only"),
        ({"DISCOPT_SEPGRAD": "jax"}, "sepgrad opt-out only"),
    ],
)
def test_opt_outs_still_reach_the_legacy_jax_path(env, label):
    """§5 graduation keeps the opt-out and the legacy path intact.

    Each opt-out must still route to JAX — otherwise the escape hatch is
    decorative. Asserting jax IS imported also proves these solves do real
    nonlinear work, so the JAX-free assertions above cannot be passing because
    nothing ran (CLAUDE.md §6).
    """
    res = _run(MODELS["nlp_exp_log"], env)
    assert int(res["JAXMODS"]) > 0, f"{label}: opt-out did not reach the JAX path"


@pytest.mark.unit
def test_cut_augmented_wrapper_over_a_tape_stays_jax_free():
    """`_AugmentedEvaluator` must not import JAX when it wraps a tape evaluator.

    Found by review, not by the tests above, and they *structurally* could not
    have found it: this wrapper is built on a live solve path (the cut-augmented
    node NLP, `solver.py`) but its `_cons_fn` has no in-tree consumer, so no
    end-to-end solve ever reads it. It used to `import jax.numpy` unconditionally
    -- measured at 210 JAX modules and a `jaxlib` return value on an otherwise
    JAX-free solve. "No consumer today" is one attribute access away from false,
    so the property is pinned directly rather than through `solve()`.
    """
    src = textwrap.dedent("""
        import sys
        import numpy as np
        from discopt import Model
        import discopt.modeling as dm
        from discopt import solver as S
        from discopt._jax.cutting_planes import CutPool, LinearCut
        from discopt._tape_nlp_evaluator import try_build

        m = Model()
        x = m.continuous("x", lb=0.1, ub=5.0)
        y = m.continuous("y", lb=0.1, ub=5.0)
        m.subject_to(x * y <= 12.0)
        m.minimize(x * x - 2.0 * y)
        tape = try_build(m)
        assert tape is not None, "fixture must lower to a tape or this proves nothing"

        pool = CutPool(max_cuts=10)
        pool.add(LinearCut(coeffs=np.array([1.0, 1.0]), rhs=9.0, sense="<="))
        aug = S._AugmentedEvaluator(tape, pool)

        fn = aug._cons_fn                      # the property that used to leak
        got = fn(np.array([1.0, 2.0]))
        base = np.asarray(tape.evaluate_constraints(np.array([1.0, 2.0])))
        assert got.shape[0] == base.shape[0] + 1, (got.shape, base.shape)
        assert np.allclose(got[:-1], base), (got, base)
        assert np.isclose(got[-1], 1.0 + 2.0 - 9.0), got[-1]
        assert isinstance(got, np.ndarray), type(got)

        leaked = sorted(k for k in sys.modules if k == "jax" or k.startswith("jax."))
        print("JAXMODS:" + str(len(leaked)))
        print("LEAKED:" + ",".join(leaked[:8]))
    """)
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"probe failed\nstdout={out.stdout}\nstderr={out.stderr[-2000:]}"
    assert "JAXMODS:0" in out.stdout, (
        f"the cut-augmented wrapper imported JAX over a tape evaluator: {out.stdout}"
    )


@pytest.mark.slow
def test_jax_arm_of_the_same_check_does_import_jax():
    """The control: with the opt-outs, JAX *is* imported.

    Without this, a bug that silently stopped the solve from doing any nonlinear
    work would make every assertion above pass for the wrong reason (CLAUDE.md §6
    — prove the probe fires).
    """
    res = _run(MODELS["nlp_exp_log"], JAX_ENV)
    assert int(res["JAXMODS"]) > 0, (
        "the JAX arm imported no jax modules; this check cannot distinguish "
        "'tape works' from 'nothing ran'"
    )
