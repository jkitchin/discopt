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

# Both backends must be on: Stage 3 removed the NLP-derivative trigger
# (nlp_evaluator.py:22) and Stage 2 removed the separation-tangent one
# (uniform_relax.py:813). Either alone leaves the other importing JAX.
TAPE_ENV = {"DISCOPT_NLP_EVAL": "tape", "DISCOPT_SEPGRAD": "tape"}

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
    jax_res = _run(MODELS[name], {"DISCOPT_NLP_EVAL": "jax", "DISCOPT_SEPGRAD": "jax"})
    tape_res = _run(MODELS[name], TAPE_ENV)

    assert jax_res["STATUS"] == tape_res["STATUS"], (
        f"{name}: status {jax_res['STATUS']} (jax) vs {tape_res['STATUS']} (tape)"
    )
    jo, to = eval(jax_res["OBJ"]), eval(tape_res["OBJ"])  # noqa: S307 - repr of a float or None
    if jo is None or to is None:
        assert jo == to, f"{name}: one arm found an incumbent and the other did not"
        return
    assert abs(jo - to) / max(1.0, abs(jo)) <= 1e-6, f"{name}: objective {jo} (jax) vs {to} (tape)"


@pytest.mark.slow
def test_jax_arm_of_the_same_check_does_import_jax():
    """The control: without the flags, JAX *is* imported.

    Without this, a bug that silently stopped the solve from doing any nonlinear
    work would make every assertion above pass for the wrong reason (CLAUDE.md §6
    — prove the probe fires).
    """
    res = _run(MODELS["nlp_exp_log"], {"DISCOPT_NLP_EVAL": "jax", "DISCOPT_SEPGRAD": "jax"})
    assert int(res["JAXMODS"]) > 0, (
        "the JAX arm imported no jax modules; this check cannot distinguish "
        "'tape works' from 'nothing ran'"
    )
