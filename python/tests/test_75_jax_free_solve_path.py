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


@pytest.mark.slow
def test_eager_imports_stay_jax_free():
    """``DISCOPT_EAGER_IMPORTS=1`` must not reintroduce JAX.

    Found by review, not by the tests above: every other test here runs with the
    eager path OFF (its default), so none of them touches this list. It named
    ``jax.numpy`` as its first entry -- a documented, supported configuration in
    which the #75 result was simply false. Measured on the merged tree before the
    fix: **210** JAX modules on the same solve that imports 0 by default.

    The list is *by name*, so no amount of removing ``import jax`` statements from
    the solve path can fix it and no import-site grep would find it. That makes it
    exactly the class of leak this file exists to catch, and it is pinned here
    rather than in the ``__init__`` unit tests so it lives beside the assertion it
    protects.
    """
    res = _run(MODELS["nlp_exp_log"], {"DISCOPT_EAGER_IMPORTS": "1"})
    assert res["JAXMODS"] == "0", (
        f"DISCOPT_EAGER_IMPORTS=1 imported {res['JAXMODS']} jax modules "
        f"({res.get('LEAKED', '?')}) -- the eager list has reacquired a jax entry"
    )


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


# --------------------------------------------------------------------------
# QP standard-form extraction (#75): ``extract_qp_data`` falls through to the
# autodiff extractor whenever ``_extract_qp_data_from_repr``'s numeric probe
# cannot reproduce the objective. That fallback imported ``jax`` unconditionally,
# so an ordinary MIQP with no nonlinear constraint anywhere pulled in 210 jax
# modules on an all-defaults solve (found on ``chimera_mis-01`` via
# ``qubo_local_search`` -> ``extract_qp_data``). These pin the JAX-free route and
# its numeric agreement with the JAX one.
# --------------------------------------------------------------------------

# f(x, y) = 2x^2 + 3y^2 + xy - 4x + 1  ->  Q = [[4, 1], [1, 6]], c = (-4, 0), d = 1
_QP_MODEL = """
m = Model()
x = m.continuous("x", lb=-5.0, ub=5.0)
y = m.continuous("y", lb=-5.0, ub=5.0)
m.minimize(2.0 * x * x + 3.0 * y * y + x * y - 4.0 * x + 1.0)
m.subject_to(x + y >= 1.0)
"""

_QP_FALLBACK_DRIVER = (
    """
import sys
import numpy as np
from discopt import Model
import discopt._jax.problem_classifier as pc

# extract_qp_data is a ladder: repr(builder) -> algebraic -> repr(probe) ->
# autodiff. Only the last rung is under test, so BOTH earlier rungs must refuse
# -- the first draft patched only the repr extractor, the algebraic one answered,
# and the fallback never ran. The per-rung counters below are what caught that
# (CLAUDE.md §6: prove the probe fired).
#
# On the real instance this is not synthetic: repr(probe) refuses on its own when
# its numeric probe cannot reproduce the objective.
_reached = {"repr": 0, "alg": 0}


def _refuse_repr(model):
    _reached["repr"] += 1
    raise pc._NotQuadraticError("test: forcing the autodiff fallback")


def _refuse_alg(model):
    _reached["alg"] += 1
    raise RuntimeError("test: forcing the autodiff fallback")


pc._extract_qp_data_from_repr = _refuse_repr
pc.extract_qp_data_algebraic = _refuse_alg
"""
    + _QP_MODEL
    + """
qp = pc.extract_qp_data(m)
Q = pc.dense_Q(qp.Q)
c = np.asarray(qp.c, dtype=float)
print("REFUSALS_REPR:" + str(_reached["repr"]))
print("REFUSALS_ALG:" + str(_reached["alg"]))
print("Q00:" + repr(float(Q[0, 0])))
print("Q01:" + repr(float(Q[0, 1])))
print("Q11:" + repr(float(Q[1, 1])))
print("C0:" + repr(float(c[0])))
print("D:" + repr(float(qp.obj_const)))
leaked = sorted(k for k in sys.modules if k == "jax" or k.startswith("jax."))
print("JAXMODS:" + str(len(leaked)))
if leaked:
    print("LEAKED:" + ",".join(leaked[:8]))
"""
)

_QP_AGREEMENT_DRIVER = (
    """
import sys
import numpy as np
from discopt import Model
import discopt._jax.problem_classifier as pc
"""
    + _QP_MODEL
    + """
n = sum(v.size for v in m._variables)

# Tape arm FIRST, so "tape works" cannot be jax quietly doing the work.
tape = pc._qp_terms_tape(m, n)
print("TAPE_NONE:" + str(tape is None))
mid = sorted(k for k in sys.modules if k == "jax" or k.startswith("jax."))
print("JAXMODS_AFTER_TAPE:" + str(len(mid)))

Qt, ct, dt = tape
Qj, cj, dj = pc._qp_terms_jax(m, n)
_after = [k for k in sys.modules if k == "jax" or k.startswith("jax.")]
print("JAXMODS_AFTER_JAX:" + str(len(_after)))
print("DQ:" + repr(float(np.max(np.abs(np.asarray(Qt) - np.asarray(Qj))))))
print("DC:" + repr(float(np.max(np.abs(np.asarray(ct) - np.asarray(cj))))))
print("DD:" + repr(abs(float(dt) - float(dj))))
"""
)


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


# Marked ``correctness`` and NOT ``slow`` on purpose. Every CI lane's ``-m``
# expression carries ``not slow``, except the nightly one which needs
# ``correctness and slow`` -- so the 25 ``slow``-only tests above are selected by
# NO lane. These two run in ~1.7 s, so ``correctness`` puts them in the
# "Python correctness (fast subset)" lane, where a reintroduced jax import
# actually fails a build instead of waiting for someone to run pytest by hand.
@pytest.mark.correctness
def test_qp_extraction_fallback_stays_jax_free():
    """``extract_qp_data``'s autodiff fallback must not import JAX."""
    res = _run_raw(_QP_FALLBACK_DRIVER)
    # Vacuity control: the forced refusal must actually have fired, or the
    # assertions below say nothing about the fallback.
    assert res["REFUSALS_ALG"] == "1", f"the algebraic extractor was not reached: {res}"
    assert res["REFUSALS_REPR"] != "0", f"the repr extractor was not reached: {res}"
    assert res["JAXMODS"] == "0", (
        f"QP autodiff fallback imported {res['JAXMODS']} jax modules "
        f"({res.get('LEAKED', '?')}) -- #75 regression"
    )
    # The fallback must still be *right*, not merely JAX-free.
    assert float(res["Q00"]) == pytest.approx(4.0)
    assert float(res["Q01"]) == pytest.approx(1.0)
    assert float(res["Q11"]) == pytest.approx(6.0)
    assert float(res["C0"]) == pytest.approx(-4.0)
    assert float(res["D"]) == pytest.approx(1.0)


@pytest.mark.correctness
def test_qp_extraction_tape_matches_jax():
    """The tape and JAX differentiators must agree on ``(Q, c, d)``."""
    res = _run_raw(_QP_AGREEMENT_DRIVER)
    assert res["TAPE_NONE"] == "False", "tape could not represent a plain QP"
    assert res["JAXMODS_AFTER_TAPE"] == "0", (
        f"the tape arm imported jax ({res['JAXMODS_AFTER_TAPE']} modules) -- "
        "it is not a JAX-free route"
    )
    # Control: the JAX arm must really run, else the agreement is vacuous.
    assert int(res["JAXMODS_AFTER_JAX"]) > 0, "the JAX arm imported no jax modules"
    assert float(res["DQ"]) == pytest.approx(0.0, abs=1e-9)
    assert float(res["DC"]) == pytest.approx(0.0, abs=1e-9)
    assert float(res["DD"]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Structure cuts (#75): the symbolic constraint-cut derivation used JAX twice --
# ``jax.grad`` for a *univariate* tangent slope and ``lambdify(modules="jax")``
# for a 200-sample Jensen convexity check. Neither needed autodiff, and together
# they pulled 211 jax modules onto a default gas-network solve, where
# ``structure_cuts`` is ON by default. Both now go through SymPy + numpy.
# --------------------------------------------------------------------------

_STRUCTURE_CUT_DRIVER = """
import sys
from discopt._jax.symbolic import cut_recognizer as R
from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp

m = build_gas_network_minlp()

# Exercise the derivation directly so the measurement below is about the cut
# machinery, not merely about a solve that might never have recognized anything.
cuts = R.recognize_and_derive_cuts(build_gas_network_minlp())
print("CUTS:" + str(len(cuts)))
tangents = 0
for c in cuts:
    lo, hi = R.under_domain_of(c.underestimator)
    v, g = c.underestimator.tangent_cut(0.5 * (lo + hi))
    tangents += 1
    print("TANGENT_V:" + repr(float(v)))
    print("TANGENT_G:" + repr(float(g)))
    print("CONVEX:" + str(bool(c.underestimator.is_convex)))
print("TANGENTS:" + str(tangents))

r = m.solve(time_limit=90, gap_tolerance=1e-4)
print("STATUS:" + str(r.status))
print("NODES:" + str(r.node_count))
print("BOUND:" + repr(float(r.bound)) if r.bound is not None else "BOUND:None")

leaked = sorted(k for k in sys.modules if k == "jax" or k.startswith("jax."))
print("JAXMODS:" + str(len(leaked)))
print("LEAKED:" + ",".join(leaked[:6]))
"""


@pytest.mark.correctness
def test_structure_cuts_stay_jax_free():
    """Deriving and applying structure cuts must not import JAX.

    Both JAX uses in ``constraint_cuts`` were removed for #75: the tangent slope
    now comes from ``sp.diff`` (exact, since ``h`` is univariate) and the Jensen
    check runs on numpy arrays. Measured on this model before the swap: 211 jax
    modules; after: 0, with ``node_count`` unchanged.
    """
    res = _run_raw(_STRUCTURE_CUT_DRIVER)
    # Vacuity controls first: a recognizer that matched nothing, or a cut whose
    # tangent generator was never called, would make the JAXMODS assertion below
    # pass while proving nothing about the code that used to import jax.
    assert int(res["CUTS"]) == 2, f"the gas-network recognizer did not fire: {res}"
    assert int(res["TANGENTS"]) == 2, f"tangent_cut was never exercised: {res}"
    assert res["CONVEX"] == "True", f"underestimator lost its convexity verdict: {res}"
    assert res["JAXMODS"] == "0", (
        f"structure-cut derivation imported {res['JAXMODS']} jax modules "
        f"({res.get('LEAKED', '?')}) -- #75 regression"
    )
    # Still correct, not merely JAX-free: the cut must still close the gap.
    assert res["STATUS"] == "optimal", f"structure cuts stopped certifying: {res}"


def test_tangent_slope_matches_the_jax_reference():
    """``dh_fn`` must reproduce the slope the ``jax.grad`` arm produced.

    Reference values are the shipped JAX numbers, captured on the real
    gas-network cut at the domain midpoint before the swap. Symbolic
    differentiation agreed to 2.2e-16 (one ulp) over 401 points; the tolerance
    here is loose enough for that and far tighter than any tolerance that would
    let a genuinely different derivative through.
    """
    from discopt._jax.symbolic import cut_recognizer as R
    from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp

    cuts = R.recognize_and_derive_cuts(build_gas_network_minlp())
    assert cuts, "recognizer produced no cuts -- nothing to compare"
    under = cuts[0].underestimator
    v, g = under.tangent_cut(37.0)
    assert v == pytest.approx(1.1938449576366978, rel=1e-12)
    assert g == pytest.approx(0.1022698440660875, rel=1e-12)
