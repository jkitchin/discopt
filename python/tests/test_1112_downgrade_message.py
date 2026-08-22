"""Issue #1112: the McCormick LP downgrade must explain itself, not misdirect.

A model whose nonlinearity lives inside a ``dm.custom``/``CustomCall`` node has no
*lifted* relaxation — the function body is opaque to the DAG walker by
construction — so ``MccormickLPRelaxer.has_relaxable_nonlinearity`` is False and
``solve_model`` discards the LP relaxer, leaving ``_mc_mode == "nlp"``. The
issue-#120 soundness guard then demotes that to ``"none"``. The demotion is
correct and is NOT what this file tests.

What was wrong is the message. It told the user to

    Use mccormick_bounds='lp' for a valid spatial relaxation

which is (a) circular when the user passed exactly that, and (b) emitted even on
models where the *reduced-space* engine is about to supply a real spatial bound,
where there is no fallback to announce at all.

REACHABILITY, established while writing this file and pinned by
``test_untraceable_custom_model_never_reaches_the_guard``: a CustomCall model
reaches the #120 guard **only** with reduced-space bounding forced on. A
non-admissible one returns early at ``solver.py`` via
``_withhold_local_optimality_certificate`` and never gets there. That is why the
fix is a single suppression rather than a re-worded warning — a branch explaining
the opaque cause to a non-reduced-space caller would be dead code (§3).

MEASUREMENT THAT SCOPED THIS FILE (recorded so it is not re-litigated). #1112's
section (b) claims "the node bound it actually gets is alphaBB". That is false: on
an MCBox-traceable 3-DOF CustomCall model the reduced engine is active and bounds
0.3275 against an incumbent of 0.3354 — a 2.3% gap, not an alphaBB-grade bound.
And the ``discopt.nn`` rows in the issue's table are a DIFFERENT bug: those models
report ``has_relaxable_nonlinearity is True`` and are never downgraded (their loose
bound is the missing spanning-box tanh envelope). ``test_nn_style_model_is_not_
downgraded`` pins that distinction so the two are never conflated again.

Every test here counts the assertions it actually executed and the module exits
non-zero if any count is zero (CLAUDE.md §6): a log-scraping test whose model
silently stops reaching the code path under test degrades into a no-op that passes.
No test swallows an exception (§7).
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.mcbox import MCBox
from discopt._relax.mccormick_lp import MccormickLPRelaxer

pytestmark = [pytest.mark.relaxation]

#: The circular advice #1112 is about.
CIRCULAR = "Use mccormick_bounds='lp'"
#: Emitted when the reduced-space engine takes over.
REDUCED_ACTIVE = "reduced-space McCormick per-node bounding active"

#: Executed-assertion tally (§6). Incremented by ``_check``; asserted non-zero by
#: ``test_probes_actually_fired``, which pytest runs last in file order.
_CHECKS = {"n": 0}


def _check(condition: bool, message: str) -> None:
    """An assertion that RECORDS that it ran, so a silent no-op cannot pass."""
    _CHECKS["n"] += 1
    assert condition, message


def _mexp(x):
    """``exp`` dispatched to MCBox for the relaxation, numpy for the value path.

    A raw ``jnp.exp`` does NOT trace through MCBox — the model would be rejected as
    reduced-inadmissible and the test would exercise a different branch than the one
    it names. This is the canonical idiom from
    ``docs/notebooks/reduced_space_customcall.md``.
    """
    return x.exp() if isinstance(x, MCBox) else np.exp(x)


def _traceable_custom_model():
    """A nonconvex CustomCall model that IS reduced-space admissible.

    Scalar leaves only — a non-scalar variable leaf is rejected by the reduced
    gate, which would again route the test somewhere other than where it claims.
    """
    m = dm.Model("traceable")
    x = m.continuous("x", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
    # ALL the nonlinearity is inside the opaque node — a bilinear term left
    # outside it would give the LP relaxer something to bound, the discard would
    # not happen, and the guard under test would never fire.
    f = dm.custom(lambda a, b: a * _mexp(-b) + b * _mexp(-a), name="f")
    m.minimize(f(x[0], x[1]))
    m.subject_to(x[0] + x[1] >= 1.0)
    return m


def _opaque_untraceable_model():
    """A CustomCall model whose body does NOT trace: a RAW ``jnp`` intrinsic.

    ``jnp.exp`` evaluates fine on the local-NLP value path but has no MCBox
    dispatch, so the reduced gate declines. (``np.exp`` would be the wrong choice
    here — it raises ``TracerArrayConversionError`` on the JAX value path too, so
    the model would be broken end to end rather than merely reduced-inadmissible,
    and the test would prove something else.)
    """
    import jax.numpy as jnp

    m = dm.Model("opaque")
    x = m.continuous("x", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
    f = dm.custom(lambda a, b: jnp.exp(-a) * b, name="g")
    m.minimize(f(x[0], x[1]))
    m.subject_to(x[0] + x[1] >= 1.0)
    return m


def _factorable_nonconvex_model():
    """A plain factorable nonconvex model — the guard's PRE-EXISTING arm."""
    m = dm.Model("factorable")
    x = m.continuous("x", lb=0.1, ub=2.0)
    y = m.continuous("y", lb=0.1, ub=2.0)
    m.minimize(x * y - dm.exp(x))
    m.subject_to(x + y >= 1.0)
    return m


def _nn_style_tanh_model():
    """A one-hidden-layer tanh network written out algebraically."""
    m = dm.Model("nn")
    x = m.continuous("x", 2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    h = [dm.tanh(0.7 * x[0] - 0.4 * x[1] + 0.1), dm.tanh(-0.3 * x[0] + 0.9 * x[1])]
    m.minimize(1.3 * h[0] - 0.8 * h[1])
    return m


def _solve_capturing(model, **kwargs):
    """Solve and return ``(result, [log messages])`` from the ``discopt`` logger."""
    records: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Sink(level=logging.DEBUG)
    log = logging.getLogger("discopt")
    prev_level, prev_prop = log.level, log.propagate
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        result = model.solve(**kwargs)
    finally:
        log.removeHandler(handler)
        log.setLevel(prev_level)
        log.propagate = prev_prop
    return result, records


# --------------------------------------------------------------------------- #
# 1. Reduced-space active ⇒ the circular advice must be gone, bound still sound
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reduced_solve():
    """One solve of the traceable CustomCall model, shared by the tests below."""
    return _solve_capturing(_traceable_custom_model(), mccormick_bounds="lp", time_limit=25)


def test_reduced_space_solve_does_not_recommend_the_flag_already_passed(reduced_solve):
    """The headline #1112 defect.

    With ``mccormick_bounds="lp"`` on a traceable CustomCall model, the reduced
    engine takes over and supplies the node bound. Recommending ``'lp'`` there is
    both circular and wrong about what is happening."""
    _, logs = reduced_solve
    joined = "\n".join(logs)
    _check(
        not MccormickLPRelaxer(_traceable_custom_model()).has_relaxable_nonlinearity,
        "the model must reach the LP-relaxer DISCARD site — if it keeps its relaxer, "
        "the #120 guard is never consulted and this test proves nothing",
    )
    _check(
        any(REDUCED_ACTIVE in m for m in logs),
        "reduced-space bounding did not activate — this test is not exercising the "
        f"path it names. Logs:\n{joined}",
    )
    _check(
        any("issue #120" in m for m in logs),
        f"the #120 guard did not fire — nothing was suppressed. Logs:\n{joined}",
    )
    _check(
        not any(CIRCULAR in m for m in logs),
        f"the circular #120 advice is still emitted under reduced-space bounding:\n{joined}",
    )


def test_reduced_space_bound_is_sound_against_an_independent_feasible_point(reduced_solve):
    """Suppressing the message must not have touched the bound (message-only fix).

    Also the falsification of #1112's section (b), which assumes the node bound is
    alphaBB's. The check is against an upper bound this test computes ITSELF — the
    objective at an explicitly feasible point — rather than against the solver's own
    incumbent, which on this model may not appear inside the time limit. A test that
    silently skips when ``result.objective is None`` would assert nothing (§6)."""
    result, _ = reduced_solve
    bound = result.bound
    _check(bound is not None and np.isfinite(bound), f"no finite dual bound: {bound}")

    x0 = x1 = 0.5  # x0 + x1 == 1.0, so the >= 1.0 constraint holds
    true_opt_upper = x0 * np.exp(-x1) + x1 * np.exp(-x0)
    _check(
        bound <= true_opt_upper + 1e-6,
        f"dual bound {bound} exceeds f(0.5,0.5) = {true_opt_upper} at a FEASIBLE point — "
        "the bound is above the true optimum, which is a false bound",
    )
    if result.objective is not None:
        _check(
            bound <= result.objective + 1e-6,
            f"dual bound {bound} exceeds incumbent {result.objective}",
        )


# --------------------------------------------------------------------------- #
# 2. A non-admissible CustomCall model never reaches the guard at all
# --------------------------------------------------------------------------- #
def test_untraceable_custom_model_never_reaches_the_guard():
    """Why the fix is a suppression and not a re-wording.

    A raw ``jnp`` intrinsic in the body evaluates fine on the value path but does
    not trace through MCBox, so the model is reduced-inadmissible. Being pure
    continuous it does not raise; it takes the local-NLP path with the optimality
    certificate withheld, returning BEFORE the #120 guard. So no CustomCall model
    can reach that guard without ``_force_reduced_space``, and a branch there
    explaining the opaque cause would be unreachable."""
    model = _opaque_untraceable_model()
    _check(
        not MccormickLPRelaxer(model).has_relaxable_nonlinearity,
        "this model must be one the LP relaxer discards, or it proves nothing",
    )
    result, logs = _solve_capturing(model, mccormick_bounds="lp", time_limit=15)
    joined = "\n".join(logs)
    _check(
        not any("issue #120" in m for m in logs),
        f"the #120 guard fired on the withheld-certificate path — the reachability "
        f"argument behind the fix is wrong and the fix must be revisited:\n{joined}",
    )
    _check(
        not any(CIRCULAR in m for m in logs),
        f"circular advice on a model that cannot use 'lp' at all:\n{joined}",
    )
    _check(
        result.bound is None,
        f"the local-NLP path must withhold the certificate, got bound={result.bound}",
    )


# --------------------------------------------------------------------------- #
# 3. The pre-existing arm is NOT regressed
# --------------------------------------------------------------------------- #
def test_factorable_nonconvex_nlp_keeps_the_original_wording():
    """An explicit ``mccormick_bounds="nlp"`` on a factorable nonconvex model is
    the case the original #120 text was written for. It must be untouched — the
    fix narrows the message, it does not delete it."""
    _, logs = _solve_capturing(_factorable_nonconvex_model(), mccormick_bounds="nlp", time_limit=20)
    guard = [m for m in logs if "issue #120" in m]
    _check(bool(guard), "the #120 guard did not fire on the factorable nonconvex arm")
    _check(
        any(CIRCULAR in m for m in guard),
        f"the original advice was lost on the arm it is correct for:\n{guard}",
    )


# --------------------------------------------------------------------------- #
# 4. Regression pin: the discopt.nn rows in #1112's table are a DIFFERENT bug
# --------------------------------------------------------------------------- #
def test_nn_style_model_is_not_downgraded():
    """Measured, and pinned so the two bugs are never conflated again.

    A tanh network classifies as ``general_nl`` and keeps its LP relaxer, so it
    never reaches the #1112 downgrade at all. Its loose bound is the missing
    spanning-box S-envelope, which is separate work."""
    relaxer = MccormickLPRelaxer(_nn_style_tanh_model())
    _check(
        relaxer.has_relaxable_nonlinearity,
        "a tanh network no longer keeps its LP relaxer — if this flips, #1112's "
        "downgrade path and the S-envelope defect have genuinely merged and the "
        "scoping in this file's docstring must be redone",
    )


# --------------------------------------------------------------------------- #
# 5. §6: prove the probes fired
# --------------------------------------------------------------------------- #
#: Number of ``_check`` calls a complete run of this module must execute. Bumped
#: deliberately when a check is added; a DROP means a test stopped reaching the
#: code path it names and degraded into a silent pass.
EXPECTED_CHECKS = 13


@pytest.fixture(scope="module", autouse=True)
def _assert_probes_fired():
    """Teardown-time tally, so it holds under any test ordering.

    A file-order ``test_...`` would under-count whenever pytest-randomly (active by
    default here) shuffles it away from last, making the guard itself the flaky
    thing it exists to prevent."""
    yield
    assert _CHECKS["n"] == EXPECTED_CHECKS, (
        f"{_CHECKS['n']} checks executed, expected {EXPECTED_CHECKS}. Fewer means a "
        "test above stopped reaching the code path it names; more means a check was "
        "added without updating EXPECTED_CHECKS."
    )
