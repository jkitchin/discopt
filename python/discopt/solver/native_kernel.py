"""The native Rust spatial-B&B kernel: engagement gate, seeding, and driver.

Extracted verbatim from ``discopt.solver`` by consolidation-plan Card 4b — a
pure move, no behaviour change (Regime N). The cluster is self-contained: eight
functions and three constants that decide *whether* the #764 native kernel runs,
build and rigorously verify its incumbent seed, and drive one attempt, returning
``None`` on any incomplete exit so the caller falls back to the trusted Python
spatial loop.

Why this cluster moved first: it is the only engine-adjacent group in
``solve_model``'s file whose dependency closure reaches exactly one solver-level
helper (``_unpack_solution``, imported lazily below). Everything else measured in
the Card 4b coupling census pulls in 10-44 module-level names and needs a shared
leaf-helper layer first — see the plan's §6 entry.

**Monkeypatch note.** ``_native_kernel_seed`` and ``_native_kernel_seed_candidates``
are deliberately NOT re-exported from ``discopt.solver``. Tests patch them, and a
re-export would let ``monkeypatch.setattr(discopt.solver, ...)`` succeed while the
in-module call site kept using the real function — a patch that silently does
nothing, which is exactly the CLAUDE.md §6 failure mode. Without the re-export the
stale patch raises ``AttributeError`` instead.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

import numpy as np

from discopt._env import env_bool
from discopt.constants import STARTING_POINT_CLIP as _SPC
from discopt.modeling.core import SolveResult

logger = logging.getLogger(__name__)

__all__ = [
    "_native_spatial_kernel_enabled",
    "_native_kernel_feature_safe",
    "kernel_engagement_stats",
    "reset_kernel_engagement_stats",
    "_native_kernel_verify_point",
    "_native_seed_bracket",
    "_native_kernel_seed_candidates",
    "_native_kernel_seed",
    "_try_native_spatial_kernel",
    "_native_nlp_enabled",
    "_NATIVE_SEED_HEURISTIC_S",
    "_NATIVE_SEED_MAX_FREE_INTEGERS",
    "_NLP_NATIVE_DEFAULT",
]


# --------------------------------------------------------------------------- #
# Engagement instrumentation (consolidation plan Phase 5, Step 1)             #
#                                                                             #
# Card 3c measured "2 producer calls, 0 served" and could not say why. Three   #
# distinct things can stop this kernel serving a solve, and a census that      #
# cannot tell them apart ranks nothing:                                        #
#                                                                             #
#   1. the solve never reaches the gate at all (another route dispatched);     #
#   2. :func:`_native_kernel_feature_safe` declines (a Python-engine contract   #
#      the kernel does not fill);                                              #
#   3. the producer declines (a relaxation feature the kernel cannot build) or  #
#      the driver rejects the kernel's own answer (verification, status).       #
#                                                                             #
# Case 3's producer half is instrumented in                                    #
# ``discopt._jax.spatial_producer.producer_stats``; cases 2 and the driver half #
# of 3 are instrumented here. Regime N: every recorder is a dict write on a     #
# thread-local that nothing in the solve path reads, and every branch returns   #
# exactly what it returned before.                                             #
# --------------------------------------------------------------------------- #
class _EngagementStats(threading.local):
    def __init__(self) -> None:  # pragma: no cover - trivial
        self.gate_calls = 0
        self.gate_safe = 0
        self.gate_reasons: dict[str, int] = {}
        self.attempts = 0
        self.served = 0
        self.outcomes: dict[str, int] = {}
        self.last_gate_reason: Optional[str] = None
        self.last_outcome: Optional[str] = None


_ENGAGEMENT = _EngagementStats()


def _eng() -> _EngagementStats:
    if not hasattr(_ENGAGEMENT, "gate_calls"):  # pragma: no cover - fresh thread
        _ENGAGEMENT.__init__()
    return _ENGAGEMENT


def reset_kernel_engagement_stats() -> None:
    """Clear this thread's native-kernel engagement counters."""
    _eng().__init__()


def kernel_engagement_stats() -> dict:
    """This thread's native-kernel engagement counters.

    ``gate_calls`` / ``attempts`` are the executed-probe counts a census must
    assert non-zero (CLAUDE.md §6); ``gate_reasons`` names which Python-engine
    contract made the feature gate decline, and ``outcomes`` names how each
    driver attempt ended (``served`` on the one path that returns a result).
    """
    st = _eng()
    return {
        "gate_calls": int(st.gate_calls),
        "gate_safe": int(st.gate_safe),
        "gate_reasons": dict(st.gate_reasons),
        "attempts": int(st.attempts),
        "served": int(st.served),
        "outcomes": dict(st.outcomes),
        "last_gate_reason": st.last_gate_reason,
        "last_outcome": st.last_outcome,
    }


def _gate_decline(code: str) -> bool:
    """Record which feature-gate contract declined, and return ``False``."""
    st = _eng()
    st.gate_reasons[code] = st.gate_reasons.get(code, 0) + 1
    st.last_gate_reason = code
    return False


def _outcome(code: str):
    """Record how one driver attempt ended. Returns ``None`` (the decline)."""
    st = _eng()
    st.outcomes[code] = st.outcomes.get(code, 0) + 1
    st.last_outcome = code
    if code == "served":
        st.served += 1
    return None


def _native_spatial_kernel_enabled() -> bool:
    """Whether the #764 native Rust spatial-B&B kernel is engaged (**default OFF**;
    ``DISCOPT_NATIVE_SPATIAL_KERNEL=1`` opts in). Bound-changing / experimental — the
    whole per-node loop moves into ``discopt-core`` (envelope patch + warm OBBT sweep
    + safe-bound pruning + spatial branch). The producer declines any model it cannot
    reproduce bound-neutrally, and the driver falls back to the Python path on decline,
    so ON never changes a certified answer, only which engine computes it.

    Graduation status (#764, panel 2026-07-19,
    ``discopt_benchmarks/results/issue764_native_kernel_graduation_panel_20260719T155819Z.json``):
    the Regime-2 graduation panel (ON vs OFF over the 66-instance in-repo corpus, 60 s
    budget) PASSED BOTH bars — cert-clean with ZERO violations (every ON-optimal
    objective matches OFF to tol, no dual bound past a reference optimum, no
    optimal->non-optimal regression, all 4 engaged incumbents — dispatch, nvs13,
    st_e13, tanksize — independently feasibility-verified) AND net-positive (median
    non-engaged wall Δ = −0.146 s, i.e. ON slightly faster; tanksize moves
    feasible→**optimal**, the issue's headline win, in ~50 s seeded).

    Default-ON decision, re-measured on the current base (#802, 2026-07-19). The two
    historical blockers below are now BOTH RESOLVED, so the decision rests purely on
    net-positive — which is NOT reliably established (it flips with machine load), so
    the default stays **OFF** per CLAUDE.md §4 (a cert-clean but not-cleanly-net-positive
    flag stays OFF — the ``DISCOPT_CUT_INHERIT`` lesson):
      1. Blast radius — RESOLVED. With the flag forced ON, ``-m smoke`` now passes
         **831/831, 0 failures** (was 20 failures / 807 pass on the pre-#789 base). The
         feature-safe routing (#789/#794) made the native kernel decline / pass through
         the models those tests exercise (callbacks, RENS/SubNLP, pools, warm-start,
         deadlines, lazy constraints), so a default-ON no longer silently disables them.
      2. Runaway — RESOLVED by #788/#795: ``_try_native_spatial_kernel`` passes
         ``time_limit_s = remaining`` (from the outer ``Model.solve`` deadline), so the
         kernel no longer runs unbounded to ``max_nodes``.
      3. Net-positive — NOT ESTABLISHED (machine-load-dependent). The headline
         ``tanksize`` win flips with load: on an idle machine the Python path certifies
         in ~21 s ≈ the kernel's ~22 s (helped 0); under load the Python path degrades to
         ``feasible``/uncertified (~90 s, timing out) while the kernel still certifies
         ``optimal`` in ~21 s. So the kernel is more ROBUST (consistent certification),
         but a stable corpus-wide net-positive is unproven — per §4 an unstable/neutral
         result does not graduate a default.
    GRADUATED default-ON 2026-07-27, then UNGRADUATED 2026-07-28 (#902) — see the
    note at the end of this docstring; the graduating panel below was blind to an
    incumbent-quality regression, to half the corpus, and to machine load, and the
    default is now OFF again. Original graduation record retained for history, but
    read it knowing the instrument that produced it was defective (panel
    ``issue764_native_kernel_graduation_panel_20260728T002442Z.json``). The last bar —
    a clean, reproducible net-positive on a VERIFIED-IDLE machine — is now met, and the
    load confound that blocked it four times is gone rather than argued away: the runner
    refused to start until no foreign ``cargo``/``pytest`` was running and 1-min load was
    < 2.5 (it waited 300 s, started at load1 = **2.31**), on a tree verified identical to
    ``origin/main`` with the loaded ``_rust`` extension checked current.

      * **cert-clean: PASS.** The panel reports 0 violations over the 31 instances in
        ``cert-optima.json``. Verified INDEPENDENTLY and more strictly against
        ``minlplib.solu`` (980 ``=opt=``): **307 comparisons, 0 violations** — no false
        primal, no dual bound past a reference optimum (sense read per arm), and the
        ``bound <= incumbent`` certificate invariant checked on every ``optimal`` row.
        That covers 52 of 66 instances; only 14 genuinely lack an ``=opt=``. Note the
        panel's own oracle file is MISSING 21 optima that ``minlplib.solu`` carries,
        including ``tanksize`` — so the panel skipped the oracle check on its own
        headline win. Widening its oracle source is tracked as follow-up.
      * **net-positive: PASS.** engaged 4/66 (dispatch, nvs13, st_e13, tanksize),
        helped 1 (tanksize ``feasible`` -> ``optimal``), median non-engaged wall
        delta **-0.015 s** over 62 instances (i.e. the producer-probe decline overhead is
        not measurable), no instance regressed.

    The ``helped=1`` was checked for the load artifact this docstring previously warned
    about, because the whole bar rests on it. Isolated, interleaved, 2 reps on a quiet
    box: OFF ``feasible`` 61.5 s / 61.5 s, ON ``optimal`` 18.6 s / 18.7 s — reproducible
    to 0.1 s, with ON's objective matching ``=opt= 1.268643754`` to 1.4e-9. So the win is
    real. That also **falsifies** this docstring's earlier claim that "on an idle machine
    the Python path certifies in ~21 s (helped 0)": on today's tree OFF never certifies
    ``tanksize`` at a 60 s budget. Recorded per CLAUDE.md §11.

    **UNGRADUATED 2026-07-28 (#902) — back to opt-in, default OFF, pending a VALID
    panel.** The flag shipped default-ON returning incumbents ~71% from the reference
    optimum on nvs17/nvs19/nvs24. Two things were wrong, and both are now fixed; the
    default is nonetheless OFF, for the reason in "Why still OFF" below.

    *The defect.* :func:`_native_kernel_seed_candidates` enumerated the literal
    ``itertools.product((0.0, 1.0), ...)`` over every FREE integer — a list filtered
    only on span > 0.5 ("not pinned by presolve"), never on being binary. Correct for
    the binary models the seed was built against (tanksize: 9 integers, every box
    ``[0,0]``/``[0,1]``/``[1,1]``), it pins every variable to 0 or 1 on an
    all-general-integer model, so on the nvs family (eight-ish integers in ``[0, 200]``,
    continuous relaxation at ``[1.98, 6.54, 2.82, 2.20, 7.48, 6.20, 6.77]``) every
    candidate was a ``{0,1}`` corner point. nvs19's seed of -315.0 was *exactly* the
    incumbent the kernel then reported after 9,303 nodes — the tree never improved on
    it. The seed now brackets each free integer's relaxation value
    (:func:`_native_seed_bracket`), which on a binary box still yields exactly
    ``{0, 1}``, so the graduated class enumerates identically:

    | instance | seed before | seed after | seed wall |
    |---|---|---|---|
    | nvs17 | -312.6 (71.6% off) | **-1100.4 (exact)** | 8.1 s -> **1.0 s** |
    | nvs19 | -315.0 (71.3% off) | **-1098.2 (0.0% off)** | 12.0 s -> **1.4 s** |
    | nvs24 | -292.6 (71.7% off) | **-1031.8 (0.1% off)** | 12.1 s -> **4.2 s** |

    Separately, :func:`_native_kernel_verify_point` built a fresh ``NLPEvaluator`` per
    call — 697 JAX retraces / 8.5 s inside a 12 s seed phase on nvs19 — and now uses
    ``cached_evaluator``. That is a pure cost fix: it changes the verification's speed,
    never its verdict. Together these also dissolve the "fixed 12 s seed phase"
    symptom: the 12 s was a cap the enumeration burned because every sub-NLP started
    from a garbage corner, not a fixed price.

    *The instrument.* The panel that graduated this flag could not see what it shipped,
    for three independent reasons — all now fixed in
    ``discopt_benchmarks/scripts/issue764_native_kernel_graduation_panel.py``: it
    enumerated only ``minlplib_nl`` (66), which does NOT contain nvs17/19/24 (while
    ``tanksize``, its headline win, exists only there — so neither corpus alone can
    both justify and falsify the flag; it now panels the 119-instance union); every
    certification check required a side to be ``optimal``, so when neither run
    certified nothing fired at all (a ``quality_clean`` gate now compares incumbents
    directly); and it was silently load-sensitive — the budget is wall-clock, so load
    changes which instances hit the limit and therefore which *statuses* the verdict is
    computed from, and it would emit a verdict under load 24 indistinguishable from a
    clean one.

    That third hole was first plugged with a blocking load gate (refuse to start until
    1-min load < 2.5). **That was replaced**: on a real workstation it never runs, so it
    is a wish rather than a test. Two further designs were tried and MEASURED to fail —
    a deterministic ``max_nodes`` budget makes solves bit-reproducible (6/6 identical
    across replicates) but ``node_limit`` sends the kernel back to the Python path, so
    the panel compares OFF against OFF; and a static producer pre-filter drops
    ``tanksize``, the instance carrying the verdict, because the producer is handed the
    *presolved* box and the filter sees declared bounds. What ships instead is
    replication: the decisive instances are re-run with the arms interleaved, a win must
    hold in EVERY replicate, a regression in a majority, and an instance whose
    replicates disagree is quarantined as unresolved. Load can now only move an instance
    to "unresolved" — it can no longer make the verdict wrong — and the panel runs on a
    busy machine (verified end-to-end at load 2.2 rising to 5.9: 4/4 decisive instances
    STABLE, 0 quarantined).

    *Why still OFF.* Re-measured post-fix on nvs17/19/24 + tanksize the panel returns
    ``GRADUATE: YES`` — nvs17 goes ``feasible`` -> ``optimal``, nvs24 gains a primal OFF
    never finds, nvs19 beats OFF (-1098.2 vs -1097.6). But that is a 4-instance probe,
    and §5 requires a corpus-wide differential panel. Since the only panel this flag
    ever passed was blind on all three counts above, **no valid graduation panel has
    ever been run for it** — the 2026-07-27 graduation was not validly earned
    independently of the seed defect. So the default returns to OFF until a clean
    119-instance replicated run passes, which is the re-graduation gate
    tracked in #902. The cost of that conservatism is explicit: ``tanksize`` and
    ``nvs17`` both go ``optimal`` -> ``feasible`` at a 60 s budget with the kernel off.

    ``DISCOPT_NATIVE_SPATIAL_KERNEL=1`` opts back in. Still open on the kernel itself
    (measured, not yet root-caused): it abandons ~1/3 of its wall budget — nvs19/nvs24
    exit at ~39 s of a 60 s limit — and its accounting fields are wrong on this path
    (``rust_time`` reads ~1e-4 s for a ~7 s Rust tree, ``jax_time`` 0.0), so do not
    build gates on them."""
    return env_bool("DISCOPT_NATIVE_SPATIAL_KERNEL", False)


def _native_kernel_feature_safe(
    *,
    mccormick_bounds,
    initial_point,
    lazy_constraints,
    incumbent_callback,
    node_callback,
    kwargs,
) -> bool:
    """Whether the native spatial kernel may take over this solve (#789).

    The kernel runs the ENTIRE spatial B&B inside ``discopt-core`` and so does
    not exercise the Python-engine machinery layered on the interpreter loop:
    node/incumbent/iteration callbacks, lazy-constraint injection, the solution
    pool, warm-start incumbents, non-default McCormick bound modes, and any
    per-solve ``tuning`` whose stats/behaviour a caller inspects. Those are real
    contracts (encoded by the smoke suite), so when a solve REQUESTS one, we
    decline here and route it to the trusted Python engine, taking the native
    path only when the solve is feature-safe. This is the #789 "route
    unsupported models to the Python engine" resolution — no test is weakened,
    and the fast native path is still taken for the plain certified-optimal
    solves it was graduated on (tanksize et al.).

    Declining is always sound: it only ever *widens* which solves use the
    already-trusted default path.
    """
    _eng().gate_calls += 1
    if incumbent_callback is not None:
        return _gate_decline("incumbent_callback")
    if node_callback is not None:
        return _gate_decline("node_callback")
    if kwargs.get("iteration_callback") is not None:
        return _gate_decline("iteration_callback")
    if lazy_constraints:
        return _gate_decline("lazy_constraints")
    if initial_point is not None:
        return _gate_decline("initial_point")
    # A non-default McCormick bound mode changes the relaxation the caller asked
    # for; the kernel always builds its own McCormick relaxation, so honour the
    # request by routing to the Python engine (e.g. ``mccormick_bounds="none"``).
    if mccormick_bounds is not None and str(mccormick_bounds) != "auto":
        return _gate_decline("mccormick_bounds")
    # Solution-pool collection is a Python-engine feature the kernel does not fill.
    if kwargs.get("solution_pool") is not None or kwargs.get("solution_pool_capacity"):
        return _gate_decline("solution_pool")
    # An explicit per-solve tuning object may enable/disable levers whose
    # solver_stats a caller inspects (e.g. cut-inherit pool stats); the kernel
    # emits none of those, so route explicit-tuning solves to the Python engine.
    if kwargs.get("tuning") is not None:
        return _gate_decline("tuning")
    _eng().gate_safe += 1
    return True


# Wall budget (seconds) for the incumbent-seeding NLP heuristics (#764 Task 1). A
# SubNLP solve (fix rounded integers, solve the continuous NLP) from the presolved
# box midpoint — plus a stratified continuous multistart for pure-continuous models —
# finds a genuine feasible point whose TRUE objective seeds the native full solve's
# cutoff. Cheap relative to the full search, and seeding turns tanksize's ~190 s
# unseeded solve into a small fraction of that. (Best-bound node selection is a poor
# incumbent finder — an unseeded node-budget probe found no feasible point for
# tanksize even at 20k nodes — so we use the NLP heuristics the trusted path already
# uses to seed its first incumbent.)
_NATIVE_SEED_HEURISTIC_S = 12.0


def _native_kernel_verify_point(model, x_flat):
    """Rigorously verify that ``x_flat`` (length ``n_orig``, original-variable order)
    is feasible for the ORIGINAL model, and return ``(True, model_objective)`` with the
    point's TRUE objective in model units, or ``(False, None)``.

    Soundness (#764 Task 1): this gates whether a value may seed the native kernel's
    incumbent cutoff. An unverified seed would poison every downstream certificate, so
    the contract is strict — it returns ``True`` ONLY when the model evaluator
    successfully evaluated every constraint AND the objective and every residual is
    within the repo tolerances (bounds abs=1e-6 + rel=1e-4 * |bound|, rows
    abs=1e-6 * row-scale, integrality 1e-5). Any evaluator failure, shape mismatch, or
    non-finite value yields ``(False, None)`` — never an optimistic pass. The objective
    returned is recomputed from the original variables (independent of the kernel's
    McCormick aux columns), so it is the genuinely-attained value, not an optimistic
    relaxation reading.

    The row check delegates to :func:`discopt.validation.feasibility.verify_point`,
    which is the single verifier for the whole repo. It replaced this function's own
    loop after that loop was measured to (a) reject ``nvs22``'s certified optimum
    because its tolerance ``abs + rel*|residual|`` is self-referential and collapses
    to a pure absolute 1e-6 on any row scale, and (b) ACCEPT a point violating row 2
    of a size-3 vector constraint by 5.0, because it advanced one row index per
    constraint object while the evaluator emits one row per flat element. See that
    module's header for the full defect list."""
    from discopt.modeling.core import ObjectiveSense
    from discopt.validation.feasibility import verify_point

    x_flat = np.asarray(x_flat, dtype=np.float64)
    if not np.all(np.isfinite(x_flat)):
        return False, None

    try:
        # ``cached_evaluator``, not ``NLPEvaluator(model)``: this is called once per
        # seed candidate (and once more on the final incumbent), and constructing an
        # evaluator re-traces and re-compiles the model's JAX constraint/objective/
        # Jacobian callables every time. Profiling #902 measured 697 traces / 8.5 s
        # inside a 12 s seed phase on nvs19 — the same defect ``cached_evaluator`` was
        # introduced to fix for the diving heuristic. The cache is keyed on the model's
        # structural fingerprint and reads bounds/parameters live, so a cached
        # evaluator computes byte-identical residuals: this changes only the cost of
        # the verification, never its verdict.
        from discopt._jax.nlp_evaluator import cached_evaluator

        evaluator = cached_evaluator(model)
        verdict = verify_point(model, x_flat, evaluator=evaluator)
        if not verdict.ok:
            logger.debug("native seed verification failed: %s", verdict.describe())
            return False, None
        obj_min = float(evaluator.evaluate_objective(x_flat))
    except Exception as exc:  # evaluator could not vouch -> NOT verified
        logger.debug("native seed verification skipped (evaluator error): %s", exc)
        return False, None
    if not math.isfinite(obj_min):
        return False, None
    # ``evaluate_objective`` negates the body for a MAXIMIZE model (it minimizes the
    # negation); undo that so the returned value is the objective in model units.
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        model_obj = -obj_min
    else:
        model_obj = obj_min
    return True, float(model_obj)


# Cap on the number of FREE integers (span > 0.5 in the presolved box) the seed
# enumerates over: 2**k sub-NLP solves. Presolve typically fixes most, leaving a
# handful (tanksize: 5 of 9 free). Above this the enumeration is skipped for a single
# rounding sub-NLP so a wide MINLP cannot cause a combinatorial blow-up.
_NATIVE_SEED_MAX_FREE_INTEGERS = 10


def _native_seed_bracket(x_rel: float, lo_b: float, up_b: float) -> tuple[float, ...]:
    """The two integer values BRACKETING ``x_rel`` inside ``[lo_b, up_b]``, nearest first.

    This is the per-variable candidate set the seed enumeration crosses (#902). It is
    written as a bracket rather than a literal ``(0.0, 1.0)`` because the enumeration
    must generalize from binaries to GENERAL integers:

    * On a **binary** box ``[0, 1]`` it returns exactly ``{0, 1}`` for every relaxation
      value — including ``x_rel == 1.0``, thanks to the ``up_b - 1`` clamp on the floor.
      So the class the #764 seed was graduated on (tanksize: 5 free binaries -> 32
      sub-NLPs) enumerates precisely the same 2**k assignments as before.
    * On a **general-integer** box (nvs17/19/24: eight-ish integers in ``[0, 200]``) it
      returns the floor/ceil pair around the continuous relaxation. The previous code
      crossed the literal ``(0.0, 1.0)`` over every free integer, pinning each one to 0
      or 1 — a corner of a ``[0, 200]^n`` box nowhere near the relaxation, which landed
      at ``[1.98, 6.54, 2.82, 2.20, 7.48, 6.20, 6.77]`` on nvs17. Every seed it produced
      was a ``{0,1}`` vector ~71% off the reference optimum (#902).

    Nearest-first ordering matters because the enumeration is deadline-bounded and
    ``itertools.product`` is lexicographic: the FIRST combination crossed is the
    nearest-rounding point (the classic sub-NLP start), and the combinations that follow
    flip the least-confident variables first. On a wide box only a prefix of the 2**k
    product is reached before the deadline, so that prefix must be the promising one.
    """
    if not math.isfinite(x_rel):
        x_rel = 0.5 * (lo_b + up_b)
    # Clamp against the INTEGER box, not the raw bounds. A box may carry non-integer
    # endpoints (presolve tightening a general integer to e.g. ``[0.5, 3.5]``), and
    # clamping to those directly would emit a fractional candidate — these values are
    # assigned straight into integer slots, so that would be a defect.
    ilo = math.ceil(lo_b)
    ihi = math.floor(up_b)
    if ihi < ilo:
        # No integer lies inside the box at all (e.g. ``[0.5, 0.7]``). Nothing here is a
        # valid assignment; hand back the nearest integer as a sub-NLP START point only
        # (the caller re-verifies every candidate, and ``subnlp`` clips to bounds), so a
        # degenerate box costs one wasted solve rather than a bogus seed.
        return (float(ilo),)
    lo = min(max(math.floor(x_rel), ilo), max(ilo, ihi - 1))
    hi = min(lo + 1, ihi)
    if hi <= lo:
        # A box admitting exactly one integer (e.g. ``[0, 0.6]``).
        return (float(lo),)
    return (float(lo), float(hi)) if abs(x_rel - lo) <= abs(x_rel - hi) else (float(hi), float(lo))


def _native_kernel_seed_candidates(model, lb, ub, n_orig, deadline):
    """Yield candidate feasible points (flat) for the native-kernel seed, from the SAME
    NLP heuristics the trusted spatial path uses to find its first incumbent — general
    across the covered subset (MINLP *and* pure-continuous). Each yielded point is only
    a CANDIDATE; the caller re-verifies it rigorously before it may seed anything.

    Strategy: solve the continuous NLP relaxation once (from the presolved-box
    midpoint) for a warm continuous base, then FIX the presolve-pinned integers to
    their box value and ENUMERATE the FREE integers (span > 0.5 in the presolved box)
    over the two values BRACKETING each one's relaxation value
    (:func:`_native_seed_bracket`), running a sub-NLP per combo. This lands genuine
    integer-feasible points a single nearest-rounding sub-NLP misses on tightly-coupled
    integers (tanksize: rounding the relaxation is integer-infeasible, but 5 free
    binaries enumerate to 32 sub-NLPs, several feasible). A continuous multistart is
    added for the pure-continuous case.

    The bracket is what makes the enumeration general (#902). It previously crossed the
    literal ``(0.0, 1.0)`` over every free integer — correct for the binaries the #764
    seed was graduated on, but on an all-general-integer model it pinned each variable
    to 0 or 1 regardless of its box, so on nvs17/19/24 (``[0, 200]`` integers) every
    candidate was a ``{0,1}`` corner point and the best verified seed came out ~71% off
    the reference optimum. On a binary box the bracket still yields exactly ``{0, 1}``,
    so this generalizes the enumeration without perturbing that class."""
    if time.perf_counter() >= deadline:
        return

    import itertools

    from discopt.modeling.core import VarType

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    lb_c = np.clip(lb, -_SPC, _SPC)
    ub_c = np.clip(ub, -_SPC, _SPC)
    midpoint = 0.5 * (lb_c + ub_c)

    # Flat integer positions (scalar-variable covered subset: flat index == var index).
    int_pos = [
        i for i, v in enumerate(model._variables) if v.var_type in (VarType.INTEGER, VarType.BINARY)
    ]
    free_int = [i for i in int_pos if i < n_orig and (ub[i] - lb[i]) > 0.5]

    ev = None
    backend = None
    xr = None
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solvers.nlp_backend import get_nlp_solver

        ev = NLPEvaluator(model)
        backend = get_nlp_solver("auto")
        r = backend(
            ev,
            midpoint,
            options={
                "print_level": 0,
                "max_iter": 500,
                "max_wall_time": max(1e-3, min(4.0, deadline - time.perf_counter())),
            },
        )
        if getattr(r, "x", None) is not None:
            xr = np.asarray(r.x, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("native seed relaxation solve failed: %s", exc)

    base = xr.copy() if xr is not None else midpoint.copy()
    # Pin presolve-fixed integers to their box value in the continuous base.
    for i in int_pos:
        if i < base.shape[0] and i not in free_int:
            base[i] = float(np.round(0.5 * (lb[i] + ub[i])))

    try:
        from discopt._jax.primal_heuristics import subnlp

        if len(free_int) <= _NATIVE_SEED_MAX_FREE_INTEGERS:
            # Per-free-integer candidate values: the two integers bracketing this
            # variable's continuous-relaxation value, nearest first (#902). On a binary
            # box this is exactly ``(0.0, 1.0)`` — the graduated behaviour — and on a
            # general-integer box it tracks the relaxation instead of pinning to 0/1.
            cand_vals = [
                _native_seed_bracket(float(base[i]), float(lb[i]), float(ub[i]))
                for i in free_int
                if i < base.shape[0]
            ]
            enum_idx = [i for i in free_int if i < base.shape[0]]
            for combo in itertools.product(*cand_vals) if cand_vals else [()]:
                if time.perf_counter() >= deadline:
                    break
                x0 = base.copy()
                for idx, val in zip(enum_idx, combo):
                    x0[idx] = val
                try:
                    cand = subnlp(
                        model,
                        x0,
                        evaluator=ev,
                        backend=backend,
                        time_budget=max(1e-3, min(3.0, deadline - time.perf_counter())),
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("native seed subnlp raised: %s", exc)
                    cand = None
                if cand is not None and cand[0] is not None:
                    yield np.asarray(cand[0], dtype=np.float64)
        elif xr is not None and time.perf_counter() < deadline:
            # Too many free binaries to enumerate: one nearest-rounding sub-NLP.
            try:
                cand = subnlp(
                    model,
                    xr,
                    evaluator=ev,
                    backend=backend,
                    time_budget=max(1e-3, min(4.0, deadline - time.perf_counter())),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("native seed subnlp raised: %s", exc)
                cand = None
            if cand is not None and cand[0] is not None:
                yield np.asarray(cand[0], dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("native seed subnlp import/setup failed: %s", exc)

    if not int_pos and time.perf_counter() < deadline:
        try:
            from discopt._jax.primal_heuristics import continuous_multistart

            cand = continuous_multistart(
                model,
                n_starts=int(min(64, max(16, 2 * int(n_orig)))),
                evaluator=ev,
                deadline=deadline,
            )
            if cand is not None and cand[0] is not None:
                yield np.asarray(cand[0], dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("native seed multistart raised: %s", exc)


def _native_kernel_seed(model, lb, ub, sign, off, n_orig, outer_deadline=None):
    """Return ``(internal_value, point)`` — a genuinely-attained incumbent to seed the
    native solve (``internal_value`` in kernel-internal minimize units, ``point`` the
    verified original-variable vector) — or ``(None, None)`` if no verified feasible
    point is available.

    Draws candidate feasible points from :func:`_native_kernel_seed_candidates`,
    RIGOROUSLY re-verifies each against the original model
    (:func:`_native_kernel_verify_point`), maps its TRUE objective to internal units via
    ``internal = sign * model_obj - offset`` (the inverse of the producer's
    ``model = sign * (internal + offset)``, exact since ``sign in {+1,-1}``), and keeps
    the BEST (lowest internal value = tightest sound cutoff) over all candidates found
    before the deadline. A heuristic's own reported objective is never trusted directly.
    The verified point is returned too so the driver can report it if the seeded full
    solve proves optimality without ever improving on (and thus re-recording) the seed.
    A verified feasible objective is only ever an *upper bound*: seeding it can prune and
    cutoff-propagate but can never loosen a bound or invent an incumbent, so an
    unverifiable candidate is simply dropped and the full solve runs unseeded."""
    deadline = time.perf_counter() + float(_NATIVE_SEED_HEURISTIC_S)
    if outer_deadline is not None:
        deadline = min(deadline, float(outer_deadline))
    if time.perf_counter() >= deadline:
        return None, None
    best_internal: Optional[float] = None
    best_point = None
    for x_cand in _native_kernel_seed_candidates(model, lb, ub, n_orig, deadline):
        if x_cand.shape[0] < n_orig:
            continue
        point = x_cand[:n_orig].copy()
        ok, model_obj = _native_kernel_verify_point(model, point)
        if not ok:
            continue
        internal = sign * float(model_obj) - off
        if not math.isfinite(internal):
            continue
        if best_internal is None or internal < best_internal:
            best_internal = internal
            best_point = point
    if best_internal is None:
        return None, None
    return best_internal, best_point


def _try_native_spatial_kernel(
    model,
    lb,
    ub,
    n_vars,
    gap_tolerance,
    max_nodes,
    time_limit,
    t_start,
    rust_time,
    jax_time,
):
    """Issue #764: if the native Rust spatial kernel is enabled and the model is in
    its covered subset — scalar variables; bilinear / monomial / affine-square / sqrt
    terms; a valid McCormick relaxation at the presolved box ``[lb, ub]`` — run the
    ENTIRE spatial B&B inside ``discopt-core`` and return a :class:`SolveResult`; else
    ``None`` so the caller runs the trusted Python path.

    Soundness: the producer declines (``None``) any model it cannot reproduce
    bound-neutrally. A fully-certified native solve (``status == 'optimal'``) is
    returned as optimal; a wall-clock-limited native solve returns its rigorous
    partial incumbent/bound with ``status == 'time_limit'``. Node-limited or declined
    runs retain the established Python fallback. The kernel's incumbent satisfies the
    model's linear rows (they are fixed rows in its LP) and every lifted term to
    ``mccormick_tol``.
    The internal minimize-convention objective/bound are mapped to model units via the
    producer's ``sign*(value + offset)`` metadata.

    Task 1 (#764): before the full solve, a short unseeded probe run finds a
    McCormick-tight candidate point which is RIGOROUSLY verified feasible against the
    original model; its true objective seeds ``initial_incumbent`` so the full search
    prunes with a tight cutoff from node 0 (tanksize: ~190 s -> ~27 s). Seeding only
    ever changes *which* certified answer path is walked (a valid upper bound prunes,
    it never loosens a bound or invents an incumbent); an unverified candidate is
    discarded and the full solve runs unseeded. Issue #788 caps both this seed phase
    and the native tree against the outer ``Model.solve(time_limit=...)`` deadline."""
    if not _native_spatial_kernel_enabled():
        return None
    _eng().attempts += 1
    try:
        from discopt import _rust
        from discopt._jax.spatial_producer import build_spatial_kernel_spec

        spec = build_spatial_kernel_spec(
            model,
            bounds=(
                np.asarray(lb, dtype=np.float64)[:n_vars],
                np.asarray(ub, dtype=np.float64)[:n_vars],
            ),
        )
        if spec is None:
            # The producer's own reason code says WHICH feature is missing —
            # ``spatial_producer.producer_stats()['last']``.
            return _outcome("producer_declined")  # outside the covered subset
        meta = {k: spec.pop(k) for k in list(spec) if k.startswith("meta_")}
        sign = float(meta["meta_obj_sense_sign"])
        off = float(meta["meta_obj_offset"])
        n_orig = int(spec["n_orig"])

        # Task 1: obtain a cheap, rigorously-verified feasible seed for the cutoff.
        # A non-finite budget means "no wall-clock cap": carry it as ``None`` rather
        # than an infinite deadline, so the kernel is asked for an uncapped search
        # instead of rejecting a non-finite ``time_limit_s`` (which the defensive
        # ``except`` below would swallow, silently disabling the kernel outright).
        _outer_budget = float(time_limit)
        outer_deadline = t_start + _outer_budget if math.isfinite(_outer_budget) else None
        initial_incumbent, seed_point = _native_kernel_seed(
            model,
            np.asarray(lb, dtype=np.float64)[:n_vars],
            np.asarray(ub, dtype=np.float64)[:n_vars],
            sign,
            off,
            n_orig,
            outer_deadline,
        )

        remaining = (
            None if outer_deadline is None else max(0.0, outer_deadline - time.perf_counter())
        )
        solve_kwargs = dict(
            max_nodes=int(max_nodes),
            gap_tol=float(gap_tolerance),
            time_limit_s=remaining,
        )
        if initial_incumbent is not None:
            solve_kwargs["initial_incumbent"] = float(initial_incumbent)
        res = _rust.solve_spatial_tree_py(**spec, **solve_kwargs)
        res.update(meta)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("native spatial kernel skipped: %s", exc)
        return _outcome(f"exception:{type(exc).__name__}")
    if res is None:
        return _outcome("kernel_returned_none")  # outside the covered subset
    native_status = res.get("status")
    if native_status not in ("optimal", "time_limit"):
        # other incomplete exits retain the established Python fallback
        return _outcome(f"status_{native_status}")
    if native_status == "optimal" and res.get("incumbent") is None:
        return _outcome("optimal_without_incumbent")

    sign = float(res["meta_obj_sense_sign"])
    off = float(res["meta_obj_offset"])
    obj_val = sign * (float(res["incumbent"]) + off) if res.get("incumbent") is not None else None
    bound_val = sign * (float(res["bound"]) + off)
    x_incumbent = np.asarray(res["incumbent_x"], dtype=np.float64)
    if obj_val is None:
        x_flat = None
    elif x_incumbent.shape[0] >= n_vars:
        # The kernel found (and re-recorded) its own improving incumbent point.
        x_flat = x_incumbent[:n_vars]
    elif seed_point is not None and seed_point.shape[0] >= n_vars:
        # Seeded solve proved optimality without ever improving on the seed, so the
        # kernel carries no point (``incumbent_x`` empty) — report OUR verified seed
        # point, whose true objective equals ``res['incumbent']`` by construction.
        x_flat = np.asarray(seed_point, dtype=np.float64)[:n_vars]
    elif native_status == "optimal":
        # no usable incumbent point -> Python path (never fabricate one)
        return _outcome("optimal_without_point")
    else:
        # A time-limited search may carry an incumbent VALUE (e.g. the seeded cutoff)
        # with no corresponding point in this process. Report the rigorous bound but
        # never fabricate a primal witness for it.
        obj_val = None
        x_flat = None

    # #789: rigorously verify the FINAL reported incumbent against the ORIGINAL
    # model before returning it as a certified optimum. The kernel solves a
    # McCormick relaxation with its own integrality handling; on some models
    # (e.g. a bilinear coupled to a binary in a way the kernel's rounding does
    # not reproduce exactly) its tree incumbent is infeasible in the original —
    # a false primal the #779 final-incumbent guard would catch and withhold,
    # turning a solvable model into a null result. Verifying here instead means
    # the kernel *declines* such a model (returns None), so the trusted Python
    # engine solves it and reports the true optimum. Sound and conservative:
    # declining only ever widens use of the already-trusted default path, and
    # the kernel is still taken on every model whose incumbent verifies (the
    # tanksize-class it was graduated on). ``x_flat`` is original-variable order
    # (``n_vars`` slots) — the same layout ``_native_kernel_verify_point`` reads.
    # #788: a time-limited exit may carry no primal at all; there is nothing to
    # verify then, and the bound-only result below is reported without one.
    if x_flat is not None:
        _ok, _model_obj = _native_kernel_verify_point(model, x_flat[:n_orig])
        if not _ok:
            logger.debug(
                "native spatial kernel: final incumbent failed original-model "
                "verification (obj=%.6g) — routing to the Python engine (#789)",
                obj_val,
            )
            return _outcome("final_incumbent_unverified")
        # Prefer the independently-recomputed true objective (exact model units) over
        # the kernel's mapped relaxation reading when they agree within tolerance; a
        # gross disagreement is itself a decline signal.
        if _model_obj is not None and abs(_model_obj - obj_val) > 1e-4 * (1.0 + abs(obj_val)):
            logger.debug(
                "native spatial kernel: reported obj %.6g disagrees with verified "
                "obj %.6g — routing to the Python engine (#789)",
                obj_val,
                _model_obj,
            )
            return _outcome("objective_disagreement")

    # Deferred (Card 4b): the package __init__ imports this module, so a
    # module-level import of its helper would be a cycle.
    from discopt.solver import _unpack_solution

    x_dict = _unpack_solution(model, x_flat) if x_flat is not None else None
    wall_time = time.perf_counter() - t_start
    gap_val = abs(obj_val - bound_val) / (abs(obj_val) + 1e-10) if obj_val is not None else None
    logger.info(
        "native spatial kernel (#764) exited %s: obj=%s bound=%.6g nodes=%d",
        native_status,
        obj_val,
        bound_val,
        int(res["node_count"]),
    )
    _outcome("served")
    return SolveResult(
        status=native_status,
        objective=obj_val,
        bound=bound_val,
        gap=gap_val,
        x=x_dict,
        wall_time=wall_time,
        node_count=int(res["node_count"]),
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=wall_time - rust_time - jax_time,
        gap_certified=math.isfinite(bound_val),
    )


# Native-AD node NLP solves (discopt#281): route the per-node NLP through
# POUNCE's own AD on the .nl problem instead of the JAX callback bridge. Opt-in
# (DISCOPT_NLP_NATIVE / options["nlp_native"]); falls back to the JAX path
# automatically whenever a native base cannot be built/validated for the model
# (see solvers.nlp_native). Default OFF: POUNCE's PyNlProblem is unsendable
# (pyo3), so caching it on the model and using it across the batch/parallel paths
# trips "unsendable ... dropped on another thread" under pytest-xdist and can
# perturb MIQP-batch certification; and the speedup is neutral-to-modest. Enable
# explicitly once PyNlProblem is made Send-safe.
_NLP_NATIVE_DEFAULT = env_bool("DISCOPT_NLP_NATIVE", False)


def _native_nlp_enabled(options: dict) -> bool:
    """Whether to attempt the POUNCE-native node NLP path for this solve."""
    val = options.get("nlp_native") if options else None
    return _NLP_NATIVE_DEFAULT if val is None else bool(val)
