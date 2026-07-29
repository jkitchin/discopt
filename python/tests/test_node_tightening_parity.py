"""Card 3c — the standing node-tightening parity test across engines.

Review §2.5.1: the Python spatial path, the Python NLP-BB path, the native Rust
spatial kernel and the MILP driver run **different** node-tightening stacks, and
nothing asserts they reach compatible fixed points.  Flipping
``DISCOPT_NATIVE_SPATIAL_KERNEL`` changes *which tightenings a node sees*, not
merely which language runs them.  Phase 5's whole strategy is to widen native
kernel coverage feature by feature; this file is the guard that says a widened
kernel has not silently weakened node tightening or, far worse, started removing
feasible points.

## What is compared, and why it is comparable

Two engines branch differently, so their node *streams* can never be aligned.
What **is** comparable is the tightening stack as a *function on a box*.  For
every node the shipped Python loop decides, this test captures:

    B0  the node box as exported by the tree
    P   = Python(B0)          — the Jacobian + 17-rule structural pass
    S   = Kernel(Python(B0))  — what the node actually gets (the shipped stack)
    K   = Kernel(B0)          — the counterfactual: kernel alone on the same box

``in_tree_presolve`` takes ``&self`` on the Rust side — it is a pure function of
(repr, box, depth, incumbent) — so evaluating ``K`` on the same repr is
side-effect free and cannot perturb the search it observes.  This is the same
counterfactual construction Card 2b used.

## The invariants

* **I1 — contraction.** Every stack only ever shrinks its input box.  A stack
  that *grows* a box is not a tightening.
* **I2 — soundness floor (the one that must never be relaxed).**  A box that
  contains a known-feasible point must still contain it after tightening, on
  *every* stack.  This is the invariant a Phase 5 kernel expansion could break
  catastrophically and silently: dropping a feasible point is how a false
  ``infeasible`` or a false ``optimal`` gets certified.
* **I3 — kernel monotonicity.** The kernel applied to a tighter box must not
  return a *looser* box: ``Kernel(Python(B0)) ⊆ Kernel(B0)``.  A violation means
  the kernel's inference is path-dependent, which makes "the kernel is at least
  as tight" unprovable.
* **I4 — the documented asymmetry, with counts.** Card 2b measured that the
  kernel does **not** subsume the Python pass: 278 of 1,495 decided nodes (18.6 %)
  carried a Python-only inference — 147 from the Jacobian linear-row FBBT alone,
  83 from the structural/interval nonlinear rules alone, 44 from both.  That gap
  is *expected*, so this test asserts a **ceiling**, not equality: the test fails
  when the gap **grows** past the recorded envelope (a new divergence), and never
  when Phase 5 shrinks it.

I1–I3 are hard assertions.  I4 is a ledger with a ceiling.

## Instrumentation

Per CLAUDE.md §6 every test prints its executed comparison count and the
file-level check fails when the totals are zero — a probe that decided no nodes
must not read as a pass.  Per §7 no exception around a comparison is swallowed.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_DATA = Path(__file__).parent / "data"
_CORPUS_DIRS = (_DATA / "minlplib_nl", _DATA / "minlplib")

#: Relative tolerance for "strictly tighter" (Card 2b's 1e-12).
_REL = 1e-12

#: Absolute slack for the soundness containment check. Node boxes are produced by
#: floating-point branching and the reference point comes back from an NLP solve,
#: so an exact ``lb <= x <= ub`` would flag ulp-level noise as a lost feasible
#: point. 1e-6 is the repo's ``abs`` tolerance (conftest).
_CONTAIN_TOL = 1e-6

#: Instances chosen to exercise BOTH Python loops — the spatial McCormick loop
#: and the convex NLP-BB loop — and to actually *decide nodes* under the budget
#: below. Drawn from Card 2b's instance list (itself drawn from the Phase 0
#: baseline by ``node_count > 0 and not convex_fast_path``); ``ex1264``/``ex1263``
#: were dropped after measuring **zero** decided nodes at this budget, i.e. they
#: made their arm of the comparison vacuous. The totals check at the bottom is
#: what keeps that from happening silently again.
PARITY_INSTANCES = ("st_e05", "nvs22", "m3", "tanksize", "syn05m", "nvs05")

#: Per-instance solve budget. Short on purpose: the invariants are per node, so
#: node *volume* buys more than node depth, and this file has to stay runnable.
_BUDGET_S = 8.0

#: I4 ceiling. Card 2b measured 18.6 % of decided nodes carrying a Python-only
#: inference over 25 instances. This file runs a 6-instance subset at a shorter
#: budget, so its own rate is measured, not inherited; the ceiling is set well
#: above the observed value and exists to catch a *step change* — a Phase 5
#: kernel edit that starts losing inferences it used to make. Shrinking the gap
#: can never fail this assertion.
_PYTHON_ONLY_NODE_RATE_CEILING = 0.60

#: Below this many decided nodes a per-instance rate is noise (one node moves it
#: by tens of points), so the ceiling is applied only to the pooled rate.
_MIN_NODES_FOR_RATE = 10


def _instance_path(stem: str) -> Path:
    for d in _CORPUS_DIRS:
        p = d / f"{stem}.nl"
        if p.exists():
            return p
    pytest.skip(f"{stem}.nl not in the in-repo corpus")


def _contains(lb, ub, x, mask) -> bool:
    """Is the witness inside the box, on the coordinates the witness covers?

    ``mask`` selects the flat coordinates for which a witness value exists.  The
    solve may reformulate the model and introduce auxiliary columns that the
    reported solution has no entry for; those coordinates are simply not checked
    rather than assumed.  Coordinates whose bound is NaN are skipped and counted
    by the caller — a NaN bound is its own defect, not a containment failure.
    """
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    sel = mask & np.isfinite(lb) & np.isfinite(ub)
    if not sel.any():
        return True
    xv = x[sel]
    tol = _CONTAIN_TOL * np.maximum(1.0, np.abs(xv))
    return bool(np.all(xv >= lb[sel] - tol) and np.all(xv <= ub[sel] + tol))


def _witness(model, x_dict, n_flat):
    """Flat witness vector + coverage mask, matched by variable NAME.

    No positional assumption is made about a reformulated model's column order:
    a flat coordinate is witnessed only when its own variable's name appears in
    the reported solution.
    """
    vals = np.zeros(n_flat, dtype=np.float64)
    mask = np.zeros(n_flat, dtype=bool)
    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1) or 1)
        if off + size > n_flat:
            break
        got = x_dict.get(v.name)
        if got is not None:
            arr = np.asarray(got, dtype=np.float64).ravel()
            if arr.size == size:
                vals[off : off + size] = arr
                mask[off : off + size] = True
        off += size
    return vals, mask


class _Ledger:
    """Counters for one instance's captured node stream."""

    def __init__(self) -> None:
        self.py_calls = 0
        self.decided_nodes = 0
        self.bounds_compared = 0
        self.unmatched = 0
        # I1
        self.contraction_checks = 0
        self.contraction_violations: list[str] = []
        # I2 (filled post-hoc, once the reference point is known)
        self.captured: list[tuple] = []
        self.soundness_checks = 0
        self.soundness_violations: list[str] = []
        # I3
        self.monotonicity_checks = 0
        self.monotonicity_violations: list[str] = []
        # I4
        self.python_only_nodes = 0
        self.python_only_bounds = 0
        self.nlp_bb_loop = False
        self.eval_model = None
        self.witness_skipped = 0


def _capture(instance: str, budget: float) -> tuple[_Ledger, object]:
    """Solve ``instance`` with both node-tightening entry points instrumented."""
    import discopt.solver as _solver
    from discopt._rust import PyModelRepr
    from discopt.modeling.core import from_nl

    led = _Ledger()
    box_of: dict[bytes, tuple] = {}
    last_repr: list = []

    def _key(lb, ub) -> bytes:
        return (
            np.asarray(lb, dtype=np.float64).tobytes() + np.asarray(ub, dtype=np.float64).tobytes()
        )

    orig_py = _solver._tighten_node_bounds_with_status
    orig_itp = PyModelRepr.in_tree_presolve

    def py_wrap(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=3):
        b0_lb = np.asarray(node_lb, dtype=np.float64).copy()
        b0_ub = np.asarray(node_ub, dtype=np.float64).copy()
        t_lb, t_ub, inf = orig_py(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds)
        led.py_calls += 1
        if inf:
            return t_lb, t_ub, inf
        p_lb = np.asarray(t_lb, dtype=np.float64)
        p_ub = np.asarray(t_ub, dtype=np.float64)
        # I1 for the Python stack: P must be inside B0.
        led.contraction_checks += 1
        _grew = (
            ((p_lb < b0_lb - 1e-9) | (p_ub > b0_ub + 1e-9)) & np.isfinite(p_lb) & np.isfinite(p_ub)
        )
        if _grew.any():
            j = int(np.flatnonzero(_grew)[0])
            led.contraction_violations.append(
                f"Python stack GREW var[{j}]: [{b0_lb[j]:.12g},{b0_ub[j]:.12g}] "
                f"-> [{p_lb[j]:.12g},{p_ub[j]:.12g}]"
            )
        if led.eval_model is None:
            led.eval_model = getattr(evaluator, "_model", None)
        box_of[_key(t_lb, t_ub)] = (b0_lb, b0_ub)
        return t_lb, t_ub, inf

    def itp_wrap(self, node_lb, node_ub, **kw):
        last_repr.append(self)
        del last_repr[:-1]
        d_s = orig_itp(self, node_lb, node_ub, **kw)
        if not d_s["ran"]:
            return d_s
        b0 = box_of.pop(_key(node_lb, node_ub), None)
        if b0 is None:
            led.unmatched += 1
            return d_s
        b0_lb, b0_ub = b0
        d_k = orig_itp(self, b0_lb, b0_ub, **kw)
        led.decided_nodes += 1

        if d_s["infeasible"] or d_k["infeasible"]:
            # A fathom on either arm: the box comparison is undefined, but the
            # asymmetry still counts (Python-only fathom = shipped arm empty,
            # kernel-alone arm not).
            if d_s["infeasible"] and not d_k["infeasible"]:
                led.python_only_nodes += 1
            return d_s

        s_lb = np.asarray(d_s["lb"], dtype=np.float64)
        s_ub = np.asarray(d_s["ub"], dtype=np.float64)
        k_lb = np.asarray(d_k["lb"], dtype=np.float64)
        k_ub = np.asarray(d_k["ub"], dtype=np.float64)
        led.bounds_compared += 2 * s_lb.size

        # I1 for the kernel arms.
        led.contraction_checks += 2
        for name, lo, hi in (("shipped", s_lb, s_ub), ("kernel-alone", k_lb, k_ub)):
            with np.errstate(invalid="ignore"):
                grew = (
                    ((lo < b0_lb - 1e-9) | (hi > b0_ub + 1e-9)) & np.isfinite(lo) & np.isfinite(hi)
                )
            if grew.any():
                j = int(np.flatnonzero(grew)[0])
                led.contraction_violations.append(
                    f"{name} GREW var[{j}]: [{b0_lb[j]:.12g},{b0_ub[j]:.12g}] "
                    f"-> [{lo[j]:.12g},{hi[j]:.12g}]"
                )

        # I3 — kernel monotonicity: Kernel(Python(B0)) must be inside Kernel(B0).
        led.monotonicity_checks += 1
        tol_lo = _REL * np.maximum(1.0, np.abs(k_lb))
        tol_hi = _REL * np.maximum(1.0, np.abs(k_ub))
        finite = np.isfinite(s_lb) & np.isfinite(s_ub) & np.isfinite(k_lb) & np.isfinite(k_ub)
        # A NaN bound is its own defect, not a parity failure; comparisons are
        # masked to the finite coordinates, and numpy's warning on the masked-out
        # arithmetic is suppressed rather than left to pollute every run.
        with np.errstate(invalid="ignore"):
            looser = ((s_lb < k_lb - tol_lo) | (s_ub > k_ub + tol_hi)) & finite
        if looser.any():
            j = int(np.flatnonzero(looser)[0])
            led.monotonicity_violations.append(
                f"var[{j}]: Kernel(Python(B0))=[{s_lb[j]:.12g},{s_ub[j]:.12g}] is LOOSER "
                f"than Kernel(B0)=[{k_lb[j]:.12g},{k_ub[j]:.12g}]"
            )

        # I4 — the documented Card 2b asymmetry: shipped strictly inside
        # kernel-alone means the Python pass contributed an inference the kernel
        # does not make.
        with np.errstate(invalid="ignore"):
            strict = ((s_lb > k_lb + tol_lo) | (s_ub < k_ub - tol_hi)) & finite
        n_strict = int(strict.sum())
        if n_strict:
            led.python_only_nodes += 1
            led.python_only_bounds += n_strict

        # I2 material, checked once the reference point is known.
        led.captured.append((b0_lb, b0_ub, s_lb, s_ub, k_lb, k_ub))
        return d_s

    _solver._tighten_node_bounds_with_status = py_wrap
    PyModelRepr.in_tree_presolve = itp_wrap
    try:
        model = from_nl(str(_instance_path(instance)))
        result = model.solve(time_limit=budget)
    finally:
        _solver._tighten_node_bounds_with_status = orig_py
        PyModelRepr.in_tree_presolve = orig_itp

    led.nlp_bb_loop = bool(getattr(result, "nlp_bb", False))
    return led, result


# Module-level totals so the final check can prove the file measured something.
TOTALS = {
    "instances": 0,
    "decided_nodes": 0,
    "bounds_compared": 0,
    "contraction_checks": 0,
    "monotonicity_checks": 0,
    "soundness_checks": 0,
    "python_only_nodes": 0,
    "spatial_instances": 0,
    "nlp_bb_instances": 0,
    "native_calls": 0,
    "native_served": 0,
    "native_checks": 0,
}


@pytest.mark.slow
@pytest.mark.parametrize("instance", PARITY_INSTANCES)
def test_node_tightening_parity(instance):
    """I1–I4 on one instance's real node stream."""
    led, result = _capture(instance, _BUDGET_S)

    TOTALS["instances"] += 1
    TOTALS["decided_nodes"] += led.decided_nodes
    TOTALS["bounds_compared"] += led.bounds_compared
    TOTALS["contraction_checks"] += led.contraction_checks
    TOTALS["monotonicity_checks"] += led.monotonicity_checks
    TOTALS["python_only_nodes"] += led.python_only_nodes
    if led.nlp_bb_loop:
        TOTALS["nlp_bb_instances"] += 1
    else:
        TOTALS["spatial_instances"] += 1

    # ---- I2: soundness floor, using the solve's own incumbent as the witness --
    x = getattr(result, "x", None)
    if isinstance(x, dict) and x and led.eval_model is not None and led.captured:
        n_flat = int(led.captured[0][0].size)
        xv, mask = _witness(led.eval_model, x, n_flat)
        if not mask.any():
            led.witness_skipped = len(led.captured)
        for b0_lb, b0_ub, s_lb, s_ub, k_lb, k_ub in led.captured:
            if b0_lb.size != n_flat or not _contains(b0_lb, b0_ub, xv, mask):
                # The witness is not in this node's subtree; nothing to assert.
                led.witness_skipped += 1
                continue
            led.soundness_checks += 2
            if not _contains(s_lb, s_ub, xv, mask):
                led.soundness_violations.append("shipped stack dropped the incumbent")
            if not _contains(k_lb, k_ub, xv, mask):
                led.soundness_violations.append("kernel-alone dropped the incumbent")
    TOTALS["soundness_checks"] += led.soundness_checks

    rate = led.python_only_nodes / led.decided_nodes if led.decided_nodes else 0.0
    print(
        f"[parity] {instance}: loop={'nlp_bb' if led.nlp_bb_loop else 'spatial'} "
        f"status={result.status} decided_nodes={led.decided_nodes} "
        f"bounds={led.bounds_compared} unmatched={led.unmatched} "
        f"python_only_nodes={led.python_only_nodes} ({rate:.1%}) "
        f"python_only_bounds={led.python_only_bounds} "
        f"soundness_checks={led.soundness_checks}"
    )

    # I1 — hard.
    assert not led.contraction_violations, (
        f"{instance}: a node-tightening stack GREW a box "
        f"({len(led.contraction_violations)} violations): "
        f"{led.contraction_violations[:3]}"
    )
    # I2 — hard, and never to be relaxed (CLAUDE.md §1).
    assert not led.soundness_violations, (
        f"{instance}: a node-tightening stack removed a known-feasible point "
        f"({len(led.soundness_violations)} violations): {led.soundness_violations[:3]}. "
        "This is a false-infeasibility / false-optimality bug, not a tuning issue."
    )
    # I3 — hard.
    assert not led.monotonicity_violations, (
        f"{instance}: the Rust node kernel is NOT monotone in its input box "
        f"({len(led.monotonicity_violations)} violations): "
        f"{led.monotonicity_violations[:3]}. 'The kernel is at least as tight' "
        "cannot be relied on while this holds."
    )
    # I4 — ceiling, not equality: Card 2b established the gap exists. Applied
    # per instance only above _MIN_NODES_FOR_RATE decided nodes; below that a
    # single node moves the rate by tens of points and the assertion would be
    # noise, so those instances are judged only through the pooled rate in
    # ``test_parity_probe_actually_decided_nodes``.
    if led.decided_nodes >= _MIN_NODES_FOR_RATE:
        assert rate <= _PYTHON_ONLY_NODE_RATE_CEILING, (
            f"{instance}: Python-only-inference node rate {rate:.1%} over "
            f"{led.decided_nodes} decided nodes exceeds the "
            f"{_PYTHON_ONLY_NODE_RATE_CEILING:.0%} envelope. The kernel has LOST "
            "inferences it used to make — a new divergence, not the documented one "
            "(Card 2b: 18.6% over 25 instances)."
        )


#: Instances the native spatial kernel actually SERVES, drawn from the Phase 5.1
#: coverage census (``reports/phase5_kernel_coverage_census_c346fd73.json``:
#: 20 of 119 served, 397.7 s of baseline wall). Card 3c parametrized this arm on
#: ``nvs05``/``st_e05`` and recorded that the kernel served zero solves; the census
#: showed that was a two-instance sampling artifact (``nvs05`` declines
#: ``term_trilinear``, ``st_e05`` declines ``blf_row_count:6``) on a corpus where
#: 1 in 6 instances IS served. An arm that only ever exercises declines compares
#: the Python fallback against itself — the CLAUDE.md §6 failure mode — so the
#: parametrization now carries both kinds:
#:
#: * ``st_e13`` / ``dispatch`` / ``nvs13`` — SERVED (census wall 1.06 / 1.45 /
#:   2.41 s, all ``optimal``), so the ON arm really is the kernel's certificate;
#: * ``nvs05`` / ``st_e05`` — DECLINED, kept deliberately, because "the producer
#:   declines and the fallback still certifies" is itself a property worth guarding
#:   and is what the original arm proved.
_NATIVE_SERVED = ("st_e13", "dispatch", "nvs13")
_NATIVE_DECLINED = ("nvs05", "st_e05")


@pytest.mark.slow
@pytest.mark.parametrize("instance", _NATIVE_SERVED + _NATIVE_DECLINED)
def test_native_spatial_kernel_agrees_end_to_end(instance):
    """The fourth engine, compared where it can be: at the certificate.

    ``propagate_spec_fixpoint`` — the native spatial kernel's node propagation —
    has no PyO3 binding, so it cannot be compared box-by-box the way the other
    stacks are.  What *can* be asserted without a Rust change is the property the
    per-node comparison exists to protect: flipping
    ``DISCOPT_NATIVE_SPATIAL_KERNEL`` must not change the certified answer.  When
    a binding is added (Phase 5), this test should be upgraded to the box-level
    comparison above; until then it is honest about being end-to-end.

    Crucially it also asserts the ON arm **actually took the kernel**.  The
    producer declines feature-unsafe models (``_native_kernel_feature_safe``), so
    an ON arm that quietly fell back to the Python loop would compare the default
    path against itself and pass while measuring nothing — exactly the failure
    CLAUDE.md §6 is about.
    """
    import discopt.solver as _solver
    from discopt.modeling.core import from_nl

    path = _instance_path(instance)
    results = {}
    engaged = {}

    orig_try = _solver._try_native_spatial_kernel

    for flag in ("0", "1"):
        calls = {"n": 0, "served": 0}

        def counting_try(*a, _c=calls, **kw):
            _c["n"] += 1
            out = orig_try(*a, **kw)
            if out is not None:
                _c["served"] += 1
            return out

        prev = os.environ.get("DISCOPT_NATIVE_SPATIAL_KERNEL")
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = flag
        _solver._try_native_spatial_kernel = counting_try
        try:
            results[flag] = from_nl(str(path)).solve(time_limit=_BUDGET_S)
        finally:
            _solver._try_native_spatial_kernel = orig_try
            if prev is None:
                os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
            else:
                os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = prev
        engaged[flag] = dict(calls)

    off, on = results["0"], results["1"]
    print(
        f"[parity-native] {instance}: engaged={engaged} | "
        f"OFF status={off.status} obj={off.objective} bound={off.bound} "
        f"nodes={off.node_count} | ON status={on.status} obj={on.objective} "
        f"bound={on.bound} nodes={on.node_count}"
    )
    TOTALS["native_calls"] += engaged["1"]["n"]
    TOTALS["native_served"] += engaged["1"]["served"]

    checks = 0
    if off.objective is not None:
        checks += 1
        assert on.status != "infeasible", (
            "native kernel ON reports infeasible where OFF found an incumbent"
        )
    if on.objective is not None:
        checks += 1
        assert off.status != "infeasible", (
            "native kernel OFF reports infeasible where ON found an incumbent"
        )
    # A certified bound from either arm must not exceed the other's incumbent
    # (minimisation): the certificate invariant CLAUDE.md §1 names.
    pairs = (
        (off, on, "OFF-bound vs ON-incumbent"),
        (on, off, "ON-bound vs OFF-incumbent"),
    )
    for a, b, name in pairs:
        if a.bound is not None and b.objective is not None and getattr(a, "gap_certified", False):
            checks += 1
            assert a.bound <= b.objective + 1e-4 * max(1.0, abs(b.objective)), (
                f"{name}: certified bound {a.bound} exceeds the other arm's incumbent {b.objective}"
            )
    # An instance the census recorded as SERVED must still be served here, or this
    # arm has silently degraded back to comparing the fallback against itself. A
    # coverage REGRESSION in the producer would show up exactly this way.
    if instance in _NATIVE_SERVED:
        assert engaged["1"]["served"] > 0, (
            f"{instance} is on the Phase 5.1 served list "
            f"(reports/phase5_kernel_coverage_census_c346fd73.json) but the kernel "
            f"served 0 of {engaged['1']['n']} producer calls here — either kernel "
            "coverage regressed or this instance no longer belongs on the list. "
            "Do not silence this by shortening the list; re-run the census."
        )
    assert checks > 0, "the native-kernel arm asserted nothing"
    TOTALS["native_checks"] += checks


@pytest.mark.slow
def test_parity_probe_actually_decided_nodes():
    """CLAUDE.md §6: a probe that decided no nodes is a FAILURE, not a pass.

    Ordered last so the per-instance tests have run.  It also asserts that both
    Python loops were exercised — a run that only ever touched one of them would
    silently leave the other engine unguarded, which is precisely the hole this
    file exists to close.
    """
    print(f"[parity-totals] {TOTALS}")
    assert TOTALS["instances"] > 0, "no parity instance ran"
    assert TOTALS["decided_nodes"] > 0, (
        "zero nodes decided — the wrappers never fired, so every assertion above was vacuous"
    )
    assert TOTALS["bounds_compared"] > 0, "zero bound comparisons executed"
    assert TOTALS["monotonicity_checks"] > 0, "monotonicity never checked"
    assert TOTALS["contraction_checks"] > 0, "contraction never checked"
    assert TOTALS["spatial_instances"] > 0 and TOTALS["nlp_bb_instances"] > 0, (
        f"only one Python loop was exercised: {TOTALS['spatial_instances']} spatial / "
        f"{TOTALS['nlp_bb_instances']} nlp_bb. The parity guard must cover both."
    )
    pooled = TOTALS["python_only_nodes"] / TOTALS["decided_nodes"]
    print(f"[parity-totals] pooled Python-only-inference node rate: {pooled:.1%}")
    assert pooled <= _PYTHON_ONLY_NODE_RATE_CEILING, (
        f"pooled Python-only-inference rate {pooled:.1%} exceeds the "
        f"{_PYTHON_ONLY_NODE_RATE_CEILING:.0%} envelope — the kernel lost inferences."
    )
    assert TOTALS["native_checks"] > 0, "the native-kernel arm asserted nothing"
    # Card 3c printed a NOTE here because it measured zero served solves. Phase
    # 5.1's census showed that was a sampling artifact — the kernel serves 20 of
    # 119 corpus instances — so the parametrization now includes served instances
    # and this is a hard assertion. A run in which the kernel serves nothing means
    # the native arm proved only that the fallback is safe, which is a strictly
    # weaker claim than the one this file is named for.
    print(
        f"[parity-native] served {TOTALS['native_served']} of "
        f"{TOTALS['native_calls']} producer calls"
    )
    assert TOTALS["native_served"] > 0, (
        f"the native spatial kernel served ZERO solves across "
        f"{TOTALS['native_calls']} producer calls. Per the Phase 5.1 census "
        "(20/119 served) at least the _NATIVE_SERVED instances must engage it; "
        "zero means kernel coverage regressed and this arm is now vacuous."
    )


@pytest.mark.smoke
def test_milp_driver_node_propagation_is_off_by_default():
    """The fourth stack is *declared*, not silently ignored.

    The review lists the MILP driver as a fourth node-tightening stack, but its
    node propagation ships default-false, so on defaults there is no MILP-driver
    node box to compare.  Asserting that here means the day it graduates ON, this
    test fails and forces the driver into the comparison above rather than
    letting it join the corpus unguarded.
    """
    import re

    import discopt._rust as _rust

    for fn_name in ("solve_milp_py", "solve_milp_sparse_py"):
        fn = getattr(_rust, fn_name, None)
        if fn is None:
            continue
        sig = str(getattr(fn, "__text_signature__", "") or "")
        assert "node_propagation" in sig, (
            f"{fn_name} no longer exposes `node_propagation`; the MILP driver's "
            "node-tightening stack changed shape — extend the Card 3c parity test."
        )
        m = re.search(r"node_propagation=(\w+)", sig)
        assert m is not None, f"could not read node_propagation's default from {sig!r}"
        assert m.group(1) in ("False", "false"), (
            f"{fn_name}: MILP-driver node propagation now defaults to {m.group(1)!r}. "
            "It is now an engaged node-tightening stack and must be added to the "
            "parity comparison in this file (Card 3c / review §2.5.1)."
        )
        print(f"[parity-milp] {fn_name}: node_propagation default = {m.group(1)}")
