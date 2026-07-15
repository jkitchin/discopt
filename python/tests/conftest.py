"""Configure JAX for testing."""

import os

# Force CPU backend — Metal/GPU backend is experimental and may fail.
os.environ["JAX_PLATFORMS"] = "cpu"
# Enable 64-bit precision for float64 support.
os.environ["JAX_ENABLE_X64"] = "1"

import pytest


@pytest.fixture
def heterogeneous_array_bounds():
    """Shared fixture for the X-2 "variable block treated as a scalar" class (#413).

    Returns a factory ``make(shape=(3,), lb=..., ub=...)`` that builds a fresh
    ``Model`` with a single array variable ``x`` whose elements carry *distinct*
    per-element bounds, plus a trivial linear objective/constraint so the model
    is well-formed for classify/extract/export/reformulate consumers.

    Element 0's bounds are deliberately the *tightest* so that any consumer that
    collapses the block to element 0 (``v.lb.flat[0]`` / ``.first()`` /
    block-as-scalar) is exposed: it will illegally narrow the wider elements.
    Every code site that reads array-variable bounds should be exercised through
    this fixture (`.nl`/MPS/LP/GAMS export, FBBT seeding, big-M, classify).

    Returns
    -------
    callable
        ``make(shape=(3,), lb=[0,2,4], ub=[1,5,9]) -> (model, x, lb, ub)`` where
        ``lb``/``ub`` are the flattened float arrays actually applied.
    """
    import numpy as np
    from discopt import Model

    def make(shape=(3,), lb=(0.0, 2.0, 4.0), ub=(1.0, 5.0, 9.0), name="model"):
        lb_arr = np.asarray(lb, dtype=np.float64).reshape(shape)
        ub_arr = np.asarray(ub, dtype=np.float64).reshape(shape)
        if np.all(lb_arr == lb_arr.flat[0]) and np.all(ub_arr == ub_arr.flat[0]):
            raise ValueError(
                "heterogeneous_array_bounds fixture requires distinct per-element "
                "bounds (homogeneous bounds hide the collapse-to-element-0 bug)."
            )
        m = Model(name)
        x = m.continuous("x", shape=shape, lb=lb_arr, ub=ub_arr)
        first = tuple(0 for _ in shape) if len(shape) > 1 else 0
        m.minimize(x[first])
        m.subject_to(x[first] >= float(lb_arr.flat[0]))
        return m, x, lb_arr.ravel(), ub_arr.ravel()

    return make


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "correctness: Known-optimum correctness validation")
    config.addinivalue_line("markers", "minlptests: MINLPTests.jl standardized NLP/MINLP problems")
    config.addinivalue_line("markers", "integration: solver-dependent integration tests")
    config.addinivalue_line("markers", "amp_benchmark: opt-in AMP benchmark/incidence tests")
    config.addinivalue_line("markers", "requires_cyipopt: requires cyipopt/Ipopt")
    config.addinivalue_line(
        "markers", "relaxation: per-operator relaxation soundness/coverage audit"
    )
    config.addinivalue_line(
        "markers",
        "amp_cert_heavy: AMP end-to-end certification tests whose in-house MILP "
        "B&B is pathologically slow (#606) and thrashes under accumulated XLA "
        "memory when packed into a shared multi-solve worker; the CI workflow "
        "runs these in a separate single-process step (fresh interpreter, no "
        "accumulation) so they execute at true speed as real gating tests",
    )
    config.addinivalue_line(
        "markers",
        "claim_boundary: relaxation claim-arbitration tests; run SERIALLY (-n0) "
        "so claimer collisions cannot be order-masked under pytest-xdist (#632)",
    )


@pytest.fixture(autouse=True)
def _guard_discopt_env_leaks():
    """Fail any test that mutates a ``DISCOPT_*`` env var without ``monkeypatch``.

    Claim-arbitration behaviour is read fresh from ``os.environ`` on every
    relaxation build (the ``DISCOPT_*`` atom/probing gates), and there is no
    module-level caching. A test that writes ``os.environ["DISCOPT_..."]``
    directly therefore leaks that setting into *every later test in the same
    xdist worker*, silently flipping claim behaviour and producing results that
    pass under ``-n>=2`` (leaker and victim land in different workers) yet fail
    serially or in CI. That is the order-masked-collision class issue #632 is
    about.

    This autouse fixture snapshots the ``DISCOPT_*`` environment before each
    test and, afterwards, (a) restores it so one leak cannot cascade to the next
    test, and (b) **fails the leaking test** with the diff, so the mutation is
    fixed at the source (use ``monkeypatch.setenv``/``delenv``, which revert
    automatically) rather than masked. Non-``DISCOPT_*`` vars are out of scope.
    """
    import os

    before = {k: v for k, v in os.environ.items() if k.startswith("DISCOPT_")}
    try:
        yield
    finally:
        after = {k: v for k, v in os.environ.items() if k.startswith("DISCOPT_")}
        if after != before:
            added = {k: after[k] for k in after.keys() - before.keys()}
            removed = {k: before[k] for k in before.keys() - after.keys()}
            changed = {
                k: (before[k], after[k])
                for k in before.keys() & after.keys()
                if before[k] != after[k]
            }
            # Restore first so the leak cannot cascade to unrelated later tests.
            for k in after.keys() - before.keys():
                del os.environ[k]
            for k, v in before.items():
                os.environ[k] = v
            raise AssertionError(
                "test leaked DISCOPT_* environment mutations (use "
                "monkeypatch.setenv/delenv instead of writing os.environ "
                "directly, so the change reverts automatically and cannot be "
                f"order-masked under xdist): added={added} removed={removed} "
                f"changed={changed}"
            )


@pytest.fixture(autouse=True)
def _clear_multilinear_facet_cache():
    """Reset the EP4a multilinear-facet memo (`multilinear_separation._FACET_CACHE`)
    before each test.

    The cache is module-level mutable state whose *result* is byte-identical to a
    cold derivation, but whose *observable backend behaviour* is not: on a warm
    hit neither the simplex nor the POUNCE fallback runs. Tests that assert on
    which backend was invoked (``test_f3_multilinear_separation_lp_backend``'s
    ``pounce_calls >= 1``) therefore pass or fail depending on whether an earlier
    test in the same xdist worker warmed the same atom/box/point key — the exact
    order-masked-collision class the ``_guard_discopt_env_leaks`` fixture above
    guards against. Clearing before each test makes those tests deterministic;
    the intra-solve caching benefit (recurring atoms within one solve) is
    unaffected, since that recurrence is within a single test.
    """
    try:
        from discopt._jax import multilinear_separation

        multilinear_separation._FACET_CACHE.clear()
    except Exception:
        pass
    yield


def _cyipopt_available() -> bool:
    """True iff the optional cyipopt/Ipopt stack imports cleanly."""
    import importlib.util

    try:
        return importlib.util.find_spec("cyipopt") is not None
    except (ImportError, ValueError):
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``requires_cyipopt`` tests when cyipopt/Ipopt is unavailable.

    The optional Ipopt system stack is installed via apt in CI, which can fail on
    a transient mirror outage. Marked tests should then SKIP (not ERROR) so a flaky
    install can't red the suite; locally, the same guard means the stack is only
    needed when actually running those tests.
    """
    # #632 federation cutover: build_milp_relaxation now routes through the uniform
    # factorable engine (uniform_relax.build_uniform_relaxation). The remaining
    # deferred tests fall in two honest classes, each carrying a *precise* reason
    # (see the dict below): (a) genuine deferred TIGHTNESS the static engine pass
    # does not yet reach — product-side RLT/PSD/edge-concave separators on synthetic
    # shapes, wide-unbounded-box instances (nvs05/nvs22/nvs16), piecewise/finite-
    # domain trig tables, the convex-claimer lift — all SOUND (feasible-point clean,
    # bound never crosses the oracle; verified) but looser, deferred to the uniform
    # OA loop (blueprint S8); and (b) federation-only MACHINERY the engine bypasses
    # by construction (incremental McCormick node patching, the cut-pool / LP-spatial
    # inheritance path). The soundness content these once carried is covered by
    # test_uniform_relax.py (per-kind feasible-point soundness, corpus 0-fallback)
    # and by the *_is_sound companion tests in the same files (which still run and
    # pass). Marked run=False so the (deterministic) failure does not execute; a
    # later engine that reaches the tightness will xpass and flag the removal.
    for item in items:
        reason = _CUTOVER_DEFERRED_TESTS.get(item.originalname)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False, run=False))

    if _cyipopt_available():
        return
    skip_no_cyipopt = pytest.mark.skip(reason="cyipopt/Ipopt not installed")
    for item in items:
        if "requires_cyipopt" in item.keywords:
            item.add_marker(skip_no_cyipopt)


# Tests deferred by the #632 uniform-engine cutover, each with a PRECISE reason
# (matched by function name, parametrisation-agnostic). Recovery/un-deferral is
# tracked in issue #640 (grouped by capability bucket, tied to blueprint S8).
# Two classes only:
#   (T) deferred TIGHTNESS — sound (feasible-point clean, bound never crosses the
#       oracle) but looser than the old separators; recovered by the uniform OA
#       loop / branch-and-reduce (blueprint S8), NOT by faking the bound; and
#   (M) federation-only MACHINERY the engine bypasses by construction.
# Whole-function deferrals live here; parametrised partial failures (e.g. only the
# nvs05/nvs22 wide-box instances of a corpus-parametrised soundness test) are marked
# xfail on the specific pytest.param IN the test file so the passing params keep
# running. See pytest_collection_modifyitems.
_CUTOVER_DEFERRED_TESTS: dict[str, str] = {
    # NOTE (#640 Bucket 1 — CLOSED): all eight Bucket 1 tests are un-deferred.
    #   * separable-objective floor (test_integer_affine_cos_*,
    #     test_x_exp_minlptests_*) — the federation separable-objective analyzer was
    #     ported back into milp_relaxation.py and wired into build_uniform_relaxation
    #     as a sound ``obj_lin >= sep_lb`` cut (added only when it strictly improves
    #     the box floor);
    #   * finite-domain trig-square selector tables
    #     (test_finite_domain_trig_square_tables_*) — exact one-hot ``λ`` encodings
    #     of sin/cos(int-affine)^2 over a small integer domain, emitted by the engine;
    #   * fixed-argument tightening (test_mixed_curvature_tan_*) — recovered EXACTLY
    #     by ``_fix_single_var_equalities`` collapsing an ``x == c`` box to a point;
    #   * the remaining piecewise-partition-structure tests
    #     (test_continuous_trig_square_*, test_mixed_curvature_affine_trig_*,
    #     test_partitioned_square_secants_*, test_gas_square_difference_*) were
    #     converted to assert the SOUND-but-looser contract the uniform engine
    #     produces (valid bound + envelope/constraint enforced), since the deleted
    #     piecewise-MILP tables are intentionally not reproduced (issue #640 DoD).
    # ── Bucket 2 (RLT/PSD/product-side) — RECOVERED except nvs22 ──
    # * PSD: registering the PURE product column for a SCALED bilinear/product in
    #   uniform_relax._build_product (separators need the column to equal x_i·x_j,
    #   not scalar·x_i·x_j; bound-neutral) recovered
    #   test_psd_cut_closes_indefinite_qcqp_root_gap,
    #   test_separator_emits_no_cut_at_a_consistent_moment_point,
    #   test_psd_closes_plain_mccormick_root_gap, and (side effect) ex1252.
    # * RLT: the quadratic constraint-factor RLT pass (uniform_relax.
    #   _emit_quadratic_rlt, gated on rlt_level1 + DISCOPT_RLT_QUAD) recovered
    #   test_quadratic_rlt_build_path_emits_lifted_rows and
    #   test_rlt_wide_box_lp_not_false_infeasible (the RLT audit in test_rlt_api.py
    #   verifies no cut removes a feasible point).
    # * nvs22: the node solver now falls back to the relaxation's rigorous
    #   box-interval objective floor when the conditioning clamp makes the fast
    #   simplex spuriously report ``unbounded`` on a provably-bounded objective —
    #   a sound global lower bound (test_nvs22_objective_term_lifts_to_sound_root_bound).
    # Bucket 2 is CLOSED.
    # ── Bucket 3 (incremental per-node McCormick caching) — RECOVERED ──
    # The incremental patch (incremental_mccormick.py) now reproduces the uniform
    # engine's per-atom envelope row-for-row: the monomial hull is the 4-row
    # secant+3-tangent form ``_emit_1d`` emits, affine squares ``(c·x+d)**2`` are
    # registered + patched, and the incremental reference build skips the two
    # box-dependent OBJECTIVE-level tightenings the closed-form patch cannot
    # regenerate (the separable floor and the composite convex lift) — both only
    # loosen the fast-path bound, never invent one. Degenerate (fixed-variable)
    # boxes are NaN-guarded. This recovered
    # test_monomial_patch_matches_cold_build, test_cube_negative_is_concave_and_covered,
    # test_incremental_active_for_integer_qcqp,
    # test_incremental_infeasible_node_pruned_without_cold_rebuild, and
    # test_incremental_node_bound_is_sound_and_matches_cold.
    # ── cut-pool / LP-spatial / incremental machinery the engine bypasses ──
    "test_serial_convex_iteration_limit_does_not_certify": (
        "engine bypasses the serial convex-iteration cut machinery"
    ),
    "test_box_dependent_child_rows_would_be_invalid_and_are_excluded": (
        "engine bypasses the cut-inheritance child-row machinery"
    ),
    "test_root_pool_cuts_valid_on_every_child_feasible_point": (
        "engine bypasses the root-pool cut-inheritance machinery"
    ),
}
