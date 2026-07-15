"""The committed claim baseline must match the current build's SHAPE (#632, R0.3).

The canonical work must keep each vendored instance's relaxation at the committed
**shape** (row / column / integer-column counts) in ``docs/dev/data/claim-baseline.jsonl``
— shape is what a claim or structural change moves, and it is stable across
environments. The baseline is regenerated when a *deliberate* bound-changing
change lands: the #640 S8 recovery (separable floor, quadratic RLT, the incremental
McCormick 4-row monomial hull, pure-product columns, monic-product CSE) added valid
cuts and bound-neutral column refactors that intentionally moved 35 instances'
shapes. Those changes are SOUND by the engine's construction (every emitted row is
a valid outer inequality) and are gated for soundness elsewhere (the differential
bound tests + feasible-point sweeps, ``incorrect_count = 0`` on the panels); this
shape gate now tracks the recovered baseline so a FUTURE unintended structural drift
is still caught.

Deliberately NOT gated here: the exact float **fingerprint**. The in-house
FBBT/parse path produces last-digit-different matrix coefficients across Rust
builds/platforms (``contvar``/``tanksize`` drift with identical shape — confirmed
on a pristine tree), so a committed-hash equality check is not reproducible on a
different CI runner. Fingerprint drift with identical shape is surfaced here as an
informational count, not a failure.

**What still guards coefficient-level neutrality, and what does not** (per the
#636 review): the H-LOG flag-OFF byte-identity guardrail
(``test_lr2_offneutral_relaxation.py``, #630) has been removed together with the
H-LOG flag deprecation — the log-space envelope now lives in the uniform engine,
not as an off-by-default collector, so there is nothing to prove inert. That test
was in any case NOT a frozen-reference gate: a uniform
coefficient change from a refactor would move both fingerprints identically and
still pass. At the R0 stage this PR needed no frozen-reference coefficient gate
because its only build-path change was inert instrumentation, since removed
(everything else was unwired, so it provably could not change a coefficient). The
frozen-reference *coefficient* gate
arrives with the first bound-changing cutover (R1.2), built as its own differential
gate per CLAUDE.md §5 — a **tolerance-based** coefficient comparison (``_A_ub``/
``_c`` within ~1e-9), robust to the FBBT last-digit non-determinism that makes an
exact committed hash unachievable cross-build.
"""

from __future__ import annotations

import pytest
from support.claim_differential import load_baseline, partition_corpus

# slow: rebuilds all 62 corpus relaxations in one test (~120s+), so it runs in the
# serial claim-boundary CI job (generous timeout), not the parallel python-fast job.
pytestmark = [pytest.mark.claim_boundary, pytest.mark.slow]


def test_current_build_matches_committed_baseline_shape():
    baseline = load_baseline()
    assert baseline, "claim-baseline.jsonl is empty or missing"
    buckets = partition_corpus(baseline)
    changed = buckets["changed"]
    errored = buckets["error"]
    assert not changed, "relaxation SHAPE drifted vs committed baseline: " + "; ".join(
        f"{d.instance} ({d.detail})" for d in changed
    )
    assert not errored, "instances failed to build vs baseline: " + "; ".join(
        f"{d.instance} ({d.detail})" for d in errored
    )
    # Sanity: the bulk of the corpus is actually compared (not all skipped).
    n_compared = len(buckets["unchanged"]) + len(buckets["fingerprint_drift"])
    assert n_compared >= 50
    # Informational: last-digit float drift across the build boundary is expected
    # on a few instances and is not a claim change (shape identical).
    drift = buckets["fingerprint_drift"]
    if drift:
        print(
            f"\n[info] {len(drift)} instance(s) with identical shape but drifted "
            f"matrix bytes (cross-build float noise): {[d.instance for d in drift]}"
        )
