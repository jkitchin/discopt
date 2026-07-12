"""The committed claim baseline must match the current build's SHAPE (#632, R0.3).

Until a bound-changing cutover lands (R1.2+), the canonical work changes no solver
math, so every vendored instance's relaxation must keep the same **shape** (row /
column / integer-column counts) as ``docs/dev/data/claim-baseline.jsonl`` — shape
is what a claim or structural change moves, and it is stable across environments.

Deliberately NOT gated here: the exact float **fingerprint**. The in-house
FBBT/parse path produces last-digit-different matrix coefficients across Rust
builds/platforms (``contvar``/``tanksize`` drift with identical shape — confirmed
on a pristine tree), so a committed-hash equality check is not reproducible on a
different CI runner. Environment-independent byte-identity is guarded instead by
the in-process ``test_lr2_offneutral_relaxation.py`` (#630, OFF-vs-code-absent in
one process) and, for the cutover, by the canonical-ON-vs-OFF in-process
differential gate (R1.2). Fingerprint drift with identical shape is surfaced here
as an informational count, not a failure.
"""

from __future__ import annotations

import pytest
from support.claim_differential import load_baseline, partition_corpus

pytestmark = [pytest.mark.claim_boundary]


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
