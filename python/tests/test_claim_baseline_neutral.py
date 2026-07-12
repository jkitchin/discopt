"""The committed claim baseline must match the current build (issue #632, R0.3).

Until a bound-changing cutover lands (R1.2+), the canonical work changes no solver
math, so every vendored instance must fingerprint **byte-identically** to
``docs/dev/data/claim-baseline.jsonl``. This test is the standing differential
gate: a spurious ``changed`` here means either an unintended relaxation drift or a
stale baseline. Bound-changing stages will re-point their own tests at the changed
bucket with per-instance attribution; this one stays as the neutrality guard for
the unchanged bucket.
"""

from __future__ import annotations

import pytest
from support.claim_differential import load_baseline, partition_corpus

pytestmark = [pytest.mark.claim_boundary]


def test_current_build_matches_committed_baseline():
    baseline = load_baseline()
    assert baseline, "claim-baseline.jsonl is empty or missing"
    buckets = partition_corpus(baseline)
    changed = buckets["changed"]
    errored = buckets["error"]
    assert not changed, "relaxation drifted vs committed baseline: " + "; ".join(
        f"{d.instance} ({d.detail})" for d in changed
    )
    assert not errored, "instances failed to build vs baseline: " + "; ".join(
        f"{d.instance} ({d.detail})" for d in errored
    )
    # Sanity: the bulk of the corpus is actually compared (not all skipped).
    assert len(buckets["unchanged"]) >= 50
