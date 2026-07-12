"""Differential gate over the committed claim baseline (issue #632, R0.3/§3.2).

Reusable helpers a per-stage test calls to compare the *current* relaxation build
against ``docs/dev/data/claim-baseline.jsonl``:

- :func:`current_row` — build one instance and produce a baseline-shaped row.
- :func:`diff_instance` — classify one instance vs its baseline row.
- :func:`partition_corpus` — classify the whole corpus into
  ``unchanged`` / ``changed`` / ``error`` / ``missing`` buckets.

For a bound-neutral change (R0..R1.1, refactors), every instance must be
``unchanged`` (byte-identical fingerprint). For a bound-changing cutover
(R1.2 onward) the ``changed`` bucket is expected but every changed instance must
be independently attributed and re-proved sound by the calling stage's test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[3]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_BASELINE = _REPO / "docs" / "dev" / "data" / "claim-baseline.jsonl"


def baseline_path() -> Path:
    return _BASELINE


def nl_dir() -> Path:
    return _NL_DIR


def load_baseline(path: Optional[Path] = None) -> dict[str, dict]:
    """Parse the committed baseline jsonl, keyed by instance name."""
    path = path or _BASELINE
    out: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["instance"]] = row
    return out


def current_row(name: str) -> dict:
    """Build ``name`` now and produce a row in the baseline schema."""
    import numpy as np
    import scipy.sparse as sp
    from discopt._jax.claim_audit import relaxation_fingerprint
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import from_nl

    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(model)
    relax, _info = build_milp_relaxation(model, terms, DiscretizationState())
    A = relax._A_ub
    A = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
    n_int = 0
    if relax._integrality is not None:
        n_int = int(np.count_nonzero(np.asarray(relax._integrality)))
    return {
        "instance": name,
        "fingerprint": relaxation_fingerprint(relax),
        "n_rows": int(A.shape[0]) if A.ndim == 2 else 0,
        "n_cols": int(A.shape[1]) if A.ndim == 2 else int(len(relax._c)),
        "n_integer_cols": n_int,
    }


@dataclasses.dataclass(frozen=True)
class InstanceDiff:
    instance: str
    status: str  # "unchanged" | "changed" | "error" | "missing"
    detail: str = ""


def diff_instance(name: str, baseline: dict[str, dict]) -> InstanceDiff:
    """Classify one instance vs the committed baseline by relaxation **shape**.

    Shape (row/column/integer-column counts) is the cross-environment-stable
    signal a claim or structural change moves; the exact float **fingerprint** is
    NOT reproducible across Rust builds/platforms (the in-house FBBT/parse path
    produces last-digit-different matrix coefficients on ``contvar``/``tanksize``
    — confirmed on a pristine tree), so this reference test does not gate on it.
    Environment-independent byte-identity is guarded separately and in-process by
    ``test_lr2_offneutral_relaxation.py`` (#630) and, for the cutover, by the
    canonical-ON-vs-OFF in-process differential gate (R1.2). ``fingerprint_drift``
    records where the hash differs despite identical shape (informational).
    """
    base = baseline.get(name)
    if base is None:
        return InstanceDiff(name, "missing", "no baseline row")
    if base.get("fingerprint") is None:
        # Baseline itself could not build this instance; skip comparison.
        return InstanceDiff(name, "missing", f"baseline unbuildable: {base.get('error', '')}")
    try:
        cur = current_row(name)
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return InstanceDiff(name, "error", repr(exc))
    shape_keys = ("n_rows", "n_cols", "n_integer_cols")
    if any(cur[k] != base[k] for k in shape_keys):
        detail = (
            f"shape changed; rows {base['n_rows']}->{cur['n_rows']} "
            f"cols {base['n_cols']}->{cur['n_cols']} "
            f"int {base['n_integer_cols']}->{cur['n_integer_cols']}"
        )
        return InstanceDiff(name, "changed", detail)
    if cur["fingerprint"] != base["fingerprint"]:
        # Same shape, different bytes — cross-build float noise, not a claim change.
        return InstanceDiff(name, "fingerprint_drift", "identical shape; matrix bytes differ")
    return InstanceDiff(name, "unchanged")


def partition_corpus(baseline: Optional[dict[str, dict]] = None) -> dict[str, list[InstanceDiff]]:
    """Classify every vendored instance vs the baseline into status buckets."""
    baseline = baseline or load_baseline()
    buckets: dict[str, list[InstanceDiff]] = {
        "unchanged": [],
        "changed": [],
        "fingerprint_drift": [],
        "error": [],
        "missing": [],
    }
    for p in sorted(_NL_DIR.glob("*.nl")):
        d = diff_instance(p.stem, baseline)
        buckets[d.status].append(d)
    return buckets
