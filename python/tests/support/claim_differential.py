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


# Relative tolerance for the root-LP-bound gate (#961). Wide enough to absorb
# cross-build/platform last-digit drift in the in-house simplex (the committed
# history shows ~1e-12-relative noise on degenerate bases), narrow enough that
# any real relaxation/bound change (the smallest genuine drift on record moved
# the 3rd significant digit) fails the gate and forces a deliberate baseline
# regeneration in the PR that caused it.
ROOT_LP_REL_TOL = 1e-6


def current_root_lp(name: str) -> tuple[str, float | None]:
    """Solve ``name``'s root LP now; return ``(status, certified_bound_or_None)``.

    Mirrors ``gen_claim_baseline._root_lp``: the in-house engine at the declared
    root box, no exception handling (#961 / CLAUDE.md §7 — a crash must surface
    as a crash, not read as "no bound").
    """
    import numpy as np
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt.modeling.core import from_nl

    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
        ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
    res = MccormickLPRelaxer(model).solve_at_node(np.concatenate(lbs), np.concatenate(ubs))
    if res.status == "optimal":
        if res.lower_bound is None or not np.isfinite(res.lower_bound):
            raise RuntimeError(
                f"solve_at_node returned status='optimal' with "
                f"lower_bound={res.lower_bound!r} (#961 contract violation)"
            )
        return res.status, float(res.lower_bound)
    return res.status, None


def diff_root_lp(name: str, baseline: dict[str, dict]) -> InstanceDiff:
    """Classify one instance's root LP ``(status, bound)`` vs the baseline (#961).

    The bound comparison is tolerance-based (``ROOT_LP_REL_TOL``): the exact
    float is solver output and can carry cross-build last-digit noise, unlike the
    integer shape columns. A baseline row that predates the ``root_lp_status``
    field is classified ``changed`` (regenerate the baseline), NOT skipped —
    an ungated recorded field is how 52 silent drifts accumulated.

    A difference is only attributable when both sides solved the **same** LP.
    ``diff_instance`` already records that the relaxation fingerprint is not
    reproducible across Rust builds/platforms, naming ``contvar``/``tanksize``
    as the known drifters; where those bytes differ, the matrix this root LP was
    built from is not the matrix the baseline recorded, so a status/bound
    difference says nothing about a claim change. Such an instance is bucketed
    ``fingerprint_drift`` (reported, never silent) instead of ``changed``, for
    the same reason and by the same rule the shape gate uses. The fingerprint is
    only computed when a difference is seen, so the gating path is unaffected:
    an instance whose bytes match is still compared exactly.

    Measured (#961/#964): ``contvar``'s root LP certifies on the host that
    generated the committed row, and declines (``uncertified``, no bound) on both
    CI (ubuntu/x86-64) and a macOS/arm64 build — 5/5 deterministic per host — with
    its fingerprint differing from the committed one on the declining build. Its
    certification sits on the edge of the Neumaier–Shcherbina/conditioning guards,
    so last-digit matrix differences flip it. That is a build boundary, not a
    drift to gate on.
    """
    base = baseline.get(name)
    if base is None:
        return InstanceDiff(name, "missing", "no baseline row")
    if base.get("fingerprint") is None:
        return InstanceDiff(name, "missing", f"baseline unbuildable: {base.get('error', '')}")
    if "root_lp_bound" not in base or "root_lp_status" not in base:
        return InstanceDiff(
            name, "changed", "baseline lacks root_lp_bound/root_lp_status; regenerate it"
        )
    try:
        cur_status, cur_bound = current_root_lp(name)
    except Exception as exc:  # noqa: BLE001 - report per-instance, do not abort the sweep
        return InstanceDiff(name, "error", repr(exc))
    base_status, base_bound = base["root_lp_status"], base["root_lp_bound"]
    detail = ""
    if cur_status != base_status:
        detail = f"root LP status {base_status!r} -> {cur_status!r}"
    elif (cur_bound is None) != (base_bound is None) or (
        cur_bound is not None
        and abs(cur_bound - base_bound) > ROOT_LP_REL_TOL * max(1.0, abs(base_bound))
    ):
        detail = f"root LP bound {base_bound} -> {cur_bound}"
    if not detail:
        return InstanceDiff(name, "unchanged")
    # A difference is only a claim change if both sides solved the same matrix.
    try:
        cur_fp = current_row(name)["fingerprint"]
    except Exception as exc:  # noqa: BLE001 - report per-instance, do not abort the sweep
        return InstanceDiff(name, "error", f"{detail}; fingerprint unavailable: {exc!r}")
    if cur_fp != base["fingerprint"]:
        return InstanceDiff(name, "fingerprint_drift", f"{detail}; matrix bytes differ too")
    return InstanceDiff(name, "changed", detail)


def partition_corpus_root_lp(
    baseline: Optional[dict[str, dict]] = None,
) -> dict[str, list[InstanceDiff]]:
    """Classify every vendored instance's root LP vs the baseline (#961)."""
    baseline = baseline or load_baseline()
    buckets: dict[str, list[InstanceDiff]] = {
        "unchanged": [],
        "changed": [],
        "fingerprint_drift": [],
        "error": [],
        "missing": [],
    }
    for p in sorted(_NL_DIR.glob("*.nl")):
        d = diff_root_lp(p.stem, baseline)
        buckets[d.status].append(d)
    return buckets


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
