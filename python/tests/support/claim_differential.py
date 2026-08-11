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
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[3]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_BASELINE = _REPO / "docs" / "dev" / "data" / "claim-baseline.jsonl"
_TESTS_DIR = _REPO / "python" / "tests"
_CERT_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"


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
    from discopt._relax.claim_audit import relaxation_fingerprint
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import build_milp_relaxation
    from discopt._relax.term_classifier import classify_nonlinear_terms
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
    # Shape gate: "unchanged" | "changed" | "fingerprint_drift" | "error" | "missing".
    # Root-LP gate adds "unsound" (a host-independent property broken) and
    # "unreproduced" (same matrix bytes, different answer — #971).
    status: str
    detail: str = ""
    # #971: whether this instance's certified root LP bound was actually checked
    # against a reference optimum. Counted by the gate so the validity sweep can
    # never quietly degrade to "no oracle, nothing compared" (CLAUDE.md §6).
    oracle_checked: bool = False


# Relative tolerance for the root-LP-bound comparison (#961). Wide enough to
# absorb cross-build/platform last-digit drift in the in-house simplex (the
# committed history shows ~1e-12-relative noise on degenerate bases), narrow
# enough that any real relaxation/bound change (the smallest genuine drift on
# record moved the 3rd significant digit) is reported and forces a deliberate
# baseline regeneration in the PR that caused it.
ROOT_LP_REL_TOL = 1e-6

# #971: how many instances may land in the ``unreproduced`` bucket — a root LP
# whose (status, bound) differs from the committed row while the relaxation bytes
# match, i.e. a difference the *arithmetic path*, not the matrix, produced. This
# is an escape hatch, so it is bounded: raising it requires the same evidence any
# other gate relaxation does. Measured basis: ``contvar`` is the only corpus
# instance ever observed to do this — two ubuntu-x86-64 CI runners, identical
# fingerprint, bounds 172170.3107274997 and 187283.4711213944 (#971), each
# bit-stable on its own host — and the corpus sweep on the host that generated
# this note put all 66 instances in ``unchanged``. The headroom of 3 covers that
# class without letting a corpus-wide drift pass as "host noise". The bucket is
# loud (printed per instance), and every instance in it has already passed the
# host-independent soundness gate below — an unsound result is never excused as
# host noise.
MAX_UNREPRODUCED_ROOT_LP = 3

# Slack for the "certified root bound does not exceed the reference optimum"
# check. Same absolute/relative convention as ``metrics.incorrect_count``, the
# house-standard oracle comparison, so a violation here means the same thing it
# means on a benchmark panel: a claim above the true optimum.
OPTIMUM_ABS_TOL = 1e-4
OPTIMUM_REL_TOL = 1e-3


@lru_cache(maxsize=1)
def reference_optima() -> dict[str, float]:
    """Published global optima for corpus instances, keyed by instance name.

    Union of the two committed registries: ``docs/dev/data/cert-optima.json``
    (the corpus-wide oracle the benchmark panels' ``incorrect_count`` uses) and
    ``python/tests/data/known_optima.toml`` (the curated per-instance registry),
    with the curated file winning on the few names in both. These values are
    published optima in the instance's **reported** objective sense.
    """
    optima: dict[str, float] = {}
    if _CERT_OPTIMA.exists():
        optima.update({k: float(v) for k, v in json.loads(_CERT_OPTIMA.read_text()).items()})
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from _optima import optima_registry  # noqa: PLC0415 - tests dir must be on sys.path first

    optima.update({k: float(v["optimum"]) for k, v in optima_registry().items()})
    return optima


@dataclasses.dataclass(frozen=True)
class RootLpProbe:
    """One root LP solve, with what is needed to judge it without a baseline."""

    instance: str
    status: str
    bound: Optional[float]  # certified lower bound on the INTERNAL (minimize) objective
    maximize: bool


def current_root_lp_probe(name: str) -> RootLpProbe:
    """Solve ``name``'s root LP now and report the result plus its objective sense.

    Mirrors ``gen_claim_baseline._root_lp``: the in-house engine at the declared
    root box, no exception handling (#961 / CLAUDE.md §7 — a crash must surface
    as a crash, not read as "no bound"). ``bound`` is a lower bound on the
    *internal* objective, which is the negated one for a MAXIMIZE model; the
    sense travels with it so the validity check can compare against a published
    optimum recorded in the reported sense.
    """
    import numpy as np
    from discopt._relax.mccormick_lp import MccormickLPRelaxer
    from discopt.modeling.core import ObjectiveSense, from_nl

    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    maximize = model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE
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
        return RootLpProbe(name, res.status, float(res.lower_bound), maximize)
    return RootLpProbe(name, res.status, None, maximize)


def current_root_lp(name: str) -> tuple[str, float | None]:
    """``(status, certified_bound_or_None)`` for ``name``'s root LP (see the probe)."""
    probe = current_root_lp_probe(name)
    return probe.status, probe.bound


def root_lp_violations(probe: RootLpProbe) -> tuple[str, ...]:
    """Host-independent properties this root LP result must satisfy (#971).

    These are the things the engine *guarantees* on any machine, as opposed to the
    exact float it happens to produce on one:

    1. **The #961 status/bound contract.** ``optimal`` carries a finite bound and
       nothing else carries one; a solved-but-uncertified node is ``uncertified``.
    2. **Validity.** The certified bound is a rigorous lower bound on the root
       relaxation, hence on the true global optimum, so it may not exceed a
       published optimum (sense-corrected: for a MAXIMIZE model the internal
       objective is the negated one, so the internal lower bound must not exceed
       ``-optimum`` — the #759 framing, and the same convention as
       ``benchmarks.metrics.dual_bound_crosses_optimum``). This is the
       certificate invariant of CLAUDE.md §1, checked against the committed
       oracles rather than against a remembered float.

    Returns the violated properties (empty when the result is admissible).
    """
    import numpy as np

    out: list[str] = []
    if probe.status == "optimal":
        if probe.bound is None or not np.isfinite(probe.bound):
            out.append(f"status='optimal' with bound={probe.bound!r} (#961 contract)")
    elif probe.bound is not None:
        out.append(f"status={probe.status!r} carries bound={probe.bound!r} (#961 contract)")
    optimum = reference_optima().get(probe.instance)
    if optimum is not None and probe.bound is not None and np.isfinite(probe.bound):
        internal_opt = -optimum if probe.maximize else optimum
        slack = OPTIMUM_ABS_TOL + OPTIMUM_REL_TOL * abs(internal_opt)
        if probe.bound > internal_opt + slack:
            sense = "maximize" if probe.maximize else "minimize"
            out.append(
                f"root LP bound {probe.bound!r} exceeds the reference optimum "
                f"{optimum!r} ({sense}; internal floor {internal_opt!r}, slack {slack:.3e})"
            )
    return tuple(out)


def _oracle_checked(probe: RootLpProbe) -> bool:
    """Whether ``root_lp_violations`` actually ran its validity comparison."""
    return probe.bound is not None and probe.instance in reference_optima()


def diff_root_lp(name: str, baseline: dict[str, dict]) -> InstanceDiff:
    """Classify one instance's root LP result vs the baseline (#961, rules per #971).

    Two different questions are asked of the same single solve:

    **Is the result admissible?** — ``root_lp_violations``: the #961 status/bound
    contract, and the bound's validity against the committed reference optima.
    These hold on every machine, so a violation is a hard finding (``unsound``)
    no matter what the baseline says.

    **Does the result reproduce the committed row?** — status compared exactly,
    bound within ``ROOT_LP_REL_TOL``. A baseline row that predates the
    ``root_lp_status`` field is classified ``changed`` (regenerate the baseline),
    NOT skipped — an ungated recorded field is how 52 silent drifts accumulated.

    A difference is attributable to the code under test only when the two sides
    solved the same LP **by the same arithmetic**. Two ways that fails, both
    bucketed informationally rather than as a claim change:

    - ``fingerprint_drift`` — the relaxation bytes differ, so the matrix this root
      LP was built from is not the matrix the baseline recorded (the in-house
      FBBT/parse path is not byte-reproducible across Rust builds/platforms; see
      ``diff_instance``).
    - ``unreproduced`` (#971) — the bytes are identical and the answer still
      differs. The certified bound is the Neumaier–Shcherbina safe bound built
      from *the simplex's own dual vertex*; the optimal dual of a degenerate LP is
      not unique, so an arithmetic-path difference that flips one tie-break
      selects a different — equally valid — certificate, and the recorded float
      moves by far more than a last digit. Measured on CI: ``contvar`` produced
      ``172170.3107274997`` and ``187283.4711213944`` from an *identical*
      fingerprint on two ubuntu-x86-64 runners, deterministic per host (#971).
      ``issue971_root_lp_arithmetic_path.py`` reproduces the *mechanism* on
      demand — holding the tree, the instance and the fingerprint fixed while
      varying only the OpenBLAS kernel gives 5 kernels, 1 fingerprint, 3 distinct
      bounds — though at that grain the spread is last-digit; the CI pair is the
      evidence for the magnitude. Before #971 this case was bucketed ``changed``
      and hard-failed the gate on whichever runners took the other path.

    The escape is bounded and conditional, not a blanket skip: ``unreproduced`` is
    capped at ``MAX_UNREPRODUCED_ROOT_LP`` by the calling gate, every instance in
    it has already passed ``root_lp_violations``, and the fingerprint is computed
    only when a difference is seen, so an instance whose bytes match is still
    compared exactly.
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
        probe = current_root_lp_probe(name)
    except Exception as exc:  # noqa: BLE001 - report per-instance, do not abort the sweep
        return InstanceDiff(name, "error", repr(exc))
    checked = _oracle_checked(probe)
    violations = root_lp_violations(probe)
    if violations:
        return InstanceDiff(name, "unsound", "; ".join(violations), oracle_checked=checked)
    cur_status, cur_bound = probe.status, probe.bound
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
        return InstanceDiff(name, "unchanged", oracle_checked=checked)
    # A difference is attributable only if both sides solved the same matrix...
    try:
        cur_fp = current_row(name)["fingerprint"]
    except Exception as exc:  # noqa: BLE001 - report per-instance, do not abort the sweep
        return InstanceDiff(name, "error", f"{detail}; fingerprint unavailable: {exc!r}")
    if cur_fp != base["fingerprint"]:
        return InstanceDiff(
            name, "fingerprint_drift", f"{detail}; matrix bytes differ too", oracle_checked=checked
        )
    # ...and by the same arithmetic, which the fingerprint does not record (#971).
    return InstanceDiff(
        name,
        "unreproduced",
        f"{detail}; identical matrix bytes, so the arithmetic path differs",
        oracle_checked=checked,
    )


def partition_corpus_root_lp(
    baseline: Optional[dict[str, dict]] = None,
) -> dict[str, list[InstanceDiff]]:
    """Classify every vendored instance's root LP vs the baseline (#961/#971)."""
    baseline = baseline or load_baseline()
    buckets: dict[str, list[InstanceDiff]] = {
        "unchanged": [],
        "changed": [],
        "unsound": [],
        "unreproduced": [],
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
