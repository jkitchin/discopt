"""Differential bound-neutrality check for Phase 1 (cert:T1.2, direction (a)).

Phase 1's incremental engine reproduces the *result*, not the exact search tree
(it solves each node with the Neumaier–Shcherbina safe bound + a warm-started
basis — §0.3 mechanisms — so the tree differs *safely*). The neutrality check is
therefore differential, not exact:

  * **certified objective** unchanged to tolerance (a certified optimum jitters
    ~1e-10 across runs; bit-exact equality is not meaningful);
  * **status** still ``optimal`` (the instance still certifies);
  * **node_count** is a *one-directional* performance guard — it may improve, but
    must not get materially worse than the baseline.

The stronger "identical relaxation math" guarantee is enforced separately and
directly by ``IncrementalMcCormickLP._validate`` (row-set equality per box) and
the T0.4 ``assert_bound_sound`` differential-bound harness; this module checks the
end-to-end solve against the committed ``cert-baseline.jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

# Objective reproducibility tolerance (matches gen_cert_baseline). This is a
# *byte-reproducibility* tolerance: the baseline was produced by the same solver
# math, so a bound-*neutral* change (refactor/cache/marshaling) must reproduce the
# certified objective to ~1e-10. It is DELIBERATELY ~4 orders tighter than the
# correctness tolerance below — for a bound-neutral change, any drift beyond it is
# evidence the change altered the search, i.e. a bug (CLAUDE.md §5, bound-neutral
# regime).
OBJ_TOL = 1e-8
OBJ_RTOL = 1e-9
# Correctness tolerance (matches benchmarks.metrics.incorrect_count / conftest):
# the objective disagrees with the *true optimum* only beyond this. A bound-CHANGING
# flag (a relaxation/reduction/cut behind a default-OFF env flag) legitimately
# changes the search tree, so its certified objective may drift beyond OBJ_TOL while
# staying well within CORRECTNESS tolerance and (crucially) not crossing the true
# optimum on the worsening side. Judging such a flag against OBJ_TOL is a category
# error — it flags a *sound, more-accurate* result as a violation (the ex1225 /
# st_e38 shape). For the bound-changing regime the objective check therefore uses
# this correctness tolerance + an oracle-bracket guard (never cross =opt=).
CORRECTNESS_ATOL = 1e-4
CORRECTNESS_RTOL = 1e-3
# Allowed node_count regression before it's a violation (one-directional guard).
NODE_REGRESSION_FRAC = 0.05


@dataclass
class NeutralityViolation:
    instance: str
    kind: str  # "objective" | "status" | "node_regression" | "missing"
    detail: str


def load_baseline(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["instance"]] = d
    return out


def _objective_violation(
    inst: str,
    nb: float,
    no: float,
    *,
    regime: str,
    oracle: dict[str, float] | None,
    obj_tol: float,
    obj_rtol: float,
) -> NeutralityViolation | None:
    """Decide whether the certified objective drift ``nb -> no`` is a violation.

    - ``bound_neutral`` regime (default): byte-reproducibility. Any drift beyond
      ``obj_tol + obj_rtol*|nb|`` is a violation — a bound-neutral change must
      reproduce the certified objective exactly.
    - ``bound_changing`` regime: the flag legitimately alters the search, so a drift
      within *correctness* tolerance is expected and NOT a violation. It is a
      violation only if the new objective is a **false certificate**: it disagrees
      with the true optimum (``oracle``) by more than the correctness tolerance, OR
      (when no oracle is available for the instance) it drifts from the baseline by
      more than the correctness tolerance. This never masks a wrong answer — it only
      stops flagging a sound, tolerance-accurate drift (e.g. one that lands closer to
      or exactly on the true optimum) as a soundness fault.
    """
    if regime != "bound_changing":
        if abs(no - nb) > obj_tol + obj_rtol * abs(nb):
            return NeutralityViolation(
                inst, "objective", f"|Δobj|={abs(no - nb):.3e} (obj {nb} -> {no})"
            )
        return None
    ctol = CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(nb)
    opt = (oracle or {}).get(inst)
    if opt is not None:
        # Genuine soundness: the certified value must agree with the TRUE optimum to
        # correctness tolerance. A drift that crosses =opt= beyond tolerance is a
        # real false certificate and is still flagged.
        if abs(no - opt) > CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(opt):
            return NeutralityViolation(
                inst,
                "objective",
                f"certified obj {no} disagrees with true optimum {opt} "
                f"(|Δ|={abs(no - opt):.3e} > correctness tol) — FALSE CERTIFICATE",
            )
        return None
    # No oracle for this instance: fall back to a correctness-tolerance drift guard
    # vs the baseline (still catches a gross wrong answer; tolerant of benign jitter).
    if abs(no - nb) > ctol:
        return NeutralityViolation(
            inst,
            "objective",
            f"|Δobj|={abs(no - nb):.3e} exceeds correctness tol {ctol:.3e} "
            f"(obj {nb} -> {no}; no oracle to bracket against)",
        )
    return None


def check_neutrality(
    new_rows: dict[str, dict],
    baseline: dict[str, dict],
    *,
    obj_tol: float = OBJ_TOL,
    obj_rtol: float = OBJ_RTOL,
    node_regression_frac: float = NODE_REGRESSION_FRAC,
    known_perf_gated: dict[str, str] | None = None,
    regime: str = "bound_neutral",
    oracle: dict[str, float] | None = None,
) -> list[NeutralityViolation]:
    """Return the list of neutrality violations of ``new_rows`` vs ``baseline``.

    ``new_rows`` / ``baseline`` map instance -> a dict with at least ``status``,
    ``objective``, ``node_count`` (as produced by ``SolveResult.to_dict``). Every
    baseline instance must be present in ``new_rows``; extras in ``new_rows`` are
    ignored. An empty list means neutral.

    ``known_perf_gated`` maps instance -> reason for instances with a *documented*
    performance-only regression (a slower per-node / near-budget certification that
    a later task fixes — e.g. T1.4 warm-starts). For those, the **perf-class**
    checks (``status`` completeness, ``node_regression``) are downgraded to
    non-violations; the **soundness-class** checks (``objective``, ``missing``) are
    *always* enforced. This keeps a known, tracked perf issue from blocking a
    sound, node-improving change while never masking a wrong answer.

    ``regime`` selects the objective check (see :func:`_objective_violation`):
    ``bound_neutral`` (default) demands byte-reproducibility; ``bound_changing``
    demands agreement with the true optimum ``oracle`` (or a correctness-tolerance
    drift bound when no oracle is present). Node-regression stays a one-directional
    perf guard in both regimes; the caller decides whether to treat it as fatal.
    """
    perf_gated = known_perf_gated or {}
    violations: list[NeutralityViolation] = []
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None:
            violations.append(NeutralityViolation(inst, "missing", "absent from new run"))
            continue
        gated = inst in perf_gated
        # status: baseline rows are all certified-optimal; the new run must be too
        # (perf-class — suppressed for a documented perf-gated instance).
        if new.get("status") != "optimal" and not gated:
            violations.append(
                NeutralityViolation(
                    inst, "status", f"status={new.get('status')} (baseline optimal)"
                )
            )
        # certified objective (soundness-class — ALWAYS enforced, regime-aware tol).
        nb, no = base.get("objective"), new.get("objective")
        if no is None and gated:
            pass  # a perf-gated instance that didn't certify has no objective to check
        elif nb is None or no is None:
            violations.append(NeutralityViolation(inst, "objective", f"objective {nb!r} -> {no!r}"))
        else:
            ov = _objective_violation(
                inst, nb, no, regime=regime, oracle=oracle, obj_tol=obj_tol, obj_rtol=obj_rtol
            )
            if ov is not None:
                violations.append(ov)
        # node_count one-directional guard (perf-class — suppressed if perf-gated).
        base_nc, new_nc = base.get("node_count", 0), new.get("node_count", 0)
        if base_nc > 0 and new_nc > base_nc * (1.0 + node_regression_frac) and not gated:
            violations.append(
                NeutralityViolation(
                    inst,
                    "node_regression",
                    f"node_count {base_nc} -> {new_nc} (+{100 * (new_nc / base_nc - 1):.0f}%)",
                )
            )
    return violations
