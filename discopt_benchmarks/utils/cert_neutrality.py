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

# Objective reproducibility tolerance (matches gen_cert_baseline).
OBJ_TOL = 1e-8
OBJ_RTOL = 1e-9
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


def check_neutrality(
    new_rows: dict[str, dict],
    baseline: dict[str, dict],
    *,
    obj_tol: float = OBJ_TOL,
    obj_rtol: float = OBJ_RTOL,
    node_regression_frac: float = NODE_REGRESSION_FRAC,
    known_perf_gated: dict[str, str] | None = None,
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
        # certified objective to tolerance (soundness-class — ALWAYS enforced).
        nb, no = base.get("objective"), new.get("objective")
        if no is None and gated:
            pass  # a perf-gated instance that didn't certify has no objective to check
        elif nb is None or no is None:
            violations.append(NeutralityViolation(inst, "objective", f"objective {nb!r} -> {no!r}"))
        elif abs(no - nb) > obj_tol + obj_rtol * abs(nb):
            violations.append(
                NeutralityViolation(
                    inst, "objective", f"|Δobj|={abs(no - nb):.3e} (obj {nb} -> {no})"
                )
            )
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
