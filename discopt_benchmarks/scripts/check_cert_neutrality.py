"""Check Phase 1 differential bound-neutrality against cert-baseline.jsonl.

Re-solves the deterministic-certifying subset at the baseline budgets and checks
each row against the committed baseline with the differential criteria
(objective-to-tolerance, still-optimal, node_count one-directional). Prints any
violations and exits non-zero if there are any.

Two reporting-only aids, added by #1134 because their absence turned a
one-lookup question into a bisect. Neither changes a verdict:

  * the **reference's provenance** (generating commit and host, from
    ``cert-baseline-meta.json``) is printed up front, so "is this reference stale
    relative to the tree?" is answerable without archaeology; and
  * a **host-speed calibration**, measured on the **unrouted** instances whose
    ``node_count`` reproduced *exactly* — same tree and no time spent outside it
    means the two runs did the same work, so their wall-time ratio is a clean speed
    proxy. (Equal node counts alone are not enough: an auto-routed algorithm that
    abstains at a budget checkpoint burns wall the tree does not account for. That
    is #1134's Cause 2, and it is why the routed rows are dropped.)

**#1204 promoted that calibration from an annotation to the budget itself.** It was
added here to stop a wall-clock verdict being *misread*: an instance that certifies
in 20 s of a 60 s budget on the reference machine is ``time_limit`` on a box 3x
slower, with nothing wrong in the tree — and the note said so while the violation
still stood. Annotating a verdict the box decided does not make it a verdict about
the tree. So the panel now measures this box **first**, with a bounded probe of
cheap settled rows, and scales every budget by the result, giving each instance the
reference's *work* allowance instead of its seconds. That is not a weakened guard
(CLAUDE.md §1): the comparison is unchanged and every class of finding is still
fatal here. Measured on a box 3.17x the reference, it is the difference between
``tanksize``/``nvs05`` reporting ``time_limit``/``feasible`` and both certifying the
reference's answer — see ``docs/dev/data/README-1204-calibration.md``.

Where a row still ends on the clock — a box past the cap, a probe that could not be
measured, an instance slower than the panel median — the residue is classified
rather than hidden: ``wall_limited_arms`` says which side ran out of budget, and a
certification lost at the wall is reported as ``wall_regression`` (perf-class)
because a wall-limited row is never ``optimal`` and so carries no certificate to be
false.

Usage:
    python discopt_benchmarks/scripts/check_cert_neutrality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from scripts.gen_cert_baseline import _CERT_OPTIMA, _instance_budgets  # noqa: E402
from utils.cert_neutrality import (  # noqa: E402
    PERF_CLASS_KINDS,
    check_neutrality,
    load_baseline,
    oracle_bracket_coverage,
    wall_limited_arms,
    wall_limited_rows,
)
from utils.host_calibration import (  # noqa: E402
    MIN_CALIBRATION_SAMPLES,
    HostScale,
    calibration_probe_instances,
    fit_scale_to_panel,
    host_speed_ratio,  # noqa: F401  (re-export: imported from here since #1134)
    measure_host_scale,
    scale_budgets,
)

_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"
_CERT_BASELINE_META = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline-meta.json"

# Re-exported so the name keeps working where it has always lived; the
# implementation moved to utils/host_calibration.py when #1204 made it load-bearing
# for both gate scripts rather than a post-hoc annotation in this one.
_MIN_CALIBRATION_SAMPLES = MIN_CALIBRATION_SAMPLES

# Documented performance-only regressions (soundness still enforced). T1.2's
# monomial coverage moves nvs17 from the cold path to the incremental path, which
# gives fewer nodes (205 -> 117) but a slower per-node cost from rejected warm
# starts (~45s of a 60s budget) — the T1.4 warm-start work resolves this. Tracked,
# not masked: its objective is still checked; only its wall/status is exempt.
#
# #1204 is the general form of the same problem — an instance that certifies at
# roughly half its budget on the reference machine tips on any runner ~2x slower —
# and it is handled for the whole class by calibrating the budgets to the host
# (utils/host_calibration.py). This entry stays because nvs17's slowness is a
# property of the TREE (rejected warm starts), not of the box.
_KNOWN_PERF_GATED = {
    "nvs17": "T1.2 monomial coverage -> incremental path; ~45s/60s wall pending T1.4 warm-starts",
}


def _calibrate(baseline: dict, nominal: dict, runner_solve) -> tuple[dict, HostScale]:
    """Measure this box against the reference machine and scale the budgets (#1204).

    Returns ``(budgets, scale)``. The probe is solved flag-OFF at NOMINAL budgets --
    it has to be comparable to the reference, which was generated that way -- and
    its rows are re-solved in the panel afterwards rather than reused, because the
    solver's role-2 sub-budgets are fractions of ``time_limit`` and so a row solved
    under a different budget is not the same solve even when the budget never binds.
    """
    probe = calibration_probe_instances(baseline)
    print(f"Calibration probe: {len(probe)} row(s) at nominal budgets", flush=True)
    probe_rows = {}
    for i, name in enumerate(probe, 1):
        row = runner_solve(name, nominal.get(name, 60.0))
        probe_rows[name] = row
        print(f"  [probe {i}/{len(probe)}] {name:20s} {row.get('status')}", flush=True)
    scale = measure_host_scale(probe_rows, baseline)
    print(f"  {scale.reason}", flush=True)
    fitted, note = fit_scale_to_panel(baseline, nominal, scale)
    if note:
        print(f"  {note}", flush=True)
    return scale_budgets(nominal, fitted), scale


def meta_describes_the_committed_reference(meta: dict) -> bool:
    """Whether ``meta`` is the provenance of the reference now on disk.

    ``gen_cert_baseline`` writes the meta on **every** run, including one whose
    write the shrink guard refused — deliberately, because a run that shrank
    coverage is the run whose evidence matters. So the meta on disk is the
    reference's provenance only when that run actually wrote the reference.
    ``baseline_written`` records it; for a meta written before that field existed
    it is re-derived from the guard's own condition (a run loses coverage and does
    not pass ``--allow-shrink`` ⇒ the write was refused).
    """
    written = meta.get("baseline_written")
    if written is not None:
        return bool(written)
    return not (meta.get("coverage_lost") or []) or bool(meta.get("allow_shrink"))


def provenance_lines(meta: dict | None) -> list[str]:
    """The provenance report for ``meta`` (None when the meta file is absent).

    Pure so the *stale/refused* case is testable without touching the committed
    ``docs/dev/data`` files. Reporting-only: nothing here decides a verdict.
    """
    if meta is None:
        return [
            f"  reference provenance: NONE ({_CERT_BASELINE_META.name} absent) — this "
            "reference predates the\n    #1134 provenance record, so the commit it was "
            "generated at is not recoverable from the tree."
        ]
    host = meta.get("host") or {}
    stamp = (
        f"commit {meta.get('commit')} at {meta.get('generated_at')}, "
        f"budget {meta.get('time_limit')}s (default; perf-panel instances run at their "
        f"own), host {host.get('platform')} ({host.get('cpu_count')} cpu)"
    )
    lost = meta.get("coverage_lost") or []
    if not meta_describes_the_committed_reference(meta):
        # The meta is a REFUSED run's. Attributing its commit and host to the
        # reference on disk would be a confidently wrong answer to exactly the
        # question #1134 exists to make answerable — worse than the missing answer
        # the NONE branch above gives. Say what it is instead.
        return [
            f"  reference provenance: UNKNOWN — {_CERT_BASELINE_META.name} records a "
            "REFUSED regeneration",
            f"    ({stamp}),",
            f"    which dropped {len(lost)} instance(s) the reference covered "
            f"({', '.join(lost)}) and so did NOT",
            "    overwrite cert-baseline.jsonl. The committed reference is OLDER than "
            "this record and its",
            "    own provenance is not recoverable from the tree.",
        ]
    out = [f"  reference provenance: {stamp}"]
    if lost:
        out.append(
            f"  reference was written under --allow-shrink, deliberately losing "
            f"{len(lost)} instance(s): {', '.join(lost)}"
        )
    return out


def _print_reference_provenance() -> None:
    """Print who generated the committed reference, so staleness is a lookup.

    Absent for a reference generated before #1134 added the record; that absence is
    itself the finding, and is reported rather than passed over.
    """
    meta = json.loads(_CERT_BASELINE_META.read_text()) if _CERT_BASELINE_META.exists() else None
    for line in provenance_lines(meta):
        print(line)


def main() -> int:
    baseline = load_baseline(_CERT_BASELINE)
    nominal = _instance_budgets(60.0)
    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    print(f"Neutrality check: {len(baseline)} certifying instances vs {_CERT_BASELINE.name}")
    _print_reference_provenance()

    def _solve(name: str, budget: float) -> dict:
        cfg = BenchmarkConfig(
            suite_name="cert-neutral", time_limit=int(budget), num_runs=1, solvers=[solver]
        )
        return BenchmarkRunner(cfg)._run_discopt(solver, name, 0).to_dict()

    # #1204: the budgets are wall-clock seconds chosen on the machine that generated
    # the reference, where the slowest rows certify at ~half their budget. On a box
    # ~2x slower they all tip, and "did this instance certify" stops being a fact
    # about the model. Measure this box and scale the budgets so a row gets the
    # reference's WORK allowance.
    budgets, scale = _calibrate(baseline, nominal, _solve)

    new_rows: dict[str, dict] = {}
    for i, name in enumerate(sorted(baseline), 1):
        row = _solve(name, budgets.get(name, 60.0))
        new_rows[name] = row
        b = baseline[name]
        d_obj = (
            abs(row["objective"] - b["objective"])
            if row.get("objective") is not None and b["objective"] is not None
            else float("nan")
        )
        print(
            f"  [{i}/{len(baseline)}] {name:20s} {row['status']:10s} "
            f"nodes {b['node_count']}->{row.get('node_count')}  |Δobj|={d_obj:.2e}",
            flush=True,
        )

    for inst, why in _KNOWN_PERF_GATED.items():
        if inst in baseline:
            print(f"  [perf-gated] {inst}: {why} (soundness still checked)")
    # #1187 / #1204: the wall clock decided part of this panel, and which part is
    # not something to guess at. ``wall_limited_arms`` says, per row, WHICH side ran
    # out of budget, and ``check_neutrality`` then keeps, suppresses or reclassifies
    # each check on its own merits:
    #   * both arms out of clock -> no verdict (#1187): two runs that each did what
    #     their budget allowed are not two measurements of the same search;
    #   * THIS run out of clock -> the lost certification is ``wall_regression``,
    #     perf-class. A wall-limited row is never ``optimal``, so it carries no
    #     certificate and can hide no false one (#1204);
    #   * the REFERENCE out of clock -> its node_count and incumbent are not
    #     yardsticks, but a certificate THIS run holds is still bracketed against
    #     the oracle.
    # Nothing is dropped silently: every excluded or reclassified row is printed.
    arms = wall_limited_arms(new_rows, baseline, budgets=budgets)
    unmeasured = wall_limited_rows(new_rows, baseline, budgets=budgets)
    # The oracle arms the one ABSOLUTE check here: the true optimum must lie between
    # each row's own dual bound and its incumbent. It is passed in this regime too --
    # it does not change the byte-reproducibility comparison, and it is the check no
    # exclusion above may switch off.
    oracle = json.loads(Path(_CERT_OPTIMA).read_text()) if Path(_CERT_OPTIMA).exists() else {}
    violations = check_neutrality(
        new_rows, baseline, known_perf_gated=_KNOWN_PERF_GATED, wall_limited=arms, oracle=oracle
    )
    bracketed, unbracketable = oracle_bracket_coverage(new_rows, oracle)
    print("\n─── neutrality result ───")
    print(f"  {scale.reason}")
    # An executed-assertion count, not a claim: "no violations" over zero rows read
    # is not a pass, and a run whose oracle file went missing would otherwise look
    # exactly like a clean one.
    print(
        f"  oracle bracket: {bracketed} of {len(new_rows)} row(s) checked against the true "
        f"optimum (bound <= opt <= incumbent, either sense)"
    )
    if unbracketable:
        print(f"    {len(unbracketable)} row(s) NOT checked:")
        for inst, why in sorted(unbracketable.items()):
            print(f"      {inst:20s} {why}")
    if not scale.measured:
        # Say it where the verdict is read, not only where the probe ran: an
        # uncalibrated panel is one whose wall-limited rows are the box's doing.
        print("  budgets are NOMINAL — rows below may be wall-limited by this box alone")
    if unmeasured:
        print(f"  {len(unmeasured)} instance(s) UNMEASURED (#1187) — not compared:")
        for inst, why in sorted(unmeasured.items()):
            print(f"    {inst:20s} {why}")
    one_sided = {i: w for i, w in arms.items() if not w.both}
    if one_sided:
        print(f"  {len(one_sided)} instance(s) wall-limited on ONE side (#1204):")
        for inst, w in sorted(one_sided.items()):
            print(f"    {inst:20s} {w.reason}")
    measured = len(baseline) - len(unmeasured)
    if not violations:
        print(f"  NEUTRAL over the {measured} instance(s) compared (objective to tol, "
              "still optimal, node_count not materially worse).")
        return 0
    # This script runs the BOUND-NEUTRAL regime against the committed reference,
    # where every class of finding is fatal — a refactor has no licence to change
    # anything, perf included. #1204 changes how a wall-lost certification is
    # CLASSIFIED and, via the calibration above, stops the box manufacturing them;
    # it does not change what this regime treats as a failure. The graduation gate's
    # bound-CHANGING arms are where perf-class findings become notes, and that split
    # is made there, by regime, exactly as it already is for node_regression.
    print(f"  {len(violations)} VIOLATION(S):")
    for v in violations:
        cls = "perf" if v.kind in PERF_CLASS_KINDS else "soundness"
        print(f"    {v.instance:20s} [{v.kind}/{cls}] {v.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
