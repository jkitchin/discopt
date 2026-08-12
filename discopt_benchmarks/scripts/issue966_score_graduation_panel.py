"""Score the #966 graduation panel (2 budgets x 3 reps) against CLAUDE.md §5.

Bar 1 -- cert-clean (soundness, non-negotiable): every rep must have empty
``unsound`` over a NON-ZERO ``oracle_comparisons_executed`` count, no
``incumbent_verification_failed``, and on the graduation pair (cand vs base)
no ``cert_regressions``, no ``lost_incumbents``, no ``lost_bound``.

Bar 2 -- net-positive: the flags must be measurably helpful broadly, not merely
sound (the DISCOPT_CUT_INHERIT lesson). Scored with a spread across reps, per
budget, so a sign flip between budgets or a mean inside its own noise is
visible rather than averaged away.

§6: prints the number of scored artifacts and per-bar assertion counts, and
exits non-zero if it scored nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics
import sys

_RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
_PAIR = "cand vs base"
_DEFAULT_GLOB = "issue966_grad_bench*_rep*.json"


def _pair(summary: dict, name: str) -> dict:
    for p in summary["pairs"]:
        if p["pair"] == name:
            return p
    raise KeyError(f"{name} not in {[p['pair'] for p in summary['pairs']]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        default=_DEFAULT_GLOB,
        help=(
            "filename pattern under discopt_benchmarks/results/ to score "
            f"(default {_DEFAULT_GLOB!r}). Every matched artifact MUST come from the "
            "same code revision -- mixing revisions silently averages two different "
            "solvers into one verdict."
        ),
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(str(_RESULTS / args.glob)))
    if not paths:
        print(f"FAIL: no artifacts matched {args.glob!r}", file=sys.stderr)
        return 1

    runs = []
    for p in paths:
        with open(p) as fh:
            s = json.load(fh)["summary"]
        m = re.search(r"_rep(\d+)", p)
        if m is None:
            # §6: a silently-unparsed name would drop a rep from the spread and
            # narrow the noise estimate that bar 2 turns on. Refuse instead.
            print(f"FAIL: cannot parse a rep number from {p!r}", file=sys.stderr)
            return 1
        runs.append(
            {
                "path": pathlib.Path(p).name,
                "budget": s["budget"],
                "rep": int(m.group(1)),
                "s": s,
                "g": _pair(s, _PAIR),
            }
        )

    # ---------------- Bar 1: cert-clean ----------------
    checks = 0
    violations: list[str] = []
    for r in runs:
        s, g, tag = r["s"], r["g"], r["path"]
        if s["oracle_comparisons_executed"] <= 0:
            violations.append(f"{tag}: ZERO oracle comparisons -- 'unsound: []' is meaningless")
        checks += 1
        for field, src in (
            ("unsound", s),
            ("incumbent_verification_failed", s),
            ("cert_regressions", g),
            ("lost_incumbents", g),
            ("lost_bound", g),
        ):
            checks += 1
            if src[field]:
                violations.append(f"{tag}: {field} = {src[field]}")

    print("=" * 78)
    print("BAR 1 -- CERT-CLEAN (soundness)")
    print("=" * 78)
    print(
        f"{'artifact':38s} {'oracle_cmps':>11s} {'unsound':>8s} {'certreg':>8s} "
        f"{'lostinc':>8s} {'lostbnd':>8s}"
    )
    for r in runs:
        s, g = r["s"], r["g"]
        print(
            f"{r['path']:38s} {s['oracle_comparisons_executed']:>11d} "
            f"{len(s['unsound']):>8d} {len(g['cert_regressions']):>8d} "
            f"{len(g['lost_incumbents']):>8d} {len(g['lost_bound']):>8d}"
        )
    bar1 = not violations
    print(f"\nASSERTIONS_EXECUTED={checks}  VIOLATIONS={len(violations)}")
    for v in violations:
        print(f"  ! {v}")
    print(f"BAR1_CERT_CLEAN={bar1}")

    # ---------------- Bar 2: net-positive ----------------
    print()
    print("=" * 78)
    print("BAR 2 -- NET-POSITIVE (cand vs base)")
    print("=" * 78)
    print(
        f"{'artifact':38s} {'overrun_d':>10s} {'over_base':>10s} {'over_cand':>10s} "
        f"{'nodes_b':>9s} {'nodes_c':>9s} {'tight':>6s} {'loose':>6s} {'certgain':>9s}"
    )
    for r in runs:
        g = r["g"]
        print(
            f"{r['path']:38s} {g['overrun_delta_s']:>10.1f} "
            f"{g['cells_over_budget']['base']:>10d} {g['cells_over_budget']['cand']:>10d} "
            f"{g['node_count_total']['base']:>9d} {g['node_count_total']['cand']:>9d} "
            f"{len(g['tighter_bound']):>6d} {len(g['looser_bound']):>6d} "
            f"{len(g['cert_gains']):>9d}"
        )

    print()
    metric_rows = []
    for budget in sorted({r["budget"] for r in runs}):
        sub = [r for r in runs if r["budget"] == budget]
        for label, vals in (
            ("overrun_delta_s", [r["g"]["overrun_delta_s"] for r in sub]),
            ("cells_over_budget base", [r["g"]["cells_over_budget"]["base"] for r in sub]),
            ("cells_over_budget cand", [r["g"]["cells_over_budget"]["cand"] for r in sub]),
            ("node_total base", [r["g"]["node_count_total"]["base"] for r in sub]),
            ("node_total cand", [r["g"]["node_count_total"]["cand"] for r in sub]),
        ):
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            metric_rows.append((budget, label, vals, mean, sd))
            print(f"  budget={budget:<5} {label:24s} {str(vals):22s} mean={mean:9.2f} sd={sd:8.2f}")

    od = {
        b: [r["g"]["overrun_delta_s"] for r in runs if r["budget"] == b]
        for b in sorted({r["budget"] for r in runs})
    }
    signs = {b: {(1 if v > 0 else -1 if v < 0 else 0) for v in vs} for b, vs in od.items()}
    flips_within = {b: len(s) > 1 for b, s in signs.items()}
    means = {b: statistics.mean(vs) for b, vs in od.items()}
    sds = {b: statistics.stdev(vs) for b, vs in od.items()}
    flip_across = len({(1 if m > 0 else -1 if m < 0 else 0) for m in means.values()}) > 1
    inside_noise = {b: abs(means[b]) <= sds[b] for b in means}

    n_tighter = sum(len(r["g"]["tighter_bound"]) for r in runs)
    n_looser = sum(len(r["g"]["looser_bound"]) for r in runs)
    n_certgain = sum(len(r["g"]["cert_gains"]) for r in runs)
    n_better = sum(len(r["g"]["better_objective"]) for r in runs)
    n_worse = sum(len(r["g"]["worse_objective"]) for r in runs)

    print()
    print(
        "  overrun_delta_s per budget: "
        + ", ".join(f"{b}: mean={means[b]:+.2f} sd={sds[b]:.2f}" for b in means)
    )
    print(f"  sign flips WITHIN a budget across reps: {flips_within}")
    print(f"  sign flip ACROSS budget means:          {flip_across}")
    print(f"  mean inside its own rep spread:         {inside_noise}")
    print(f"  bound outcomes summed over 6 reps: tighter={n_tighter} looser={n_looser}")
    print(f"  cert_gains={n_certgain}  better_obj={n_better}  worse_obj={n_worse}")

    helpful_timing = (
        (not flip_across)
        and all(not v for v in inside_noise.values())
        and all(m < 0 for m in means.values())
    )
    bar2 = helpful_timing and n_certgain > 0
    print(f"\nBAR2_NET_POSITIVE={bar2}  (timing_helpful={helpful_timing}, cert_gains={n_certgain})")

    print()
    print("=" * 78)
    print(f"ARTIFACTS_SCORED={len(runs)}")
    print(f"GRADUATION_VERDICT={'PASS' if (bar1 and bar2) else 'FAIL'} (bar1={bar1}, bar2={bar2})")
    print("=" * 78)
    return 0 if runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
