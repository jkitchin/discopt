#!/usr/bin/env python3
"""Flag-graduation gate (G1.2) — the per-flag validation instrument + ledger.

**The process gap this closes (cause C-B).** discopt's bound-changing regime
(CLAUDE.md §5) correctly parks each new capability behind a default-OFF env flag
"pending 3 green nightlies". But no nightly pipeline exists, so **nothing ever
graduates** — five sound, class-validated capabilities are inert on main. This
script is the missing instrument: it runs, *per parked flag*, the evidence a
graduation PR needs, emits a machine-readable verdict, and appends it to a
durable ledger so "3 consecutive green" is checkable. The flip itself stays a
separate, reviewed PR — this gate produces the evidence, it does not automate the
decision.

**Per flag it runs three checks** (all infra; NO solver-math change):

1. **Held-out per-flag arm** — via :mod:`generality_sweep` ARMS: solve a seeded,
   held-out sample (excludes the 61-panel + named probes) OFF vs the flag ON *in
   isolation* (every other capability at its default OFF — the isolation the N=20
   pilot lacked). Yields ``benefit_fraction`` / ``regression_rate`` /
   structural-prevalence and any soundness violation (oracle-bracket crossing /
   false-optimal in either config).

2. **Cert-panel neutrality** — re-runs :mod:`check_cert_neutrality` **in a fresh
   subprocess with the flag ON** (so the fresh interpreter reads the env flag at
   import). For a *bound-neutral* flag this must be byte-identical (node_count +
   objective) vs ``cert-baseline.jsonl``; for a *bound-changing / heuristic-policy*
   flag the certified **objective is always enforced** (soundness) while node
   drift on the cert panel is a documented, non-fatal perf note (it is expected
   where the flag's structure is present). Either way, a changed *objective* or a
   lost *optimal status* is a hard fail.

3. **incorrect_count = 0 + no oracle cross** — folded into checks (1) and (2):
   the arm's soundness bracket ( ``[=bestdual=, =best=]`` ) is the held-out
   incorrect-count guard; the cert panel's objective-to-tolerance is the panel
   incorrect-count guard.

**Eligibility.** A flag's verdict is ``eligible`` iff:

  * **0** soundness violations in the held-out arm (either config), AND
  * cert-neutral (objective unchanged + still optimal; node byte-identical too for
    a bound-neutral flag), AND
  * ``regression_rate`` at/below the documented threshold
    (:data:`MAX_REGRESSION_RATE`).

A flag is **graduation-eligible** (ready for its flip PR) only after
:data:`GREEN_STREAK_REQUIRED` *consecutive* eligible verdicts in the ledger — the
"3 green nightlies" rule made checkable.

**Corpus honesty (read §4 of the plan doc).** The held-out arm needs the full
~4,800-instance MINLPLib corpus in ``~/Dropbox/projects/discopt-minlp-benchmark``,
which **GitHub CI does not have**. So:

  * **local / nightly** (a machine with the corpus): run the *full* gate — the
    held-out arm + cert-neutrality + ledger append. ``make graduation-gate`` or
    the documented cron line drives this.
  * **CI subset** (``.github/workflows/graduation-gate.yml``): runs only the
    **cert-neutrality + incorrect_count** portion over the **vendored** cert panel
    (which IS in the repo) as a regression guard — ``--ci-subset``. It skips the
    held-out arm (no corpus) and does NOT append a ledger verdict (a nightly-only
    fact).

Usage (full, one flag, tiny smoke):
    PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/graduation_gate.py \
        --flags psd_cost_gate --n 8 --time-limit 15

Usage (CI subset — cert-neutrality only, vendored panel, no corpus):
    python discopt_benchmarks/scripts/graduation_gate.py \
        --flags all --ci-subset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_BENCH_ROOT = _SCRIPTS.parent
_REPO = _BENCH_ROOT.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import generality_sweep as gs  # noqa: E402

DEFAULT_CORPUS = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark"))
CERT_NEUTRALITY_SCRIPT = _SCRIPTS / "check_cert_neutrality.py"
LEDGER_PATH = _REPO / "docs" / "dev" / "data" / "graduation-ledger.jsonl"

# --------------------------------------------------------------------------- #
# Documented thresholds (global, not per-instance — CLAUDE.md §2). A flip PR must
# cite these; changing them is a reviewed decision, recorded in the protocol doc.
# --------------------------------------------------------------------------- #
# Max held-out regression-rate for eligibility. The pilot measured branch-and-
# reduce at 8 % and PSD at 10 % (bundled); the isolated arms must land at/below
# this. 0.10 is the documented ceiling — a flag regressing >10 % of its
# structure-carrying held-out instances is not ready to be a default.
MAX_REGRESSION_RATE = 0.10
# Consecutive green verdicts required before a flag is graduation-eligible (the
# "3 green nightlies" rule).
GREEN_STREAK_REQUIRED = 3


# --------------------------------------------------------------------------- #
# verdict record
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    flag: str
    eligible: bool
    benefit_fraction: float | None
    regression_rate: float | None
    soundness_ok: bool
    cert_neutral: bool
    # context / provenance
    regime: str = ""
    structural_prevalence: int = 0
    scored: int = 0
    n_soundness_violations: int = 0
    cert_kind: str = ""  # "byte_identical" | "objective_only" | "skipped"
    cert_violations: list[dict] = field(default_factory=list)
    benefit_instances: list[str] = field(default_factory=list)
    regression_instances: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# cert-panel neutrality — fresh subprocess so the env flag is read at import
# --------------------------------------------------------------------------- #
@dataclass
class CertResult:
    neutral: bool
    kind: str  # "byte_identical" | "objective_only" | "skipped"
    violations: list[dict]
    note: str


def run_cert_neutrality(arm: str, env_on: dict[str, str]) -> CertResult:
    """Run ``check_cert_neutrality.py`` in a fresh subprocess with the flag ON.

    The check compares the flagged re-solve of the 41-instance cert panel against
    the committed ``cert-baseline.jsonl`` and exits non-zero on any violation. For
    a bound-changing flag, a *node_count* change on a cert instance carrying the
    flag's structure is expected and NOT a soundness fault, but a changed
    *objective* or a lost *optimal* status is.

    The objective check is **regime-aware** (the CUTOFF-SOUND-1 fix). For a
    *bound-neutral* flag the underlying ``check_neutrality`` demands byte-
    reproducibility (~1e-8). For a *bound-changing* flag that would be a category
    error: the flag legitimately alters the search, so its certified objective may
    drift beyond 1e-8 while staying within *correctness* tolerance and not crossing
    the true optimum (the ex1225 31.0 / st_e38 shape — a drift *toward* =opt=). We
    therefore pass the ``bound_changing`` regime and the ``cert-optima.json`` oracle
    down to ``check_neutrality``, which flags an objective only when it disagrees
    with the TRUE optimum beyond correctness tolerance (a genuine false certificate).
    node_regression is downgraded to a perf note for a bound-changing flag."""
    regime = gs.ARMS.get(arm, {}).get("regime", "bound_changing")
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    env.update(env_on)
    # Emit machine-readable violations by importing the util in the subprocess and
    # dumping JSON — reusing check_cert_neutrality's own runner + baseline so we do
    # not fork its logic. The oracle (cert-optima.json) lets the bound-changing
    # objective check bracket against the TRUE optimum, not byte-identity.
    worker = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(_BENCH_ROOT)!r}); sys.path.insert(0, {str(_REPO)!r})\n"
        "from pathlib import Path\n"
        "from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig\n"
        "from scripts.gen_cert_baseline import _instance_budgets, _CERT_OPTIMA\n"
        "from utils.cert_neutrality import check_neutrality, load_baseline\n"
        "from scripts.check_cert_neutrality import _CERT_BASELINE, _KNOWN_PERF_GATED\n"
        "baseline = load_baseline(_CERT_BASELINE)\n"
        "_op = Path(_CERT_OPTIMA)\n"
        "oracle = json.loads(_op.read_text()) if _op.exists() else {}\n"
        f"regime = {regime!r}\n"
        "budgets = _instance_budgets(60.0)\n"
        "solver = SolverConfig(name='discopt', command='', solver_type='internal')\n"
        "new_rows = {}\n"
        "for name in sorted(baseline):\n"
        "    cfg = BenchmarkConfig(suite_name='cert-neutral',\n"
        "        time_limit=int(budgets.get(name, 60)), num_runs=1, solvers=[solver])\n"
        "    res = BenchmarkRunner(cfg)._run_discopt(solver, name, 0)\n"
        "    new_rows[name] = res.to_dict()\n"
        "viol = check_neutrality(new_rows, baseline, known_perf_gated=_KNOWN_PERF_GATED,\n"
        "    regime=regime, oracle=oracle)\n"
        "print('CERTJSON:' + json.dumps([{'instance': v.instance, 'kind': v.kind,\n"
        "    'detail': v.detail} for v in viol]))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return CertResult(False, "skipped", [], "cert-neutrality subprocess timed out (1h)")
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("CERTJSON:")),
        None,
    )
    if line is None:
        return CertResult(
            False,
            "skipped",
            [],
            f"cert-neutrality produced no JSON (stderr tail: {proc.stderr.strip()[-300:]!r})",
        )
    viol = json.loads(line[len("CERTJSON:") :])
    # Soundness-class violations (objective / status / missing) are hard fails in
    # every regime. node_regression is perf-class: fatal for a bound-neutral flag,
    # a documented note for a bound-changing / heuristic-policy flag.
    hard = [v for v in viol if v["kind"] in ("objective", "status", "missing")]
    node_only = [v for v in viol if v["kind"] == "node_regression"]
    if regime == "bound_neutral":
        neutral = not viol
        kind = "byte_identical"
        note = "byte-identical required (bound-neutral flag)"
        return CertResult(neutral, kind, viol, note)
    # bound-changing / control: objective must hold; node drift is a perf note.
    neutral = not hard
    kind = "objective_only"
    note = "certified objective + optimal-status enforced; node drift is a perf note"
    if node_only:
        note += (
            f" ({len(node_only)} instance(s) changed node_count — expected where structure present)"
        )
    return CertResult(neutral, kind, hard, note)


# --------------------------------------------------------------------------- #
# ledger (durable, append-only) + consecutive-green streak
# --------------------------------------------------------------------------- #
def append_ledger(verdict: Verdict, ts: str, meta: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": ts, **meta, **asdict(verdict)}
    with open(LEDGER_PATH, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(ln) for ln in LEDGER_PATH.read_text().splitlines() if ln.strip()]


def green_streak(flag: str, ledger: list[dict]) -> int:
    """Count trailing consecutive eligible verdicts for ``flag`` (most-recent-last
    order). Any non-eligible verdict resets the streak."""
    streak = 0
    for row in ledger:
        if row.get("flag") != flag:
            continue
        if row.get("eligible"):
            streak += 1
        else:
            streak = 0
    return streak


# --------------------------------------------------------------------------- #
# the gate, per flag
# --------------------------------------------------------------------------- #
def evaluate_flag(
    flag: str,
    instances: list,
    nl_dir: Path,
    tl: float,
    off_cache: dict,
    ci_subset: bool,
) -> Verdict:
    env_on = gs.ARMS[flag]["env"]
    regime = gs.ARMS[flag]["regime"]
    notes: list[str] = []

    # (2/3) cert-panel neutrality + incorrect_count (both regimes run this).
    cert = run_cert_neutrality(flag, env_on)
    notes.append(f"cert: {cert.note}")

    if ci_subset:
        # CI has no corpus → skip the held-out arm; verdict reports cert-only.
        # NOT ledger-appended (a nightly-only fact). eligible here means "cert
        # regression guard passed" — the CI signal, not graduation-eligibility.
        notes.append(
            "CI-SUBSET: held-out arm SKIPPED (no corpus in CI); this verdict is a "
            "cert regression guard only, NOT a graduation verdict (not ledgered)."
        )
        return Verdict(
            flag=flag,
            eligible=cert.neutral,
            benefit_fraction=None,
            regression_rate=None,
            soundness_ok=cert.neutral,
            cert_neutral=cert.neutral,
            regime=regime,
            cert_kind=cert.kind,
            cert_violations=cert.violations,
            notes=notes,
        )

    # (1) held-out per-flag arm (isolated).
    arm_rows = gs.run_arm(instances, nl_dir, flag, tl, off_cache=off_cache)
    stats = gs.arm_stats(arm_rows, flag)
    n_viol = sum(len(r.violations) for r in arm_rows)
    soundness_ok = n_viol == 0
    rr = stats.regression_rate
    regression_ok = (rr is None) or (rr <= MAX_REGRESSION_RATE)
    if stats.structural_prevalence == 0:
        notes.append(
            "held-out arm: 0 structure-carrying instances at this N — benefit/"
            "regression UNMEASURED; grow N or target the stratum to get signal."
        )
    if not regression_ok:
        notes.append(
            f"regression_rate {rr:.2f} > threshold {MAX_REGRESSION_RATE:.2f} — not eligible."
        )
    if not soundness_ok:
        notes.append(f"{n_viol} SOUNDNESS violation(s) in held-out arm — P0, not eligible.")

    eligible = soundness_ok and cert.neutral and regression_ok
    return Verdict(
        flag=flag,
        eligible=eligible,
        benefit_fraction=stats.benefit_fraction,
        regression_rate=rr,
        soundness_ok=soundness_ok,
        cert_neutral=cert.neutral,
        regime=regime,
        structural_prevalence=stats.structural_prevalence,
        scored=stats.scored,
        n_soundness_violations=n_viol,
        cert_kind=cert.kind,
        cert_violations=cert.violations,
        benefit_instances=stats.benefit_instances,
        regression_instances=stats.regression_instances,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_reports(
    verdicts: list[Verdict], args: argparse.Namespace, ts: str, out_dir: Path, ledger: list[dict]
) -> tuple[Path, Path]:
    def pct(x: object) -> str:
        return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"

    payload = {
        "timestamp": ts,
        "mode": "ci_subset" if args.ci_subset else "full",
        "n": args.n,
        "seed": args.seed,
        "time_limit": args.time_limit,
        "max_vars": args.max_vars,
        "max_regression_rate": MAX_REGRESSION_RATE,
        "green_streak_required": GREEN_STREAK_REQUIRED,
        "verdicts": [asdict(v) for v in verdicts],
        "streaks": {v.flag: green_streak(v.flag, ledger) for v in verdicts if not args.ci_subset},
    }
    js = out_dir / f"graduation_gate_{ts}.json"
    js.write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Flag-graduation gate — {'CI-subset (cert-only)' if args.ci_subset else 'full'} run",
        "",
        f"- Generated: {ts}",
        f"- Mode: **{'CI-subset' if args.ci_subset else 'full (held-out arm + cert panel)'}**",
        f"- Eligibility: 0 soundness violations AND cert-neutral AND "
        f"regression_rate ≤ **{MAX_REGRESSION_RATE:.0%}**; graduation-eligible after "
        f"**{GREEN_STREAK_REQUIRED}** consecutive green verdicts.",
        "",
        "| flag | regime | benefit | regression | soundness | cert-neutral | eligible | "
        + ("streak |" if not args.ci_subset else "|"),
        "|---|---|---|---|:-:|:-:|:-:|" + ("---|" if not args.ci_subset else ""),
    ]
    for v in verdicts:
        streak_cell = (
            f" {green_streak(v.flag, ledger)}/{GREEN_STREAK_REQUIRED} |"
            if not args.ci_subset
            else ""
        )
        lines.append(
            f"| {v.flag} | {v.regime} | {pct(v.benefit_fraction)} | {pct(v.regression_rate)} | "
            f"{'Y' if v.soundness_ok else 'N'} | {'Y' if v.cert_neutral else 'N'} | "
            f"{'YES' if v.eligible else 'no'} |{streak_cell}"
        )
    for v in verdicts:
        lines += ["", f"### {v.flag}", ""]
        if not args.ci_subset:
            lines.append(
                f"- structural prevalence {v.structural_prevalence}, scored {v.scored}, "
                f"soundness violations {v.n_soundness_violations}"
            )
            if v.benefit_instances:
                lines.append(f"- benefit: {', '.join(v.benefit_instances)}")
            if v.regression_instances:
                lines.append(f"- regression: {', '.join(v.regression_instances)}")
        if v.cert_violations:
            lines.append(f"- cert violations: {v.cert_violations}")
        for note in v.notes:
            lines.append(f"- {note}")

    md = out_dir / f"graduation_gate_{ts}.md"
    md.write_text("\n".join(lines) + "\n")
    return js, md


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--flags",
        type=str,
        default=",".join(gs.GRADUATION_ARMS),
        help=(
            "comma-separated parked flags to gate "
            f"(choices: {','.join(gs.GRADUATION_ARMS)}; or 'all' for the bundle "
            "cross-check). Default: all parked flags."
        ),
    )
    ap.add_argument("--n", type=int, default=100, help="held-out sample size (nightly default 100)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (reproducibility)")
    ap.add_argument("--time-limit", type=float, default=30.0, help="seconds per held-out solve")
    ap.add_argument("--max-vars", type=int, default=500, help="cap on variable count per instance")
    ap.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    ap.add_argument("--out-dir", type=str, default=str(_REPO / "reports"))
    ap.add_argument(
        "--ci-subset",
        action="store_true",
        help=(
            "CI-runnable subset: cert-neutrality + incorrect_count over the VENDORED "
            "cert panel only; skips the held-out corpus arm and does NOT append to "
            "the ledger. For .github CI where the ~4,800-instance corpus is absent."
        ),
    )
    ap.add_argument(
        "--no-ledger",
        action="store_true",
        help="do not append verdicts to the ledger (dry run of a full gate).",
    )
    args = ap.parse_args()

    requested = [f.strip() for f in args.flags.split(",") if f.strip()]
    valid = set(gs.GRADUATION_ARMS) | {"all"}
    unknown = [f for f in requested if f not in valid]
    if unknown:
        print(f"# ERROR: unknown flag(s): {unknown}; choices: {sorted(valid)}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")

    instances: list = []
    nl_dir = Path()
    selection_meta: dict = {}
    if not args.ci_subset:
        corpus = Path(os.path.expanduser(args.corpus))
        if not (corpus / "minlplib" / "nl").is_dir():
            print(
                f"# ERROR: corpus not found at {corpus}. The full gate needs the "
                "MINLPLib snapshot; use --ci-subset on a machine without it.",
                file=sys.stderr,
            )
            return 2
        instances, selection_meta = gs.select_instances(corpus, args.n, args.seed, args.max_vars)
        nl_dir = corpus / "minlplib" / "nl"
        print(
            f"# graduation gate (full): {len(instances)}/{args.n} held-out instances, "
            f"seed {args.seed}, tl {int(args.time_limit)}s; flags {requested}",
            flush=True,
        )
    else:
        print(
            f"# graduation gate (CI-subset): cert-neutrality over the vendored cert "
            f"panel for flags {requested} (held-out arm skipped — no corpus)",
            flush=True,
        )

    off_cache: dict = {}
    verdicts: list[Verdict] = []
    meta = {
        "mode": "ci_subset" if args.ci_subset else "full",
        "n": args.n,
        "seed": args.seed,
        "time_limit": args.time_limit,
        "max_vars": args.max_vars,
        "selection_meta": selection_meta,
    }
    for flag in requested:
        print(f"\n=== gating flag: {flag} ===", flush=True)
        v = evaluate_flag(flag, instances, nl_dir, args.time_limit, off_cache, args.ci_subset)
        verdicts.append(v)
        if not args.ci_subset and not args.no_ledger:
            append_ledger(v, ts, meta)

    ledger = read_ledger()
    js, md = write_reports(verdicts, args, ts, out_dir, ledger)

    print("\n─── verdicts ───", flush=True)
    any_hard_fail = False
    for v in verdicts:
        streak = green_streak(v.flag, ledger) if not args.ci_subset else None
        streak_s = f" streak {streak}/{GREEN_STREAK_REQUIRED}" if streak is not None else ""
        grad = (
            " → GRADUATION-ELIGIBLE"
            if streak is not None and streak >= GREEN_STREAK_REQUIRED
            else ""
        )
        print(
            f"  {v.flag:20} eligible={'YES' if v.eligible else 'no ':3} "
            f"benefit={_p(v.benefit_fraction)} regression={_p(v.regression_rate)} "
            f"soundness={'ok' if v.soundness_ok else 'FAIL'} "
            f"cert={'neutral' if v.cert_neutral else 'FAIL'}{streak_s}{grad}",
            flush=True,
        )
        # A soundness or cert-objective failure is a hard, non-zero exit (P0 / CI red).
        if not v.soundness_ok or not v.cert_neutral:
            any_hard_fail = True
    print(f"# JSON: {js}\n# MD:   {md}", flush=True)
    if not args.ci_subset and not args.no_ledger:
        print(f"# LEDGER: appended {len(verdicts)} verdict(s) to {LEDGER_PATH}", flush=True)
    return 1 if any_hard_fail else 0


def _p(x: object) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
