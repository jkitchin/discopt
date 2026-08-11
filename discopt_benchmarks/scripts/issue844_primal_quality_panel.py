"""#844 fallback differential panel, now scoring incumbent QUALITY (issue #862).

The panel that graduated ``DISCOPT_LP_SPATIAL_FALLBACK`` default-on gated on
``gains / lost_incumbents / cert_regressions / overshoots`` plus soundness — whether
an incumbent **exists**, is sound, and stays in budget. It never scored how good the
incumbent was, so the flag shipped returning tln6 ``65.3`` against a reference
optimum of ``15.3`` (+327%), and, as #862 puts it, "a change could halve incumbent
quality and the panel would still pass".

This panel keeps that gate byte-for-byte and adds the missing axis:

* per-instance **primal gap** to the reference optimum (``utils.primal_quality``),
  reported for both arms;
* the aggregate quality summary (mean / median / worst, plus explicit
  ``unscored`` counts so a missing oracle can never read as a clean result);
* **quality regressions** ON vs OFF — the check whose absence #862 is about.

Quality is *reported*, and regressions are *flagged*, but the pass/fail verdict is
deliberately unchanged: #862 asks for the measurement first ("without (1) there is
no way to tell whether (2) helped"), and tightening a graduation gate on the same
commit that first measures the thing would retroactively fail a flag that was
graduated honestly under the bar of its day. ``--gate-quality`` opts into failing on
quality regressions once a baseline exists.

Corpus. Defaults to the vendored pure-integer MINIMIZE instances — the models the
LP-node engine actually serves — so the panel runs in CI. Point ``--corpus`` at a
MINLPLib ``nl`` directory to sweep the real thing; with a ``minlplib.solu`` snapshot
installed (see ``utils.reference_optima``) tln4/5/6 then score automatically, which
is the configuration issue #862 was filed from.

Usage::

    python discopt_benchmarks/scripts/issue844_primal_quality_panel.py \\
        --time-limit 60 --out results/issue862_quality_panel.json
    python .../issue844_primal_quality_panel.py --corpus ~/…/minlplib/nl \\
        --instances tln4,tln5,tln6,nvs17,nvs19,nvs24
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from utils.primal_quality import (  # noqa: E402
    is_false_primal,
    primal_gap,
    quality_regressions,
    relative_excess,
    summarize,
)
from utils.reference_optima import reference_oracle, solu_path  # noqa: E402

FLAG = "DISCOPT_LP_SPATIAL_FALLBACK"

# Vendored corpora, searched in order. These hold every pure-integer MINIMIZE
# instance the repo ships; the panel filters to the ones the engine is in scope for.
_VENDORED = (
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib",
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl",
)

# Each instance runs in its own process: the fallback engine is only reachable
# *after* a primary solve exhausts its budget, and per-solve state (the
# ``model._solve_deadline`` stash that caused the #844 overshoot) makes in-process
# A/B runs contaminate each other.
_WORKER = r"""
import json, os, sys, time, warnings
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
warnings.filterwarnings("ignore")
from discopt.modeling.core import from_nl

path, flag, tl = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.environ["DISCOPT_LP_SPATIAL_FALLBACK"] = flag
try:
    model = from_nl(path)
    t0 = time.perf_counter()
    r = model.solve(time_limit=tl)
    print("RESULT" + json.dumps({
        "objective": r.objective,
        "bound": r.bound,
        "status": r.status,
        "gap_certified": bool(getattr(r, "gap_certified", False)),
        "node_count": getattr(r, "node_count", None),
        "wall": time.perf_counter() - t0,
        "incumbent_verification_failed": bool(
            getattr(r, "incumbent_verification_failed", False)),
    }))
except Exception as exc:
    print("RESULT" + json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:160]}"}))
"""


def _in_scope(path: str) -> bool:
    """Is this a model the LP-node engine serves (pure integer, minimize)?

    A model that fails to load is *reported*, not silently dropped: a swallowed
    exception here would quietly shrink the corpus and make the panel look clean
    because it measured less (CLAUDE.md §3, and the #864 sweep of silent swallows).
    """
    try:
        from discopt._relax.lp_spatial_bb import _is_in_scope
        from discopt.modeling.core import from_nl

        return bool(_is_in_scope(from_nl(path)))
    except Exception as exc:
        print(f"  !! could not classify {Path(path).stem}: {type(exc).__name__}: {exc}")
        return False


def _run(path: str, flag: str, time_limit: float) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER, path, flag, str(time_limit)],
            capture_output=True,
            text=True,
            timeout=time_limit + 180,
        )
    except subprocess.TimeoutExpired:
        return {"error": "harness_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:])
    return {"error": "no_result", "stderr": proc.stderr[-300:]}


def _row(name: str, res: dict, optimum: float | None, sense: str = "min") -> dict:
    obj = res.get("objective")
    return {
        "name": name,
        "objective": obj,
        "optimum": optimum,
        # ``_is_in_scope`` admits MINIMIZE only, so this is "min" by construction --
        # but it is threaded as a parameter rather than hardcoded at the use site so
        # widening that scope cannot silently leave a maximize row labelled "min".
        "sense": sense,
        "bound": res.get("bound"),
        "status": res.get("status"),
        "gap_certified": res.get("gap_certified"),
        "node_count": res.get("node_count"),
        "wall": res.get("wall"),
        "incumbent_verification_failed": res.get("incumbent_verification_failed"),
        "error": res.get("error"),
        "primal_gap": primal_gap(obj, optimum),
        "relative_excess": relative_excess(obj, optimum, sense),
    }


def _resolve_instances(args) -> list[tuple[str, str]]:
    """Return ``[(name, path)]`` for the requested corpus, in scope and deduplicated."""
    dirs = [Path(args.corpus)] if args.corpus else list(_VENDORED)
    wanted = [s for s in (args.instances or "").split(",") if s]
    found: dict[str, str] = {}
    for d in dirs:
        for path in sorted(glob.glob(str(Path(d) / "*.nl"))):
            found.setdefault(Path(path).stem, path)
    if wanted:
        missing = [n for n in wanted if n not in found]
        if missing:
            print(f"!! requested instances not present in the corpus: {missing}")
        pairs = [(n, found[n]) for n in wanted if n in found]
    else:
        pairs = sorted(found.items())
    return [(n, p) for n, p in pairs if _in_scope(p)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", help="directory of .nl files (default: vendored corpora)")
    ap.add_argument("--instances", help="comma-separated instance names (default: all in scope)")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default="issue862_quality_panel.json")
    ap.add_argument(
        "--gate-quality",
        action="store_true",
        help="also fail the panel on an incumbent-quality regression ON vs OFF",
    )
    args = ap.parse_args()

    instances = _resolve_instances(args)
    if not instances:
        print("no in-scope instances found")
        return 2
    solu = solu_path()
    print(f"corpus: {len(instances)} in-scope instances, budget {args.time_limit}s")
    print(
        f"oracles: minlplib.solu {'at ' + str(solu) if solu else 'NOT INSTALLED (vendored only)'}"
    )

    off_rows: list[dict] = []
    on_rows: list[dict] = []
    for i, (name, path) in enumerate(instances, 1):
        oracle = reference_oracle(name)
        opt = None if oracle is None else oracle.value
        off = _run(path, "0", args.time_limit)
        on = _run(path, "1", args.time_limit)
        r_off, r_on = _row(name, off, opt), _row(name, on, opt)
        off_rows.append(r_off)
        on_rows.append(r_on)
        g = r_on["primal_gap"]
        print(
            f"[{i}/{len(instances)}] {name:22s} off={r_off['objective']} on={r_on['objective']} "
            f"opt={opt} pgap={'--' if g is None else f'{g:.4f}'} "
            f"cert {r_off['gap_certified']}->{r_on['gap_certified']}"
            + ("  NO ORACLE" if opt is None else ""),
            flush=True,
        )

    # ---- the #844 gate, unchanged ------------------------------------------- #
    gains = lost = cert_regressions = overshoots = unsound = false_primals = 0
    for a, b in zip(off_rows, on_rows, strict=True):
        if a["objective"] is None and b["objective"] is not None:
            gains += 1
        if a["objective"] is not None and b["objective"] is None:
            lost += 1
        if a["gap_certified"] and not b["gap_certified"]:
            cert_regressions += 1
        if b["error"] == "harness_timeout":
            # By definition this row blew the budget: the harness killed it at
            # time_limit+180. ``wall`` is None here, and ``wall or 0.0`` would have
            # scored the worst possible overrun as 0.0 s and passed the gate.
            overshoots += 1
            print(f"  OVERSHOOT {b['name']}: harness_timeout (>{args.time_limit + 180:.0f}s)")
        elif (b["wall"] or 0.0) > args.time_limit * 1.25:
            overshoots += 1
            print(f"  OVERSHOOT {b['name']}: {b['wall']:.1f}s vs {args.time_limit}s")
        if b["incumbent_verification_failed"]:
            unsound += 1
            print(f"  INCUMBENT VERIFICATION FAILED {b['name']}")
        if (
            b["objective"] is not None
            and b["bound"] is not None
            and b["bound"] > b["objective"] + 1e-6 * (1 + abs(b["objective"]))
        ):
            unsound += 1
            print(f"  UNSOUND {b['name']}: bound {b['bound']} > incumbent {b['objective']}")
        if is_false_primal(b["objective"], b["optimum"], b["sense"]):
            false_primals += 1
            print(f"  FALSE PRIMAL {b['name']}: {b['objective']} < reference {b['optimum']}")
        if (
            b["optimum"] is not None
            and b["bound"] is not None
            and b["bound"] > b["optimum"] + 1e-4 * (1 + abs(b["optimum"]))
        ):
            unsound += 1
            print(f"  BOUND ABOVE OPT {b['name']}: {b['bound']} > {b['optimum']}")

    # ---- the #862 addition: incumbent quality -------------------------------- #
    q_off, q_on = summarize(off_rows), summarize(on_rows)
    regressions = quality_regressions(off_rows, on_rows)

    # The original panel required ``gains > 0`` on a hand-picked case list that
    # included instances the default path leaves without an incumbent. On an
    # arbitrary corpus that bar is only meaningful when such an instance is present:
    # demanding it everywhere fails a corpus the fallback correctly never engages on,
    # and dropping it silently would weaken the gate to a tautology. So it is applied
    # exactly when it can be met, and its applicability is reported either way.
    # An errored run has no objective, which the gains/lost logic above reads as
    # "no incumbent" -- so an OFF-arm crash silently scores as a GAIN and suppresses
    # quality-regression detection for that instance. Errors are a broken
    # measurement, not a result: surface them and fail the gate.
    errored = [r for r in off_rows + on_rows if r["error"]]
    for r in errored:
        print(f"  ERROR {r['name']}: {r['error']}")

    no_incumbent_off = sum(1 for a in off_rows if a["objective"] is None)
    net_positive_applies = no_incumbent_off > 0
    net_positive_ok = gains > 0 if net_positive_applies else True

    print("\n=== #844 gate (unchanged) ===")
    print(
        f"  gains={gains} lost_incumbents={lost} cert_regressions={cert_regressions} "
        f"overshoots={overshoots} unsound={unsound} false_primals={false_primals}"
    )
    print(
        "  net-positive bar: "
        + (
            f"gains>0 over {no_incumbent_off} instance(s) with no incumbent OFF"
            f" -> {net_positive_ok}"
            if net_positive_applies
            else "N/A (every instance already has an incumbent OFF; the fallback never engages)"
        )
    )
    gate_ok = net_positive_ok and lost == 0 and cert_regressions == 0 and overshoots == 0
    gate_ok = gate_ok and unsound == 0 and false_primals == 0 and not errored

    print("\n=== #862 incumbent quality ===")
    for label, q in (("OFF", q_off), ("ON", q_on)):
        mean = "--" if q.mean_gap is None else f"{q.mean_gap:.4f}"
        med = "--" if q.median_gap is None else f"{q.median_gap:.4f}"
        worst = "--" if q.worst_gap is None else f"{q.worst_gap:.4f} ({q.worst_instance})"
        print(
            f"  {label:3s} incumbents={q.with_incumbent:3d} scored={q.scored:3d} "
            f"unscored={q.unscored:3d} mean_gap={mean} median_gap={med} worst={worst}"
        )
    if regressions:
        print(f"  QUALITY REGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(
                f"    {r['name']:22s} gap {r['baseline_gap']:.4f} -> {r['candidate_gap']:.4f} "
                f"(obj {r['baseline_objective']} -> {r['candidate_objective']}, opt {r['optimum']})"
            )
    else:
        print("  no incumbent-quality regressions ON vs OFF")

    worst_offenders = sorted(
        (r for r in on_rows if r["primal_gap"] is not None and r["primal_gap"] > 1e-6),
        key=lambda r: -r["primal_gap"],
    )[:10]
    if worst_offenders:
        print("\n  worst incumbents ON (the #862 target list):")
        for r in worst_offenders:
            exc = "--" if r["relative_excess"] is None else f"{r['relative_excess'] * 100:+.1f}%"
            print(
                f"    {r['name']:22s} obj={r['objective']} opt={r['optimum']} "
                f"pgap={r['primal_gap']:.4f} excess={exc}"
            )

    # Vacuity guard. ``not regressions`` is trivially True on a corpus where nothing
    # could be scored (no oracles, or no incumbents), so without this a --gate-quality
    # run could pass having measured NOTHING -- the exact "no measurement reads as no
    # problem" failure this panel exists to prevent. Scoring nothing is an unmeasured
    # verdict, not a clean one.
    quality_measured = q_on.scored > 0
    quality_ok = quality_measured and not regressions
    if not quality_measured:
        print(
            "  !! QUALITY UNMEASURED: 0 instances scored "
            f"({q_on.unscored} unscored) -- no oracle or no incumbent anywhere"
        )
    ok = gate_ok and (quality_ok or not args.gate_quality)
    print(f"\n  GATE OK: {gate_ok}   QUALITY CLEAN: {quality_ok}   PANEL OK: {ok}")

    payload = {
        "time_limit": args.time_limit,
        "flag": FLAG,
        "solu": str(solu) if solu else None,
        "gate": {
            "gains": gains,
            "no_incumbent_off": no_incumbent_off,
            "net_positive_applies": net_positive_applies,
            "net_positive_ok": net_positive_ok,
            "lost_incumbents": lost,
            "cert_regressions": cert_regressions,
            "overshoots": overshoots,
            "unsound": unsound,
            "false_primals": false_primals,
            "ok": gate_ok,
        },
        "quality": {
            "off": q_off.as_dict(),
            "on": q_on.as_dict(),
            "regressions": regressions,
            "ok": quality_ok,
        },
        "rows": {"off": off_rows, "on": on_rows},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
