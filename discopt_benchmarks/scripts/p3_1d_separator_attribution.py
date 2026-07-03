"""Phase 3 entry experiment 1d (cert:P3.1d) — SCIP per-separator attribution.

The decisive spike that follows the 1c NO-GO (``certification-gap-plan.md`` §7,
"Phase 3 1c"): 1c falsified the *reachability* hypothesis — making discopt's
cover/clique/GMI/1-row-MIR/2-row-c-MIR family reachable and armed at the root
closes ~0% of the gap SCIP closes ~100% of on the graphpart / integer-product
class. The residual is **separator DEPTH**, not plumbing. Before any invasive
Rust cut-seam work, this experiment answers the one question that pins the next
build target:

    Of SCIP's cut families, WHICH ONE(S) carry the root bound on this class?

Design — SCIP per-separator attribution at the ROOT (node limit 1), pyscipopt:

  * anchors:
      - all-OFF  : every separator disabled -> the LP/relaxation floor.
      - all-ON   : SCIP's default separator set -> the full SCIP root bound.
  * only-one-on (per family F): disable ALL separators, enable exactly F at its
      default freq -> the root gap closed BY F alone.
  * leave-one-out (per family F): all separators on, disable F -> the root gap
      LOST without F (marginal contribution).

Gap closed by a root bound B (min sense), anchored on THIS SCIP build so every
config is comparable:
      gap_closed(B) = (B - all_off) / (opt - all_off)
with ``opt`` the oracle optimum from ``minlplib.solu``. all_off is the shared LP
floor (separators off); opt - all_off is the reachable root gap.

Presolve is DISABLED across every config (``presolving/maxrounds = 0``) and
propagation is left at SCIP defaults but node-limited to 1, so the only thing
that moves the root bound between configs is the separator set — a clean
attribution. (With presolve on, a presolve reduction could masquerade as a
separator win; we want the separator signal, isolated.)

Hard guardrails (P3.1d): panel <= 6; root-only; per-instance SCIP solve 60s hard
cap; results persisted to ``discopt_benchmarks/results/`` as JSON; errored /
over-cap configs are LABELED skipped, never silently dropped. SCIP-side only —
touches no discopt solver code; SCIP reads the ``.nl`` directly, so discopt need
not import.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p3_1d_separator_attribution.py
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
_NL_DIR = _SNAPSHOT / "minlplib/nl"
_SOLU = _SNAPSHOT / "minlplib.solu"
_CERT_OPTIMA = _REPO_ROOT / "docs/dev/data/cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

# Small integer-product / graphpart panel (the 0b/1c set; fac1-3 dropped — they
# route to spatial-McCormick and are already ~1.0 closed at the root). <= 6.
PANEL = [
    "ex1263",
    "ex1263a",
    "graphpart_2pm-0044-0044",
    "graphpart_2g-0044-1601",
    "graphpart_2pm-0055-0055",
]

CAP_SECONDS = 60.0


def _load_oracle(name: str) -> float | None:
    """Return the known optimum for *name* from solu, else cert-optima.json."""
    if _SOLU.exists():
        best = None
        with _SOLU.open() as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == name:
                    tag = parts[0]
                    try:
                        val = float(parts[2])
                    except ValueError:
                        continue
                    if tag == "=opt=":
                        return val
                    if tag == "=best=" and best is None:
                        best = val
        if best is not None:
            return best
    if _CERT_OPTIMA.exists():
        data = json.loads(_CERT_OPTIMA.read_text())
        if name in data:
            return float(data[name])
    return None


def _list_separators(nl_path: Path) -> list[str]:
    """Enumerate the separator families THIS SCIP build exposes (freq params)."""
    from pyscipopt import Model

    m = Model()
    m.hideOutput()
    m.readProblem(str(nl_path))
    params = m.getParams()
    return sorted(
        {k.split("/")[1] for k in params if k.startswith("separating/") and k.endswith("/freq")}
    )


def _new_model(nl_path: Path):
    from pyscipopt import Model

    m = Model()
    m.hideOutput()
    m.readProblem(str(nl_path))
    # Root-only; presolve OFF so separators are the only mover of the root bound.
    m.setParam("limits/nodes", 1)
    m.setParam("limits/time", CAP_SECONDS)
    m.setParam("presolving/maxrounds", 0)
    return m


def _root_bound(m) -> tuple[float | None, str]:
    m.optimize()
    status = m.getStatus()
    try:
        db = float(m.getDualbound())
        if not math.isfinite(db):
            db = None
    except Exception:
        db = None
    return db, status


def _set_all_separators(m, seps: list[str], freq: int) -> None:
    """Force every separator freq to *freq* (-1 disables)."""
    for s in seps:
        # a family without a settable freq — leave it
        with contextlib.suppress(Exception):
            m.setParam(f"separating/{s}/freq", freq)


def _reset_separator(m, seps: list[str], enable: str) -> None:
    """Disable all separators except *enable*, which is reset to its default."""
    _set_all_separators(m, seps, -1)
    with contextlib.suppress(Exception):
        m.resetParam(f"separating/{enable}/freq")


def _gap_closed(bound: float | None, all_off: float | None, opt: float | None) -> float | None:
    if bound is None or all_off is None or opt is None:
        return None
    denom = opt - all_off
    if abs(denom) < 1e-9:
        return 1.0
    return (bound - all_off) / denom


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate the separator families once (build-specific).
    probe = _NL_DIR / f"{PANEL[0]}.nl"
    if not probe.exists():
        # fall back to the first existing instance
        for name in PANEL:
            if (_NL_DIR / f"{name}.nl").exists():
                probe = _NL_DIR / f"{name}.nl"
                break
    separators = _list_separators(probe)
    print(f"cert:P3.1d — SCIP per-separator attribution over {len(PANEL)} instances")
    print(f"SCIP separator families ({len(separators)}): {', '.join(separators)}\n")

    rows: list[dict] = []
    n_run = 0
    n_skip = 0

    for name in PANEL:
        nl_path = _NL_DIR / f"{name}.nl"
        row: dict = {"instance": name}
        if not nl_path.exists():
            row["status"] = "skipped"
            row["note"] = "nl-missing"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP (nl missing)")
            continue

        opt = _load_oracle(name)
        row["oracle_opt"] = opt

        t0 = time.monotonic()

        # --- anchor: all separators OFF (LP floor) ---
        try:
            m = _new_model(nl_path)
            _set_all_separators(m, separators, -1)
            all_off, st_off = _root_bound(m)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "skipped"
            row["note"] = f"scip-alloff-error: {type(exc).__name__}: {exc}"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP  all-off error: {exc}")
            continue

        # --- anchor: all separators ON (full SCIP root) ---
        try:
            m = _new_model(nl_path)
            all_on, st_on = _root_bound(m)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "skipped"
            row["note"] = f"scip-allon-error: {type(exc).__name__}: {exc}"
            rows.append(row)
            n_skip += 1
            print(f"{name:<26} SKIP  all-on error: {exc}")
            continue

        row["scip_all_off"] = all_off
        row["scip_all_on"] = all_on
        row["gap_closed_all_on"] = _gap_closed(all_on, all_off, opt)

        # --- only-one-on: disable all, enable exactly F ---
        only_one: dict[str, dict] = {}
        for f in separators:
            try:
                m = _new_model(nl_path)
                _reset_separator(m, separators, f)
                b, _ = _root_bound(m)
                only_one[f] = {
                    "bound": b,
                    "gap_closed": _gap_closed(b, all_off, opt),
                }
            except Exception as exc:  # noqa: BLE001
                only_one[f] = {
                    "bound": None,
                    "gap_closed": None,
                    "note": f"{type(exc).__name__}: {exc}",
                }

        # --- leave-one-out: all on, disable F -> marginal loss ---
        leave_one_out: dict[str, dict] = {}
        for f in separators:
            try:
                m = _new_model(nl_path)
                with contextlib.suppress(Exception):
                    m.setParam(f"separating/{f}/freq", -1)
                b, _ = _root_bound(m)
                gc = _gap_closed(b, all_off, opt)
                gc_on = row["gap_closed_all_on"]
                marginal = None
                if _finite(gc) and _finite(gc_on):
                    marginal = gc_on - gc  # gap lost when F removed
                leave_one_out[f] = {
                    "bound": b,
                    "gap_closed": gc,
                    "marginal_loss": marginal,
                }
            except Exception as exc:  # noqa: BLE001
                leave_one_out[f] = {
                    "bound": None,
                    "gap_closed": None,
                    "marginal_loss": None,
                    "note": f"{type(exc).__name__}: {exc}",
                }

        row["only_one_on"] = only_one
        row["leave_one_out"] = leave_one_out
        row["scip_status_off"] = st_off
        row["scip_status_on"] = st_on
        row["wall"] = time.monotonic() - t0
        row["status"] = "run"
        note = "" if opt is not None else "no-oracle"
        row["note"] = note
        rows.append(row)
        n_run += 1

        # Per-instance console summary: top only-one-on families + top LOO losses.
        def _fmt(x, w=8):
            return f"{x:>{w}.3f}" if _finite(x) else f"{'--':>{w}}"

        oo_sorted = sorted(
            ((f, d.get("gap_closed")) for f, d in only_one.items()),
            key=lambda kv: (kv[1] is not None, kv[1] or 0.0),
            reverse=True,
        )
        loo_sorted = sorted(
            ((f, d.get("marginal_loss")) for f, d in leave_one_out.items()),
            key=lambda kv: (kv[1] is not None, kv[1] or 0.0),
            reverse=True,
        )
        print(
            f"{name:<26} opt={_fmt(opt, 10)} off={_fmt(all_off, 10)} "
            f"on={_fmt(all_on, 10)} gc_on={_fmt(row['gap_closed_all_on'])} "
            f"({row['wall']:.1f}s) {note}"
        )
        top_oo = ", ".join(f"{f}={_fmt(g, 0).strip()}" for f, g in oo_sorted[:4] if _finite(g))
        top_loo = ", ".join(
            f"{f}={_fmt(g, 0).strip()}" for f, g in loo_sorted[:4] if _finite(g) and g > 1e-6
        )
        print(f"    only-one-on top: {top_oo or '(none close gap)'}")
        print(f"    leave-one-out loss: {top_loo or '(no family is load-bearing)'}")

    # --- Aggregate attribution across run instances ---
    run_rows = [r for r in rows if r.get("status") == "run"]

    def _agg(field: str, sub: str) -> dict[str, float]:
        """Median of *sub* over run instances, per family."""
        acc: dict[str, list[float]] = {}
        for r in run_rows:
            for f, d in r.get(field, {}).items():
                v = d.get(sub)
                if _finite(v):
                    acc.setdefault(f, []).append(v)
        out = {}
        for f, xs in acc.items():
            xs = sorted(xs)
            n = len(xs)
            out[f] = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        return out

    only_med = _agg("only_one_on", "gap_closed")
    loo_med = _agg("leave_one_out", "marginal_loss")

    print("\n--- aggregate (median over run instances) ---")
    print("only-one-on gap-closed (top 8):")
    for f, v in sorted(only_med.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        print(f"    {f:<16} {v:>7.3f}")
    print("leave-one-out marginal loss (top 8):")
    for f, v in sorted(loo_med.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        print(f"    {f:<16} {v:>7.3f}")
    print(f"\nRun: {n_run}   Skipped: {n_skip}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = _RESULTS_DIR / f"p3_1d_separator_attribution_{stamp}.json"
    payload = {
        "task": "cert:P3.1d",
        "generated": stamp,
        "panel": PANEL,
        "cap_seconds": CAP_SECONDS,
        "presolve": "disabled (presolving/maxrounds=0)",
        "scip_separators": separators,
        "n_run": n_run,
        "n_skipped": n_skip,
        "aggregate_only_one_on_median_gap_closed": only_med,
        "aggregate_leave_one_out_median_marginal_loss": loo_med,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nPersisted -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
