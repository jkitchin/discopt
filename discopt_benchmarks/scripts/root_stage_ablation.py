#!/usr/bin/env python
"""THRU-1 — root-relaxation per-stage cost + value-vs-cost ablation (reusable).

Entry-experiment harness for the re-scoped BARON/SCIP certification gap: measure
where the ROOT-relaxation wall goes by stage, and how much *bound* each expensive
stage actually contributes. Measurement-only — it does NOT change solver defaults;
the ``--gate-psd`` mode is a *prototype* cost-aware policy kept here (never shipped
into the default path) so the findings are reproducible.

Stages timed via the T0.3 ``solver_stats`` families that ``SolveResult`` already
exposes:

    reduce/fbbt   reduce/obbt
    separate/multilinear   separate/edge_concave   separate/univariate_square
    separate/convex        separate/psd            separate/rlt

Stage toggles (env / solve kwargs) this harness drives:

    PSD moment cuts        cuts="manual"  (skips the auto-policy that turns PSD on
                           for box-QP; PSD stays at its constructor default OFF)
    univariate-square sep  DISCOPT_SQUARE_SEPARATE=0
    multilinear sep        DISCOPT_MULTILINEAR_SEPARATE=0
    edge-concave           DISCOPT_EDGE_CONCAVE=0
    root cut pool (PSD)    DISCOPT_ROOT_CUT_ROUNDS=N  (route PSD once at root)

Usage
-----
  # per-stage cost table (all stages on) for one instance
  root_stage_ablation.py INST.nl --time-limit 60

  # value-vs-cost: base vs a stage turned off (bound loss + wall saved)
  root_stage_ablation.py INST.nl --config base
  root_stage_ablation.py INST.nl --config nopsd
  root_stage_ablation.py INST.nl --config nopsd_nosq   # DISCOPT_SQUARE_SEPARATE=0 too

  # PROTOTYPE cost-aware gate (default-OFF policy, findings-only): probe whether
  # PSD moves the root LP bound; if not, solve with PSD skipped.
  root_stage_ablation.py INST.nl --gate-psd --time-limit 60

Configs set env in-process for ``nosq``; export ``DISCOPT_SQUARE_SEPARATE=0``
yourself if you want the child env to carry it (the harness also sets it).
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return str(v)


def _solve(inst: str, tl: float, cuts: str | None) -> dict:
    from discopt.modeling.core import from_nl

    kw: dict = {"time_limit": tl, "gap_tolerance": 1e-4}
    if cuts is not None:
        kw["cuts"] = cuts
    t0 = time.perf_counter()
    m = from_nl(inst)
    r = m.solve(**kw)
    wall = time.perf_counter() - t0
    return {
        "wall": wall,
        "status": str(r.status),
        "objective": _f(r.objective),
        "bound": _f(r.bound),
        "root_time": _f(r.root_time),
        "root_bound": _f(r.root_bound),
        "root_gap": _f(r.root_gap),
        "node_count": r.node_count,
        "solver_stats": r.solver_stats or {},
    }


def _config_env(config: str) -> None:
    """Apply the environment side of a named config (square separation lives in env)."""
    if "nosq" in config:
        os.environ["DISCOPT_SQUARE_SEPARATE"] = "0"


def _config_cuts(config: str) -> str | None:
    # "nopsd*" disables the auto cut policy (which is what turns PSD on for box-QP),
    # leaving the relaxer's constructor default (PSD off). "manual" == no auto policy.
    return "manual" if config.startswith("nopsd") else None


def run(args: argparse.Namespace) -> dict:
    name = os.path.basename(args.instance).replace(".nl", "")

    if args.gate_psd:
        # Prototype cost-aware gate (findings-only; NOT a shipped default):
        #   1. cheap probe: does turning PSD on move the certified bound at all on
        #      a short budget? Compare a short base solve vs a short manual-cuts
        #      solve.  2. commit to the cheaper config for the full budget.
        # This exists to reproduce the "skip PSD when it is bound-inert" result;
        # a real gate would decide per-node from a single separation round's bound
        # delta (see the findings doc), not by a probe solve.
        probe_tl = max(args.probe_time, 1.0)
        base_probe = _solve(args.instance, probe_tl, cuts=None)
        man_probe = _solve(args.instance, probe_tl, cuts="manual")
        # PSD is "worth it" only if base reaches a *strictly better* (higher for a
        # min-sense dual bound) root bound than manual within the probe budget.
        rb_base = base_probe["root_bound"]
        rb_man = man_probe["root_bound"]
        # PSD is "worth it" only if, within the probe budget, the base (PSD-on)
        # stack reaches a strictly *tighter* root bound than manual (PSD-off).
        # If base fails to even produce a root bound in the probe budget (its root
        # fixpoint did not converge — the nvs24 pathology), that non-convergence is
        # itself the signal that the expensive stack does not pay off → drop it.
        psd_helps = (
            isinstance(rb_base, float)
            and isinstance(rb_man, float)
            and rb_base > rb_man + 1e-6 * (1.0 + abs(rb_man))
        )
        base_root_converged = isinstance(rb_base, float)
        chosen = None if (psd_helps and base_root_converged) else "manual"
        full = _solve(args.instance, args.time_limit, cuts=chosen)
        return {
            "instance": name,
            "mode": "gate_psd",
            "psd_helps_root_bound": psd_helps,
            "base_root_converged_in_probe": base_root_converged,
            "probe_root_bound_base": rb_base,
            "probe_root_bound_manual": rb_man,
            "chosen_cuts": chosen or "auto",
            **full,
        }

    _config_env(args.config)
    cuts = _config_cuts(args.config)
    rec = _solve(args.instance, args.time_limit, cuts=cuts)
    rec.update({"instance": name, "mode": "ablate", "config": args.config})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument(
        "--config",
        default="base",
        help="base | nopsd | nosq | nopsd_nosq | nomultilinear (env/cuts ablation)",
    )
    ap.add_argument(
        "--gate-psd",
        action="store_true",
        help="prototype cost-aware PSD gate (probe then commit; findings-only)",
    )
    ap.add_argument("--probe-time", type=float, default=8.0, help="--gate-psd probe budget (s)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    if args.config == "nomultilinear":
        os.environ["DISCOPT_MULTILINEAR_SEPARATE"] = "0"

    rec = run(args)
    text = json.dumps(rec, indent=1)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
