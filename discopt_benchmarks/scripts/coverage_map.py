#!/usr/bin/env python
"""G3 coverage map: which solve path + which capabilities fire, per instance.

Measurement-only. Monkeypatches (does NOT edit) the *actual* solver.py dispatch
targets and the current default-path capabilities with call/fire counters, then
solves one ``.nl`` instance and emits a JSON coverage row.

Solve paths (mutually exclusive; the one that ran for this instance):

  amp           solver="amp"  -> discopt.solvers.amp.solve_amp
  convex-fast   pure-continuous convex -> _solve_continuous (convex_fast_path)
  lp / qp       classified LP/QP -> _solve_lp / _solve_qp
  miqp-bb       convex MIQP  -> _solve_miqp_bb
  milp-driver   classified MILP -> _solve_milp_bb
  nlp-bb        convex MINLP -> _solve_nlp_bb
  spatial       the fall-through McCormick spatial B&B loop *inside* solve_model
                (this is where the governor / alphaBB / PSD-gate / zero-span lift
                capabilities live). Detected by elimination: solve_model was
                entered, none of the delegated paths above fired, and the result
                was not the convex fast path.

Capabilities counted (the current default-path set on origin/main; the older
run_root_fixpoint / reduce_node / DISCOPT_PSD_COST_GATE names do NOT exist here):

  ils_calls           integer_local_search() invocations (VOLUME-1 ILS-capped)
  pump/dive/rins/rens governed primal-heuristic fire counts (the effort governor
    _calls             #541 gates the entry of these; a reachable-but-dark
                      heuristic shows 0 here)
  alphabb_alpha_calls rigorous_alpha() invocations (alphaBB bounding fired)
  psd_gate_calls      _apply_auto_cut_policy() invocations (PSD/RLT auto policy)
  psd_separate_calls  MccormickLPRelaxer._separate_psd() invocations
  psd_enabled         auto-cut policy turned PSD on at least once
  rlt_enabled         auto-cut policy turned RLT on at least once
  zerospan_lift_fired model._zero_spanning_factor_auxes non-empty at solve end
                      (R4 zero-spanning product-factor lift tagged an aux)

Usage
-----
  python coverage_map.py INST.nl --time-limit 60 --json out.json
  python coverage_map.py INST.nl --solver amp        # force the AMP path
"""

from __future__ import annotations

import argparse
import json
import time


def run(args: argparse.Namespace) -> dict:
    import discopt.solver as solver_mod
    from discopt.modeling.core import from_nl

    c = {
        "delegated_path": None,  # amp/lp/qp/miqp-bb/milp-driver/nlp-bb/convex-fast
        "alphabb_alpha_calls": 0,
        "psd_gate_calls": 0,
        "psd_separate_calls": 0,
        "psd_enabled": False,
        "rlt_enabled": False,
    }
    errs: dict[str, str] = {}

    def mark_delegate(name, fn):
        def w(*a, **k):
            # First delegated dispatch target wins.
            if c["delegated_path"] is None:
                c["delegated_path"] = name
            return fn(*a, **k)

        return w

    # --- wrap the real dispatch targets inside solver.py ---
    for name, attr in [
        ("lp", "_solve_lp"),
        ("qp", "_solve_qp"),
        ("miqp-bb", "_solve_miqp_bb"),
        ("milp-driver", "_solve_milp_bb"),
        ("nlp-bb", "_solve_nlp_bb"),
        ("convex-fast", "_solve_continuous"),
    ]:
        if hasattr(solver_mod, attr):
            setattr(solver_mod, attr, mark_delegate(name, getattr(solver_mod, attr)))

    # AMP: wrap the module-level entry it imports.
    try:
        import discopt.solvers.amp as amp_mod

        _orig_amp = amp_mod.solve_amp

        def w_amp(*a, **k):
            if c["delegated_path"] is None:
                c["delegated_path"] = "amp"
            return _orig_amp(*a, **k)

        amp_mod.solve_amp = w_amp
    except Exception as e:  # pragma: no cover
        errs["amp"] = str(e)

    # NOTE on the governor: the effort governor `_root_heur_nlp_entry_ok`
    # (#541) and the improver governor `_improver_allowed` are *local closures*
    # in solve_model, so their per-decision throttle count is not observable from
    # outside without tracing. We instead measure the governor's *effect*: the
    # actual fire counts of the heuristics it gates (below). A heuristic that is
    # reachable-but-dark on a path shows 0 calls; one the governor keeps
    # throttling shows few calls relative to node_count.

    # The governed root/primal heuristics. Wrapping each gives an actual fire
    # count on this path: the effort governor (_root_heur_nlp_entry_ok, a closure
    # we cannot patch) *gates the entry of these*, so a heuristic that is
    # reachable-but-dark on a path shows 0 calls here, and one the governor keeps
    # throttling shows few calls relative to nodes. ``ils_calls`` is the
    # VOLUME-1-capped integer local search.
    try:
        import discopt._jax.primal_heuristics as ph

        _heur_names = {
            "ils_calls": "integer_local_search",
            "pump_calls": "feasibility_pump",
            "dive_calls": "fractional_diving",
            "rins_calls": "rins",
            "rens_calls": "rens",
        }
        for ckey, fname in _heur_names.items():
            c.setdefault(ckey, 0)
            if hasattr(ph, fname):
                _orig = getattr(ph, fname)

                def _mk(ckey, _orig):
                    def w(*a, **k):
                        c[ckey] += 1
                        return _orig(*a, **k)

                    return w

                setattr(ph, fname, _mk(ckey, _orig))
    except Exception as e:
        errs["heur"] = str(e)

    # alphaBB rigorous alpha
    try:
        import discopt._jax.alphabb as ab

        _orig_alpha = ab.rigorous_alpha

        def w_alpha(*a, **k):
            c["alphabb_alpha_calls"] += 1
            return _orig_alpha(*a, **k)

        ab.rigorous_alpha = w_alpha
    except Exception as e:
        errs["alphabb"] = str(e)

    # PSD/RLT auto-cut policy (the "PSD gate")
    try:
        _orig_policy = solver_mod._apply_auto_cut_policy

        def w_policy(model, relaxer):
            c["psd_gate_calls"] += 1
            _orig_policy(model, relaxer)
            if getattr(relaxer, "_psd_cuts", False):
                c["psd_enabled"] = True
            if getattr(relaxer, "_rlt_cuts", False):
                c["rlt_enabled"] = True

        solver_mod._apply_auto_cut_policy = w_policy
    except Exception as e:
        errs["psd_gate"] = str(e)

    # zero-spanning product-factor lift (R4 / #538). The lift sets
    # ``_zero_spanning_factor_auxes`` on the *reformulated* model created inside
    # solve_model (not the model we hold), so wrap the reform and record when it
    # tags an aux (that is the fire signal).
    c["zerospan_lift_fired"] = False
    try:
        import discopt._jax.factorable_reform as fr

        _orig_reform = fr.factorable_reformulate

        def w_reform(model, **k):
            out = _orig_reform(model, **k)
            if getattr(out, "_zero_spanning_factor_auxes", None):
                c["zerospan_lift_fired"] = True
            return out

        fr.factorable_reformulate = w_reform
        # solver.py imports the name at call sites via ``from ... import``; also
        # patch the module attr it resolves through.
        if hasattr(solver_mod, "factorable_reformulate"):
            solver_mod.factorable_reformulate = w_reform
    except Exception as e:
        errs["zerospan"] = str(e)

    # PSD separator
    try:
        import discopt._jax.mccormick_lp as mlp

        if hasattr(mlp.MccormickLPRelaxer, "_separate_psd"):
            _orig_psd = mlp.MccormickLPRelaxer._separate_psd

            def w_psd(self, *a, **k):
                c["psd_separate_calls"] += 1
                return _orig_psd(self, *a, **k)

            mlp.MccormickLPRelaxer._separate_psd = w_psd
    except Exception as e:
        errs["psd_sep"] = str(e)

    # G2 effort governor (#541): process-lifetime singleton with a real
    # throttled_events counter per governed source. Fresh subprocess = clean
    # singleton, so snapshot() after the solve is the governor fire proof.
    try:
        from discopt.heuristic_governor import governor as _gov

        _gov().reset()
    except Exception as e:
        errs["governor"] = str(e)

    model = from_nl(args.instance)
    t0 = time.perf_counter()
    solve_kwargs = {"time_limit": args.time_limit, "gap_tolerance": 1e-4}
    if args.solver:
        solve_kwargs["solver"] = args.solver
    result = model.solve(**solve_kwargs)
    wall = time.perf_counter() - t0

    try:
        from discopt.heuristic_governor import governor as _gov

        gov_snap = _gov().snapshot()
    except Exception:
        gov_snap = {}
    gov_throttled = sum(int(s.get("throttled_events", 0)) for s in gov_snap.values())

    # zero-spanning lift fires on the reformulated model (see wrap above);
    # also honor a tag on the original model as a fallback.
    zerospan = bool(c.get("zerospan_lift_fired")) or bool(
        getattr(model, "_zero_spanning_factor_auxes", None)
    )

    # Resolve the path. A delegated target wins; else convex-fast if the result
    # flagged it; else the fall-through spatial loop.
    path = c["delegated_path"]
    if path == "convex-fast" and not getattr(result, "convex_fast_path", False):
        # _solve_continuous also runs as the pure-continuous non-convex fallback;
        # only call it convex-fast when the result confirms it.
        path = "continuous-fallback"
    if path is None:
        path = "spatial" if getattr(result, "convex_fast_path", False) is False else "convex-fast"

    return {
        "instance": args.instance,
        "path": path,
        "ils_calls": c.get("ils_calls", 0),
        "pump_calls": c.get("pump_calls", 0),
        "dive_calls": c.get("dive_calls", 0),
        "rins_calls": c.get("rins_calls", 0),
        "rens_calls": c.get("rens_calls", 0),
        "alphabb_alpha_calls": c["alphabb_alpha_calls"],
        "psd_gate_calls": c["psd_gate_calls"],
        "psd_separate_calls": c["psd_separate_calls"],
        "psd_enabled": c["psd_enabled"],
        "rlt_enabled": c["rlt_enabled"],
        "governor_snapshot": gov_snap,
        "governor_throttled_events": gov_throttled,
        "zerospan_lift_fired": zerospan,
        "wall_s": wall,
        "node_count": getattr(result, "node_count", None),
        "status": str(result.status),
        "objective": None if result.objective is None else float(result.objective),
        "bound": None if getattr(result, "bound", None) is None else float(result.bound),
        "gap_certified": bool(getattr(result, "gap_certified", False)),
        "_errs": errs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--solver", default=None, help="force a solver family, e.g. 'amp'")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    rec = run(args)
    text = json.dumps(rec, indent=1, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
