"""SOTA-P1 A1 (#97) entry experiment: FBBT-before-root-probe ON vs OFF.

Measures, per instance (nvs05, tanksize, casctanks), TL=60:
  - McCormick-LP root probe status + whether it yielded a safe bound
  - whether _mc_lp_relaxer stayed LIVE (engagement) -> did OBBT/node_reduce engage
  - certified dual bound, root_bound, node_count, status, gap_certified

Run:  python a1_entry.py <mode:off|on> <instance> <time_limit>
Prints a single JSON line to stdout.
"""

import json
import os
import sys

NL_DIR = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


def main():
    mode = sys.argv[1]  # off | on
    inst = sys.argv[2]
    tl = float(sys.argv[3])

    os.environ["DISCOPT_FBBT_BEFORE_ROOT_PROBE"] = "1" if mode == "on" else "0"

    import discopt.modeling as dm

    # ---- instrumentation (measurement only) ----
    probe_info = {"status": None, "lower_bound": None, "called": 0}
    obbt_calls = {"n": 0}

    # Tap the McCormick relaxer's solve_at_node to capture the ROOT probe result.
    # The root probe is the first solve_at_node call on the relaxer object during
    # setup; we record every call's status and note the first one as the probe.
    try:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        _orig_solve = MccormickLPRelaxer.solve_at_node

        def _tapped_solve(self, lb, ub, *a, **k):
            res = _orig_solve(self, lb, ub, *a, **k)
            if probe_info["called"] == 0:
                probe_info["status"] = getattr(res, "status", None)
                probe_info["lower_bound"] = (
                    None if getattr(res, "lower_bound", None) is None else float(res.lower_bound)
                )
            probe_info["called"] += 1
            return res

        MccormickLPRelaxer.solve_at_node = _tapped_solve
    except Exception as e:
        probe_info["tap_error"] = repr(e)

    # Tap root OBBT engagement (per-node OBBT / root fixpoint uses obbt_tighten_root).
    try:
        import discopt._jax.obbt as OBBT

        _orig_obbt = OBBT.obbt_tighten_root

        def _tapped_obbt(*a, **k):
            obbt_calls["n"] += 1
            return _orig_obbt(*a, **k)

        OBBT.obbt_tighten_root = _tapped_obbt
    except Exception:
        pass

    model = dm.from_nl(os.path.join(NL_DIR, inst + ".nl"))
    res = model.solve(time_limit=tl)

    out = {
        "instance": inst,
        "mode": mode,
        "time_limit": tl,
        "status": res.status,
        "objective": None if res.objective is None else float(res.objective),
        "bound": None if res.bound is None else float(res.bound),
        "root_bound": None if res.root_bound is None else float(res.root_bound),
        "node_count": res.node_count,
        "gap_certified": res.gap_certified,
        "probe_status": probe_info["status"],
        "probe_lower_bound": probe_info["lower_bound"],
        "obbt_root_calls": obbt_calls["n"],
        "mc_relaxer_live": obbt_calls["n"] > 0,  # OBBT only runs when relaxer is live
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
