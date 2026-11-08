"""pytest plugin (-p lp_switch_probe): record the #8 convex node-bound LP sources and
the P1.2 rust-first rescue verdicts on every call; pickle non-decisive rescue LPs.
Writes JSON lines to $LP_PROBE_OUT; never changes a result."""
import json, os, pickle, time

OUT = os.environ["LP_PROBE_OUT"]
DUMP = os.environ.get("LP_PROBE_DUMP", os.path.dirname(OUT))


def _log(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


def pytest_configure(config):
    import discopt.solvers.convex_node_bound as CNB
    import discopt.solvers.lp_milp_highs as LMH

    real_dual = CNB._lp_dual

    def lp_dual(sf, J, xl, xu, s_lo, s_hi, grad):
        t0 = time.perf_counter()
        y_r, v_r = CNB._simplex_dual(J, xl, xu, s_lo, s_hi, grad)
        t_r = time.perf_counter() - t0
        t0 = time.perf_counter()
        y, v, why = real_dual(sf, J, xl, xu, s_lo, s_hi, grad)
        t_h = time.perf_counter() - t0
        _log({"site": "cnb", "rust_ok": y_r is not None, "rust_val": v_r if y_r is not None else str(v_r),
              "orig_ok": y is not None, "orig_val": v, "why": why, "t_rust": t_r, "t_orig": t_h,
              "test": os.environ.get("PYTEST_CURRENT_TEST", "")})
        return y, v, why

    CNB._lp_dual = lp_dual

    real_rf = LMH.solve_lp_std_rust_first

    def rust_first(sf, *, time_limit=None):
        out = LMH.solve_lp_std(sf, time_limit=time_limit, engine="rust")
        decisive = out.status == "unbounded" or (out.status in ("optimal", "infeasible") and out.gap_certified)
        row = {"site": "p12", "rust_status": out.status, "gap_certified": bool(out.gap_certified),
               "decisive": decisive, "message": str(out.message)[:200],
               "test": os.environ.get("PYTEST_CURRENT_TEST", "")}
        if not decisive:
            fn = os.path.join(DUMP, f"p12_nondecisive_{abs(hash(row['test']))}_{int(time.time()*1e6)}.pkl")
            with open(fn, "wb") as f:
                pickle.dump(sf, f)
            row["dump"] = fn
        back = real_rf(sf, time_limit=time_limit)
        row["final_status"] = back.status
        _log(row)
        return back

    LMH.solve_lp_std_rust_first = rust_first
