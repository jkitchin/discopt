#!/usr/bin/env python
"""Pure LP-relaxation optimum per instance (integrality dropped), the pre-MIP root bound.

Same arrays as the ablation (lp_milp_loader.read_instance). Prints executed count; exits
non-zero on 0. Output: d2_lp_bounds.json {name: {lp_obj, status, wall}}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lp_milp_loader as L  # noqa: E402, N812
from d2_highs_ablation import load_instances  # noqa: E402


def main() -> int:
    import highspy

    out = {}
    executed = 0
    for inst_d in load_instances():
        inst = L.read_instance(inst_d["path"])
        h = highspy.Highs()
        for k, v in [("output_flag", False), ("threads", 1), ("time_limit", 60.0)]:
            assert h.setOptionValue(k, v) == highspy.HighsStatus.kOk, k
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = inst.n, inst.m
        lp.col_cost_ = inst.c
        lp.offset_ = inst.offset
        lp.sense_ = highspy.ObjSense.kMaximize if inst.maximize else highspy.ObjSense.kMinimize
        lp.col_lower_, lp.col_upper_ = inst.col_lo, inst.col_hi
        lp.row_lower_, lp.row_upper_ = inst.row_lo, inst.row_hi
        a = inst.A.tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
        # no integrality_ -> pure LP
        assert h.passModel(lp) == highspy.HighsStatus.kOk
        t0 = time.perf_counter()
        st = h.run()
        if st == highspy.HighsStatus.kError:
            raise RuntimeError(f"{inst_d['name']}: LP run error")
        status = h.modelStatusToString(h.getModelStatus())
        if status != "Optimal":
            raise RuntimeError(f"{inst_d['name']}: LP status {status}")
        obj = float(h.getInfo().objective_function_value)
        o = inst_d["oracle"]
        gap = abs(obj - o) / max(1.0, abs(o))
        out[inst_d["name"]] = {"lp_obj": obj, "oracle": o, "lp_gap": gap, "status": status,
                               "wall": time.perf_counter() - t0, "maximize": inst.maximize}
        executed += 1
        print(f"{inst_d['name']:20s} LP={obj:.8g} oracle={o:.8g} gap={gap:.4%}", flush=True)
    Path(__file__).with_name("d2_lp_bounds.json").write_text(json.dumps(out, indent=1))
    print(f"executed LPs: {executed}")
    return 0 if executed else 2


if __name__ == "__main__":
    sys.exit(main())
