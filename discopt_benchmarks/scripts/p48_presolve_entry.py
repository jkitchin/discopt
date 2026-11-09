"""highs-parity P4.8 entry: how much does a dimension-reducing presolve remove?

Upper-bound proxy: HiGHS's own MILP presolve (``highspy``, default options) on the same
converted arrays the Rust route solves (``lp_milp_loader``), MIPLIB-38 + the M panel.
Records rows/columns/nonzeros before and after. The plan's entry bar: reductions >= 20 %
of rows on >= 8 instances. Prints executed count; exits non-zero on 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812
import p42_rens_panel as R  # noqa: E402, N812


def presolved_size(inst) -> dict:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    a = inst.A.tocsc()
    lp.num_col_, lp.num_row_ = inst.n, inst.m
    lp.col_cost_ = np.asarray(-inst.c if inst.maximize else inst.c, dtype=float)
    lp.col_lower_ = np.asarray(inst.col_lo, dtype=float)
    lp.col_upper_ = np.asarray(inst.col_hi, dtype=float)
    lp.row_lower_ = np.asarray(inst.row_lo, dtype=float)
    lp.row_upper_ = np.asarray(inst.row_hi, dtype=float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = a.indptr.astype(np.int32)
    lp.a_matrix_.index_ = a.indices.astype(np.int32)
    lp.a_matrix_.value_ = a.data.astype(float)
    lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous
                       for t in inst.is_int]  # fmt: skip
    h.passModel(lp)
    st = h.presolve()
    p = h.getPresolvedLp()
    return {"presolve_status": str(st), "model_status": h.modelStatusToString(h.getModelStatus()),
            "m": inst.m, "n": inst.n, "nnz": int(inst.A.nnz), "m_after": int(p.num_row_),
            "n_after": int(p.num_col_), "nnz_after": int(len(p.a_matrix_.value_))}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, executed = [], 0
    for mem in R.population():
        if any(r["name"] == mem["name"] for r in rows):
            continue
        inst = L.read_instance(str(REPO / mem["path"]))
        r = {"name": mem["name"], "panel": mem.get("panel"), **presolved_size(inst)}
        r["row_reduction"] = 1.0 - r["m_after"] / r["m"] if r["m"] else 0.0
        r["col_reduction"] = 1.0 - r["n_after"] / r["n"] if r["n"] else 0.0
        rows.append(r)
        executed += 1
        print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    q = sum(1 for r in rows if r["row_reduction"] >= 0.20)
    print(f"p48 entry: executed={executed} rows_reduced_ge_20pct={q}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
