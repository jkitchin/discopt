"""Retraction check (2026-11-08): does the Rust "optimal" answer on P1.3's 3 looser
clay0303hfsg node LPs violate optimality? Each column's reduced cost d = c - A^T y is
judged by where x sits: at its lower bound d >= -tol, at its upper d <= tol, strictly
inside |d| <= tol (tol = 1e-7 * (1 + |c_j|)). The earlier diagnosis counted wrong-sign
d on open-sided columns without looking at x. Prints the count of columns checked;
exits 1 when none were."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "cap", "discopt_benchmarks/scripts/capture_highs_parity_fixtures.py"
)
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)
import discopt.solvers.lp_milp_highs as H  # noqa: E402
from discopt.solvers.lp_std_engine import RustEngine  # noqa: E402

HERE = Path(__file__).parent
rows = json.loads((HERE / "p13_entry_with_zero_dual.json").read_text())
D = Path("python/tests/data/highs_parity")
files = sorted({r["file"] for r in rows if "clay" in r["file"]})
checked = 0
for f in files:
    rf = cap.load_fixture(D / f)["rf"]
    sf = H.StdForm.from_arrays(
        rf["c"], rf["A"].tocsc(), rf["row_lo"], rf["col_lo"], rf["col_hi"], rf["obj_const"]
    )
    raw = RustEngine().solve(sf, time_limit=60, presolve=False, tight_tolerances=False)
    if raw.status != "optimal":
        continue
    x, d = raw.x, sf.c - sf.A.T @ raw.y
    tol = 1e-7 * (1 + np.abs(sf.c))
    xtol = 1e-9 * (1 + np.abs(x))
    at_lo = np.abs(x - sf.xl) <= xtol
    at_hi = np.abs(x - sf.xu) <= xtol
    inside = ~at_lo & ~at_hi
    bad_lo = at_lo & ~at_hi & (d < -tol)
    bad_hi = at_hi & ~at_lo & (d > tol)
    bad_in = inside & (np.abs(d) > tol)
    open_side = (sf.xl <= -1e20) | (sf.xu >= 1e20)
    naive = open_side & (((sf.xl <= -1e20) & (d > tol)) | ((sf.xu >= 1e20) & (d < -tol)))
    checked += sf.n
    print(
        f"{f[-12:]} n={sf.n} |y|max={np.abs(raw.y).max():.3g} violations: at-lower "
        f"{bad_lo.sum()} at-upper {bad_hi.sum()} inside {bad_in.sum()} | naive "
        f"open-side count {naive.sum()} (of which at a finite bound {(naive & ~inside).sum()})"
    )
print("columns checked", checked)
sys.exit(0 if checked else 1)
