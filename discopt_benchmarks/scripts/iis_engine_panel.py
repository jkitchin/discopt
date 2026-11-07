"""highs-parity P3.2: the MILP IIS engine on Rust vs HiGHS, over P0.6's 53 fixtures.

Each fixture (``python/tests/data/highs_parity/iis/``) is run with ``iis_engine="rust"``
and ``iis_engine="highs"`` in fresh processes, interleaved with rotated order, ``--reps``
times. Per fixture: whether the Rust IIS equals P0.6's (HiGHS) baseline element for
element, and -- for every Rust set that differs -- an independent validity check in raw
HiGHS on the ``from_nl`` model: the set is infeasible, and dropping any one of its rows,
bounds or integralities makes it feasible (irreducible). Walls: median per engine and the
per-fixture Rust/HiGHS ratio. Prints the comparison and validity-check counts and exits
non-zero on 0 compared or on any invalid Rust set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DIR = REPO / "python" / "tests" / "data" / "highs_parity" / "iis"
sys.path.insert(0, str(HERE.parent))

import iis_parity_panel as P  # noqa: E402, N812


def _subsystem_feasible(path: str, sig: dict, drop=None) -> str:
    """Raw-HiGHS status of the IIS ``sig`` (minus element ``drop``) on the model the
    engines saw: kept rows with their bounds, kept bounds, kept integralities; every
    other row free, bound open and column continuous."""
    import highspy
    import lp_milp_loader as L  # noqa: N812
    from discopt.modeling.core import from_nl

    inst = L.read_instance(path)
    with tempfile.TemporaryDirectory() as td:
        nl = os.path.join(td, "m.nl")
        L.write_nl(inst, nl)
        mps = os.path.join(td, "m.mps")
        from_nl(nl).to_mps(mps)
        h0 = highspy.Highs()
        h0.setOptionValue("output_flag", False)
        h0.readModel(mps)
    lp = h0.getLp()
    names = list(lp.col_names_)
    inf = highspy.kHighsInf
    rows = [r for r in sig["constraints"] if drop != ("c", r)]
    bnds = {(b[0], b[1]) for b in sig["bounds"] if drop != ("b", b[0], b[1])}
    ints = {i[0] for i in sig["integrality"] if drop != ("i", i[0])}
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    a = lp.a_matrix_
    A = np.zeros((lp.num_row_, lp.num_col_))  # noqa: N806
    for j in range(lp.num_col_):
        for k in range(a.start_[j], a.start_[j + 1]):
            A[a.index_[k], j] = a.value_[k]
    lo0, hi0 = np.array(lp.col_lower_), np.array(lp.col_upper_)
    for j, nm in enumerate(names):
        held = nm in ints
        lo = lo0[j] if (held or (nm, "lower") in bnds) else -inf
        hi = hi0[j] if (held or (nm, "upper") in bnds) else inf
        h.addVar(lo, hi)
        if held:
            h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
    for r in rows:
        idx = np.flatnonzero(A[r])
        h.addRow(lp.row_lower_[r], lp.row_upper_[r], len(idx), idx.astype(np.int32), A[r, idx])
    h.run()
    return h.modelStatusToString(h.getModelStatus())


def validate(path: str, sig: dict) -> tuple[bool, str]:
    """Is ``sig`` an IIS: infeasible, and feasible after dropping any one element?"""
    full = _subsystem_feasible(path, sig)
    if full != "Infeasible":
        return False, f"set is not infeasible ({full})"
    elems = ([("c", r) for r in sig["constraints"]]
             + [("b", b[0], b[1]) for b in sig["bounds"]]
             + [("i", i[0]) for i in sig["integrality"]])  # fmt: skip
    for e in elems:
        st = _subsystem_feasible(path, sig, drop=e)
        if st == "Infeasible":
            return False, f"element {e} is removable"
    return True, f"irreducible over {len(elems)} elements"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    expected = json.loads((DIR / "expected.json").read_text())["models"]
    rows, compared, checked, invalid = [], 0, 0, []
    keys = ("constraints", "bounds", "integrality")
    for i, sha in enumerate(sorted(expected), 1):
        path = str(DIR / f"{sha}.mps")
        per: dict[str, list] = {"rust": [], "highs": []}
        for rep in range(a.reps):
            for eng in ("rust", "highs") if rep % 2 == 0 else ("highs", "rust"):
                per[eng].append(P._run_one(path, a.time_limit, eng))
        errs = [r["error"] for e in per for r in per[e] if "error" in r]
        if errs:
            raise RuntimeError(f"{sha}: {errs}")
        want = expected[sha]
        rust_sigs = {json.dumps({k: r[k] for k in keys}) for r in per["rust"]}
        same = all({k: r[k] for k in keys} == {k: want[k] for k in keys} for r in per["rust"])
        compared += 1
        row = {"fixture": sha, "identical": same, "rust_deterministic": len(rust_sigs) == 1,
               "rust_proven_irreducible": [r["proven_irreducible"] for r in per["rust"]],
               "wall_rust": statistics.median(r["wall"] for r in per["rust"]),
               "wall_highs": statistics.median(r["wall"] for r in per["highs"])}  # fmt: skip
        if not same:
            for sig_s in rust_sigs:
                ok, why = validate(path, json.loads(sig_s))
                checked += 1
                row.setdefault("validity", []).append(why)
                if not ok:
                    invalid.append((sha, why))
        rows.append(row)
        print(f"[{i}/{len(expected)}] {sha} identical={same} "
              f"rust {row['wall_rust']:.3f}s highs {row['wall_highs']:.3f}s "
              f"{row.get('validity', '')}", flush=True)  # fmt: skip
    ratios = [r["wall_rust"] / r["wall_highs"] for r in rows if r["wall_highs"] > 0]
    summ = {"fixtures": compared, "identical": sum(r["identical"] for r in rows),
            "validity_checks": checked, "invalid": invalid,
            "median_wall_ratio": statistics.median(ratios) if ratios else None,
            "max_wall_ratio": max(ratios) if ratios else None,
            "total_wall_rust": sum(r["wall_rust"] for r in rows),
            "total_wall_highs": sum(r["wall_highs"] for r in rows)}  # fmt: skip
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1) + "\n")
    print("summary", json.dumps(summ), flush=True)
    return 0 if compared and not invalid else 1


if __name__ == "__main__":
    sys.exit(main())
